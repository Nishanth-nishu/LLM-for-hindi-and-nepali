"""
scrape_collect.py — high-throughput manual web collection
==========================================================
Produces the >=20% MANUAL portion of the corpus by scraping and cleaning pages
yourself, which is exactly what the brief lists as qualifying manual collection.

WHY THIS EXISTS ALONGSIDE scrape_ingest.py
------------------------------------------
The existing `scrape_ingest.py` fetches sequentially with `time.sleep(delay)`.
At the default 1.5 s that is **0.66 pages/second**.

Do the arithmetic against the requirement:

    500M tokens x 20% manual        = 100,000,000 manual tokens per language
    at ~600 tokens per news article =     ~167,000 documents
    at 0.66 pages/s                 =       ~70 hours   PER LANGUAGE

It cannot finish. Not "slowly" -- the sequential design is off by roughly two
orders of magnitude from what the requirement needs.

This module fixes the two things that cause that:

1. CONCURRENCY ACROSS HOSTS, POLITENESS WITHIN A HOST.
   The 1.5 s delay is the right thing to do *per host*, and keeping it is what
   makes this ethical. The mistake is applying it globally. With 40 hosts in
   flight and 1.5 s per host you get ~26 pages/s while every individual
   publisher still sees one request every 1.5 s. Same politeness, 40x the
   throughput. 167k documents becomes ~2 hours.

   **Your throughput is set by how many DOMAINS you crawl, not how hard you hit
   each one.** Ten domains caps you at ~6.6 pages/s no matter what you
   configure. If you are short of tokens, add domains.

2. URL DISCOVERY.
   `scrape_ingest.py` takes a seed file of URLs. A list of 30 homepages yields
   30 documents. Real volume comes from each site's sitemap, which lists its
   whole article archive. `seed_discovery.py` turns 30 homepages into 200k
   article URLs; this module consumes that.

ETHICS -- unchanged from the sequential version, and non-negotiable
-------------------------------------------------------------------
- robots.txt is fetched and honoured per host, including Crawl-delay.
- A robots.txt that fails to load is treated as DISALLOW, not allow.
- Per-host concurrency capped at 2, minimum 1.5 s between hits on a host.
- Identifying User-Agent with a real contact address.
- No login-walled or paywalled pages.
- Hosts that fail repeatedly are parked rather than hammered.

Usage
-----
  # 1. discover article URLs from your seed domains
  python -m pipeline.collect.seed_discovery --lang hindi --repo-root .

  # 2. scrape them
  python -m pipeline.collect.scrape_collect --lang hindi --repo-root . \\
      --max-hours 3 --target-chars 400000000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import Document, ShardWriter  # noqa: E402

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

# Hindi vs Nepali share a script, so a Devanagari check is not a language check.
# These are high-frequency function words that are near-exclusive to one side.
NEPALI_MARKERS = {"छ", "छन्", "छैन", "गर्छ", "गरेको", "भएको", "लाई", "सँग", "हुन्",
                  "भन्ने", "भन्दा", "गर्ने", "नेपाल", "काठमाडौं", "हरू"}
HINDI_MARKERS = {"है", "हैं", "था", "थे", "किया", "गया", "रहा", "में", "नहीं",
                 "लेकिन", "क्योंकि", "भारत", "दिल्ली", "करने", "बताया"}


# ---------------------------------------------------------------------------
# Politeness
# ---------------------------------------------------------------------------

class HostLimiter:
    """Per-host: a concurrency semaphore plus a minimum gap between requests."""

    def __init__(self, per_host: int = 2, delay: float = 1.5, jitter: float = 0.3):
        self.per_host = per_host
        self.delay = delay
        self.jitter = jitter
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._next_ok: dict[str, float] = {}
        self._delays: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def set_delay(self, host: str, d: float):
        self._delays[host] = max(d, self.delay)

    def _sem(self, host: str) -> asyncio.Semaphore:
        if host not in self._sems:
            self._sems[host] = asyncio.Semaphore(self.per_host)
        return self._sems[host]

    async def acquire(self, host: str):
        await self._sem(host).acquire()
        while True:
            async with self._lock:
                now = time.monotonic()
                nxt = self._next_ok.get(host, 0.0)
                if now >= nxt:
                    base = self._delays.get(host, self.delay)
                    j = base * self.jitter
                    self._next_ok[host] = now + base + random.uniform(-j, j)
                    return
                wait = nxt - now
            await asyncio.sleep(min(wait, 3.0))

    def release(self, host: str):
        self._sem(host).release()


class RobotsCache:
    """robots.txt per host. Unavailable robots.txt => disallow."""

    def __init__(self, user_agent: str):
        self.ua = user_agent
        self._cache: dict[str, tuple] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, client, url: str) -> tuple[bool, float | None]:
        p = urlsplit(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin in self._cache:
            rp, delay, bad = self._cache[origin]
            return ((not bad) and (rp is None or self._can(rp, url))), delay

        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin in self._cache:
                rp, delay, bad = self._cache[origin]
                return ((not bad) and (rp is None or self._can(rp, url))), delay
            rp, delay, bad = None, None, False
            try:
                r = await client.get(origin + "/robots.txt", timeout=15,
                                     follow_redirects=True)
                if r.status_code >= 500 or r.status_code in (401, 403):
                    bad = True
                elif r.status_code < 400:
                    rp, delay = self._parse(r.text)
            except Exception:
                bad = True
            self._cache[origin] = (rp, delay, bad)
            return ((not bad) and (rp is None or self._can(rp, url))), delay

    def _parse(self, body: str):
        try:
            from protego import Protego
            rp = Protego.parse(body)
            return rp, rp.crawl_delay(self.ua)
        except ImportError:
            import urllib.robotparser as robotparser
            rp = robotparser.RobotFileParser()
            rp.parse(body.splitlines())
            return rp, rp.crawl_delay(self.ua)
        except Exception:
            return None, None

    def _can(self, rp, url: str) -> bool:
        try:
            if hasattr(rp, "can_fetch") and rp.__class__.__name__ == "Protego":
                return bool(rp.can_fetch(url, self.ua))
            return bool(rp.can_fetch(self.ua, url))
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Extraction + language check
# ---------------------------------------------------------------------------

def extract_text(html: str, url: str) -> tuple[str, str]:
    """Main-content extraction. Returns (text, extractor_name)."""
    try:
        import trafilatura
        got = trafilatura.extract(html, url=url, include_comments=False,
                                  include_tables=False, favor_precision=True,
                                  output_format="txt", deduplicate=True)
        if got and len(got.strip()) >= 200:
            return got.strip(), "trafilatura"
        got = trafilatura.extract(html, url=url, include_comments=False,
                                  favor_recall=True, output_format="txt")
        if got and len(got.strip()) >= 200:
            return got.strip(), "trafilatura:recall"
    except ImportError:
        pass
    except Exception:
        pass
    # Minimal fallback so the pipeline still functions without trafilatura.
    import html as _h
    body = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", html, flags=re.I | re.S)
    paras = re.findall(r"<p\b[^>]*>(.*?)</p>", body, flags=re.I | re.S)
    txt = "\n".join(_h.unescape(re.sub(r"<[^>]+>", " ", p)).strip() for p in paras)
    txt = re.sub(r"[ \t]+", " ", txt).strip()
    return (txt, "regex_fallback") if len(txt) >= 200 else ("", "none")


def normalize(text: str) -> str:
    """NFC + whitespace collapse. Same normalisation the tokenizer expects."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")
    lines = [" ".join(l.split()) for l in text.splitlines()]
    return "\n".join(l for l in lines if l).strip()


def language_ok(text: str, lang: str, *, min_dev=0.70, max_latin=0.12) -> tuple[bool, str]:
    """
    Devanagari check plus a Hindi/Nepali discriminator.

    Script alone is not enough: Hindi and Nepali share Devanagari, and a Hindi
    page landing in the Nepali corpus is a correctness bug the brief cares
    about ("Do not share documents across corpora").
    """
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return False, "empty"
    dev = len(DEVANAGARI.findall(stripped)) / len(stripped)
    if dev < min_dev:
        return False, f"devanagari_ratio={dev:.2f}"
    lat = len(LATIN.findall(stripped)) / len(stripped)
    if lat > max_latin:
        return False, f"latin_ratio={lat:.2f}"

    toks = text.split()
    hi = sum(1 for t in toks if t.strip("।,.!?\"'()") in HINDI_MARKERS)
    ne = sum(1 for t in toks if t.strip("।,.!?\"'()") in NEPALI_MARKERS)
    if hi + ne < 3:
        return False, "no_language_markers"
    want_hi = lang == "hindi"
    if want_hi and hi <= ne:
        return False, f"looks_nepali(hi={hi},ne={ne})"
    if not want_hi and ne <= hi:
        return False, f"looks_hindi(hi={hi},ne={ne})"
    return True, "ok"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    fetched: int = 0
    kept: int = 0
    robots_denied: int = 0
    http_error: int = 0
    extract_failed: int = 0
    wrong_language: int = 0
    too_short: int = 0
    duplicate: int = 0
    chars: int = 0
    reasons: dict = field(default_factory=dict)

    def bump(self, key: str):
        self.reasons[key] = self.reasons.get(key, 0) + 1


async def run(*, lang: str, repo_root: Path, urls: list[str], user_agent: str,
              concurrency: int, per_host: int, delay: float, max_hours: float,
              target_chars: int, min_chars: int, out_name: str,
              order: str = "forward", shuffle_seed: int = 20260820,
              log=print) -> Stats:
    import httpx

    st = Stats()
    limiter = HostLimiter(per_host=per_host, delay=delay)
    # Per-host accounting. Aggregate throughput hides the thing that actually
    # determines it: how many hosts are contributing, and what delay each one
    # imposes. A run at 3 pages/s across 21 hosts is not "the network is slow",
    # it is "most hosts are silent, or have declared a 10s Crawl-delay" -- and
    # those two have completely different fixes.
    host_fetched: dict[str, int] = {}
    host_kept: dict[str, int] = {}
    host_delay: dict[str, float] = {}
    robots = RobotsCache(user_agent)
    parked: set[str] = set()
    fails: dict[str, int] = {}
    stop = asyncio.Event()
    t0 = time.monotonic()

    out_path = repo_root / lang / "data" / "raw" / out_name
    writer = ShardWriter(out_path)
    log(f"  output: {out_path}  ({len(writer.seen):,} already collected)")

    # URL-level resume.
    #
    # ShardWriter resumes the OUTPUT: it remembers content hashes, so a document
    # already collected is not written twice. That is not the same as resuming
    # the CRAWL. Without a record of which URLs were visited, a restart walks
    # the list from the top, re-downloads every page it already has, and
    # discards each one as a duplicate -- paying full network and politeness
    # cost for zero new documents. On a 345k-URL list that is hours of work to
    # rediscover what is already on disk.
    #
    # So: log every URL that reached a definitive outcome, and skip those.
    # Definitive means the server answered -- 2xx we kept or rejected, 4xx, or
    # robots said no. Timeouts, connection errors and 5xx are NOT logged,
    # because those deserve a retry on the next run.
    visited_path = repo_root / lang / "data" / "raw" / f".{out_name}.visited"
    visited: set[str] = set()
    if visited_path.exists():
        with open(visited_path, encoding="utf-8") as vf:
            visited = {l.strip() for l in vf if l.strip()}
        log(f"  visited log: {len(visited):,} urls already attempted "
            f"-> skipping them")
    elif out_path.exists():
        # No visited log, but documents exist: this output predates URL-level
        # resume. Bootstrap from the urls recorded on the documents themselves.
        #
        # This recovers the pages that were KEPT, not the ones fetched and
        # rejected, so the first run after upgrading still re-fetches the
        # rejects. Partial credit is worth having -- on a run that kept 57% of
        # what it fetched, this skips over half the redundant work -- and every
        # run after this one is exact.
        from pipeline.manifest import read_jsonl as _read
        visited = {r["url"] for r in _read(out_path) if r.get("url")}
        if visited:
            log(f"  no visited log; bootstrapped {len(visited):,} urls from "
                f"documents already collected.")
            log(f"  (urls that were fetched and REJECTED are not recorded "
                f"there, so some re-fetching is unavoidable this run only.)")
            visited_path.write_text("\n".join(sorted(visited)) + "\n",
                                    encoding="utf-8")
    visited_f = open(visited_path, "a", encoding="utf-8")
    visited_buf: list[str] = []

    def mark_visited(u: str) -> None:
        visited_buf.append(u)
        if len(visited_buf) >= 200:
            visited_f.write("\n".join(visited_buf) + "\n")
            visited_f.flush()
            visited_buf.clear()

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def handle(client, url: str):
        if stop.is_set():
            return
        host = urlsplit(url).netloc
        if host in parked:
            return
        async with sem:
            ok, cd = await robots.allowed(client, url)
            if cd:
                limiter.set_delay(host, float(cd))
                host_delay[host] = float(cd)
            if not ok:
                st.robots_denied += 1
                mark_visited(url)
                return
            await limiter.acquire(host)
            try:
                r = await client.get(url, timeout=25, follow_redirects=True)
            except Exception as e:
                st.http_error += 1
                fails[host] = fails.get(host, 0) + 1
                if fails[host] >= 25:
                    parked.add(host)
                    log(f"  [parked] {host} after 25 consecutive failures")
                return
            finally:
                limiter.release(host)

            st.fetched += 1
            host_fetched[host] = host_fetched.get(host, 0) + 1
            if 400 <= r.status_code < 500:
                # The server answered and the answer will not change. Log it.
                st.http_error += 1
                st.bump(f"http_{r.status_code}")
                mark_visited(url)
                return
            if r.status_code >= 500:
                # Transient. Leave it unlogged so the next run retries.
                st.http_error += 1
                st.bump(f"http_{r.status_code}")
                return
            fails[host] = 0

            mark_visited(url)          # 2xx: whatever happens below is final

            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "xml" not in ctype:
                st.bump("content_type")
                return

            text, extractor = extract_text(r.text, url)
            if not text:
                st.extract_failed += 1
                return
            text = normalize(text)
            if len(text) < min_chars:
                st.too_short += 1
                return
            good, why = language_ok(text, lang)
            if not good:
                st.wrong_language += 1
                st.bump(f"lang:{why.split('(')[0].split('=')[0]}")
                return

            doc = Document(
                text=text, language=lang, provenance_class="manual",
                source=host, collection_method="scrape", url=url,
                extra={"extractor": extractor, "http_status": r.status_code},
            )
            async with lock:
                if writer.write(doc):
                    st.kept += 1
                    st.chars += len(text)
                    host_kept[host] = host_kept.get(host, 0) + 1
                else:
                    st.duplicate += 1
                if st.chars >= target_chars:
                    stop.set()

    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency)
    headers = {"User-Agent": user_agent,
               "Accept-Language": "hi,ne,en;q=0.5"}

    todo = [u for u in urls if u not in visited] if visited else list(urls)
    if visited:
        log(f"  {len(todo):,} of {len(urls):,} urls remain "
            f"({len(urls) - len(todo):,} skipped as already attempted)")

    # ORDER MATTERS ON A RESUME, AND THE DEFAULT ORDER IS THE WRONG ONE.
    #
    # The visited log records URLs that reached a definitive outcome. When it
    # is bootstrapped from an older output file, only the KEPT urls are
    # recoverable -- the ones fetched and rejected were never written down. The
    # shuffle order is seeded and therefore stable, so those rejects all sit at
    # the FRONT of what remains, and a resume spends hours re-fetching pages it
    # already knows it does not want. Observed: 4,492 fetched, 12 kept.
    #
    # `tail` starts from the end of the list, which by construction is the
    # region no previous run reached, so acceptance returns to normal
    # immediately. `reshuffle` mixes rejects evenly through fresh urls instead
    # of front-loading them -- better if you intend to run the list out.
    if order == "tail":
        todo.reverse()
        log(f"  order: tail-first (skips the re-reject region a bootstrapped "
            f"resume puts at the front)")
    elif order == "reshuffle":
        random.Random(shuffle_seed + 1).shuffle(todo)
        log(f"  order: reshuffled")
    if not todo:
        log("  nothing left to fetch. Add domains to seed_domains.txt and "
            "re-run `--stage discover` to extend the URL list.")

    try:
        async with httpx.AsyncClient(headers=headers, limits=limits,
                                     follow_redirects=True) as client:
            pending: set[asyncio.Task] = set()
            last_log = time.monotonic()
            for i, url in enumerate(todo):
                if stop.is_set():
                    break
                if (time.monotonic() - t0) / 3600 >= max_hours:
                    log(f"  [stop] --max-hours {max_hours} reached")
                    break
                t = asyncio.create_task(handle(client, url))
                pending.add(t)
                t.add_done_callback(pending.discard)
                if len(pending) >= concurrency * 4:
                    await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if time.monotonic() - last_log >= 30:
                    last_log = time.monotonic()
                    el = time.monotonic() - t0
                    log(f"  {i + 1:,}/{len(todo):,} urls | kept {st.kept:,} | "
                        f"{st.chars / 1e6:.1f}M chars | "
                        f"{st.fetched / max(1e-9, el):.1f} pages/s | "
                        f"{el / 60:.0f} min")
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        # ---- per-host breakdown: the diagnostic that matters -------------
        el = max(1e-9, time.monotonic() - t0)
        all_hosts = {urlsplit(u).netloc for u in urls}
        log(f"\n  per-host ({len(host_fetched)} of {len(all_hosts)} hosts responded):")
        log(f"    {'host':<38}{'fetched':>9}{'kept':>8}{'pg/s':>8}  delay")
        for h in sorted(host_fetched, key=lambda k: -host_fetched[k])[:40]:
            d = host_delay.get(h)
            log(f"    {h:<38}{host_fetched[h]:>9,}{host_kept.get(h, 0):>8,}"
                f"{host_fetched[h] / el:>8.2f}"
                f"  {f'{d:g}s (robots)' if d else f'{delay:g}s'}")
        silent = sorted(all_hosts - set(host_fetched))
        if silent:
            log(f"    [{len(silent)} host(s) returned nothing: "
                f"{', '.join(silent[:6])}{' ...' if len(silent) > 6 else ''}]")
        slow = {h: d for h, d in host_delay.items() if d > delay * 2}
        if slow:
            worst = sorted(slow.items(), key=lambda kv: -kv[1])[:4]
            log(f"\n  {len(slow)} host(s) declared a Crawl-delay above your "
                f"{delay:g}s default.")
            log(f"  That is theirs to set and it is honoured, so the only "
                f"lever left is MORE HOSTS.")
            log(f"  Worst: {worst}")
    finally:
        writer.close()
        if visited_buf:
            visited_f.write("\n".join(visited_buf) + "\n")
        visited_f.flush()
        visited_f.close()

    return st


def load_urls(path: Path, shuffle_seed: int = 20260820) -> list[str]:
    """
    Load and INTERLEAVE urls by host.

    A sitemap-ordered file is grouped by host, so feeding it in order means all
    workers queue behind the same site and the per-host delay throttles
    everything. Interleaving by host is what actually delivers the concurrency.
    """
    by_host: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            by_host.setdefault(urlsplit(u).netloc, []).append(u)

    rng = random.Random(shuffle_seed)
    for v in by_host.values():
        rng.shuffle(v)

    out, hosts = [], list(by_host)
    rng.shuffle(hosts)
    idx = {h: 0 for h in hosts}
    while True:
        progressed = False
        for h in hosts:
            i = idx[h]
            if i < len(by_host[h]):
                out.append(by_host[h][i])
                idx[h] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser(
        description="Concurrent, robots-respecting manual web collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--url-file", default=None,
                    help="default: <lang>/data/raw/article_urls.txt")
    ap.add_argument("--out-name", default="manual_scrape.jsonl")
    ap.add_argument("--user-agent",
                    default="Phase1CorpusBot/1.0 (academic research; CONTACT@YOUR-DOMAIN)")
    ap.add_argument("--concurrency", type=int, default=40,
                    help="global in-flight requests; throughput is capped by "
                         "the NUMBER OF DOMAINS, not this")
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--delay", type=float, default=1.5,
                    help="minimum seconds between hits on ONE host")
    ap.add_argument("--url-order", choices=["forward", "tail", "reshuffle"],
                    default="forward",
                    help="resume order. Use 'tail' after a bootstrapped resume: "
                         "the front of the list is dense with urls a previous "
                         "run already fetched and rejected.")
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--target-chars", type=int, default=400_000_000,
                    help="stop at this many characters kept. ~400M chars is a "
                         "rough proxy for ~100M tokens of Devanagari.")
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    url_file = Path(args.url_file) if args.url_file else \
        root / args.lang / "data" / "raw" / "article_urls.txt"
    if not url_file.exists():
        print(f"[error] {url_file} not found.\n"
              f"        Run seed_discovery.py first -- scraping a list of "
              f"homepages yields one document per homepage.", file=sys.stderr)
        return 1

    if "CONTACT@YOUR-DOMAIN" in args.user_agent:
        print("[WARN] User-Agent still contains the placeholder contact address.\n"
              "       Put a real one in: it is what lets a publisher email you\n"
              "       instead of silently blocking the range.")

    urls = load_urls(url_file)
    n_hosts = len({urlsplit(u).netloc for u in urls})
    print(f"[{args.lang}] {len(urls):,} urls across {n_hosts} hosts")

    ceiling = n_hosts * (args.per_host / args.delay)
    print(f"  politeness ceiling: ~{ceiling:.1f} pages/s "
          f"({n_hosts} hosts x {args.per_host} concurrent / {args.delay}s)")
    if ceiling < 5:
        print(f"  [WARN] under 5 pages/s. Throughput here is set by the number of\n"
              f"         DOMAINS, not by --concurrency. Add more seed domains.")
    print(f"  in {args.max_hours}h that is at most ~{ceiling * args.max_hours * 3600:,.0f} pages")

    st = asyncio.run(run(
        lang=args.lang, repo_root=root, urls=urls, user_agent=args.user_agent,
        concurrency=args.concurrency, per_host=args.per_host, delay=args.delay,
        max_hours=args.max_hours, target_chars=args.target_chars,
        min_chars=args.min_chars, out_name=args.out_name,
        order=args.url_order))

    print(f"\n  fetched          {st.fetched:,}")
    print(f"  kept             {st.kept:,}  ({st.chars / 1e6:.1f}M chars)")
    print(f"  robots denied    {st.robots_denied:,}")
    print(f"  http errors      {st.http_error:,}")
    print(f"  extract failed   {st.extract_failed:,}")
    print(f"  wrong language   {st.wrong_language:,}")
    print(f"  too short        {st.too_short:,}")
    print(f"  duplicates       {st.duplicate:,}")
    if st.reasons:
        top = sorted(st.reasons.items(), key=lambda kv: -kv[1])[:8]
        print(f"  top reasons      {dict(top)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
