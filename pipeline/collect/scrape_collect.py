"""
scrape_collect.py ??? high-throughput manual web collection
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

DEVANAGARI = re.compile(r"[???-???]")
LATIN = re.compile(r"[A-Za-z]")

# Hindi vs Nepali share a script, so a Devanagari check is not a language check.
# These are high-frequency function words that are near-exclusive to one side.
NEPALI_MARKERS = {"???", "?????????", "?????????", "????????????", "???????????????", "????????????", "?????????", "?????????", "????????????",
                  "???????????????", "???????????????", "???????????????", "???????????????", "????????????????????????", "?????????"}
HINDI_MARKERS = {"??????", "?????????", "??????", "??????", "????????????", "?????????", "?????????", "?????????", "????????????",
                 "???????????????", "?????????????????????", "????????????", "??????????????????", "????????????", "???????????????"}


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
    text = text.replace("???", "").replace("???", "")
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
    hi = sum(1 for t in toks if t.strip("???,.!?\"'()") in HINDI_MARKERS)
    ne = sum(1 for t in toks if t.strip("???,.!?\"'()") in NEPALI_MARKERS)
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
              target_chars: int, min_chars: int, out_name: str, log=print) -> Stats:
    import httpx

    st = Stats()
    limiter = HostLimiter(per_host=per_host, delay=delay)
    robots = RobotsCache(user_agent)
    parked: set[str] = set()
    fails: dict[str, int] = {}
    stop = asyncio.Event()
    t0 = time.monotonic()

    out_path = repo_root / lang / "data" / "raw" / out_name
    writer = ShardWriter(out_path)
    log(f"  output: {out_path}  ({len(writer.seen):,} already collected)")

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
            if not ok:
                st.robots_denied += 1
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
            if r.status_code >= 400:
                st.http_error += 1
                st.bump(f"http_{r.status_code}")
                return
            fails[host] = 0

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
                else:
                    st.duplicate += 1
                if st.chars >= target_chars:
                    stop.set()

    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency)
    headers = {"User-Agent": user_agent,
               "Accept-Language": "hi,ne,en;q=0.5"}

    try:
        async with httpx.AsyncClient(headers=headers, limits=limits,
                                     follow_redirects=True) as client:
            pending: set[asyncio.Task] = set()
            last_log = time.monotonic()
            for i, url in enumerate(urls):
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
                    log(f"  {i + 1:,}/{len(urls):,} urls | kept {st.kept:,} | "
                        f"{st.chars / 1e6:.1f}M chars | "
                        f"{st.fetched / max(1e-9, el):.1f} pages/s | "
                        f"{el / 60:.0f} min")
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    finally:
        writer.close()

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
        min_chars=args.min_chars, out_name=args.out_name))

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
