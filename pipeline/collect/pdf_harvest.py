"""
pdf_harvest.py — fetch PDFs so the ocr stage has something to read
==================================================================
The `ocr` stage consumes a directory of PDFs. This fills that directory.

WHY THIS IS THE RIGHT LEVER FOR A LOW-RESOURCE LANGUAGE
--------------------------------------------------------
Page scraping is politeness-bound: throughput is

    hosts x per_host_concurrency / crawl_delay

and on the Nepali web that lands around 3 pages/s no matter how many workers
or terminals you throw at it. You cannot fix that by working harder, because
the limit is set by other people's robots.txt.

Documents change the arithmetic. A single statute PDF from lawcommission.gov.np
carries more text than fifty news articles, and fetching 400 PDFs costs 400
requests instead of 400,000. The brief names OCR from books and PDFs as a
manual-collection method, so these tokens count toward the >=20% exactly as
scraped pages do.

For Nepali specifically the highest-value targets are the statute corpus, the
gazette (rajpatra), and ministry publications: natively drafted long-form
prose, no syndication, essentially zero duplication against Sangraha.

WHAT IT DOES
------------
Reads seed pages from <lang>/configs/pdf_sources.txt, walks each one up to
--depth links deep staying on the same host, and downloads every .pdf it finds.
robots.txt is honoured, including Crawl-delay, and each host is fetched at most
--per-host at a time. Already-downloaded files are skipped, so re-running
extends the collection rather than repeating it.

Usage
-----
  python -m pipeline.collect.pdf_harvest --lang nepali --out-dir ~/nepali_pdfs
  python -m pipeline.collect.pdf_harvest --lang nepali --out-dir ~/nepali_pdfs \\
      --depth 3 --max-pdfs 2000

Then:
  python run_phase1.py --lang nepali --stage ocr --ocr-input-dir ~/nepali_pdfs
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)

# Directory-listing and pagination links are worth following; login pages and
# calendars are rabbit holes that never end in a PDF.
DENY_LINK = re.compile(
    r"/(login|signin|register|cart|search|tag|share|print|comment)"
    r"|[?&](replytocom|share=|print=)"
    r"|\.(jpg|jpeg|png|gif|svg|css|js|zip|mp4|mp3|docx?|xlsx?)($|\?)", re.I)


def is_pdf_link(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return path.endswith(".pdf")


def safe_name(url: str, out_dir: Path) -> Path:
    """
    A filesystem-safe name that stays recognisable and cannot collide.

    The URL basename alone collides constantly on government sites -- every
    ministry has a `notice.pdf` -- so an 8-char hash of the full URL is
    appended. Keeping the readable stem matters when you later need to cite
    where a document came from.
    """
    base = unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "document.pdf"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:80]
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    h = hashlib.blake2b(url.encode(), digest_size=4).hexdigest()
    return out_dir / f"{base[:-4]}__{h}.pdf"


async def harvest(seeds: list[str], out_dir: Path, *, depth: int,
                  max_pdfs: int, per_host: int, delay: float,
                  max_hours: float, user_agent: str, min_bytes: int,
                  log=print) -> dict:
    import httpx
    from pipeline.collect.scrape_collect import HostLimiter, RobotsCache

    out_dir.mkdir(parents=True, exist_ok=True)
    limiter = HostLimiter(per_host=per_host, delay=delay)
    robots = RobotsCache(user_agent)

    seen_pages: set[str] = set()
    seen_pdfs: set[str] = set()
    stats: Counter[str] = Counter()
    downloaded_bytes = 0
    t0 = time.monotonic()

    existing = {p.name for p in out_dir.glob("*.pdf")}
    if existing:
        log(f"  {len(existing):,} pdfs already in {out_dir} — will skip those")

    queue: list[tuple[str, int]] = [(s, 0) for s in seeds]
    headers = {"User-Agent": user_agent, "Accept-Language": "ne,hi,en;q=0.5"}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        while queue and len(seen_pdfs) < max_pdfs:
            if (time.monotonic() - t0) / 3600 >= max_hours:
                log(f"  [stop] --max-hours {max_hours} reached")
                break

            url, d = queue.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            host = urlsplit(url).netloc

            ok, cd = await robots.allowed(client, url)
            if cd:
                limiter.set_delay(host, float(cd))
            if not ok:
                stats["robots_denied"] += 1
                continue

            await limiter.acquire(host)
            try:
                r = await client.get(url, timeout=40)
            except Exception:
                stats["fetch_error"] += 1
                continue
            finally:
                limiter.release(host)

            if r.status_code >= 400:
                stats[f"http_{r.status_code}"] += 1
                continue

            ctype = r.headers.get("content-type", "").lower()

            # ---- a PDF -------------------------------------------------------
            if "pdf" in ctype or is_pdf_link(url):
                dest = safe_name(url, out_dir)
                if dest.name in existing:
                    stats["already_have"] += 1
                    continue
                body = r.content
                if len(body) < min_bytes:
                    stats["too_small"] += 1
                    continue
                if not body[:5].startswith(b"%PDF"):
                    # Content-Type lied, or it is an error page served as a PDF
                    # link. Cheap to check, and OCR on a HTML error page
                    # produces convincing-looking garbage.
                    stats["not_a_pdf"] += 1
                    continue
                dest.write_bytes(body)
                seen_pdfs.add(url)
                downloaded_bytes += len(body)
                stats["downloaded"] += 1
                if len(seen_pdfs) % 25 == 0:
                    log(f"  {len(seen_pdfs):,} pdfs, "
                        f"{downloaded_bytes / 1e6:.0f} MB, "
                        f"{len(queue):,} links queued, "
                        f"{(time.monotonic() - t0) / 60:.0f} min")
                continue

            # ---- an HTML page: harvest links --------------------------------
            if "html" not in ctype:
                stats["skipped_content_type"] += 1
                continue
            stats["pages_walked"] += 1
            if d >= depth:
                continue

            for href in HREF_RE.findall(r.text):
                nxt = urljoin(str(r.url), href.strip())
                if not nxt.startswith(("http://", "https://")):
                    continue
                nxt, _, _ = nxt.partition("#")
                if nxt in seen_pages:
                    continue
                if is_pdf_link(nxt):
                    queue.append((nxt, d))          # PDFs do not cost depth
                elif urlsplit(nxt).netloc == host and not DENY_LINK.search(nxt):
                    queue.append((nxt, d + 1))

    return {"pdfs": len(seen_pdfs), "bytes": downloaded_bytes,
            "pages_walked": stats["pages_walked"], "stats": dict(stats),
            "minutes": (time.monotonic() - t0) / 60}


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", default=None,
                    help="default <lang>/configs/pdf_sources.txt")
    ap.add_argument("--depth", type=int, default=2,
                    help="how many link hops from each seed to follow")
    ap.add_argument("--max-pdfs", type=int, default=3000)
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--min-bytes", type=int, default=20000,
                    help="skip tiny PDFs: cover sheets, single-page forms")
    ap.add_argument("--user-agent", default=None)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    src = Path(args.sources) if args.sources else \
        root / args.lang / "configs" / "pdf_sources.txt"
    if not src.exists():
        print(f"[error] {src} not found. It lists the pages to harvest PDFs "
              f"from, one URL per line.", file=sys.stderr)
        return 1

    seeds = [l.strip() for l in src.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    if not seeds:
        print(f"[error] {src} has no urls", file=sys.stderr)
        return 1

    ua = args.user_agent or (
        "corpus-research/1.0 (academic language-model dataset; "
        "contact: set --user-agent)")
    if "set --user-agent" in ua:
        print("[WARN] no contact address in the User-Agent. Set one with "
              "--user-agent: it is what lets a publisher email you instead of "
              "silently blocking you.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    print(f"[{args.lang}] harvesting pdfs from {len(seeds)} seed pages")
    print(f"  out: {out_dir}")
    print(f"  depth {args.depth}, max {args.max_pdfs:,} pdfs, "
          f"{args.per_host} concurrent/host at {args.delay}s\n")

    res = asyncio.run(harvest(seeds, out_dir, depth=args.depth,
                              max_pdfs=args.max_pdfs, per_host=args.per_host,
                              delay=args.delay, max_hours=args.max_hours,
                              user_agent=ua, min_bytes=args.min_bytes))

    print(f"\n  {res['pdfs']:,} pdfs ({res['bytes'] / 1e6:.0f} MB) "
          f"from {res['pages_walked']:,} pages in {res['minutes']:.0f} min")
    for k, v in sorted(res["stats"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} {v:,}")

    if not res["pdfs"]:
        print("\n  Nothing downloaded. Check that the seed pages actually link "
              "to PDFs,\n  and try --depth 3 — many government sites put the "
              "documents two or three\n  clicks below the landing page.")
        return 1

    print(f"\n  next:  python run_phase1.py --lang {args.lang} "
          f"--stage ocr --ocr-input-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
