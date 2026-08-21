"""
seed_discovery.py ??? turn seed domains into article URLs
========================================================
The missing step between "I have 40 news sites" and "I have 200,000 article
URLs to scrape".

`scrape_ingest.py` takes a file of URLs. If that file contains 40 homepages you
collect 40 documents. Volume lives in each site's sitemap, which lists its whole
published archive -- often tens or hundreds of thousands of URLs, partitioned by
year and month.

Handles what actually breaks in the wild:
  - gzipped sitemaps (.xml.gz) -- very common on large Indian news sites
  - sitemap index files nested two or three levels deep
  - Google News sitemaps carrying <news:publication_date>
  - sitemaps served as text/plain, one URL per line
  - malformed XML (regex-first parsing, ElementTree only as a fallback)
  - year/month archive partitions, which is where the depth is

Writes <lang>/data/raw/article_urls.txt, which scrape_collect.py consumes.

Usage
-----
  python -m pipeline.collect.seed_discovery --lang hindi --repo-root .
  python -m pipeline.collect.seed_discovery --lang nepali --repo-root . --max-per-host 40000
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)

CONVENTIONAL = [
    "/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
    "/wp-sitemap.xml", "/sitemap.xml.gz", "/sitemap_news.xml",
    "/news-sitemap.xml", "/sitemapindex.xml", "/sitemap/sitemap.xml",
]

# Pages that are not articles. Filtering here saves a fetch each later.
DENY = re.compile(
    r"/tag/|/tags/|/author/|/search|/login|/register|/cart|/wp-json/|"
    r"/feed/?$|/amp/?$|/photo-gallery|/video/|/videos/|/live-|"
    r"/horoscope|/rashifal|/panchang|/page/\d+/?$|"
    r"\.(jpg|jpeg|png|gif|webp|svg|mp4|mp3|pdf|zip|docx?|xlsx?)$", re.I)


def maybe_gunzip(body: bytes) -> bytes:
    if body[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(body)
        except OSError:
            return body
    return body


def parse_sitemap(text: str) -> tuple[list[str], list[str]]:
    """Returns (child_sitemaps, page_urls)."""
    head = text[:2000].lower()
    locs = [m.strip() for m in LOC_RE.findall(text)]
    if locs:
        return (locs, []) if "<sitemapindex" in head else ([], locs)
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("http")]
    if lines:
        return [], lines
    try:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(text.strip())
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        if root.tag.lower().endswith("sitemapindex"):
            return [e.text.strip() for e in root.findall(".//sm:sitemap/sm:loc", ns) if e.text], []
        return [], [e.text.strip() for e in root.findall(".//sm:url/sm:loc", ns) if e.text]
    except Exception:
        return [], []


async def discover_host(client, base: str, *, max_urls: int, max_sitemaps: int,
                        log=print) -> list[str]:
    p = urlsplit(base)
    origin = f"{p.scheme or 'https'}://{p.netloc}"

    queue: list[str] = []
    seen_sm: set[str] = set()

    # robots.txt Sitemap: directives are authoritative -- try them first.
    try:
        r = await client.get(origin + "/robots.txt", timeout=15)
        if r.status_code < 400:
            for line in r.text.splitlines():
                if line.lower().lstrip().startswith("sitemap:"):
                    u = line.split(":", 1)[1].strip()
                    if u not in seen_sm:
                        seen_sm.add(u)
                        queue.append(u)
    except Exception:
        pass

    for path in CONVENTIONAL:
        u = urljoin(origin + "/", path.lstrip("/"))
        if u not in seen_sm:
            seen_sm.add(u)
            queue.append(u)

    pages: list[str] = []
    seen_pages: set[str] = set()
    processed = 0

    while queue and len(pages) < max_urls and processed < max_sitemaps:
        sm = queue.pop(0)
        processed += 1
        try:
            r = await client.get(sm, timeout=30)
            if r.status_code >= 400:
                continue
            text = maybe_gunzip(r.content).decode("utf-8", errors="replace")
        except Exception:
            continue

        children, urls = parse_sitemap(text)
        for c in children:
            if c not in seen_sm and len(seen_sm) < max_sitemaps * 4:
                seen_sm.add(c)
                queue.append(c)
        for u in urls:
            if u in seen_pages or DENY.search(u):
                continue
            seen_pages.add(u)
            pages.append(u)
            if len(pages) >= max_urls:
                break

    log(f"    {p.netloc:<40} {len(pages):>8,} urls  ({processed} sitemaps)")
    return pages


def load_seeds(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line if line.startswith("http") else "https://" + line)
    return out


async def amain(args) -> int:
    import httpx

    root = Path(args.repo_root).resolve()
    seeds_file = Path(args.seeds) if args.seeds else \
        root / args.lang / "configs" / "seed_domains.txt"
    if not seeds_file.exists():
        print(f"[error] {seeds_file} not found. It should list one domain per line.",
              file=sys.stderr)
        return 1

    seeds = load_seeds(seeds_file)
    print(f"[{args.lang}] discovering from {len(seeds)} seed domains")

    out_path = root / args.lang / "data" / "raw" / "article_urls.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": args.user_agent}
    limits = httpx.Limits(max_connections=args.concurrency * 2)
    sem = asyncio.Semaphore(args.concurrency)
    all_urls: list[str] = []

    async def one(client, base):
        async with sem:
            try:
                return await discover_host(client, base, max_urls=args.max_per_host,
                                           max_sitemaps=args.max_sitemaps)
            except Exception as e:
                print(f"    {base:<40} FAILED: {type(e).__name__}")
                return []

    async with httpx.AsyncClient(headers=headers, limits=limits,
                                 follow_redirects=True) as client:
        results = await asyncio.gather(*[one(client, s) for s in seeds])
    for r in results:
        all_urls.extend(r)

    seen = set()
    deduped = [u for u in all_urls if not (u in seen or seen.add(u))]
    out_path.write_text("\n".join(deduped), encoding="utf-8")

    hosts = len({urlsplit(u).netloc for u in deduped})
    print(f"\n  {len(deduped):,} unique article urls across {hosts} hosts")
    print(f"  wrote {out_path}")

    # The number that actually decides whether the target is reachable.
    need_docs = args.target_tokens / 600
    print(f"\n  target {args.target_tokens:,} manual tokens ~= {need_docs:,.0f} documents")
    if len(deduped) < need_docs * 1.6:
        print(f"  [WARN] only {len(deduped):,} urls for a target of ~{need_docs:,.0f} "
              f"accepted documents.\n"
              f"         Acceptance after extraction + language + quality filtering\n"
              f"         is typically 40-60%, so you want ~{need_docs * 1.8:,.0f}+ urls.\n"
              f"         Add more seed domains -- that is the only lever that works.")
    else:
        print(f"  headroom looks adequate ({len(deduped) / max(1, need_docs):.1f}x)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--seeds", default=None,
                    help="default: <lang>/configs/seed_domains.txt")
    ap.add_argument("--max-per-host", type=int, default=30000)
    ap.add_argument("--max-sitemaps", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--target-tokens", type=int, default=100_000_000,
                    help="manual-token target, for the headroom check")
    ap.add_argument("--user-agent",
                    default="Phase1CorpusBot/1.0 (academic research; CONTACT@YOUR-DOMAIN)")
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
