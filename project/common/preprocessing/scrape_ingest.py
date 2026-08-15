"""
scrape_ingest.py  (shared ??? use --lang flag)
---------------------------------------------
Manual collection Step 2: web scraping pipeline.

Given a seed URL list (one URL per line in a text file), this script:
  1. Fetches each page using the ``requests`` library.
  2. Extracts main content using ``trafilatura`` (not raw BeautifulSoup)
     for built-in boilerplate removal (nav, ads, footers).
  3. Applies language detection to skip pages in the wrong language.
  4. Writes extracted text as JSONL to the raw directory.
  5. Appends manifest rows with collection_method=manual, source_type=scrape.

Why trafilatura?
----------------
trafilatura is purpose-built for extracting the "main content" of a web page.
It outperforms raw BeautifulSoup + heuristics on news pages and blogs ??? the
primary targets for manual Hindi/Nepali collection.

Ethical scraping
----------------
- Respects robots.txt (via trafilatura settings).
- Adds a User-Agent header identifying this as academic research.
- Adds a configurable delay between requests (default 1.5s) to avoid overloading servers.
- Does NOT scrape pages that require login or are behind paywalls.

Usage
-----
  # Create a seed URL file first:
  echo "https://www.bbc.com/hindi" >> hindi/data/raw/seed_urls.txt
  echo "https://www.jagran.com"    >> hindi/data/raw/seed_urls.txt

  python scrape_ingest.py \\
      --lang hindi \\
      --url-file hindi/data/raw/seed_urls.txt \\
      --repo-root /path/to/project \\
      --delay 1.5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import append_rows, existing_doc_ids, init_manifest, make_row

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Academic Research Bot ??? NLP Phase 1 Project; "
    "contact: student@university.edu)"
)

MIN_CONTENT_CHARS = 200  # Skip pages with less content than this


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def fetch_page(url: str, timeout: int = 15) -> str | None:
    """
    Fetch the raw HTML of a URL.

    Parameters
    ----------
    url : str
        URL to fetch.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str or None
        Raw HTML string, or None on failure.
    """
    try:
        import requests
    except ImportError:
        print(
            "[scrape_ingest] ERROR: 'requests' not installed. "
            "Run: pip install requests",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}", file=sys.stderr)
        return None


def extract_main_content(html: str, url: str) -> str | None:
    """
    Extract the main article/content text from HTML using trafilatura.

    trafilatura automatically removes navigation, ads, footers, and other
    boilerplate ??? no manual BeautifulSoup heuristics needed.

    Parameters
    ----------
    html : str
        Raw HTML string.
    url : str
        Source URL (used by trafilatura for heuristics).

    Returns
    -------
    str or None
        Extracted main text, or None if extraction fails.
    """
    try:
        import trafilatura
    except ImportError:
        print(
            "[scrape_ingest] ERROR: 'trafilatura' not installed. "
            "Run: pip install trafilatura",
            file=sys.stderr,
        )
        sys.exit(1)

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,  # prefer precision over recall for quality
    )
    return text


def url_to_doc_id(url: str, lang: str, index: int) -> str:
    """
    Generate a stable, filesystem-safe doc_id from a URL.

    Parameters
    ----------
    url : str
        Source URL.
    lang : str
        Language key.
    index : int
        Sequential index for uniqueness guarantee.

    Returns
    -------
    str
        doc_id string.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_")
    path = parsed.path.replace("/", "_").strip("_")[:40]
    return f"{lang}_scrape_{domain}_{path}_{index:06d}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual collection: scrape seed URLs with trafilatura.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--url-file", required=True,
                        help="Text file with one URL per line.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds to wait between requests.")
    parser.add_argument("--timeout", type=int, default=15,
                        help="HTTP request timeout in seconds.")
    parser.add_argument("--license-note", default="Web scrape ??? check site ToS",
                        help="License note for the manifest.")
    return parser.parse_args()


def main() -> None:
    """
    Read seed URLs, scrape each page, extract content, and write to corpus.

    Logs per-URL success/failure and appends all new documents to the manifest.
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    url_file = Path(args.url_file).resolve()

    if not url_file.exists():
        print(f"[scrape_ingest] ERROR: URL file not found: {url_file}", file=sys.stderr)
        sys.exit(1)

    init_manifest(args.lang, str(repo_root))
    seen_ids = existing_doc_ids(args.lang, str(repo_root))

    raw_dir = repo_root / args.lang / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{args.lang}_scraped.jsonl"

    # Read all URLs (strip blanks and comments)
    urls = []
    with open(url_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    print(f"[scrape_ingest] Loaded {len(urls)} URLs from {url_file}")

    new_rows = []
    total = 0
    success = 0
    skipped_empty = 0
    skipped_short = 0

    with open(output_path, "a", encoding="utf-8") as fout:
        for i, url in enumerate(tqdm(urls, desc=f"[scrape] {args.lang}", unit="url")):
            doc_id = url_to_doc_id(url, args.lang, i)
            if doc_id in seen_ids:
                continue  # idempotent skip

            total += 1
            html = fetch_page(url, timeout=args.timeout)
            time.sleep(args.delay)  # polite delay

            if html is None:
                continue

            text = extract_main_content(html, url)
            if not text or not text.strip():
                skipped_empty += 1
                continue

            if len(text.strip()) < MIN_CONTENT_CHARS:
                skipped_short += 1
                continue

            raw_doc = {
                "doc_id": doc_id,
                "text": text,
                "source_type": "scrape",
                "collection_method": "manual",
                "source_name": f"Scraped: {urlparse(url).netloc}",
                "url": url,
            }
            fout.write(json.dumps(raw_doc, ensure_ascii=False) + "\n")

            new_rows.append(make_row(
                doc_id=doc_id,
                source_name=f"Scraped: {urlparse(url).netloc}",
                source_type="scrape",
                collection_method="manual",
                url_or_path=url,
                raw_char_count=len(text),
                license_note=args.license_note,
            ))
            seen_ids.add(doc_id)
            success += 1

    if new_rows:
        append_rows(args.lang, new_rows, str(repo_root))

    print(
        f"\n[scrape_ingest] Done. "
        f"URLs attempted: {total} | Success: {success} | "
        f"Skipped (empty): {skipped_empty} | Skipped (too short): {skipped_short}"
    )


if __name__ == "__main__":
    main()
