"""
boilerplate_strip.py
--------------------
Step 3 of the cleaning pipeline (scraped sources only).

Removes common boilerplate patterns from web-scraped text:
  - Navigation menus (repeated link-like lines)
  - Footer / header text patterns
  - Cookie consent notices
  - Social-media share prompts
  - Advertisement / sponsored content markers
  - Excessive URL / email lines

This step is applied *only* to documents whose source_type is 'scrape'.
Other source types (wikipedia, hf_corpus, ocr, transcription) are passed
through unchanged, since trafilatura already handles most boilerplate.

Usage (CLI)
-----------
  python boilerplate_strip.py \\
      --lang hindi \\
      --input /path/to/lang_filtered.jsonl \\
      --output /path/to/boilerplate_stripped.jsonl \\
      --manifest /path/to/manifest.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Boilerplate patterns
# ---------------------------------------------------------------------------

# Lines that look like pure navigation / menu items:
# e.g. short lines that are just a sequence of | or ??? separated words
_NAV_LINE = re.compile(
    r"^(?:[\w\s]{1,30}\s*[|??????????]\s*){2,}[\w\s]{1,30}$"
)

# Lines that are mostly URLs
_URL_HEAVY = re.compile(
    r"https?://\S+", re.IGNORECASE
)

# Common footer / boilerplate phrases (case-insensitive)
_BOILERPLATE_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"all rights reserved",
        r"copyright\s+???\s*\d{4}",
        r"terms of (use|service)",
        r"privacy policy",
        r"cookie policy",
        r"subscribe to our newsletter",
        r"follow us on",
        r"share this (article|post|story)",
        r"click here to",
        r"advertisement",
        r"sponsored content",
        r"?????????????????????????????? ????????????????????????",      # "all rights reserved" in Hindi
        r"????????????????????????",                   # "disclaimer"
        r"?????????????????? ????????????",                 # "contact us"
        r"??????????????????????????????",                  # "newsletter"
        r"???????????? ??????????????????",                # "social media"
    ]
]

# Line that contains more URLs than words ??? likely a link dump
_MAX_URL_FRACTION = 0.5  # drop line if >50% of tokens are URLs

# Minimum fraction of alphabetic Devanagari chars in a kept line
_MIN_ALPHA_FRACTION = 0.3

# Lines shorter than this (in chars) that aren't sentence-ending are nav-like
_MIN_LINE_CHARS = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strip_boilerplate_line(line: str) -> Optional[str]:
    """
    Evaluate a single line and return it if it should be kept, else None.

    Parameters
    ----------
    line : str
        A single line of text (already stripped of leading/trailing whitespace).

    Returns
    -------
    str or None
        The line if it passes all filters, or None if it should be dropped.
    """
    if not line:
        return ""  # keep blank lines (paragraph breaks)

    # 1. Drop lines matching boilerplate phrases
    for pattern in _BOILERPLATE_PHRASES:
        if pattern.search(line):
            return None

    # 2. Drop navigation-looking lines (multiple pipe / chevron separators)
    if _NAV_LINE.match(line):
        return None

    # 3. Drop lines where most tokens are URLs
    tokens = line.split()
    if tokens:
        url_tokens = sum(1 for t in tokens if _URL_HEAVY.match(t))
        if url_tokens / len(tokens) > _MAX_URL_FRACTION:
            return None

    # 4. Drop very short non-sentence lines (likely navigation fragments)
    if len(line) < _MIN_LINE_CHARS and not line.endswith(("???", "???", ".", "?", "!")):
        return None

    return line


def strip_boilerplate_text(text: str) -> str:
    """
    Apply boilerplate stripping to every line of a document's text.

    Parameters
    ----------
    text : str
        Full document text (may contain \\n newlines).

    Returns
    -------
    str
        Cleaned text with boilerplate lines removed.
        Consecutive blank lines are collapsed to a single blank line.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    prev_blank = False

    for line in lines:
        stripped_line = line.strip()
        result = strip_boilerplate_line(stripped_line)

        if result is None:
            continue  # dropped

        is_blank = result == ""
        if is_blank and prev_blank:
            continue  # collapse consecutive blanks
        cleaned.append(result)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


def strip_document(doc: dict, text_key: str = "text") -> dict:
    """
    Strip boilerplate from a document dict.

    Only processes documents with ``source_type == 'scrape'`` (or those
    missing a source_type field, erring on the side of cleaning).
    Other source types are returned as-is.

    Parameters
    ----------
    doc : dict
        Document dict (must have ``text_key`` field).
    text_key : str
        JSON key containing the text.

    Returns
    -------
    dict
        New doc dict with cleaned text.
    """
    source_type = doc.get("source_type", "scrape")
    if source_type != "scrape":
        return doc  # pass-through for non-scraped sources

    new_doc = dict(doc)
    new_doc[text_key] = strip_boilerplate_text(doc.get(text_key, ""))
    return new_doc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Step 3: Strip boilerplate from scraped documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"],
                        help="Target language (for logging).")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file.")
    parser.add_argument("--text-key", default="text",
                        help="JSON key containing the text.")
    return parser.parse_args()


def main() -> None:
    """
    Apply boilerplate stripping to all scrape-type documents.

    Logs: docs processed, scrape-type docs stripped, docs dropped (empty
    after stripping).
    """
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    stripped_count = 0
    dropped_empty = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"[boilerplate_strip] {args.lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed JSON: {e}", file=sys.stderr)
                continue

            total += 1
            original_len = len(doc.get(args.text_key, ""))
            cleaned_doc = strip_document(doc, text_key=args.text_key)
            cleaned_len = len(cleaned_doc.get(args.text_key, ""))

            if cleaned_len < original_len:
                stripped_count += 1

            if not cleaned_doc.get(args.text_key, "").strip():
                dropped_empty += 1
                continue

            fout.write(json.dumps(cleaned_doc, ensure_ascii=False) + "\n")

    kept = total - dropped_empty
    print(
        f"\n[boilerplate_strip] Done. "
        f"Input: {total} | Stripped (partial): {stripped_count} | "
        f"Dropped (empty): {dropped_empty} | Output: {kept}"
    )


if __name__ == "__main__":
    main()
