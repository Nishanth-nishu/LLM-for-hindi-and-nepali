"""
dedup_exact.py
--------------
Step 5a of the cleaning pipeline: exact-duplicate removal.

Computes SHA-256 of each document's normalised text. Documents whose hash
has already been seen are dropped.  This catches:
  - Identical articles syndicated across multiple news portals
  - Identical Wikipedia dumps downloaded from multiple sources
  - Identical documents that appear in both the public and manual corpora

The script is idempotent: re-running it on the same input produces the same
output (first occurrence of each hash is always kept).

Usage (CLI)
-----------
  python dedup_exact.py \\
      --lang hindi \\
      --input /path/to/quality_filtered.jsonl \\
      --output /path/to/exact_deduped.jsonl
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def sha256_of_text(text: str) -> str:
    """
    Compute the SHA-256 hex digest of a UTF-8 encoded text string.

    Parameters
    ----------
    text : str
        Input text.  Whitespace is normalised before hashing so that
        differences in trailing newlines do not create false negatives.

    Returns
    -------
    str
        64-character hex string.
    """
    # Normalise whitespace to avoid trivial mismatches
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def dedup_exact_stream(
    input_path: Path,
    output_path: Path,
    text_key: str = "text",
    lang: str = "",
) -> dict:
    """
    Stream through a JSONL file, write first-seen docs, drop exact duplicates.

    Parameters
    ----------
    input_path : Path
        Input JSONL file.
    output_path : Path
        Output JSONL file.
    text_key : str
        JSON key containing the document text.
    lang : str
        Language label used in progress-bar description.

    Returns
    -------
    dict
        Statistics: {total, kept, dropped, unique_hashes}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    total = 0
    kept = 0
    dropped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"[dedup_exact] {lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed JSON: {e}", file=sys.stderr)
                continue

            total += 1
            text = doc.get(text_key, "")
            h = sha256_of_text(text)

            if h in seen_hashes:
                dropped += 1
                continue

            seen_hashes.add(h)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    return {
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "unique_hashes": len(seen_hashes),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Step 5a: Remove exact duplicate documents using SHA-256.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--text-key", default="text",
                        help="JSON key containing the text to hash.")
    return parser.parse_args()


def main() -> None:
    """Run exact deduplication and log survival statistics."""
    args = parse_args()

    stats = dedup_exact_stream(
        input_path=Path(args.input),
        output_path=Path(args.output),
        text_key=args.text_key,
        lang=args.lang,
    )

    print(
        f"\n[dedup_exact] Done. "
        f"Input: {stats['total']} | Kept: {stats['kept']} | "
        f"Dropped (exact dups): {stats['dropped']} | "
        f"Unique hashes: {stats['unique_hashes']}"
    )


if __name__ == "__main__":
    main()
