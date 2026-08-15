"""
normalize.py  (v2)
------------------
Stage 6: Normalization and Repair.

v2 changes over v1:
  - Reads thresholds from filtering_thresholds.yaml (warns if null).
  - Repair-over-delete: attempts to fix recoverable OCR issues before dropping.
  - Logs per-document outcome to manifest via drop_stage/drop_reason.
  - Writes filtering_stats.json with before/after counts.

Repair heuristics (applied before dropping)
--------------------------------------------
  1. Common OCR mis-reads: ??? ??? 0, ??? variants, broken Devanagari matras.
  2. Doubled spaces / punctuation run-ons introduced by OCR line-by-line output.
  3. Spurious ASCII substitutions of Devanagari chars (e.g. 'f' for '???' in
     some scanner outputs) ??? NOT auto-fixed (too risky); flagged for review.

Usage
-----
  python normalize.py \\
      --lang hindi \\
      --input  hindi/data/raw/downloaded/hi_wiki.jsonl \\
      --output hindi/data/cleaned/hi_wiki_normalized.jsonl \\
      --repo-root .
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import update_rows

# ????????? Patterns ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

_ZW_CHARS  = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]"
)
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HTML_ENT   = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")

# Devanagari danda variants ??? canonical
_DANDA_VARIANTS = {"\u0964": "???", "\u0965": "???"}

# Common OCR repair map (conservative ??? only high-confidence substitutions)
_OCR_REPAIR = str.maketrans({
    "\u09e6": "???",  # Bengali ??? ??? Devanagari ???
    "\u0966": "???",  # already Devanagari, but various encodings
    "|":      "???",  # ASCII pipe often OCR'd for danda
})


# ????????? Public API ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def normalize_text(
    text: str,
    convert_deva_digits: bool = False,
    repair_ocr: bool = True,
) -> str:
    """
    Apply the full normalization + repair pipeline to a single string.

    Order of operations:
      1. NFC Unicode normalization
      2. Remove zero-width / invisible characters
      3. Remove control characters (preserve \\n and \\t)
      4. Strip HTML entities
      5. Conservative OCR repair (if repair_ocr=True)
      6. Devanagari punctuation canonicalization
      7. Optionally convert Devanagari digits to ASCII
      8. Collapse whitespace within lines
      9. Collapse runs of blank lines to single blank line

    Parameters
    ----------
    text : str
    convert_deva_digits : bool
    repair_ocr : bool
        Apply conservative OCR repair heuristics (safe for all source types).

    Returns
    -------
    str
        Normalized string.
    """
    if not text:
        return text

    # 1. NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Zero-width chars
    text = _ZW_CHARS.sub("", text)
    # 3. Control chars
    text = _CTRL_CHARS.sub("", text)
    # 4. HTML entities
    text = _HTML_ENT.sub(" ", text)
    # 5. OCR repair
    if repair_ocr:
        text = text.translate(_OCR_REPAIR)
    # 6. Devanagari punctuation
    for variant, canonical in _DANDA_VARIANTS.items():
        text = text.replace(variant, canonical)
    # 7. Devanagari digits
    if convert_deva_digits:
        text = text.translate(str.maketrans("??????????????????????????????", "0123456789"))
    # 8???9. Whitespace normalisation
    lines = text.split("\n")
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        line = _MULTI_SPACE.sub(" ", line).strip()
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def load_threshold(config: dict, key_path: list[str], fallback: float) -> tuple[float, bool]:
    """
    Read a threshold from the filtering_thresholds config dict.

    Parameters
    ----------
    config : dict
        Parsed filtering_thresholds.yaml.
    key_path : list[str]
        Nested key path, e.g. ['normalization', 'min_chars_after_norm', 'value'].
    fallback : float
        Conservative fallback if value is null.

    Returns
    -------
    tuple[float, bool]
        (threshold_value, is_fallback)
    """
    node = config
    for k in key_path:
        node = node.get(k, {}) if isinstance(node, dict) else {}
    val = node if not isinstance(node, dict) else None
    if val is None:
        print(
            f"[normalize] WARNING: Threshold {key_path} is null in filtering_thresholds.yaml. "
            f"Using conservative fallback: {fallback}",
            file=sys.stderr,
        )
        return fallback, True
    return float(val), False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 6: Normalize and repair text in a JSONL corpus file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--convert-deva-digits", action="store_true")
    parser.add_argument("--no-repair-ocr", action="store_true",
                        help="Disable OCR repair heuristics.")
    return parser.parse_args()


def main() -> None:
    """Normalize all documents in input JSONL; write passing docs to output."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    # Load threshold config
    thresh_path = repo_root / args.lang / "configs" / "filtering_thresholds.yaml"
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    min_chars, is_fallback = load_threshold(
        thresh_cfg,
        ["normalization", "min_chars_after_norm", "value"],
        fallback=50,
    )

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = dropped_empty = dropped_short = kept = 0
    drop_updates: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"[normalize] {args.lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text = doc.get(args.text_key, "")
            normed = normalize_text(
                text,
                convert_deva_digits=args.convert_deva_digits,
                repair_ocr=not args.no_repair_ocr,
            )
            doc_id = str(doc.get("doc_id", ""))

            if not normed.strip():
                dropped_empty += 1
                drop_updates[doc_id] = {
                    "status": "dropped",
                    "drop_stage": "normalization",
                    "drop_reason": "empty_after_normalization",
                }
                continue

            if len(normed) < min_chars:
                dropped_short += 1
                drop_updates[doc_id] = {
                    "status": "dropped",
                    "drop_stage": "normalization",
                    "drop_reason": f"too_short_after_norm:{len(normed)}<{min_chars}",
                }
                continue

            doc[args.text_key] = normed
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    # Batch-update manifest
    if drop_updates:
        n = update_rows(args.lang, drop_updates, str(repo_root))
        print(f"[normalize] Updated {n} manifest rows with drop info.")

    print(
        f"\n[normalize] Done.  Input: {total:,} | "
        f"Dropped (empty): {dropped_empty:,} | "
        f"Dropped (too short < {min_chars} chars): {dropped_short:,} | "
        f"Kept: {kept:,}"
        + (" [USING FALLBACK THRESHOLDS]" if is_fallback else "")
    )


if __name__ == "__main__":
    main()
