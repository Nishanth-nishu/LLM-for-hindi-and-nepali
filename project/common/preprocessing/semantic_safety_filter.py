"""
semantic_safety_filter.py  (NEW)
---------------------------------
Stage 13: Semantic / Topic / Safety Filtering.

This is a CONSERVATIVE filter ??? the goal is to remove clearly unsafe or
completely irrelevant material, NOT to aggressively prune the corpus by topic.

For Nepali specifically, extreme caution is required: the corpus is small,
and aggressive filtering destroys irreplaceable data.  Every removal category
must be manually reviewed on ??? min_manual_review_fraction of the corpus
(configured per language in filtering_thresholds.yaml) before being trusted.

Implementation
--------------
This stage uses keyword/pattern matching only ??? NO pretrained language model.
Pattern lists are in the PATTERNS dict below and must be manually reviewed and
extended for each language.  They serve as a first-pass filter; all removals
are logged so a human can audit them.

Categories
----------
  1. explicit_sexual_content   ??? explicit sexual terms (Devanagari + common transliterations)
  2. extreme_violence          ??? explicit gore / graphic violence terms
  3. hate_speech               ??? slurs, incitement terms (must be reviewed carefully)
  4. spam_gibberish            ??? high-frequency, low-entropy text (all-caps runs,
                                 repeated chars, very long "words")

All dropped documents are written to a separate rejected_<lang>.jsonl file
for manual audit ??? they are NOT silently discarded.

Usage
-----
  python semantic_safety_filter.py \\
      --lang hindi \\
      --input hindi/data/dedup/near_deduped.jsonl \\
      --output hindi/data/dedup/safety_filtered.jsonl \\
      --repo-root .
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import update_rows

# ????????? Pattern lists ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# These are intentionally minimal seed lists.
# YOU MUST REVIEW AND EXTEND THESE for your specific corpus.
# Do not add patterns without manually verifying they catch genuinely
# harmful content and do not catch legitimate text.

_PATTERNS: dict[str, list[re.Pattern]] = {
    "explicit_sexual_content": [
        re.compile(p, re.IGNORECASE | re.UNICODE)
        for p in [
            r"??????????????????",           # obscene (Hindi)
            r"????????? ????????????????????????",      # sexual harassment
            r"???????????????",            # porn transliteration
            r"????????????????????????",         # obscenity
            # Add Nepali equivalents as needed:
            r"??????????????????",           # Nepali form
        ]
    ],
    "extreme_violence": [
        re.compile(p, re.IGNORECASE | re.UNICODE)
        for p in [
            r"????????? ???????????????",        # beheading
            r"???????????????????????????",        # massacre
            # Extend with carefully reviewed terms
        ]
    ],
    "hate_speech": [
        # CAUTION: Hate speech patterns are highly context-dependent.
        # These are placeholder patterns only ??? review every removal.
        re.compile(p, re.IGNORECASE | re.UNICODE)
        for p in []             # intentionally empty ??? fill after manual review
    ],
    "spam_gibberish": [
        re.compile(p, re.UNICODE)
        for p in [
            r"(.)\1{20,}",       # any character repeated 20+ times
            r"[A-Z]{50,}",       # 50+ consecutive uppercase ASCII
        ]
    ],
}


def classify_document(text: str, enabled_categories: list[str]) -> tuple[bool, str]:
    """
    Check whether a document matches any enabled safety filter category.

    Parameters
    ----------
    text : str
    enabled_categories : list[str]

    Returns
    -------
    tuple[bool, str]
        (should_drop, category) ??? category is '' if document is clean.
    """
    for category in enabled_categories:
        patterns = _PATTERNS.get(category, [])
        for pattern in patterns:
            if pattern.search(text):
                return True, category
    return False, ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 13: Conservative semantic/safety filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--write-rejected", action="store_true", default=True,
                        help="Write rejected docs to rejected_<lang>.jsonl for audit.")
    return parser.parse_args()


def main() -> None:
    """
    Apply conservative safety filter and write audit log of all removals.

    Logs per-category drop counts.  Prints a warning if the filter is
    removing more than 1% of documents (suggests patterns are too broad).
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    thresh_path = repo_root / args.lang / "configs" / "filtering_thresholds.yaml"
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    safety_cfg = thresh_cfg.get("semantic_safety", {})
    categories_cfg = safety_cfg.get("categories", [])
    enabled_categories = [
        c["name"] for c in categories_cfg
        if c.get("enabled", True) and c.get("action", "drop") == "drop"
    ]

    if not safety_cfg.get("enabled", True):
        print("[semantic_safety] Filtering disabled in config. Passing all docs through.")
        # Just copy input to output
        import shutil
        shutil.copy(args.input, args.output)
        return

    if args.lang == "nepali":
        review_frac = safety_cfg.get("min_manual_review_fraction", 0.02)
        print(
            f"[semantic_safety] NEPALI: Manual review of {review_frac:.0%} of retained docs "
            f"is required before this stage is considered validated. "
            f"See filtering_thresholds.yaml: semantic_safety.min_manual_review_fraction"
        )

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    rejected_path = output_path.parent / f"rejected_{args.lang}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    drop_by_cat: dict[str, int] = {}
    drop_updates: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(rejected_path, "w", encoding="utf-8") as frej:

        for line in tqdm(fin, desc=f"[semantic_safety] {args.lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text   = doc.get(args.text_key, "")
            doc_id = str(doc.get("doc_id", ""))

            should_drop, category = classify_document(text, enabled_categories)

            if should_drop:
                drop_by_cat[category] = drop_by_cat.get(category, 0) + 1
                drop_updates[doc_id] = {
                    "status":     "dropped",
                    "drop_stage": "semantic_safety",
                    "drop_reason": f"safety_category:{category}",
                }
                frej.write(json.dumps(
                    {**doc, "_rejected_category": category},
                    ensure_ascii=False
                ) + "\n")
            else:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1

    if drop_updates:
        n = update_rows(args.lang, drop_updates, str(repo_root))
        print(f"[semantic_safety] Updated {n} manifest rows.")

    total_dropped = total - kept
    if total > 0 and total_dropped / total > 0.01:
        print(
            f"[semantic_safety] WARNING: Filter removed {total_dropped/total:.2%} of documents "
            f"(> 1%). Review patterns ??? they may be too broad.",
            file=sys.stderr,
        )

    print(
        f"\n[semantic_safety] Done.  Input: {total:,} | Kept: {kept:,} | "
        f"Dropped: {total_dropped:,}"
    )
    print(f"  Rejected docs written to: {rejected_path}")
    print("  Drop by category:")
    for cat, cnt in sorted(drop_by_cat.items(), key=lambda x: -x[1]):
        print(f"    {cat:<35s}: {cnt:,}")
    print(
        "\n  REQUIRED: Manually audit the rejected docs in rejected_<lang>.jsonl "
        "before finalizing this stage."
    )


if __name__ == "__main__":
    main()
