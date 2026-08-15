"""
dedup_paragraph.py  (NEW)
--------------------------
Stage 11: Sub-document / Paragraph-level Deduplication.

Some sources repeat boilerplate paragraphs across otherwise-distinct documents
(e.g. news site bylines, legal disclaimers, copyrights, syndicated lede paragraphs).
This stage deduplicates at the paragraph level:

  1. Split each document into paragraphs (split on ???2 consecutive newlines).
  2. Hash each paragraph (SHA-256 of normalised text).
  3. Count how many documents each paragraph hash appears in.
  4. Remove paragraphs that appear in ??? min_paragraph_frequency documents
     AND have ??? min_paragraph_chars characters (to skip trivially short
     repeated lines that legitimate text may share).
  5. If a document's text becomes empty after paragraph removal, drop the doc.
  6. Track `parent_document_id` in the manifest for paragraph-level children
     (not applicable at this stage since we modify in place, but the doc_id
     is preserved for the source document).

Threshold setting (Hard Constraint 7)
--------------------------------------
The min_paragraph_frequency threshold must be set by:
  1. Computing the full paragraph-frequency distribution on the corpus.
  2. Manually inspecting paragraphs at various frequency thresholds
     (e.g. freq=5: are these legitimately boilerplate? freq=2: probably not).
  3. Recording the chosen value in filtering_thresholds.yaml.

Usage
-----
  python dedup_paragraph.py \\
      --lang hindi \\
      --input hindi/data/cleaned/quality_filtered.jsonl \\
      --output hindi/data/dedup/paragraph_deduped.jsonl \\
      --repo-root .
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import update_rows

_PARA_SEP = re.compile(r"\n{2,}")


def hash_paragraph(text: str) -> str:
    """
    Compute SHA-256 of whitespace-normalised paragraph text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def split_paragraphs(text: str) -> list[str]:
    """
    Split document text into paragraphs.

    Parameters
    ----------
    text : str

    Returns
    -------
    list[str]
        Non-empty paragraph strings.
    """
    return [p.strip() for p in _PARA_SEP.split(text) if p.strip()]


def load_threshold(thresh_cfg: dict, key: str, fallback: int, lang: str) -> tuple[int, bool]:
    """
    Load a paragraph dedup threshold from config.

    Parameters
    ----------
    thresh_cfg : dict
    key : str
    fallback : int
    lang : str

    Returns
    -------
    tuple[int, bool]
        (value, is_fallback)
    """
    node = thresh_cfg.get("dedup_paragraph", {}).get(key, {})
    val  = node.get("value")
    if val is None:
        print(
            f"[dedup_paragraph] WARNING: dedup_paragraph.{key}.value is null. "
            f"Using fallback: {fallback}",
            file=sys.stderr,
        )
        return fallback, True
    return int(val), False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 11: Paragraph-level deduplication.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    """
    Two-pass paragraph deduplication:
    Pass 1 ??? count paragraph hash frequencies across all documents.
    Pass 2 ??? remove boilerplate paragraphs; drop docs that become empty.
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    thresh_path = repo_root / args.lang / "configs" / "filtering_thresholds.yaml"
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    min_freq,  fb_freq  = load_threshold(thresh_cfg, "min_paragraph_frequency", 10, args.lang)
    min_chars, fb_chars = load_threshold(thresh_cfg, "min_paragraph_chars",     100, args.lang)
    any_fb = fb_freq or fb_chars

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ?????? Pass 1: count paragraph frequencies ??????????????????????????????????????????????????????????????????????????????????????????
    print(f"[dedup_paragraph] Pass 1: counting paragraph hashes???")
    para_freq: Counter = Counter()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="  pass1", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get(args.text_key, "")
            for para in split_paragraphs(text):
                if len(para) >= min_chars:
                    para_freq[hash_paragraph(para)] += 1

    boilerplate_hashes = {h for h, cnt in para_freq.items() if cnt >= min_freq}
    print(f"  Boilerplate paragraph hashes (freq???{min_freq}): {len(boilerplate_hashes):,}")

    # ?????? Pass 2: strip boilerplate paragraphs ???????????????????????????????????????????????????????????????????????????????????????
    print(f"[dedup_paragraph] Pass 2: stripping boilerplate paragraphs???")
    total = kept = dropped_docs = paras_removed = 0
    drop_updates: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="  pass2", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text   = doc.get(args.text_key, "")
            paras  = split_paragraphs(text)
            doc_id = str(doc.get("doc_id", ""))

            clean_paras = []
            for para in paras:
                if len(para) >= min_chars and hash_paragraph(para) in boilerplate_hashes:
                    paras_removed += 1
                else:
                    clean_paras.append(para)

            clean_text = "\n\n".join(clean_paras).strip()

            if not clean_text:
                dropped_docs += 1
                drop_updates[doc_id] = {
                    "status":     "dropped",
                    "drop_stage": "dedup_paragraph",
                    "drop_reason": "empty_after_paragraph_dedup",
                }
                continue

            doc[args.text_key] = clean_text
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    if drop_updates:
        n = update_rows(args.lang, drop_updates, str(repo_root))
        print(f"[dedup_paragraph] Updated {n} manifest rows.")

    print(
        f"\n[dedup_paragraph] Done.  "
        f"Input: {total:,} | Kept: {kept:,} | Dropped (empty): {dropped_docs:,} | "
        f"Paragraphs removed: {paras_removed:,}"
        + (" [FALLBACK THRESHOLDS]" if any_fb else "")
    )


if __name__ == "__main__":
    main()
