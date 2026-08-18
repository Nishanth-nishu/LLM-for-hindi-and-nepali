"""
manifest_utils.py  (v2)
------------------------
Utilities for the per-language manifest CSV ??? the single source of provenance
truth for every document from raw acquisition through the final corpus.

v2 schema adds 12 new columns over v1:
  - Full language-ID metadata (detected code + confidence)
  - Raw size metrics (char count, word count, estimated token count)
  - SHA-256 hash for dedup cross-reference
  - Pipeline status tracking (status, drop_stage, drop_reason, split)
  - Parent-document link for OCR pages / paragraph children

Schema (20 columns)
-------------------
doc_id               unique document identifier
source_id            identifier for the source dataset/site (e.g. 'hi_wiki')
source_name          human-readable name (e.g. 'Hindi Wikipedia 20231101')
source_type          wikipedia | hf_corpus | scrape | ocr | transcription
collection_method    manual | downloaded
url_or_path          URL (downloaded/scraped) or local file path (OCR/transcription)
license_note         SPDX identifier or free-text license description
collection_date      ISO-8601 date
language_expected    BCP-47 / ISO 639 code we expect (e.g. 'hi', 'ne')
language_detected    code returned by language-ID classifier
language_confidence  classifier confidence score (0???1)
raw_char_count       character count of raw (pre-cleaning) text
raw_word_count       whitespace-word count of raw text
raw_token_estimate   crude estimate: raw_char_count / 4.5 (for planning only)
manual_or_downloaded alias for collection_method (kept for backward compat.)
parent_document_id   for OCR pages from a PDF or paragraph-level children
sha256               SHA-256 hex digest of normalised text (set after normalise step)
status               retained | dropped
drop_stage           pipeline stage that dropped this doc (null if retained)
drop_reason          free-text reason (null if retained)
split                train | val | test | null (set after splitting)

Usage
-----
  from manifest_utils import init_manifest, append_rows, make_row, update_rows
"""

import csv
import hashlib
from datetime import date
from pathlib import Path
from typing import Optional

# ????????? Column definitions ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

MANIFEST_COLUMNS: list[str] = [
    "doc_id",
    "source_id",
    "source_name",
    "source_type",
    "collection_method",
    "url_or_path",
    "license_note",
    "collection_date",
    "language_expected",
    "language_detected",
    "language_confidence",
    "raw_char_count",
    "raw_word_count",
    "raw_token_estimate",
    "manual_or_downloaded",
    "parent_document_id",
    "sha256",
    "status",
    "drop_stage",
    "drop_reason",
    "split",
]

VALID_SOURCE_TYPES       = {"wikipedia", "hf_corpus", "scrape", "ocr", "transcription"}
VALID_COLLECTION_METHODS = {"manual", "downloaded"}
VALID_STATUSES           = {"retained", "dropped", "needs_manual_review"}
VALID_SPLITS             = {"train", "val", "test", None, ""}

LANG_EXPECTED: dict[str, str] = {
    "hindi":  "hi",
    "nepali": "ne",
}


# ????????? Path helpers ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def manifest_path(lang: str, repo_root: str = ".") -> Path:
    """
    Return the canonical manifest CSV path for a given language.

    Parameters
    ----------
    lang : str
        Language directory name (e.g. 'hindi', 'nepali').
    repo_root : str
        Absolute or relative path to the project root directory.

    Returns
    -------
    Path
    """
    return Path(repo_root) / lang / "data" / "manifest.csv"


def init_manifest(lang: str, repo_root: str = ".") -> Path:
    """
    Create an empty manifest CSV with the correct 20-column header if absent.

    Idempotent: calling on an existing manifest leaves it unchanged.

    Parameters
    ----------
    lang : str
    repo_root : str

    Returns
    -------
    Path
        Path to the manifest file.
    """
    path = manifest_path(lang, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
        print(f"[manifest_utils] Initialised manifest: {path}")
    return path


# ????????? Load / query ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def load_manifest(lang: str, repo_root: str = ".") -> list[dict]:
    """
    Load all manifest rows into a list of dicts.

    Parameters
    ----------
    lang : str
    repo_root : str

    Returns
    -------
    list[dict]
        Empty list if manifest does not yet exist.
    """
    path = manifest_path(lang, repo_root)
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def existing_doc_ids(lang: str, repo_root: str = ".") -> set:
    """
    Return the set of doc_ids already recorded in the manifest.

    Used by ingestion scripts to enforce idempotency.

    Parameters
    ----------
    lang : str
    repo_root : str

    Returns
    -------
    set[str]
    """
    return {row["doc_id"] for row in load_manifest(lang, repo_root)}


# ????????? Write ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def append_rows(lang: str, rows: list[dict], repo_root: str = ".") -> None:
    """
    Append one or more validated rows to the manifest CSV.

    All rows are validated before any write occurs to prevent partial writes.

    Parameters
    ----------
    lang : str
    rows : list[dict]
        Each dict must contain all 20 MANIFEST_COLUMNS keys.
    repo_root : str

    Raises
    ------
    ValueError
        On missing columns or invalid enum values.
    """
    if not rows:
        return
    for row in rows:
        _validate_row(row)
    path = init_manifest(lang, repo_root)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writerows(rows)


def update_rows(
    lang: str,
    updates: dict[str, dict],
    repo_root: str = ".",
) -> int:
    """
    Update specific fields in existing manifest rows, identified by doc_id.

    This rewrites the entire file ??? only use for bulk updates (e.g. setting
    drop_stage after a pipeline stage completes on the full corpus).

    Parameters
    ----------
    lang : str
    updates : dict[str, dict]
        Maps doc_id ??? {field: new_value} for fields to update.
    repo_root : str

    Returns
    -------
    int
        Number of rows actually updated.
    """
    rows = load_manifest(lang, repo_root)
    updated = 0
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id in updates:
            row.update(updates[doc_id])
            updated += 1

    path = manifest_path(lang, repo_root)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return updated


# ????????? Row construction ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def make_row(
    doc_id: str,
    source_id: str,
    source_name: str,
    source_type: str,
    collection_method: str,
    url_or_path: str,
    license_note: str,
    lang: str,
    raw_char_count: int,
    raw_word_count: int = 0,
    language_detected: str = "",
    language_confidence: float = 0.0,
    parent_document_id: str = "",
    sha256: str = "",
    status: str = "retained",
    drop_stage: str = "",
    drop_reason: str = "",
    split: str = "",
    collection_date: Optional[str] = None,
) -> dict:
    """
    Construct a fully-populated manifest row dict with all 20 columns.

    Parameters
    ----------
    doc_id : str
    source_id : str
        Short identifier for the source dataset, e.g. 'hi_wiki'.
    source_name : str
    source_type : str
        One of VALID_SOURCE_TYPES.
    collection_method : str
        One of VALID_COLLECTION_METHODS.
    url_or_path : str
    license_note : str
    lang : str
        Language key ('hindi' or 'nepali') ??? used to fill language_expected.
    raw_char_count : int
    raw_word_count : int
    language_detected : str
    language_confidence : float
    parent_document_id : str
        doc_id of the parent document (for OCR pages, paragraph children).
    sha256 : str
        SHA-256 of normalised text (filled in after normalise step).
    status : str
        'retained' or 'dropped'.
    drop_stage : str
    drop_reason : str
    split : str
        'train', 'val', 'test', or '' (unfilled).
    collection_date : str, optional
        ISO-8601 date; defaults to today.

    Returns
    -------
    dict
        Row dict with all 20 MANIFEST_COLUMNS keys.
    """
    row = {
        "doc_id":               doc_id,
        "source_id":            source_id,
        "source_name":          source_name,
        "source_type":          source_type,
        "collection_method":    collection_method,
        "url_or_path":          url_or_path,
        "license_note":         license_note,
        "collection_date":      collection_date or date.today().isoformat(),
        "language_expected":    LANG_EXPECTED.get(lang, lang),
        "language_detected":    language_detected,
        "language_confidence":  round(float(language_confidence), 4),
        "raw_char_count":       int(raw_char_count),
        "raw_word_count":       int(raw_word_count),
        "raw_token_estimate":   int(raw_char_count / 4.5),
        "manual_or_downloaded": collection_method,
        "parent_document_id":   parent_document_id,
        "sha256":               sha256,
        "status":               status,
        "drop_stage":           drop_stage,
        "drop_reason":          drop_reason,
        "split":                split,
    }
    _validate_row(row)
    return row


def sha256_of_text(text: str) -> str:
    """
    Compute SHA-256 hex digest of whitespace-normalised text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        64-character hex string.
    """
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ????????? Validation ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _validate_row(row: dict) -> None:
    """
    Validate a manifest row dict for required columns and enum constraints.

    Parameters
    ----------
    row : dict

    Raises
    ------
    ValueError
    """
    missing = [c for c in MANIFEST_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Manifest row missing columns: {missing}")

    if row["source_type"] not in VALID_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type: {row['source_type']!r}")

    if row["collection_method"] not in VALID_COLLECTION_METHODS:
        raise ValueError(f"Invalid collection_method: {row['collection_method']!r}")

    if row["status"] not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {row['status']!r}")


# ????????? Summary ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def stage_filter_table(lang: str, repo_root: str = ".") -> list[dict]:
    """
    Build the stage-by-stage filtering table from manifest drop_stage fields.

    Returns one row per pipeline stage with: stage, docs_in, docs_dropped,
    tokens_in_est, tokens_dropped_est.  Generated directly from the manifest
    so the report table is always consistent with actual pipeline outcomes.

    Parameters
    ----------
    lang : str
    repo_root : str

    Returns
    -------
    list[dict]
        One dict per stage.
    """
    rows = load_manifest(lang, repo_root)
    stages = [
        "raw_inspection", "test_holdout", "language_id", "normalization",
        "boilerplate", "quality_filter", "learned_quality",
        "dedup_exact", "dedup_paragraph", "dedup_near",
        "semantic_safety", "decontamination", "final",
    ]

    # Build cumulative sets of dropped doc_ids per stage
    dropped_by_stage: dict[str, list[dict]] = {s: [] for s in stages}
    for row in rows:
        ds = row.get("drop_stage", "")
        if ds and ds in dropped_by_stage:
            dropped_by_stage[ds].append(row)

    table = []
    docs_in = len(rows)
    tokens_in = sum(int(r.get("raw_token_estimate", 0)) for r in rows)

    for stage in stages:
        dropped = dropped_by_stage[stage]
        docs_dropped = len(dropped)
        tokens_dropped = sum(int(r.get("raw_token_estimate", 0)) for r in dropped)
        table.append({
            "stage":          stage,
            "docs_in":        docs_in,
            "docs_dropped":   docs_dropped,
            "tokens_in_est":  tokens_in,
            "tokens_dropped_est": tokens_dropped,
        })
        docs_in   -= docs_dropped
        tokens_in -= tokens_dropped

    return table


def manual_downloaded_summary(lang: str, repo_root: str = ".") -> dict:
    """
    Summarise manual vs. downloaded char/token counts across retained documents,
    and check both the percentage requirement (>=20% manual) and the assignment's
    absolute token targets (manual_target_tokens / downloaded_target_tokens /
    token_target from <lang>/configs/data_config.yaml, default 100M/400M/500M).

    Token counts here use the same crude char/4.5 estimate as raw_token_estimate
    elsewhere in the manifest (planning-grade, not the final SentencePiece count ???
    compute_stats.py's tokenizer stats give the exact figure once a tokenizer
    exists).

    Parameters
    ----------
    lang : str
    repo_root : str

    Returns
    -------
    dict
    """
    rows = [r for r in load_manifest(lang, repo_root) if r.get("status") == "retained"]
    total_chars     = sum(int(r.get("raw_char_count", 0)) for r in rows)
    manual_chars    = sum(int(r.get("raw_char_count", 0)) for r in rows
                          if r.get("collection_method") == "manual")
    downloaded_chars = total_chars - manual_chars
    manual_pct = manual_chars / total_chars * 100 if total_chars else 0.0

    manual_tokens_est     = round(manual_chars / 4.5)
    downloaded_tokens_est = round(downloaded_chars / 4.5)
    total_tokens_est      = round(total_chars / 4.5)

    targets = _load_token_targets(lang, repo_root)

    return {
        "total_chars":      total_chars,
        "manual_chars":     manual_chars,
        "downloaded_chars": downloaded_chars,
        "manual_pct":       round(manual_pct, 2),
        "compliant":        manual_pct >= 20.0,
        "manual_tokens_est":     manual_tokens_est,
        "downloaded_tokens_est": downloaded_tokens_est,
        "total_tokens_est":      total_tokens_est,
        "manual_target_tokens":     targets["manual_target_tokens"],
        "downloaded_target_tokens": targets["downloaded_target_tokens"],
        "token_target":             targets["token_target"],
        "manual_target_met":     manual_tokens_est >= targets["manual_target_tokens"],
        "downloaded_target_met": downloaded_tokens_est >= targets["downloaded_target_tokens"],
        "manual_tokens_remaining":     max(0, targets["manual_target_tokens"] - manual_tokens_est),
        "downloaded_tokens_remaining": max(0, targets["downloaded_target_tokens"] - downloaded_tokens_est),
    }


def _load_token_targets(lang: str, repo_root: str = ".") -> dict:
    """
    Read manual_target_tokens / downloaded_target_tokens / token_target from
    <lang>/configs/data_config.yaml, falling back to the assignment defaults
    (100M manual / 400M downloaded / 500M total) if the config or fields are
    missing (e.g. an older config not yet updated).
    """
    defaults = {
        "manual_target_tokens": 100_000_000,
        "downloaded_target_tokens": 400_000_000,
        "token_target": 500_000_000,
    }
    cfg_path = Path(repo_root) / lang / "configs" / "data_config.yaml"
    if not cfg_path.exists():
        return defaults
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    mc = cfg.get("manual_collection", {}) or {}
    return {
        "manual_target_tokens": int(mc.get("manual_target_tokens", defaults["manual_target_tokens"])),
        "downloaded_target_tokens": int(mc.get("downloaded_target_tokens", defaults["downloaded_target_tokens"])),
        "token_target": int(cfg.get("token_target", defaults["token_target"])),
    }


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Inspect or initialise a language manifest.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--action", choices=["init", "summary", "filter-table"],
                        default="summary")
    args = parser.parse_args()

    if args.action == "init":
        init_manifest(args.lang, args.repo_root)
    elif args.action == "summary":
        print(json.dumps(manual_downloaded_summary(args.lang, args.repo_root), indent=2))
    else:
        table = stage_filter_table(args.lang, args.repo_root)
        header = f"{'Stage':<20} {'DocsIn':>8} {'DocsDrop':>9} {'TokIn':>12} {'TokDrop':>10}"
        print(header)
        print("-" * len(header))
        for r in table:
            print(f"{r['stage']:<20} {r['docs_in']:>8} {r['docs_dropped']:>9} "
                  f"{r['tokens_in_est']:>12} {r['tokens_dropped_est']:>10}")
