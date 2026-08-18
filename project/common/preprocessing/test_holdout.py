"""
test_holdout.py  (shared ??? use --lang flag)
--------------------------------------------
Stage 4: Test Holdout + Freeze.

Creates and freezes the test set from the raw document pool BEFORE any cleaning,
deduplication, or tokenizer training occurs.

Why this order matters
----------------------
If you clean/dedup first and select test documents afterward:
  - Cleaning decisions may be influenced by what ends up in the test set.
  - Deduplication may group training and test documents into the same cluster,
    causing leakage when the cluster representative is chosen.
  - The tokenizer's vocabulary is trained on cleaned data ??? if test documents
    inform that cleaning, vocabulary selection is contaminated.

By freezing the test set immediately after raw inspection, we guarantee:
  1. No cleaning step sees test text before it is frozen.
  2. Decontamination (stage 14) can remove any training-candidate document
     that resembles a test document.
  3. The tokenizer is trained only on non-test data.

Stratification
--------------
The holdout is stratified by source_type AND collection_method to ensure the
test set reflects the real diversity of the corpus ??? not forced to mirror
training proportions, but representative.

Output
------
  - <lang>/data/test_holdout/test.jsonl   ??? frozen test documents
  - Manifest rows updated: status=retained, split=test for test docs;
    drop_stage=test_holdout, drop_reason=held_out_test_set for remaining
    docs is NOT applied (those docs proceed to cleaning).

Usage
-----
  python test_holdout.py --lang hindi --repo-root . --input-dir hindi/data/raw/
  python test_holdout.py --lang nepali --repo-root .
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import load_manifest, update_rows, manifest_path

# Sentinel file that marks the holdout as frozen
FREEZE_SENTINEL = "FROZEN.txt"


def is_frozen(holdout_dir: Path) -> bool:
    """
    Check whether the test holdout directory has been frozen.

    Parameters
    ----------
    holdout_dir : Path

    Returns
    -------
    bool
    """
    return (holdout_dir / FREEZE_SENTINEL).exists()


def freeze_holdout(holdout_dir: Path, doc_ids: list[str]) -> None:
    """
    Write the freeze sentinel file, marking the holdout as immutable.

    Parameters
    ----------
    holdout_dir : Path
    doc_ids : list[str]
        Doc IDs included in this holdout (written to sentinel for auditability).
    """
    sentinel_path = holdout_dir / FREEZE_SENTINEL
    with open(sentinel_path, "w", encoding="utf-8") as f:
        f.write("THIS DIRECTORY IS FROZEN.\n")
        f.write("Do not modify, re-run, or add to this holdout after creation.\n")
        f.write("Any modification invalidates the test set.\n\n")
        f.write(f"Frozen doc_ids ({len(doc_ids)} total):\n")
        for doc_id in sorted(doc_ids):
            f.write(f"  {doc_id}\n")
    print(f"[test_holdout] Freeze sentinel written: {sentinel_path}")


def collect_all_raw_docs(raw_dir: Path, text_key: str = "text") -> list[dict]:
    """
    Collect all documents from all JSONL files under raw_dir.

    Parameters
    ----------
    raw_dir : Path
        Root raw data directory (searches recursively for *.jsonl files).
    text_key : str

    Returns
    -------
    list[dict]
        List of document dicts, each with at least doc_id and text.
    """
    docs: list[dict] = []
    jsonl_files = sorted(raw_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"[test_holdout] WARNING: No JSONL files found under {raw_dir}", file=sys.stderr)
        return docs

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  reading {fpath.name}", unit="doc"):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if doc.get(text_key, "").strip():
                        docs.append(doc)
                except json.JSONDecodeError:
                    continue
    return docs


def stratified_holdout(
    docs: list[dict],
    fraction: float,
    stratify_by: list[str],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Draw a stratified holdout from docs, stratifying by specified fields.

    Parameters
    ----------
    docs : list[dict]
        All raw documents.
    fraction : float
        Fraction of total docs to include in the holdout.
    stratify_by : list[str]
        Document field names to stratify on.
    seed : int

    Returns
    -------
    tuple[list[dict], list[dict]]
        (holdout_docs, remaining_docs)
    """
    rng = random.Random(seed)

    # Group by stratum key
    strata: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        key = tuple(str(doc.get(f, "unknown")) for f in stratify_by)
        strata[key].append(doc)

    holdout: list[dict] = []
    remaining: list[dict] = []

    for key, stratum_docs in strata.items():
        rng.shuffle(stratum_docs)
        n_holdout = max(1, int(len(stratum_docs) * fraction))
        holdout.extend(stratum_docs[:n_holdout])
        remaining.extend(stratum_docs[n_holdout:])

    return holdout, remaining


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Create and freeze the test holdout from the raw pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--force", action="store_true",
                        help="Re-create holdout even if already frozen (DANGEROUS: "
                             "invalidates any prior test-set usage). Requires --force.")
    return parser.parse_args()


def main() -> None:
    """
    Create the stratified test holdout from all raw documents and freeze it.

    If the holdout directory already contains a FROZEN.txt sentinel, this
    script exits with an error unless --force is passed (which is intentionally
    designed to be inconvenient, since re-creating the test set after any
    pipeline work has started is a serious methodological error).
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    lang = args.lang

    config_path = repo_root / lang / "configs" / "data_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    holdout_cfg  = config.get("test_holdout", {})
    fraction     = holdout_cfg.get("fraction", 0.02)
    seed         = holdout_cfg.get("seed", 42)
    stratify_by  = holdout_cfg.get("stratify_by", ["source_type", "collection_method"])

    holdout_dir = repo_root / lang / "data" / "test_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    # ?????? Guard against re-creation ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    if is_frozen(holdout_dir) and not args.force:
        print(
            f"[test_holdout] ABORT: Holdout is already frozen at {holdout_dir}.\n"
            f"  Re-creating the test set after pipeline work has started is a "
            f"methodological error.\n"
            f"  If you are certain, pass --force (this invalidates all prior work).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.force:
        print("[test_holdout] WARNING: --force passed; overwriting existing holdout!")

    # ?????? Collect all raw docs ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    raw_dir = repo_root / lang / "data" / "raw"
    print(f"\n[test_holdout] Collecting all raw documents from {raw_dir}???")
    all_docs = collect_all_raw_docs(raw_dir, text_key=config.get("text_key", "text"))
    print(f"  Total raw documents: {len(all_docs):,}")

    if not all_docs:
        print("[test_holdout] ERROR: No documents found. Run download scripts first.", file=sys.stderr)
        sys.exit(1)

    # ?????? Stratified holdout ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    print(f"\n[test_holdout] Drawing stratified holdout "
          f"(fraction={fraction}, seed={seed}, stratify_by={stratify_by})???")
    holdout_docs, remaining_docs = stratified_holdout(
        all_docs, fraction, stratify_by, seed
    )
    print(f"  Holdout: {len(holdout_docs):,} docs  |  Remaining: {len(remaining_docs):,} docs")

    # ?????? Write frozen test JSONL ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    test_jsonl_path = holdout_dir / "test.jsonl"
    holdout_doc_ids = []
    with open(test_jsonl_path, "w", encoding="utf-8") as fout:
        for doc in tqdm(holdout_docs, desc="  writing test_holdout", unit="doc"):
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            holdout_doc_ids.append(str(doc.get("doc_id", "")))

    # ?????? Update manifest ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    print("[test_holdout] Updating manifest with test-split assignments???")
    updates = {doc_id: {"split": "test", "status": "retained"}
               for doc_id in holdout_doc_ids}
    n_updated = update_rows(lang, updates, str(repo_root))
    print(f"  Updated {n_updated} manifest rows with split=test.")

    # ?????? Freeze ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    freeze_holdout(holdout_dir, holdout_doc_ids)

    # ?????? Composition summary ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    source_types    = {}
    coll_methods    = {}
    for doc in holdout_docs:
        st = doc.get("source_type", "unknown")
        cm = doc.get("collection_method", "unknown")
        source_types[st] = source_types.get(st, 0) + 1
        coll_methods[cm] = coll_methods.get(cm, 0) + 1

    print(f"\n[test_holdout] Holdout composition:")
    print(f"  By source_type:        {source_types}")
    print(f"  By collection_method:  {coll_methods}")
    print(f"\n  Test JSONL -> {test_jsonl_path}")
    print(f"  Remaining docs for pipeline: {len(remaining_docs):,}")


if __name__ == "__main__":
    main()
