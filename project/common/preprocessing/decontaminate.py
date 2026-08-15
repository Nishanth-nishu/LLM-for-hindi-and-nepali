"""
decontaminate.py  (NEW)
------------------------
Stage 14: Decontamination against the frozen test set.

After all cleaning and deduplication, any document in the training candidate
pool that is an exact or near-duplicate of a test document must be removed.
This is what makes the "freeze test first" decision in stage 4 actually pay off.

Method
------
  1. Load all documents from the frozen test holdout.
  2. Build MinHash signatures for every test document.
  3. Insert test signatures into an LSH index.
  4. Stream training candidates ??? query LSH for near-duplicates of each.
  5. Also check exact SHA-256 hashes against the test set.
  6. Drop any training candidate that matches (exact or near) a test document.

Parameters
----------
  Threshold: same Jaccard threshold as dedup_near (from filtering_thresholds.yaml
  decontamination.jaccard_threshold ??? defaults to dedup_near value if null).

Usage
-----
  python decontaminate.py \\
      --lang hindi \\
      --input hindi/data/dedup/near_deduped.jsonl \\
      --test-holdout hindi/data/test_holdout/test.jsonl \\
      --output hindi/data/decontaminated/decontaminated.jsonl \\
      --repo-root .
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import sha256_of_text, update_rows


# ????????? MinHash helpers ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _shingles(text: str, n: int = 5) -> set[bytes]:
    """Build character n-gram shingle set from text."""
    t = " ".join(text.split())
    return {t[i:i+n].encode("utf-8") for i in range(max(0, len(t) - n + 1))}


def _minhash(text: str, num_perm: int, ngram: int) -> MinHash:
    """Build MinHash from text."""
    m = MinHash(num_perm=num_perm)
    for s in _shingles(text, ngram):
        m.update(s)
    return m


# ????????? Main logic ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 14: Decontaminate training candidates against frozen test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True,
                        help="Training candidate JSONL (post dedup).")
    parser.add_argument("--test-holdout", required=True,
                        help="Frozen test JSONL.")
    parser.add_argument("--output", required=True,
                        help="Decontaminated output JSONL.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    """Build LSH from test set, then filter training candidates."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    # Load thresholds
    thresh_path = repo_root / args.lang / "configs" / "filtering_thresholds.yaml"
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    dcfg       = thresh_cfg.get("decontamination", {})
    jaccard_node = dcfg.get("jaccard_threshold", {})
    jaccard    = jaccard_node.get("value")
    fb         = jaccard_node.get("conservative_fallback", 0.80)
    is_fb      = False
    if jaccard is None:
        # Fall back to dedup_near threshold
        near_node = thresh_cfg.get("dedup_near", {}).get("jaccard_threshold", {})
        jaccard = near_node.get("value") or near_node.get("conservative_fallback", fb)
        is_fb = True
        print(
            f"[decontaminate] WARNING: decontamination.jaccard_threshold is null. "
            f"Using fallback: {jaccard}",
            file=sys.stderr,
        )

    num_perm = thresh_cfg.get("dedup_near", {}).get("num_perm", 128)
    ngram    = thresh_cfg.get("dedup_near", {}).get("n_gram_size", 5)

    test_path   = Path(args.test_holdout)
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ?????? Build test LSH + SHA-256 set ???????????????????????????????????????????????????????????????????????????????????????????????????????????????
    print(f"[decontaminate] Building test set index (threshold={jaccard})???")
    lsh = MinHashLSH(threshold=jaccard, num_perm=num_perm)
    test_sha256s: set[str] = set()
    test_docs: dict[str, MinHash] = {}

    with open(test_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="  indexing test", unit="doc")):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get(args.text_key, "")
            if not text.strip():
                continue
            test_sha256s.add(sha256_of_text(text))
            m = _minhash(text, num_perm, ngram)
            key = f"test_{i}"
            lsh.insert(key, m)
            test_docs[key] = m

    print(f"  Test set: {len(test_sha256s):,} unique documents indexed.")

    # ?????? Stream training candidates ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    print("[decontaminate] Filtering training candidates???")
    total = kept = removed_exact = removed_near = 0
    drop_updates: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="  decontaminate", unit="doc"):
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

            # Exact match
            h = sha256_of_text(text)
            if h in test_sha256s:
                removed_exact += 1
                drop_updates[doc_id] = {
                    "status":     "dropped",
                    "drop_stage": "decontamination",
                    "drop_reason": "exact_match_with_test_doc",
                }
                continue

            # Near match
            m = _minhash(text, num_perm, ngram)
            neighbors = lsh.query(m)
            if neighbors:
                removed_near += 1
                drop_updates[doc_id] = {
                    "status":     "dropped",
                    "drop_stage": "decontamination",
                    "drop_reason": f"near_duplicate_of_test:jaccard>={jaccard}",
                }
                continue

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    if drop_updates:
        n = update_rows(args.lang, drop_updates, str(repo_root))
        print(f"[decontaminate] Updated {n} manifest rows.")

    print(
        f"\n[decontaminate] Done.  "
        f"Input: {total:,} | Kept: {kept:,} | "
        f"Removed (exact): {removed_exact:,} | Removed (near): {removed_near:,}"
        + (" [FALLBACK THRESHOLD]" if is_fb else "")
    )


if __name__ == "__main__":
    main()
