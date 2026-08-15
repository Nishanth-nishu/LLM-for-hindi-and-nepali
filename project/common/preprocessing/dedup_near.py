"""
dedup_near.py
-------------
Step 5b of the cleaning pipeline: near-duplicate removal using MinHash + LSH.

Algorithm
---------
1. For each document, compute a set of character n-gram shingles (default n=5).
2. Build a MinHash signature for each shingle set using ``datasketch.MinHash``.
3. Insert all signatures into a ``datasketch.MinHashLSH`` index.
4. For each document, query the LSH index for near-duplicates (Jaccard > threshold).
5. Within each near-duplicate cluster, keep the longest document (most content)
   and discard the rest.

This step is the most memory- and compute-intensive part of the pipeline.
For large corpora (>1M documents) consider running on a machine with 32+ GB RAM
or processing in shards.

Complexity notes
----------------
- MinHash with ``num_perm=128`` gives ~2% error on Jaccard estimates.
- LSH with threshold=0.8 means documents sharing >80% of 5-grams are
  considered near-duplicates ??? appropriate for catching wire-copy republication.
- Time complexity: O(N * L / b) where L = text length, b = LSH bands.

Usage (CLI)
-----------
  python dedup_near.py \\
      --lang hindi \\
      --input /path/to/exact_deduped.jsonl \\
      --output /path/to/near_deduped.jsonl \\
      --threshold 0.8 \\
      --num-perm 128 \\
      --ngram 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Generator

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------

def char_ngram_shingles(text: str, n: int = 5) -> set[bytes]:
    """
    Generate the set of character n-gram shingles from a text string.

    Character n-grams are used rather than word n-grams because they are more
    robust to minor OCR errors and tokenisation differences.

    Parameters
    ----------
    text : str
        Input text.
    n : int
        Shingle size (number of characters per shingle).

    Returns
    -------
    set[bytes]
        Set of UTF-8 encoded n-gram byte strings.
    """
    # Compact whitespace so position-only differences don't create new shingles
    text = " ".join(text.split())
    return {text[i: i + n].encode("utf-8") for i in range(max(0, len(text) - n + 1))}


def build_minhash(shingles: set[bytes], num_perm: int = 128) -> MinHash:
    """
    Build a MinHash object from a set of shingles.

    Parameters
    ----------
    shingles : set[bytes]
        Set of shingle byte strings.
    num_perm : int
        Number of permutation functions (higher = more accurate, more memory).

    Returns
    -------
    MinHash
        datasketch MinHash object.
    """
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle)
    return m


# ---------------------------------------------------------------------------
# Core deduplication logic
# ---------------------------------------------------------------------------

def deduplicate_near(
    input_path: Path,
    output_path: Path,
    text_key: str = "text",
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 5,
    lang: str = "",
) -> dict:
    """
    Remove near-duplicate documents using MinHash LSH.

    Two-pass algorithm:
      Pass 1 ??? build MinHash for every document and find duplicate clusters via LSH.
      Pass 2 ??? stream documents again, keep only the elected representative
               (longest document) from each cluster.

    Parameters
    ----------
    input_path : Path
        Input JSONL file (output of dedup_exact.py).
    output_path : Path
        Output JSONL file.
    text_key : str
        JSON key containing document text.
    threshold : float
        Jaccard similarity threshold above which two docs are near-duplicates.
    num_perm : int
        Number of MinHash permutations.
    ngram : int
        Character n-gram size for shingling.
    lang : str
        Language label for progress-bar display.

    Returns
    -------
    dict
        Statistics: {total, kept, dropped, clusters}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Pass 1: build LSH index and find cluster representatives
    # -----------------------------------------------------------------------
    print(f"[dedup_near] Pass 1: Building MinHash signatures (n={ngram}, perm={num_perm}, threshold={threshold})")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    # doc_id ??? (char_count, minhash)
    doc_meta: dict[str, tuple[int, MinHash]] = {}

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"[dedup_near pass1] {lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = str(doc.get("doc_id", id(doc)))
            text = doc.get(text_key, "")
            shingles = char_ngram_shingles(text, n=ngram)
            if not shingles:
                continue

            m = build_minhash(shingles, num_perm=num_perm)
            doc_meta[doc_id] = (len(text), m)

    # Insert into LSH and build clusters
    # We process in order to give stable, reproducible cluster membership
    print(f"[dedup_near] Building LSH index over {len(doc_meta)} documents???")

    # elected_rep: maps each doc_id to the representative doc_id for its cluster
    elected_rep: dict[str, str] = {}

    for doc_id, (char_count, m) in tqdm(
        doc_meta.items(), desc="[dedup_near LSH]", unit="doc"
    ):
        neighbors = lsh.query(m)  # returns doc_ids already in index

        if not neighbors:
            # No near-duplicate found ??? this doc starts a new cluster
            lsh.insert(doc_id, m)
            elected_rep[doc_id] = doc_id
        else:
            # Find the longest document in this cluster (including current doc)
            candidates = neighbors + [doc_id]
            best = max(candidates, key=lambda d: doc_meta[d][0])

            if doc_id not in lsh:
                lsh.insert(doc_id, m)

            # All candidates in this cluster point to the same best rep
            for cand in candidates:
                elected_rep[cand] = elected_rep.get(best, best)

    # Determine which doc_ids to keep (those that are their own representative)
    to_keep: set[str] = {
        doc_id for doc_id, rep in elected_rep.items() if rep == doc_id
    }

    # -----------------------------------------------------------------------
    # Pass 2: write kept documents
    # -----------------------------------------------------------------------
    print(f"[dedup_near] Pass 2: Writing {len(to_keep)} / {len(doc_meta)} documents???")

    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"[dedup_near pass2] {lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            doc_id = str(doc.get("doc_id", ""))

            if doc_id in to_keep:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1

    dropped = total - kept
    num_clusters = len({rep for rep in elected_rep.values()})

    return {
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "clusters": num_clusters,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Step 5b: Near-duplicate removal using MinHash + LSH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Jaccard similarity threshold for near-duplicates.")
    parser.add_argument("--num-perm", type=int, default=128,
                        help="Number of MinHash permutations (accuracy vs. memory).")
    parser.add_argument("--ngram", type=int, default=5,
                        help="Character n-gram size for shingling.")
    return parser.parse_args()


def main() -> None:
    """Run near-deduplication and log survival statistics."""
    args = parse_args()

    stats = deduplicate_near(
        input_path=Path(args.input),
        output_path=Path(args.output),
        text_key=args.text_key,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram=args.ngram,
        lang=args.lang,
    )

    print(
        f"\n[dedup_near] Done. "
        f"Input: {stats['total']} | Kept: {stats['kept']} | "
        f"Dropped (near-dups): {stats['dropped']} | "
        f"Clusters found: {stats['clusters']}"
    )


if __name__ == "__main__":
    main()
