"""
compute_stats.py  (v2)
-----------------------
Computes all Phase 1 statistics for a single language and writes them to JSON.

Statistics computed
-------------------
  1. Stage-by-stage filtering table (from manifest drop_stage fields)
  2. Source-level table (from manifest source_name fields)
  3. Split-level counts (train/val/test docs + token counts)
  4. Manual vs. downloaded breakdown (on final training corpus)
  5. Tokenizer stats: vocab_size, avg_chars_per_token, UNK rate, fertility,
     compression_ratio, top-20 tokens, 5 example tokenizations
  6. Vocab sweep comparison table (from tokenizer/analysis/sweep_results.csv)

Output
------
  <lang>/data/filtering_stats.json ??? stage + source tables
  <lang>/data/stats.json           ??? all stats for generate_report.py

Usage
-----
  python compute_stats.py --lang hindi --repo-root . --output hindi/data/stats.json
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from manifest_utils import (load_manifest, stage_filter_table,
                             manual_downloaded_summary)


# ????????? Token counting ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def count_tokens_whitespace(text: str) -> int:
    """Count whitespace-delimited words as a fast token proxy."""
    return len(text.split())


def jsonl_stats(jsonl_path: Path, text_key: str = "text") -> dict:
    """
    Compute document and token counts from a JSONL file.

    Parameters
    ----------
    jsonl_path : Path
    text_key : str

    Returns
    -------
    dict
        n_docs, total_chars, total_tokens, avg_chars, avg_tokens.
    """
    if not jsonl_path.exists():
        return {"n_docs": 0, "total_chars": 0, "total_tokens": 0,
                "avg_chars": 0.0, "avg_tokens": 0.0}

    n_docs = total_chars = total_tokens = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  {jsonl_path.name}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get(text_key, "")
            n_docs      += 1
            total_chars += len(text)
            total_tokens += count_tokens_whitespace(text)

    return {
        "n_docs":       n_docs,
        "total_chars":  total_chars,
        "total_tokens": total_tokens,
        "avg_chars":    round(total_chars / n_docs, 1) if n_docs else 0.0,
        "avg_tokens":   round(total_tokens / n_docs, 1) if n_docs else 0.0,
    }


# ????????? Source-level table ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def source_level_table(lang: str, repo_root: Path) -> list[dict]:
    """
    Build a per-source statistics table from the manifest.

    Columns: source_name, raw_docs, raw_tokens_est, final_docs,
             final_tokens_est, manual.

    Parameters
    ----------
    lang : str
    repo_root : Path

    Returns
    -------
    list[dict]
    """
    rows = load_manifest(lang, str(repo_root))
    src_raw:   dict[str, dict] = defaultdict(lambda: {"docs": 0, "tokens": 0, "manual": False})
    src_final: dict[str, dict] = defaultdict(lambda: {"docs": 0, "tokens": 0})

    for row in rows:
        name   = row.get("source_name", "unknown")
        tokens = int(row.get("raw_token_estimate", 0))
        manual = row.get("collection_method") == "manual"
        src_raw[name]["docs"]   += 1
        src_raw[name]["tokens"] += tokens
        src_raw[name]["manual"] = src_raw[name]["manual"] or manual

        if row.get("status") == "retained":
            src_final[name]["docs"]   += 1
            src_final[name]["tokens"] += tokens

    table = []
    for name, raw in src_raw.items():
        final = src_final.get(name, {"docs": 0, "tokens": 0})
        table.append({
            "source_name":       name,
            "raw_docs":          raw["docs"],
            "raw_tokens_est":    raw["tokens"],
            "final_docs":        final["docs"],
            "final_tokens_est":  final["tokens"],
            "manual":            "Yes" if raw["manual"] else "No",
        })
    return sorted(table, key=lambda x: -x["raw_docs"])


# ????????? Tokenizer stats ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def compute_tokenizer_stats(
    sp_model_path: Path,
    val_jsonl: Path,
    text_key: str = "text",
    n_examples: int = 5,
    top_k: int = 20,
) -> dict:
    """
    Compute tokenizer evaluation metrics on the validation set.

    Metrics: vocab_size, avg_chars_per_token, UNK rate, fertility,
    compression_ratio, top-20 token frequencies, 5 example tokenizations.

    Parameters
    ----------
    sp_model_path : Path
    val_jsonl : Path
    text_key : str
    n_examples : int
    top_k : int

    Returns
    -------
    dict
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print("[compute_stats] WARNING: sentencepiece not installed.", file=sys.stderr)
        return {}

    if not sp_model_path.exists():
        print(f"[compute_stats] WARNING: Model not found: {sp_model_path}", file=sys.stderr)
        return {}

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    vocab_size   = sp.get_piece_size()
    unk_id       = sp.unk_id()
    total_tokens = total_chars = unk_tokens = total_words = 0
    token_counter: Counter = Counter()
    examples: list[dict]   = []

    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  tokenizer eval", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get(text_key, "")
            if not text.strip():
                continue

            pieces = sp.encode(text, out_type=str)
            ids    = sp.encode(text, out_type=int)
            total_tokens += len(pieces)
            total_chars  += len(text)
            total_words  += len(text.split())
            unk_tokens   += sum(1 for t in ids if t == unk_id)
            token_counter.update(pieces)

            if len(examples) < n_examples:
                sent = (text.split("???")[0].split(".")[0].strip())[:200]
                if sent:
                    examples.append({
                        "original": sent,
                        "tokens":   sp.encode(sent, out_type=str),
                        "n_tokens": len(sp.encode(sent, out_type=str)),
                    })

    avg_cpt    = total_chars  / total_tokens if total_tokens else 0.0
    unk_rate   = unk_tokens   / total_tokens if total_tokens else 0.0
    fertility  = total_tokens / total_words  if total_words  else 0.0
    comp_ratio = total_chars  / total_tokens if total_tokens else 0.0  # bytes/token ??? chars/token for UTF-8 BMP

    return {
        "vocab_size":          vocab_size,
        "avg_chars_per_token": round(avg_cpt, 3),
        "unk_rate":            round(unk_rate, 6),
        "unk_rate_pct":        round(unk_rate * 100, 4),
        "fertility":           round(fertility, 4),
        "compression_ratio":   round(comp_ratio, 3),
        "total_tokens_eval":   total_tokens,
        "top_tokens":          [{"token": t, "count": c}
                                for t, c in token_counter.most_common(top_k)],
        "examples":            examples,
    }


def load_sweep_results(analysis_dir: Path) -> list[dict]:
    """
    Load vocab-size sweep results CSV.

    Parameters
    ----------
    analysis_dir : Path

    Returns
    -------
    list[dict]
    """
    csv_path = analysis_dir / "sweep_results.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute all Phase 1 statistics for a language.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True,
                        help="Path to write stats.json.")
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    """Compute all statistics and save to JSON for report generation."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    lang = args.lang

    splits_dir   = repo_root / lang / "data" / "splits"
    vocab_dir    = repo_root / lang / "tokenizer" / "vocab"
    analysis_dir = repo_root / lang / "tokenizer" / "analysis"

    print(f"\n[compute_stats] Computing stats for [{lang}]???")

    # 1. Stage-by-stage filtering table
    print("[compute_stats] Building stage filter table from manifest???")
    filter_table = stage_filter_table(lang, str(repo_root))

    # 2. Source-level table
    print("[compute_stats] Building source table???")
    src_table = source_level_table(lang, repo_root)

    # 3. Split-level JSONL stats
    print("[compute_stats] Computing split-level stats???")
    split_stats = {}
    for split_name in ["train", "val", "test"]:
        split_stats[split_name] = jsonl_stats(
            splits_dir / f"{split_name}.jsonl", text_key=args.text_key
        )

    # 4. Manual/downloaded breakdown
    breakdown = manual_downloaded_summary(lang, str(repo_root))
    if not breakdown["compliant"]:
        print(
            f"[compute_stats] NON-COMPLIANT: Manual fraction is "
            f"{breakdown['manual_pct']:.1f}% (< 20% requirement). "
            f"More manual tokens needed: approx "
            f"{int(breakdown['total_chars'] * 0.20 - breakdown['manual_chars'])} chars.",
            file=sys.stderr,
        )

    # 5. Tokenizer stats
    print("[compute_stats] Computing tokenizer stats???")
    sp_models = list(vocab_dir.glob("*.model"))
    tokenizer_stats = {}
    if sp_models and (splits_dir / "val.jsonl").exists():
        tokenizer_stats = compute_tokenizer_stats(
            sp_models[0], splits_dir / "val.jsonl", text_key=args.text_key
        )

    # 6. Sweep results
    sweep_results = load_sweep_results(analysis_dir)

    # Assemble and save
    stats = {
        "lang":                lang,
        "filter_table":        filter_table,
        "source_table":        src_table,
        "splits":              split_stats,
        "manual_breakdown":    breakdown,
        "tokenizer":           tokenizer_stats,
        "vocab_sweep":         sweep_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Also write filtering_stats.json
    fs_path = repo_root / lang / "data" / "filtering_stats.json"
    with open(fs_path, "w", encoding="utf-8") as f:
        json.dump({"filter_table": filter_table, "source_table": src_table}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n[compute_stats] Stats saved ??? {out_path}")
    tr = split_stats.get("train", {})
    print(
        f"  Train docs    : {tr.get('n_docs', 0):,}\n"
        f"  Train tokens  : {tr.get('total_tokens', 0):,}\n"
        f"  Manual pct    : {breakdown.get('manual_pct', 0):.1f}%  "
        f"({'PASS' if breakdown.get('compliant') else 'FAIL ???20% required'})\n"
        f"  Vocab size    : {tokenizer_stats.get('vocab_size', 'N/A')}\n"
        f"  Fertility     : {tokenizer_stats.get('fertility', 'N/A')}\n"
        f"  UNK rate      : {tokenizer_stats.get('unk_rate_pct', 'N/A')}%"
    )


if __name__ == "__main__":
    main()
