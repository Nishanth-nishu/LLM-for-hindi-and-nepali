"""
train_tokenizer.py  (shared ??? use --lang flag)
-----------------------------------------------
Step 7 of the Phase 1 pipeline: SentencePiece tokenizer training + vocab sweep.

Design decisions
----------------
- **SentencePiece Unigram** (default): Unigram LM-based tokenization generalises
  better to morphologically rich Devanagari languages than BPE.  BPE is
  configurable via ``tokenizer_config.yaml`` for ablation.
- **Train only on train split**: tokenizer is trained exclusively on
  ``splits/train.jsonl`` to avoid any information leakage into val/test.
- **Vocab sweep**: trains with vocab sizes [16000, 24000, 32000] and picks
  the best size based on fertility (avg tokens/word) subject to UNK rate < 1%
  on the validation split.
- **Per-language independence**: this script is called once per language with
  ``--lang``; the two tokenizers share no parameters.

Fertility definition
--------------------
  fertility = total_tokens / total_whitespace_words (lower is better)
  A fertility of 1.0 means every word is a single token (ideal but unrealistic).
  Typical Devanagari BPE/Unigram fertility: 1.8???2.5.

Usage
-----
  python train_tokenizer.py \\
      --lang hindi \\
      --repo-root /path/to/project \\
      --model-type unigram \\
      --vocab-sizes 16000 24000 32000
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_texts_from_jsonl(jsonl_path: Path, text_key: str = "text") -> list[str]:
    """
    Load all text strings from a JSONL file.

    Parameters
    ----------
    jsonl_path : Path
        Path to the JSONL file.
    text_key : str
        JSON key containing the text.

    Returns
    -------
    list[str]
        List of document text strings.
    """
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get(text_key, "")
                if text.strip():
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def write_text_file_for_spm(texts: list[str], path: Path) -> None:
    """
    Write a plain-text file (one document per line) for SentencePiece training.

    SentencePiece's ``SentencePieceTrainer.train`` accepts a ``input`` path to
    a plain text file where each line is treated as a training sentence.

    Parameters
    ----------
    texts : list[str]
        List of document strings.
    path : Path
        Output path for the plain-text training corpus.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            # SentencePiece processes one sentence per line; write each doc
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    f.write(line + "\n")


def train_sentencepiece(
    corpus_path: Path,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "unigram",
    character_coverage: float = 0.9999,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
) -> None:
    """
    Train a SentencePiece model on the given corpus.

    Parameters
    ----------
    corpus_path : Path
        Path to the plain-text training corpus (one sentence per line).
    model_prefix : str
        Path prefix for output .model and .vocab files.
    vocab_size : int
        Target vocabulary size.
    model_type : str
        'unigram' (default) or 'bpe'.
    character_coverage : float
        Fraction of characters to cover in the vocabulary.
        0.9999 is appropriate for Devanagari (covers rare combining marks).
    pad_id, unk_id, bos_id, eos_id : int
        Special token IDs.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print(
            "[train_tokenizer] ERROR: sentencepiece not installed. "
            "Run: pip install sentencepiece",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            normalization_rule_name="nmt_nfkc",
            remove_extra_whitespaces=True,
            input_sentence_size=5_000_000,  # max training sentences
            shuffle_input_sentence=True,
        )
    except RuntimeError as e:
        print(f"[train_tokenizer] WARNING: Skipping vocab_size={vocab_size}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_fertility(sp_model, texts: list[str]) -> float:
    """
    Compute fertility: average number of tokens per whitespace-delimited word.

    A lower fertility means the tokenizer represents text more efficiently.
    Fertility = total_pieces / total_words.

    Parameters
    ----------
    sp_model : sentencepiece.SentencePieceProcessor
        Loaded SentencePiece model.
    texts : list[str]
        Sample texts to evaluate on (e.g., validation split).

    Returns
    -------
    float
        Mean fertility score across all texts.
    """
    total_pieces = 0
    total_words = 0

    for text in texts:
        words = text.split()
        if not words:
            continue
        pieces = sp_model.encode(text, out_type=str)
        total_pieces += len(pieces)
        total_words += len(words)

    return total_pieces / total_words if total_words > 0 else float("inf")


def compute_unk_rate(sp_model, texts: list[str]) -> float:
    """
    Compute the unknown-token (UNK) rate on a set of texts.

    UNK rate = fraction of tokens that are the UNK token.
    A rate > 1% typically indicates the vocabulary is too small.

    Parameters
    ----------
    sp_model : sentencepiece.SentencePieceProcessor
        Loaded SentencePiece model.
    texts : list[str]
        Sample texts to evaluate on.

    Returns
    -------
    float
        UNK rate in [0, 1].
    """
    unk_id = sp_model.unk_id()
    total_tokens = 0
    unk_tokens = 0

    for text in texts:
        ids = sp_model.encode(text, out_type=int)
        total_tokens += len(ids)
        unk_tokens += sum(1 for t in ids if t == unk_id)

    return unk_tokens / total_tokens if total_tokens > 0 else 0.0


def choose_best_vocab_size(
    sweep_results: list[dict],
    max_unk_rate: float = 0.01,
) -> int:
    """
    Choose the best vocabulary size from sweep results.

    Selection criterion:
      - Among vocab sizes with UNK rate < max_unk_rate,
        pick the one with the lowest fertility.
      - If none pass the UNK threshold, pick the largest vocab size.

    Parameters
    ----------
    sweep_results : list[dict]
        List of dicts with keys: vocab_size, fertility, unk_rate.
    max_unk_rate : float
        Maximum acceptable UNK rate (default 1%).

    Returns
    -------
    int
        The selected vocabulary size.
    """
    passing = [r for r in sweep_results if r["unk_rate"] < max_unk_rate]
    if not passing:
        # Fallback: use the largest vocab size to minimise UNK rate
        best = max(sweep_results, key=lambda r: r["vocab_size"])
        print(
            f"[choose_vocab] WARNING: No vocab size achieved UNK rate < {max_unk_rate:.1%}. "
            f"Falling back to largest size: {best['vocab_size']}"
        )
        return best["vocab_size"]

    best = min(passing, key=lambda r: r["fertility"])
    print(
        f"[choose_vocab] Selected vocab_size={best['vocab_size']} "
        f"(fertility={best['fertility']:.3f}, unk_rate={best['unk_rate']:.4%})"
    )
    return best["vocab_size"]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    train_texts: list[str],
    val_texts: list[str],
    vocab_sizes: list[int],
    output_dir: Path,
    model_type: str,
    lang: str,
) -> list[dict]:
    """
    Train tokenizers at multiple vocab sizes and evaluate on the validation split.

    Parameters
    ----------
    train_texts : list[str]
        Training documents.
    val_texts : list[str]
        Validation documents for fertility / UNK rate evaluation.
    vocab_sizes : list[int]
        List of vocabulary sizes to sweep over.
    output_dir : Path
        Directory to save sweep models and results.
    model_type : str
        'unigram' or 'bpe'.
    lang : str
        Language key (for logging).

    Returns
    -------
    list[dict]
        Sweep results ??? one dict per vocab_size with keys:
        vocab_size, fertility, unk_rate, model_path.
    """
    import sentencepiece as spm

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write training corpus to a temp file
    corpus_path = output_dir / "train_corpus.txt"
    print(f"[train_tokenizer] Writing SentencePiece training corpus -> {corpus_path}")
    write_text_file_for_spm(train_texts, corpus_path)

    results = []

    for vocab_size in vocab_sizes:
        print(f"\n[train_tokenizer] Training {model_type.upper()} tokenizer "
              f"vocab_size={vocab_size} for [{lang}]???")

        model_prefix = str(output_dir / f"spm_{lang}_{model_type}_v{vocab_size}")

        res = train_sentencepiece(
            corpus_path=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
        )

        model_file = Path(f"{model_prefix}.model")
        if not model_file.exists():
            print(f"  [train_tokenizer] Skipped evaluation for vocab_size={vocab_size} (model not generated).")
            continue

        # Load and evaluate
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_file))

        print(f"  Evaluating on {len(val_texts)} validation documents???")
        # Use a subset for speed (up to 5000 docs)
        eval_texts = val_texts[:5000]
        fertility = compute_fertility(sp, eval_texts)
        unk_rate = compute_unk_rate(sp, eval_texts)

        print(f"  vocab_size={vocab_size:6d} | fertility={fertility:.3f} | unk_rate={unk_rate:.4%}")

        results.append({
            "vocab_size": vocab_size,
            "fertility": round(fertility, 4),
            "unk_rate": round(unk_rate, 6),
            "model_path": f"{model_prefix}.model",
        })

    return results


def save_sweep_results(results: list[dict], output_path: Path) -> None:
    """
    Save sweep results to a CSV file for the report.

    Parameters
    ----------
    results : list[dict]
        Sweep results from run_sweep.
    output_path : Path
        CSV output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["vocab_size", "fertility", "unk_rate", "model_path"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[train_tokenizer] Sweep results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer with vocab-size sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"],
                        help="SentencePiece model type.")
    parser.add_argument("--vocab-sizes", nargs="+", type=int,
                        default=[16000, 24000, 32000],
                        help="Vocab sizes to sweep over.")
    parser.add_argument("--max-unk-rate", type=float, default=0.01,
                        help="Max UNK rate for vocab selection (default 1%%).")
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    """
    Run the full tokenizer training sweep for a language and save the best model.

    Steps:
    1. Load train and val splits from <lang>/data/splits/.
    2. Run vocab sweep at all specified sizes.
    3. Select best vocab size (min fertility, UNK rate < threshold).
    4. Copy best model to <lang>/tokenizer/vocab/.
    5. Save sweep results CSV to <lang>/tokenizer/analysis/.
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    lang = args.lang

    splits_dir = repo_root / lang / "data" / "splits"
    vocab_dir = repo_root / lang / "tokenizer" / "vocab"
    analysis_dir = repo_root / lang / "tokenizer" / "analysis"

    train_jsonl = splits_dir / "train.jsonl"
    val_jsonl = splits_dir / "val.jsonl"

    if not train_jsonl.exists():
        print(f"[train_tokenizer] ERROR: train.jsonl not found: {train_jsonl}", file=sys.stderr)
        print("  Run split_data.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[train_tokenizer] Loading train texts for [{lang}]???")
    train_texts = load_texts_from_jsonl(train_jsonl, text_key=args.text_key)
    print(f"  Loaded {len(train_texts):,} training documents.")

    print(f"[train_tokenizer] Loading val texts for [{lang}]???")
    val_texts = load_texts_from_jsonl(val_jsonl, text_key=args.text_key) if val_jsonl.exists() else []
    print(f"  Loaded {len(val_texts):,} validation documents.")

    # Run sweep in a temporary directory; copy winner to vocab_dir
    sweep_dir = analysis_dir / "sweep_models"

    results = run_sweep(
        train_texts=train_texts,
        val_texts=val_texts,
        vocab_sizes=args.vocab_sizes,
        output_dir=sweep_dir,
        model_type=args.model_type,
        lang=lang,
    )

    # Save sweep CSV
    save_sweep_results(results, analysis_dir / "sweep_results.csv")

    # Select best
    best_vocab_size = choose_best_vocab_size(results, max_unk_rate=args.max_unk_rate)
    best_result = next(r for r in results if r["vocab_size"] == best_vocab_size)

    # Copy best model to vocab/
    import shutil
    vocab_dir.mkdir(parents=True, exist_ok=True)
    src_model = Path(best_result["model_path"])
    src_vocab = src_model.with_suffix(".vocab")
    dst_model = vocab_dir / f"{lang}_{args.model_type}.model"
    dst_vocab = vocab_dir / f"{lang}_{args.model_type}.vocab"

    shutil.copy(src_model, dst_model)
    if src_vocab.exists():
        shutil.copy(src_vocab, dst_vocab)

    print(
        f"\n[train_tokenizer] ??? Best tokenizer saved:"
        f"\n  Model : {dst_model}"
        f"\n  Vocab : {dst_vocab}"
        f"\n  vocab_size={best_vocab_size} | "
        f"fertility={best_result['fertility']} | "
        f"unk_rate={best_result['unk_rate']:.4%}"
    )


if __name__ == "__main__":
    main()
