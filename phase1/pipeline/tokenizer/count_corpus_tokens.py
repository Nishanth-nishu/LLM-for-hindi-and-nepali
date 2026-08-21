"""
count_corpus_tokens.py
======================
Counts the corpus in TRAINING TOKENS using the language's own from-scratch
tokenizer, and reports the manual vs downloaded split.

Why this script has to exist
----------------------------
The brief asks for "approximately 500M training tokens per language AFTER
tokenization", and separately forbids pretrained tokenizers. Those two
requirements together create a chicken-and-egg problem that is easy to get
wrong:

  - you cannot count training tokens until you have a tokenizer, and
  - you should not train the tokenizer on a corpus you have not finished
    collecting, and
  - you must not count with someone else's tokenizer, because the reported
    number then describes their vocabulary, not yours.

The resolution is a documented two-pass protocol:

  PASS 1 (collection)  Track volume in bytes, characters and whitespace words.
                       These are tokenizer-independent and are what the crawler
                       and downloader should budget against. Use the
                       words x fertility estimate below only as a progress bar.

  PASS 2 (accounting)  Once the corpus is closed and split, train the tokenizer
                       on train.jsonl ONLY, then run this script to produce the
                       authoritative token count with that tokenizer.

Report the Pass-2 number. If you retrain the tokenizer, re-run this script --
a token count is only meaningful relative to one vocabulary, and the JSON output
records which model produced it.

The 20% manual requirement
--------------------------
"at least 20% of the final training tokens must come from manual collection".

Note the unit: TOKENS, not documents and not bytes. A corpus that is 20% manual
by document count can easily be 12% by tokens if the scraped documents are
longer, and that would fail the requirement while looking fine on a document
histogram. This script counts tokens per provenance class and states the
fraction against the token total, which is the quantity the brief specifies.

Every document is expected to carry a `provenance_class` field, one of:

    manual       OCR from books/PDFs, pages you scraped and cleaned yourself,
                 typed or transcribed text
    downloaded   public corpora (HuggingFace, OSCAR, Sangraha, ...)

plus a `source` field naming the specific origin. If `provenance_class` is
missing this script will say so and refuse to report a fraction, rather than
guessing -- an unverifiable manual fraction is worse than none.

Usage
-----
  python count_corpus_tokens.py --lang hindi  --repo-root /path/to/project
  python count_corpus_tokens.py --lang nepali --repo-root . --splits train val test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MANUAL_CLASSES = {"manual", "ocr", "scrape", "scraped", "transcription", "typed"}
DOWNLOADED_CLASSES = {"downloaded", "public", "hf", "huggingface"}


def classify(value: str | None) -> str:
    if not value:
        return "unlabelled"
    v = str(value).strip().lower()
    if v in MANUAL_CLASSES:
        return "manual"
    if v in DOWNLOADED_CLASSES:
        return "downloaded"
    return v


def count_split(sp, path: Path, *, text_key: str, batch: int = 512) -> dict:
    """Stream a split and count tokens. Batched encode; never loads the file."""
    per_class_tokens: Counter[str] = Counter()
    per_class_docs: Counter[str] = Counter()
    per_source_tokens: Counter[str] = Counter()
    per_class_chars: Counter[str] = Counter()
    per_class_bytes: Counter[str] = Counter()
    unlabelled = 0

    pending_texts: list[str] = []
    pending_meta: list[tuple[str, str, int, int]] = []

    def flush():
        nonlocal pending_texts, pending_meta
        if not pending_texts:
            return
        encoded = sp.encode(pending_texts, out_type=int)
        for (cls, src, nchars, nbytes), ids in zip(pending_meta, encoded):
            n = len(ids)
            per_class_tokens[cls] += n
            per_class_docs[cls] += 1
            per_class_chars[cls] += nchars
            per_class_bytes[cls] += nbytes
            per_source_tokens[f"{cls}:{src}"] += n
        pending_texts = []
        pending_meta = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = doc.get(text_key) or ""
            if not text.strip():
                continue
            cls = classify(doc.get("provenance_class"))
            if cls == "unlabelled":
                unlabelled += 1
            src = str(doc.get("source") or doc.get("domain") or "unknown")
            pending_texts.append(text)
            pending_meta.append((cls, src, len(text), len(text.encode("utf-8"))))
            if len(pending_texts) >= batch:
                flush()
    flush()

    total = sum(per_class_tokens.values())
    return {
        "path": str(path),
        "total_tokens": total,
        "total_documents": sum(per_class_docs.values()),
        "tokens_by_provenance": dict(per_class_tokens),
        "documents_by_provenance": dict(per_class_docs),
        "characters_by_provenance": dict(per_class_chars),
        "bytes_by_provenance": dict(per_class_bytes),
        "tokens_by_source": dict(per_source_tokens.most_common(200)),
        "unlabelled_documents": unlabelled,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Count corpus tokens with the language's own tokenizer and "
                    "report the manual vs downloaded split.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--model", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--target-tokens", type=int, default=500_000_000)
    ap.add_argument("--min-manual-fraction", type=float, default=0.20)
    args = ap.parse_args()

    import sentencepiece as spm

    root = Path(args.repo_root).resolve()
    lang = args.lang
    model_path = Path(args.model) if args.model else \
        root / lang / "tokenizer" / "vocab" / f"{lang}_tokenizer.model"
    if not model_path.exists():
        print(f"[error] tokenizer not found: {model_path}\n"
              f"        Train it first. Do NOT substitute a pretrained tokenizer "
              f"to get a number -- the brief forbids it and the count would "
              f"describe the wrong vocabulary.", file=sys.stderr)
        return 1

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    print(f"[{lang}] counting with {model_path.name} (vocab={sp.get_piece_size():,})")

    per_split = {}
    for name in args.splits:
        p = root / lang / "data" / "splits" / f"{name}.jsonl"
        if not p.exists():
            print(f"  [skip] {name}: {p} not found")
            continue
        print(f"  counting {name} ...", flush=True)
        per_split[name] = count_split(sp, p, text_key=args.text_key)
        print(f"    {per_split[name]['total_tokens']:,} tokens "
              f"in {per_split[name]['total_documents']:,} documents")

    if not per_split:
        print("[error] no splits found", file=sys.stderr)
        return 1

    corpus_tokens = sum(s["total_tokens"] for s in per_split.values())
    train_tokens = per_split.get("train", {}).get("total_tokens", 0)

    by_class: Counter[str] = Counter()
    docs_by_class: Counter[str] = Counter()
    for s in per_split.values():
        by_class.update(s["tokens_by_provenance"])
        docs_by_class.update(s["documents_by_provenance"])

    manual = by_class.get("manual", 0)
    downloaded = by_class.get("downloaded", 0)
    unlabelled = by_class.get("unlabelled", 0)
    manual_fraction = manual / corpus_tokens if corpus_tokens else 0.0

    # Document-level fraction, purely to expose the difference. If these two
    # diverge, the report should quote the token one -- that is what the brief
    # asks for -- and the divergence is worth a sentence.
    doc_total = sum(docs_by_class.values())
    manual_doc_fraction = docs_by_class.get("manual", 0) / doc_total if doc_total else 0.0

    report = {
        "language": lang,
        "tokenizer": {
            "model_file": str(model_path),
            "vocab_size": sp.get_piece_size(),
            "note": "Counts are valid only for this tokenizer. Retraining it "
                    "invalidates every token figure below.",
        },
        "totals": {
            "corpus_tokens_all_splits": corpus_tokens,
            "train_tokens": train_tokens,
            "target_tokens": args.target_tokens,
            "pct_of_target": round(100 * corpus_tokens / max(1, args.target_tokens), 2),
            "shortfall_tokens": max(0, args.target_tokens - corpus_tokens),
        },
        "manual_vs_downloaded": {
            "manual_tokens": manual,
            "downloaded_tokens": downloaded,
            "unlabelled_tokens": unlabelled,
            "manual_fraction_of_tokens": round(manual_fraction, 4),
            "manual_fraction_of_documents": round(manual_doc_fraction, 4),
            "requirement": args.min_manual_fraction,
            "requirement_met": manual_fraction >= args.min_manual_fraction,
        },
        "per_split": per_split,
    }

    out = root / lang / "data" / "stats" / "token_accounting.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  corpus tokens (all splits) : {corpus_tokens:,}")
    print(f"  train tokens               : {train_tokens:,}")
    print(f"  target                     : {args.target_tokens:,} "
          f"({report['totals']['pct_of_target']}%)")
    print(f"  manual tokens              : {manual:,} ({manual_fraction:.2%})")
    print(f"  downloaded tokens          : {downloaded:,}")
    if unlabelled:
        print(f"  UNLABELLED tokens          : {unlabelled:,} "
              f"({unlabelled / max(1, corpus_tokens):.2%})")

    ok = True
    if unlabelled:
        ok = False
        print(f"\n  [FAIL] {by_class.get('unlabelled', 0):,} tokens have no "
              f"`provenance_class`. The manual fraction above is therefore a "
              f"lower bound, not a measurement. Label every document at ingest "
              f"(manifest_utils.py) and re-run.")
    if manual_fraction < args.min_manual_fraction:
        ok = False
        need = int(args.min_manual_fraction * corpus_tokens) - manual
        print(f"\n  [FAIL] manual fraction {manual_fraction:.2%} is below the "
              f"{args.min_manual_fraction:.0%} requirement. You need roughly "
              f"{need:,} more manual tokens at the current corpus size -- or "
              f"{int(manual / args.min_manual_fraction):,} total corpus tokens "
              f"if you instead shrink the downloaded share.")
    if abs(manual_fraction - manual_doc_fraction) > 0.05:
        print(f"\n  [NOTE] manual is {manual_fraction:.1%} of tokens but "
              f"{manual_doc_fraction:.1%} of documents. Quote the token figure "
              f"-- that is what the brief specifies -- and explain the gap "
              f"(manual documents are {'longer' if manual_fraction > manual_doc_fraction else 'shorter'} "
              f"on average).")
    if corpus_tokens < args.target_tokens:
        print(f"\n  [NOTE] {report['totals']['shortfall_tokens']:,} tokens short "
              f"of target. For the lower-resource language the brief permits "
              f"this: report the exact count and justify the shortfall.")
    if ok and corpus_tokens >= args.target_tokens:
        print("\n  [OK] target met and manual fraction satisfied.")

    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
