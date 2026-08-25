"""
dataset_stats.py — the per-language dataset statistics report (deliverable 3)
=============================================================================
`corpus_stats.json` records what the BUILD did: how many documents went in, how
many survived each filter, what the budget trimmed. `token_accounting.json`
records what the TOKENIZER measured. Neither describes the dataset itself --
how long its documents are, what it is made of, how lexically varied it is.
That is what a dataset statistics report is, and this produces it.

    python -m pipeline.stats.dataset_stats --lang hindi --repo-root .

Writes `<lang>/data/stats/dataset_statistics.json` and prints a summary.

WHAT IT MEASURES AND WHY EACH ONE IS THERE
-------------------------------------------
* **Document length distribution** -- mean, median and percentiles in
  characters, words and (if a tokenizer exists) tokens. A mean alone hides the
  shape: a corpus of 400-character news snippets and one of 40,000-character
  book chapters can share a mean and behave completely differently in a
  fixed-context language model. The percentiles are what tell you how many
  documents will be truncated at a 512- or 1024-token context.

* **Composition by source and by provenance class** -- documents, characters
  and share. "20% manual" is a single number; this is the breakdown behind it,
  and it exposes whether one source dominates.

* **Top domains for manually collected text** -- from the `url` field. A manual
  corpus scraped from 45 hosts where two hosts contribute 80% of the text is
  not really a 45-host corpus, and only this table shows that.

* **Type-token ratio and hapax rate** -- lexical variety, on a fixed random
  sample of documents (TTR is not comparable across different sample sizes, so
  the sample size is fixed and recorded rather than "all documents"). Hapax
  rate -- the share of word types seen exactly once -- is the part that
  matters for a tokenizer: it is the vocabulary tail that byte fallback ends
  up handling.

* **Script composition** -- mean Devanagari share, plus how many documents fall
  below the filter threshold. Should be ~0 after cleaning; if it is not, the
  language filter is not doing its job.

* **Sentence statistics** -- counts and mean length, splitting on danda (।) as
  well as ASCII terminators, because Devanagari prose does not end sentences
  with a full stop.

SAMPLING, STATED HONESTLY
-------------------------
Everything except TTR/hapax is computed over EVERY document in the splits.
TTR and hapax are computed on `--ttr-sample` documents drawn by a seeded hash
of the document id, which is deterministic and reproducible. The sample size
appears in the output next to the numbers it produced, because a type-token
ratio without its sample size is not a comparable quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

SPLITS = ("train", "val", "test")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SENT_SPLIT = re.compile(r"[।॥.!?]+")
WORD_SPLIT = re.compile(r"\s+")


def percentile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolated percentile. Values must already be sorted."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def describe(vals: list[int]) -> dict:
    if not vals:
        return {}
    vals = sorted(vals)
    n = len(vals)
    total = sum(vals)
    mean = total / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "count": n,
        "total": total,
        "mean": round(mean, 1),
        "std": round(math.sqrt(var), 1),
        "min": vals[0],
        "p1": round(percentile(vals, 0.01), 1),
        "p5": round(percentile(vals, 0.05), 1),
        "p25": round(percentile(vals, 0.25), 1),
        "median": round(percentile(vals, 0.50), 1),
        "p75": round(percentile(vals, 0.75), 1),
        "p95": round(percentile(vals, 0.95), 1),
        "p99": round(percentile(vals, 0.99), 1),
        "max": vals[-1],
    }


def in_sample(doc_id: str, rate: float, seed: int) -> bool:
    """Deterministic sampling. A hash threshold rather than random.random()
    means the same documents are sampled on every run and on every machine,
    so two runs of this report are comparable."""
    if rate >= 1.0:
        return True
    h = hashlib.blake2b(f"{seed}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64 < rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--ttr-sample", type=int, default=50_000,
                    help="documents sampled for type-token ratio and hapax rate "
                         "(default 50,000). TTR is not comparable across sample "
                         "sizes, so this is fixed and reported alongside.")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--no-tokens", action="store_true",
                    help="skip per-document token lengths even if a tokenizer "
                         "exists (roughly halves the runtime)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang
    splits_dir = root / lang / "data" / "splits"
    if not (splits_dir / "train.jsonl").exists():
        print(f"[error] {splits_dir / 'train.jsonl'} not found. Run build first.",
              file=sys.stderr)
        return 1

    # Token lengths need the shipped tokenizer. Without it every other statistic
    # is still valid -- characters and words do not depend on a tokenizer -- so
    # this degrades rather than fails.
    sp = None
    model = root / lang / "tokenizer" / "vocab" / f"{lang}_tokenizer.model"
    if not args.no_tokens and model.exists():
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(str(model))
            print(f"[{lang}] token lengths from {model.name} "
                  f"(vocab {sp.get_piece_size():,})")
        except Exception as e:
            print(f"  [warn] could not load tokenizer: {type(e).__name__}; "
                  f"token statistics will be omitted")
    elif not model.exists():
        print(f"  [note] no tokenizer at {model}; reporting characters and "
              f"words only")

    # ---- accumulators ------------------------------------------------------
    per_split: dict[str, dict] = {}
    all_chars: list[int] = []
    all_words: list[int] = []
    all_tokens: list[int] = []
    by_source: dict[str, dict] = defaultdict(
        lambda: {"documents": 0, "characters": 0, "words": 0,
                 "provenance_class": None})
    by_class: dict[str, dict] = defaultdict(
        lambda: {"documents": 0, "characters": 0, "words": 0})
    by_method: Counter = Counter()
    domains: Counter = Counter()
    domain_chars: Counter = Counter()
    deva_ratios: list[float] = []
    low_deva = 0
    sentences_total = 0
    sample_types: Counter = Counter()
    sampled_docs = 0
    sampled_words = 0

    # Estimate the sampling rate from the train split's line count so the TTR
    # sample lands near --ttr-sample regardless of corpus size.
    n_docs_est = sum(1 for _ in open(splits_dir / "train.jsonl", "rb"))
    rate = min(1.0, args.ttr_sample / max(1, n_docs_est))

    for split in SPLITS:
        p = splits_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        s_chars: list[int] = []
        s_words: list[int] = []
        s_tokens: list[int] = []
        s_class: Counter = Counter()
        print(f"  scanning {split} ...", flush=True)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get(args.text_key) or ""
                if not text:
                    continue
                nchars = len(text)
                words = [w for w in WORD_SPLIT.split(text) if w]
                nwords = len(words)
                s_chars.append(nchars)
                s_words.append(nwords)
                all_chars.append(nchars)
                all_words.append(nwords)

                if sp is not None:
                    n = len(sp.encode(text, out_type=int))
                    s_tokens.append(n)
                    all_tokens.append(n)

                cls = rec.get("provenance_class") or "unlabelled"
                src = rec.get("source") or "unknown"
                s_class[cls] += 1
                by_class[cls]["documents"] += 1
                by_class[cls]["characters"] += nchars
                by_class[cls]["words"] += nwords
                b = by_source[src]
                b["documents"] += 1
                b["characters"] += nchars
                b["words"] += nwords
                b["provenance_class"] = cls
                by_method[rec.get("collection_method") or "unknown"] += 1

                if cls == "manual" and rec.get("url"):
                    host = urlsplit(rec["url"]).netloc.lower()
                    if host:
                        domains[host] += 1
                        domain_chars[host] += nchars

                ndeva = len(DEVANAGARI.findall(text))
                ratio = ndeva / nchars if nchars else 0.0
                deva_ratios.append(ratio)
                if ratio < 0.5:
                    low_deva += 1

                sentences_total += sum(
                    1 for s in SENT_SPLIT.split(text) if s.strip())

                doc_id = rec.get("doc_id") or str(nchars)
                if in_sample(doc_id, rate, args.seed):
                    sampled_docs += 1
                    sampled_words += nwords
                    sample_types.update(words)

        per_split[split] = {
            "documents": len(s_chars),
            "characters": sum(s_chars),
            "words": sum(s_words),
            "tokens": sum(s_tokens) if s_tokens else None,
            "documents_by_provenance": dict(s_class),
            "characters_per_document": describe(s_chars),
            "words_per_document": describe(s_words),
            "tokens_per_document": describe(s_tokens) if s_tokens else None,
        }
        print(f"    {len(s_chars):,} docs, {sum(s_chars):,} chars")

    # ---- lexical -----------------------------------------------------------
    types = len(sample_types)
    hapax = sum(1 for v in sample_types.values() if v == 1)
    lexical = {
        "sample_documents": sampled_docs,
        "sample_words": sampled_words,
        "word_types_in_sample": types,
        "type_token_ratio": round(types / sampled_words, 6) if sampled_words else None,
        "hapax_types": hapax,
        "hapax_rate_of_types": round(hapax / types, 4) if types else None,
        "note": ("Type-token ratio falls as sample size grows, so it is only "
                 "comparable against another measurement at the same "
                 "sample_words. Types are whitespace-delimited surface forms, "
                 "not lemmas -- for a morphologically rich language this "
                 "overstates vocabulary relative to a lemmatised count."),
    }

    total_chars = sum(v["characters"] for v in by_class.values())
    total_docs = sum(v["documents"] for v in by_class.values())
    for v in by_class.values():
        v["share_of_characters"] = round(v["characters"] / total_chars, 4) if total_chars else 0
        v["share_of_documents"] = round(v["documents"] / total_docs, 4) if total_docs else 0
    for v in by_source.values():
        v["share_of_characters"] = round(v["characters"] / total_chars, 4) if total_chars else 0

    top_domains = [
        {"host": h, "documents": domains[h], "characters": domain_chars[h],
         "share_of_manual_characters": round(
             domain_chars[h] / max(1, by_class.get("manual", {}).get("characters", 1)), 4)}
        for h, _ in domain_chars.most_common(25)
    ]

    report = {
        "language": lang,
        "totals": {
            "documents": total_docs,
            "characters": total_chars,
            "words": sum(v["words"] for v in by_class.values()),
            "tokens": sum(all_tokens) if all_tokens else None,
            "sentences": sentences_total,
            "tokenizer": (str(model.name) if sp is not None else None),
            "tokenizer_vocab_size": (sp.get_piece_size() if sp is not None else None),
        },
        "document_length": {
            "characters": describe(all_chars),
            "words": describe(all_words),
            "tokens": describe(all_tokens) if all_tokens else None,
        },
        "composition_by_provenance": {k: dict(v) for k, v in by_class.items()},
        "composition_by_source": dict(sorted(
            ((k, dict(v)) for k, v in by_source.items()),
            key=lambda kv: -kv[1]["characters"])),
        "documents_by_collection_method": dict(by_method.most_common()),
        "top_manual_domains": top_domains,
        "manual_domain_count": len(domains),
        "script": {
            "mean_devanagari_ratio": round(sum(deva_ratios) / len(deva_ratios), 4)
            if deva_ratios else None,
            "median_devanagari_ratio": round(percentile(
                sorted(int(r * 10000) for r in deva_ratios), 0.5) / 10000, 4)
            if deva_ratios else None,
            "documents_below_50pct_devanagari": low_deva,
            "note": "Should be near zero after the language filter. A non-trivial "
                    "count means filtering is leaking.",
        },
        "sentences": {
            "total": sentences_total,
            "mean_per_document": round(sentences_total / total_docs, 2) if total_docs else None,
            "mean_words_per_sentence": round(
                sum(v["words"] for v in by_class.values()) / sentences_total, 2)
            if sentences_total else None,
            "note": "Split on danda (U+0964), double danda, and ASCII . ! ? — "
                    "Devanagari prose does not terminate sentences with a full stop.",
        },
        "lexical": lexical,
        "per_split": per_split,
    }

    out = root / lang / "data" / "stats" / "dataset_statistics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    dl = report["document_length"]
    print(f"\n[{lang}] dataset statistics")
    print(f"  documents        {total_docs:,}")
    print(f"  characters       {total_chars:,}")
    print(f"  words            {report['totals']['words']:,}")
    if report["totals"]["tokens"]:
        print(f"  tokens           {report['totals']['tokens']:,}")
    print(f"  sentences        {sentences_total:,}")
    print(f"\n  chars/doc        median {dl['characters']['median']:,.0f}  "
          f"mean {dl['characters']['mean']:,.1f}  "
          f"p95 {dl['characters']['p95']:,.0f}  max {dl['characters']['max']:,}")
    print(f"  words/doc        median {dl['words']['median']:,.0f}  "
          f"mean {dl['words']['mean']:,.1f}")
    if dl["tokens"]:
        print(f"  tokens/doc       median {dl['tokens']['median']:,.0f}  "
              f"mean {dl['tokens']['mean']:,.1f}  "
              f"p95 {dl['tokens']['p95']:,.0f}")
    print(f"\n  sources          {len(by_source)}")
    for k, v in list(report["composition_by_source"].items())[:8]:
        print(f"    {k:<24} {v['documents']:>10,} docs  "
              f"{v['share_of_characters']:>7.2%} of chars")
    print(f"\n  manual domains   {len(domains):,}")
    for d in top_domains[:5]:
        print(f"    {d['host']:<32} {d['share_of_manual_characters']:>7.2%} "
              f"of manual chars")
    print(f"\n  TTR              {lexical['type_token_ratio']} "
          f"over {sampled_words:,} words in {sampled_docs:,} sampled docs")
    print(f"  word types       {types:,}  ({lexical['hapax_rate_of_types']:.1%} hapax)")
    print(f"  Devanagari       mean {report['script']['mean_devanagari_ratio']:.3f}, "
          f"{low_deva:,} docs below 50%")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
