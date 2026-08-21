"""
yi_quality_score.py
===================
LLM-based corpus quality scoring using Yi (01.AI), with a distillation step so
it finishes on a student compute budget.

READ THE THROUGHPUT MATH BEFORE YOU RUN THIS
--------------------------------------------
Yi-6B on a free Colab T4, 4-bit, ~600-token documents, generating one digit:
roughly **2-5 documents/second**. Against a 500M-token corpus (~800k documents
at 600 tokens each) that is **44 to 111 hours**. It does not finish. Not
"slowly" -- it does not finish before any deadline you have.

So this script has two stages and you need both:

  STAGE A  --score      Yi scores a SAMPLE (default 30k documents).
                        ~2-4 hours on a T4. This is the expensive part.

  STAGE B  --distill    Train a cheap character n-gram classifier on Yi's
                        labels, then score EVERY document with it.
                        ~10 minutes on CPU for the whole corpus.

This is the FineWeb-Edu recipe: use the expensive model once to define what
"good" means, then distil it into something you can afford to run everywhere.
Stage B is not a shortcut around Stage A -- it is what makes Stage A usable.

Running Stage A alone gives you quality labels for 4% of your corpus, which is
enough for a report table and not enough to filter with.

WHAT IT SCORES
--------------
A 0-5 rubric adapted from the FineWeb-Edu educational-quality prompt, rewritten
for a monolingual Devanagari pretraining corpus. The dimensions that matter here
are different from English web filtering: OCR damage and boilerplate survival
are the dominant failure modes in a corpus that is >=20% manually collected from
books and scraped pages.

  5  clean, coherent, well-formed prose; publishable as-is
  4  good prose, minor noise
  3  usable but noticeably noisy (some boilerplate, light OCR damage)
  2  substantially damaged: heavy OCR errors, fragments, list spam
  1  barely usable: mostly navigation, tables, repeated strings
  0  not usable prose, or not the target language at all

A CONSTRAINT YOU SHOULD CHECK BEFORE USING THIS
-----------------------------------------------
The project brief's hard-constraints box says "No pretrained models; no
pretrained tokenizers; no HuggingFace Transformer models." In context that
governs the model you BUILD -- ??2.1 is explicit that it is about implementing
the architecture yourself. Using an LLM as a *data filter* is standard practice
(FineWeb-Edu, DCLM) and is a different activity from using one as your model.

But the wording does not carve that out explicitly, and I am not your grader.
Before you spend a night of Colab on this:

  - ask on the course HackMD (the brief directs doubts there), and
  - either way, state plainly in your report that Yi was used ONLY to score
    training data, that no pretrained weights or tokenizer enter Model H or
    Model L, and that the distilled classifier is trained from scratch.

`--no-llm` runs the same pipeline with a non-LLM scorer (n-gram perplexity
against your trusted manual subset + Gopher-style heuristics) if you want zero
exposure to that question. It is weaker, and it is safe.

USAGE
-----
  # Stage A: Yi scores a sample (GPU, hours)
  python yi_quality_score.py --lang hindi --repo-root . --score --sample 30000

  # Stage B: distil to a fast classifier and score the whole corpus (CPU, minutes)
  python yi_quality_score.py --lang hindi --repo-root . --distill

  # Apply a threshold to produce the filtered corpus
  python yi_quality_score.py --lang hindi --repo-root . --apply --min-score 2.5

  # No-LLM variant, if you'd rather not touch the constraint question
  python yi_quality_score.py --lang hindi --repo-root . --score --no-llm --sample 30000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Verified model IDs on the HuggingFace Hub (01.AI).
#   01-ai/Yi-6B-Chat-4bits   pre-quantised, no bitsandbytes build needed -- best
#                            first choice on a free Colab T4
#   01-ai/Yi-1.5-6B-Chat     newer 1.5 series, quantise at load time
#   01-ai/Yi-6B-Chat         fp16, needs ~13GB -- will not fit alongside a batch on a T4
DEFAULT_MODEL = "01-ai/Yi-6B-Chat-4bits"

RUBRIC = """You are grading text for inclusion in a language-model training corpus.

Rate the passage from 0 to 5:
5 = clean, coherent, well-formed prose; publishable as-is
4 = good prose with minor noise
3 = usable but noticeably noisy (some boilerplate or light OCR damage)
2 = substantially damaged: heavy OCR errors, sentence fragments, list spam
1 = barely usable: mostly navigation text, tables, or repeated strings
0 = not usable prose, or not written in {language}

Judge only the text quality, never the opinions expressed.
Reply with a single digit and nothing else."""

_DIGIT = re.compile(r"[0-5]")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def reservoir(path: Path, n: int, *, text_key="text", seed=20260820):
    rng = random.Random(seed)
    res, seen = [], 0
    for doc in iter_jsonl(path):
        if not (doc.get(text_key) or "").strip():
            continue
        seen += 1
        if len(res) < n:
            res.append(doc)
        else:
            j = rng.randrange(seen)
            if j < n:
                res[j] = doc
    return res, seen


def load_done(path: Path) -> set:
    """Resume support: which doc_ids already have a score."""
    if not path.exists():
        return set()
    return {d.get("doc_id") for d in iter_jsonl(path) if d.get("doc_id") is not None}


# ---------------------------------------------------------------------------
# Scorer A: Yi
# ---------------------------------------------------------------------------

class YiScorer:
    """Wraps a Yi chat model as a 0-5 quality rater."""

    def __init__(self, model_id: str = DEFAULT_MODEL, *, language: str = "Hindi",
                 max_chars: int = 2400, batch_size: int = 8, log=print):
        self.model_id = model_id
        self.language = language
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.log = log
        self._model = None
        self._tok = None

    def load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                f"transformers/torch required for --score: {e}\n"
                f"  pip install 'transformers>=4.40' accelerate torch") from e

        self.log(f"  loading {self.model_id} ...")
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"          # required for batched generation
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self._model.eval()
        self.log(f"  loaded on {next(self._model.parameters()).device}")

    def _prompt(self, text: str) -> str:
        snippet = text.strip()[: self.max_chars]
        messages = [
            {"role": "user",
             "content": RUBRIC.format(language=self.language)
                        + f"\n\nPassage:\n{snippet}\n\nRating:"},
        ]
        return self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)

    def score_batch(self, texts: list[str]) -> list[float | None]:
        import torch
        self.load()
        prompts = [self._prompt(t) for t in texts]
        enc = self._tok(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1536).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=4, do_sample=False,
                pad_token_id=self._tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = self._tok.batch_decode(gen, skip_special_tokens=True)
        scores = []
        for d in decoded:
            m = _DIGIT.search(d)
            scores.append(float(m.group()) if m else None)
        return scores


# ---------------------------------------------------------------------------
# Scorer B: no-LLM fallback
# ---------------------------------------------------------------------------

class HeuristicScorer:
    """
    Non-LLM 0-5 scorer: Gopher-style structural signals plus a character 5-gram
    log-probability against a reference profile built from your own trusted
    (manual) text. No pretrained weights involved anywhere.
    """

    def __init__(self, reference_texts: list[str] | None = None, log=print):
        self.log = log
        self.ref: Counter[str] = Counter()
        self.ref_total = 0
        if reference_texts:
            for t in reference_texts:
                flat = re.sub(r"\s+", " ", t)
                for i in range(len(flat) - 4):
                    self.ref[flat[i:i + 5]] += 1
            self.ref_total = sum(self.ref.values())
            log(f"  reference profile: {len(self.ref):,} 5-grams "
                f"from {len(reference_texts):,} trusted documents")

    def _ngram_lp(self, text: str) -> float:
        if not self.ref_total:
            return 0.0
        flat = re.sub(r"\s+", " ", text)[:4000]
        if len(flat) < 10:
            return -20.0
        floor = math.log(0.5 / self.ref_total)
        tot = n = 0.0
        for i in range(len(flat) - 4):
            c = self.ref.get(flat[i:i + 5], 0)
            tot += math.log(c / self.ref_total) if c else floor
            n += 1
        return tot / max(1.0, n)

    def score_batch(self, texts: list[str]) -> list[float | None]:
        out = []
        for text in texts:
            words = text.split()
            if len(words) < 20:
                out.append(0.0)
                continue
            lines = [l for l in text.split("\n") if l.strip()]
            score = 5.0
            uniq = len(set(words)) / len(words)
            if uniq < 0.30:
                score -= 2.0
            elif uniq < 0.45:
                score -= 1.0
            if lines:
                dup = 1 - len(set(lines)) / len(lines)
                score -= 2.0 * min(1.0, dup / 0.3)
            mean_wl = sum(len(w) for w in words) / len(words)
            if not (2.0 <= mean_wl <= 12.0):
                score -= 1.5
            sym = sum(1 for w in words if any(c in "#???$%^&*_=+~<>|\\/@" for c in w))
            score -= 2.0 * min(1.0, (sym / len(words)) / 0.10)
            if self.ref_total:
                lp = self._ngram_lp(text)
                # Calibrated loosely: -8 or better looks like the trusted set,
                # -13 or worse looks like a different distribution entirely.
                score -= 2.0 * min(1.0, max(0.0, (-lp - 8.0) / 5.0))
            out.append(round(max(0.0, min(5.0, score)), 2))
        return out


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

def stage_score(args, root: Path) -> int:
    lang = args.lang
    split = root / lang / "data" / "splits" / f"{args.split}.jsonl"
    if not split.exists():
        print(f"[error] {split} not found", file=sys.stderr)
        return 1

    out_path = root / lang / "data" / "quality" / "llm_scores.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    if done:
        print(f"  resuming: {len(done):,} documents already scored")

    print(f"[{lang}] sampling up to {args.sample:,} documents from {args.split} ...")
    docs, total_seen = reservoir(split, args.sample, text_key=args.text_key,
                                 seed=args.seed)
    docs = [d for d in docs if d.get("doc_id") not in done]
    print(f"  corpus has {total_seen:,} documents; {len(docs):,} left to score")
    if not docs:
        print("  nothing to do")
        return 0

    if args.no_llm:
        ref = [d.get(args.text_key, "") for d in iter_jsonl(split)
               if d.get("provenance_class") == "manual"][:4000]
        scorer = HeuristicScorer(ref)
        backend = "heuristic-ngram"
    else:
        scorer = YiScorer(args.model, language=("Hindi" if lang == "hindi" else "Nepali"),
                          batch_size=args.batch_size)
        backend = args.model

    # Throughput reality check before committing hours to it.
    est_dps = 40.0 if args.no_llm else 3.0
    eta_h = len(docs) / est_dps / 3600
    print(f"  backend: {backend}")
    print(f"  rough ETA: {eta_h:.1f} h at ~{est_dps:.0f} docs/s")
    if eta_h > args.max_hours:
        print(f"\n  [STOP] estimated {eta_h:.1f} h exceeds --max-hours {args.max_hours}.\n"
              f"         Lower --sample (30000 is usually enough to train the\n"
              f"         distilled classifier) or raise --max-hours deliberately.")
        return 2

    t0 = time.time()
    n_written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(docs), args.batch_size):
            chunk = docs[i:i + args.batch_size]
            texts = [d.get(args.text_key, "") for d in chunk]
            try:
                scores = scorer.score_batch(texts)
            except Exception as e:
                print(f"  [batch failed at {i}] {type(e).__name__}: {e}")
                continue
            for d, s in zip(chunk, scores):
                if s is None:
                    continue
                f.write(json.dumps({
                    "doc_id": d.get("doc_id"),
                    "score": s,
                    "provenance_class": d.get("provenance_class"),
                    "source": d.get("source"),
                    "backend": backend,
                }, ensure_ascii=False) + "\n")
                n_written += 1
            if i and (i // args.batch_size) % 20 == 0:
                el = time.time() - t0
                rate = (i + len(chunk)) / max(1e-9, el)
                rem = (len(docs) - i - len(chunk)) / max(1e-9, rate) / 60
                print(f"  {i + len(chunk):,}/{len(docs):,}  "
                      f"{rate:.1f} docs/s  ~{rem:.0f} min left", flush=True)
                f.flush()

    print(f"\n  wrote {n_written:,} scores -> {out_path}")
    _summarise(out_path)
    return 0


def _summarise(path: Path):
    scores, by_class = [], {}
    for d in iter_jsonl(path):
        s = d.get("score")
        if s is None:
            continue
        scores.append(s)
        by_class.setdefault(d.get("provenance_class") or "unlabelled", []).append(s)
    if not scores:
        return
    hist = Counter(int(s) for s in scores)
    print(f"\n  score distribution over {len(scores):,} documents:")
    for b in range(6):
        n = hist.get(b, 0)
        bar = "#" * int(40 * n / max(1, len(scores)))
        print(f"    {b}: {n:>7,}  {bar}")
    print(f"    mean {sum(scores) / len(scores):.2f}")
    if len(by_class) > 1:
        print("\n  mean score by provenance:")
        for k, v in sorted(by_class.items()):
            print(f"    {k:<12} {sum(v) / len(v):.2f}  (n={len(v):,})")
        print("\n  If manual scores materially below downloaded, your OCR/scrape "
              "cleaning needs work before you filter -- do not simply threshold "
              "away the 20% the brief requires you to collect.")


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------

def stage_distill(args, root: Path) -> int:
    """Train a cheap classifier on the LLM labels, then score the full corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    import joblib

    lang = args.lang
    scores_path = root / lang / "data" / "quality" / "llm_scores.jsonl"
    if not scores_path.exists():
        print(f"[error] no labels at {scores_path}. Run --score first.", file=sys.stderr)
        return 1

    labels = {d["doc_id"]: d["score"] for d in iter_jsonl(scores_path)
              if d.get("doc_id") is not None and d.get("score") is not None}
    print(f"[{lang}] {len(labels):,} labelled documents")
    if len(labels) < 2000:
        print("  [WARN] under 2000 labels. The distilled scorer will be noisy; "
              "3000+ is a reasonable floor.")

    split = root / lang / "data" / "splits" / f"{args.split}.jsonl"
    X, y = [], []
    for d in iter_jsonl(split):
        did = d.get("doc_id")
        if did in labels:
            X.append((d.get(args.text_key) or "")[:4000])
            y.append(labels[did])
    print(f"  matched {len(X):,} documents to text")
    if len(X) < 200:
        print("[error] too few matched documents -- check doc_id consistency",
              file=sys.stderr)
        return 1

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  max_features=120_000, min_df=2, sublinear_tf=True)),
        ("reg", Ridge(alpha=1.0)),
    ])
    print("  fitting distilled scorer ...")
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    corr = float(__import__("numpy").corrcoef(pred, yte)[0, 1])
    print(f"  held-out MAE {mae:.3f} on a 0-5 scale, Pearson r {corr:.3f}")
    if mae > 1.0:
        print("  [WARN] MAE above 1.0 means the distilled scorer disagrees with Yi "
              "by more than a whole rubric band. Report this rather than filtering "
              "on it, or collect more labels.")

    model_dir = root / lang / "data" / "quality"
    joblib.dump({"pipeline": pipe, "mae": mae, "pearson_r": corr,
                 "n_labels": len(X)}, model_dir / "quality_scorer.joblib", compress=3)

    out = model_dir / "corpus_scores.jsonl"
    print(f"  scoring the full corpus -> {out}")
    n = 0
    buf_ids, buf_txt = [], []
    with open(out, "w", encoding="utf-8") as f:
        def flush():
            nonlocal n
            if not buf_txt:
                return
            for did, s in zip(buf_ids, pipe.predict(buf_txt)):
                f.write(json.dumps({"doc_id": did,
                                    "quality_score": round(float(s), 3)}) + "\n")
                n += 1
            buf_ids.clear()
            buf_txt.clear()

        for d in iter_jsonl(split):
            buf_ids.append(d.get("doc_id"))
            buf_txt.append((d.get(args.text_key) or "")[:4000])
            if len(buf_txt) >= 2048:
                flush()
        flush()
    print(f"  scored {n:,} documents")
    print(f"  wrote {model_dir / 'quality_scorer.joblib'}\n        {out}")
    return 0


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------

def stage_apply(args, root: Path) -> int:
    lang = args.lang
    scores_path = root / lang / "data" / "quality" / "corpus_scores.jsonl"
    if not scores_path.exists():
        print(f"[error] {scores_path} not found. Run --distill first.", file=sys.stderr)
        return 1
    scores = {d["doc_id"]: d["quality_score"] for d in iter_jsonl(scores_path)}

    split = root / lang / "data" / "splits" / f"{args.split}.jsonl"
    out = root / lang / "data" / "splits" / f"{args.split}.filtered.jsonl"
    kept = dropped = 0
    kept_by_class: Counter[str] = Counter()
    drop_by_class: Counter[str] = Counter()

    with open(out, "w", encoding="utf-8") as f:
        for d in iter_jsonl(split):
            s = scores.get(d.get("doc_id"))
            cls = d.get("provenance_class") or "unlabelled"
            if s is not None and s < args.min_score:
                dropped += 1
                drop_by_class[cls] += 1
                continue
            d["quality_score"] = s
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
            kept_by_class[cls] += 1

    tot = kept + dropped
    print(f"[{lang}] threshold {args.min_score}: kept {kept:,} / {tot:,} "
          f"({kept / max(1, tot):.1%}), dropped {dropped:,}")
    for cls in sorted(set(kept_by_class) | set(drop_by_class)):
        k, dr = kept_by_class[cls], drop_by_class[cls]
        print(f"    {cls:<12} kept {k:>7,}  dropped {dr:>7,}  ({k / max(1, k + dr):.1%} retained)")
    print(f"\n  wrote {out}")
    print("  Re-run count_corpus_tokens.py afterwards: filtering changes both your "
          "token total and your manual fraction, and the >=20% requirement is "
          "measured on the FINAL corpus.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Yi-based corpus quality scoring, with distillation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--seed", type=int, default=20260820)

    ap.add_argument("--score", action="store_true", help="Stage A: LLM scores a sample")
    ap.add_argument("--distill", action="store_true", help="Stage B: train + score all")
    ap.add_argument("--apply", action="store_true", help="Stage C: write filtered split")

    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=6.0,
                    help="refuse to start if the ETA exceeds this")
    ap.add_argument("--no-llm", action="store_true",
                    help="use the heuristic scorer instead of Yi")
    ap.add_argument("--min-score", type=float, default=2.5)
    args = ap.parse_args()

    if not (args.score or args.distill or args.apply):
        ap.error("pick a stage: --score, --distill or --apply")

    root = Path(args.repo_root).resolve()
    rc = 0
    if args.score:
        rc = stage_score(args, root) or rc
    if args.distill:
        rc = stage_distill(args, root) or rc
    if args.apply:
        rc = stage_apply(args, root) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
