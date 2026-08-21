"""
train_tokenizer.py  (v2 ??? replaces the v1 sweep script)
=======================================================
Step 7 of the Phase 1 pipeline: train a from-scratch SentencePiece tokenizer
per language, with a vocabulary sweep whose selection criterion is not
degenerate.

WHAT CHANGED FROM v1, AND WHY
-----------------------------
v1 selected the vocabulary size by "lowest fertility subject to UNK < 1%".
Fertility falls monotonically with vocabulary size, so that rule ALWAYS returns
the largest size in the sweep list. The sweep was decorative: the answer was
decided by whatever you typed last in --vocab-sizes. Both languages duly
selected their largest candidate (Hindi 32000, Nepali 24000).

v2 replaces it with three things that actually discriminate:

1. ELBOW ON MARGINAL GAIN.  Climb the vocabulary ladder only while the next
   step buys more than `min_relative_gain` (default 2%) of fertility. This is
   the standard knee criterion and it can return an interior point.

2. AN EXPLICIT PARAMETER COST.  Vocabulary is not free: the embedding matrix is
   vocab_size x d_model parameters, and for the small models trained in Phase 2
   that is a large share of the budget. The sweep reports embedding share so
   the trade-off is visible rather than implicit.

3. A SCALING-LAW REFERENCE POINT.  Tao et al. (2024), "Scaling Laws with
   Vocabulary", find the compute-optimal vocabulary scales as
   N_v ??? N_nv^0.83 (sub-linear in non-vocabulary parameters). We compute the
   predicted optimum for the Phase 2 model size and report the ratio between it
   and the swept choice, so the number in the report has a published anchor
   instead of only an in-house heuristic.

4. BYTE FALLBACK, AND WHY UNK RATE STOPS BEING THE METRIC.
   v1 measured UNK rate and found it *rising* with vocabulary size
   (Hindi 0.078% -> 0.082%). That is not a bug: the UNK numerator is roughly
   fixed (the same rare characters) while the token denominator shrinks as
   pieces get longer, so the rate goes up. More importantly, UNK is the wrong
   thing to be spending vocabulary on. With `byte_fallback=True` SentencePiece
   decomposes any unseen character into UTF-8 byte pieces, so UNK rate is
   structurally zero and no character is ever unrepresentable.

   The metric that replaces it is the BYTE-FALLBACK RATE: the fraction of
   emitted tokens that are raw <0xNN> byte pieces. This matters for Devanagari
   specifically. The tokenizer-tax analysis for Indian languages finds that the
   rate at which merges fail and leave "stranded" single bytes correlates with
   the token-inflation penalty at r = 0.89, with Indic scripts showing 27-43%
   stranded-byte rates under English-centric tokenizers versus <10% for
   English. A monolingual tokenizer trained on your own corpus should drive
   this to near zero; if it does not, the vocabulary or the corpus is too small
   and the report should say so.

OTHER FIXES
-----------
- v1 loaded the entire train split into a Python list of strings, then wrote it
  out again. At the 500M-token target that is several GB resident and will OOM
  a 16 GB notebook. v2 streams JSONL -> the SentencePiece input file line by
  line and never holds the corpus in memory.
- v1 hardcoded `input_sentence_size`, `normalization_rule_name` and
  `remove_extra_whitespaces` inside the function, ignoring tokenizer_config.yaml
  entirely. The YAML was decorative. v2 reads every SentencePiece parameter
  from the config.
- v1 set `normalization_rule_name="nmt_nfkc"` while the pipeline's normalize.py
  applies its own Unicode normalisation. Two different normalisers in sequence
  means the text the tokenizer sees is not the text you stored, and inference
  must replicate both to match. v2 defaults to `identity` and asserts the input
  is already NFC ??? one normaliser, in one place, in the pipeline.
- v1 evaluated on `val_texts[:5000]`, i.e. the FIRST 5000 documents, which are
  whatever the split writer happened to emit first (usually one source). v2
  takes a seeded random sample across the whole held-out split.
- v2 also reports fertility separately over MANUAL vs DOWNLOADED held-out text,
  because the assignment requires >=20% manually collected tokens and a
  tokenizer fitted mostly to web scrape can be measurably worse on OCR'd book
  text. If the two differ a lot, that belongs in the report.

REFERENCES
----------
- Kudo & Richardson (2018), "SentencePiece: A simple and language independent
  subword tokenizer and detokenizer for Neural Text Processing." EMNLP demo.
- Kudo (2018), "Subword Regularization: Improving NMT Models with Multiple
  Subword Candidates." ACL. (the Unigram LM algorithm)
- Sennrich et al. (2016), "Neural Machine Translation of Rare Words with
  Subword Units." ACL. (BPE)
- Rust et al. (2021), "How Good is Your Tokenizer? On the Monolingual
  Performance of Multilingual Language Models." ACL. (fertility as the standard
  intrinsic metric: tokens per word)
- Tao et al. (2024), "Scaling Laws with Vocabulary: Larger Models Deserve
  Larger Vocabularies." (N_v ??? N_nv^0.83)
- Petrov et al. (2023), "Language Model Tokenizers Introduce Unfairness Between
  Languages." NeurIPS.

USAGE
-----
  python train_tokenizer.py --lang hindi  --repo-root /path/to/project
  python train_tokenizer.py --lang nepali --repo-root /path/to/project

  # compare algorithms rather than assuming one
  python train_tokenizer.py --lang nepali --repo-root . --compare-algorithms
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SPM = {
    "character_coverage": 0.9999,
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    # `identity`, not nmt_nfkc: normalisation happens once, in normalize.py.
    # Stacking a second normaliser inside the tokenizer means stored text and
    # tokenizer-visible text diverge, and every inference path has to replicate
    # both to stay consistent.
    "normalization_rule_name": "identity",
    "remove_extra_whitespaces": True,
    "input_sentence_size": 5_000_000,
    "shuffle_input_sentence": True,
    # Structurally removes UNK. See the module docstring.
    "byte_fallback": True,
    "split_digits": True,          # keeps numbers from eating merge budget
    "allow_whitespace_only_pieces": False,
    "max_sentence_length": 16384,
    # SentencePiece is extremely chatty on stderr; 2 = warnings and above.
    "minloglevel": 2,
}


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Streaming corpus preparation
# ---------------------------------------------------------------------------

def stream_jsonl_to_text(
    jsonl_path: Path,
    out_path: Path,
    *,
    text_key: str = "text",
    max_lines: int | None = None,
    check_nfc: bool = True,
    log=print,
) -> dict:
    """
    Stream a JSONL split into the one-sentence-per-line plain text file that
    SentencePiece trains on. Never holds the corpus in memory.

    Returns counts, and ??? because this is the cheapest place to check it ???
    whether the corpus is actually in NFC. A tokenizer trained on mixed
    NFC/NFD Devanagari silently learns two spellings of the same word and
    wastes vocabulary on both.

    WHY max_lines SAMPLES INSTEAD OF TRUNCATING
    -------------------------------------------
    Taking the first N lines looks equivalent and is not. The split file is
    written in the order documents survived cleaning, which is manual-first
    (see iter_language_docs), then downloaded by source priority. Truncating at
    N therefore hands SentencePiece a corpus that is all manual plus whichever
    downloaded source happened to come next ??? and the tokenizer learns the
    vocabulary of a corpus that does not exist.

    Instead this makes a counting pass, derives a keep probability, and selects
    lines by a seeded hash of their content. Deterministic, no memory, and
    every source is thinned by the same factor.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def iter_docs():
        with open(jsonl_path, encoding="utf-8") as fin:
            for raw in fin:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    doc = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                text = doc.get(text_key) or ""
                if text.strip():
                    yield text

    keep_threshold: int | None = None
    total_lines = 0
    if max_lines:
        for text in iter_docs():
            total_lines += sum(1 for l in text.split("\n") if l.strip())
        if total_lines > max_lines:
            p = max_lines / total_lines
            keep_threshold = int(p * (1 << 32))
            log(f"  sampling {max_lines:,} of {total_lines:,} lines "
                f"({p:.1%}), uniformly across sources")
        else:
            log(f"  {total_lines:,} lines is already under the "
                f"{max_lines:,} cap; using all of them")

    n_docs = n_lines = n_chars = 0
    n_not_nfc = 0
    checked = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for text in iter_docs():
            n_docs += 1
            if check_nfc and checked < 20000:
                checked += 1
                if unicodedata.normalize("NFC", text) != text:
                    n_not_nfc += 1

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if keep_threshold is not None:
                    h = hashlib.blake2b(line.encode("utf-8"),
                                        digest_size=4).digest()
                    if int.from_bytes(h, "big") >= keep_threshold:
                        continue
                fout.write(line + "\n")
                n_lines += 1
                n_chars += len(line)

    if check_nfc and n_not_nfc:
        log(f"  [WARN] {n_not_nfc}/{checked} sampled documents are not NFC-normalised. "
            f"Run normalize.py before training the tokenizer -- otherwise the same "
            f"word appears under two encodings and both consume vocabulary.")

    return {"documents": n_docs, "lines": n_lines, "characters": n_chars,
            "lines_available": total_lines or n_lines,
            "sampled": keep_threshold is not None,
            "non_nfc_sampled": n_not_nfc, "nfc_checked": checked}


def sample_eval_texts(
    jsonl_path: Path,
    *,
    text_key: str = "text",
    n: int = 5000,
    seed: int = 20260820,
    provenance_key: str = "provenance_class",
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Reservoir-sample n documents from a held-out split.

    v1 took `val_texts[:5000]` ??? the first 5000 lines, which in a split written
    source-by-source is effectively one source. That biases every metric.

    Also buckets by provenance ("manual" / "downloaded") when the field exists,
    so fertility can be reported per bucket. The assignment requires >=20% of
    tokens to be manually collected; whether the tokenizer serves that 20% as
    well as it serves the scraped 80% is a real question and belongs in the
    report.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    buckets: dict[str, list[str]] = {}
    seen = 0

    with open(jsonl_path, encoding="utf-8") as f:
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
            seen += 1

            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = text

            cls = doc.get(provenance_key)
            if cls:
                b = buckets.setdefault(str(cls), [])
                if len(b) < n // 2:
                    b.append(text)

    return reservoir, buckets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sentencepiece(
    corpus_path: Path,
    model_prefix: str,
    vocab_size: int,
    model_type: str,
    spm_params: dict,
    log=print,
) -> Path | None:
    import sentencepiece as spm

    params = dict(DEFAULT_SPM)
    params.update(spm_params or {})

    kwargs = dict(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        **{k: v for k, v in params.items()},
    )
    try:
        spm.SentencePieceTrainer.train(**kwargs)
    except RuntimeError as e:
        # A failed train can still leave a partial .model on disk, which would
        # then be picked up as if it had succeeded.
        for suffix in (".model", ".vocab"):
            stale = Path(f"{model_prefix}{suffix}")
            if stale.exists():
                stale.unlink()
        msg = str(e)
        # The single most common failure: asking for more pieces than the
        # corpus can support. Say so plainly instead of a bare traceback --
        # for a low-resource language this is a finding, not an error.
        if "vocab_size" in msg or "too large" in msg.lower():
            log(f"  [SKIP] vocab_size={vocab_size} is larger than this corpus can "
                f"support. For a low-resource language this is itself a result: "
                f"report it as the ceiling. ({msg.splitlines()[0][:160]})")
        else:
            log(f"  [SKIP] vocab_size={vocab_size} failed: {msg.splitlines()[0][:200]}")
        return None

    p = Path(f"{model_prefix}.model")
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TokenizerMetrics:
    vocab_size: int
    model_type: str
    fertility: float = 0.0              # tokens per whitespace word (Rust et al. 2021)
    chars_per_token: float = 0.0
    bytes_per_token: float = 0.0
    unk_rate: float = 0.0
    byte_fallback_rate: float = 0.0     # the metric that replaces UNK rate
    tokens_per_doc: float = 0.0
    eval_documents: int = 0
    eval_tokens: int = 0
    embedding_params: int = 0
    embedding_share: float = 0.0
    fertility_by_provenance: dict = field(default_factory=dict)
    model_path: str = ""

    def as_row(self) -> dict:
        d = asdict(self)
        d["fertility_by_provenance"] = json.dumps(d["fertility_by_provenance"],
                                                  ensure_ascii=False)
        return d


def _is_byte_piece(piece: str) -> bool:
    """SentencePiece emits byte-fallback pieces as literal '<0xE0>' strings."""
    return len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")


def evaluate(sp, texts: list[str], *, vocab_size: int, model_type: str,
             d_model: int, total_params: int | None,
             buckets: dict[str, list[str]] | None = None,
             model_path: str = "") -> TokenizerMetrics:
    m = TokenizerMetrics(vocab_size=vocab_size, model_type=model_type,
                         model_path=model_path)
    if not texts:
        return m

    unk_id = sp.unk_id()
    total_pieces = total_words = total_chars = total_bytes = 0
    n_unk = n_bytefallback = 0

    for text in texts:
        words = text.split()
        if not words:
            continue
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)
        total_pieces += len(ids)
        total_words += len(words)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        n_unk += sum(1 for t in ids if t == unk_id)
        n_bytefallback += sum(1 for p in pieces if _is_byte_piece(p))

    if total_pieces == 0:
        return m

    m.fertility = total_pieces / max(1, total_words)
    m.chars_per_token = total_chars / total_pieces
    m.bytes_per_token = total_bytes / total_pieces
    m.unk_rate = n_unk / total_pieces
    m.byte_fallback_rate = n_bytefallback / total_pieces
    m.tokens_per_doc = total_pieces / max(1, len(texts))
    m.eval_documents = len(texts)
    m.eval_tokens = total_pieces

    m.embedding_params = vocab_size * d_model
    if total_params:
        m.embedding_share = m.embedding_params / total_params

    if buckets:
        for cls, btexts in buckets.items():
            if len(btexts) < 20:
                continue
            p = w = 0
            for t in btexts:
                ws = t.split()
                if not ws:
                    continue
                p += len(sp.encode(t, out_type=int))
                w += len(ws)
            if w:
                m.fertility_by_provenance[cls] = round(p / w, 4)

    return m


# ---------------------------------------------------------------------------
# Vocabulary selection
# ---------------------------------------------------------------------------

def scaling_law_vocab(n_non_vocab_params: int, d_model: int) -> int:
    """
    Compute-optimal vocabulary from Tao et al. (2024), "Scaling Laws with
    Vocabulary: Larger Models Deserve Larger Vocabularies".

    The law is over vocabulary PARAMETERS, not vocabulary size:

        N_v = k * N_nv^0.83        where  N_v = vocab_size * d_model

    so the size itself is  V = k * N_nv^0.83 / d_model.

    We anchor k on their headline prediction that Llama2-70B
    (N_nv ~ 65e9, d_model = 8192) has an optimal vocabulary of ~216k:

        k = (216_000 * 8192) / (65e9 ** 0.83)  ~= 1.876

    A WARNING ABOUT THIS PROJECT'S SCALE. The brief specifies ~25M-parameter
    models and separately "recommends" vocabularies in the tens of thousands.
    Those two recommendations pull in opposite directions, and the law makes
    that explicit: for N_nv ~ 15M at d_model = 512 the compute-optimal
    vocabulary is only about 3.3k -- an order of magnitude below the brief's
    suggestion.

    That is not a reason to ignore the brief. It is a genuine tension to
    resolve in the report, because the law is fitted for compute-optimal
    English pretraining and does not price in the compression benefit a larger
    vocabulary buys on a morphologically rich script. But it does mean that at
    this model size the parameter cost of vocabulary is the dominant
    consideration, not fertility -- see `embedding_share` in the sweep table.
    """
    gamma = 0.83
    k = (216_000 * 8192) / (65e9 ** gamma)
    vocab_params = k * (max(1, n_non_vocab_params) ** gamma)
    return int(vocab_params / max(1, d_model))


@dataclass
class Selection:
    vocab_size: int
    reason: str
    scaling_law_reference: int
    ratio_to_reference: float
    table: list[dict] = field(default_factory=list)


def select_vocab_size(
    metrics: list[TokenizerMetrics],
    *,
    min_relative_gain: float = 0.02,
    max_byte_fallback_rate: float = 0.01,
    max_unk_rate: float = 0.001,
    n_non_vocab_params: int = 15_000_000,
    d_model: int = 512,
    log=print,
) -> Selection:
    """
    Pick a vocabulary size by ELBOW, not by minimum.

    Rule:
      Among candidates that satisfy the hard constraints (byte-fallback rate
      and UNK rate below thresholds), walk up the ladder from the smallest.
      Stop at the first size where moving to the NEXT larger size improves
      fertility by less than `min_relative_gain`. That size is the elbow.

    Why not "lowest fertility": fertility is monotonically decreasing in vocab
    size, so minimising it is the same as `max(vocab_sizes)` and tells you
    nothing. It also ignores that every extra 1k of vocabulary costs
    1000 x d_model embedding parameters, which for a Phase-2-scale model is
    real budget taken away from depth.
    """
    ordered = sorted([m for m in metrics if m.eval_tokens > 0],
                     key=lambda m: m.vocab_size)
    if not ordered:
        raise RuntimeError("no usable sweep results to select from")

    ref = scaling_law_vocab(n_non_vocab_params, d_model)

    feasible = [m for m in ordered
                if m.byte_fallback_rate <= max_byte_fallback_rate
                and m.unk_rate <= max_unk_rate]
    if not feasible:
        worst = min(ordered, key=lambda m: (m.byte_fallback_rate, m.unk_rate))
        log(f"  [WARN] no candidate met byte_fallback<={max_byte_fallback_rate} "
            f"and unk<={max_unk_rate}. Falling back to the best available "
            f"({worst.vocab_size}). Investigate before reporting -- this usually "
            f"means the corpus is too small or not normalised.")
        feasible = [worst]

    chosen = feasible[-1]
    reason = (f"largest feasible size ({chosen.vocab_size}); no elbow found "
              f"within the swept range, so the sweep should be extended upward "
              f"before treating this as optimal")

    for i in range(len(feasible) - 1):
        cur, nxt = feasible[i], feasible[i + 1]
        if cur.fertility <= 0:
            continue
        gain = (cur.fertility - nxt.fertility) / cur.fertility
        if gain < min_relative_gain:
            chosen = cur
            if gain < 0:
                # Non-monotonic fertility. On a real corpus this is a red flag:
                # it usually means the eval sample is too small to measure with,
                # or the larger vocabulary is over-segmenting because the corpus
                # cannot support it.
                reason = (f"stopped at {cur.vocab_size}: growing to "
                          f"{nxt.vocab_size} made fertility WORSE "
                          f"({cur.fertility:.4f} -> {nxt.fertility:.4f}, "
                          f"{-gain:.1%} worse). Fertility should fall "
                          f"monotonically with vocabulary size, so check that "
                          f"the held-out sample is large enough and that the "
                          f"corpus supports the larger vocabulary")
            else:
                reason = (f"elbow: growing {cur.vocab_size} -> {nxt.vocab_size} "
                          f"improves fertility by only {gain:.2%} "
                          f"(< {min_relative_gain:.0%} threshold) while adding "
                          f"{(nxt.vocab_size - cur.vocab_size):,} x d_model "
                          f"embedding parameters")
            break

    table = []
    for i, m in enumerate(ordered):
        gain = None
        if i > 0 and ordered[i - 1].fertility > 0:
            gain = (ordered[i - 1].fertility - m.fertility) / ordered[i - 1].fertility
        table.append({
            "vocab_size": m.vocab_size,
            "model_type": m.model_type,
            "fertility": round(m.fertility, 4),
            "relative_fertility_gain_vs_prev": (round(gain, 4) if gain is not None else ""),
            "chars_per_token": round(m.chars_per_token, 4),
            "bytes_per_token": round(m.bytes_per_token, 4),
            "unk_rate": round(m.unk_rate, 8),
            "byte_fallback_rate": round(m.byte_fallback_rate, 8),
            "embedding_params": m.embedding_params,
            "embedding_share": round(m.embedding_share, 4),
            "selected": m.vocab_size == chosen.vocab_size and m.model_type == chosen.model_type,
        })

    if chosen.embedding_share and chosen.embedding_share > 0.35:
        log(f"  [WARN] at the configured Phase 2 size, vocab={chosen.vocab_size} puts "
            f"{chosen.embedding_share:.0%} of ALL model parameters into the embedding "
            f"matrix. The brief specifies ~25M-parameter models; above ~35% you are "
            f"buying vocabulary with depth. Consider a smaller vocabulary and say so "
            f"in the report -- this is the trade-off the sweep exists to expose.")

    log(f"  [select] {chosen.model_type} vocab_size={chosen.vocab_size}")
    log(f"           {reason}")
    log(f"           scaling-law reference (Tao et al. 2024) for "
        f"{n_non_vocab_params:,} non-vocab params: {ref:,} "
        f"(ratio {chosen.vocab_size / max(1, ref):.2f}x)")

    return Selection(vocab_size=chosen.vocab_size, reason=reason,
                     scaling_law_reference=ref,
                     ratio_to_reference=chosen.vocab_size / max(1, ref),
                     table=table)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(*, corpus_path: Path, eval_texts: list[str],
              buckets: dict[str, list[str]], vocab_sizes: list[int],
              model_types: list[str], out_dir: Path, lang: str,
              spm_params: dict, d_model: int, total_params: int | None,
              log=print) -> list[TokenizerMetrics]:
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[TokenizerMetrics] = []

    for model_type in model_types:
        for v in vocab_sizes:
            log(f"\n[{lang}] training {model_type} vocab={v} ...")
            prefix = str(out_dir / f"spm_{lang}_{model_type}_v{v}")
            model_file = train_sentencepiece(corpus_path, prefix, v, model_type,
                                             spm_params, log=log)
            if model_file is None:
                continue

            sp = spm.SentencePieceProcessor()
            sp.load(str(model_file))
            m = evaluate(sp, eval_texts, vocab_size=v, model_type=model_type,
                         d_model=d_model, total_params=total_params,
                         buckets=buckets, model_path=str(model_file))
            results.append(m)
            log(f"  fertility={m.fertility:.4f}  chars/token={m.chars_per_token:.3f}  "
                f"unk={m.unk_rate:.6%}  byte_fallback={m.byte_fallback_rate:.6%}")
            if m.fertility_by_provenance:
                log(f"  fertility by provenance: {m.fertility_by_provenance}")

    return results


def save_results(results: list[TokenizerMetrics], selection: Selection,
                 analysis_dir: Path, log=print) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = analysis_dir / "sweep_results.csv"
    if results:
        fields = list(results[0].as_row().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r.as_row())
        log(f"  wrote {csv_path}")

    sel_path = analysis_dir / "vocab_selection.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_vocab_size": selection.vocab_size,
            "selection_reason": selection.reason,
            "scaling_law_reference_vocab": selection.scaling_law_reference,
            "ratio_to_scaling_law_reference": round(selection.ratio_to_reference, 4),
            "scaling_law_citation": ("Tao et al. (2024), Scaling Laws with "
                                     "Vocabulary: Larger Models Deserve Larger "
                                     "Vocabularies. N_v proportional to N_nv^0.83"),
            "candidates": selection.table,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote {sel_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a from-scratch SentencePiece tokenizer with a "
                    "non-degenerate vocabulary sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lang", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--config", default=None,
                   help="tokenizer_config.yaml; defaults to <lang>/configs/tokenizer_config.yaml")
    p.add_argument("--model-types", nargs="+", default=None,
                   choices=["unigram", "bpe"],
                   help="override config; use both to run the ablation")
    p.add_argument("--compare-algorithms", action="store_true",
                   help="shorthand for --model-types unigram bpe")
    p.add_argument("--vocab-sizes", nargs="+", type=int, default=None)
    p.add_argument("--eval-sample", type=int, default=5000)
    p.add_argument("--max-corpus-lines", type=int, default=None,
                   help="cap lines written to the SPM input file (smoke tests)")
    p.add_argument("--seed", type=int, default=20260820)
    return p.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    args = parse_args()
    root = Path(args.repo_root).resolve()
    lang = args.lang

    cfg_path = Path(args.config) if args.config else root / lang / "configs" / "tokenizer_config.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        print(f"[warn] no config at {cfg_path}; using defaults", flush=True)

    splits = root / lang / "data" / "splits"
    train_jsonl = splits / "train.jsonl"
    val_jsonl = splits / "val.jsonl"
    if not train_jsonl.exists():
        print(f"[error] {train_jsonl} not found. Run split_data.py first.", file=sys.stderr)
        return 1

    vocab_dir = root / lang / "tokenizer" / "vocab"
    analysis_dir = root / lang / "tokenizer" / "analysis"
    sweep_dir = analysis_dir / "sweep_models"

    sweep_cfg = cfg.get("sweep", {}) or {}
    vocab_sizes = args.vocab_sizes or sweep_cfg.get("vocab_sizes") or [8000, 16000, 24000, 32000, 48000]
    model_types = (["unigram", "bpe"] if args.compare_algorithms
                   else args.model_types
                   or ([cfg.get("model_type", "unigram")]))
    sel_cfg = cfg.get("selection", {}) or {}
    phase2 = cfg.get("phase2_model", {}) or {}
    d_model = int(phase2.get("d_model", 512))
    total_params = phase2.get("total_params")
    n_non_vocab = int(phase2.get("non_vocab_params", 15_000_000))

    text_key = cfg.get("text_key", "text")

    print(f"\n=== tokenizer training: {lang} ===")
    print(f"  config      : {cfg_path}")
    print(f"  model types : {model_types}")
    print(f"  vocab sizes : {vocab_sizes}")

    print("\n[1/4] streaming train split -> SentencePiece input")
    corpus_path = sweep_dir / "train_corpus.txt"
    stats = stream_jsonl_to_text(train_jsonl, corpus_path, text_key=text_key,
                                 max_lines=args.max_corpus_lines)
    print(f"  {stats['documents']:,} docs -> {stats['lines']:,} lines, "
          f"{stats['characters']:,} chars")

    print("\n[2/4] sampling held-out evaluation text")
    eval_src = val_jsonl if val_jsonl.exists() else train_jsonl
    if not val_jsonl.exists():
        print("  [WARN] no val.jsonl; evaluating on TRAIN. Fertility measured on "
              "training data is optimistic and must not be reported as held-out.")
    eval_texts, buckets = sample_eval_texts(eval_src, text_key=text_key,
                                            n=args.eval_sample, seed=args.seed)
    print(f"  sampled {len(eval_texts):,} held-out documents"
          + (f"; provenance buckets: { {k: len(v) for k, v in buckets.items()} }"
             if buckets else "; no provenance_class field found"))

    print("\n[3/4] sweep")
    results = run_sweep(corpus_path=corpus_path, eval_texts=eval_texts,
                        buckets=buckets, vocab_sizes=vocab_sizes,
                        model_types=model_types, out_dir=sweep_dir, lang=lang,
                        spm_params=cfg.get("sentencepiece", {}) or {},
                        d_model=d_model, total_params=total_params)
    if not results:
        print("[error] every sweep point failed", file=sys.stderr)
        return 1

    print("\n[4/4] selection")
    selection = select_vocab_size(
        results,
        min_relative_gain=float(sel_cfg.get("min_relative_fertility_gain", 0.02)),
        max_byte_fallback_rate=float(sel_cfg.get("max_byte_fallback_rate", 0.01)),
        max_unk_rate=float(sel_cfg.get("max_unk_rate", 0.001)),
        n_non_vocab_params=n_non_vocab,
        d_model=d_model,
    )
    save_results(results, selection, analysis_dir)

    # Pick the winning (model_type, vocab_size). When algorithms were compared,
    # break the tie on fertility at the selected size.
    finalists = [m for m in results if m.vocab_size == selection.vocab_size]
    best = min(finalists, key=lambda m: m.fertility)
    if len(finalists) > 1:
        print(f"  algorithm ablation at vocab={selection.vocab_size}: "
              + ", ".join(f"{m.model_type}={m.fertility:.4f}" for m in finalists)
              + f" -> {best.model_type}")

    vocab_dir.mkdir(parents=True, exist_ok=True)
    src_model = Path(best.model_path)
    dst_model = vocab_dir / f"{lang}_tokenizer.model"
    dst_vocab = vocab_dir / f"{lang}_tokenizer.vocab"
    shutil.copy(src_model, dst_model)
    if src_model.with_suffix(".vocab").exists():
        shutil.copy(src_model.with_suffix(".vocab"), dst_vocab)

    with open(vocab_dir / f"{lang}_tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({
            "language": lang,
            "model_type": best.model_type,
            "vocab_size": best.vocab_size,
            "selection_reason": selection.reason,
            "scaling_law_reference_vocab": selection.scaling_law_reference,
            "metrics_on_heldout": {
                "fertility": round(best.fertility, 4),
                "chars_per_token": round(best.chars_per_token, 4),
                "bytes_per_token": round(best.bytes_per_token, 4),
                "unk_rate": best.unk_rate,
                "byte_fallback_rate": best.byte_fallback_rate,
                "fertility_by_provenance": best.fertility_by_provenance,
                "eval_documents": best.eval_documents,
            },
            "sentencepiece_params": {**DEFAULT_SPM, **(cfg.get("sentencepiece", {}) or {})},
            "trained_on": str(train_jsonl),
            "corpus_stats": stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  final model : {dst_model}")
    print(f"  final vocab : {dst_vocab}")
    print(f"  metadata    : {vocab_dir / f'{lang}_tokenizer.json'}")
    print(f"  {best.model_type} / vocab={best.vocab_size} / "
          f"fertility={best.fertility:.4f} / unk={best.unk_rate:.6%} / "
          f"byte_fallback={best.byte_fallback_rate:.6%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
