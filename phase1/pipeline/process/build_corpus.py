"""
build_corpus.py ??? raw documents to train/val/test splits
=========================================================
One pass that does everything ??1.2 asks for: "remove invalid/duplicate data;
normalize Unicode; create splits; report corpus statistics".

Stages, in the order that makes each one cheap:

  1. NORMALISE      NFC, zero-width strip, whitespace collapse, danda spacing.
                    Done first so every later comparison sees one encoding.
  2. VALIDATE       length, script ratio, Hindi-vs-Nepali check.
  3. QUALITY        Gopher-style repetition and symbol heuristics.
  4. EXACT DEDUP    hash of the aggressively-normalised form.
  5. NEAR DEDUP     MinHash + banded LSH, global across sources.
  6. SPLIT          train/val/test, stratified by provenance and source.

WHY DEDUP RUNS BEFORE SPLIT AND WHY IT IS GLOBAL
-------------------------------------------------
If a document appears in both Sangraha and your scrape, and dedup runs *after*
splitting, the two copies can land in train and test. Your Phase 2 perplexity
then measures memorisation, not language modelling, and the number is not
recoverable after the fact.

Global also means across sources within a language, not just within a file.
Syndicated news is the common case: the same wire copy on eight sites.

WHY THE SPLIT IS STRATIFIED
---------------------------
A random split of a corpus that is 20% manual gives you a test set that is
roughly 20% manual only in expectation. Stratifying by `provenance_class` and
`source` guarantees every split has representative proportions, so held-out
fertility and perplexity are comparable across them -- and so the tokenizer's
`fertility_by_provenance` reporting has enough of each class to be meaningful.

Usage
-----
  python -m pipeline.process.build_corpus --lang hindi --repo-root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import iter_language_docs  # noqa: E402

DEVANAGARI = re.compile(r"[???-???]")
LATIN = re.compile(r"[A-Za-z]")
SYMBOLS = set("#???$%^&*_=+~<>|\\/@")

NEPALI_MARKERS = {"???", "?????????", "?????????", "????????????", "???????????????", "????????????", "?????????", "?????????", "????????????",
                  "???????????????", "???????????????", "???????????????", "???????????????", "????????????????????????", "?????????", "?????????"}
HINDI_MARKERS = {"??????", "?????????", "??????", "??????", "????????????", "?????????", "?????????", "?????????", "????????????",
                 "???????????????", "?????????????????????", "????????????", "??????????????????", "????????????", "???????????????", "??????"}

_NUKTA = {"??????": "??????", "??????": "??????", "??????": "??????", "??????": "??????", "??????": "??????",
          "??????": "??????", "??????": "??????", "??????": "??????"}
_NUKTA_RE = re.compile("|".join(map(re.escape, _NUKTA)))
_DANDA_RE = re.compile(r"\s*([??????])\s*")

BOILERPLATE = re.compile(
    r"^?????? ?????? ??????|^????????? ?????? ??????|^?????? ??????|^???????????????|^?????? ?????????|^???????????????????????????|^?????? ??????|"
    r"^????????????|^share|^???????????? ??????|^copyright|^??????????????????????????????|^all rights reserved|"
    r"^????????????????????????$|^advertisement$|^tags?:|^???????????????|^?????????|^comments?$|"
    r"^?????????????????????$|^?????????????????????????????????$|^\s*\d+\s*$", re.I)


# ---------------------------------------------------------------------------
# 1. Normalise
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFC + nukta recomposition + zero-width strip + whitespace/danda tidy."""
    text = text.replace("???", "").replace("???", "").replace("??", "")
    text = unicodedata.normalize("NFC", text)
    text = _NUKTA_RE.sub(lambda m: _NUKTA[m.group(0)], text)
    text = _DANDA_RE.sub(r"\1 ", text)
    lines = [" ".join(l.split()) for l in text.splitlines()]
    return "\n".join(l for l in lines if l).strip()


def strip_boilerplate(text: str) -> tuple[str, int]:
    kept, dropped = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if len(s) < 120 and BOILERPLATE.search(s):
            dropped += 1
            continue
        kept.append(s)
    return "\n".join(kept), dropped


def hash_form(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = t.translate(dict.fromkeys([0x0964, 0x0965, 0x0970]))
    return re.sub(r"[^\w???-???]+", "", t)


# ---------------------------------------------------------------------------
# 2-3. Validity and quality
# ---------------------------------------------------------------------------

def check(text: str, lang: str, cfg: dict) -> tuple[bool, str]:
    if len(text) < cfg["min_chars"]:
        return False, "too_short_chars"
    words = text.split()
    if len(words) < cfg["min_words"]:
        return False, "too_few_words"

    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return False, "empty"
    dev = len(DEVANAGARI.findall(stripped)) / len(stripped)
    if dev < cfg["min_devanagari_ratio"]:
        return False, "low_devanagari"
    if len(LATIN.findall(stripped)) / len(stripped) > cfg["max_latin_ratio"]:
        return False, "code_mixed"

    hi = sum(1 for w in words if w.strip("???,.!?\"'()") in HINDI_MARKERS)
    ne = sum(1 for w in words if w.strip("???,.!?\"'()") in NEPALI_MARKERS)
    if hi + ne < 3:
        return False, "no_language_markers"
    if lang == "hindi" and hi <= ne:
        return False, "wrong_language"
    if lang == "nepali" and ne <= hi:
        return False, "wrong_language"

    uniq = len(set(words)) / len(words)
    if uniq < cfg["min_unique_word_ratio"]:
        return False, "repetitive"
    mean_wl = sum(len(w) for w in words) / len(words)
    if not (2.0 <= mean_wl <= 12.0):
        return False, "bad_mean_word_length"
    sym = sum(1 for w in words if any(c in SYMBOLS for c in w)) / len(words)
    if sym > cfg["max_symbol_ratio"]:
        return False, "symbol_heavy"

    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        dup_lines = 1 - len(set(lines)) / len(lines)
        if dup_lines > cfg["max_duplicate_line_ratio"]:
            return False, "duplicate_lines"
    return True, "ok"


# ---------------------------------------------------------------------------
# 5. Near-duplicate detection
# ---------------------------------------------------------------------------

class MinHashLSH:
    """
    Banded MinHash over word 5-grams. Pure stdlib + a little arithmetic so the
    pipeline has no heavyweight dependency; ~40 bytes per (document, band).
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 96, k: int = 5,
                 seed: int = 20260820):
        self.k = k
        self.num_perm = num_perm
        self.bands, self.rows = self._geometry(num_perm, threshold)
        rng = random.Random(seed)
        self.a = [rng.randrange(1, (1 << 61) - 1) for _ in range(num_perm)]
        self.b = [rng.randrange(0, (1 << 61) - 1) for _ in range(num_perm)]
        self.buckets: dict[tuple[int, bytes], str] = {}

    @staticmethod
    def _geometry(num_perm: int, threshold: float) -> tuple[int, int]:
        best, err = (num_perm, 1), 1e9
        for b in range(1, num_perm + 1):
            r = num_perm // b
            if r < 1:
                continue
            e = abs(((1.0 / b) ** (1.0 / r)) - threshold)
            if e < err:
                err, best = e, (b, r)
        return best

    def _signature(self, text: str) -> list[int]:
        words = text.split()
        if len(words) < self.k:
            shingles = {" ".join(words)} if words else {""}
        else:
            shingles = {" ".join(words[i:i + self.k])
                        for i in range(len(words) - self.k + 1)}
        hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")
              for s in shingles]
        if not hs:
            hs = [0]
        P = (1 << 61) - 1
        return [min((a * h + b) % P for h in hs)
                for a, b in zip(self.a, self.b)]

    def check_and_add(self, doc_id: str, text: str) -> str | None:
        sig = self._signature(text)
        keys = []
        for i in range(self.bands):
            chunk = sig[i * self.rows:(i + 1) * self.rows]
            key = (i, hashlib.blake2b(
                b"".join(x.to_bytes(8, "little") for x in chunk),
                digest_size=8).digest())
            keys.append(key)
            hit = self.buckets.get(key)
            if hit is not None and hit != doc_id:
                return hit
        for key in keys:
            self.buckets.setdefault(key, doc_id)
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    "min_chars": 400,
    "min_words": 50,
    "min_devanagari_ratio": 0.70,
    "max_latin_ratio": 0.12,
    "min_unique_word_ratio": 0.30,
    "max_symbol_ratio": 0.10,
    "max_duplicate_line_ratio": 0.30,
    "near_dup_threshold": 0.80,
    "split_ratios": [0.98, 0.01, 0.01],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config", default=None,
                    help="YAML overriding DEFAULT_CFG; default "
                         "<lang>/configs/data_config.yaml if present")
    ap.add_argument("--no-near-dedup", action="store_true",
                    help="skip MinHash (faster; keeps near-duplicates)")
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang
    cfg = dict(DEFAULT_CFG)

    cfg_path = Path(args.config) if args.config else \
        root / lang / "configs" / "data_config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            user = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg.update({k: v for k, v in (user.get("filters") or {}).items()
                        if k in cfg})
            if user.get("split_ratios"):
                cfg["split_ratios"] = user["split_ratios"]
            print(f"  config: {cfg_path}")
        except Exception as e:
            print(f"  [warn] could not read {cfg_path}: {e}")

    print(f"[{lang}] building corpus")
    counters: Counter[str] = Counter()
    seen_exact: set[str] = set()
    lsh = None if args.no_near_dedup else MinHashLSH(cfg["near_dup_threshold"])

    interim_dir = root / lang / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    clean_path = interim_dir / "clean.jsonl"

    kept_records = []
    n_in = 0
    with open(clean_path, "w", encoding="utf-8") as out:
        for rec in iter_language_docs(root / lang):
            n_in += 1
            if n_in % 50000 == 0:
                print(f"  {n_in:,} in, {counters['kept']:,} kept", flush=True)

            text = rec.get("text") or ""
            if not text.strip():
                counters["empty"] += 1
                continue
            if rec.get("language") not in (None, lang):
                counters["wrong_language_field"] += 1
                continue

            text = normalize(text)
            text, n_bp = strip_boilerplate(text)
            counters["boilerplate_lines_removed"] += n_bp

            ok, why = check(text, lang, cfg)
            if not ok:
                counters[f"drop:{why}"] += 1
                continue

            hf = hash_form(text)
            h = hashlib.blake2b(hf.encode(), digest_size=16).hexdigest()
            if h in seen_exact:
                counters["drop:exact_duplicate"] += 1
                continue
            seen_exact.add(h)

            doc_id = rec.get("doc_id") or h[:20]
            if lsh is not None:
                match = lsh.check_and_add(doc_id, text)
                if match:
                    counters["drop:near_duplicate"] += 1
                    continue

            rec["text"] = text
            rec["doc_id"] = doc_id
            rec.setdefault("provenance_class", "UNLABELLED")
            if rec["provenance_class"] == "UNLABELLED":
                counters["unlabelled_provenance"] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept_records.append((doc_id, rec["provenance_class"],
                                 rec.get("source", "?"), len(text)))
            counters["kept"] += 1

    print(f"\n  read {n_in:,} raw documents, kept {counters['kept']:,}")
    for k, v in sorted(counters.items(), key=lambda kv: -kv[1]):
        if k.startswith("drop:") or k in ("empty", "unlabelled_provenance"):
            print(f"    {k:<32} {v:>9,}")

    if counters["unlabelled_provenance"]:
        print(f"\n  [WARN] {counters['unlabelled_provenance']:,} documents have no "
              f"provenance_class.\n"
              f"         The >=20% manual requirement cannot be verified for them.\n"
              f"         Fix the collector, not this stage.")

    # ---- split, stratified by (provenance_class, source) -------------------
    rng = random.Random(args.seed)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for doc_id, cls, src, _ in kept_records:
        strata[(cls, src)].append(doc_id)

    tr_r, va_r, te_r = cfg["split_ratios"]
    assign: dict[str, str] = {}
    for key, ids in strata.items():
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(n * va_r)) if n >= 20 else 0
        n_test = max(1, int(n * te_r)) if n >= 20 else 0
        for i, d in enumerate(ids):
            assign[d] = ("val" if i < n_val
                         else "test" if i < n_val + n_test
                         else "train")

    splits_dir = root / lang / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    handles = {name: open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8")
               for name in ("train", "val", "test")}
    per_split: Counter[str] = Counter()
    per_split_class: dict[str, Counter] = {k: Counter() for k in handles}
    chars_split_class: dict[str, Counter] = {k: Counter() for k in handles}
    try:
        for rec in (json.loads(l) for l in open(clean_path, encoding="utf-8") if l.strip()):
            name = assign.get(rec["doc_id"], "train")
            handles[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_split[name] += 1
            cls = rec.get("provenance_class", "UNLABELLED")
            per_split_class[name][cls] += 1
            chars_split_class[name][cls] += len(rec["text"])
    finally:
        for h in handles.values():
            h.close()

    print(f"\n  splits (stratified by provenance and source):")
    for name in ("train", "val", "test"):
        cc = chars_split_class[name]
        tot = sum(cc.values())
        frac = cc.get("manual", 0) / tot if tot else 0
        print(f"    {name:<6} {per_split[name]:>9,} docs  "
              f"{dict(per_split_class[name])}  manual={frac:.1%} of chars")

    stats = {
        "language": lang,
        "raw_documents": n_in,
        "kept_documents": counters["kept"],
        "counters": dict(counters),
        "splits": {n: per_split[n] for n in ("train", "val", "test")},
        "split_documents_by_class": {n: dict(per_split_class[n]) for n in per_split_class},
        "split_characters_by_class": {n: dict(chars_split_class[n]) for n in chars_split_class},
        "config": cfg,
        "note": ("Character fractions are a proxy. The graded manual fraction is "
                 "in TOKENS -- run pipeline/tokenizer/count_corpus_tokens.py after "
                 "the tokenizer exists."),
    }
    stats_path = root / lang / "data" / "stats" / "corpus_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n  wrote {splits_dir}/{{train,val,test}}.jsonl")
    print(f"  wrote {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
