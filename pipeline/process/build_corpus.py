"""
build_corpus.py — raw documents to train/val/test splits
=========================================================
One pass that does everything §1.2 asks for: "remove invalid/duplicate data;
normalize Unicode; create splits; report corpus statistics".

Stages, in the order that makes each one cheap:

  1. NORMALISE      NFC, zero-width strip, whitespace collapse, danda spacing.
                    Done first so every later comparison sees one encoding.
  2. VALIDATE       length, script ratio, Hindi-vs-Nepali check.
  3. QUALITY        Gopher-style repetition and symbol heuristics.
  4. EXACT DEDUP    hash of the aggressively-normalised form.
  5. NEAR DEDUP     MinHash + banded LSH, global across sources.
  6. BUDGET         trim to the manual/downloaded token targets.
  7. SPLIT          train/val/test, stratified by provenance and source.

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

WHY THE BUDGET TRIM RUNS HERE AND NOT AT COLLECTION
----------------------------------------------------
The target is 100M manual + 400M downloaded. You cannot hit that at collection
time, because deduplication has not happened yet: the collectors overshoot, and
how much survives depends on how much your scrape overlaps Sangraha -- which is
substantial, since Sangraha is itself built from crawls of the same publishers.

So collect long, dedup globally, and trim last, when the surviving counts are
known. The trim enforces three things in order:

    manual     <= --manual-target-tokens
    downloaded <= --downloaded-target-tokens
    downloaded <= manual * (1 - f') / f'        <-- the >=20% guarantee

where f' is the requirement plus --manual-fraction-margin (default 0.5pp).
The margin exists because the trim measures characters and the requirement
grades tokens: manual text and Sangraha have slightly different
characters-per-token, so trimming to exactly 20.00% of characters lands either
side of 20% of tokens, and half of those runs fail. Aiming at 20.5% makes the
error one-sided. Pass the measured per-class ratios from token_accounting.json
via --manual-chars-per-token / --downloaded-chars-per-token on a second pass
and the conversion becomes exact.

The third clause is the one that matters. If the scrape yields 70M manual
tokens instead of 100M, the first two clauses would happily ship 70M + 400M =
15% manual and fail the requirement. The third caps downloaded at 280M and
ships 350M at exactly 20%. A smaller corpus that meets the ratio beats a larger
one that does not, and the brief says so explicitly for the lower-resource
language.

Trimming spends the downloaded budget in source-priority order: Wikipedia
first (edited prose, small), then verified Sangraha, then unverified Sangraha
as the elastic filler. What gets dropped is the least-curated data, which is
the correct thing to lose.

Usage
-----
  python -m pipeline.process.build_corpus --lang hindi --repo-root .
  python -m pipeline.process.build_corpus --lang hindi \\
      --manual-target-tokens 100000000 --downloaded-target-tokens 400000000
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import iter_language_docs  # noqa: E402

# Lower number = spend the budget here first. Anything unlisted sorts last,
# which is the right default for a source nobody has vouched for.
SOURCE_PRIORITY = {
    "wikipedia": 1,
    "sangraha_verified": 2,
    "sangraha": 3,
    "sangraha_unverified": 4,
}
UNKNOWN_SOURCE_PRIORITY = 9

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
SYMBOLS = set("#…$%^&*_=+~<>|\\/@")

NEPALI_MARKERS = {"छ", "छन्", "छैन", "गर्छ", "गरेको", "भएको", "लाई", "सँग", "हुन्",
                  "भन्ने", "भन्दा", "गर्ने", "नेपाल", "काठमाडौं", "हरू", "बाट"}
HINDI_MARKERS = {"है", "हैं", "था", "थे", "किया", "गया", "रहा", "में", "नहीं",
                 "लेकिन", "क्योंकि", "भारत", "दिल्ली", "करने", "बताया", "को"}

_NUKTA = {"क़": "क़", "ख़": "ख़", "ग़": "ग़", "ज़": "ज़", "ड़": "ड़",
          "ढ़": "ढ़", "फ़": "फ़", "य़": "य़"}
_NUKTA_RE = re.compile("|".join(map(re.escape, _NUKTA)))
_DANDA_RE = re.compile(r"\s*([।॥])\s*")

BOILERPLATE = re.compile(
    r"^यह भी पढ|^इसे भी पढ|^और पढ|^पढ़ें|^यो पनि|^सम्बन्धित|^थप पढ|"
    r"^शेयर|^share|^फॉलो कर|^copyright|^सर्वाधिकार|^all rights reserved|"
    r"^विज्ञापन$|^advertisement$|^tags?:|^ट्याग|^टैग|^comments?$|"
    r"^टिप्पणी$|^प्रतिक्रिया$|^\s*\d+\s*$", re.I)


# ---------------------------------------------------------------------------
# 1. Normalise
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFC + nukta recomposition + zero-width strip + whitespace/danda tidy."""
    text = text.replace("﻿", "").replace("​", "").replace("­", "")
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
    return re.sub(r"[^\wऀ-ॿ]+", "", t)


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

    hi = sum(1 for w in words if w.strip("।,.!?\"'()") in HINDI_MARKERS)
    ne = sum(1 for w in words if w.strip("।,.!?\"'()") in NEPALI_MARKERS)
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

    def band_keys(self, text: str) -> list[tuple[int, bytes]]:
        """
        The banded LSH keys for one document. PURE -- no shared state read or
        written -- which is what lets it run in a worker process.

        This is where the time goes: ~5.6 ms of the 6.8 ms per document, all of
        it hashing shingles. The bucket bookkeeping that follows is dictionary
        operations measured in microseconds, and it MUST stay sequential
        because whether a document is a duplicate depends on every document
        seen before it.
        """
        sig = self._signature(text)
        keys = []
        for i in range(self.bands):
            chunk = sig[i * self.rows:(i + 1) * self.rows]
            keys.append((i, hashlib.blake2b(
                b"".join(x.to_bytes(8, "little") for x in chunk),
                digest_size=8).digest()))
        return keys

    def check_keys(self, doc_id: str, keys: list) -> str | None:
        """Sequential half: look the keys up, then claim them."""
        for key in keys:
            hit = self.buckets.get(key)
            if hit is not None and hit != doc_id:
                return hit
        for key in keys:
            self.buckets.setdefault(key, doc_id)
        return None

    def check_and_add(self, doc_id: str, text: str) -> str | None:
        return self.check_keys(doc_id, self.band_keys(text))


# ---------------------------------------------------------------------------
# Parallel document transform
# ---------------------------------------------------------------------------
#
# Every per-document step -- normalise, boilerplate strip, quality gates,
# content hash, MinHash signature -- is a pure function of the document text.
# Only the three pieces of bookkeeping that follow are order-dependent: the
# exact-duplicate set, the LSH buckets, and the output file.
#
# So the expensive part fans out across processes and the cheap part stays in
# the main one. Measured at 6.84 ms/doc serial; on 8 cores this brings a
# 1.6M-document Hindi build from ~3 hours to well under one.

_W: dict = {}


def _init_worker(lang: str, cfg: dict, threshold: float, num_perm: int,
                 seed: int) -> None:
    _W["lang"] = lang
    _W["cfg"] = cfg
    _W["lsh"] = MinHashLSH(threshold, num_perm=num_perm, seed=seed)


def _transform_batch(texts: list[str]) -> list[tuple]:
    """(kept_text | None, why, boilerplate_lines, content_hash, band_keys)."""
    lang, cfg, lsh = _W["lang"], _W["cfg"], _W["lsh"]
    out = []
    for raw in texts:
        text = normalize(raw)
        text, nbp = strip_boilerplate(text)
        ok, why = check(text, lang, cfg)
        if not ok:
            out.append((None, why, nbp, None, None))
            continue
        h = hashlib.blake2b(hash_form(text).encode(), digest_size=16).hexdigest()
        keys = lsh.band_keys(text) if lsh is not None else None
        out.append((text, "ok", nbp, h, keys))
    return out


def _batched(it, n: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


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


# ---------------------------------------------------------------------------
# 6. Budget trim
# ---------------------------------------------------------------------------

def apply_budget(records: list[tuple[str, str, str, int]],
                 *,
                 manual_target_tokens: int,
                 downloaded_target_tokens: int,
                 min_manual_fraction: float,
                 margin: float,
                 manual_cpt: float,
                 downloaded_cpt: float,
                 rng: random.Random) -> tuple[set[str], dict]:
    """
    Choose which surviving documents make the final corpus.

    `records` is [(doc_id, provenance_class, source, n_chars)] after dedup.
    Returns (keep_ids, report).

    THE THREE CAPS
    --------------
        manual     <= manual_target_tokens
        downloaded <= downloaded_target_tokens
        downloaded <= manual_kept * (1 - f') / f'

    The third is why this function exists. Without it a short scrape silently
    produces a corpus that fails the graded requirement, and you find out at
    the count stage after the tokenizer has already been trained on it.

    WHY f' IS THE REQUIREMENT PLUS A MARGIN
    ---------------------------------------
    Budgets are enforced in characters, because no tokenizer exists yet, but
    the requirement is graded in TOKENS. The two are not proportional: manual
    text (scraped news, OCR) and downloaded text (Sangraha) have slightly
    different characters-per-token, so a corpus trimmed to exactly 20.00% of
    characters lands a hair either side of 20% of tokens -- and half the time
    that is 19.99%, which fails.

    Targeting f + margin instead makes the outcome one-sided. The cost is a
    slightly smaller corpus; the brief says "~500M tokens" but ">=20% manual",
    and only one of those two is a hard threshold.

    On a second pass you can pass the measured per-class chars/token from
    token_accounting.json, at which point the conversion is exact and the
    margin is doing almost nothing.
    """
    by_class: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for doc_id, cls, src, n in records:
        by_class[cls].append((doc_id, src, n))

    def fill(items: list[tuple[str, str, int]], budget: int
             ) -> tuple[set[str], int, Counter, Counter]:
        """Greedy fill in source-priority order, shuffled within each source."""
        by_src: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for doc_id, src, n in items:
            by_src[src].append((doc_id, n))
        order = sorted(by_src, key=lambda s: (
            SOURCE_PRIORITY.get(s, UNKNOWN_SOURCE_PRIORITY), s))
        keep: set[str] = set()
        kept_chars = 0
        kept_by_src: Counter[str] = Counter()
        dropped_by_src: Counter[str] = Counter()
        for src in order:
            docs = by_src[src]
            rng.shuffle(docs)
            for doc_id, n in docs:
                if kept_chars + n <= budget:
                    keep.add(doc_id)
                    kept_chars += n
                    kept_by_src[src] += 1
                else:
                    dropped_by_src[src] += 1
        return keep, kept_chars, kept_by_src, dropped_by_src

    manual_items = by_class.get("manual", [])
    down_items = [d for cls, lst in by_class.items() if cls != "manual"
                  for d in lst]

    manual_available = sum(n for _, _, n in manual_items)
    down_available = sum(n for _, _, n in down_items)

    manual_target_chars = int(manual_target_tokens * manual_cpt)
    downloaded_target_chars = int(downloaded_target_tokens * downloaded_cpt)

    keep_manual, manual_chars, m_kept, m_dropped = fill(
        manual_items, manual_target_chars)

    # Ratio arithmetic in TOKENS -- the unit the requirement is written in --
    # then converted back to a character budget for the fill.
    f = min(0.99, min_manual_fraction + margin)
    manual_tokens_kept = manual_chars / manual_cpt
    ratio_cap_tokens = manual_tokens_kept * (1 - f) / f if f > 0 else float("inf")
    ratio_cap_chars = int(ratio_cap_tokens * downloaded_cpt)

    down_budget = min(downloaded_target_chars, ratio_cap_chars)

    # Which constraint actually bound. Greedy fill never lands exactly on a
    # budget -- it stops at the last document that fits -- so a naive
    # `ratio_cap < target` comparison reports the requirement as binding on
    # runs where it was comfortably met, and sends the user off to collect data
    # they do not need. Only call it binding when manual missed by something
    # that matters.
    shortfall = max(0, manual_target_chars - manual_chars)
    material = max(int(0.01 * manual_target_chars), 50_000)
    binding = ("ratio" if shortfall > material and ratio_cap_chars < downloaded_target_chars
               else "target")

    keep_down, down_chars, d_kept, d_dropped = fill(down_items, down_budget)

    total = manual_chars + down_chars
    est_manual_tokens = manual_chars / manual_cpt
    est_down_tokens = down_chars / downloaded_cpt
    est_total_tokens = est_manual_tokens + est_down_tokens

    report = {
        "manual_target_tokens": manual_target_tokens,
        "downloaded_target_tokens": downloaded_target_tokens,
        "min_manual_fraction": min_manual_fraction,
        "margin": margin,
        "effective_fraction_targeted": round(f, 4),
        "chars_per_token": {"manual": manual_cpt, "downloaded": downloaded_cpt},
        "manual_target_chars": manual_target_chars,
        "downloaded_target_chars": downloaded_target_chars,
        "available_chars": {"manual": manual_available,
                            "downloaded": down_available},
        "kept_chars": {"manual": manual_chars, "downloaded": down_chars},
        "estimated_kept_tokens": {"manual": int(est_manual_tokens),
                                  "downloaded": int(est_down_tokens),
                                  "total": int(est_total_tokens)},
        "downloaded_budget_used": down_budget,
        "downloaded_budget_binding_constraint": binding,
        "ratio_cap_chars": ratio_cap_chars,
        "manual_char_fraction": round(manual_chars / total, 4) if total else 0.0,
        "estimated_manual_token_fraction": (
            round(est_manual_tokens / est_total_tokens, 4) if est_total_tokens else 0.0),
        "documents_kept_by_source": {"manual": dict(m_kept),
                                     "downloaded": dict(d_kept)},
        "documents_dropped_over_budget_by_source": {
            "manual": dict(m_dropped), "downloaded": dict(d_dropped)},
        "manual_shortfall_chars": shortfall,
        "materiality_threshold_chars": material,
    }
    return keep_manual | keep_down, report


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
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
    ap.add_argument("--workers", type=int, default=0,
                    help="processes for the per-document transform "
                         "(0 = cpu_count-1). The order-dependent dedup "
                         "bookkeeping stays sequential regardless.")
    ap.add_argument("--batch-size", type=int, default=400,
                    help="documents per task sent to a worker")
    ap.add_argument("--minhash-perm", type=int, default=96,
                    help="MinHash permutations; fewer is faster and slightly "
                         "less precise")
    # These four default to None so the config file is the source of truth and
    # a flag means "override the config", not "silently match its default".
    ap.add_argument("--manual-target-tokens", type=int, default=None,
                    help="cap on manual tokens; default from data_config.yaml")
    ap.add_argument("--downloaded-target-tokens", type=int, default=None,
                    help="cap on downloaded tokens; also capped by the ratio")
    ap.add_argument("--min-manual-fraction", type=float, default=None,
                    help="the graded requirement; downloaded is trimmed until "
                         "this holds even if manual falls short of its target")
    ap.add_argument("--chars-per-token", type=float, default=None,
                    help="proxy used to convert the token budgets to character "
                         "budgets; replace with the measured value on pass 2")
    ap.add_argument("--manual-chars-per-token", type=float, default=None,
                    help="pass 2: measured value for manual text, from "
                         "token_accounting.json -> measured_chars_per_token")
    ap.add_argument("--downloaded-chars-per-token", type=float, default=None,
                    help="pass 2: measured value for downloaded text")
    ap.add_argument("--manual-fraction-margin", type=float, default=None,
                    help="aim this far above --min-manual-fraction, so that "
                         "character-to-token drift cannot land you under it")
    ap.add_argument("--from-clean", action="store_true",
                    help="skip normalise/filter/dedup and re-trim from the "
                         "existing interim/clean.jsonl. Use when only the "
                         "budget targets changed -- minutes instead of hours. "
                         "Re-run the full pass if raw data or filters changed.")
    ap.add_argument("--no-budget", action="store_true",
                    help="keep everything that survives dedup, ignore the "
                         "targets (use to measure how much you actually have)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang
    cfg = dict(DEFAULT_CFG)

    targets: dict = {}
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
            targets = user.get("targets") or {}
            print(f"  config: {cfg_path}")
        except Exception as e:
            print(f"  [warn] could not read {cfg_path}: {e}")

    def pick(cli, key, fallback):
        return cli if cli is not None else targets.get(key, fallback)

    manual_target_tokens = pick(args.manual_target_tokens,
                                "manual_target_tokens", 100_000_000)
    downloaded_target_tokens = pick(args.downloaded_target_tokens,
                                    "downloaded_target_tokens", 400_000_000)
    min_manual_fraction = pick(args.min_manual_fraction,
                               "min_manual_fraction", 0.20)
    chars_per_token = pick(args.chars_per_token, "chars_per_token_proxy", 4.0)
    margin = pick(args.manual_fraction_margin, "manual_fraction_margin", 0.005)
    manual_cpt = args.manual_chars_per_token or chars_per_token
    down_cpt = args.downloaded_chars_per_token or chars_per_token

    if args.workers <= 0:
        args.workers = max(1, (os.cpu_count() or 2) - 1)

    print(f"[{lang}] building corpus  "
          f"({args.workers} worker{'s' if args.workers != 1 else ''}, "
          f"near-dedup {'off' if args.no_near_dedup else f'on/{args.minhash_perm}perm'})")
    counters: Counter[str] = Counter()
    seen_exact: set[str] = set()
    lsh = None if args.no_near_dedup else MinHashLSH(
        cfg["near_dup_threshold"], num_perm=args.minhash_perm)

    interim_dir = root / lang / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    clean_path = interim_dir / "clean.jsonl"

    kept_records = []
    n_in = 0
    t_start = time.monotonic()

    # ---- fast path: re-trim from the already-cleaned pool -------------------
    #
    # Cleaning and deduplication are deterministic and expensive -- 85 minutes
    # on 1.4M documents. The BUDGET TRIM that follows is neither: it is a
    # greedy fill over records already in memory, and it is the part you
    # actually want to re-run when a token target changes.
    #
    # `clean.jsonl` is the output of the expensive half. If it exists and you
    # only want different budget numbers, read it back and skip straight to the
    # trim. Minutes instead of hours, and byte-identical to what a full rebuild
    # would produce, because nothing upstream of the trim depends on the
    # budget.
    #
    # Re-run the full pass whenever the RAW data or the cleaning config
    # changes -- this path cannot see either.
    if args.from_clean:
        if not clean_path.exists():
            print(f"[error] --from-clean needs {clean_path}, which does not "
                  f"exist. Run a full build first.", file=sys.stderr)
            return 1
        prev = root / lang / "data" / "stats" / "corpus_stats.json"
        if prev.exists():
            try:
                old_stats = json.loads(prev.read_text(encoding="utf-8"))
                counters.update(old_stats.get("counters") or {})
                n_in = old_stats.get("raw_documents", 0)
            except Exception:
                pass
        print(f"  --from-clean: reusing {clean_path.name} "
              f"(skipping normalise/filter/dedup)")
        with open(clean_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kept_records.append((rec["doc_id"],
                                     rec.get("provenance_class", "UNLABELLED"),
                                     rec.get("source", "?"),
                                     len(rec.get("text") or "")))
        counters["kept"] = len(kept_records)
        print(f"  {len(kept_records):,} cleaned documents loaded in "
              f"{time.monotonic() - t_start:.0f}s")

    def consume(rec: dict, res: tuple) -> None:
        """Sequential bookkeeping. Order-dependent, so it stays single-threaded."""
        nonlocal n_in
        text, why, n_bp, h, keys = res
        counters["boilerplate_lines_removed"] += n_bp
        if text is None:
            counters[f"drop:{why}"] += 1
            return
        if h in seen_exact:
            counters["drop:exact_duplicate"] += 1
            return
        seen_exact.add(h)
        doc_id = rec.get("doc_id") or h[:20]
        if lsh is not None and keys is not None:
            if lsh.check_keys(doc_id, keys):
                counters["drop:near_duplicate"] += 1
                return
        rec["text"] = text
        rec["doc_id"] = doc_id
        rec.setdefault("provenance_class", "UNLABELLED")
        if rec["provenance_class"] == "UNLABELLED":
            counters["unlabelled_provenance"] += 1
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept_records.append((doc_id, rec["provenance_class"],
                             rec.get("source", "?"), len(text)))
        counters["kept"] += 1

    def progress() -> None:
        el = time.monotonic() - t_start
        print(f"  {n_in:,} in, {counters['kept']:,} kept, "
              f"{n_in / max(1e-9, el):.0f} docs/s, {el / 60:.0f} min", flush=True)

    def prefilter(rec: dict) -> bool:
        if not (rec.get("text") or "").strip():
            counters["empty"] += 1
            return False
        if rec.get("language") not in (None, lang):
            counters["wrong_language_field"] += 1
            return False
        return True

    # nullcontext when re-trimming: clean.jsonl was just READ, so it must not
    # be reopened for write.
    out_cm = (contextlib.nullcontext(None) if args.from_clean
              else open(clean_path, "w", encoding="utf-8"))
    with out_cm as out:
        if args.from_clean:
            pass                      # kept_records already populated above
        elif args.workers == 1:
            # Serial path: same code, same worker globals, no process pool.
            # Kept so the parallel result can be diffed against it.
            _init_worker(lang, cfg, cfg["near_dup_threshold"],
                         args.minhash_perm, 20260820)
            for rec in iter_language_docs(root / lang):
                n_in += 1
                if not prefilter(rec):
                    continue
                consume(rec, _transform_batch([rec["text"]])[0])
                if n_in % 50000 == 0:
                    progress()
        else:
            from concurrent.futures import ProcessPoolExecutor
            src = (r for r in iter_language_docs(root / lang) if prefilter(r))
            with ProcessPoolExecutor(
                    max_workers=args.workers, initializer=_init_worker,
                    initargs=(lang, cfg, cfg["near_dup_threshold"],
                              args.minhash_perm,
                              args.seed if lsh is None else 20260820)) as ex:
                # Batches keep IPC overhead per document negligible; `map`
                # preserves order, which the exact-duplicate and LSH tie-breaks
                # depend on -- manual documents must still be seen first.
                batches = _batched(src, args.batch_size)
                pending: list = []
                for batch in batches:
                    pending.append((batch, ex.submit(_transform_batch,
                                                     [r["text"] for r in batch])))
                    if len(pending) < args.workers * 2:
                        continue
                    recs, fut = pending.pop(0)
                    for rec, res in zip(recs, fut.result()):
                        n_in += 1
                        consume(rec, res)
                    if n_in % 50000 < args.batch_size:
                        progress()
                for recs, fut in pending:
                    for rec, res in zip(recs, fut.result()):
                        n_in += 1
                        consume(rec, res)
            progress()

    print(f"\n  read {n_in:,} raw documents, kept {counters['kept']:,}")
    for k, v in sorted(counters.items(), key=lambda kv: -kv[1]):
        if k.startswith("drop:") or k in ("empty", "unlabelled_provenance"):
            print(f"    {k:<32} {v:>9,}")

    if counters["unlabelled_provenance"]:
        print(f"\n  [WARN] {counters['unlabelled_provenance']:,} documents have no "
              f"provenance_class.\n"
              f"         The >=20% manual requirement cannot be verified for them.\n"
              f"         Fix the collector, not this stage.")

    # ---- budget trim -------------------------------------------------------
    rng = random.Random(args.seed)
    budget_report = None
    if args.no_budget:
        keep_ids = {doc_id for doc_id, _, _, _ in kept_records}
        print("\n  [budget] disabled (--no-budget): keeping everything that "
              "survived dedup")
        av = Counter()
        for _, cls, _, n in kept_records:
            av[cls] += n
        tot = sum(av.values())
        print(f"           available: manual {av['manual'] / 1e6:.1f}M chars, "
              f"downloaded {av['downloaded'] / 1e6:.1f}M chars, "
              f"manual share {av['manual'] / tot if tot else 0:.1%}")
    else:
        keep_ids, budget_report = apply_budget(
            kept_records,
            manual_target_tokens=manual_target_tokens,
            downloaded_target_tokens=downloaded_target_tokens,
            min_manual_fraction=min_manual_fraction,
            margin=margin,
            manual_cpt=manual_cpt,
            downloaded_cpt=down_cpt,
            rng=rng)

        br = budget_report
        km, kd = br["kept_chars"]["manual"], br["kept_chars"]["downloaded"]
        am, ad = br["available_chars"]["manual"], br["available_chars"]["downloaded"]
        et = br["estimated_kept_tokens"]
        measured = args.manual_chars_per_token is not None
        print(f"\n  --- budget trim ---")
        print(f"    chars/token used: manual {manual_cpt}, downloaded {down_cpt}"
              + ("  (measured)" if measured else "  (proxy)"))
        print(f"    targeting >={br['effective_fraction_targeted']:.1%} manual "
              f"= requirement {min_manual_fraction:.0%} + {margin:.1%} margin "
              f"for char/token drift")
        print(f"    manual      {am:>15,} chars available -> {km:>15,} kept "
              f" (~{et['manual']:,} tokens)")
        print(f"    downloaded  {ad:>15,} chars available -> {kd:>15,} kept "
              f" (~{et['downloaded']:,} tokens)")
        print(f"    estimated corpus: {et['total']:,} tokens, "
              f"{br['estimated_manual_token_fraction']:.2%} manual")
        print(f"    downloaded budget set by: {br['downloaded_budget_binding_constraint']}")
        if br["downloaded_budget_binding_constraint"] == "ratio":
            print(f"      the >={min_manual_fraction:.0%} requirement is binding, not "
                  f"your {downloaded_target_tokens:,}-token target.")
            print(f"      manual came up "
                  f"{br['manual_shortfall_chars'] / manual_cpt:,.0f} tokens short, "
                  f"so downloaded was capped at "
                  f"{br['ratio_cap_chars'] / down_cpt:,.0f} to hold the ratio.")
            print(f"      Final corpus will be about {et['total']:,} tokens, not "
                  f"{(manual_target_tokens + downloaded_target_tokens):,}.")
            print(f"      To ship the full target instead, collect more manual data "
                  f"(--stage scrape,ocr) and re-run build.")
        dropped = sum(sum(v.values()) for v in
                      br["documents_dropped_over_budget_by_source"].values())
        if dropped:
            print(f"    {dropped:,} documents dropped as over budget "
                  f"(least-curated sources first)")

        kept_records = [r for r in kept_records if r[0] in keep_ids]

    # ---- split, stratified by (provenance_class, source) -------------------
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
            if rec["doc_id"] not in keep_ids:
                counters["drop:over_budget"] += 1
                continue
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
        "documents_in_final_corpus": len(keep_ids),
        "budget": budget_report,
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
