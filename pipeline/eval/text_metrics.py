"""Self-contained BLEU-4 / chrF / ROUGE-L / distinct-n / repetition-rate.

Implemented directly (no sacrebleu / rouge-score dependency) so the eval
step has no extra network install to fail during a time-boxed cloud run.
Formulas follow the standard definitions (Papineni et al. 2002 for BLEU,
Popovic 2015 for chrF, Lin 2004 for ROUGE-L).
"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu4(hyps: list[list[str]], refs: list[list[str]]) -> float:
    """Corpus-level BLEU-4, one reference per hypothesis, add-1 smoothing on
    each n-gram precision (avoids the score collapsing to 0 for a partially
    trained model that gets zero 4-gram matches)."""
    clipped = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, 5):
            h_ng = _ngrams(hyp, n)
            r_ng = _ngrams(ref, n)
            clipped[n - 1] += sum(min(c, r_ng[g]) for g, c in h_ng.items())
            total[n - 1] += max(0, len(hyp) - n + 1)
    precisions = [(clipped[i] + 1) / (total[i] + 1) for i in range(4)]
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    return bp * geo_mean


def sentence_chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    def char_ngrams(s: str, n: int) -> Counter:
        s = s.replace(" ", "")
        return Counter(s[i : i + n] for i in range(len(s) - n + 1))

    p_sum, r_sum, count = 0.0, 0.0, 0
    for k in range(1, n + 1):
        h_ng, r_ng = char_ngrams(hyp, k), char_ngrams(ref, k)
        match = sum(min(c, r_ng[g]) for g, c in h_ng.items())
        h_total, r_total = sum(h_ng.values()), sum(r_ng.values())
        if h_total == 0 or r_total == 0:
            continue
        p_sum += match / h_total
        r_sum += match / r_total
        count += 1
    if count == 0:
        return 0.0
    p, r = p_sum / count, r_sum / count
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    scores = [sentence_chrf(h, r) for h, r in zip(hyps, refs)]
    return sum(scores) / max(1, len(scores))


def _lcs_len(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def sentence_rouge_l(hyp: list[str], ref: list[str], beta: float = 1.2) -> float:
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_len(hyp, ref)
    p, r = lcs / len(hyp), lcs / len(ref)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (r + beta**2 * p)


def corpus_rouge_l(hyps: list[list[str]], refs: list[list[str]]) -> float:
    scores = [sentence_rouge_l(h, r) for h, r in zip(hyps, refs)]
    return sum(scores) / max(1, len(scores))


def distinct_n(token_lists: list[list[str]], n: int) -> float:
    """Unique n-grams / total n-grams across every generation."""
    total, uniq = 0, set()
    for toks in token_lists:
        for g in _ngrams(toks, n):
            uniq.add(g)
            total += 1
    return len(uniq) / max(1, total)


def repetition_rate(tokens: list[str], n: int = 4) -> float:
    """Fraction of n-grams in a single generation that are repeats of an
    earlier n-gram in the same generation."""
    ng = _ngrams(tokens, n)
    total = sum(ng.values())
    repeated = sum(c - 1 for c in ng.values() if c > 1)
    return repeated / max(1, total)
