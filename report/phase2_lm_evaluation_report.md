# Phase 2 — Language Modeling Evaluation

| | |
|---|---|
| Generated (UTC) | 2026-09-07 18:23:01 |
| Git commit | `3d2fb366c7fcf2e0757186cd592d0b4076e307a7` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## PPL / BPB, side by side

| | Model H (Hindi) | Model L (Nepali) |
|---|--:|--:|
| Checkpoint step | 6500 | 4999 |
| Test cross-entropy (nats) | 2.7803 | 3.4681 |
| Test perplexity | 16.12 | 32.07 |
| Test bits-per-byte | 0.3088 | 0.5158 |
| Val perplexity | 14.62 | 32.51 |



## Why BPB, not just PPL

Perplexity is computed per *token*, and the two languages use separate tokenizers with different measured fertility (Phase 1: 1.6522 Hindi vs 1.8338 Nepali tokens/word) — a token is not the same amount of text in both languages, so comparing raw PPL across languages is misleading. Bits-per-byte normalizes by UTF-8 bytes of the underlying text instead, which is tokenizer- and language-agnostic.

## Discussing the H vs L gap

Model L (Nepali) has higher test perplexity than Model H (32.07 vs 16.12, +98.9%). Plausible contributors: fewer training tokens (471.6M L vs 474.8M H, ~0.7% less — the corpus actually tokenized for this run, not the Phase 1 handoff's claimed final-corpus figures; see `report/phase2_resource_comparison.md`), higher tokenizer fertility (1.8338 L vs 1.6522 H tokens/word — the same text costs more, harder-to-predict tokens in Nepali), and a lower manual-token share (20.83% L vs 21.29% H, both Phase 1 measurements). On bits-per-byte, which removes the tokenizer effect, the gap persists in the same direction (0.5158 vs 0.3088 bits/byte) — Model L remains behind on the byte-normalized metric too, suggesting the corpus-size gap dominates over the tokenizer-fertility effect. Byte-fallback rate (0.73% H vs 0.28% L, from `report/phase1_tokenizer_report.md`) is small for both and an unlikely major driver. With only two languages and one run each, this is a plausible attribution, not a controlled ablation.
