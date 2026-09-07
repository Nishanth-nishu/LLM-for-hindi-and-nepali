# Phase 2 — Language Modeling Evaluation

| | |
|---|---|
| Generated (UTC) | 2026-09-07 06:54:26 |
| Git commit | `2c72a3b797bc6a52b00fcb81d0b357ca59d72870` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## PPL / BPB, side by side

| | Model H (Hindi) | Model L (Nepali) |
|---|--:|--:|
| Checkpoint step | 4999 | 4999 |
| Test cross-entropy (nats) | 3.3740 | 3.4681 |
| Test perplexity | 29.19 | 32.07 |
| Test bits-per-byte | 0.6147 | 0.5158 |
| Val perplexity | 29.85 | 32.51 |



## Why BPB, not just PPL

Perplexity is computed per *token*, and the two languages use separate tokenizers with different measured fertility (Phase 1: 1.6522 Hindi vs 1.8338 Nepali tokens/word) — a token is not the same amount of text in both languages, so comparing raw PPL across languages is misleading. Bits-per-byte normalizes by UTF-8 bytes of the underlying text instead, which is tokenizer- and language-agnostic.

## Discussing the H vs L gap

Model L (Nepali) has higher test perplexity than Model H (32.07 vs 29.19, +9.9%). Plausible contributors, all measured in Phase 1: fewer training tokens (501.2M L vs 559.9M H, ~10.5% less), higher tokenizer fertility (1.8338 L vs 1.6522 H tokens/word — the same text costs more, harder-to-predict tokens in Nepali), and a lower manual-token share (20.83% L vs 21.29% H). On bits-per-byte, which removes the tokenizer effect, the gap reverses (0.5158 vs 0.6147 bits/byte) — Model L is the more byte-efficient model despite its higher token-level PPL, since its higher-fertility tokenizer spreads each byte of text over more, individually easier, token-prediction steps. Byte-fallback rate (0.73% H vs 0.28% L, from `report/phase1_tokenizer_report.md`) is small for both and an unlikely major driver. With only two languages and one run each, this is a plausible attribution, not a controlled ablation.
