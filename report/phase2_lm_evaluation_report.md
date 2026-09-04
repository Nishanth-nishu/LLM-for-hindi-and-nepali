# Phase 2 — Language Modeling Evaluation

| | |
|---|---|
| Generated (UTC) | 2026-09-04 18:20:27 |
| Git commit | `593e6e79a54d90f533b73353471ce30cd6fe237a` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## PPL / BPB, side by side

| | Model H (Hindi) | Model L (Nepali) |
|---|--:|--:|
| Checkpoint step | 1750 | 1250 |
| Test cross-entropy (nats) | 3.3814 | 4.2042 |
| Test perplexity | 29.41 | 66.96 |
| Test bits-per-byte | 0.4070 | 0.7023 |
| Val perplexity | 28.88 | 70.81 |



## Why BPB, not just PPL

Perplexity is computed per *token*, and the two languages use separate tokenizers with different measured fertility (Phase 1: 1.6522 Hindi vs 1.8338 Nepali tokens/word) — a token is not the same amount of text in both languages, so comparing raw PPL across languages is misleading. Bits-per-byte normalizes by UTF-8 bytes of the underlying text instead, which is tokenizer- and language-agnostic.

## Discussing the H vs L gap

Once both are measured, this section should relate the gap to: training-token count (559.9M H vs 501.2M L), manual-token share (21.29% H vs 20.83% L), tokenizer fertility (1.65 vs 1.83) and byte-fallback rate (0.73% H vs 0.28% L — see `report/phase1_tokenizer_report.md`), and script/orthographic differences between the two languages. Fill in with the measured numbers once both checkpoints have been evaluated.
