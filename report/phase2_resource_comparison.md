# Phase 2 — Resource-Level Comparison (Model H vs Model L)

| | |
|---|---|
| Generated (UTC) | 2026-09-04 06:39:46 |
| Git commit | `d68cff126061309a911334f15371e3df0d4197c3` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

| | Model H (Hindi) | Model L (Nepali) |
|---|--:|--:|
| Training tokens (Phase 1, measured) | 559,913,515 | 501,226,336 |
| Manual token share | 21.29% | 20.83% |
| Tokenizer vocab | 4,000 | 4,000 |
| Fertility (tokens/word) | 1.6522 | 1.8338 |
| Byte-fallback rate | 0.7316% | 0.2757% |
| Parameters | 24,377,856 | 24,377,856 |

| Test perplexity | 311.76 | 510.64 |
| Test BPB | 1.0419 | 0.9275 |

## Write-up

Same architecture, same parameter budget, same vocabulary size, independently trained on independently collected corpora that share no documents (Phase 1 verification check C1-C3). Any gap in the metrics above is therefore attributable to the corpus (size, manual/downloaded mix, quality) and the language/script itself (Phase 1's own finding: manual text tokenizes *better* than downloaded in Nepali but *worse* in Hindi — an observation, not a generalizable finding, with only two languages) — not to any architectural difference between the two models.

Both models were evaluated at the same checkpoint step (199), so the comparison above is apples-to-apples in training progress, not just in architecture. Model L (Nepali) has higher token-level perplexity than Model H (510.6 vs 311.8), consistent with training on ~11% fewer tokens and a higher-fertility tokenizer. On bits-per-byte, which controls for the tokenizer difference, the gap narrows or reverses (0.9275 vs 1.0419 bits/byte) — Model L is actually *more* byte-efficient despite the higher PPL, because its higher-fertility tokenizer spreads the same text over more (individually easier-to-predict) tokens, each carrying less information. Note this reflects a 199-step pilot for both models (0.5% of the planned 40,000-step budget), not a converged comparison — see `docs/PHASE2_GCP_TRAINING.md` for the full-run plan.
