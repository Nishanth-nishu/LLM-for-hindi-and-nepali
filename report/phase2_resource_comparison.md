# Phase 2 — Resource-Level Comparison (Model H vs Model L)

| | |
|---|---|
| Generated (UTC) | 2026-09-07 13:48:08 |
| Git commit | `510c558f3df77d0bbf1bd1312dc1127551ded0b7` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

| | Model H (Hindi) | Model L (Nepali) |
|---|--:|--:|
| Training tokens (this run, measured) | 474,751,658 | 471,622,290 |
| Training tokens (Phase 1 handoff, claimed final corpus) | 559,913,515 | 501,226,336 |
| Manual token share | 21.29% | 20.83% |
| Tokenizer vocab | 4,000 | 4,000 |
| Fertility (tokens/word) | 1.6522 | 1.8338 |
| Byte-fallback rate | 0.7316% | 0.2757% |
| Parameters | 24,377,856 | 24,377,856 |

| Test perplexity | 29.19 | 32.07 |
| Test BPB | 0.6147 | 0.5158 |

**Note on the two token-count rows:** the corpus actually tokenized and trained on for Phase 2 (474,751,658 Hindi / 471,622,290 Nepali train tokens) is smaller than the Phase 1 handoff's claimed final-corpus figures (559,913,515 / 501,226,336). The data synced from GCS `work/<lang>/data/splits/` onto the training VM was an earlier pipeline pass, not the literal final rebuild the handoff quotes — both are still real, substantial corpora (~484M Hindi / ~481M Nepali across all splits, roughly 96-97% of the ~500M target each) — just not the exact numbers previously reported here.

## Write-up

Same architecture, same parameter budget, same vocabulary size, independently trained on independently collected corpora that share no documents (Phase 1 verification check C1-C3). Any gap in the metrics above is therefore attributable to the corpus (size, manual/downloaded mix, quality) and the language/script itself (Phase 1's own finding: manual text tokenizes *better* than downloaded in Nepali but *worse* in Hindi — an observation, not a generalizable finding, with only two languages) — not to any architectural difference between the two models.

Both models were evaluated at the same checkpoint step (4999), so the comparison above is apples-to-apples in training progress, not just in architecture. Model L (Nepali) has higher token-level perplexity than Model H (32.1 vs 29.2), consistent with training on 0.7% fewer tokens (471,622,290 vs 474,751,658) and a higher-fertility tokenizer. On bits-per-byte, which controls for the tokenizer difference, the gap narrows or reverses (0.5158 vs 0.6147 bits/byte) — Model L is actually *more* byte-efficient despite the higher PPL, because its higher-fertility tokenizer spreads the same text over more (individually easier-to-predict) tokens, each carrying less information. Note this reflects a completed 5,000-step training run for both models, not a larger from-scratch budget — see `docs/PHASE2_GCP_TRAINING.md` for cost/time estimates of a longer run.
