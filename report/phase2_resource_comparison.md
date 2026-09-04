# Phase 2 — Resource-Level Comparison (Model H vs Model L)

| | |
|---|---|
| Generated (UTC) | 2026-09-03 23:42:39 |
| Git commit | `c2a82c02fa66e6395fd229e237a102fc4cb385f5` |
| Branch | `phase-1` |

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

Fill in the specific gap size and direction once both models have been trained to a comparable step count and evaluated.
