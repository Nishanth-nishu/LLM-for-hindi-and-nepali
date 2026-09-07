# Phase 2 — Resource-Level Comparison (Model H vs Model L)

| | |
|---|---|
| Generated (UTC) | 2026-09-07 17:40:27 |
| Git commit | `a5902e127c823786f344be37373a6ffc0a09a543` |
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

| Test perplexity | 16.12 | 32.07 |
| Test BPB | 0.3088 | 0.5158 |

**Note on the two token-count rows:** the corpus actually tokenized and trained on for Phase 2 (474,751,658 Hindi / 471,622,290 Nepali train tokens) is smaller than the Phase 1 handoff's claimed final-corpus figures (559,913,515 / 501,226,336). The data synced from GCS `work/<lang>/data/splits/` onto the training VM was an earlier pipeline pass, not the literal final rebuild the handoff quotes — both are still real, substantial corpora (~484M Hindi / ~481M Nepali across all splits, roughly 96-97% of the ~500M target each) — just not the exact numbers previously reported here.

## Write-up

Same architecture, same parameter budget, same vocabulary size, independently trained on independently collected corpora that share no documents (Phase 1 verification check C1-C3). Any gap in the metrics above is therefore attributable to the corpus (size, manual/downloaded mix, quality) and the language/script itself (Phase 1's own finding: manual text tokenizes *better* than downloaded in Nepali but *worse* in Hindi — an observation, not a generalizable finding, with only two languages) — not to any architectural difference between the two models.

Fill in the specific gap size and direction once both models have been evaluated at the same checkpoint step.
