# Phase 2 — Resource-Level Comparison (Model H vs Model L)

| | |
|---|---|
| Generated (UTC) | 2026-09-08 16:45:21 |
| Git commit | `9ca6c1de4c71bedc835ee996a6960d672e850a4d` |
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

**Not an apples-to-apples comparison on training progress:** Model H was evaluated at step 6,500 and Model L at step 4,999 — Model H has seen substantially more training, not just a different corpus (474,751,658 Hindi vs 471,622,290 Nepali train tokens in the corpus itself, only 0.7% apart). This happened because Hindi got a second, longer GPU run (Kaggle, after the shared CPU-VM run both models completed at step 4,999) while Nepali's equivalent retry did not produce a usable checkpoint in time — see `report/phase2_checkpoint_links.json` for the full account. Read literally, Model L has higher test perplexity than Model H (32.07 vs 16.12) and higher BPB (0.5158 vs 0.3088), but with this much of a step gap between them that difference is not a clean signal about the languages or corpora — it is dominated by how much more Model H has been trained. The corpus-level comparison (fertility, byte-fallback rate, manual token share, and the near-equal raw token counts above) remains valid regardless, since those are properties of the data and tokenizer, not the training run.
