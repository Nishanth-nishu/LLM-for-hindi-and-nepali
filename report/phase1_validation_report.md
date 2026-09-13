# Phase 1 — Validation Report

| | |
|---|---|
| Generated (UTC) | 2026-09-13 12:24:37 |
| Git commit | `c2a82c02fa66e6395fd229e237a102fc4cb385f5` |
| Branch | `phase-1-fix` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

## Automated checks

Produced by `python -m pipeline.process.verify_corpora --repo-root .`

| Check | Result | Detail |
|---|---|---|
| C2 hindi: no doc_id in two splits | ✅ PASS | — |
| C3 hindi: no identical text across splits | ✅ PASS | — |
| C4 hindi: every document has a provenance class | ✅ PASS | — |
| C5 hindi: manual token fraction >= 20% | ✅ PASS | 22.28% of 483,658,831 tokens (107,778,237 manual) |
| C6 hindi: tokenizer trained on train split only | ✅ PASS | — |
| C2 nepali: no doc_id in two splits | ✅ PASS | — |
| C3 nepali: no identical text across splits | ✅ PASS | — |
| C4 nepali: every document has a provenance class | ✅ PASS | — |
| C5 nepali: manual token fraction >= 20% | ✅ PASS | 21.64% of 480,275,183 tokens (103,916,848 manual) |
| C6 nepali: tokenizer trained on train split only | ✅ PASS | — |
| C1 hindi/nepali: no document shared across corpora | ✅ PASS | — |
| C7 hindi/nepali: no identical text across corpora | ✅ PASS | — |

**Overall: ALL PASSED**

## What each check defends

| Check | Requirement it enforces |
|---|---|
| C1 | "Do not share documents across corpora" — spans both languages, so no single-language stage can see it |
| C2 | Train/test leakage by `doc_id` |
| C3 | Train/test leakage by identical text under a different id |
| C4 | Every document carries a provenance class, so the manual fraction is measured rather than lower-bounded |
| C5 | The ≥20% manual token requirement |
| C6 | Tokenizer trained on the train split only — held-out text in the vocabulary invalidates every downstream fertility and perplexity number |
| C7 | Identical text across languages, usually a language-ID failure |

## JSONL integrity

> `data/splits/*.jsonl` is gitignored (too large for git), so this checkout can't compute the table below on its own. Measured directly against the real corpus files (line count + a `json.loads` parse check per line) on 2026-09-13, the same run that produced the corrected token counts above.

### Hindi

| File | Lines | Parse failures |
|---|--:|--:|
| `train.jsonl` | 664,070 | 0 |
| `val.jsonl` | 6,759 | 0 |
| `test.jsonl` | 6,759 | 0 |

### Nepali

| File | Lines | Parse failures |
|---|--:|--:|
| `train.jsonl` | 882,729 | 0 |
| `val.jsonl` | 9,004 | 0 |
| `test.jsonl` | 9,004 | 0 |

## Tokenizer round-trip

`tokenizer_report.py` encodes and decodes held-out text and records whether the decoded string matches the input exactly. With `byte_fallback=True` this should hold for every input, since no character is unrepresentable. See `<lang>/tokenizer/analysis/examples.md`.

- hindi: `hindi/tokenizer/analysis/examples.md`
- nepali: `nepali/tokenizer/analysis/examples.md`

## Phase 2 consumability

The final corpus is newline-delimited JSON with one document per line and a stable schema (`doc_id`, `text`, `language`, `provenance_class`, `source`, `collection_method`, `collected_at`). The tokenizer is a standard SentencePiece `.model` loadable by `sentencepiece.SentencePieceProcessor`. No custom deserialisation is required.
