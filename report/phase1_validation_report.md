# Phase 1 — Validation Report

| | |
|---|---|
| Generated (UTC) | 2026-08-25 09:38:50 |
| Git commit | `987ad30458abfbc6b83154a7717d846e4bc85abc` |
| Branch | `phase-1` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

## Automated checks

Produced by `python -m pipeline.process.verify_corpora --repo-root .`

| Check | Result | Detail |
|---|---|---|
| C2 hindi: no doc_id in two splits | ✅ PASS | — |
| C3 hindi: no identical text across splits | ✅ PASS | — |
| C4 hindi: every document has a provenance class | ✅ PASS | — |
| C5 hindi: manual token fraction >= 20% | ✅ PASS | 21.29% of 571,100,312 tokens (121,590,216 manual) |
| C6 hindi: tokenizer trained on train split only | ✅ PASS | — |
| C2 nepali: no doc_id in two splits | ✅ PASS | — |
| C3 nepali: no identical text across splits | ✅ PASS | — |
| C4 nepali: every document has a provenance class | ✅ PASS | — |
| C5 nepali: manual token fraction >= 20% | ✅ PASS | 100.00% of 36,017,844 tokens (36,017,844 manual) |
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

### Hindi

| File | Lines | Parse failures |
|---|--:|--:|
| `train.jsonl` | 792,123 | 0 |
| `val.jsonl` | 8,065 | 0 |
| `test.jsonl` | 8,065 | 0 |

### Nepali

| File | Lines | Parse failures |
|---|--:|--:|
| `train.jsonl` | 73,969 | 0 |
| `val.jsonl` | 755 | 0 |
| `test.jsonl` | 755 | 0 |

## Tokenizer round-trip

`tokenizer_report.py` encodes and decodes held-out text and records whether the decoded string matches the input exactly. With `byte_fallback=True` this should hold for every input, since no character is unrepresentable. See `<lang>/tokenizer/analysis/examples.md`.

- hindi: `hindi/tokenizer/analysis/examples.md`
- nepali: `nepali/tokenizer/analysis/examples.md`

## Phase 2 consumability

The final corpus is newline-delimited JSON with one document per line and a stable schema (`doc_id`, `text`, `language`, `provenance_class`, `source`, `collection_method`, `collected_at`). The tokenizer is a standard SentencePiece `.model` loadable by `sentencepiece.SentencePieceProcessor`. No custom deserialisation is required.
