# Phase 1 — Validation Report

## 1. Objective

The validation stage verifies that the corpus is structurally valid, linguistically appropriate and ready for tokenizer training and Phase 2.

## 2. GCS Validation

The pipeline successfully accesses the configured GCS bucket:

`gs://lma-01-hi-ne-corpus/raw/`

Language-specific source paths are checked before long-running ingestion.

## 3. Data Integrity

The validation process checks:

- JSONL readability
- valid JSON records
- non-empty text
- UTF-8 compatibility
- expected language
- duplicate records
- corrupted files

## 4. Tokenizer Validation

The tokenizer validation checks:

- successful encoding
- successful decoding
- encode/decode round trip
- Devanagari coverage
- byte fallback behavior
- vocabulary integrity

## 5. Language Validation

Hindi and Nepali datasets are validated independently.

Source-level distributions are recorded so that a single large source does not unintentionally dominate the corpus.

## 6. Phase 2 Compatibility

The final corpus must be consumable by the next training stage without requiring source-specific preprocessing.

Required properties:

- deterministic format
- valid UTF-8
- normalized text
- tokenizer compatibility
- reproducible splits
- documented provenance

## 7. Current Status

| Validation | Status |
|---|---|
| GCS connectivity | PASS |
| Source discovery | PASS |
| JSONL validation | Pipeline validation |
| Language validation | Pipeline validation |
| Deduplication | Pipeline stage |
| Tokenizer round-trip | Final tokenizer stage |
| Final token count | Pending |
| Phase 2 compatibility | Final validation stage |

## 8. Final Sign-Off

Phase 1 will receive final sign-off after:

1. manual corpus is frozen
2. cleaning is complete
3. tokenizer vocabulary is selected
4. final tokenizer is trained
5. entire corpus is encoded
6. exact token counts are generated
7. final train/validation/test splits are verified
