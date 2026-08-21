# Phase 1 — Data Cleaning Report

## 1. Objective

The cleaning stage converts heterogeneous source material into training-ready text while removing malformed, empty, duplicated, and unsuitable records.

## 2. Cleaning Operations

The Phase 1 pipeline applies the following classes of processing:

1. Empty-document removal
2. Invalid-record removal
3. Text normalization
4. Language filtering
5. Duplicate detection
6. Near-duplicate handling where configured
7. Length filtering
8. Source-specific filtering
9. Final corpus validation

## 3. Empty and Invalid Records

Records are rejected when they contain:

- missing text
- empty text
- invalid JSON
- malformed source records
- unusable OCR output
- text that fails required validation

## 4. Language Filtering

Hindi data is evaluated against Hindi-language requirements.
Nepali data is evaluated against Nepali-language requirements.

This prevents unrelated multilingual material from dominating the final corpus.

## 5. Deduplication

Deduplication is performed before final tokenizer training.

The objective is to avoid:

- exact duplicates
- repeated documents
- repeated scraped pages
- duplicated OCR content
- source overlap

Final before/after counts will be populated from the completed deduplication run.

## 6. Reporting Convention

The final statistics table will contain:

| Metric | Before Cleaning | After Cleaning | Final |
|---|---:|---:|---:|
| Documents | TBD | TBD | TBD |
| Characters | TBD | TBD | TBD |
| Tokens | TBD | TBD | TBD |
| Duplicate records | TBD | Removed | 0/remaining |
| Invalid records | TBD | Removed | 0/remaining |

## 7. Statistical Qualification

No estimated token count is treated as a final measurement.

Final tokens will be obtained by running the production tokenizer over the final cleaned/deduplicated corpus.
