# Phase 1 — Manual Data Collection Report

## 1. Objective

The manual-data component supplements large public corpora with manually collected and OCR-derived Hindi and Nepali material.

The planned contribution is approximately:

**100M tokens per language corpus allocation.**

## 2. Collection Types

Manual collection includes:

- web-page extraction
- manually selected sources
- OCR-derived text
- source-specific text extraction
- quality filtering

## 3. Collection Progress

The manual pipeline processes large URL collections and reports:

- URLs processed
- documents retained
- characters collected
- pages/second
- elapsed time
- failures

Example observed progress:

```text
59,198 / 345,597 URLs
35,893 documents retained
102.1M characters
~13 pages/sec
~69 minutes
```

This represents collection progress and should not be interpreted as 102.1M tokens.

## 4. Target

| Language | Manual Target |
|---|---:|
| Hindi | ~100M tokens |
| Nepali | ~100M tokens |

The final tokenizer-based counts will replace the provisional target values.

## 5. OCR Quality

OCR-derived documents are subject to:

- empty-text filtering
- malformed-text filtering
- language filtering
- duplicate detection
- text-quality checks

## 6. Final Manual Statistics

The final report will record:

- URLs attempted
- URLs successfully processed
- pages retained
- pages rejected
- OCR documents
- scraped documents
- total characters
- final tokenizer tokens
- failure count
- duplicate count

## 7. Current Status

The manual corpus is considered provisionally sufficient for the Phase 1 acquisition target.

Final token counts remain dependent on the production tokenizer.
