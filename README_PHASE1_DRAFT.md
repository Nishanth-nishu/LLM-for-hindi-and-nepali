# Phase 1 — Hindi & Nepali Corpus Preparation

**Status:** Collection target reached — provisional final statistics pending tokenizer encoding

## Executive Summary

Phase 1 establishes the large-scale Hindi and Nepali corpus required for downstream language-model training.

The collection plan targets approximately:

**400M tokens from downloaded/public corpora + 100M tokens from manually collected/OCR-derived data = approximately 500M tokens.**

The corpus is stored in Google Cloud Storage and processed through a reproducible collection, cleaning, deduplication and tokenizer-training pipeline.

## Languages

- Hindi
- Nepali

## Corpus Allocation

| Language | Downloaded | Manual | Total |
|---|---:|---:|---:|
| Hindi | ~400M | ~100M | ~500M |
| Nepali | ~400M | ~100M | ~500M |

## Sources

### Hindi

- Wikipedia
- Sangraha
- OSCAR
- mC4
- IndicCorp v2
- CC100
- OPUS
- manual/OCR sources

### Nepali

- Wikipedia
- Sangraha
- manual/OCR sources

## Storage

```text
gs://lma-01-hi-ne-corpus/raw/
├── hi/
└── ne/
```

## Tokenizer

SentencePiece Unigram is used with high Devanagari character coverage and byte fallback.

Multiple vocabulary sizes are evaluated before selecting the final vocabulary.

## Downloads

| Artifact | Link |
|---|---|
| Dataset (corpus) | https://drive.google.com/drive/folders/1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6?usp=drive_link |
| Trained tokenizer model | https://drive.google.com/drive/folders/1yKSysO5UJdSmYw_H7XrhEY_jx0cfKQIY?usp=sharing |

## Important Statistical Note

The 500M-token figure represents the Phase 1 corpus acquisition target.

It must not be interpreted as the final exact tokenizer count until the final tokenizer has been trained and applied to the complete cleaned corpus.

The final report will contain measured token counts.

## Phase 1 Deliverables

- data collection report
- provenance report
- cleaning report
- manual data report
- tokenizer report
- final statistics
- reproducibility report
- validation report
- configuration/statistics manifests
