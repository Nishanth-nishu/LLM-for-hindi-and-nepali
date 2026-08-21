# Phase 1 — Data Collection Report

**Status:** Collection target reached — provisional final statistics pending tokenizer encoding

## 1. Corpus Target

| Component | Target |
|---|---:|
| Downloaded/public corpora | ~400M tokens |
| Manual/OCR/web data | ~100M tokens |
| **Total Phase 1 target** | **~500M tokens** |

Note: The 400M + 100M figures represent the Phase 1 corpus acquisition targets. Exact final token counts will be computed after training the production SentencePiece tokenizer and encoding the final deduplicated corpus. Therefore, these figures should be treated as provisional corpus targets rather than tokenizer-measured token counts.

## 2. Public Corpus Sources

- Sangraha verified
- Sangraha unverified

The GCS-ingest pipeline selects the appropriate sources according to the Phase 1 configuration.

## 3. Manual Data

The manual component consists of manually collected/web-derived and OCR-derived material.

The manual collection target is approximately:

**100M tokens per the Phase 1 corpus plan.**

The final tokenizer-based token count will be measured after the production tokenizer is trained.

## 4. Corpus Allocation

| Language | Downloaded Target | Manual Target | Total Target |
|---|---:|---:|---:|
| Hindi | ~400M | ~100M | ~500M |
| Nepali | ~400M | ~100M | ~500M |

**Important:** These targets describe the corpus acquisition plan. Final measured token counts will be reported after tokenizer training.

## 5. GCS Storage

Primary bucket:

`gs://lma-01-hi-ne-corpus/raw/`

Language-specific data is organized under:

`gs://lma-01-hi-ne-corpus/raw/hi/`
`gs://lma-01-hi-ne-corpus/raw/ne/`

**Dataset (Drive mirror):** https://drive.google.com/drive/folders/1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6?usp=drive_link

## 6. Collection Statistics

The collection pipeline records:

- documents processed
- documents retained
- characters collected
- source size
- download failures
- parsing failures
- filtering failures
- processing throughput

These statistics are preserved for reproducibility.

## 7. Current Phase 1 Status

| Requirement | Status |
|---|---|
| Hindi public corpus collection | Complete/provisionally complete |
| Nepali public corpus collection | Complete/provisionally complete |
| Manual collection | Collection target reached/provisionally reached |
| GCS organization | Complete |
| Deduplication | Pipeline stage |
| Tokenizer training | Pending/finalization |
| Exact token counting | Pending production tokenizer |
| Final train/validation/test split | Pending finalization |

## 8. Important Statistical Qualification

The reported 400M + 100M figures should not be interpreted as exact tokenizer measurements.

The final report will replace the provisional values with:

`final token count = number of tokens produced by the production tokenizer over the final corpus`

This prevents token estimates based solely on character/token ratios from being presented as measured values.
