# Phase 1 — Data Collection Report

| | |
|---|---|
| Generated (UTC) | 2026-08-25 09:38:50 |
| Git commit | `987ad30458abfbc6b83154a7717d846e4bc85abc` |
| Branch | `phase-1` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

## Hindi

### Downloaded sources (streamed from GCS)

Bucket root: `gs://lma-01-hi-ne-corpus/raw`  
Sampling: windowed byte-range sampling (see `gcs.sampling` in `hindi/configs/data_config.yaml`)

| Source | HF origin | Documents | Characters | Sampling | Windows done |
|---|---|--:|--:|---|--:|
| `wikipedia` | wikimedia/wikipedia | 122,802 | 242,437,606 | sequential | 1 |
| `sangraha_verified` | ai4bharat/sangraha | 953,905 | 2,077,564,879 | strided | 200 |

- Characters ingested: **2,320,002,485**
- Token estimate at ingest: **580,000,621** (ESTIMATE, 4.0 chars/token proxy)
- Stopped on budget: True

Rejected at ingest:

| Reason | Documents |
|---|--:|
| too_short | 38,363 |
| not_devanagari | 16,540 |

Sources present in the bucket and deliberately excluded:

- `opus/OpenSubtitles_en-hi` — translated subtitle text; wrong register for a monolingual LM
- `hi_ne_corpus.zip` — mixes both languages; the brief forbids sharing documents across corpora

### Manual sources

| File | Documents | Characters | Size (MB) |
|---|--:|--:|--:|
| `manual_scrape.jsonl` | 168,387 | 467,813,284 | 1,253.6 |

### Raw data size

- `hindi/data/raw/` — **7.59 GB**
- `hindi/data/splits/` — **4.83 GB**

### Token counts

| Metric | Value | Status |
|---|--:|---|
| Corpus tokens (all splits) | 571,100,312 | **MEASURED** |
| Training tokens | 559,913,515 | **MEASURED** |
| Target | 500,000,000 | — |
| % of target | 114.22% | **MEASURED** |

Measured with `hindi_tokenizer.model` (vocab 4,000). Counts are valid only for this tokenizer.

## Nepali

### Downloaded sources (streamed from GCS)

Bucket root: `gs://lma-01-hi-ne-corpus/raw`  
Sampling: windowed byte-range sampling (see `gcs.sampling` in `nepali/configs/data_config.yaml`)

| Source | HF origin | Documents | Characters | Sampling | Windows done |
|---|---|--:|--:|---|--:|
| `wikipedia` | wikimedia/wikipedia | 23,787 | 37,090,392 | sequential | 1 |
| `sangraha_verified` | ai4bharat/sangraha | 1,177,648 | 2,282,910,101 | strided | 200 |

- Characters ingested: **2,320,000,493**
- Token estimate at ingest: **580,000,123** (ESTIMATE, 4.0 chars/token proxy)
- Stopped on budget: True

Rejected at ingest:

| Reason | Documents |
|---|--:|
| too_short | 9,974 |
| not_devanagari | 1,428 |

Sources present in the bucket and deliberately excluded:

- `opus/OpenSubtitles_en-hi` — translated subtitle text; wrong register for a monolingual LM
- `hi_ne_corpus.zip` — mixes both languages; the brief forbids sharing documents across corpora

### Manual sources

| File | Documents | Characters | Size (MB) |
|---|--:|--:|--:|
| `manual_ocr.jsonl` | 8,409 | 11,093,576 | 32.9 |
| `manual_scrape.jsonl` | 231,303 | 412,909,259 | 1,189.1 |

### Raw data size

- `nepali/data/raw/` — **7.82 GB**
- `nepali/data/splits/` — **0.38 GB**

### Token counts

| Metric | Value | Status |
|---|--:|---|
| Corpus tokens (all splits) | 36,017,844 | **MEASURED** |
| Training tokens | 35,314,141 | **MEASURED** |
| Target | 500,000,000 | — |
| % of target | 7.2% | **MEASURED** |

Measured with `nepali_tokenizer.model` (vocab 4,000). Counts are valid only for this tokenizer.
