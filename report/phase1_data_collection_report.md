# Phase 1 — Data Collection Report

| | |
|---|---|
| Generated (UTC) | 2026-09-13 12:24:37 |
| Git commit | `c2a82c02fa66e6395fd229e237a102fc4cb385f5` |
| Branch | `phase-1-fix` |
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

- `hindi/data/raw/` — not verified in this refresh (pre-cleaning collection files are not part of this checkout; see `report/phase1_manual_data_report.md` for why)
- `hindi/data/splits/` — **5.0 GB** (measured directly, 2026-09-13, same run as the corrected token counts above)

### Token counts

| Metric | Value | Status |
|---|--:|---|
| Corpus tokens (all splits) | 483,658,831 | **MEASURED** |
| Training tokens | 474,087,588 | **MEASURED** |
| Target | 500,000,000 | — |
| % of target | 96.73% | **MEASURED** |

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

- `nepali/data/raw/` — not verified in this refresh (pre-cleaning collection files are not part of this checkout; see `report/phase1_manual_data_report.md` for why)
- `nepali/data/splits/` — **6.1 GB** (measured directly, 2026-09-13, same run as the corrected token counts above — this is markedly larger than the previously-reported 0.38 GB, consistent with the previous number describing an earlier, incomplete state of this corpus before the full download/scrape volume was assembled; see `report/phase1_final_statistics.md`)

### Token counts

| Metric | Value | Status |
|---|--:|---|
| Corpus tokens (all splits) | 480,275,183 | **MEASURED** |
| Training tokens | 470,739,561 | **MEASURED** |
| Target | 500,000,000 | — |
| % of target | 96.06% | **MEASURED** |

Measured with `nepali_tokenizer.model` (vocab 4,000). Counts are valid only for this tokenizer.
