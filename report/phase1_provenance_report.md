# Phase 1 — Data Provenance Report

| | |
|---|---|
| Generated (UTC) | 2026-08-25 09:38:50 |
| Git commit | `987ad30458abfbc6b83154a7717d846e4bc85abc` |
| Branch | `phase-1` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

## Sources consumed

Read from `<lang>/configs/data_config.yaml` — the same file the ingest stage reads, so this list cannot disagree with what ran.

### Hindi

Bucket root: `gs://lma-01-hi-ne-corpus/raw`  ·  language code: `hi`  ·  sampling: `strided` over 200 windows

| Source key | Object path | Priority | Upstream |
|---|---|--:|---|
| `wikipedia` | `wikipedia/data.jsonl` | 1 | `wikimedia/wikipedia` |
| `sangraha_verified` | `sangraha/verified/data.jsonl` | 2 | `ai4bharat/sangraha` |
| `sangraha_unverified` | `sangraha/unverified/data.jsonl` | 3 | `ai4bharat/sangraha` |

Priority is the deduplication tie-break: when two documents collide, the copy from the lower-priority number survives, so edited prose outranks verified crawl, which outranks unverified crawl. Manual documents outrank all downloaded sources.

### Nepali

Bucket root: `gs://lma-01-hi-ne-corpus/raw`  ·  language code: `ne`  ·  sampling: `strided` over 200 windows

| Source key | Object path | Priority | Upstream |
|---|---|--:|---|
| `wikipedia` | `wikipedia/data.jsonl` | 1 | `wikimedia/wikipedia` |
| `sangraha_verified` | `sangraha/verified/data.jsonl` | 2 | `ai4bharat/sangraha` |
| `sangraha_unverified` | `sangraha/unverified/data.jsonl` | 3 | `ai4bharat/sangraha` |

Priority is the deduplication tie-break: when two documents collide, the copy from the lower-priority number survives, so edited prose outranks verified crawl, which outranks unverified crawl. Manual documents outrank all downloaded sources.

## Excluded artifacts

Present in the bucket and deliberately not ingested. Recorded so the choice is visible in the repository rather than implied by an absence — an unexplained gap between what a bucket holds and what a corpus contains is indistinguishable from an oversight.

### Hindi

| Artifact | Why it was excluded |
|---|---|
| `opus/OpenSubtitles_en-hi.zip` | Subtitle text from a translation pair. Much of the Hindi side is translated rather than originally composed, and subtitle register (fragmentary dialogue, no paragraph structure) is a poor match for a general-purpose monolingual LM. Including it would also contradict the no-machine-translated-data constraint this project set for itself. |
| `hi_ne_corpus.zip` | Mixes Hindi and Nepali in one archive. The brief requires two independent corpora with no shared documents; this cannot be ingested without first splitting it by language and re-checking for cross-language collisions. |

### Nepali

| Artifact | Why it was excluded |
|---|---|
| `hi_ne_corpus.zip` | Mixes Hindi and Nepali in one archive. The brief requires two independent corpora with no shared documents; this cannot be ingested without first splitting it by language and re-checking for cross-language collisions. |

## Source → local → processed mapping

Every record carries `provenance_class`, `source`, `collection_method` and `collected_at`; the mapping below is therefore inspectable per document, not only in aggregate (see `pipeline/manifest.py`).

### Hindi

| GCS object | Local raw file | Processed into | provenance_class |
|---|---|---|---|
| `gs://lma-01-hi-ne-corpus/raw/hi/wikipedia/data.jsonl` | `hindi/data/raw/downloaded_wikipedia.jsonl` | `hindi/data/splits/*.jsonl` | `downloaded` |
| `gs://lma-01-hi-ne-corpus/raw/hi/sangraha/verified/data.jsonl` | `hindi/data/raw/downloaded_sangraha_verified.jsonl` | `hindi/data/splits/*.jsonl` | `downloaded` |
| seed_domains.txt → sitemaps → article URLs | `hindi/data/raw/manual_scrape.jsonl` | `hindi/data/splits/*.jsonl` | `manual` |

### Nepali

| GCS object | Local raw file | Processed into | provenance_class |
|---|---|---|---|
| `gs://lma-01-hi-ne-corpus/raw/ne/wikipedia/data.jsonl` | `nepali/data/raw/downloaded_wikipedia.jsonl` | `nepali/data/splits/*.jsonl` | `downloaded` |
| `gs://lma-01-hi-ne-corpus/raw/ne/sangraha/verified/data.jsonl` | `nepali/data/raw/downloaded_sangraha_verified.jsonl` | `nepali/data/splits/*.jsonl` | `downloaded` |
| pdf_sources.txt → pdf_harvest → PDFs | `nepali/data/raw/manual_ocr.jsonl` | `nepali/data/splits/*.jsonl` | `manual` |
| seed_domains.txt → sitemaps → article URLs | `nepali/data/raw/manual_scrape.jsonl` | `nepali/data/splits/*.jsonl` | `manual` |

## Dataset / licence information

| Source | Origin | Licence | Notes |
|---|---|---|---|
| Sangraha (verified, unverified) | `ai4bharat/sangraha` | CC-BY-4.0 (verify at source) | Built from web crawls of Indic publishers |
| Wikipedia | `wikimedia/wikipedia` | CC-BY-SA-3.0 / GFDL | Edited prose |
| Manual scrape | publisher sites in `<lang>/configs/seed_domains.txt` | per-site terms; `robots.txt` honoured incl. `Crawl-delay` | Collected by this project |
| Manual OCR | government publications in `<lang>/configs/pdf_sources.txt` | government publications, generally permissive — verify per source | Text-layer extraction, OCR fallback |

> Licence column records the commonly stated licence for each upstream dataset. Confirm against the source before redistribution; this project consumes the data rather than republishing it.

## Version

- Commit: `987ad30458abfbc6b83154a7717d846e4bc85abc`
- Branch: `phase-1`
- Generated: 2026-08-25 09:38:50 UTC
- Working tree: dirty


```
UU README.md
A  pipeline/stats/dataset_stats.py
M  run_phase1.py
A  tools/make_figures.py
M  tools/make_reports.py
?? hindi/data/stats/dataset_statistics.json
?? nepali/data/stats/dataset_statistics.json
?? report/figures/cleaning_hindi.png
?? report/figures/cleaning_hindi.svg
?? report/figures/corpus_composition.png
?? report/figures/corpus_composition.svg
?? report/figures/doc_length.png
?? report/figures/doc_length.svg
?? report/figures/vocab_tradeoff.png
?? report/figures/vocab_tradeoff.svg
?? report/figures/zipf.png
?? report/figures/zipf.svg
```
