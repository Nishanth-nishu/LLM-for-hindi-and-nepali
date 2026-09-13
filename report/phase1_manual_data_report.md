# Phase 1 — Manual Data Report

| | |
|---|---|
| Generated (UTC) | 2026-09-13 12:24:37 |
| Git commit | `c2a82c02fa66e6395fd229e237a102fc4cb385f5` |
| Branch | `phase-1-fix` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

Manual collection means text this project gathered itself: pages scraped and cleaned, and PDFs whose text was extracted. It is distinguished from downloaded public corpora by the `provenance_class` field, set at ingest and never inferred afterwards.

> **Note on the two measurements below:** "Documents retained" reflects the raw collection stage (`data/raw/manual_*.jsonl`, not committed — too large for git and not re-downloaded for this refresh); "Manual tokens (MEASURED, final tokenizer)" reflects the final train/val/test corpus after cleaning, deduplication, and splitting. The two numbers describe different pipeline stages and are not expected to divide out evenly against each other — cleaning and dedup between raw collection and the final corpus reduce the retained volume.

## Hindi

**Scraping seed domains:** 37 (`hindi/configs/seed_domains.txt`)

**PDF seed pages:** 3 (`hindi/configs/pdf_sources.txt`)

**URLs discovered:** 345,597 across 45 hosts

### Documents retained

| Source file | Documents | Characters |
|---|--:|--:|
| `manual_scrape.jsonl` | 168,387 | 467,813,284 |
| **total** | **168,387** | **467,813,284** |

**Manual tokens (MEASURED, final tokenizer):** 107,778,237 (22.28% of corpus tokens)

Requirement ≥20% — **MET**

### Failures

Scrape and OCR failure counts are printed by their stages (robots-denied, HTTP errors, extraction failures, wrong language, too short, and for OCR `SKIP:no_text_layer`). Paste the run summary here — it is not currently written to a JSON file.

## Nepali

**Scraping seed domains:** 41 (`nepali/configs/seed_domains.txt`)

**PDF seed pages:** 6 (`nepali/configs/pdf_sources.txt`)

**URLs discovered:** 392,234 across 21 hosts

### Documents retained

| Source file | Documents | Characters |
|---|--:|--:|
| `manual_ocr.jsonl` | 8,409 | 11,093,576 |
| `manual_scrape.jsonl` | 231,303 | 412,909,259 |
| **total** | **239,712** | **424,002,835** |

**Manual tokens (MEASURED, final tokenizer):** 103,916,848 (21.64% of corpus tokens)

Requirement ≥20% — **MET**

### Failures

Scrape and OCR failure counts are printed by their stages (robots-denied, HTTP errors, extraction failures, wrong language, too short, and for OCR `SKIP:no_text_layer`). Paste the run summary here — it is not currently written to a JSON file.
