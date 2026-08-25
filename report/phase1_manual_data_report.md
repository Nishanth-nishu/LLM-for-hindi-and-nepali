# Phase 1 — Manual Data Report

| | |
|---|---|
| Generated (UTC) | 2026-08-25 09:38:50 |
| Git commit | `987ad30458abfbc6b83154a7717d846e4bc85abc` |
| Branch | `phase-1` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

Manual collection means text this project gathered itself: pages scraped and cleaned, and PDFs whose text was extracted. It is distinguished from downloaded public corpora by the `provenance_class` field, set at ingest and never inferred afterwards.

## Hindi

**Scraping seed domains:** 37 (`hindi/configs/seed_domains.txt`)

**PDF seed pages:** 3 (`hindi/configs/pdf_sources.txt`)

**URLs discovered:** 345,597 across 45 hosts

### Documents retained

| Source file | Documents | Characters |
|---|--:|--:|
| `manual_scrape.jsonl` | 168,387 | 467,813,284 |
| **total** | **168,387** | **467,813,284** |

**Manual tokens (MEASURED, final tokenizer):** 121,590,216 (21.29% of corpus tokens)

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

**Manual tokens (MEASURED, final tokenizer):** 36,017,844 (100.00% of corpus tokens)

Requirement ≥20% — **MET**

### Failures

Scrape and OCR failure counts are printed by their stages (robots-denied, HTTP errors, extraction failures, wrong language, too short, and for OCR `SKIP:no_text_layer`). Paste the run summary here — it is not currently written to a JSON file.
