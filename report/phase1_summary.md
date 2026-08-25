# Phase 1 — Hindi and Nepali corpora and tokenizers

_Generated 2026-08-25 09:38 UTC from commit `987ad30458ab`._

## Executive summary

Two independent monolingual corpora and two from-scratch SentencePiece tokenizers, one per language. No pretrained tokenizer or model is used anywhere. Every document carries its provenance, so the manual versus downloaded split is measured rather than asserted.

The downloaded side streams from Cloud Storage with **windowed byte-range sampling** rather than reading each blob from the top: at a 400M-token budget against a 173 GB Sangraha file, a prefix read would have covered 2% of the file and delivered whichever source happened to be written first. The manual side comes from scraping (robots-respecting, `Crawl-delay` honoured) and from text extraction over government PDFs.

## Statistics

| | Hindi | Nepali |
|---|--:|--:|
| Corpus tokens (MEASURED) | 571,100,312 | 36,017,844 |
| Training tokens | 559,913,515 | 35,314,141 |
| Manual tokens | 121,590,216 | 36,017,844 |
| Downloaded tokens | 449,510,096 | 0 |
| Vocabulary size | 4,000 | 4,000 |
| Fertility (test) | 1.6522 | 1.8457 |

## Dataset locations

```
<lang>/data/raw/       collected documents, one JSONL per source
<lang>/data/interim/   cleaned pool before the budget trim
<lang>/data/splits/    train.jsonl / val.jsonl / test.jsonl
<lang>/data/stats/     corpus_stats.json, token_accounting.json, ...
<lang>/tokenizer/vocab/    <lang>_tokenizer.model / .vocab / .json
<lang>/tokenizer/analysis/ sweep_results.csv, token_stats.json, examples.md
report/                the eight Phase 1 reports
```

## Reports

| File | Contents |
|---|---|
| `phase1_data_collection_report.md` | Sources, counts, sizes, failures |
| `phase1_provenance_report.md` | GCS paths, source→local→processed mapping, licences |
| `phase1_cleaning_report.md` | Filters, dedup methodology, before/after |
| `phase1_manual_data_report.md` | Scraping and OCR sources, retention |
| `phase1_tokenizer_report.md` | Sweep, selection, fertility, examples |
| `phase1_final_statistics.md` | Final measured counts per language |
| `phase1_reproducibility.md` | Commit, environment, commands |
| `phase1_validation_report.md` | All checks, integrity, round-trip |

## Data availability

The corpora are far too large for Git and are not committed. Only code, configuration, statistics, reports and the tokenizer models live here.

| Artifact | Location |
|---|---|
| Tokenizer models (`.model`, `.vocab`, metadata, sweep evidence) | [Google Drive](https://drive.google.com/drive/folders/1yKSysO5UJdSmYw_H7XrhEY_jx0cfKQIY) |
| Final corpus splits — `clean_data_hindi/`, `clean_data_nepali/` | [Google Drive](https://drive.google.com/drive/folders/1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6) |
| Raw pre-cleaning collection — `raw_data_hindi/`, `raw_data_nepali/` | [Google Drive](https://drive.google.com/drive/folders/1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6) |

The tokenizer models are also committed to this repository under `<lang>/tokenizer/vocab/`, since they are small and are the deliverable a reader needs in order to reproduce any token count in these reports.

## Known limitations

_Edit this section before submitting._

- **Token counts** are only valid for the tokenizer that produced them. Retraining the tokenizer invalidates every token figure.
- **Collection is not reproducible** from code alone; the live web changes. The raw JSONL is the reproducible artifact.
- **OCR** used `--text-layer-only` where noted, so scanned PDFs without an embedded text layer were skipped rather than rasterised. State the count if you used it.
- **The corpus size was reached by iteration, not by prediction.** Characters-per-token is not a constant: it depends on which documents the budget admits, and the trim drops least-curated sources first, so raising the budget changes the mix and therefore the ratio. Three trim passes were needed before the measured count matched the target.
- **The manual-versus-downloaded fertility comparison supports no general claim.** Manual text tokenizes better than downloaded in Nepali and worse in Hindi, at every swept vocabulary size. With two languages this is an observation, not a finding.
- **Hindi was ratio-bound, Nepali target-bound.** Hindi's URL frontier was exhausted (345,585 of 345,597 URLs attempted), so its corpus size is capped by its manual yield rather than by the token target.
