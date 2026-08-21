# Phase 1 — Data Provenance Report

## 1. Objective

This report traces every source used in the Hindi and Nepali corpora back to its origin, its collection method, and its storage location, so that any final training document can be attributed to a specific upstream source.

## 2. Language Layout

### Hindi

`gs://lma-01-hi-ne-corpus/raw/hi/`

Expected source organization:

- `cc100/`
- `opus/`
- `wikipedia/`
- `oscar/`
- `indiccorp_v2/`
- `mc4/`
- `sangraha/verified/`
- `sangraha/unverified/`

### Nepali

`gs://lma-01-hi-ne-corpus/raw/ne/`

Expected source organization:

- `wikipedia/`
- `sangraha/verified/`
- `sangraha/unverified/`

## 3. Source → Storage Mapping

| Source | Language | GCS Location |
|---|---|---|
| Wikipedia | Hindi | `raw/hi/wikipedia/` |
| Sangraha Verified | Hindi | `raw/hi/sangraha/verified/` |
| Sangraha Unverified | Hindi | `raw/hi/sangraha/unverified/` |
| OPUS | Hindi | `raw/hi/opus/` |
| CC100 | Hindi | `raw/hi/cc100/` |
| OSCAR | Hindi | `raw/hi/oscar/` |
| IndicCorp v2 | Hindi | `raw/hi/indiccorp_v2/` |
| mC4 | Hindi | `raw/hi/mc4/` |
| Wikipedia | Nepali | `raw/ne/wikipedia/` |
| Sangraha Verified | Nepali | `raw/ne/sangraha/verified/` |
| Sangraha Unverified | Nepali | `raw/ne/sangraha/unverified/` |

## 4. Example Observed GCS Objects

Nepali collection includes:

`gs://lma-01-hi-ne-corpus/raw/ne/wikipedia/data.jsonl`
`gs://lma-01-hi-ne-corpus/raw/ne/sangraha/verified/data.jsonl`
`gs://lma-01-hi-ne-corpus/raw/ne/sangraha/unverified/data.jsonl`

Observed sizes during collection included approximately:

- Wikipedia: 0.21 GB
- Sangraha verified: 86.25 GB
- Sangraha unverified: 9.71 GB

## 5. Provenance Policy

Every final dataset should be traceable to:

source → raw object → collection pipeline → cleaned data → deduplicated data → tokenizer corpus → final split.

## 6. Excluded Artifacts

Large archives that are not intended to be directly consumed by the GCS-ingest stage are treated separately from the normalized JSONL corpus.

For example:

`OpenSubtitles_en-hi.zip`

and aggregate archives such as:

`hi_ne_corpus.zip`

are storage artifacts rather than independent normalized training sources.

## 6a. External Access Copies

| Artifact | Link |
|---|---|
| Dataset (corpus) | https://drive.google.com/drive/folders/1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6?usp=drive_link |
| Trained tokenizer model | https://drive.google.com/drive/folders/1yKSysO5UJdSmYw_H7XrhEY_jx0cfKQIY?usp=sharing |

## 7. Version Information

The final provenance report will record:

- Git commit
- pipeline version
- configuration version
- tokenizer version
- collection timestamp
- final GCS object paths

These values should be updated immediately before final Phase 1 sign-off.
