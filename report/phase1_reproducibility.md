# Phase 1 — Reproducibility Report

## 1. Repository

GitHub repository:

`https://github.com/Nishanth-nishu/LLM-for-hindi-and-nepali`

## 2. Repository Structure

Important components include:

```text
pipeline/
├── collect/
├── ...
hindi/
nepali/
requirements.txt
run_phase1.py
tests/
docs/
```

## 3. Main Pipeline

Phase 1 is orchestrated using:

```bash
python run_phase1.py --lang hindi
python run_phase1.py --lang nepali
```

The GCS ingestion stage can be tested using:

```bash
python -m pipeline.collect.gcs_ingest --lang hindi --dry-run
python -m pipeline.collect.gcs_ingest --lang nepali --dry-run
```

## 4. GCS

Primary bucket:

`gs://lma-01-hi-ne-corpus/`

Raw root:

`gs://lma-01-hi-ne-corpus/raw/`

## 4a. External Access Copies

In addition to the GCS bucket, working copies are mirrored to Google Drive for convenience:

| Artifact | Link |
|---|---|
| Dataset (corpus) | https://drive.google.com/drive/folders/1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6?usp=drive_link |
| Trained tokenizer model | https://drive.google.com/drive/folders/1yKSysO5UJdSmYw_H7XrhEY_jx0cfKQIY?usp=sharing |

GCS remains the source of truth; the Drive links are a convenience mirror and should be re-synced whenever the corpus or tokenizer is retrained.

## 5. Environment

The final report should record:

- `python --version`
- `pip freeze`
- `gcloud --version`
- `git rev-parse HEAD`

## 6. Version

Final reproducibility sign-off must record the exact Git commit used for:

- data collection
- cleaning
- tokenizer training
- final counting

## 7. Important Rule

The corpus itself is stored in GCS and should not be committed to Git.

Only code, configuration, reports, manifests and statistics should be committed.
