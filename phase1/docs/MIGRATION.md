# Migrating your existing repo into this structure

Your current layout nests everything under `project/`, which puts the language
directories one level below the repo root and `report/` two levels down. The
brief's layout example places language directories and `report/` at the **repo
root**, and says "Keep every figure and table that you want graded inside
`report/`".

## What maps where

| Yours | Here | Note |
|---|---|---|
| `project/hindi/` | `hindi/` | flattened |
| `project/nepali/` | `nepali/` | flattened |
| `project/common/preprocessing/` | `pipeline/` | split into `collect/`, `process/`, `tokenizer/` |
| `project/report/` | `report/` | flattened |
| `project/scripts/download_drive_datasets.py` | keep — Drive fetch is separate from corpus collection |
| `download_hi_ne_corpora_gcp.py` | `pipeline/collect/download_public.py` | rewritten, see below |

## Your files that this replaces

| Your file | Replaced by | Why |
|---|---|---|
| `common/preprocessing/train_tokenizer.py` | `pipeline/tokenizer/train_tokenizer.py` | v1's vocabulary selection was degenerate — see the audit |
| `common/preprocessing/scrape_ingest.py` | `pipeline/collect/scrape_collect.py` | sequential at 0.66 pages/s; cannot reach the manual target |
| `common/preprocessing/ocr_ingest.py` | `pipeline/collect/ocr_collect.py` | now emits manifest records and parallelises |
| `common/preprocessing/download_public.py` + your GCP script | `pipeline/collect/download_public.py` | streaming, manifest-aware, mC4 bug fixed |
| `dedup_exact.py`, `dedup_near.py`, `dedup_paragraph.py`, `normalize.py`, `quality_filters.py`, `lang_id_filter.py`, `boilerplate_strip.py`, `split_data.py` | `pipeline/process/build_corpus.py` | one pass in the correct order — dedup before split |

## Your files worth keeping

- `manifest_utils.py` — richer than `pipeline/manifest.py` in places. If you keep
  it, make sure it writes `provenance_class` as `"manual"` or `"downloaded"`,
  because every downstream stage keys on that exact field.
- `decontaminate.py` — train/test contamination checking. Not in this pipeline;
  keep it and run it after `build`.
- `llm_quality_repair.py`, `semantic_safety_filter.py` — keep if they work.
- `compute_stats.py`, `generate_report.py`, `plots.py` — this pipeline writes its
  own stats JSON and a Markdown report, but your plotting code is what produces
  the figures. **Every plot needs a title, both axis labels and a legend** or it
  can score zero.
- `audit_phase1.py` — keep and re-point at the new paths.

## Migration commands

```bash
cd your-repo
git checkout phase-1        # or phase-2 if phase-1 is already graded

# flatten
git mv project/hindi  hindi
git mv project/nepali nepali
git mv project/report report
mkdir -p pipeline/collect pipeline/process pipeline/tokenizer pipeline/quality
git mv project/common/preprocessing/manifest_utils.py pipeline/
git mv project/common/preprocessing/decontaminate.py  pipeline/process/
git mv project/common/stats                            pipeline/stats
git mv project/common/audit_phase1.py                  pipeline/

# drop the empty shell
git rm -r --cached project 2>/dev/null; rmdir project/common project 2>/dev/null

# then copy this toolkit's pipeline/, run_phase1.py, README.md,
# requirements.txt and the two configs/ over the top
```

## After migrating — the paths that changed

- `--repo-root` is now the repo root, not `project/`.
- Splits: `<lang>/data/splits/{train,val,test}.jsonl` (unchanged shape).
- Tokenizer artifacts: `<lang>/tokenizer/vocab/<lang>_tokenizer.{model,vocab,json}`
  — note **`_tokenizer`**, not `_unigram`. Your existing files are
  `hindi_unigram.model`; the new trainer writes `hindi_tokenizer.model`. Re-run
  the trainer rather than renaming, since the selection rule changed.

## The one thing to check first

Your raw documents must carry `provenance_class`. Without it the ≥20% manual
requirement is unverifiable and `count_corpus_tokens.py` will refuse to report a
fraction. If your existing raw JSONL lacks the field, the fastest fix is a
one-off backfill keyed on which collector produced each file:

```python
import json, pathlib
MAP = {"manual_scrape": "manual", "manual_ocr": "manual",
       "downloaded_": "downloaded"}
for p in pathlib.Path("hindi/data/raw").glob("*.jsonl"):
    cls = next((v for k, v in MAP.items() if p.name.startswith(k)), None)
    if not cls:
        print("SKIP — cannot infer provenance:", p.name); continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("provenance_class", cls)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{p.name}: {len(rows)} rows -> {cls}")
```

Infer it from the *collector*, never from the source string — only the collector
knows whether `ekantipur.com` text was scraped by you or pulled out of Sangraha.
