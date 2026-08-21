# Phase 1 ??? Data Collection and Tokenizer Construction

Two fully independent monolingual pipelines: **Hindi** (Model H, higher-resource)
and **Nepali** (Model L, from the allowed lower-resource list). Separate corpora,
separate tokenizers, separate vocabularies. No document, vocabulary or artifact
is shared between them.

## Google Drive links

> **Fill these in before submitting.** Large artifacts must not be committed to
> git, and graders must not have to request access.

| Artifact | Link |
|---|---|
| Hindi raw corpus | _paste shareable link_ |
| Hindi splits (train/val/test) | _paste shareable link_ |
| Nepali raw corpus | _paste shareable link_ |
| Nepali splits (train/val/test) | _paste shareable link_ |
| Tokenizer models + vocabularies | _paste shareable link_ |

Tokenizer `.model`/`.vocab` files are small enough to commit and are in the repo
under `<lang>/tokenizer/vocab/`.

---

## Reproduction

```bash
pip install -r requirements.txt

# End to end, one language at a time
python run_phase1.py --lang hindi  --stage all
python run_phase1.py --lang nepali --stage all
```

Stages run individually too, which is what you want when one fails:

```bash
python run_phase1.py --lang hindi --stage discover,scrape
python run_phase1.py --lang hindi --stage build
python run_phase1.py --lang hindi --stage tokenizer --compare-algorithms
python run_phase1.py --lang hindi --stage count,report
```

| Stage | What it does | Output |
|---|---|---|
| `ingest-existing` | fold corpora you already downloaded into the pipeline | `<lang>/data/raw/downloaded_imported.jsonl` |
| `discover` | seed domains ??? article URLs via sitemaps | `<lang>/data/raw/article_urls.txt` |
| `plan` | can you reach ???20% manual? run **before** a long collection | printed budget |
| `scrape` | article URLs ??? **manual** documents | `<lang>/data/raw/manual_scrape.jsonl` |
| `ocr` | PDFs/images ??? **manual** documents (optional) | `<lang>/data/raw/manual_ocr.jsonl` |
| `download` | HuggingFace corpora ??? **downloaded** documents | `<lang>/data/raw/downloaded_*.jsonl` |
| `build` | normalise, filter, dedup, stratified split | `<lang>/data/splits/*.jsonl` |
| `tokenizer` | train from scratch + vocabulary sweep + statistics | `<lang>/tokenizer/` |
| `count` | authoritative token count + manual fraction | `<lang>/data/stats/token_accounting.json` |
| `report` | per-language dataset + tokenizer report | `report/<lang>_dataset_report.md` |

### Already downloaded corpora? Fold them in ??? as the ???80%

If you already ran a bulk download, don't redo it:

```bash
python -m pipeline.collect.ingest_existing --lang hindi --repo-root . \
    --input-dir /path/to/extracted/zips --dry-run   # inspect first
```

It labels them `downloaded`, because that is what they are. Sangraha,
IndicCorp, OSCAR, Wikipedia and mC4 are public corpora ??? the brief's ???80%.
They do not become manual by being renamed or moved.

### The ???20% manual requirement, as arithmetic

Manual data is **collected, never downloaded** ??? that is what makes it manual.
`scrape_collect.py` and `ocr_collect.py` produce it; `download_public.py`
produces the other ???80%.

With M = manual tokens and D = downloaded tokens, the brief requires
`M / (M + D) ??? 0.20`, which rearranges to **`D ??? 4M`** ??? so your
**total corpus is capped at 5?? your manual tokens**. There are two ways to
satisfy it, and most people only think of the first:

1. collect more manual data ??? slow, bounded by how much Devanagari web exists;
2. **download less** ??? instant, entirely in your control.

Run `python -m pipeline.collect.plan_budget --lang <lang>` before committing to
a long run. A 175M-token corpus that meets the ratio beats a 500M-token one that
fails it, and the brief explicitly permits the shortfall for the lower-resource
language provided you report the exact count and justify it.

### Two inputs only you can supply

1. **`<lang>/configs/seed_domains.txt`** ??? the domains you scrape yourself.
   Throughput is set by the *number of domains*, not by concurrency: each domain
   yields at most ~1.3 pages/s under the 1.5 s politeness delay. 40 domains ??? 53
   pages/s; 10 domains ??? 13 pages/s. If you are short of manual tokens, add
   domains.
2. **PDFs for OCR** (optional) ??? pass with `--ocr-input-dir`.

### Before any real run

Put a real contact address in the User-Agent (`--user-agent`). It is what lets a
publisher email you instead of silently blocking the range.

---

## Layout

```
README.md                     reproduction steps + Drive links
run_phase1.py                 end-to-end orchestrator
requirements.txt
pipeline/                     shared code (artifacts stay per-language)
  manifest.py                 the provenance contract every stage depends on
  collect/
    seed_discovery.py         seed domains -> article URLs
    scrape_collect.py         concurrent, robots-respecting manual scraping
    ocr_collect.py            PDF/image OCR -> manual documents
    download_public.py        HuggingFace corpora -> downloaded documents
  process/
    build_corpus.py           normalise, filter, dedup, split
  tokenizer/
    train_tokenizer.py        from-scratch SentencePiece + vocabulary sweep
    tokenizer_report.py       token-frequency stats, chars/token, examples
    count_corpus_tokens.py    token accounting + manual fraction
  quality/
    yi_quality_score.py       LLM quality scoring + distillation (optional)
hindi/                        Model H ??? self-contained
  configs/                    data_config, tokenizer_config, seed_domains
  data/{raw,interim,splits,stats,quality}
  tokenizer/{vocab,analysis}
nepali/                       Model L ??? same structure, nothing shared
report/                       per-phase reports, figures, tables
tests/make_fixture.py         synthetic data to exercise the pipeline offline
```

---

## Phase 1 deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `pipeline/collect/` |
| 2 | Dataset preprocessing pipelines | `pipeline/process/build_corpus.py` |
| 3 | Per-language dataset statistics reports | `report/<lang>_dataset_report.md`, `<lang>/data/stats/` |
| 4 | Per-language train/validation/test splits | `<lang>/data/splits/` |
| 5 | Tokenizer training code (both languages) | `pipeline/tokenizer/train_tokenizer.py` |
| 6 | Vocabulary files (one per language) | `<lang>/tokenizer/vocab/<lang>_tokenizer.vocab` |
| 7 | Tokenizer model files (one per language) | `<lang>/tokenizer/vocab/<lang>_tokenizer.model` |

## Language selection

- **Model H ??? Hindi.** The largest public-text footprint of any non-English
  Indian language: multiple national dailies with deep sitemap-exposed archives,
  government publication in Hindi by statute, and substantial coverage in public
  Indic corpora (Sangraha, IndicCorp).
- **Model L ??? Nepali.** On the permitted list. The Nepali web is roughly an
  order of magnitude smaller than the Hindi web, which is the resource
  constraint this project is meant to expose.

## Independence

Enforced structurally, not by convention:

- `pipeline/manifest.py` refuses any document whose `language` is not one of the
  two, and every collector writes into `<lang>/data/` only.
- `doc_id` is a content hash, so `find_cross_language_collisions()` detects any
  document that ended up in both corpora.
- The tokenizer trains on `<lang>/data/splits/train.jsonl` alone; vocabularies
  are never merged.

## Testing offline

```bash
python tests/make_fixture.py .
python -m pipeline.process.build_corpus --lang hindi --repo-root .
python run_phase1.py --lang hindi --stage tokenizer,count,report --vocab-sizes 256 384 512
```

The fixture is deliberately small, so it only supports a few hundred vocabulary
pieces. It exists to prove the wiring, not to produce a usable tokenizer.
