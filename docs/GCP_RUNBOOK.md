# GCP runbook — from `git clone` to a trained tokenizer

Everything below assumes project `lma-01`, bucket `gs://lma-01-hi-ne-corpus`,
and the target 100M manual + 400M downloaded tokens per language.

---

## 1. What to provision, and why

### The machine

| | Value | Why this and not something else |
|---|---|---|
| Machine type | **`n2-highmem-8`** (8 vCPU, 64 GB) | Memory, not cores, is the binding constraint — see below |
| Boot/data disk | **500 GB `pd-balanced`** | See the disk note |
| Region | **`us-central1`** | Same region as the bucket. Cross-region reads cost money and add latency to a 30 GB stream |
| Idle shutdown | **`0s`** (disabled) or 240 min | Immutable after creation. A 180-minute idle timer kills a tokenizer training run that prints nothing for hours |
| Python | 3.12 | What the pipeline is tested on |
| External IP | **Enabled** | The scrape stage needs outbound HTTP |
| Personal credentials (EUC) | **Enabled** | So the notebook reads the bucket as you |

**Why 64 GB and not the 16 GB on `e2-standard-4`.** SentencePiece Unigram
training is the memory peak of the whole pipeline, and it is not close. It
loads the sampled sentences into memory, builds a suffix array over them, and
runs EM over candidate pieces. The working set is roughly an order of magnitude
larger than the sampled text. At `input_sentence_size: 5000000` on Devanagari
that is comfortably 30–45 GB. On a 16 GB machine it does not run slowly, it
gets OOM-killed — usually forty minutes in, with no output.

If you would rather stay on a small machine, the alternative is to shrink the
sample rather than the vocabulary:

```bash
python run_phase1.py --lang hindi --stage tokenizer --max-corpus-lines 2000000
```

That samples 2M lines uniformly across sources (it does *not* take the first
2M — see the docstring in `train_tokenizer.py` for why that distinction
matters) and brings the peak down to roughly 12–16 GB. You lose a little
vocabulary quality on rare word forms. It is a real trade, not a free one.

**Why 8 vCPU.** The `build` stage is CPU-bound at a measured **6.84 ms per
document** — normalisation, the quality gates, and MinHash signatures, of which
MinHash is 5.6 ms. All of that is a pure function of the document text, so it
runs in a process pool while only the order-dependent bookkeeping (the
exact-duplicate set, the LSH buckets, the output file) stays sequential.

Measured scaling: 30,000 documents took 217.6 s with one worker and 110.8 s
with two — 1.96x on two cores. At `--workers 7` a 1.6M-document Hindi build
goes from roughly three hours to well under one. Verified byte-identical to
the serial path, because `map` preserves order and the tie-breaks depend on it.

SentencePiece training is threaded too, so the cores are used twice.

**Why 500 GB and not the 1 TB you already have.** You never hold the whole
bucket locally — `gcs-ingest` streams and stops at the budget. The disk holds
the ingested JSONL (~15 GB per language), the cleaned pool, the splits, and the
SentencePiece plain-text input (~8 GB per language). Peak is under 120 GB for
both languages. 1 TB is not wrong, just idle spend. If you keep the existing
1 TB `pd-standard` disk, note that `pd-standard` IOPS scale with provisioned
size, so at 1 TB you get 750 read / 1500 write IOPS, which is adequate. A
smaller `pd-standard` disk would *not* be — if you shrink the disk, switch to
`pd-balanced` at the same time.

### Creating the template

```bash
gcloud config set project lma-01

gcloud workbench instances create nishanth-corpus-500m \
  --location=us-central1-a \
  --machine-type=n2-highmem-8 \
  --data-disk-size=500 \
  --data-disk-type=PD_BALANCED \
  --metadata=idle-timeout-seconds=0
```

If you create it through the Colab Enterprise runtime-template UI instead, the
fields are: machine type `n2-highmem-8`, disk 500 GB balanced, idle shutdown
disabled, external IP on, EUC on.

**The auto-deletion trap.** Colab Enterprise runtimes are deleted 18 hours
after *creation*, not after last use, and that clock cannot be extended. A full
two-language run is longer than that if you include a serious scrape. Plan it
as chunks that each end with a sync to GCS:

```bash
gsutil -m rsync -r hindi/data gs://lma-01-hi-ne-corpus/work/hindi/data
gsutil -m rsync -r nepali/data gs://lma-01-hi-ne-corpus/work/nepali/data
```

and pull it back at the start of the next runtime. Every stage is resumable, so
a chunk boundary costs nothing but the sync.

---

## 2. Setup

```bash
git clone https://github.com/Nishanth-nishu/courps_scraper.git
cd courps_scraper

pip install -q -r requirements.txt

# Auth — EUC usually covers this, but be explicit
gcloud auth application-default login          # skip if EUC is already active
gcloud config set project lma-01

# Confirm the pipeline can see the bucket, before anything long starts
python -m pipeline.collect.gcs_ingest --lang hindi  --dry-run
python -m pipeline.collect.gcs_ingest --lang nepali --dry-run
```

The dry run prints each resolved URI, whether it exists, and its size. If a
path is wrong you find out in ten seconds instead of two hours.

Optional, only for the OCR stage:

```bash
sudo apt-get update && sudo apt-get install -y \
    tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep poppler-utils
pip install pytesseract pdf2image Pillow
```

---

## 3. What gets used, and what does not

Read directly from the bucket:

```
raw/hi/wikipedia/data.jsonl              raw/ne/wikipedia/data.jsonl
raw/hi/sangraha/verified/data.jsonl      raw/ne/sangraha/verified/data.jsonl
raw/hi/sangraha/unverified/data.jsonl    raw/ne/sangraha/unverified/data.jsonl
```

Deliberately **not** used, with the reasons recorded under `excluded_sources:`
in `<lang>/configs/data_config.yaml` so the choice is visible in the repo:

- `raw/hi/opus/OpenSubtitles_en-hi.zip` — the Hindi side of a translation pair.
  Much of it is translated rather than originally composed, which contradicts
  the no-machine-translated-data constraint this project set for itself, and
  subtitle register (fragmentary dialogue, no paragraphs) is a poor match for a
  general-purpose monolingual LM.
- `raw/hi/hi_ne_corpus.zip` — mixes both languages in one archive. The brief
  requires two corpora with no shared documents; this cannot be ingested
  without splitting it by language first and re-checking for cross-language
  collisions.

Sources fill the budget in priority order — Wikipedia, then verified Sangraha,
then unverified Sangraha as the filler. If the first two cover the budget, the
unverified split is never read, and that is the correct outcome.

### Why the reader samples windows instead of reading from the top

Hindi's verified Sangraha is **173 GiB**. A 400M-token budget touches about 2%
of it. The first 2% of a file assembled by concatenating sources is not a
sample of that file — it is whichever source the authors happened to write
first, and a tokenizer trained on it learns one publisher's vocabulary.

Measured on a fixture of 20,000 documents written as ten ordered source blocks,
with a budget covering ~8% of the file:

```
SEQUENTIAL: {0: 1405}
STRIDED   : {0: 135, 1: 137, 2: 123, 3: 124, 4: 120,
             5: 129, 6: 123, 7: 127, 8: 123, 9: 80}
```

Same bytes transferred. One reads a single source; the other reads all ten.

So `gcs_ingest` divides each large blob into `sampling_windows` evenly spaced
byte ranges (200 by default) and takes an equal share of the budget from each.
Byte offsets never land on line boundaries, so the fragment at the start of
each window is discarded — 200 lost records out of millions, the entire cost of
the scheme. Small blobs where the budget would cover most of the file
(Wikipedia) still read sequentially, and `--sampling sequential` forces a
prefix if you ever want one.

Configured under `gcs:` in `<lang>/configs/data_config.yaml`.

---

## 4. The run

### Order, and why

```
gcs-ingest  →  discover  →  plan  →  scrape  →  build  →  tokenizer  →  count  →  report
```

`plan` sits between discovery and scraping on purpose: it tells you whether the
domains you have can produce 100M manual tokens *before* you spend four hours
finding out they cannot.

### Run everything under `tmux`

Not optional. A closed terminal sends SIGHUP and kills the job; a six-hour
ingest dies having transferred tens of gigabytes.

```bash
tmux new -s hindi          # detach with Ctrl-b then d, reattach with tmux a -t hindi
tmux new -s nepali
```

`gcs_ingest` now traps SIGHUP and SIGTERM so it writes its resume state before
dying, which limits the damage — but it still dies. Use `tmux`.

### Hindi

```bash
# 1. Downloaded side — streams from GCS, stops at the budget       ~45-90 min
python run_phase1.py --lang hindi --stage gcs-ingest

# 2. Manual collection prep and feasibility check                  ~20 min
python run_phase1.py --lang hindi --stage discover,plan

# 3. Manual collection — the graded 20%                            hours
python run_phase1.py --lang hindi --stage scrape --scrape-hours 6

# 4. Clean, dedup globally, trim to budget, split                  ~30 min
python run_phase1.py --lang hindi --stage build

# 5. Tokenizer sweep, token count, report                          ~2-4 h
python run_phase1.py --lang hindi --stage tokenizer,count,report
```

### 6. The pre-submission gate — once, after BOTH languages

```bash
python -m pipeline.process.verify_corpora --repo-root .
```

Twelve checks, each tied to something the brief requires. Exit code 0 only if
all pass. Three of them cannot be checked anywhere else in the pipeline:
"do not share documents across corpora" spans both languages, and train/test
leakage spans three files that the split writer produces independently.

```
C1  no doc_id in both languages      C5  manual token fraction >= 20%
C2  no doc_id in two splits          C6  tokenizer trained on train only
C3  no identical text across splits  C7  no identical text across languages
C4  every document has a provenance
```

C3 and C7 are the text-level twins of C2 and C1: a document that survives dedup
under a different id leaks exactly as hard as one that keeps its id. C7 failing
usually means language ID let Hindi documents into the Nepali corpus — the two
share a script, and short documents are genuinely ambiguous.

### Nepali

Same commands with `--lang nepali`. Run it in a second terminal concurrently —
the two languages share no state, and the scrape stage is politeness-bound
rather than CPU-bound, so they do not compete.

### The second pass

`count` prints the measured characters-per-token and, if it differs materially
from the 4.0 proxy the budgets used, the exact command to correct it:

```bash
python run_phase1.py --lang hindi --stage build,tokenizer,count \
    --chars-per-token 3.42
```

Or, more precisely, using the per-class figures from
`hindi/data/stats/token_accounting.json` → `measured_chars_per_token`:

```bash
python -m pipeline.process.build_corpus --lang hindi \
    --manual-chars-per-token 3.51 --downloaded-chars-per-token 3.38
```

This is minutes, not hours: `build` re-trims from the already-cleaned pool and
does not re-read the bucket or re-scrape anything.

---

## 5. Rough wall-clock and cost

Per language, on `n2-highmem-8`:

| Stage | Time | Bound by |
|---|---|---|
| `gcs-ingest` | 45–90 min | Network from GCS; ~15–20 GB read |
| `discover` | 10–20 min | Sitemap fetches, politeness-limited |
| `scrape` | 4–8 h | Politeness: ~1.3 pages/s **per domain**. Scales with domain count, not threads |
| `build` | 30–45 min | CPU, parallel. ~6.8 ms/doc serial; scales near-linearly with `--workers` |
| `tokenizer` | 2–4 h | Six-point sweep; each Unigram fit is 20–40 min |
| `count` | 15–30 min | Encoding the whole corpus once |

Both languages, with the scrape overlapped: roughly 12–16 hours of runtime.
At `n2-highmem-8` on-demand in `us-central1` that is a few dollars of compute.
Same-region GCS reads are free of network egress; you pay Class B operations
only, which is negligible at this object count.

**The scrape is the long pole and the one you control.** Throughput is
`domains × 2 concurrent ÷ 1.5 s` ≈ 1.3 pages/s per domain. Thirty-seven domains
gives ~49 pages/s; ten domains gives ~13, and no amount of concurrency changes
that. If `plan` says you are short, the fix is more entries in
`<lang>/configs/seed_domains.txt`, not more threads.

---

## 6. What "success" looks like

`count` should print, per language:

```
  corpus tokens (all splits) : ~500,000,000
  manual tokens              : ~100,000,000 (20.4x%)
  [OK] target met and manual fraction satisfied.
```

The manual fraction lands slightly *above* 20% by design. The budget trim
measures characters while the requirement grades tokens, and the two are not
exactly proportional across provenance classes — manual text and Sangraha have
slightly different characters-per-token. Trimming to exactly 20.00% of
characters lands either side of 20% of tokens, and half those runs come out at
19.99%, which fails. `manual_fraction_margin: 0.005` in the config makes the
error one-sided. It costs about 2% of corpus size to guarantee the one number
that is actually graded.

If instead you see:

```
  downloaded budget set by: ratio
      the >=20% requirement is binding, not your 400,000,000-token target.
```

then the scrape came up short and the pipeline capped the downloaded side to
protect the ratio, shipping a smaller corpus that meets the requirement rather
than a larger one that fails it. Two ways forward, and the second is usually
right: collect more manual data, or accept the smaller corpus and justify it —
the brief explicitly permits this for the lower-resource language ("collect the
maximum feasible amount, report the exact token count, and justify the
shortfall").

---

## 7. If something breaks

**`google-cloud-storage is not installed`** — `pip install google-cloud-storage`,
or pass `--bucket-root` pointing at a local copy of the tree.

**403 on the bucket** — EUC is off or the account lacks `storage.objectViewer`.
Check with `gsutil ls gs://lma-01-hi-ne-corpus/raw/hi/`.

**`gcs-ingest` interrupted** — just re-run it. It records lines consumed and
windows completed per blob in `<lang>/data/stats/gcs_ingest_state.json` and
resumes from there. Use `--no-resume` to force a full re-read.

**A run produced no output and then vanished** — two separate things, both
fixed but worth recognising. Output that never appears is stdout block-buffering
on a non-TTY (a notebook cell, a pipe, a redirect); every long-running stage now
line-buffers, and `python -u` forces it for anything that doesn't. A run that
vanishes with no `gcs_ingest_summary.json` was killed — almost always SIGHUP
from a closed terminal. The presence or absence of that summary file is the
reliable way to tell a completed run from a killed one; don't judge by the
output files, which look identical either way.

**Tokenizer OOM-killed** — you are on too small a machine. Either move to
`n2-highmem-8` or add `--max-corpus-lines 2000000`, as described in §1.

**`vocab_size is larger than this corpus can support`** — genuine for a small
corpus, and for the lower-resource language it is itself a reportable result:
it is the vocabulary ceiling your data supports. Extend the sweep downward in
`<lang>/configs/tokenizer_config.yaml`.

**The runtime vanished mid-run** — the 18-hour auto-deletion. Recover from the
last GCS sync and continue from the stage that was running; every stage is
resumable.
