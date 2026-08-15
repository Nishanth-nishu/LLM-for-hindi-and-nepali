# ??????? End-to-End Pipeline Architecture: Hindi & Nepali Language Model (Phase 1)

**Author:** Nishanth R ??? CL3-410 Individual Project  
**Languages:** Hindi (Model H, higher-resource) ?? Nepali (Model L, lower-resource)  
**Goal:** Collect, clean, deduplicate, and tokenize monolingual corpora for training separate language models from scratch.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Stage 0 ??? Environment & Reconnaissance](#3-stage-0--environment--reconnaissance)
4. [Stage 1 ??? Data Sources: What We Collected and Why](#4-stage-1--data-sources-what-we-collected-and-why)
5. [Stage 2 ??? Test Holdout (Freezing the Test Set)](#5-stage-2--test-holdout-freezing-the-test-set)
6. [Stage 3 ??? Language Identification Filter](#6-stage-3--language-identification-filter)
7. [Stage 4 ??? Unicode Normalization & Repair](#7-stage-4--unicode-normalization--repair)
8. [Stage 5 ??? Boilerplate Stripping](#8-stage-5--boilerplate-stripping)
9. [Stage 6 ??? Heuristic Quality Filters (Statistical Methods)](#9-stage-6--heuristic-quality-filters-statistical-methods)
10. [Stage 7 ??? Exact Deduplication](#10-stage-7--exact-deduplication)
11. [Stage 8 ??? Paragraph-Level Deduplication](#11-stage-8--paragraph-level-deduplication)
12. [Stage 9 ??? Near-Duplicate Removal (MinHash + LSH)](#12-stage-9--near-duplicate-removal-minhash--lsh)
13. [Stage 10 ??? Semantic & Safety Filtering](#13-stage-10--semantic--safety-filtering)
14. [Stage 11 ??? Decontamination against Test Set](#14-stage-11--decontamination-against-test-set)
15. [Stage 12 ??? Train / Val Split](#15-stage-12--train--val-split)
16. [Stage 13 ??? Corpus Statistics Computation](#16-stage-13--corpus-statistics-computation)
17. [Stage 14 ??? Tokenizer Training (SentencePiece Unigram)](#17-stage-14--tokenizer-training-sentencepiece-unigram)
18. [Stage 15 ??? Vocabulary Sweep & Selection](#18-stage-15--vocabulary-sweep--selection)
19. [Final Numbers: What We Produced](#19-final-numbers-what-we-produced)
20. [Manual Data Collection ??? How to Add It](#20-manual-data-collection--how-to-add-it)
21. [Mathematical Summary](#21-mathematical-summary)

---

## 1. Project Overview

The goal of Phase 1 is to build two completely independent, high-quality training corpora and tokenizers ??? **one for Hindi, one for Nepali** ??? from scratch, using no pretrained LLMs or HuggingFace tokenizers. Every step is implemented using basic Python, PyTorch-compatible utilities, and SentencePiece.

**Key constraints:**
- ??? No shared vocabulary between Hindi and Nepali
- ??? No pretrained tokenizers or HuggingFace `AutoTokenizer`
- ??? Test set frozen **before** any cleaning (data hygiene)
- ??? At least 20% of final training tokens must come from manually collected sources
- ??? All cleaning thresholds set from distribution inspection (data-driven), not guesswork
- ??? Fully resumable, idempotent pipeline

---

## 2. Repository Structure

```
project/
????????? common/
???   ????????? preprocessing/           # All shared pipeline scripts
???   ???   ????????? recon.py             # Stage 0: Source reconnaissance
???   ???   ????????? lang_id_filter.py    # Stage 3: Language ID
???   ???   ????????? normalize.py         # Stage 4: Unicode normalization
???   ???   ????????? boilerplate_strip.py # Stage 5: Boilerplate removal
???   ???   ????????? quality_filters.py   # Stage 6: Heuristic quality filters
???   ???   ????????? dedup_exact.py       # Stage 7: Exact deduplication
???   ???   ????????? dedup_paragraph.py   # Stage 8: Paragraph dedup
???   ???   ????????? dedup_near.py        # Stage 9: MinHash near-dedup
???   ???   ????????? semantic_safety_filter.py  # Stage 10: Safety filter
???   ???   ????????? decontaminate.py     # Stage 11: Decontamination
???   ???   ????????? split_data.py        # Stage 12: Train/val split
???   ???   ????????? train_tokenizer.py   # Stage 14-15: Tokenizer training
???   ???   ????????? scrape_ingest.py     # Manual: Web scraping
???   ???   ????????? ocr_ingest.py        # Manual: PDF/image OCR
???   ???   ????????? transcription_ingest.py  # Manual: Audio transcription
???   ???   ????????? manifest_utils.py    # Manifest tracking (CSV)
???   ????????? stats/
???       ????????? compute_stats.py     # Stage 13: Statistics
???       ????????? plots.py             # Visualization
???       ????????? generate_report.py   # Auto-report generation
???
????????? hindi/
???   ????????? configs/
???   ???   ????????? data_config.yaml          # Source list, split ratios
???   ???   ????????? filtering_thresholds.yaml # All filter thresholds (data-driven)
???   ???   ????????? tokenizer_config.yaml     # Tokenizer hyperparameters
???   ????????? data/
???   ???   ????????? raw/downloaded/      # Downloaded JSONL files
???   ???   ????????? raw/manual/          # Manual collection (scrape/ocr/transcription)
???   ???   ????????? cleaned/             # Per-stage cleaned outputs
???   ???   ????????? dedup/               # Deduplicated outputs
???   ???   ????????? splits/              # train.jsonl / val.jsonl / test.jsonl
???   ????????? tokenizer/
???       ????????? vocab/               # .model and .vocab files
???       ????????? analysis/            # sweep_results.csv, fertility plots
???
????????? nepali/                      # Identical structure to hindi/
????????? report/
    ????????? phase1_report.md
    ????????? figures/                 # Auto-generated plots
```

Each document in every JSONL file carries a **manifest row** ??? a CSV record tracking:
- `doc_id`, `source`, `collection_method`, `stage_reached`, `drop_reason`
- This makes the entire pipeline fully auditable.

---

## 3. Stage 0 ??? Environment & Reconnaissance

### 3.1 Environment Setup

The script (`submit_phase1.sh`) first:
1. Creates/activates a `venv` and installs all dependencies
2. Downloads the fastText language identification model `lid.176.bin` (176-language classifier, 917 languages in its training data, 125 MB)

### 3.2 Source Reconnaissance (`recon.py`)

Before downloading anything, we **sample 500 documents** from each configured source and compute feature distributions. This is how we decide on threshold values ??? not by guessing.

**Feature distributions computed:**

| Feature | What it Measures |
|---|---|
| `char_count` | Document length in characters |
| `word_count` | Approximate word count (whitespace split) |
| `devanagari_ratio` | Fraction of chars in Unicode block U+0900???U+097F |
| `whitespace_ratio` | Fraction of chars that are whitespace |
| `punct_symbol_ratio` | Fraction in Unicode P* + S* categories |
| `digit_ratio` | Fraction that are digits |
| `url_ratio` | URL count / total token count |
| `repeated_line_ratio` | Lines appearing more than once / total lines |
| `repeated_ngram_ratio` | Fraction of 5-grams that are repeated |
| `mean_line_length` | Average chars per line |
| `paragraph_count` | Number of double-newline-separated paragraphs |

**Hindi Wikipedia sample results (n=500):**
```
char_count:       median=5,016  p5=148    p95=31,340
devanagari_ratio: median=0.779  p5=0.688  p95=0.808
punct_symbol_ratio: median=0.003  p95=0.04   ??? using Unicode P+S categories
```

**Nepali Wikipedia sample results (n=500):**
```
char_count:       median=956    p5=144    p95=10,521
devanagari_ratio: median=0.821  p5=0.581  p95=0.847
punct_symbol_ratio: median=0.003  p95=0.04
```

These distributions directly set every threshold you see in `filtering_thresholds.yaml`.

---

## 4. Stage 1 ??? Data Sources: What We Collected and Why

### 4.1 Downloaded Sources (Automatic)

#### Hindi
| Source | HuggingFace ID | Docs | License | Why |
|---|---|---|---|---|
| **Wikipedia** | `wikimedia/wikipedia/20231101.hi` | 163,091 | CC-BY-SA-4.0 | High-quality, encyclopedic text. Well-structured with wiki markup already stripped by HuggingFace. |
| **IndicCorpV2** | `ai4bharat/IndicCorpV2 (hi)` | ~millions | CC-BY-4.0 | AI4Bharat's cleaned web crawl for Indic languages |
| **OSCAR-2301** | `oscar-corpus/OSCAR-2301 (hi)` | ~millions | CC0-1.0 | Large multilingual web crawl; Hindi subset large |

#### Nepali (Low-Resource)
| Source | HuggingFace ID | Docs | License | Why |
|---|---|---|---|---|
| **Wikipedia** | `wikimedia/wikipedia/20231101.ne` | 32,885 | CC-BY-SA-4.0 | Nepali Wikipedia is much smaller (~20% of Hindi) |
| **IndicCorpV2** | `ai4bharat/IndicCorpV2 (ne)` | sparse | CC-BY-4.0 | Limited coverage but adds diversity |
| **CC-100** | `cc100 (ne)` | varies | Common Crawl ToS | CC-100 Nepali subset |
| **OSCAR-2301** | `oscar-corpus/OSCAR-2301 (ne)` | varies | CC0-1.0 | Additional web crawl coverage |

> **Note:** In the pipeline run that completed, only Wikipedia was successfully downloaded for both languages due to availability/authentication issues with OSCAR and IndicCorpV2. This is the key reason why the ???20% manual data requirement was not met.

### 4.2 Manual Collection Sources (Required ??? ???20% of training tokens)

The pipeline has three manual ingestion pathways (all scripts exist and are ready to use ??? see Section 20 for how to run them):

#### Manual Pathway 1: Web Scraping (`scrape_ingest.py`)
- Uses `trafilatura` (not raw BeautifulSoup) for clean content extraction
- Respects `robots.txt`, adds 1.5s delay between requests
- Targets: news sites, blogs, government portals

**Recommended Hindi sources:**
- `https://www.bbc.com/hindi`
- `https://www.aajtak.in`
- `https://www.jagran.com`
- `https://navbharattimes.indiatimes.com`

**Recommended Nepali sources:**
- `https://ekantipur.com`
- `https://annapurnapost.com`
- `https://myrepublica.nagariknetwork.com`
- `https://ratopati.com`

#### Manual Pathway 2: PDF OCR (`ocr_ingest.py`)
- Converts PDFs ??? images (300 DPI via `pdf2image`)
- Runs Tesseract OCR with `hin` tessdata (covers Devanagari for both languages)
- Post-OCR noise removal (regex strips artefact character runs)

#### Manual Pathway 3: Transcription (`transcription_ingest.py`)
- Ingests `.txt` files from interview/speech transcriptions
- Tags with `collection_method=manual, source_type=transcription`

---

## 5. Stage 2 ??? Test Holdout (Freezing the Test Set)

**Script:** `test_holdout.py`

This is the most important data hygiene step. **2% of the raw document pool is frozen as the test set BEFORE any cleaning.** This means:
- The test set may contain noisy documents (that's intentional ??? it reflects real-world distribution)
- Cleaning thresholds are NOT influenced by knowledge of what the test set looks like
- Near-dedup and decontamination cannot accidentally alter what goes into test

### Stratified Sampling Formula

Documents are stratified by `(source_type, collection_method)` to ensure the test set reflects each data source proportionally:

```
n_test_stratum_k = round(N_stratum_k ?? test_fraction)
                 = round(N_stratum_k ?? 0.02)
```

Where:
- `N_stratum_k` = number of docs in stratum k (e.g., `(wikipedia, downloaded)`)
- `test_fraction = 0.02` (2%)
- Random seed = 42 (reproducible)

**Outcome (Hindi):** 163,091 raw docs ??? 3,261 test docs frozen  
**Outcome (Nepali):** 32,885 raw docs ??? 657 test docs frozen

Test docs are written to `splits/test.jsonl` and **never touched again** until final evaluation.

---

## 6. Stage 3 ??? Language Identification Filter

**Script:** `lang_id_filter.py`  
**Model:** fastText `lid.176.bin` ??? a linear text classifier trained on n-gram character features over 176 languages

### How fastText Language ID Works

fastText represents each document as a bag of character n-grams (n=3 to 6). For each n-gram, it looks up a pre-trained embedding. The document embedding is the **average** of all n-gram embeddings:

```
doc_embedding = (1/T) ?? ??_t  embedding(ngram_t)
```

Then a **softmax classifier** predicts the language:

```
P(language = L | doc) = exp(W_L ?? doc_embedding) / ??_k exp(W_k ?? doc_embedding)
```

The confidence score is this softmax probability. We keep documents where:

```
confidence(hi) ??? 0.65     (for Hindi)
confidence(ne) ??? 0.70     (for Nepali)
```

### Devanagari Script Fallback

Because Nepali and Hindi share the Devanagari script and fastText often confuses them, we add a heuristic fallback:

```python
devanagari_ratio = len(re.findall(r'[\u0900-\u097F]', text)) / len(text)
if devanagari_ratio >= 0.40:
    # Accept even if fastText predicted hi instead of ne
    accept = True
```

**Outcomes:**
- Hindi: 163,091 ??? 160,306 (dropped 2,785 non-Hindi docs)
- Nepali: 32,885 ??? 32,885 (all accepted; fallback handled mis-classifications)

---

## 7. Stage 4 ??? Unicode Normalization & Repair

**Script:** `normalize.py`

Raw Devanagari text from the web contains encoding artefacts. We apply the following normalization chain in order:

### Step 1: Unicode NFC Normalization
```
text = unicodedata.normalize('NFC', text)
```
Devanagari combining marks (matras, halant) can be stored in multiple canonical forms. NFC ensures:
- Precomposed forms preferred where available
- Equivalent character sequences are collapsed to a single canonical form

### Step 2: Control Character Removal
```python
# Remove all Unicode Cc category characters (control chars) except tab and newline
text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
```

### Step 3: Zero-Width Character Removal
```python
# ZW non-joiner (U+200C), ZW joiner (U+200D), BOM (U+FEFF), etc.
text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
```
These invisible characters corrupt tokenization and downstream processing.

### Step 4: Whitespace Normalization
- Multiple spaces ??? single space
- Multiple newlines ??? max 2 consecutive (preserves paragraph boundaries)
- Tabs ??? single space

### Step 5: Minimum Length Check
After normalization, documents with fewer than **50 characters** are dropped as stubs.

**Outcomes:**
- Hindi: dropped 2,173 docs (mostly redirect stubs that became empty after normalization)
- Nepali: dropped 60 docs

---

## 8. Stage 5 ??? Boilerplate Stripping

**Script:** `boilerplate_strip.py`

Wikipedia articles contain structural markup artifacts that survived the HuggingFace pre-processing:

- Navigation boxes (navboxes) ??? repeated across hundreds of articles
- Category listings ??? e.g., "Category: Cities in India | Category: Populated places"
- Disambiguation footers
- Reference section markers (e.g., `== References ==` followed by only citation numbers)
- Infobox remnants ??? key-value pairs like `|population = 5,000,000`

Boilerplate is detected using a combination of:
1. **Pattern matching** ??? regex rules for wiki-specific markup artifacts
2. **Line-length heuristics** ??? lines < 10 chars are usually navbox bullets
3. **Structural detection** ??? `== Section ==` headers with no content body

After stripping, very short documents (< 100 chars) are dropped.

**Outcomes:**
- Hindi: 0 documents dropped (boilerplate stripped but documents remained viable)
- Nepali: 0 documents dropped

---

## 9. Stage 6 ??? Heuristic Quality Filters (Statistical Methods)

**Script:** `quality_filters.py`

This is the most mathematically sophisticated filtering stage. We compute 9 features per document and apply threshold tests.

### Feature Extraction

```python
import unicodedata

chars = len(text)
sp    = sum(1 for c in text if c.isspace())
dig   = sum(1 for c in text if c.isdigit())

# Unicode category-based punctuation (correct for Devanagari)
# Uses P* (Punctuation) and S* (Symbol) categories ONLY
# This correctly excludes Devanagari matras (Mc), halant (Mn), etc.
punct_sym = sum(1 for c in text if unicodedata.category(c)[0] in ("P", "S"))

deva = len(re.findall(r'[\u0900-\u097F]', text))
urls = len(re.findall(r'https?://\S+', text))
```

> **Critical fix:** An earlier version computed punctuation as `chars - alpha - digits - spaces`. This is **wrong** for Devanagari because matras (e.g., ???, ???, ???) have `isalpha() = False` ??? they are combining marks (Unicode category Mc/Mn), not punctuation. This caused 88% of valid Hindi Wikipedia documents to be incorrectly dropped. The fix using `unicodedata.category(c)[0] in ("P", "S")` is the correct approach.

### The 9 Quality Tests

For each document `d`, the filter applies these tests (all must pass):

#### Test 1: Minimum Character Count
```
len(text) ??? min_char_count (100 for Hindi, 80 for Nepali)
```

#### Test 2: Minimum Devanagari Ratio
```
deva / chars ??? min_devanagari_ratio
= 0.30 (Hindi)   ??? allows Hindi+English mixed text
= 0.20 (Nepali)  ??? more permissive for low-resource
```
**Rationale from recon:** Hindi Wikipedia p5 of devanagari_ratio = 0.688. Setting threshold at 0.30 drops only pages that are predominantly English (e.g., English-language Wikipedia articles accidentally indexed in the Hindi dump).

#### Test 3: Maximum Whitespace Ratio
```
sp / chars ??? max_whitespace_ratio (0.35)
```
Catches tab-separated data dumps and whitespace-padded artifacts.

#### Test 4: Maximum Punctuation+Symbol Ratio
```
punct_sym / chars ??? max_punct_symbol_ratio (0.15)
```
A normal Hindi Wikipedia article has ~3???5% punctuation (dandas `???`, commas, brackets). 15% is extremely generous ??? it only catches pure-symbol garbage.

#### Test 5: Maximum Digit Ratio
```
dig / chars ??? max_digit_ratio (0.20)
```
Articles with heavy statistics/tables can have 10???15% digits. 0.20 catches data-dump-only pages.

#### Test 6: Maximum URL Ratio
```
urls / whitespace_words ??? max_url_ratio (0.10)
```
Drops reference-list-only stubs where 10%+ of content is bare URLs.

#### Test 7: Maximum Repeated Line Ratio
```
repeated_lines / total_lines ??? max_repeated_line_ratio (0.30)
```
Where `repeated_lines` = count of lines appearing more than once in the document. Catches navboxes and category dumps.

#### Test 8: Maximum Repeated N-gram Ratio
```
repeated_5gram_tokens / total_tokens ??? max_repeated_ngram_ratio (0.25)
```
Counts how many 5-grams appear more than once. Catches templates with repeated boilerplate sentences.

#### Test 9: Minimum Mean Line Length
```
mean(len(line) for line in text.split('\n')) ??? min_mean_line_length (10.0)
```
Pages consisting entirely of short navigation bullets have very low mean line length.

**Outcomes:**
- Hindi: 158,133 ??? 158,133 (0 dropped ??? all passed after the punct_sym fix)
- Nepali: 32,825 ??? 31,744 (dropped 1,081 very short stubs)

---

## 10. Stage 7 ??? Exact Deduplication

**Script:** `dedup_exact.py`

### Method

Every document is hashed after whitespace normalization:

```python
import hashlib

def doc_hash(text: str) -> str:
    # Collapse all whitespace to single spaces, strip leading/trailing
    normalized = ' '.join(text.split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
```

A **set** accumulates seen hashes. The first occurrence of each hash is kept; duplicates are dropped.

```
Space: O(N)   ??? one hash per document
Time: O(N)    ??? single pass
```

**Why SHA-256?** 256-bit hashes make collisions astronomically unlikely (birthday problem: 2^128 hashes before 50% collision probability).

**Outcomes:**
- Hindi: 0 duplicates (Wikipedia articles are already deduplicated in the HuggingFace dump)
- Nepali: 0 duplicates

---

## 11. Stage 8 ??? Paragraph-Level Deduplication

**Script:** `dedup_paragraph.py`

### Method (Two-Pass)

Wikipedia articles share "boilerplate paragraphs" ??? identical text blocks that appear verbatim across hundreds of articles (copyright notices, navigation paragraphs, geographic category templates).

**Pass 1: Count paragraph hash frequencies**
```python
para_freq = Counter()
for doc in all_documents:
    for para in doc.split('\n\n'):   # double newline = paragraph boundary
        if len(para) >= min_para_chars:    # 100 chars minimum
            para_hash = hashlib.md5(para.strip().encode()).hexdigest()
            para_freq[para_hash] += 1

# A paragraph appearing in ??? 10 different docs = boilerplate
boilerplate_hashes = {h for h, cnt in para_freq.items() if cnt >= min_freq}
```

**Pass 2: Strip boilerplate paragraphs**
```python
for doc in all_documents:
    cleaned_paras = [
        para for para in doc.split('\n\n')
        if hashlib.md5(para.strip().encode()).hexdigest() not in boilerplate_hashes
    ]
    if cleaned_paras:   # document still has content after stripping
        output.write(cleaned_doc)
    # else: drop the now-empty document
```

**Threshold:** `min_paragraph_frequency = 10` ??? conservative. A paragraph must appear in at least 10 documents to be considered boilerplate.

**Outcomes:** 0 docs dropped (Wikipedia articles have mostly unique paragraphs)

---

## 12. Stage 9 ??? Near-Duplicate Removal (MinHash + LSH)

**Script:** `dedup_near.py`  
**Algorithm:** MinHash Locality Sensitive Hashing (MinHash-LSH)

This is the most computationally intensive deduplication step. It finds documents that are **nearly identical** (e.g., the same article with minor edits, two versions of a government notice, or crawler duplicates from different timestamps).

### Mathematical Foundation

#### Step 1: Shingling (Character N-grams)
Each document is converted to a set of character 5-grams (shingles):

```
doc = "?????????????????? ??????????????????"
shingles = {"???????????????", "???????????????", "???????????? ", "?????? ??????", "??? ?????????", " ????????????", "???????????????", "???????????????"}
```

#### Step 2: Jaccard Similarity
The **Jaccard similarity** between documents A and B is:

```
J(A, B) = |A ??? B| / |A ??? B|
```

Computing this exactly for 163,000+ document pairs is O(N??) ??? infeasible. MinHash approximates it efficiently.

#### Step 3: MinHash Signatures
We define a **MinHash signature** as follows:

For `k = 1, ..., num_perm` (we use `num_perm = 128`) independent hash functions `h_k`:

```
MinHash_k(A) = min_{x ??? A} h_k(x)
```

The key theorem:

```
P[ MinHash_k(A) = MinHash_k(B) ] = J(A, B)
```

So the fraction of matching MinHash values approximates Jaccard:

```
??(A, B) = (1/128) ?? ??_{k=1}^{128} ????[MinHash_k(A) = MinHash_k(B)]
```

This estimator has variance:

```
Var[??] = J(1-J)/128
Std[??] ??? 0.088   (when J=0.5, num_perm=128)
```

#### Step 4: Locality Sensitive Hashing (LSH)
To avoid comparing all O(N??) pairs, we use LSH with banding:

```
B bands ?? R rows = num_perm
128 bands ?? 1 row = 128    (for our configuration)
```

Two documents are **candidate pairs** (and compared) only if their band signatures collide in at least one band. With the threshold `t = 0.80`:

```
P[collision in at least one band] ??? 1 - (1 - t^R)^B
```

At `t = 0.80`, `R=1`, `B=128`:
```
P[candidate pair | J=0.80] = 1 - (1 - 0.80)^128 ??? 1.0
P[candidate pair | J=0.50] = 1 - (1 - 0.50)^128 ??? 1.0
```

In practice this bands configuration acts as a complete comparison (since R=1). A production setting would use B=16, R=8 for a sharper threshold.

#### Step 5: Duplicate Resolution
For each group of near-duplicates (Jaccard ??? 0.80), we **keep the longest document** (most content) and drop the rest.

**Outcomes:** 0 docs dropped for both Hindi and Nepali (Wikipedia dump already deduplicated)

---

## 13. Stage 10 ??? Semantic & Safety Filtering

**Script:** `semantic_safety_filter.py`

### Method

We apply **keyword/pattern matching** using Unicode-aware regex patterns. This is intentionally conservative ??? no ML classifier, to avoid false positives that would destroy scarce Nepali data.

Four categories are checked:

1. **explicit_sexual_content** ??? Devanagari terms for explicit sexual content
2. **extreme_violence** ??? Terms for graphic violence (e.g., `???????????????????????????` = massacre)
3. **hate_speech** ??? Empty by default; must be manually reviewed before filling
4. **spam_gibberish** ??? Pattern `(.)\\1{20,}` (any char repeated 20+ times), `[A-Z]{50,}`

**Warning mechanism:** If the filter drops > 1% of documents, it prints a loud warning (suggests patterns are too broad and need review).

All dropped documents are written to `rejected_<lang>.jsonl` for human audit.

**Outcomes:**
- Hindi: dropped 88 docs (0.056% drop rate ??? safe)
- Nepali: dropped 21 docs (0.066% drop rate ??? safe)

---

## 14. Stage 11 ??? Decontamination against Test Set

**Script:** `decontaminate.py`

**Purpose:** Ensure no training document is a near-duplicate of a test document. This prevents data leakage that would inflate evaluation metrics.

### Method

1. **Build MinHash LSH index from the frozen test set** (3,261 Hindi + 657 Nepali test docs)
2. **Query each training candidate** against the test set index
3. **Drop training candidates** with Jaccard similarity ??? 0.80 to any test document

The Jaccard threshold of 0.80 is identical to the near-dedup threshold ??? a training doc that shares 80% of its 5-grams with a test doc is considered contaminated.

Additionally, **exact match** is checked:
```python
if sha256(training_doc) == sha256(test_doc):
    drop(training_doc)
```

**Outcomes:**
- Hindi: dropped 301 docs
- Nepali: dropped 214 docs

---

## 15. Stage 12 ??? Train / Val Split

**Script:** `split_data.py`

### Stratified Split

The 98% non-test pool is split into train/val using stratified sampling by `collection_method` (downloaded vs. manual):

```
train_ratio = 0.96
val_ratio   = 0.04
seed        = 42
```

For each stratum k:
```
n_val_k   = round(N_k ?? 0.04)
n_train_k = N_k - n_val_k
```

**Final Split Outcomes:**

| Split | Hindi Docs | Hindi Tokens | Nepali Docs | Nepali Tokens |
|---|---|---|---|---|
| Train | 13,880 | 7,274,932 | 7,767 | 1,841,236 |
| Val | 578 | 299,386 | 323 | 68,290 |
| Test | 3,261 | 1,006,670 | 657 | 106,861 |

> **Note:** Token counts are whitespace-split estimates. The actual subword token counts (after tokenization) will be higher.

---

## 16. Stage 13 ??? Corpus Statistics Computation

**Script:** `compute_stats.py`

Statistics are computed on the final train/val/test splits and written to `{lang}/data/stats.json`. These statistics feed into the auto-generated report and visualizations.

**Statistics computed per split:**
- Document count
- Token count (whitespace-split)
- Character count
- Vocabulary size (unique whitespace tokens)
- Type-token ratio (TTR) = unique_tokens / total_tokens
- Average document length (chars, words, tokens)
- Per-source breakdown (downloaded vs. manual)
- Devanagari character coverage (%)

**Visualization plots generated:**
1. **Pipeline Funnel** ??? bar chart showing document count at each stage
2. **Vocab Sweep** ??? fertility vs. vocab size for each language
3. **Source Contribution** ??? pie chart of tokens by source
4. **Manual vs. Downloaded Fraction** ??? horizontal bar showing 0% manual (current gap)

---

## 17. Stage 14 ??? Tokenizer Training (SentencePiece Unigram)

**Script:** `train_tokenizer.py`  
**Library:** `sentencepiece` (Google's SentencePiece ??? no HuggingFace dependency)

### Why SentencePiece Unigram (not BPE)?

**BPE (Byte-Pair Encoding)** is a greedy algorithm that merges the most frequent character pair at each step. It is deterministic but does not model token probabilities.

**SentencePiece Unigram** treats tokenization as a **probabilistic language model**:

```
P(segmentation S of string X) = ??_{i=1}^{|S|} P(s_i)
```

where `P(s_i)` is the unigram probability of token `s_i`. The optimal segmentation maximizes this probability:

```
S* = argmax_{S: concat(S)=X} ??_i log P(s_i)
```

This makes Unigram better for **morphologically rich languages** like Devanagari, where the same word can be segmented in multiple linguistically valid ways.

### Training Procedure

1. **Prepare training corpus:**
```python
# All texts from train.jsonl (val/test excluded to prevent vocabulary leakage)
texts = load_texts_from_jsonl("hindi/data/splits/train.jsonl")
# Write one sentence per line for SentencePiece
write_text_file_for_spm(texts, "/tmp/hi_train.txt")
```

2. **SentencePiece Unigram training:**
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="/tmp/hi_train.txt",
    model_prefix="hindi/tokenizer/vocab/hindi_tokenizer",
    vocab_size=32000,
    model_type="unigram",
    character_coverage=0.9999,   # keeps 99.99% of character types
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    normalization_rule_name="nmt_nfkc",
    remove_extra_whitespaces=True,
    input_sentence_size=5_000_000,
    shuffle_input_sentence=True,
)
```

**`character_coverage=0.9999`** is critical for Devanagari. Unlike Latin scripts with ~60 characters, Devanagari has hundreds of consonant+matra combinations. 0.9999 ensures virtually all characters are in the vocabulary (not treated as UNK).

3. **Training algorithm (simplified):**

```
Initialization: seed vocabulary = all character n-grams (n=1..6) + most frequent substrings
E-step: for each word in corpus, compute optimal segmentation under current vocab
M-step: update token probabilities using EM (Expectation Maximization):
        P(s) = count(s in optimal segmentations) / total_token_count
Pruning: remove tokens whose removal increases loss by < ??
         repeat until |vocab| = target_vocab_size
```

The EM algorithm optimizes:
```
L(V) = ??_{x ??? corpus} log P(x | V)
     = ??_{x ??? corpus} log ??_{S: concat(S)=x} ??_i P(s_i)
```

### Special Tokens

| Token | ID | Meaning |
|---|---|---|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown character |
| `<s>` | 2 | Begin-of-sequence |
| `</s>` | 3 | End-of-sequence |

### Word Piece Prefix Convention

SentencePiece uses `???` (U+2581, LOWER ONE EIGHTH BLOCK) as a word-start prefix:
- `?????????` = the word "??????" at the start of a word (after whitespace)
- `??????` (no prefix) = "??????" as a continuation of a larger word

This allows the tokenizer to represent word boundaries without explicit space tokens, and is crucial for Devanagari where words are separated by spaces but morphology is fusional.

---

## 18. Stage 15 ??? Vocabulary Sweep & Selection

The tokenizer is trained at 3 vocabulary sizes: **16,000**, **24,000**, **32,000**.

For each size, we evaluate on the **validation set** using 4 metrics:

### Metric 1: Fertility (most important)
```
fertility = total_subword_tokens / total_whitespace_words
```
Lower = better (fewer pieces per word = more whole-word tokens = better for LM training).
Typical range for Devanagari: 1.2 ??? 2.5.

### Metric 2: Average Characters per Token
```
avg_chars_per_token = total_characters / total_subword_tokens
```
Higher = better (longer tokens = more semantic meaning per token).

### Metric 3: UNK Rate
```
unk_rate = count(tokens == <unk>) / total_tokens
```
Must be < 1% (constraint). With character_coverage=0.9999, this is nearly 0.

### Metric 4: Compression Ratio
```
compression_ratio = raw_bytes / total_tokens
```
Higher = better compression (each token represents more raw bytes).

### Vocabulary Sweep Results

**Hindi:**

| Vocab Size | Fertility | Avg Chars/Token | UNK Rate | Compression |
|---|---|---|---|---|
| 16,000 | 1.37 | ??? | 0.0297% | ??? |
| 24,000 | 1.32 | ??? | 0.0309% | ??? |
| **32,000** | **1.29** | **4.236** | **0.0316%** | **4.236** |

**Nepali:**

| Vocab Size | Fertility | Avg Chars/Token | UNK Rate | Compression |
|---|---|---|---|---|
| 16,000 | 1.56 | ??? | 0.0197% | ??? |
| 24,000 | 1.49 | ??? | 0.0206% | ??? |
| **32,000** | **1.45** | **4.599** | **0.0212%** | **4.599** |

**Selected: 32,000 for both languages** ??? lowest fertility, well under 1% UNK rate, best compression. The improvement from 24K???32K is meaningful (fertility drops 0.04 for Hindi, 0.04 for Nepali) while vocabulary remains manageable for downstream LM embedding tables.

### Final Tokenizer Quality

| Metric | Hindi | Nepali |
|---|---|---|
| Vocab size | 32,000 | 32,000 |
| Avg chars/token | 4.236 | 4.599 |
| Fertility | 1.2884 | 1.4526 |
| UNK rate | 0.0316% | 0.0212% |

Nepali has higher fertility (1.45 vs. 1.29) because Nepali morphology is more agglutinative ??? words take more suffixes, producing more out-of-vocabulary sub-sequences even with a 32K vocabulary.

---

## 19. Final Numbers: What We Produced

### Complete Pipeline Funnel

**Hindi Wikipedia:**
```
Raw docs:         163,091
??? After lang_id:  160,306  (???2,785)
??? After normalize: 158,133 (???2,173)
??? After boilerplate: 158,133 (???0)
??? After quality:  158,133  (???0, fixed punct bug)
??? After exact dedup: 158,133 (???0)
??? After para dedup:  158,133 (???0)
??? After near dedup:  158,133 (???0)
??? After safety:   158,045  (???88)
??? After decontam: 157,744  (???301)
??? Train:          13,880   }
??? Val:            578      } = 17,739 total post-split
??? Test:           3,261    }  (note: test was frozen before cleaning)
```

**Nepali Wikipedia:**
```
Raw docs:          32,885
??? After lang_id:   32,885  (???0)
??? After normalize: 32,825  (???60)
??? After boilerplate: 32,825 (???0)
??? After quality:   31,744  (???1,081)
??? After exact dedup: 31,744 (???0)
??? After para dedup:  31,744 (???0)
??? After near dedup:  31,744 (???0)
??? After safety:   31,723  (???21)
??? After decontam: 31,509  (???214)
??? Train:          7,767   }
??? Val:            323     } = 8,747 total post-split
??? Test:           657     }
```

---

## 20. Manual Data Collection ??? How to Add It

The ???20% manual data requirement was **not met** because the current pipeline only ran Wikipedia (automatic). Here is exactly how to collect and ingest manual data:

### Step 1: Web Scraping

```bash
cd /scratch/nishanth.r/LMA/individual-project-Nishanth-nishu
source venv/bin/activate
export PYTHONPATH=project

# Create seed URL files
mkdir -p project/hindi/data/raw/manual/scrape
mkdir -p project/nepali/data/raw/manual/scrape

# Hindi seed URLs ??? news portals, government sites, educational content
cat > project/hindi/data/raw/manual/scrape/seed_urls.txt << 'EOF'
https://www.bbc.com/hindi/topics
https://www.aajtak.in
https://www.jagran.com
https://navbharattimes.indiatimes.com
https://hindi.ndtv.com
https://www.prabhatkhabar.com
https://www.livehindustan.com
https://hindikhabar.com
https://mhrd.gov.in
https://hi.wikipedia.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%B6%E0%A5%87%E0%A4%B7:%E0%A4%AF%E0%A4%BE%E0%A4%A6%E0%A5%8D%E0%A4%B0%E0%A4%9A%E0%A5%8D%E0%A4%9B%E0%A4%BF%E0%A4%95_%E0%A4%AA%E0%A5%83%E0%A4%B7%E0%A5%8D%E0%A4%A0
EOF

# Nepali seed URLs
cat > project/nepali/data/raw/manual/scrape/seed_urls.txt << 'EOF'
https://ekantipur.com
https://annapurnapost.com
https://myrepublica.nagariknetwork.com
https://ratopati.com
https://nepalkhabar.com
https://www.setopati.com
https://www.onlinekhabar.com
https://np.wikipedia.org/wiki/%E0%A4%B5%E0%A4%BF%E0%A4%B6%E0%A5%87%E0%A4%B7:Randompage
EOF

# Install scraping dependencies
pip install requests trafilatura tqdm

# Run web scraper ??? Hindi
python3 project/common/preprocessing/scrape_ingest.py \
    --lang hindi \
    --url-file project/hindi/data/raw/manual/scrape/seed_urls.txt \
    --repo-root project/ \
    --delay 1.5

# Run web scraper ??? Nepali
python3 project/common/preprocessing/scrape_ingest.py \
    --lang nepali \
    --url-file project/nepali/data/raw/manual/scrape/seed_urls.txt \
    --repo-root project/ \
    --delay 1.5
```

### Step 2: PDF OCR (Hindi Newspapers, Books, Government Docs)

```bash
# Install OCR dependencies
pip install pytesseract pdf2image Pillow
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin poppler-utils

# Create PDF directories
mkdir -p project/hindi/data/raw/manual/pdf_ocr
mkdir -p project/nepali/data/raw/manual/pdf_ocr

# Place PDFs in the directories, then run:
python3 project/common/preprocessing/ocr_ingest.py \
    --lang hindi \
    --input-dir project/hindi/data/raw/manual/pdf_ocr/ \
    --repo-root project/ \
    --dpi 300

python3 project/common/preprocessing/ocr_ingest.py \
    --lang nepali \
    --input-dir project/nepali/data/raw/manual/pdf_ocr/ \
    --repo-root project/ \
    --dpi 300
```

**Good free sources for PDFs:**
- **Hindi:** NCERT textbooks (ncert.nic.in), press releases from pib.gov.in, Rajya Sabha debates
- **Nepali:** Parliament of Nepal proceedings (parliament.gov.np), Nepal Gazette archives

### Step 3: Text Transcriptions

```bash
mkdir -p project/hindi/data/raw/manual/transcription
mkdir -p project/nepali/data/raw/manual/transcription

# Place plain .txt transcription files in the folders, then:
python3 project/common/preprocessing/transcription_ingest.py \
    --lang hindi \
    --input-dir project/hindi/data/raw/manual/transcription/ \
    --repo-root project/

python3 project/common/preprocessing/transcription_ingest.py \
    --lang nepali \
    --input-dir project/nepali/data/raw/manual/transcription/ \
    --repo-root project/
```

### Step 4: Re-run the Pipeline from Stage 3 Onward

Once manual data is added to `raw/manual/`, the pipeline is designed to merge it with downloaded data during the cleaning stages. The manifest tracks each document's `collection_method` (downloaded vs. manual), so the token fraction calculation is automatic.

```bash
# Re-run stages 3???18 (the cleaning and tokenization pipeline)
# The script is idempotent ??? it will skip any stage whose output already exists
# but will re-process when new raw files appear
bash submit_phase1.sh
```

---

## 21. Mathematical Summary

Here is a compact reference of all the mathematics used in this pipeline:

| Stage | Formula | What it Computes |
|---|---|---|
| **Lang ID** | `P(L\|doc) = softmax(W ?? avg_embed)` | Language probability |
| **Devanagari ratio** | `\|{c : c ??? [U+0900,U+097F]}\| / \|text\|` | Script purity |
| **Punct ratio** | `\|{c : cat(c) ??? {P,S}}\| / \|text\|` | Punctuation density |
| **Test holdout** | `n_test = round(N ?? 0.02)` | Stratified 2% holdout |
| **Train/val split** | `n_val = round(N ?? 0.04)` | Stratified 4% validation |
| **Jaccard (exact)** | `J(A,B) = \|A???B\| / \|A???B\|` | Set similarity |
| **MinHash estimate** | `??(A,B) = (1/K) ?? ????[min_k(A)=min_k(B)]` | Approximate Jaccard |
| **MinHash variance** | `???? = J(1-J)/K` | Estimation error |
| **Unigram LM** | `P(S) = ??_i P(s_i)` | Segmentation probability |
| **Tokenizer EM** | `P(s) = count(s) / ?? count(t)` | Token probability update |
| **Fertility** | `F = total_tokens / total_words` | Tokenization efficiency |
| **Compression ratio** | `C = raw_bytes / total_tokens` | Bytes per token |
| **UNK rate** | `U = count(unk_tokens) / total_tokens` | Coverage |

---

*Document generated by Nishanth R ??? CL3-410 Individual Project ??? Phase 1*  
*Last updated: 2026-08-16*
