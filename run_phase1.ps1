# =============================================================================
# run_phase1.ps1 ??? Windows PowerShell runner for Phase 1 pipeline
# =============================================================================
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$REPO = (Get-Item .).FullName
$PROJECT = "$REPO\project"
$COMMON = "$PROJECT\common\preprocessing"
$STATS = "$PROJECT\common\stats"
$VENV_PYTHON = "$REPO\venv\Scripts\python.exe"
$FASTTEXT_MODEL = "$REPO\lid.176.bin"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PHASE 1 - Data Collection and Tokenizer Construction" -ForegroundColor Cyan
Write-Host "  Author : Nishanth R  |  CL3-410 Individual Project" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Step 0: Download fastText model if missing
if (-not (Test-Path $FASTTEXT_MODEL)) {
    Write-Host "Downloading fastText lid.176.bin (~125 MB)..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" -OutFile $FASTTEXT_MODEL
    Write-Host "??? lid.176.bin downloaded." -ForegroundColor Green
} else {
    Write-Host "??? lid.176.bin already present." -ForegroundColor Green
}

$LANGS = @("hindi", "nepali")

# Clean existing raw/downloaded data to ensure fresh run if needed
foreach ($LANG in $LANGS) {
    $RAW_DIR = "$PROJECT\$LANG\data\raw\downloaded"
    if (Test-Path $RAW_DIR) { Remove-Item -Recurse -Force $RAW_DIR }
    $MAN_FILE = "$PROJECT\$LANG\data\manifest.csv"
    if (Test-Path $MAN_FILE) { Remove-Item -Force $MAN_FILE }
}

# Stage 1: Reconnaissance
Write-Host "`n--- STAGE 1: Source Reconnaissance ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    Write-Host "Running recon for $LANG..." -ForegroundColor Yellow
    & $VENV_PYTHON "$COMMON\recon.py" --lang $LANG --repo-root $PROJECT
}

# Stage 2: Raw Acquisition (Public Corpora)
Write-Host "`n--- STAGE 2: Raw Data Acquisition ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    Write-Host "Downloading public datasets for $LANG..." -ForegroundColor Yellow
    & $VENV_PYTHON "$PROJECT\$LANG\data\download_public.py" --source all --max-docs 1000 --repo-root $PROJECT
}

# Stage 3: Manual Collection Ingestion
Write-Host "`n--- STAGE 3: Manual Collection Ingestion ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $SEED_FILE = "$PROJECT\$LANG\data\raw\manual\scrape\seed_urls.txt"
    if (Test-Path $SEED_FILE) {
        Write-Host "Ingesting manual web scrape for $LANG..." -ForegroundColor Yellow
        & $VENV_PYTHON "$COMMON\scrape_ingest.py" --lang $LANG --url-file $SEED_FILE --repo-root $PROJECT
    }
}

# Stage 4: Test Holdout Freeze
Write-Host "`n--- STAGE 4: Test Holdout Freeze ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $FROZEN = "$PROJECT\$LANG\data\test_holdout\FROZEN.txt"
    if (Test-Path $FROZEN) { Remove-Item -Force $FROZEN }
    $TEST_FILE = "$PROJECT\$LANG\data\test_holdout\test.jsonl"
    if (Test-Path $TEST_FILE) { Remove-Item -Force $TEST_FILE }

    Write-Host "Freezing test holdout for $LANG..." -ForegroundColor Yellow
    & $VENV_PYTHON "$COMMON\test_holdout.py" --lang $LANG --repo-root $PROJECT
}

# Stages 5-8: Cleaning Pipeline
Write-Host "`n--- STAGES 5-8: Cleaning Pipeline ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $RAW_DIR = "$PROJECT\$LANG\data\raw\downloaded"
    $CLEAN_DIR = "$PROJECT\$LANG\data\cleaned"
    if (-not (Test-Path $CLEAN_DIR)) { New-Item -ItemType Directory -Force -Path $CLEAN_DIR | Out-Null }

    $RAW_FILES = Get-ChildItem -Path "$PROJECT\$LANG\data\raw" -Recurse -Filter "*.jsonl" -ErrorAction SilentlyContinue
    foreach ($file in $RAW_FILES) {
        $BASE = $file.BaseName
        Write-Host "  Cleaning $LANG / $BASE..." -ForegroundColor Yellow

        $OUT_LANGID = "$CLEAN_DIR\${BASE}_langid.jsonl"
        & $VENV_PYTHON "$COMMON\lang_id_filter.py" --lang $LANG --input $file.FullName --output $OUT_LANGID --fasttext-model $FASTTEXT_MODEL --repo-root $PROJECT

        $OUT_NORM = "$CLEAN_DIR\${BASE}_normalized.jsonl"
        & $VENV_PYTHON "$COMMON\normalize.py" --lang $LANG --input $OUT_LANGID --output $OUT_NORM --repo-root $PROJECT

        $OUT_BPLR = "$CLEAN_DIR\${BASE}_boilerplate.jsonl"
        & $VENV_PYTHON "$COMMON\boilerplate_strip.py" --lang $LANG --input $OUT_NORM --output $OUT_BPLR

        $OUT_QUAL = "$CLEAN_DIR\${BASE}_quality.jsonl"
        & $VENV_PYTHON "$COMMON\quality_filters.py" --lang $LANG --input $OUT_BPLR --output $OUT_QUAL --repo-root $PROJECT
    }

    # Merge cleaned quality files
    $MERGED = "$CLEAN_DIR\all_quality_filtered.jsonl"
    Write-Host "  Merging cleaned files into $MERGED..." -ForegroundColor Yellow
    Get-Content "$CLEAN_DIR\*_quality.jsonl" | Set-Content $MERGED
}

# Stage 10: Exact Deduplication
Write-Host "`n--- STAGE 10: Exact Deduplication ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $CLEAN_DIR = "$PROJECT\$LANG\data\cleaned"
    $DEDUP_DIR = "$PROJECT\$LANG\data\dedup"
    if (-not (Test-Path $DEDUP_DIR)) { New-Item -ItemType Directory -Force -Path $DEDUP_DIR | Out-Null }
    & $VENV_PYTHON "$COMMON\dedup_exact.py" --lang $LANG --input "$CLEAN_DIR\all_quality_filtered.jsonl" --output "$DEDUP_DIR\exact_deduped.jsonl"
}

# Stage 11: Paragraph Deduplication
Write-Host "`n--- STAGE 11: Paragraph Deduplication ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $DEDUP_DIR = "$PROJECT\$LANG\data\dedup"
    & $VENV_PYTHON "$COMMON\dedup_paragraph.py" --lang $LANG --input "$DEDUP_DIR\exact_deduped.jsonl" --output "$DEDUP_DIR\para_deduped.jsonl" --repo-root $PROJECT
}

# Stage 12: Near Deduplication
Write-Host "`n--- STAGE 12: Near Deduplication (MinHash + LSH) ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $DEDUP_DIR = "$PROJECT\$LANG\data\dedup"
    & $VENV_PYTHON "$COMMON\dedup_near.py" --lang $LANG --input "$DEDUP_DIR\para_deduped.jsonl" --output "$DEDUP_DIR\near_deduped.jsonl"
}

# Stage 13: Semantic / Safety Filtering
Write-Host "`n--- STAGE 13: Safety Filter ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $DEDUP_DIR = "$PROJECT\$LANG\data\dedup"
    & $VENV_PYTHON "$COMMON\semantic_safety_filter.py" --lang $LANG --input "$DEDUP_DIR\near_deduped.jsonl" --output "$DEDUP_DIR\safety_filtered.jsonl" --repo-root $PROJECT
}

# Stage 14: Decontamination
Write-Host "`n--- STAGE 14: Decontamination against Test Set ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $DEDUP_DIR = "$PROJECT\$LANG\data\dedup"
    $DECONTAM_DIR = "$PROJECT\$LANG\data\decontaminated"
    if (-not (Test-Path $DECONTAM_DIR)) { New-Item -ItemType Directory -Force -Path $DECONTAM_DIR | Out-Null }
    $TEST_HOLDOUT = "$PROJECT\$LANG\data\test_holdout\test.jsonl"

    & $VENV_PYTHON "$COMMON\decontaminate.py" --lang $LANG --input "$DEDUP_DIR\safety_filtered.jsonl" --test-holdout $TEST_HOLDOUT --output "$DECONTAM_DIR\decontaminated.jsonl" --repo-root $PROJECT
}

# Stage 16: Train / Val Splits
Write-Host "`n--- STAGE 16: Train / Val Splits ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    $SPLITS_DIR = "$PROJECT\$LANG\data\splits"
    if (-not (Test-Path $SPLITS_DIR)) { New-Item -ItemType Directory -Force -Path $SPLITS_DIR | Out-Null }
    $DECONTAM_DIR = "$PROJECT\$LANG\data\decontaminated"

    & $VENV_PYTHON "$COMMON\split_data.py" --lang $LANG --input "$DECONTAM_DIR\decontaminated.jsonl" --repo-root $PROJECT
}

# Stages 17-18: Tokenizer Training + Vocab Sweep
Write-Host "`n--- STAGES 17-18: Tokenizer Training and Sweep ---" -ForegroundColor Cyan
foreach ($LANG in $LANGS) {
    # Remove old model files to force training
    $VOCAB_DIR = "$PROJECT\$LANG\tokenizer\vocab"
    if (Test-Path $VOCAB_DIR) { Remove-Item -Recurse -Force $VOCAB_DIR }

    & $VENV_PYTHON "$COMMON\train_tokenizer.py" --lang $LANG --repo-root $PROJECT
}

# Final Stats & Report Generation
Write-Host "`n--- FINAL: Statistics, Plots and Report Generation ---" -ForegroundColor Cyan
$HINDI_STATS = "$PROJECT\hindi\data\stats.json"
$NEPALI_STATS = "$PROJECT\nepali\data\stats.json"
$FIGURES_DIR = "$PROJECT\report\figures"
if (-not (Test-Path $FIGURES_DIR)) { New-Item -ItemType Directory -Force -Path $FIGURES_DIR | Out-Null }

& $VENV_PYTHON "$STATS\compute_stats.py" --lang hindi --repo-root $PROJECT --output $HINDI_STATS
& $VENV_PYTHON "$STATS\compute_stats.py" --lang nepali --repo-root $PROJECT --output $NEPALI_STATS

& $VENV_PYTHON "$STATS\plots.py" --hindi-stats $HINDI_STATS --nepali-stats $NEPALI_STATS --output-dir $FIGURES_DIR
& $VENV_PYTHON "$STATS\generate_report.py" --hindi-stats $HINDI_STATS --nepali-stats $NEPALI_STATS --figures-dir $FIGURES_DIR --output "$PROJECT\report\phase1_report.md" --repo-root $PROJECT

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Phase 1 Pipeline Execution COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
