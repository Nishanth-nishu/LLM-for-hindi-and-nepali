#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# upload_to_drive.sh — Phase 1 deliverables from the GCP VM to Google Drive
# ---------------------------------------------------------------------------
# Stages two things:
#   1. both tokenizers (model, vocab, metadata, sweep evidence)  -> TOKENIZER_FOLDER
#   2. the final corpus splits, gzipped                          -> DATA_FOLDER
#      as clean_data_hindi/ and clean_data_nepali/
#
# Requires an rclone remote named `gdrive` (see SETUP in the message that
# shipped this file). Run from the repo root:
#
#     bash tools/upload_to_drive.sh            # stage, then upload
#     bash tools/upload_to_drive.sh --stage-only    # package, don't upload
#     bash tools/upload_to_drive.sh --dry-run       # show what would transfer
#
# WHY GZIP. The two corpora are ~11 GB of UTF-8 Devanagari. Devanagari costs 3
# bytes per character and the text is highly repetitive at the byte level, so
# gzip typically returns ~3.5x. That turns a multi-hour upload into a
# manageable one and costs nothing at the far end -- `zcat train.jsonl.gz` and
# every JSONL reader that accepts a stream still work.
# ---------------------------------------------------------------------------
set -euo pipefail

TOKENIZER_FOLDER="1yKSysO5UJdSmYw_H7XrhEY_jx0cfKQIY"
DATA_FOLDER="1pwRASlKuWFjfS1DvV9iks-PZN4Orwfb6"
REMOTE="gdrive"
STAGE="${HOME}/drive_upload"

DRY=""
STAGE_ONLY=0
WITH_RAW=0
RAW_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY="--dry-run" ;;
    --stage-only) STAGE_ONLY=1 ;;
    --with-raw)   WITH_RAW=1 ;;
    --raw-only)   WITH_RAW=1; RAW_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

REPO="$(pwd)"
for f in hindi/data/splits/train.jsonl nepali/data/splits/train.jsonl \
         hindi/tokenizer/vocab/hindi_tokenizer.model \
         nepali/tokenizer/vocab/nepali_tokenizer.model; do
  [[ -f "$REPO/$f" ]] || { echo "[error] missing $f -- run from the repo root" >&2; exit 1; }
done

# pigz uses every core; gzip uses one. On an 8-core VM that is the difference
# between ~8 minutes and ~50 for 11 GB.
if command -v pigz >/dev/null 2>&1; then ZIP="pigz -c"; else ZIP="gzip -c"; fi
echo "[info] compressing with: $ZIP"

# ---- raw preflight --------------------------------------------------------
# Raw is several times the size of the finished corpus -- it is everything
# collected BEFORE filtering, deduplication and the budget trim. Print the
# number before spending an hour on it, because Drive's free tier is 15 GB
# shared with Gmail and this can exhaust it.
if [[ $WITH_RAW -eq 1 ]]; then
  echo
  echo "=== raw preflight ==="
  total=0
  for L in hindi nepali; do
    if [[ -d "$REPO/$L/data/raw" ]]; then
      sz=$(find "$REPO/$L/data/raw" -maxdepth 1 -name '*.jsonl' -printf '%s\n' \
           2>/dev/null | awk '{s+=$1} END {print s+0}')
      n=$(find "$REPO/$L/data/raw" -maxdepth 1 -name '*.jsonl' | wc -l)
      echo "  $L: $n raw files, $(numfmt --to=iec "$sz" 2>/dev/null || echo "$sz")B"
      total=$((total + sz))
    fi
  done
  echo "  uncompressed total: $(numfmt --to=iec "$total" 2>/dev/null || echo "$total")B"
  echo "  expect roughly $(numfmt --to=iec $((total / 3)) 2>/dev/null || echo $((total/3)))B after gzip"
  echo "  (Devanagari is 3 bytes/char and compresses ~3-3.5x)"
fi

if [[ $RAW_ONLY -eq 0 ]]; then
echo
echo "=== 1/3  staging tokenizers ==="
for L in hindi nepali; do
  dst="$STAGE/tokenizers/${L}_tokenizer"
  mkdir -p "$dst"
  cp -v "$REPO/$L/tokenizer/vocab/${L}_tokenizer.model" "$dst/"
  cp -v "$REPO/$L/tokenizer/vocab/${L}_tokenizer.vocab" "$dst/"
  cp -v "$REPO/$L/tokenizer/vocab/${L}_tokenizer.json"  "$dst/" 2>/dev/null || true
  # The evidence for the vocabulary choice travels with the model. Without it
  # "vocab_size=4000" is an assertion; with it, it is a decision.
  for extra in analysis/vocab_selection.json analysis/sweep_results.csv \
               analysis/sweep_results_selection.csv analysis/token_stats.json \
               analysis/examples.md; do
    [[ -f "$REPO/$L/tokenizer/$extra" ]] && cp "$REPO/$L/tokenizer/$extra" "$dst/"
  done
done

echo
echo "=== 2/3  staging corpus splits (gzip) ==="
for L in hindi nepali; do
  dst="$STAGE/clean_data_$L"
  mkdir -p "$dst"
  for S in train val test; do
    src="$REPO/$L/data/splits/$S.jsonl"
    [[ -f "$src" ]] || { echo "[warn] no $src, skipping"; continue; }
    if [[ -s "$dst/$S.jsonl.gz" ]]; then
      echo "  $L/$S.jsonl.gz already staged, skipping"
    else
      echo "  compressing $L/$S.jsonl ($(du -h "$src" | cut -f1)) ..."
      $ZIP "$src" > "$dst/$S.jsonl.gz"
    fi
  done
  # Provenance rides along with the data. Anyone who opens this folder in six
  # months can see what the token counts were and how they were produced.
  for extra in data/stats/corpus_stats.json data/stats/token_accounting.json \
               data/stats/manifest.json; do
    [[ -f "$REPO/$L/$extra" ]] && cp "$REPO/$L/$extra" "$dst/"
  done
  [[ -f "$REPO/report/verification.json" ]] && cp "$REPO/report/verification.json" "$dst/"
done
fi   # RAW_ONLY

# ---- raw, pre-cleaning ----------------------------------------------------
# One .gz per source file rather than one concatenated blob: the filenames ARE
# the provenance. manual_scrape.jsonl and gcs_sangraha_verified.jsonl tell you
# where a document came from without opening it, and merging them would throw
# that away.
if [[ $WITH_RAW -eq 1 ]]; then
  echo
  echo "=== staging RAW (pre-cleaning) ==="
  for L in hindi nepali; do
    src_dir="$REPO/$L/data/raw"
    [[ -d "$src_dir" ]] || { echo "[warn] no $src_dir"; continue; }
    dst="$STAGE/raw_data_$L"
    mkdir -p "$dst"
    while IFS= read -r src; do
      base="$(basename "$src")"
      if [[ -s "$dst/$base.gz" ]]; then
        echo "  $L/$base.gz already staged, skipping"
      else
        echo "  compressing $L/$base ($(du -h "$src" | cut -f1)) ..."
        $ZIP "$src" > "$dst/$base.gz"
      fi
    done < <(find "$src_dir" -maxdepth 1 -name '*.jsonl' | sort)
    # The URL frontier is small and is the only record of what the crawler was
    # ever pointed at, including the pages it fetched and rejected.
    for extra in article_urls.txt; do
      [[ -f "$src_dir/$extra" ]] && $ZIP "$src_dir/$extra" > "$dst/$extra.gz"
    done
    [[ -f "$REPO/$L/configs/seed_domains.txt" ]] && \
      cp "$REPO/$L/configs/seed_domains.txt" "$dst/"
  done
fi

echo
echo "  staged sizes:"
du -sh "$STAGE"/* 2>/dev/null

if [[ $STAGE_ONLY -eq 1 ]]; then
  echo
  echo "[done] staged at $STAGE (upload skipped)"
  exit 0
fi

command -v rclone >/dev/null 2>&1 || { echo "[error] rclone not on PATH" >&2; exit 1; }
rclone listremotes | grep -qx "${REMOTE}:" || {
  echo "[error] no rclone remote named '${REMOTE}'. Run the SETUP steps first." >&2
  exit 1; }

echo
echo "=== 3/3  uploading ==="
# --checksum makes a re-run resume rather than re-send: rclone compares MD5s
# and skips what already matches. Safe to interrupt and restart.
COMMON=(--progress --transfers 4 --checkers 8 --checksum --drive-chunk-size 64M $DRY)

if [[ $RAW_ONLY -eq 0 ]]; then
  echo "-> tokenizers"
  rclone copy "$STAGE/tokenizers" "${REMOTE},root_folder_id=${TOKENIZER_FOLDER}:" "${COMMON[@]}"

  for L in hindi nepali; do
    echo "-> clean_data_$L"
    rclone copy "$STAGE/clean_data_$L" \
      "${REMOTE},root_folder_id=${DATA_FOLDER}:clean_data_$L" "${COMMON[@]}"
  done
fi

if [[ $WITH_RAW -eq 1 ]]; then
  for L in hindi nepali; do
    [[ -d "$STAGE/raw_data_$L" ]] || continue
    echo "-> raw_data_$L"
    rclone copy "$STAGE/raw_data_$L" \
      "${REMOTE},root_folder_id=${DATA_FOLDER}:raw_data_$L" "${COMMON[@]}"
  done
fi

echo
echo "[done] verify with:"
echo "  rclone ls \"${REMOTE},root_folder_id=${TOKENIZER_FOLDER}:\""
echo "  rclone ls \"${REMOTE},root_folder_id=${DATA_FOLDER}:\""
