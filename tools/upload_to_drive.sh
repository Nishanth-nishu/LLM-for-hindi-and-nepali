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
for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY="--dry-run" ;;
    --stage-only) STAGE_ONLY=1 ;;
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

echo
echo "  staged sizes:"
du -sh "$STAGE"/tokenizers "$STAGE"/clean_data_hindi "$STAGE"/clean_data_nepali

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

echo "-> tokenizers"
rclone copy "$STAGE/tokenizers" "${REMOTE},root_folder_id=${TOKENIZER_FOLDER}:" "${COMMON[@]}"

for L in hindi nepali; do
  echo "-> clean_data_$L"
  rclone copy "$STAGE/clean_data_$L" \
    "${REMOTE},root_folder_id=${DATA_FOLDER}:clean_data_$L" "${COMMON[@]}"
done

echo
echo "[done] verify with:"
echo "  rclone ls \"${REMOTE},root_folder_id=${TOKENIZER_FOLDER}:\""
echo "  rclone ls \"${REMOTE},root_folder_id=${DATA_FOLDER}:\""
