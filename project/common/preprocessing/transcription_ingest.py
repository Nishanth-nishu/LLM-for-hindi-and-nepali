"""
transcription_ingest.py  (shared ??? use --lang flag)
-----------------------------------------------------
Manual collection Step 3: ingest manually typed / transcribed text files.

Reads a directory of plain .txt files that you have personally typed,
transcribed from audio, or manually annotated.  Each .txt file becomes one
document in the corpus.

Usage
-----
  # Put your transcribed .txt files into:
  #   hindi/data/raw/transcribed/
  # Then run:
  python transcription_ingest.py \\
      --lang hindi \\
      --input-dir hindi/data/raw/transcribed/ \\
      --repo-root /path/to/project
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import append_rows, existing_doc_ids, init_manifest, make_row

MIN_CHARS = 50  # minimum characters to accept a transcription file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest manually transcribed .txt files into the corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .txt transcription files.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-name", default=None,
                        help="Override source name in manifest.")
    parser.add_argument("--license-note", default="Own transcription ??? no restrictions")
    return parser.parse_args()


def main() -> None:
    """
    Ingest all .txt files in input_dir as manual transcription documents.

    Each file becomes one document.  doc_id is derived from the filename.
    Idempotent: already-ingested files (by doc_id) are skipped.
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists():
        print(f"[transcription_ingest] ERROR: Directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    init_manifest(args.lang, str(repo_root))
    seen_ids = existing_doc_ids(args.lang, str(repo_root))

    raw_dir = repo_root / args.lang / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{args.lang}_transcribed.jsonl"

    source_name = args.source_name or f"Manual transcription ({args.lang})"
    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"[transcription_ingest] Found {len(txt_files)} .txt file(s).")

    new_rows = []
    success = 0

    with open(output_path, "a", encoding="utf-8") as fout:
        for txt_file in tqdm(txt_files, desc=f"[transcription] {args.lang}", unit="file"):
            doc_id = f"{args.lang}_transcription_{txt_file.stem}"
            if doc_id in seen_ids:
                continue

            text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) < MIN_CHARS:
                print(f"  [SKIP] Too short: {txt_file.name}", file=sys.stderr)
                continue

            raw_doc = {
                "doc_id": doc_id,
                "text": text,
                "source_type": "transcription",
                "collection_method": "manual",
                "source_name": source_name,
            }
            fout.write(json.dumps(raw_doc, ensure_ascii=False) + "\n")

            new_rows.append(make_row(
                doc_id=doc_id,
                source_name=source_name,
                source_type="transcription",
                collection_method="manual",
                url_or_path=str(txt_file),
                raw_char_count=len(text),
                license_note=args.license_note,
            ))
            seen_ids.add(doc_id)
            success += 1

    if new_rows:
        append_rows(args.lang, new_rows, str(repo_root))

    print(f"\n[transcription_ingest] Done. Ingested: {success} files ??? {output_path}")


if __name__ == "__main__":
    main()
