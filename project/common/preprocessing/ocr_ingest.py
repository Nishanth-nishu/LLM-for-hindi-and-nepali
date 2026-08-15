"""
ocr_ingest.py  (shared ??? use --lang flag)
------------------------------------------
Manual collection Step 1: OCR pipeline for PDF / image scans.

Given a directory of PDF or image files (PNG, JPG, TIFF), this script:
  1. Converts PDFs to images (using pdf2image / poppler).
  2. Runs Tesseract OCR with the appropriate language pack.
  3. Applies basic post-OCR cleaning (removes obvious garbage lines).
  4. Writes cleaned text as JSONL to the raw directory.
  5. Appends manifest rows with collection_method=manual, source_type=ocr.

Tesseract language packs
------------------------
- Hindi   : ``hin``  (well-supported, good accuracy)
- Nepali: No native Nepali traineddata exists in Tesseract as of 2024.
  We fall back to ``hin`` (Devanagari script) and flag this in the manifest
  and report as a known limitation.  For best results, consider using a
  fine-tuned Tesseract model trained on Nepali-specific fonts.

Dependencies
------------
  pip install pytesseract pdf2image Pillow tqdm
  apt-get install tesseract-ocr tesseract-ocr-hin poppler-utils

Usage
-----
  python ocr_ingest.py \\
      --lang hindi \\
      --input-dir /path/to/pdfs_and_images/ \\
      --repo-root /path/to/project \\
      --dpi 300
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import append_rows, existing_doc_ids, init_manifest, make_row

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tesseract language pack per language
TESS_LANG_MAP = {
    "hindi": "hin",
    "nepali": "hin",  # fallback ??? see module docstring
}

# Source name for manifest
SOURCE_NAMES = {
    "hindi": "Manual OCR (Hindi PDFs/scans)",
    "nepali": "Manual OCR (Nepali PDFs/scans ??? hin tessdata fallback)",
}

# Minimum character count after OCR to consider a page usable
MIN_OCR_CHARS = 50

# OCR noise patterns to strip (common Tesseract artefacts)
_OCR_NOISE = re.compile(r"[|}{\\<>~`_]{3,}")  # runs of noise chars
_MOSTLY_NUMBERS = re.compile(r"^\s*[\d\s.,\-:]{5,}\s*$")  # lines of just numbers


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list:
    """
    Convert a PDF file to a list of PIL Images using pdf2image.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.
    dpi : int
        Resolution for rendering (higher = better OCR, slower).

    Returns
    -------
    list[PIL.Image]
        List of page images.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print(
            "[ocr_ingest] ERROR: pdf2image not installed. "
            "Run: pip install pdf2image",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"  [WARN] Could not convert PDF {pdf_path.name}: {e}", file=sys.stderr)
        return []


def ocr_image(image, lang: str, tess_lang: str) -> str:
    """
    Run Tesseract OCR on a PIL Image and return the extracted text.

    Parameters
    ----------
    image : PIL.Image
        Page image.
    lang : str
        Target language key (for logging).
    tess_lang : str
        Tesseract language code (e.g. 'hin').

    Returns
    -------
    str
        Raw OCR output text.
    """
    try:
        import pytesseract
    except ImportError:
        print(
            "[ocr_ingest] ERROR: pytesseract not installed. "
            "Run: pip install pytesseract",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return pytesseract.image_to_string(image, lang=tess_lang)
    except Exception as e:
        print(f"  [WARN] Tesseract error: {e}", file=sys.stderr)
        return ""


def clean_ocr_text(text: str) -> str:
    """
    Apply basic post-OCR cleaning to remove common Tesseract artefacts.

    Parameters
    ----------
    text : str
        Raw OCR output.

    Returns
    -------
    str
        Cleaned text.  May be empty if the page was garbage.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")
            continue
        # Drop lines that are mostly OCR noise
        if _OCR_NOISE.search(line):
            continue
        # Drop lines that are only numbers / punctuation (page numbers, tables)
        if _MOSTLY_NUMBERS.match(line):
            continue
        cleaned.append(line)

    # Collapse multiple blank lines
    result_lines = []
    prev_blank = False
    for line in cleaned:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank

    return "\n".join(result_lines).strip()


def ocr_file(file_path: Path, lang: str, dpi: int = 300) -> str:
    """
    OCR a single file (PDF or image) and return the full extracted text.

    Parameters
    ----------
    file_path : Path
        Path to the input file.
    lang : str
        Target language key.
    dpi : int
        DPI for PDF rendering.

    Returns
    -------
    str
        Full extracted and cleaned text.
    """
    tess_lang = TESS_LANG_MAP[lang]
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        images = pdf_to_images(file_path, dpi=dpi)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            from PIL import Image
            images = [Image.open(str(file_path))]
        except Exception as e:
            print(f"  [WARN] Could not open image {file_path.name}: {e}", file=sys.stderr)
            return ""
    else:
        print(f"  [WARN] Unsupported file type: {file_path.suffix}", file=sys.stderr)
        return ""

    page_texts = []
    for img in images:
        raw = ocr_image(img, lang, tess_lang)
        cleaned = clean_ocr_text(raw)
        if len(cleaned) >= MIN_OCR_CHARS:
            page_texts.append(cleaned)

    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual collection: OCR pipeline for PDFs and image scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing PDF/image files to OCR.")
    parser.add_argument("--repo-root", default=".",
                        help="Project root directory.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PDF-to-image conversion.")
    parser.add_argument("--source-name", default=None,
                        help="Override the source name recorded in the manifest.")
    parser.add_argument("--license-note", default="Unknown ??? check source",
                        help="License for the OCR source material.")
    return parser.parse_args()


def main() -> None:
    """
    OCR all files in the input directory and ingest them into the corpus.

    Logs per-file outcomes and appends manifest rows for all successfully
    processed files.
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists():
        print(f"[ocr_ingest] ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    init_manifest(args.lang, str(repo_root))
    seen_ids = existing_doc_ids(args.lang, str(repo_root))

    raw_dir = repo_root / args.lang / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{args.lang}_ocr.jsonl"

    source_name = args.source_name or SOURCE_NAMES[args.lang]

    if args.lang == "nepali":
        print(
            "[ocr_ingest] NOTE: Nepali OCR uses 'hin' Tesseract traineddata "
            "(Devanagari fallback). Accuracy may vary. "
            "See report Section 3.2 for discussion."
        )

    # Find all processable files
    supported_exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_exts
    )

    print(f"[ocr_ingest] Found {len(files)} file(s) in {input_dir}")

    new_rows = []
    total = 0
    success = 0

    with open(output_path, "a", encoding="utf-8") as fout:
        for file_path in tqdm(files, desc=f"[ocr] {args.lang}", unit="file"):
            # Generate a stable doc_id from the filename
            doc_id = f"{args.lang}_ocr_{file_path.stem}"
            if doc_id in seen_ids:
                print(f"  [SKIP] Already processed: {file_path.name}")
                continue

            total += 1
            text = ocr_file(file_path, args.lang, dpi=args.dpi)

            if not text.strip():
                print(f"  [WARN] Empty OCR result for: {file_path.name}", file=sys.stderr)
                continue

            raw_doc = {
                "doc_id": doc_id,
                "text": text,
                "source_type": "ocr",
                "collection_method": "manual",
                "source_name": source_name,
            }
            fout.write(json.dumps(raw_doc, ensure_ascii=False) + "\n")

            new_rows.append(make_row(
                doc_id=doc_id,
                source_name=source_name,
                source_type="ocr",
                collection_method="manual",
                url_or_path=str(file_path),
                raw_char_count=len(text),
                license_note=args.license_note,
            ))
            seen_ids.add(doc_id)
            success += 1

    if new_rows:
        append_rows(args.lang, new_rows, str(repo_root))

    print(
        f"\n[ocr_ingest] Done. "
        f"Files attempted: {total} | Success: {success} | "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
