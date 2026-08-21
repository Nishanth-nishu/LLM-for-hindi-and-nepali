"""
ocr_collect.py — manual collection from PDFs and scanned images
================================================================
The second qualifying manual route in the brief ("OCR from books/PDFs").

Emits `pipeline.manifest.Document` records with
`provenance_class="manual", collection_method="ocr"` so the tokens count toward
the >=20% requirement automatically.

THROUGHPUT, HONESTLY
--------------------
Tesseract on 300 DPI Devanagari runs roughly **1-3 seconds per page** on a
notebook CPU. A page of a printed book is ~350 words ~= 600 tokens.

    100M manual tokens by OCR alone = ~167,000 pages = 46 to 140 hours

OCR is not how you reach 20% of 500M in a day. It is how you reach a
*defensible slice* of it, with a quality of text that scraped news cannot match
(long-form, edited, no boilerplate). Use `--workers` to parallelise across CPU
cores, and plan on OCR contributing a few million tokens while scraping carries
the bulk.

If you want the honest version in your report: say OCR contributed N tokens from
M books, that scraping contributed the rest of the manual share, and give the
per-source table. That is a stronger answer than pretending OCR scaled.

NEPALI TESSERACT
----------------
Tesseract ships no Nepali (`nep`) traineddata. `hin` covers the Devanagari
script and is the standard fallback, but it carries a Hindi language model, so
Nepali-specific orthography (हरू, छ, ँ) is where errors concentrate. This is a
real limitation and belongs in the report -- the script records
`ocr_lang_fallback: true` in every affected record so you can quantify it later.

Mitigations, in order of effort:
  1. `--psm 6` (uniform block) is usually better than the default on book scans.
  2. Post-OCR filtering (below) drops the worst pages rather than letting
     garbage into the corpus.
  3. Fine-tuned Devanagari traineddata, if you have time -- you do not.

DEPENDENCIES
------------
  pip install pytesseract pdf2image Pillow
  apt-get install tesseract-ocr tesseract-ocr-hin poppler-utils

Usage
-----
  python -m pipeline.collect.ocr_collect --lang hindi --repo-root . \\
      --input-dir /path/to/pdfs --workers 4
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import Document, ShardWriter  # noqa: E402

TESS_LANG = {"hindi": "hin", "nepali": "hin"}   # no `nep` pack exists
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def clean_ocr_text(text: str) -> str:
    """
    Post-OCR cleanup. Conservative: drops lines that are clearly artefacts and
    leaves everything else alone. Aggressive "correction" of OCR output is how
    you silently introduce errors that look like real words.
    """
    text = unicodedata.normalize("NFC", text)
    out = []
    for line in text.splitlines():
        s = " ".join(line.split())
        if not s:
            continue
        if len(s) <= 2:                                  # stray marks
            continue
        stripped = re.sub(r"\s+", "", s)
        if not stripped:
            continue
        dev = len(DEVANAGARI.findall(stripped)) / len(stripped)
        if dev < 0.5:                                    # page numbers, headers, noise
            continue
        if len(set(stripped)) <= 2:                      # "।।।।।।" style artefacts
            continue
        out.append(s)
    return "\n".join(out).strip()


def page_quality_ok(text: str, *, min_chars: int, min_dev: float = 0.75) -> tuple[bool, str]:
    if len(text) < min_chars:
        return False, "too_short"
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return False, "empty"
    dev = len(DEVANAGARI.findall(stripped)) / len(stripped)
    if dev < min_dev:
        return False, f"low_devanagari({dev:.2f})"
    words = text.split()
    if len(words) < 40:
        return False, "too_few_words"
    # OCR failure signature: many very short "words" from broken conjuncts.
    short = sum(1 for w in words if len(w) <= 2) / len(words)
    if short > 0.45:
        return False, f"fragmented({short:.2f})"
    return True, "ok"


def extract_text_layer(path: Path, max_pages: int | None) -> list[str]:
    """
    Pull the embedded text layer with `pdftotext`, or return [] if there is none.

    THIS IS TRIED BEFORE OCR, AND IT MATTERS ENORMOUSLY.

    A born-digital PDF -- anything produced by a word processor or typesetter
    rather than a scanner -- already contains its text. Statutes, gazettes,
    ministry reports and most government publications are born-digital. For
    those files:

      * pdftotext returns the EXACT characters, with no recognition step and
        therefore no recognition errors;
      * it takes milliseconds per page instead of the several seconds Tesseract
        needs to rasterise at 300 dpi and run a neural recogniser.

    On a 100-page statute that is the difference between 0.2 seconds and ten
    minutes, for a result that is not merely faster but strictly more accurate.
    Devanagari OCR is good, not perfect: it confuses conjuncts and matras, and
    every such error becomes a spurious token the tokenizer must learn.

    OCR remains essential for genuinely scanned material -- old books, image
    PDFs -- which is why it stays as the fallback rather than being replaced.
    """
    import subprocess
    cmd = ["pdftotext", "-layout", "-enc", "UTF-8"]
    if max_pages:
        cmd += ["-l", str(max_pages)]
    cmd += [str(path), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    text = out.stdout.decode("utf-8", errors="replace")
    # pdftotext separates pages with a form feed.
    pages = [p for p in text.split("\f")]
    joined = "".join(pages)
    # A scanned PDF still "succeeds" here, returning a handful of stray
    # characters from headers or an OCR layer that was never added. Require
    # real content before believing it, otherwise a scan silently yields an
    # almost-empty document and never reaches the OCR path.
    if len(joined.strip()) < 200:
        return []
    dev = sum(1 for c in joined if "\u0900" <= c <= "\u097f")
    if dev < len(joined.strip()) * 0.25:
        return []
    return pages


def ocr_one_file(args_tuple) -> tuple[str, list[str], str]:
    """Worker: extract one PDF/image. Returns (filename, page_texts, error)."""
    path_str, lang, dpi, psm, max_pages, text_layer_only = args_tuple
    path = Path(path_str)

    # Text layer first: faster and more accurate when it exists.
    if path.suffix.lower() == ".pdf":
        pages = extract_text_layer(path, max_pages)
        if pages:
            return path.name, pages, ""
        if text_layer_only:
            # Deliberate skip, not a failure. On a mixed harvest the scanned
            # minority can cost more wall-clock than the born-digital majority
            # by two orders of magnitude: pdftotext is ~50 ms per document,
            # Tesseract is 3-10 SECONDS per page. When a deadline is the
            # binding constraint, taking the 76% that is free and declaring
            # the rest is the correct trade -- and it is a trade you should
            # state in the report rather than leave implicit.
            return path.name, [], "SKIP:no_text_layer"

    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        return path.name, [], f"missing dependency: {e}"

    tess_lang = TESS_LANG[lang]
    config = f"--psm {psm}"
    pages: list[str] = []

    try:
        if path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as e:
                return path.name, [], f"pdf2image missing: {e}"
            images = convert_from_path(str(path), dpi=dpi)
            if max_pages:
                images = images[:max_pages]
            for img in images:
                pages.append(pytesseract.image_to_string(img, lang=tess_lang,
                                                         config=config))
        elif path.suffix.lower() in IMAGE_EXT:
            pages.append(pytesseract.image_to_string(Image.open(path),
                                                     lang=tess_lang, config=config))
        else:
            return path.name, [], "unsupported file type"
    except Exception as e:
        return path.name, [], f"{type(e).__name__}: {e}"

    return path.name, pages, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-name", default="manual_ocr.jsonl")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--psm", type=int, default=6,
                    help="Tesseract page segmentation mode; 6 (uniform block) "
                         "usually beats the default on book scans")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--text-layer-only", action="store_true",
                    help="skip files with no embedded text instead of running "
                         "OCR on them. pdftotext is ~50ms/doc; Tesseract is "
                         "3-10s/PAGE. Use when time is the constraint.")
    ap.add_argument("--max-pages-per-file", type=int, default=0,
                    help="0 = all pages")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--group-pages", type=int, default=1,
                    help="join N consecutive pages into one document; 3-5 gives "
                         "the tokenizer more context per record")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"[error] {in_dir} is not a directory", file=sys.stderr)
        return 1

    files = sorted([p for p in in_dir.rglob("*")
                    if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_EXT])
    if not files:
        print(f"[error] no PDFs or images under {in_dir}", file=sys.stderr)
        return 1

    print(f"[{args.lang}] OCR over {len(files)} files, {args.workers} workers")
    if args.lang == "nepali":
        print("  [note] using the `hin` Tesseract pack -- no Nepali pack exists.\n"
              "         Every record is flagged ocr_lang_fallback=true so you can\n"
              "         quantify the effect in the report.")

    out_path = root / args.lang / "data" / "raw" / args.out_name
    writer = ShardWriter(out_path)
    print(f"  output: {out_path} ({len(writer.seen):,} already collected)")

    kept = dropped = failed = 0
    drop_reasons: dict[str, int] = {}
    tasks = [(str(p), args.lang, args.dpi, args.psm, args.max_pages_per_file,
               args.text_layer_only)
             for p in files]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ocr_one_file, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            name, pages, err = fut.result()
            if err:
                failed += 1
                print(f"  [{i}/{len(files)}] {name}: FAILED — {err}")
                continue

            groups = []
            g = max(1, args.group_pages)
            for j in range(0, len(pages), g):
                groups.append("\n".join(pages[j:j + g]))

            f_kept = 0
            for k, raw in enumerate(groups):
                text = clean_ocr_text(raw)
                ok, why = page_quality_ok(text, min_chars=args.min_chars)
                if not ok:
                    dropped += 1
                    drop_reasons[why.split("(")[0]] = drop_reasons.get(why.split("(")[0], 0) + 1
                    continue
                doc = Document(
                    text=text, language=args.lang, provenance_class="manual",
                    source=f"ocr:{Path(name).stem}", collection_method="ocr",
                    title=Path(name).stem,
                    extra={"source_file": name, "page_group": k,
                           "pages_in_group": g, "dpi": args.dpi, "psm": args.psm,
                           "tesseract_lang": TESS_LANG[args.lang],
                           "ocr_lang_fallback": args.lang == "nepali"},
                )
                if writer.write(doc):
                    kept += 1
                    f_kept += 1
            print(f"  [{i}/{len(files)}] {name}: {len(pages)} pages -> {f_kept} docs")

    writer.close()
    print(f"\n  kept {kept:,} documents, dropped {dropped:,}, {failed} files failed")
    if drop_reasons:
        print(f"  drop reasons: {drop_reasons}")
    if dropped > kept:
        print("\n  [WARN] more pages dropped than kept. Usually one of:\n"
              "         wrong --psm, DPI too low (try 300-400), or scans that\n"
              "         Tesseract cannot handle at all. Inspect a few pages before\n"
              "         running the whole set.")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
