"""
ingest_existing.py — fold already-downloaded corpora into the pipeline
======================================================================
You already ran a bulk download and have zips / JSONL sitting in Drive. This
converts them into manifest records so the rest of the pipeline can use them,
without re-downloading anything.

WHAT PROVENANCE CLASS THIS ASSIGNS, AND WHY
-------------------------------------------
`downloaded`. Always, by default.

The brief separates two categories explicitly:

    "You may use public corpora (e.g. Hugging Face), BUT for both languages at
     least 20% of the final training tokens must come from manual collection
     (e.g. OCR from books/PDFs, scraping + cleaning pages you gather yourself,
     typed/transcribed text)."

Anything a download script fetched from HuggingFace -- Sangraha, IndicCorpV2,
OSCAR, mC4, CC-100, Wikipedia, OPUS, Samanantar -- is the first category. It
does not become manual by being renamed, moved into a folder called `manual`,
or relabelled at ingest. The `source` field travels with every record, so the
provenance is inspectable in the final corpus regardless of the label.

This is not a technicality to route around. It is the requirement.

IF SOME OF THIS DATA GENUINELY IS YOURS
---------------------------------------
If part of what you have really was scraped or OCR'd by you, pass
`--provenance manual --manual-justification "..."`. The justification is stored
verbatim in every record it applies to and surfaces in the dataset report, so
the claim is documented rather than asserted. The tool still warns if the
filename looks like a known public corpus, because that is the case worth
double-checking.

Usage
-----
  # extract your Drive zips somewhere first, then:
  python -m pipeline.collect.ingest_existing --lang hindi --repo-root . \\
      --input-dir /path/to/extracted/hindi

  # inspect without writing anything
  python -m pipeline.collect.ingest_existing --lang hindi --repo-root . \\
      --input-dir /path/to/extracted/hindi --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import Document, ShardWriter  # noqa: E402

# Filename fragments that identify a known public corpus. Matching one of these
# is strong evidence the data was downloaded, not collected.
PUBLIC_CORPUS_MARKERS = {
    "sangraha": "ai4bharat/sangraha",
    "indiccorp": "ai4bharat/IndicCorpV2",
    "indic_corp": "ai4bharat/IndicCorpV2",
    "oscar": "oscar-corpus/OSCAR",
    "mc4": "allenai/c4 (multilingual)",
    "c4": "allenai/c4",
    "cc100": "cc100",
    "cc_100": "cc100",
    "wikipedia": "wikimedia/wikipedia",
    "wiki": "wikimedia/wikipedia",
    "opus": "OPUS parallel corpora",
    "samanantar": "ai4bharat/samanantar",
    "culturax": "uonlp/CulturaX",
}

TEXT_KEYS = ("text", "content", "raw_content", "body", "article", "sentence")


def detect_source(path: Path) -> tuple[str, str | None]:
    """Returns (source_label, public_corpus_repo_or_None)."""
    hay = str(path).lower().replace("-", "_")
    for marker, repo in PUBLIC_CORPUS_MARKERS.items():
        if marker in hay:
            return marker, repo
    return path.stem[:60], None


def extract_text(rec: dict) -> str:
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def iter_records(path: Path):
    """Yield dicts from .jsonl, .jsonl.gz, .json, or members of a .zip."""
    suf = path.suffix.lower()
    if suf == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    low = name.lower()
                    if not any(low.endswith(e) for e in
                               (".jsonl", ".json", ".jsonl.gz", ".txt")):
                        continue
                    with zf.open(name) as fh:
                        raw = fh.read()
                    if low.endswith(".gz"):
                        try:
                            raw = gzip.decompress(raw)
                        except OSError:
                            continue
                    text = raw.decode("utf-8", errors="replace")
                    if low.endswith(".txt"):
                        for line in text.splitlines():
                            if line.strip():
                                yield {"text": line, "_member": name}
                    else:
                        for line in text.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                d = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(d, dict):
                                d["_member"] = name
                                yield d
        except zipfile.BadZipFile:
            print(f"  [skip] {path.name}: not a readable zip")
        return

    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:   # type: ignore[operator]
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    yield d
    except OSError as e:
        print(f"  [skip] {path.name}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input-dir", required=True,
                    help="directory of zips / jsonl from your existing download")
    ap.add_argument("--provenance", default="downloaded",
                    choices=["downloaded", "manual"])
    ap.add_argument("--manual-justification", default=None,
                    help="required with --provenance manual: how YOU collected it")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what is there, write nothing")
    args = ap.parse_args()

    if args.provenance == "manual" and not args.manual_justification:
        print("[error] --provenance manual requires --manual-justification.\n"
              "        Describe how YOU collected it (scraped which sites, OCR'd\n"
              "        which books). It is stored in every record and shown in the\n"
              "        report, so the claim is documented rather than asserted.\n"
              "        If it came from a download script, it is 'downloaded'.",
              file=sys.stderr)
        return 1

    root = Path(args.repo_root).resolve()
    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"[error] {in_dir} is not a directory", file=sys.stderr)
        return 1

    files = sorted([p for p in in_dir.rglob("*")
                    if p.is_file() and (p.suffix.lower() in (".zip", ".jsonl", ".json")
                                        or p.name.endswith(".jsonl.gz"))])
    if not files:
        print(f"[error] nothing ingestible under {in_dir}", file=sys.stderr)
        return 1

    print(f"[{args.lang}] scanning {len(files)} files under {in_dir}\n")

    # ---- provenance sanity check ------------------------------------------
    flagged: list[tuple[Path, str]] = []
    for p in files:
        _, repo = detect_source(p)
        if repo:
            flagged.append((p, repo))

    if flagged:
        print("  files matching a known PUBLIC CORPUS:")
        for p, repo in flagged[:20]:
            print(f"    {p.name:<45} -> {repo}")
        if len(flagged) > 20:
            print(f"    ... and {len(flagged) - 20} more")
        print()

    if args.provenance == "manual" and flagged:
        print("  " + "=" * 68)
        print("  WARNING: you are labelling these `manual`, but the filenames match")
        print("  public corpora that a download script fetches from HuggingFace.")
        print()
        print("  The brief contrasts the two categories explicitly: public corpora")
        print("  are what the <=80% is for; manual collection means OCR from")
        print("  books/PDFs, pages you scraped and cleaned yourself, or text you")
        print("  typed or transcribed.")
        print()
        print("  Downloaded data does not become manual by being relabelled, and")
        print("  the `source` field stays in every record either way.")
        print("  " + "=" * 68)
        print()

    # ---- ingest ------------------------------------------------------------
    out_name = ("manual_imported.jsonl" if args.provenance == "manual"
                else "downloaded_imported.jsonl")
    out_path = root / args.lang / "data" / "raw" / out_name
    writer = None if args.dry_run else ShardWriter(out_path)

    by_source: Counter[str] = Counter()
    chars_by_source: Counter[str] = Counter()
    kept = skipped = 0

    for p in files:
        source, repo = detect_source(p)
        n_file = 0
        for rec in iter_records(p):
            text = extract_text(rec)
            if len(text.strip()) < args.min_chars:
                skipped += 1
                continue
            by_source[source] += 1
            chars_by_source[source] += len(text)
            n_file += 1
            if writer is not None:
                extra = {"imported_from": p.name}
                if repo:
                    extra["public_corpus"] = repo
                if args.manual_justification:
                    extra["manual_collection_method"] = args.manual_justification
                doc = Document(
                    text=text.strip(), language=args.lang,
                    provenance_class=args.provenance, source=source,
                    collection_method=("scrape" if args.provenance == "manual"
                                       else "hf_download"),
                    extra=extra)
                if writer.write(doc):
                    kept += 1
        print(f"  {p.name:<45} {n_file:>9,} documents"
              + (f"   [{repo}]" if repo else ""))

    if writer is not None:
        writer.close()

    total_docs = sum(by_source.values())
    total_chars = sum(chars_by_source.values())
    print(f"\n  {total_docs:,} documents, {total_chars / 1e6:.1f}M characters "
          f"({skipped:,} skipped as too short)")
    if writer is not None:
        print(f"  wrote {kept:,} unique -> {out_path}")
    else:
        print("  (dry run — nothing written)")

    # ---- what this means for the requirement -------------------------------
    est_tokens = total_chars / 4.0     # rough Devanagari proxy
    print(f"\n  rough token estimate: {est_tokens / 1e6:.0f}M "
          f"(at ~4 chars/token; the real number needs your tokenizer)")

    if args.provenance == "downloaded":
        need_manual = est_tokens * 0.25   # 20% of total means 25% of downloaded
        print(f"\n  --- what you still need ---")
        print(f"  This is the DOWNLOADED side. To reach >=20% manual overall you")
        print(f"  need about {need_manual / 1e6:.0f}M manual tokens alongside it,")
        print(f"  OR you cap the downloaded share lower. Run:")
        print(f"      python -m pipeline.collect.plan_budget --lang {args.lang}")
        print(f"  and then collect the manual side:")
        print(f"      python run_phase1.py --lang {args.lang} --stage discover,scrape")

    return 0


if __name__ == "__main__":
    sys.exit(main())
