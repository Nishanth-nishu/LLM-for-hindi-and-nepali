#!/usr/bin/env python3
"""
download_drive_datasets.py
----------------------------
Restore the already-collected Phase 1 datasets from Google Drive instead of
re-running live HuggingFace streaming collection.

Downloads two tar archives via gdown:

  Hindi  : LMA_Phase1_Hindi_2026-08-13.tar   (~2.65 GB)  Drive ID 1GwFqH8a2rpQWuNmNF6Y-A7-m-Wn7LlhS
  Nepali : LMA_Phase1_Nepali_2026-08-13.tar  (~2.35 GB)  Drive ID 1aaatAK0euWFVrrt6sRmbKADqFjMWqv_X

Each archive is extracted directly into <lang>/data/ under the project root
(e.g. project/hindi/data/, project/nepali/data/), which is where every other
pipeline stage (download_public.py, the cleaning stages, split_data.py,
train_tokenizer.py, compute_stats.py, audit_phase1.py, ...) already expects
to find manifest.csv, raw/, cleaned/, dedup/, splits/, etc.

Resumable / idempotent
-----------------------
  - Skips the download if the tar already exists locally and its size matches
    the expected size on Drive (re-downloads if it looks truncated/corrupt).
  - Skips extraction if <lang>/data/manifest.csv already exists, unless
    --force is given.
  - Archives are deleted after a successful extraction by default (they're
    multi-GB and their content is now sitting in <lang>/data/); pass
    --keep-archive to retain them.

Usage
-----
  python scripts/download_drive_datasets.py --repo-root project
  python scripts/download_drive_datasets.py --repo-root project --lang hindi
  python scripts/download_drive_datasets.py --repo-root project --force --keep-archive

Requires
--------
  pip install gdown
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

# ????????? Dataset registry ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

DATASETS: dict[str, dict] = {
    "hindi": {
        "drive_id": "1GwFqH8a2rpQWuNmNF6Y-A7-m-Wn7LlhS",
        "filename": "LMA_Phase1_Hindi_2026-08-13.tar",
        "expected_size_bytes": int(2.65 * 1024 ** 3),
        # Drive/gdown sizes aren't exact-byte-perfect, so we accept anything
        # within this fraction of the expected size as "looks complete".
        "size_tolerance": 0.03,
    },
    "nepali": {
        "drive_id": "1aaatAK0euWFVrrt6sRmbKADqFjMWqv_X",
        "filename": "LMA_Phase1_Nepali_2026-08-13.tar",
        "expected_size_bytes": int(2.35 * 1024 ** 3),
        "size_tolerance": 0.03,
    },
}


# ????????? Helpers ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _looks_complete(path: Path, expected_bytes: int, tolerance: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    actual = path.stat().st_size
    return abs(actual - expected_bytes) <= expected_bytes * tolerance


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def download_archive(lang: str, cfg: dict, download_dir: Path, force: bool) -> Path:
    """Download one language's tar archive via gdown, skipping if already present."""
    try:
        import gdown
    except ImportError:
        print(
            "[download_drive_datasets] ERROR: gdown is not installed.\n"
            "  Install it with: pip install gdown",
            file=sys.stderr,
        )
        sys.exit(1)

    download_dir.mkdir(parents=True, exist_ok=True)
    dest = download_dir / cfg["filename"]

    if not force and _looks_complete(dest, cfg["expected_size_bytes"], cfg["size_tolerance"]):
        print(f"[download_drive_datasets] [{lang}] Archive already present and looks "
              f"complete ({_human(dest.stat().st_size)}) ??? skipping download: {dest}")
        return dest

    if dest.exists():
        print(f"[download_drive_datasets] [{lang}] Existing archive looks incomplete "
              f"or --force set ??? re-downloading.")
        dest.unlink()

    print(f"[download_drive_datasets] [{lang}] Downloading {cfg['filename']} "
          f"(~{_human(cfg['expected_size_bytes'])}) from Drive ID {cfg['drive_id']} ???")
    try:
        gdown.download(id=cfg["drive_id"], output=str(dest), quiet=False)
    except Exception as e:
        print(f"[download_drive_datasets] ERROR: gdown download failed for {lang}: {e}",
              file=sys.stderr)
        sys.exit(1)

    if not dest.exists():
        print(f"[download_drive_datasets] ERROR: download reported success but "
              f"{dest} is missing.", file=sys.stderr)
        sys.exit(1)

    if not _looks_complete(dest, cfg["expected_size_bytes"], cfg["size_tolerance"]):
        actual = dest.stat().st_size
        print(
            f"[download_drive_datasets] WARNING: [{lang}] downloaded size "
            f"{_human(actual)} differs from expected {_human(cfg['expected_size_bytes'])} "
            f"by more than {cfg['size_tolerance']*100:.0f}%. The archive may be truncated "
            f"or Drive served an HTML quota-warning page instead of the file. Inspect it "
            f"before relying on the extracted data (e.g. `tar -tvf {dest}` should list "
            f"real entries, not fail immediately).",
            file=sys.stderr,
        )

    print(f"[download_drive_datasets] [{lang}] Downloaded {_human(dest.stat().st_size)} ??? {dest}")
    return dest


def extract_archive(lang: str, archive_path: Path, lang_data_dir: Path, force: bool) -> None:
    """Extract a tar archive directly into <lang>/data/."""
    marker = lang_data_dir / "manifest.csv"
    if not force and marker.exists():
        print(f"[download_drive_datasets] [{lang}] {marker} already exists ??? "
              f"skipping extraction (pass --force to re-extract).")
        return

    lang_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_drive_datasets] [{lang}] Extracting {archive_path.name} ??? {lang_data_dir} ???")

    try:
        with tarfile.open(archive_path, "r:*") as tar:
            # Guard against path-traversal / absolute-path members in the tar.
            def _is_within(base: Path, target: Path) -> bool:
                try:
                    target.resolve().relative_to(base.resolve())
                    return True
                except ValueError:
                    return False

            safe_members = []
            for member in tar.getmembers():
                member_path = lang_data_dir / member.name
                if not _is_within(lang_data_dir, member_path):
                    print(f"[download_drive_datasets] WARNING: skipping unsafe tar "
                          f"member outside target dir: {member.name}", file=sys.stderr)
                    continue
                safe_members.append(member)
            tar.extractall(path=lang_data_dir, members=safe_members)
    except tarfile.TarError as e:
        print(f"[download_drive_datasets] ERROR: failed to extract {archive_path} "
              f"for {lang}: {e}", file=sys.stderr)
        print(f"[download_drive_datasets] Hint: if the archive size warning fired above, "
              f"the file is probably not a valid tar (e.g. an HTML quota page saved with "
              f"a .tar name). Delete {archive_path} and re-run.", file=sys.stderr)
        sys.exit(1)

    if not marker.exists():
        print(f"[download_drive_datasets] WARNING: [{lang}] extraction finished but "
              f"{marker} was not found. Check whether the archive's internal layout "
              f"nests an extra top-level folder (e.g. 'data/manifest.csv' instead of "
              f"'manifest.csv') and adjust accordingly.", file=sys.stderr)
    else:
        print(f"[download_drive_datasets] [{lang}] ??? Extraction complete, manifest found.")


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore Hindi/Nepali Phase 1 datasets from Google Drive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=".",
                         help="Path to the 'project' directory (contains hindi/, nepali/).")
    parser.add_argument("--lang", choices=["hindi", "nepali", "all"], default="all")
    parser.add_argument("--download-dir", default=None,
                         help="Where to save the .tar archives before extraction "
                              "(default: <repo-root>/_drive_archives).")
    parser.add_argument("--keep-archive", action="store_true",
                         help="Don't delete the .tar file after successful extraction.")
    parser.add_argument("--force", action="store_true",
                         help="Re-download and re-extract even if data already looks present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    download_dir = Path(args.download_dir).resolve() if args.download_dir else repo_root / "_drive_archives"

    langs = ["hindi", "nepali"] if args.lang == "all" else [args.lang]

    print("[download_drive_datasets] === Restoring Phase 1 datasets from Google Drive ===")
    print(f"  repo-root     : {repo_root}")
    print(f"  download-dir  : {download_dir}")
    print(f"  languages     : {', '.join(langs)}")
    print(f"  keep-archive  : {args.keep_archive}")
    print(f"  force         : {args.force}")

    for lang in langs:
        cfg = DATASETS[lang]
        lang_data_dir = repo_root / lang / "data"

        print(f"\n[download_drive_datasets] --- {lang} ---")
        archive_path = download_archive(lang, cfg, download_dir, args.force)
        extract_archive(lang, archive_path, lang_data_dir, args.force)

        if not args.keep_archive and archive_path.exists():
            print(f"[download_drive_datasets] [{lang}] Removing archive to save disk "
                  f"space: {archive_path} (pass --keep-archive to retain it).")
            archive_path.unlink()

    print("\n[download_drive_datasets] === Done ===")
    for lang in langs:
        manifest = repo_root / lang / "data" / "manifest.csv"
        status = "found" if manifest.exists() else "MISSING"
        print(f"  {lang}: manifest.csv {status} ({manifest})")
    print(
        "\n  Next: run common/audit_phase1.py for each language to confirm the "
        "500M-token / 20%-manual requirements are met in the restored data, e.g.\n"
        "    python common/audit_phase1.py --lang hindi  --repo-root <repo-root>\n"
        "    python common/audit_phase1.py --lang nepali --repo-root <repo-root>"
    )


if __name__ == "__main__":
    main()
