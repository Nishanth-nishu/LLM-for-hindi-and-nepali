"""
make_variant.py — a second corpus build from the same collected data
=====================================================================
Sets up a parallel tree that SHARES the raw collected documents but keeps its
own interim files, splits, statistics and tokenizer. That lets you build the
corpus two different ways and compare, without recollecting anything and
without the two runs overwriting each other.

    python tools/make_variant.py --lang hindi --name nocap

    # then point any stage at it
    python -m pipeline.process.build_corpus --lang hindi --repo-root variants/nocap \\
        --min-manual-fraction 0.0
    python pipeline/tokenizer/train_tokenizer.py --lang hindi \\
        --repo-root variants/nocap --vocab-sizes 16000

WHY A SYMLINK AND NOT A COPY
----------------------------
`<lang>/data/raw/` is tens of gigabytes and is still being appended to by the
collectors. Copying it would waste the disk and freeze a stale snapshot; the
symlink means both variants read the same live files. Everything a build
*writes* -- interim/, splits/, stats/, tokenizer/ -- is a real directory in the
variant, so the two never collide.

WHAT THIS IS FOR, AND WHAT IT IS NOT
-------------------------------------
Use it for an ablation: build once respecting the >=20% manual requirement and
once ignoring it, then compare tokenizer fertility and byte-fallback on matched
vocabulary sizes. That answers "does the manual fraction change tokenizer
quality?" with evidence rather than assertion.

The variant that ignores the ratio is NOT a submittable corpus. It fails the
graded requirement by construction. Keep the ratio-respecting build as the
deliverable and cite the variant as a comparison.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--name", required=True,
                    help="variant directory name, e.g. 'nocap'")
    ap.add_argument("--variants-dir", default="variants")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    src_lang = root / args.lang
    if not (src_lang / "data" / "raw").is_dir():
        print(f"[error] {src_lang / 'data' / 'raw'} not found — collect first.",
              file=sys.stderr)
        return 1

    dst_root = root / args.variants_dir / args.name
    dst_lang = dst_root / args.lang
    (dst_lang / "data").mkdir(parents=True, exist_ok=True)

    # shared, read-only in practice
    raw_link = dst_lang / "data" / "raw"
    if raw_link.is_symlink() or raw_link.exists():
        if raw_link.is_symlink():
            raw_link.unlink()
        else:
            print(f"[error] {raw_link} exists and is not a symlink; refusing "
                  f"to touch it.", file=sys.stderr)
            return 1
    raw_link.symlink_to(src_lang / "data" / "raw")

    # private to the variant
    for sub in ("interim", "splits", "stats", "quality"):
        (dst_lang / "data" / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("vocab", "analysis"):
        (dst_lang / "tokenizer" / sub).mkdir(parents=True, exist_ok=True)
    (dst_root / "report").mkdir(parents=True, exist_ok=True)

    # configs are copied, not linked: a variant usually wants to change one
    cfg_dst = dst_lang / "configs"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for f in (src_lang / "configs").glob("*"):
        if f.is_file():
            shutil.copy2(f, cfg_dst / f.name)

    print(f"variant ready: {dst_root}")
    print(f"  raw   -> symlink to {src_lang / 'data' / 'raw'}  (shared, live)")
    print(f"  own   -> interim/ splits/ stats/ tokenizer/ configs/ report/")
    print()
    print(f"  build it:")
    print(f"    python -m pipeline.process.build_corpus --lang {args.lang} \\")
    print(f"        --repo-root {dst_root.relative_to(root)} --workers 6 "
          f"--min-manual-fraction 0.0")
    print(f"  train on it:")
    print(f"    python pipeline/tokenizer/train_tokenizer.py --lang {args.lang} \\")
    print(f"        --repo-root {dst_root.relative_to(root)} --vocab-sizes 16000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
