"""measure_phase2_corpus_tokens.py — the REAL train/val/test token counts of
whatever corpus a Phase 2 run actually consumed, read from the cached token
arrays pipeline/train/dataset.py writes next to each split
(`<split>.jsonl.<tokenizer-stem>.tokens.npy`).

Exists because the Phase 2 reports previously hardcoded the Phase 1 handoff
document's claimed final corpus sizes (559.9M Hindi / 501.2M Nepali train
tokens) instead of measuring what was actually tokenized and trained on.
The two turned out to differ by 5-15%: the GCS `work/<lang>/data/splits/`
copies synced onto the training VM were an earlier pipeline pass, not the
literal final rebuild the handoff quotes. This script closes that gap by
reading the real numbers a real run produced, not a document's claim.

    python tools/measure_phase2_corpus_tokens.py --repo-root .

Uses mmap (not a full load) so multi-GB token arrays are read in
milliseconds — only the .npy header is touched, not the token data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

LANGS = ("hindi", "nepali")
SPLITS = ("train", "val", "test")


def cache_path(splits_dir: Path, split: str, tokenizer_model: Path) -> Path:
    return splits_dir / f"{split}.jsonl.{tokenizer_model.stem}.tokens.npy"


def measure(root: Path, lang: str) -> dict | None:
    cfg_path = root / lang / "configs" / "model_config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tcfg = cfg["training"]
    splits_dir = root / lang / "data" / "splits"
    tokenizer_model = root / tcfg["tokenizer_model"]

    counts = {}
    for split in SPLITS:
        p = cache_path(splits_dir, split, tokenizer_model)
        if not p.exists():
            counts[split] = None
            continue
        arr = np.load(p, mmap_mode="r")  # header-only read, not a full load
        counts[split] = int(arr.shape[0])

    if any(v is None for v in counts.values()):
        return {"tokens": counts, "total": None, "note": "one or more splits not yet tokenized/cached"}

    total = sum(counts.values())
    return {"tokens": counts, "total": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    for lang in LANGS:
        result = measure(root, lang)
        out = root / lang / "data" / "stats" / "phase2_corpus_token_counts.json"
        if result is None:
            print(f"  {lang}: no model_config.yaml, skipped")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if result.get("total") is not None:
            print(f"  {lang}: train={result['tokens']['train']:,} val={result['tokens']['val']:,} "
                  f"test={result['tokens']['test']:,} total={result['total']:,} -> {out}")
        else:
            print(f"  {lang}: incomplete ({result['tokens']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
