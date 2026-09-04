"""make_phase2_figures.py — loss-curve plots from training_log.jsonl.

    python tools/make_phase2_figures.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANGS = ("hindi", "nepali")


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot_loss_curve(log: list[dict], lang: str, out_path: Path) -> None:
    steps = [r["step"] for r in log]
    train_loss = [r["train_loss"] for r in log]
    val_loss = [r["val_loss"] for r in log]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, train_loss, label="train loss", marker="o", markersize=3)
    ax.plot(steps, val_loss, label="val loss", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(f"{lang.title()} — training/validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    for lang in LANGS:
        log = load_log(root / lang / "checkpoints" / "training_log.jsonl")
        out = root / "report" / "figures" / "phase2" / lang / "loss_curve.png"
        if not log:
            print(f"  {lang}: no training_log.jsonl yet, skipped")
            continue
        plot_loss_curve(log, lang, out)
        print(f"  wrote {out.relative_to(root)} ({len(log)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
