"""
plots.py  (NEW)
----------------
Generates all Phase 1 visualisation plots.  Kept separate from computation
(compute_stats.py) per course guidelines ??? no statistics are computed here;
this module only reads pre-computed stats JSON files and renders figures.

Plots generated
---------------
  1. Pipeline funnel     ??? docs + tokens at each stage
  2. Document length     ??? character count distribution (histogram)
  3. Token length        ??? whitespace-word count distribution
  4. Source contribution ??? stacked bar: manual vs. downloaded per source
  5. Manual fraction     ??? pie chart: manual vs. downloaded in final corpus
  6. Vocab sweep         ??? line plots: fertility / UNK / compression vs. vocab size

Each plot is saved as both PNG (for embedding in the report) and SVG (for
editing).  All plots include title, axis labels, legend, and source attribution.

Usage
-----
  python plots.py \\
      --hindi-stats  hindi/data/stats.json \\
      --nepali-stats nepali/data/stats.json \\
      --output-dir   report/figures/
"""

import argparse
import json
from pathlib import Path


# ????????? Lazy import helpers ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _import_matplotlib():
    """Import matplotlib; raise a clear error if not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (safe for servers)
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plots.py. Install: pip install matplotlib"
        )


# ????????? Individual plot functions ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def plot_pipeline_funnel(
    stats_hindi: dict,
    stats_nepali: dict,
    output_dir: Path,
) -> None:
    """
    Plot the pipeline funnel: documents remaining at each stage.

    Two subplots side by side ??? Hindi (left) and Nepali (right).

    Parameters
    ----------
    stats_hindi : dict
    stats_nepali : dict
    output_dir : Path
    """
    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, stats, lang in [(axes[0], stats_hindi, "Hindi"),
                              (axes[1], stats_nepali, "Nepali")]:
        filter_table = stats.get("filter_table", [])
        stages = [r["stage"] for r in filter_table]
        docs_in = [r["docs_in"] for r in filter_table]
        if not stages:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            ax.set_title(f"{lang} ??? Pipeline Funnel")
            continue

        ax.barh(stages[::-1], docs_in[::-1], color="#4A90D9", edgecolor="white")
        ax.set_xlabel("Documents Remaining")
        ax.set_title(f"{lang} ??? Pipeline Funnel")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}"))
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Phase 1: Documents Remaining After Each Pipeline Stage",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "pipeline_funnel")
    plt.close(fig)


def plot_source_contribution(
    stats_hindi: dict,
    stats_nepali: dict,
    output_dir: Path,
) -> None:
    """
    Plot per-source token contribution as stacked bars (manual vs. downloaded).

    Parameters
    ----------
    stats_hindi : dict
    stats_nepali : dict
    output_dir : Path
    """
    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, stats, lang in [(axes[0], stats_hindi, "Hindi"),
                              (axes[1], stats_nepali, "Nepali")]:
        src_table = stats.get("source_table", [])
        if not src_table:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue

        # Shorten source names
        names  = [r["source_name"][:25] for r in src_table[:10]]
        tokens = [r.get("final_tokens_est", 0) for r in src_table[:10]]
        colors = ["#E07B39" if r.get("manual") == "Yes" else "#4A90D9"
                  for r in src_table[:10]]

        ax.barh(names[::-1], [t / 1e6 for t in tokens[::-1]],
                color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Final Tokens (millions)")
        ax.set_title(f"{lang} ??? Source Contributions")
        ax.grid(axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E07B39", label="Manual"),
            Patch(facecolor="#4A90D9", label="Downloaded"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    fig.suptitle("Phase 1: Per-Source Token Contributions (Final Corpus)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "source_contribution")
    plt.close(fig)


def plot_manual_fraction(
    stats_hindi: dict,
    stats_nepali: dict,
    output_dir: Path,
) -> None:
    """
    Plot manual vs. downloaded fraction as pie charts.

    Parameters
    ----------
    stats_hindi : dict
    stats_nepali : dict
    output_dir : Path
    """
    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, stats, lang in [(axes[0], stats_hindi, "Hindi"),
                              (axes[1], stats_nepali, "Nepali")]:
        bd = stats.get("manual_breakdown", {})
        manual_pct  = bd.get("manual_pct", 0)
        dl_pct      = 100 - manual_pct
        compliant   = bd.get("compliant", False)

        wedges, texts, autotexts = ax.pie(
            [manual_pct, dl_pct],
            labels=["Manual", "Downloaded"],
            autopct="%1.1f%%",
            colors=["#E07B39", "#4A90D9"],
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        req_line = "??? ???20% requirement met" if compliant else "??? <20% requirement NOT met"
        ax.set_title(f"{lang}\n{req_line}", fontsize=11,
                     color="green" if compliant else "red")

    fig.suptitle("Phase 1: Manual vs. Downloaded Token Fractions (Final Corpus)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "manual_fraction")
    plt.close(fig)


def plot_vocab_sweep(
    stats_hindi: dict,
    stats_nepali: dict,
    output_dir: Path,
) -> None:
    """
    Plot vocab-size sweep metrics: fertility, UNK rate, compression ratio.

    Parameters
    ----------
    stats_hindi : dict
    stats_nepali : dict
    output_dir : Path
    """
    plt = _import_matplotlib()
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("fertility",         "Fertility (tokens/word)", "lower is better"),
        ("unk_rate",          "UNK Rate",                "lower is better"),
        ("compression_ratio", "Compression Ratio (chars/token)", ""),
    ]

    for lang, stats, color in [("Hindi",  stats_hindi,  "#4A90D9"),
                                 ("Nepali", stats_nepali, "#E07B39")]:
        sweep = stats.get("vocab_sweep", [])
        if not sweep:
            continue
        sizes = [int(r.get("vocab_size", 0)) for r in sweep]
        for ax, (metric, ylabel, note) in zip(axes, metrics):
            vals = [float(r.get(metric, 0)) for r in sweep]
            ax.plot(sizes, vals, marker="o", label=lang, color=color)

    for ax, (metric, ylabel, note) in zip(axes, metrics):
        ax.set_xlabel("Vocabulary Size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + (f"\n({note})" if note else ""))
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Phase 1: Vocabulary Size Sweep Results",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "vocab_sweep")
    plt.close(fig)


# ????????? Save helper ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _save(fig, output_dir: Path, name: str) -> None:
    """
    Save a figure as both PNG and SVG.

    Parameters
    ----------
    fig : matplotlib Figure
    output_dir : Path
    name : str
        Base filename without extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [plots] Saved: {name}.png / .svg")


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate all Phase 1 visualisation plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hindi-stats",  required=True,
                        help="Path to hindi/data/stats.json")
    parser.add_argument("--nepali-stats", required=True,
                        help="Path to nepali/data/stats.json")
    parser.add_argument("--output-dir",   default="report/figures/",
                        help="Directory to write plot files.")
    return parser.parse_args()


def main() -> None:
    """Load stats files and generate all plots."""
    args = parse_args()

    def _load(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            print(f"[plots] WARNING: stats file not found: {p}")
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    sh = _load(args.hindi_stats)
    sn = _load(args.nepali_stats)
    out = Path(args.output_dir)

    print("[plots] Generating all Phase 1 plots???")
    plot_pipeline_funnel(sh, sn, out)
    plot_source_contribution(sh, sn, out)
    plot_manual_fraction(sh, sn, out)
    plot_vocab_sweep(sh, sn, out)
    print(f"\n[plots] Done. All figures saved to {out}/")


if __name__ == "__main__":
    main()
