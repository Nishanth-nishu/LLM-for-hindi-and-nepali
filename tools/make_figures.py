"""
make_figures.py — the four figures the Phase 1 reports reference
================================================================
Reads the same statistics files `make_reports.py` reads and writes PNG + SVG
into `report/figures/`. Run it before `make_reports.py`:

    python tools/make_figures.py --repo-root .

WHY THESE FOUR
--------------
A figure earns its place by answering a question faster than the table does.
Four questions qualify:

1. `corpus_composition`  — is the >=20% manual requirement met, and by how much?
   A 100% stacked bar with the requirement drawn on it answers in one glance
   what a table answers in two subtractions.

2. `vocab_tradeoff`      — why 4,000 and not 32,000?
   Fertility keeps improving with vocabulary size while embedding cost explodes.
   These are two measures on incomparable scales, so they are TWO STACKED PANELS
   sharing one x-axis, never one plot with two y-axes. A dual-axis chart lets
   the author place the crossing point anywhere by choosing the scales, which
   is precisely the claim the figure is supposed to support.

3. `cleaning_funnel`     — what did cleaning actually remove?
   Magnitude across named categories spanning five orders of magnitude, so:
   horizontal bars, sorted, log axis, one series and therefore no legend.

4. `zipf`                — does the token distribution look like language?
   Rank against frequency on log-log. Natural language is near-linear with a
   slope around -1; a flat or kinked curve says the tokenizer is fragmenting.

COLOR
-----
Two categorical slots, blue and orange, assigned to (manual, downloaded) and to
(hindi, nepali) in fixed order and never cycled. Validated: worst adjacent CVD
dE 24.7, normal-vision dE 33.6, both clear of the floors, both >= 3:1 on the
chart surface. Text is ink, never the series color; the colored mark beside a
label carries identity.

Figures are rendered on their own light surface rather than transparent, so
they stay legible in a dark-themed GitHub view instead of turning into dark
text on a dark page.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

LANGS = ("hindi", "nepali")

# --- design tokens (see references/palette.md; swap here to re-theme) --------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
S1 = "#2a78d6"   # categorical slot 1 — manual / hindi
S2 = "#eb6834"   # categorical slot 2 — downloaded / nepali

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def millions(x, _pos=None):
    return f"{x / 1e6:.0f}M"


def save(fig, out_dir: Path, name: str, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_dir.name}/{name}.png  + .svg")


# ---------------------------------------------------------------------------
# 1. corpus composition
# ---------------------------------------------------------------------------

def fig_corpus_composition(data, out_dir, log=print) -> bool:
    rows = []
    for lang in LANGS:
        t = data[lang]["tokens"]
        if not t:
            continue
        mv = t["manual_vs_downloaded"]
        rows.append((lang.title(), mv["manual_tokens"], mv["downloaded_tokens"],
                     t["totals"]["corpus_tokens_all_splits"]))
    if not rows:
        log("  [skip] corpus_composition — no token_accounting.json")
        return False

    fig, ax = plt.subplots(figsize=(8.5, 1.5 + 0.62 * len(rows)))
    y = range(len(rows))
    for i, (name, man, dl, tot) in enumerate(rows):
        mp, dp = 100 * man / tot, 100 * dl / tot
        # 2px surface gap between adjacent fills: draw the second segment
        # inset rather than butted against the first.
        ax.barh(i, mp, height=0.46, color=S1, zorder=3)
        ax.barh(i, dp - 0.4, left=mp + 0.4, height=0.46, color=S2, zorder=3)
        ax.text(mp / 2, i, f"{mp:.2f}%", ha="center", va="center",
                color="#ffffff", fontsize=11, fontweight="bold", zorder=4)
        ax.text(101.5, i, f"{tot / 1e6:.1f}M tokens", ha="left", va="center",
                color=INK_2, fontsize=10)

    ax.axvline(20, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=5)
    # Label the line inside the plot, in the gap between bars. Above the axes it
    # collides with the title; below them it reads as an x-axis label. The gap
    # is the only place it is unambiguously attached to the line.
    ax.text(19.2, (len(rows) - 1) / 2, "20% requirement", ha="right",
            va="center", color=INK, fontsize=9.5, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=11)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("Corpus composition — manual share of MEASURED tokens",
                 loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=26)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S1),
               plt.Rectangle((0, 0), 1, 1, color=S2)]
    ax.legend(handles, ["Manually collected", "Downloaded (Sangraha, Wikipedia)"],
              loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, frameon=False,
              fontsize=10, labelcolor=INK_2, handlelength=1.1, handleheight=1.1)
    save(fig, out_dir, "corpus_composition", log)
    return True


# ---------------------------------------------------------------------------
# 2. vocabulary trade-off
# ---------------------------------------------------------------------------

def _sweep_rows(root: Path, lang: str) -> list[dict]:
    """Prefer the sweep CSV — it holds every point. Fall back to the candidate
    list inside vocab_selection.json, which holds the same rows for the sizes
    that were actually swept."""
    csv_path = root / lang / "tokenizer" / "analysis" / "sweep_results.csv"
    alt = root / lang / "tokenizer" / "analysis" / "sweep_results_selection.csv"
    for p in (alt, csv_path):          # the *_selection copy is the full sweep
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) > 1:
                return rows
    v = load(root / lang / "tokenizer" / "analysis" / "vocab_selection.json")
    return (v or {}).get("candidates", [])


def fig_vocab_tradeoff(root, data, out_dir, d_model=512, budget=25_000_000,
                       log=print) -> bool:
    series = {}
    for lang in LANGS:
        rows = _sweep_rows(root, lang)
        pts = []
        for r in rows:
            try:
                v = int(float(r["vocab_size"]))
                f = float(r["fertility"])
            except (KeyError, TypeError, ValueError):
                continue
            pts.append((v, f))
        if len(pts) > 1:
            series[lang] = sorted(set(pts))
    if not series:
        log("  [skip] vocab_tradeoff — no sweep results")
        return False

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})

    for lang, colour in ((("hindi"), S1), (("nepali"), S2)):
        if lang not in series:
            continue
        xs = [p[0] for p in series[lang]]
        ys = [p[1] for p in series[lang]]
        ax1.plot(xs, ys, color=colour, lw=2, marker="o", ms=8,
                 markeredgecolor=SURFACE, markeredgewidth=2, label=lang.title())
        # Direct-label at the LEFT end. The two curves converge at the largest
        # vocabulary, so labels there sit on top of each other; at the smallest
        # they are 0.18 fertility apart.
        ax1.annotate(lang.title(), (xs[0], ys[0]), textcoords="offset points",
                     xytext=(11, 0), color=INK_2, fontsize=10, ha="left",
                     va="center")

    all_v = sorted({v for s in series.values() for v, _ in s})
    shares = [100 * v * d_model / budget for v in all_v]
    ax2.plot(all_v, shares, color=MUTED, lw=2, marker="o", ms=8,
             markeredgecolor=SURFACE, markeredgewidth=2)
    for v, s in zip(all_v, shares):
        ax2.annotate(f"{s:.1f}%", (v, s), textcoords="offset points",
                     xytext=(0, 10), ha="center", color=INK_2, fontsize=9.5)
    ax2.axhline(35, color=INK, lw=1.4, ls=(0, (4, 3)))
    # Under the rule, right-aligned: the point annotations sit above their
    # markers, and 16,000 lands at 32.8% right where a left-aligned caption on
    # top of the rule would be.
    ax2.text(max(all_v) * 1.45, 31.5,
             "35% — above this, vocabulary is bought with depth",
             color=INK, fontsize=9, va="top", ha="right")

    sel = None
    for lang in LANGS:
        v = data[lang].get("vocab")
        if v and v.get("selected_vocab_size"):
            sel = v["selected_vocab_size"]
            break
    if sel:
        for ax in (ax1, ax2):
            ax.axvline(sel, color=INK, lw=1, alpha=0.35, zorder=0)
        lo, hi = ax1.get_ylim()
        ax1.text(sel, lo + (hi - lo) * 0.03, f"  shipped: {sel:,}", va="bottom",
                 ha="left", color=INK, fontsize=9.5, fontweight="bold")

    ax1.set_ylabel("Fertility (tokens per word)")
    ax1.margins(x=0.06)
    ax1.set_title("Why vocabulary 4,000 — fertility keeps improving, "
                  "the parameter budget does not",
                  loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=12)
    ax2.set_ylabel(f"Embedding share of a\n{budget / 1e6:.0f}M parameter budget")
    ax2.set_xlabel("Vocabulary size")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(all_v)
    ax2.set_xticklabels([f"{v:,}" for v in all_v])
    ax2.set_ylim(0, max(shares) * 1.22)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_xlim(min(all_v) * 0.85, max(all_v) * 1.5)
    fig.text(0.5, -0.02,
             "Two measures on incomparable scales, so two panels sharing one "
             "x-axis — never one plot with two y-axes.",
             ha="center", color=MUTED, fontsize=9)
    save(fig, out_dir, "vocab_tradeoff", log)
    return True


# ---------------------------------------------------------------------------
# 3. cleaning funnel
# ---------------------------------------------------------------------------

def fig_cleaning(data, out_dir, log=print) -> bool:
    made = False
    for lang in LANGS:
        cs = data[lang]["corpus"]
        if not cs:
            continue
        counters = cs.get("counters") or {}
        # `over_budget` belongs to the budget TRIM, a later stage than cleaning,
        # and it is an order of magnitude larger than any filter. Including it
        # would both mislabel the chart and flatten everything else.
        trimmed = counters.get("drop:over_budget", 0)
        drops = {k[len("drop:"):].replace("_", " "): v
                 for k, v in counters.items()
                 if k.startswith("drop:") and v and k != "drop:over_budget"}
        if not drops:
            continue
        items = sorted(drops.items(), key=lambda kv: kv[1])

        fig, ax = plt.subplots(figsize=(8.0, 0.34 * len(items) + 2.3))
        ax.set_xscale("log")
        ax.set_xlim(0.7, max(v for _, v in items) * 8)
        # DOT plot, not bars. The counts span five orders of magnitude, which
        # forces a log axis -- and a bar on a log axis lies: its length is the
        # log of the value, so 1 and 1,000 look like they differ by a third
        # rather than by a thousand. A dot encodes with position only.
        for i, (_, v) in enumerate(items):
            ax.plot([0.7, v], [i, i], color=AXIS, lw=1,
                    zorder=2, solid_capstyle="butt")
        ax.plot([v for _, v in items], range(len(items)), "o", ms=9,
                color=S1, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none", zorder=3)
        for i, (_, v) in enumerate(items):
            ax.text(v * 1.35, i, f"{v:,}", va="center", ha="left",
                    color=INK_2, fontsize=9.5, zorder=4)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([k for k, _ in items], color=INK_2, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(0.7, max(v for _, v in items) * 8)
        ax.set_ylim(-0.8, len(items) - 0.2)
        ax.set_xlabel("Documents removed (log scale — position, not length)")
        ax.grid(True, axis="x", which="major", color=AXIS, lw=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        kept = cs.get("kept_documents")
        raw = cs.get("raw_documents")
        sub = (f"{raw:,} read → {kept:,} survived"
               if raw and kept else "documents removed by filter")
        if trimmed:
            sub += (f"\n{trimmed:,} more were dropped later by the budget trim, "
                    f"not by cleaning")
        ax.set_title(f"{lang.title()} — what cleaning and deduplication removed\n"
                     f"{sub}",
                     loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=12)
        save(fig, out_dir, f"cleaning_{lang}", log)
        made = True
    if not made:
        log("  [skip] cleaning_* — no drop counters in corpus_stats.json")
    return made


# ---------------------------------------------------------------------------
# 4. Zipf
# ---------------------------------------------------------------------------

def fig_zipf(root, data, out_dir, log=print) -> bool:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    drawn = False
    for lang, colour in (("hindi", S1), ("nepali", S2)):
        p = root / lang / "tokenizer" / "analysis" / "token_frequency.csv"
        if not p.exists():
            continue
        ranks, shares = [], []
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ranks.append(int(row["rank"]))
                    shares.append(float(row["share"]))
                except (KeyError, ValueError):
                    continue
        if len(ranks) < 50:
            continue
        ts = data[lang].get("tokstats") or {}
        slope = ((ts.get("token_frequency") or {}).get("zipf_slope"))
        label = lang.title() + (f"  (slope {slope})" if slope is not None else "")
        ax.plot(ranks, shares, color=colour, lw=2, label=label)
        drawn = True
    if not drawn:
        plt.close(fig)
        log("  [skip] zipf — no token_frequency.csv "
            "(run tokenizer_report.py)")
        return False

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token rank")
    ax.set_ylabel("Share of all tokens")
    ax.grid(True, which="major", color=AXIS, lw=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=10)
    ax.set_title("Token frequency distribution on held-out text",
                 loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=12)
    fig.text(0.5, -0.01,
             "Natural language is near-linear on log-log with a slope around "
             "-1. Both tokenizers use vocab 4,000.",
             ha="center", color=MUTED, fontsize=9)
    save(fig, out_dir, "zipf", log)
    return True


def fig_doc_length(data, out_dir, log=print) -> bool:
    """Percentile curve, not a histogram. The question a reader actually has is
    "what fraction of my documents fit in a 512-token context?", and a
    percentile curve answers it by reading across; a histogram makes them
    integrate bins by eye. Bin widths would also have to be chosen, and on a
    distribution this skewed the choice changes the story."""
    series = {}
    for lang in LANGS:
        ds = data[lang].get("dataset")
        if not ds:
            continue
        d = (ds.get("document_length") or {}).get("tokens")
        unit = "tokens"
        if not d:
            d = (ds.get("document_length") or {}).get("words")
            unit = "words"
        if not d:
            continue
        pts = [(1, d["p1"]), (5, d["p5"]), (25, d["p25"]), (50, d["median"]),
               (75, d["p75"]), (95, d["p95"]), (99, d["p99"])]
        series[lang] = (pts, unit)
    if not series:
        log("  [skip] doc_length — no dataset_statistics.json "
            "(run pipeline.stats.dataset_stats)")
        return False

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    unit = "tokens"
    for lang, colour in (("hindi", S1), ("nepali", S2)):
        if lang not in series:
            continue
        pts, unit = series[lang]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=lang.title())
        ax.annotate(lang.title(), (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(10, 0), color=INK_2, fontsize=10, va="center")

    for ctx, style in ((512, (0, (4, 3))), (1024, (0, (1, 2)))):
        ax.axhline(ctx, color=INK, lw=1.2, ls=style, zorder=1)
        ax.text(0.6, ctx * 1.06, f"{ctx}-{unit[:-1]} context", color=INK,
                fontsize=9, va="bottom", ha="left")

    ax.set_yscale("log")
    ax.set_xlim(0, 108)
    # p1/p99 are plotted but not ticked: their labels collide with p5/p95 at
    # any figure width that keeps the rest readable.
    ax.set_xticks([5, 25, 50, 75, 95])
    ax.set_xticklabels(["p5", "p25", "median", "p75", "p95"])
    ax.set_ylabel(f"Document length ({unit}, log scale)")
    ax.set_xlabel("Percentile of documents")
    ax.grid(True, axis="y", which="major", color=AXIS, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Document length distribution",
                 loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=12)
    fig.text(0.5, -0.01,
             "Read across: where a curve crosses a context line is the share of "
             "documents that fit without truncation.",
             ha="center", color=MUTED, fontsize=9)
    save(fig, out_dir, "doc_length", log)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--param-budget", type=int, default=25_000_000)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = root / args.out_dir
    data = {}
    for lang in LANGS:
        s = root / lang / "data" / "stats"
        data[lang] = {
            "corpus": load(s / "corpus_stats.json"),
            "tokens": load(s / "token_accounting.json"),
            "vocab": load(root / lang / "tokenizer" / "analysis" / "vocab_selection.json"),
            "tokstats": load(root / lang / "tokenizer" / "analysis" / "token_stats.json"),
            "dataset": load(s / "dataset_statistics.json"),
        }

    print(f"\n=== figures -> {args.out_dir} ===")
    made = sum([
        bool(fig_corpus_composition(data, out_dir)),
        bool(fig_vocab_tradeoff(root, data, out_dir,
                                d_model=args.d_model, budget=args.param_budget)),
        bool(fig_cleaning(data, out_dir)),
        bool(fig_zipf(root, data, out_dir)),
        bool(fig_doc_length(data, out_dir)),
    ])
    print(f"\n  {made} figure group(s) written. Skipped ones name the stage "
          f"that produces their input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
