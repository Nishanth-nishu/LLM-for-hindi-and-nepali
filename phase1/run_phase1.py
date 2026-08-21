"""
run_phase1.py ??? end-to-end Phase 1 pipeline
============================================
One command per language, from empty repo to the seven Phase 1 deliverables.

    python run_phase1.py --lang hindi --stage all

Every stage is resumable and idempotent: re-running skips work already done.
Stages can be run individually, which is what you want when one of them fails
at 2am.

STAGE ORDER, AND WHY IT IS THIS ORDER
--------------------------------------
    ingest-existing  corpora you already downloaded -> the <=80% (optional)
    discover   seed domains -> article URLs         (manual collection prep)
    plan       can you reach >=20% manual? run BEFORE a long collection
    scrape     article URLs -> manual documents     (the >=20%)
    ocr        PDFs -> manual documents             (optional, slow)
    download   HuggingFace corpora -> documents     (the <=80%)
    build      normalise, filter, dedup, split
    tokenizer  train from scratch on train split only
    count      authoritative token count + manual fraction
    report     per-language statistics and figures

`build` must precede `tokenizer` because the tokenizer trains on the train
split and only the train split -- training it on the raw pool would leak
validation and test text into the vocabulary.

`count` must follow `tokenizer` because "500M training tokens after
tokenization" can only be measured with the tokenizer you actually built. See
the two-pass protocol in pipeline/tokenizer/count_corpus_tokens.py.

WHAT THIS DOES NOT DO
---------------------
It does not decide your seed domains or supply your PDFs. Those are the two
inputs only you can provide, and they are what determine whether you reach the
manual target. See <lang>/configs/seed_domains.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STAGES = ["ingest-existing", "discover", "plan", "scrape", "ocr", "download",
          "build", "tokenizer", "count", "report"]


def run(cmd: list[str], *, cwd: Path, dry: bool = False) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry:
        return 0
    t0 = time.monotonic()
    rc = subprocess.call(cmd, cwd=str(cwd))
    print(f"  [{'ok' if rc == 0 else f'rc={rc}'}] {(time.monotonic() - t0) / 60:.1f} min",
          flush=True)
    return rc


def stage_report(lang: str, root: Path) -> int:
    """Assemble the per-language dataset statistics report (deliverable 3)."""
    out = root / "report" / f"{lang}_dataset_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    def load(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    corpus = load(root / lang / "data" / "stats" / "corpus_stats.json")
    tokens = load(root / lang / "data" / "stats" / "token_accounting.json")
    tokstats = load(root / lang / "tokenizer" / "analysis" / "token_stats.json")
    vocabsel = load(root / lang / "tokenizer" / "analysis" / "vocab_selection.json")
    download = load(root / lang / "data" / "stats" / "download_summary.json")

    L = [f"# {lang.title()} ??? Phase 1 dataset and tokenizer report", ""]

    if corpus:
        L += ["## Corpus construction", "",
              f"- Raw documents collected: **{corpus['raw_documents']:,}**",
              f"- Documents after cleaning and deduplication: **{corpus['kept_documents']:,}**",
              f"- Retention: **{corpus['kept_documents'] / max(1, corpus['raw_documents']):.1%}**", "",
              "### Cleaning steps and what each removed", "",
              "| Step | Documents removed |", "|---|--:|"]
        for k, v in sorted(corpus["counters"].items(), key=lambda kv: -kv[1]):
            if k.startswith("drop:"):
                L.append(f"| {k[5:].replace('_', ' ')} | {v:,} |")
        L += ["", "### Splits", "", "| Split | Documents | Manual chars | Downloaded chars |",
              "|---|--:|--:|--:|"]
        for s in ("train", "val", "test"):
            cc = corpus["split_characters_by_class"].get(s, {})
            L.append(f"| {s} | {corpus['splits'].get(s, 0):,} | "
                     f"{cc.get('manual', 0):,} | {cc.get('downloaded', 0):,} |")
        L.append("")

    if tokens:
        t, m = tokens["totals"], tokens["manual_vs_downloaded"]
        L += ["## Token accounting", "",
              f"Counted with **{Path(tokens['tokenizer']['model_file']).name}** "
              f"(vocab {tokens['tokenizer']['vocab_size']:,}). "
              f"Token counts are only valid for this tokenizer.", "",
              f"- Corpus tokens (all splits): **{t['corpus_tokens_all_splits']:,}**",
              f"- Training tokens: **{t['train_tokens']:,}**",
              f"- Target: {t['target_tokens']:,} (**{t['pct_of_target']}%**)", ""]
        if t["shortfall_tokens"]:
            L += [f"> **Shortfall:** {t['shortfall_tokens']:,} tokens below target. "
                  f"_Justify here._", ""]
        L += ["### Manual vs downloaded", "", "| | Tokens | Share |", "|---|--:|--:|",
              f"| Manual | {m['manual_tokens']:,} | **{m['manual_fraction_of_tokens']:.2%}** |",
              f"| Downloaded | {m['downloaded_tokens']:,} | "
              f"{1 - m['manual_fraction_of_tokens']:.2%} |", "",
              f"Requirement ???{m['requirement']:.0%} ??? "
              f"**{'MET' if m['requirement_met'] else 'NOT MET'}**", "",
              f"(By document count the manual share is "
              f"{m['manual_fraction_of_documents']:.2%}; the brief specifies tokens.)", ""]

    if download and download.get("sources"):
        L += ["### Downloaded sources", "", "| Source | Status | Documents | Characters |",
              "|---|---|--:|--:|"]
        for s in download["sources"]:
            L.append(f"| {s['source']} | {s['status']} | {s.get('documents', 0):,} | "
                     f"{s.get('characters', 0):,} |")
        L.append("")

    if vocabsel:
        L += ["## Tokenizer", "",
              f"- Vocabulary size: **{vocabsel['selected_vocab_size']:,}**",
              f"- Selection: {vocabsel['selection_reason']}",
              f"- Scaling-law reference (Tao et al. 2024): "
              f"{vocabsel['scaling_law_reference_vocab']:,} "
              f"(ratio {vocabsel['ratio_to_scaling_law_reference']}x)", "",
              "### Vocabulary sweep", "",
              "| Vocab | Algo | Fertility | ?? vs prev | chars/tok | byte-fallback | Emb. share | Chosen |",
              "|--:|---|--:|--:|--:|--:|--:|:-:|"]
        for c in vocabsel["candidates"]:
            g = c["relative_fertility_gain_vs_prev"]
            L.append(f"| {c['vocab_size']:,} | {c['model_type']} | {c['fertility']} | "
                     f"{(f'{g:.2%}' if g != '' else '???')} | {c['chars_per_token']} | "
                     f"{c['byte_fallback_rate']:.2e} | {c['embedding_share']:.1%} | "
                     f"{'**yes**' if c['selected'] else ''} |")
        L.append("")

    if tokstats:
        e, u, v = tokstats["efficiency"], tokstats["unknown_tokens"], tokstats["vocabulary"]
        tf = tokstats["token_frequency"]
        L += ["### Tokenizer statistics (held-out test split)", "",
              "| Metric | Value |", "|---|--:|",
              f"| Fertility (tokens/word) | {e['fertility_tokens_per_word']} |",
              f"| Average characters per token | {e['avg_chars_per_token']} |",
              f"| Average bytes per token | {e['avg_bytes_per_token']} |",
              f"| UNK rate | {u['unk_rate']:.2e} |",
              f"| Byte-fallback rate | {u['byte_fallback_rate']:.2e} |",
              f"| Vocabulary utilisation | {v['vocab_utilisation']:.1%} |",
              f"| Pieces never used | {v['unused_pieces']:,} |",
              f"| Top-1000 piece coverage | {tf['coverage_top_1000']:.1%} |",
              f"| Zipf slope | {tf['zipf_slope']} |", ""]
        if tokstats.get("fertility_by_provenance"):
            L += ["#### Fertility by provenance", "", "| Provenance | Fertility |",
                  "|---|--:|"]
            for k, val in tokstats["fertility_by_provenance"].items():
                L.append(f"| {k} | {val} |")
            L += ["", "_If manual fertility is materially worse than downloaded, the "
                  "tokenizer is under-serving the data you collected yourself._", ""]
        L += [f"Tokenization examples: `{lang}/tokenizer/analysis/examples.md`",
              f"Full frequency table: `{lang}/tokenizer/analysis/token_frequency.csv`", ""]

    L += ["---", "", "_Generated by `run_phase1.py --stage report`. "
          "Fill in the justification prompts marked _italic_ before submitting._"]

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--stage", default="all",
                    help=f"'all' or one of {STAGES}, or a comma-separated list")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    # per-stage knobs
    ap.add_argument("--scrape-hours", type=float, default=4.0)
    ap.add_argument("--download-hours", type=float, default=3.0)
    ap.add_argument("--manual-target-chars", type=int, default=400_000_000)
    ap.add_argument("--download-target-chars", type=int, default=1_600_000_000)
    ap.add_argument("--download-sources", nargs="+",
                    default=["sangraha", "indiccorp", "wikipedia"])
    ap.add_argument("--ocr-input-dir", default=None)
    ap.add_argument("--existing-dir", default=None,
                    help="directory of already-downloaded zips/jsonl to import")
    ap.add_argument("--vocab-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--compare-algorithms", action="store_true")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang
    py = sys.executable

    stages = STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"[error] unknown stage(s) {bad}; valid: {STAGES}", file=sys.stderr)
        return 1

    print(f"=== Phase 1 pipeline: {lang} ===")
    print(f"  repo   : {root}")
    print(f"  stages : {stages}")

    failures = []

    if "ingest-existing" in stages:
        if not args.existing_dir:
            print("\n[skip] ingest-existing: no --existing-dir given")
        else:
            run([py, "-m", "pipeline.collect.ingest_existing", "--lang", lang,
                 "--repo-root", str(root), "--input-dir", args.existing_dir],
                cwd=root, dry=args.dry_run)

    if "discover" in stages:
        rc = run([py, "-m", "pipeline.collect.seed_discovery", "--lang", lang,
                  "--repo-root", str(root)], cwd=root, dry=args.dry_run)
        if rc:
            failures.append("discover")

    if "plan" in stages:
        run([py, "-m", "pipeline.collect.plan_budget", "--lang", lang,
             "--repo-root", str(root)], cwd=root, dry=args.dry_run)

    if "scrape" in stages:
        rc = run([py, "-m", "pipeline.collect.scrape_collect", "--lang", lang,
                  "--repo-root", str(root), "--max-hours", str(args.scrape_hours),
                  "--target-chars", str(args.manual_target_chars)],
                 cwd=root, dry=args.dry_run)
        if rc:
            failures.append("scrape")

    if "ocr" in stages:
        if not args.ocr_input_dir:
            print("\n[skip] ocr: no --ocr-input-dir given. OCR is optional; "
                  "scraping alone can satisfy the manual requirement.")
        else:
            rc = run([py, "-m", "pipeline.collect.ocr_collect", "--lang", lang,
                      "--repo-root", str(root), "--input-dir", args.ocr_input_dir],
                     cwd=root, dry=args.dry_run)
            if rc:
                failures.append("ocr")

    if "download" in stages:
        rc = run([py, "-m", "pipeline.collect.download_public", "--lang", lang,
                  "--repo-root", str(root), "--sources", *args.download_sources,
                  "--max-hours", str(args.download_hours),
                  "--target-chars", str(args.download_target_chars)],
                 cwd=root, dry=args.dry_run)
        if rc:
            failures.append("download")

    if "build" in stages:
        rc = run([py, "-m", "pipeline.process.build_corpus", "--lang", lang,
                  "--repo-root", str(root)], cwd=root, dry=args.dry_run)
        if rc:
            failures.append("build")

    if "tokenizer" in stages:
        cmd = [py, "pipeline/tokenizer/train_tokenizer.py", "--lang", lang,
               "--repo-root", str(root)]
        if args.compare_algorithms:
            cmd.append("--compare-algorithms")
        if args.vocab_sizes:
            cmd += ["--vocab-sizes", *map(str, args.vocab_sizes)]
        rc = run(cmd, cwd=root, dry=args.dry_run)
        if rc:
            failures.append("tokenizer")
        else:
            rc = run([py, "pipeline/tokenizer/tokenizer_report.py", "--lang", lang,
                      "--repo-root", str(root), "--split", "test"],
                     cwd=root, dry=args.dry_run)
            if rc:
                failures.append("tokenizer_report")

    if "count" in stages:
        rc = run([py, "pipeline/tokenizer/count_corpus_tokens.py", "--lang", lang,
                  "--repo-root", str(root)], cwd=root, dry=args.dry_run)
        # non-zero here means a requirement is unmet, which is information, not
        # a crash -- keep going so the report still gets written.
        if rc:
            print("  [note] count exited non-zero: a target or the >=20% manual "
                  "requirement is not met. The report will show the numbers.")

    if "report" in stages and not args.dry_run:
        stage_report(lang, root)

    print(f"\n=== done: {lang} ===")
    if failures:
        print(f"  stages that failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
