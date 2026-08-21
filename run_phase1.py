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
    gcs-ingest corpora already in Cloud Storage -> the <=80%   <-- default path
    ingest-existing  local zips/jsonl -> the <=80%             (alternative)
    discover   seed domains -> article URLs         (manual collection prep)
    plan       can you reach >=20% manual? run BEFORE a long collection
    scrape     article URLs -> manual documents     (the >=20%)
    ocr        PDFs -> manual documents             (optional, slow)
    download   HuggingFace corpora -> documents     (only if not already in GCS)
    build      normalise, filter, dedup, BUDGET TRIM, split
    tokenizer  train from scratch on train split only
    count      authoritative token count + manual fraction
    report     per-language statistics and figures

`gcs-ingest` and `download` are two routes to the same place. If the corpora
are already sitting in gs://lma-01-hi-ne-corpus/raw/ you want the first: it
streams, respects a token budget, and never re-downloads from HuggingFace.
`download` stays for sources not yet in the bucket.

`build` is where the 100M-manual / 400M-downloaded split is enforced, after
deduplication -- see the budget section in pipeline/process/build_corpus.py.

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

STAGES = ["gcs-ingest", "ingest-existing", "discover", "plan", "scrape", "ocr",
          "download", "build", "tokenizer", "count", "report"]

# `download` re-fetches from HuggingFace, which is pointless once the data is in
# the bucket -- and it is. `all` therefore means everything except that.
DEFAULT_STAGES = [s for s in STAGES if s not in ("download", "ingest-existing")]


def load_targets(root: Path, lang: str) -> dict:
    """The `targets:` block of <lang>/configs/data_config.yaml, or {}."""
    p = root / lang / "configs" / "data_config.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("targets") or {}
    except Exception as e:
        print(f"[warn] could not read {p}: {e}")
        return {}


def opt(flag: str, value) -> list[str]:
    """
    A flag, only if the user actually set it.

    Budget knobs default to None here so that <lang>/configs/data_config.yaml
    stays the single source of truth. Passing the orchestrator's own default
    would override the config with a value nobody chose, and the config would
    become decoration that silently does nothing.
    """
    return [] if value is None else [flag, str(value)]


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

    gcs = load(root / lang / "data" / "stats" / "gcs_ingest_summary.json")
    if gcs and gcs.get("sources"):
        L += ["### Downloaded sources (streamed from Cloud Storage)", "",
              f"Bucket root: `{gcs['bucket_root']}`", "",
              "| Source | HF origin | Documents | Characters |", "|---|---|--:|--:|"]
        for k, v in gcs["sources"].items():
            L.append(f"| {k} | {v.get('hf_repo', '???')} | "
                     f"{v.get('documents', 0):,} | {v.get('characters', 0):,} |")
        L += ["", "Deliberately excluded:", ""]
        for k, why in (gcs.get("excluded_deliberately") or {}).items():
            L.append(f"- `{k}` ??? {why}")
        L.append("")

    if corpus and corpus.get("budget"):
        b = corpus["budget"]
        et = b.get("estimated_kept_tokens") or {}
        L += ["### Budget trim", "",
              "Applied after deduplication, so the figures below are of "
              "surviving documents. Token figures here are estimates from the "
              "character counts; the authoritative numbers are in the token "
              "accounting section above.", "",
              "| | Available (chars) | Kept (chars) | Kept (est. tokens) | Target (tokens) |",
              "|---|--:|--:|--:|--:|",
              f"| Manual | {b['available_chars']['manual']:,} | "
              f"{b['kept_chars']['manual']:,} | {et.get('manual', 0):,} | "
              f"{b.get('manual_target_tokens', 0):,} |",
              f"| Downloaded | {b['available_chars']['downloaded']:,} | "
              f"{b['kept_chars']['downloaded']:,} | {et.get('downloaded', 0):,} | "
              f"{b.get('downloaded_target_tokens', 0):,} |", "",
              f"Manual share after trim: **{b['manual_char_fraction']:.2%}** of "
              f"characters, **{b.get('estimated_manual_token_fraction', 0):.2%}** "
              f"of tokens (estimated).", "",
              f"The trim aimed at {b.get('effective_fraction_targeted', 0):.1%} "
              f"rather than {b['min_manual_fraction']:.0%} ??? a "
              f"{b.get('margin', 0):.1%} margin, because the trim measures "
              f"characters while the requirement grades tokens and the two are "
              f"not exactly proportional across provenance classes.", "",
              f"The downloaded budget was set by the "
              f"**{b['downloaded_budget_binding_constraint']}** constraint"
              + (f" ??? manual came in "
                 f"{b['manual_shortfall_chars']:,} characters short of its "
                 f"target, so downloaded was capped at "
                 f"{b['ratio_cap_chars']:,} to hold the ???"
                 f"{b['min_manual_fraction']:.0%} ratio."
                 if b["downloaded_budget_binding_constraint"] == "ratio"
                 else " ??? the ratio was satisfied with room to spare."), ""]

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
                    help=f"'all' (= {DEFAULT_STAGES}), 'every' for literally "
                         f"every stage including {STAGES[6]}, or a "
                         f"comma-separated list from {STAGES}")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    # --- the token budget, threaded through gcs-ingest and build -------------
    # All None by default: <lang>/configs/data_config.yaml is the source of
    # truth, and a flag here means "override the config for this run".
    ap.add_argument("--manual-target-tokens", type=int, default=None)
    ap.add_argument("--downloaded-target-tokens", type=int, default=None)
    ap.add_argument("--min-manual-fraction", type=float, default=None)
    ap.add_argument("--chars-per-token", type=float, default=None,
                    help="proxy for pass 1; count_corpus_tokens prints the "
                         "measured value for pass 2")
    # --- gcs-ingest ----------------------------------------------------------
    ap.add_argument("--bucket-root", default=None)
    ap.add_argument("--gcs-sources", nargs="+", default=None,
                    help="subset of wikipedia sangraha_verified sangraha_unverified")
    ap.add_argument("--overcollect", type=float, default=None,
                    help="ingest this multiple of the downloaded target, since "
                         "dedup and quality filters remove a large minority")
    ap.add_argument("--gcs-hours", type=float, default=6.0)
    # per-stage knobs
    ap.add_argument("--scrape-hours", type=float, default=4.0)
    ap.add_argument("--download-hours", type=float, default=3.0)
    ap.add_argument("--manual-target-chars", type=int, default=None,
                    help="scrape stop condition; defaults to "
                         "manual-target-tokens x overcollect x chars-per-token")
    ap.add_argument("--download-target-chars", type=int, default=1_600_000_000)
    ap.add_argument("--download-sources", nargs="+",
                    default=["sangraha", "wikipedia"])
    ap.add_argument("--ocr-input-dir", default=None)
    ap.add_argument("--existing-dir", default=None,
                    help="directory of already-downloaded zips/jsonl to import")
    ap.add_argument("--vocab-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--compare-algorithms", action="store_true")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang
    py = sys.executable

    if args.stage == "all":
        stages = list(DEFAULT_STAGES)
    elif args.stage == "every":
        stages = list(STAGES)
    else:
        stages = [s.strip() for s in args.stage.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"[error] unknown stage(s) {bad}; valid: {STAGES}", file=sys.stderr)
        return 1

    # Resolve the budget for display and for the stages that need derived
    # numbers (scrape's stop condition, plan's and count's totals). Each stage
    # re-resolves the same way, so these are a view, not a second source of
    # truth -- but they must agree, hence the shared precedence rule.
    targets = load_targets(root, lang)

    def budget(key, fallback):
        cli = getattr(args, key, None)
        return cli if cli is not None else targets.get(key, fallback)

    manual_tokens = budget("manual_target_tokens", 100_000_000)
    downloaded_tokens = budget("downloaded_target_tokens", 400_000_000)
    min_manual = budget("min_manual_fraction", 0.20)
    cpt = args.chars_per_token if args.chars_per_token is not None else \
        targets.get("chars_per_token_proxy", 4.0)
    overcollect = args.overcollect if args.overcollect is not None else \
        targets.get("overcollect", 1.45)
    total_tokens = manual_tokens + downloaded_tokens

    # The scrape target gets the same overcollect headroom the downloaded side
    # gets. Collecting exactly 100M tokens' worth yields well under 100M after
    # dedup, and that shortfall then drags the WHOLE corpus down through the
    # ratio cap in build -- a 30% manual miss costs 150M total, not 30M.
    manual_target_chars = args.manual_target_chars or int(
        manual_tokens * overcollect * cpt)

    print(f"=== Phase 1 pipeline: {lang} ===")
    print(f"  repo   : {root}")
    print(f"  stages : {stages}")
    print(f"  budget : {manual_tokens:,} manual + {downloaded_tokens:,} downloaded "
          f"= {total_tokens:,} tokens "
          f"({manual_tokens / total_tokens:.0%} manual, requirement "
          f">={min_manual:.0%})")

    failures = []

    if "gcs-ingest" in stages:
        cmd = [py, "-m", "pipeline.collect.gcs_ingest", "--lang", lang,
               "--repo-root", str(root), "--max-hours", str(args.gcs_hours)]
        cmd += opt("--bucket-root", args.bucket_root)
        cmd += opt("--target-tokens", args.downloaded_target_tokens)
        cmd += opt("--overcollect", args.overcollect)
        cmd += opt("--chars-per-token", args.chars_per_token)
        if args.gcs_sources:
            cmd += ["--sources", *args.gcs_sources]
        rc = run(cmd, cwd=root, dry=args.dry_run)
        if rc:
            failures.append("gcs-ingest")

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
             "--repo-root", str(root),
             "--target-total", str(total_tokens),
             "--min-manual-fraction", str(min_manual)],
            cwd=root, dry=args.dry_run)

    if "scrape" in stages:
        rc = run([py, "-m", "pipeline.collect.scrape_collect", "--lang", lang,
                  "--repo-root", str(root), "--max-hours", str(args.scrape_hours),
                  "--target-chars", str(manual_target_chars)],
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
        cmd = [py, "-m", "pipeline.process.build_corpus", "--lang", lang,
               "--repo-root", str(root)]
        cmd += opt("--manual-target-tokens", args.manual_target_tokens)
        cmd += opt("--downloaded-target-tokens", args.downloaded_target_tokens)
        cmd += opt("--min-manual-fraction", args.min_manual_fraction)
        cmd += opt("--chars-per-token", args.chars_per_token)
        rc = run(cmd, cwd=root, dry=args.dry_run)
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
                  "--repo-root", str(root),
                  "--target-tokens", str(total_tokens),
                  "--min-manual-fraction", str(min_manual)],
                 cwd=root, dry=args.dry_run)
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
