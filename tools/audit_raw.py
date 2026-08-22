"""
audit_raw.py — is there enough collected text, before spending an hour finding out?
===================================================================================

A full `build_corpus` run on a large language is ~90 minutes of cleaning and
near-duplicate detection. That is a long time to wait to learn that the corpus
was never going to reach its target. This reads the RAW files and the previous
build's statistics and answers the question directly:

    python tools/audit_raw.py --lang hindi --repo-root .

WHAT IT REPORTS AND WHAT EACH NUMBER MEANS
------------------------------------------
1. RAW chars by provenance_class and source, straight off `<lang>/data/raw/`.
   This is an upper bound. Nothing here has been cleaned or deduplicated.

2. AVAILABLE chars from the last build's `corpus_stats.json`, if one exists.
   That figure IS post-clean and post-dedup, so it is the honest input to the
   budget. The gap between (1) and (2) is what cleaning threw away last time.

3. NEW raw chars since that build -- (1) now minus (1) as it stood then,
   inferred from the retention rate. This is the only number that tells you
   whether re-running the build can move the result at all.

4. The CEILING. The >=20% manual requirement caps the whole corpus at

       total <= manual_tokens / f'          f' = 0.20 + margin

   so the corpus can never be larger than about 4.878x the manual tokens, no
   matter how much downloaded text is sitting in the bucket. If that ceiling
   is below the target, more Sangraha will not help and only manual collection
   will.

WHY THIS DOES NOT COUNT TOKENS
------------------------------
It cannot. Tokens are defined by a tokenizer, and the tokenizer is trained on
the finished corpus. Everything here is characters divided by a
chars-per-token figure you supply -- pass the MEASURED one from
`count_corpus_tokens.py` if you have it, and the output will say so. Estimates
are labelled as estimates in the printout for the same reason they are
labelled in the reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def iter_raw_files(raw_dir: Path):
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.suffix == ".jsonl" and not p.name.startswith("."):
            yield p


def scan(raw_dir: Path, text_key: str = "text", log=print):
    """Stream every raw JSONL and total characters by (class, source)."""
    chars: Counter[tuple[str, str]] = Counter()
    docs: Counter[tuple[str, str]] = Counter()
    per_file: list[tuple[str, int, int]] = []
    bad = 0

    for path in iter_raw_files(raw_dir):
        f_docs = f_chars = 0
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                cls = rec.get("provenance_class") or "UNLABELLED"
                src = rec.get("source") or "unknown"
                n = len(rec.get(text_key) or "")
                chars[(cls, src)] += n
                docs[(cls, src)] += 1
                f_docs += 1
                f_chars += n
        per_file.append((path.name, f_docs, f_chars))
        log(f"    {path.name:<44} {f_docs:>10,} docs  {f_chars:>16,} chars")

    return chars, docs, per_file, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--chars-per-token", type=float, default=None,
                    help="MEASURED value from count_corpus_tokens.py if you have "
                         "one. Without it the script falls back to 4.0 and says "
                         "so on every line it affects.")
    ap.add_argument("--min-manual-fraction", type=float, default=0.20)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--target-tokens", type=int, default=500_000_000)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    raw_dir = root / args.lang / "data" / "raw"
    if not raw_dir.is_dir():
        print(f"[error] {raw_dir} not found", file=sys.stderr)
        return 1

    cpt = args.chars_per_token or 4.0
    measured = args.chars_per_token is not None
    tag = "measured" if measured else "PROXY 4.0 -- pass --chars-per-token"

    print(f"\n=== raw audit: {args.lang} ===")
    print(f"  raw dir : {raw_dir}")
    print(f"\n  [1/3] scanning raw files")
    chars, docs, per_file, bad = scan(raw_dir, args.text_key)
    if bad:
        print(f"    [warn] {bad:,} unparseable lines skipped")

    by_class: Counter[str] = Counter()
    for (cls, _src), n in chars.items():
        by_class[cls] += n

    print(f"\n  by provenance_class and source (RAW -- not cleaned, not deduped):")
    for cls in sorted(by_class, key=lambda c: -by_class[c]):
        print(f"    {cls}  {by_class[cls]:,} chars")
        for (c, src), n in sorted(chars.items(), key=lambda kv: -kv[1]):
            if c == cls:
                print(f"        {src:<32} {docs[(c, src)]:>10,} docs  {n:>16,} chars")

    if "UNLABELLED" in by_class:
        print(f"\n    [WARN] {by_class['UNLABELLED']:,} chars carry no "
              f"provenance_class. build_corpus cannot grade these and they will "
              f"not count toward the manual fraction.")

    manual_raw = by_class.get("manual", 0)
    down_raw = sum(n for c, n in by_class.items() if c not in ("manual", "UNLABELLED"))

    # ---- 2. what the last build actually had after cleaning -----------------
    print(f"\n  [2/3] last build, post-clean and post-dedup")
    stats_path = root / args.lang / "data" / "stats" / "corpus_stats.json"
    retention = None
    prev_manual_avail = None
    if stats_path.exists():
        try:
            st = json.loads(stats_path.read_text(encoding="utf-8"))
            budget = st.get("budget") or st.get("budget_report") or {}
            avail = budget.get("available_chars") or {}
            prev_manual_avail = avail.get("manual")
            prev_down_avail = avail.get("downloaded")
            if prev_manual_avail:
                print(f"    available manual     {prev_manual_avail:,} chars")
                print(f"    available downloaded {prev_down_avail:,} chars")
                # Retention is against the raw total AS IT WAS THEN, which we do
                # not have. Reporting it against today's raw understates it
                # whenever new data has arrived -- say so rather than imply a
                # clean ratio.
                retention = prev_manual_avail / manual_raw if manual_raw else None
                print(f"    manual survived cleaning: "
                      f"{retention:.1%} of TODAY's raw manual chars")
                if retention and retention < 0.95:
                    print(f"    (if raw has grown since that build, the true "
                          f"retention rate is higher than this)")
            else:
                print(f"    [warn] {stats_path.name} has no budget.available_chars")
        except Exception as e:
            print(f"    [warn] could not read {stats_path}: {type(e).__name__}")
    else:
        print(f"    no previous build found at {stats_path}")
        print(f"    cleaning typically keeps 75-90% of raw characters")

    # ---- 3. the ceiling -----------------------------------------------------
    print(f"\n  [3/3] what the >=20% manual requirement permits")
    f = args.min_manual_fraction + args.margin
    ratio = (1 - f) / f

    def ceiling(manual_chars: float, label: str):
        mt = manual_chars / cpt
        total = mt * (1 + ratio)
        need_down = mt * ratio
        pct = total / args.target_tokens
        print(f"\n    {label}")
        print(f"      manual chars            {manual_chars:>16,.0f}")
        print(f"      -> manual tokens        {mt:>16,.0f}   ({tag})")
        print(f"      downloaded permitted    {need_down:>16,.0f}   "
              f"(manual x {ratio:.3f})")
        print(f"      CEILING on total        {total:>16,.0f}   "
              f"({pct:.1%} of {args.target_tokens:,})")
        if down_raw / cpt < need_down:
            print(f"      [note] only {down_raw / cpt:,.0f} downloaded tokens are "
                  f"in raw, less than the {need_down:,.0f} permitted -- "
                  f"downloaded is the binding side here, not manual")
        return total

    ceiling(manual_raw, "using RAW manual chars (optimistic: nothing deduped yet)")
    if prev_manual_avail:
        realistic = ceiling(float(prev_manual_avail),
                            "using LAST BUILD's post-dedup manual chars (realistic)")
        gap = args.target_tokens - realistic
        print()
        if gap <= 0:
            print(f"    VERDICT: the target is reachable from data already "
                  f"collected. Re-run build_corpus with the measured "
                  f"chars-per-token; no more collection needed.")
        else:
            extra_manual_tokens = gap / (1 + ratio)
            extra_manual_chars = extra_manual_tokens * cpt
            print(f"    VERDICT: {gap:,.0f} tokens short of "
                  f"{args.target_tokens:,}.")
            print(f"    Closing it needs about {extra_manual_tokens:,.0f} more "
                  f"MANUAL tokens (~{extra_manual_chars:,.0f} chars).")
            print(f"    More downloaded text cannot close it -- the ratio caps "
                  f"downloaded at {ratio:.3f}x manual.")
            if retention:
                print(f"    At the observed {retention:.0%} cleaning retention "
                      f"that is ~{extra_manual_chars / retention:,.0f} raw chars "
                      f"to collect.")
        if manual_raw > prev_manual_avail:
            new = manual_raw - prev_manual_avail
            print(f"\n    Raw manual is {new:,} chars above what the last build "
                  f"had available. Some of that is data collected since; some is "
                  f"what cleaning removed. A rebuild is the only way to separate "
                  f"the two.")
    print()
    if not measured:
        print("  [reminder] every token figure above is chars / 4.0, an ESTIMATE. "
              "Do not report these as token counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
