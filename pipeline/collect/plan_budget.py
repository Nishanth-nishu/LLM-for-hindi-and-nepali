"""
plan_budget.py — can you actually reach the >=20% manual requirement?
======================================================================
Answers the only question that matters before you start a long collection run:
given the domains you have, how many manual tokens can you realistically get,
and what does that cap your total corpus at?

THE CONSTRAINT, RESTATED AS ARITHMETIC
--------------------------------------
Let M = manual tokens, D = downloaded tokens. The brief requires

    M / (M + D) >= 0.20        which rearranges to        D <= 4M

so your **total corpus size is capped at 5x your manual tokens**:

    total = M + D <= 5M

That single line is the most useful thing on this page, because it means there
are TWO ways to satisfy the requirement and most people only think of one:

  (a) collect more manual data                    -- slow, bounded by the web
  (b) download LESS                               -- instant, entirely in your control

If scraping yields 60M manual tokens, you do not need 100M. You need to stop
downloading at 240M, giving 300M total at exactly 20%. A 300M-token corpus that
meets the requirement beats a 500M-token corpus that fails it.

Usage
-----
  python -m pipeline.collect.plan_budget --lang hindi --repo-root .
  python -m pipeline.collect.plan_budget --lang nepali --repo-root . --manual-tokens 35000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit


def fmt(n: float) -> str:
    return f"{int(n):,}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--target-total", type=int, default=500_000_000)
    ap.add_argument("--min-manual-fraction", type=float, default=0.20)
    ap.add_argument("--manual-tokens", type=int, default=None,
                    help="skip estimation; use this measured figure "
                         "(from token_accounting.json)")
    # Estimation parameters -- override with your own measurements.
    ap.add_argument("--tokens-per-doc", type=float, default=600,
                    help="mean tokens per accepted document")
    ap.add_argument("--acceptance-rate", type=float, default=0.45,
                    help="fraction of fetched pages surviving extraction, "
                         "language ID, quality and dedup")
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--hours", type=float, default=4.0)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang

    print(f"=== manual-data budget: {lang} ===\n")

    # ---- what you have -----------------------------------------------------
    url_file = root / lang / "data" / "raw" / "article_urls.txt"
    seed_file = root / lang / "configs" / "seed_domains.txt"

    n_urls = n_hosts = 0
    if url_file.exists():
        urls = [l.strip() for l in url_file.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")]
        n_urls = len(urls)
        n_hosts = len({urlsplit(u).netloc for u in urls})
        print(f"discovered: {fmt(n_urls)} article URLs across {n_hosts} hosts")
    elif seed_file.exists():
        n_hosts = len([l for l in seed_file.read_text(encoding="utf-8").splitlines()
                       if l.strip() and not l.startswith("#")])
        print(f"seed domains: {n_hosts}  (run `--stage discover` to get URL counts)")
    else:
        print(f"[warn] neither {url_file.name} nor {seed_file.name} found")

    # ---- what that yields --------------------------------------------------
    if args.manual_tokens is not None:
        manual = args.manual_tokens
        print(f"\nusing MEASURED manual tokens: {fmt(manual)}")
    else:
        ceiling_pps = n_hosts * (args.per_host / args.delay) if n_hosts else 0
        fetchable = ceiling_pps * args.hours * 3600
        if n_urls:
            fetchable = min(fetchable, n_urls)
        accepted = fetchable * args.acceptance_rate
        manual = accepted * args.tokens_per_doc

        print(f"\nthroughput (politeness-bound, not bandwidth-bound):")
        print(f"  {n_hosts} hosts x {args.per_host} concurrent / {args.delay}s"
              f"  = {ceiling_pps:.1f} pages/s ceiling")
        print(f"  in {args.hours}h            -> {fmt(fetchable)} pages fetchable")
        print(f"  x {args.acceptance_rate:.0%} acceptance   -> {fmt(accepted)} documents kept")
        print(f"  x {args.tokens_per_doc:.0f} tokens/doc  -> {fmt(manual)} manual tokens (ESTIMATE)")

    # ---- what it implies ---------------------------------------------------
    f = args.min_manual_fraction
    max_total = manual / f if f else 0
    max_download = max_total - manual
    needed_manual = args.target_total * f

    print(f"\n--- the {f:.0%} constraint ---")
    print(f"  M / (M + D) >= {f:.2f}   =>   D <= {(1 - f) / f:.0f}M   =>   total <= {1 / f:.0f}M\n")
    print(f"  manual tokens  M          = {fmt(manual)}")
    print(f"  max downloaded D          = {fmt(max_download)}")
    print(f"  MAX TOTAL CORPUS          = {fmt(max_total)}")
    print(f"  target total              = {fmt(args.target_total)}")

    print()
    if max_total >= args.target_total:
        print(f"  [OK] enough manual data to support the full {fmt(args.target_total)} target.")
        print(f"       Cap downloaded collection at {fmt(args.target_total - needed_manual)} "
              f"tokens so the ratio holds.")
    else:
        shortfall = needed_manual - manual
        print(f"  [CONSTRAINED] {fmt(manual)} manual tokens caps your corpus at "
              f"{fmt(max_total)},")
        print(f"                which is {fmt(args.target_total - max_total)} below target.")
        print(f"\n  Two ways out, and (b) is usually the right one:")
        print(f"    (a) collect {fmt(shortfall)} more manual tokens")
        print(f"        -> roughly {fmt(shortfall / args.tokens_per_doc)} more documents")
        if n_hosts:
            extra_hosts = (shortfall / args.tokens_per_doc / args.acceptance_rate) / \
                          (args.hours * 3600 * (args.per_host / args.delay))
            print(f"        -> or about {extra_hosts:.0f} more DOMAINS at the same hours")
        print(f"    (b) ship a smaller corpus: {fmt(max_total)} tokens at exactly "
              f"{f:.0%} manual.")
        print(f"        The brief permits this for the lower-resource language --")
        print(f"        \"collect the maximum feasible amount, report the exact token")
        print(f"        count, and justify the shortfall\". A {fmt(max_total)}-token")
        print(f"        corpus that MEETS the ratio beats a {fmt(args.target_total)}-token")
        print(f"        one that fails it.")

    # ---- sanity on the URL supply -----------------------------------------
    if n_urls:
        need_docs = manual / args.tokens_per_doc
        need_urls = need_docs / args.acceptance_rate
        print(f"\n--- URL supply ---")
        print(f"  need ~{fmt(need_urls)} URLs to yield {fmt(need_docs)} accepted documents")
        print(f"  have  {fmt(n_urls)}  ({n_urls / max(1, need_urls):.1f}x headroom)")
        if n_urls < need_urls:
            print(f"  [WARN] not enough URLs. Add domains and re-run discover.")

    print(f"\n--- reminders ---")
    print(f"  * These are ESTIMATES. Replace tokens/doc and acceptance with your own")
    print(f"    measurements after one real run, then re-run this with --manual-tokens.")
    print(f"  * Collect 1.5-2x your target: global dedup removes near-duplicates, and")
    print(f"    syndicated news duplicates heavily across sites.")
    print(f"  * Throughput scales with DOMAINS, not concurrency. More threads on ten")
    print(f"    domains changes nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
