#!/usr/bin/env python3
"""
audit_phase1.py
----------------
Quick pass/fail check for one language against the Phase 1 data-scale
requirements from the assignment brief:

  - ~500M tokens in the final training corpus (token_target)
  - at least 20% of those tokens from manual collection
  - (informational) progress toward the 100M manual / 400M downloaded split

This is separate from compute_stats.py on purpose: compute_stats.py builds
the full descriptive report (filter tables, tokenizer stats, plots feed).
This script asks one narrow question and prints a clear PASS/FAIL so it can
be run on its own after a collection run, without needing a trained
tokenizer or the splits step to have happened yet.

Token counting
--------------
Counts come from manifest.csv (raw_char_count / 4.5, the same estimator
manifest_utils.py uses elsewhere in this repo), summed over rows with
status == "retained", split by collection_method. This is a planning-grade
estimate, not the exact SentencePiece token count ??? the exact number only
exists once splits + tokenizer training are done (see compute_stats.py's
tokenizer stats for that).

Usage
-----
  python common/audit_phase1.py --lang hindi  --repo-root project
  python common/audit_phase1.py --lang nepali --repo-root project

Exit status
-----------
  0  scale + manual-share requirements met
  1  not met (Nepali scale shortfalls are expected/allowed ??? see note below)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "preprocessing"))
from manifest_utils import load_manifest, manual_downloaded_summary  # noqa: E402


def build_report(lang: str, repo_root: str) -> dict:
    """
    Pull manual/downloaded token estimates from the manifest and check them
    against the token targets in <lang>/configs/data_config.yaml.

    Parameters
    ----------
    lang : str
    repo_root : str

    Returns
    -------
    dict
        Everything needed to print a human-readable PASS/FAIL summary.
    """
    summary = manual_downloaded_summary(lang, repo_root)

    total_tokens = summary["total_tokens_est"]
    manual_fraction = (
        summary["manual_tokens_est"] / total_tokens if total_tokens else 0.0
    )

    scale_met = total_tokens >= summary["token_target"]
    manual_share_met = manual_fraction >= 0.20

    return {
        "language": lang,
        "manual_tokens_est": summary["manual_tokens_est"],
        "downloaded_tokens_est": summary["downloaded_tokens_est"],
        "total_tokens_est": total_tokens,
        "manual_fraction": round(manual_fraction, 4),
        "token_target": summary["token_target"],
        "manual_target_tokens": summary["manual_target_tokens"],
        "downloaded_target_tokens": summary["downloaded_target_tokens"],
        "scale_requirement_met": scale_met,
        "manual_share_requirement_met": manual_share_met,
        "manual_target_met": summary["manual_target_met"],
        "downloaded_target_met": summary["downloaded_target_met"],
        "all_requirements_met": scale_met and manual_share_met,
    }


def print_summary(report: dict) -> None:
    """Print the JSON report plus a short human-readable verdict line."""
    print(json.dumps(report, ensure_ascii=False, indent=2))

    lang = report["language"]
    if report["all_requirements_met"]:
        print(f"\n[audit_phase1] {lang}: PASS "
              f"({report['total_tokens_est']/1e6:.1f}M tokens, "
              f"{report['manual_fraction']*100:.1f}% manual).")
        return

    reasons = []
    if not report["scale_requirement_met"]:
        shortfall = report["token_target"] - report["total_tokens_est"]
        reasons.append(f"total tokens short by ~{shortfall/1e6:.1f}M")
    if not report["manual_share_requirement_met"]:
        reasons.append(
            f"manual share {report['manual_fraction']*100:.1f}% < 20% required"
        )
    print(f"\n[audit_phase1] {lang}: FAIL ??? {'; '.join(reasons)}", file=sys.stderr)

    if lang == "nepali" and not report["scale_requirement_met"]:
        print(
            "[audit_phase1] Note: a total-token shortfall is expected and "
            "explicitly allowed for the low-resource language per the "
            "assignment ??? record the exact figures above in the Phase 1 "
            "report rather than treating this run as a hard failure.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check one language's collected data against the "
                     "500M-token / 20%-manual Phase 1 requirements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--repo-root", default=".",
                         help="Path to the 'project' directory (contains hindi/, nepali/).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not load_manifest(args.lang, args.repo_root):
        print(
            f"[audit_phase1] ERROR: no manifest found for '{args.lang}' under "
            f"{args.repo_root}. Run download_public.py / manual collection first.",
            file=sys.stderr,
        )
        return 1

    report = build_report(args.lang, args.repo_root)
    print_summary(report)
    return 0 if report["all_requirements_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
