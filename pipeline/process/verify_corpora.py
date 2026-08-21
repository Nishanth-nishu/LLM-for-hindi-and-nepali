"""
verify_corpora.py — the gate you run before submitting
=======================================================
Every check here corresponds to something the brief states as a requirement.
None of them are stylistic. Run it after both languages are built and counted:

    python -m pipeline.process.verify_corpora --repo-root .

Exit code is 0 only if every check passes.

WHY THIS EXISTS AS A SEPARATE STAGE
-----------------------------------
The individual stages each enforce their own invariants, but three of the
brief's requirements are *cross-cutting* and no single stage can see them:

  - "Do not share documents across corpora" spans two languages, and each
    language's build runs alone.
  - Train/test leakage spans three files that the split writer produces
    independently.
  - The >=20% manual fraction is measured by `count`, but nothing re-reads that
    measurement and refuses to proceed.

A check that lives inside the stage that could violate it is a check that
passes for the wrong reason. These run last, over the artefacts as they will
actually be submitted.

THE CHECKS
----------
C1  no doc_id in both languages          "do not share documents across corpora"
C2  no doc_id in two splits              train/test leakage
C3  no identical TEXT across splits      leakage that survives a different id
C4  every document has a provenance      the >=20% figure is otherwise a guess
C5  manual token fraction >= 0.20        the graded requirement
C6  tokenizer trained on train only      vocabulary must not see held-out text
C7  no identical text across languages   stronger form of C1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import content_hash, read_jsonl  # noqa: E402

SPLITS = ("train", "val", "test")


class Result:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"\n         {detail}" if detail else ""))

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.checks if not ok]


def load_splits(root: Path, lang: str) -> dict[str, list[dict]]:
    out = {}
    for s in SPLITS:
        p = root / lang / "data" / "splits" / f"{s}.jsonl"
        out[s] = list(read_jsonl(p)) if p.exists() else []
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--languages", nargs="+", default=["hindi", "nepali"])
    ap.add_argument("--min-manual-fraction", type=float, default=0.20)
    ap.add_argument("--sample-text-checks", type=int, default=0,
                    help="limit C3/C7 to this many documents per split "
                         "(0 = all; use a limit only if memory is tight)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    r = Result()

    print("=== corpus verification ===\n")

    per_lang: dict[str, dict[str, list[dict]]] = {}
    for lang in args.languages:
        per_lang[lang] = load_splits(root, lang)
        n = sum(len(v) for v in per_lang[lang].values())
        print(f"  {lang}: {n:,} documents across "
              f"{ {k: len(v) for k, v in per_lang[lang].items()} }")
    print()

    missing = [l for l in args.languages if not sum(len(v) for v in per_lang[l].values())]
    if missing:
        print(f"[error] no split files for {missing}. Run `--stage build` first.",
              file=sys.stderr)
        return 1

    # ---- C2 / C3: leakage within a language --------------------------------
    for lang in args.languages:
        splits = per_lang[lang]

        seen_id: dict[str, str] = {}
        dupes: list[tuple[str, str, str]] = []
        for s in SPLITS:
            for rec in splits[s]:
                did = rec.get("doc_id")
                if did in seen_id and seen_id[did] != s:
                    dupes.append((did, seen_id[did], s))
                else:
                    seen_id.setdefault(did, s)
        r.add(f"C2 {lang}: no doc_id in two splits", not dupes,
              "" if not dupes else
              f"{len(dupes)} leaked ids, e.g. {dupes[:3]}. Phase 2 perplexity "
              f"would measure memorisation, not modelling.")

        # Text-level, because a different id on identical text leaks just as hard.
        seen_txt: dict[str, str] = {}
        txt_dupes = 0
        example = None
        for s in SPLITS:
            docs = splits[s]
            if args.sample_text_checks:
                docs = docs[:args.sample_text_checks]
            for rec in docs:
                h = content_hash(rec.get("text") or "")
                if h in seen_txt and seen_txt[h] != s:
                    txt_dupes += 1
                    example = example or (seen_txt[h], s)
                else:
                    seen_txt.setdefault(h, s)
        r.add(f"C3 {lang}: no identical text across splits", txt_dupes == 0,
              "" if not txt_dupes else
              f"{txt_dupes} documents appear in two splits under different ids "
              f"(e.g. {example[0]} and {example[1]}). Dedup ran, but not on the "
              f"form these documents differ in.")

        # ---- C4: provenance completeness -----------------------------------
        classes: Counter[str] = Counter()
        for s in SPLITS:
            for rec in splits[s]:
                classes[rec.get("provenance_class") or "MISSING"] += 1
        bad = {k: v for k, v in classes.items()
               if k not in ("manual", "downloaded")}
        r.add(f"C4 {lang}: every document has a provenance class", not bad,
              "" if not bad else
              f"{bad} — the manual fraction is a lower bound, not a measurement. "
              f"Fix the collector, not the report.")

        # ---- C5: the graded ratio ------------------------------------------
        acc_p = root / lang / "data" / "stats" / "token_accounting.json"
        if not acc_p.exists():
            r.add(f"C5 {lang}: manual token fraction >= "
                  f"{args.min_manual_fraction:.0%}", False,
                  "token_accounting.json not found — run `--stage count` "
                  "after the tokenizer exists. Character ratios are not the "
                  "graded number.")
        else:
            acc = json.loads(acc_p.read_text(encoding="utf-8"))
            m = acc["manual_vs_downloaded"]
            frac = m["manual_fraction_of_tokens"]
            tot = acc["totals"]["corpus_tokens_all_splits"]
            r.add(f"C5 {lang}: manual token fraction >= "
                  f"{args.min_manual_fraction:.0%}",
                  frac >= args.min_manual_fraction,
                  f"{frac:.2%} of {tot:,} tokens "
                  f"({m['manual_tokens']:,} manual)")

        # ---- C6: tokenizer saw only the train split ------------------------
        meta_p = root / lang / "tokenizer" / "vocab" / f"{lang}_tokenizer.json"
        if not meta_p.exists():
            r.add(f"C6 {lang}: tokenizer trained on train split only", False,
                  "tokenizer metadata not found — run `--stage tokenizer`.")
        else:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            src = json.dumps(meta.get("corpus") or meta.get("train_split") or meta)
            looks_right = "train" in src and "val.jsonl" not in src \
                and "test.jsonl" not in src
            r.add(f"C6 {lang}: tokenizer trained on train split only",
                  looks_right,
                  "" if looks_right else
                  f"tokenizer metadata references a split other than train. "
                  f"Held-out text in the vocabulary invalidates every "
                  f"fertility and perplexity number downstream.")

    # ---- C1 / C7: across languages -----------------------------------------
    if len(args.languages) >= 2:
        a, b = args.languages[0], args.languages[1]
        ids_a = {rec.get("doc_id") for s in SPLITS for rec in per_lang[a][s]}
        ids_b = {rec.get("doc_id") for s in SPLITS for rec in per_lang[b][s]}
        shared = (ids_a & ids_b) - {None}
        r.add(f"C1 {a}/{b}: no document shared across corpora", not shared,
              "" if not shared else
              f"{len(shared)} shared doc_ids, e.g. {sorted(shared)[:3]}. The "
              f"brief forbids sharing documents across corpora.")

        h_a = {content_hash(rec.get("text") or "")
               for s in SPLITS for rec in per_lang[a][s]}
        h_b = {content_hash(rec.get("text") or "")
               for s in SPLITS for rec in per_lang[b][s]}
        shared_txt = h_a & h_b
        r.add(f"C7 {a}/{b}: no identical text across corpora", not shared_txt,
              "" if not shared_txt else
              f"{len(shared_txt)} documents have identical normalised text in "
              f"both corpora. Usually a language-ID failure: Hindi and Nepali "
              f"share a script and short documents are genuinely ambiguous.")

    # ---- verdict ------------------------------------------------------------
    print()
    out = root / "report" / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"checks": [{"name": n, "passed": ok, "detail": d}
                    for n, ok, d in r.checks],
         "all_passed": not r.failed}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    if r.failed:
        print(f"  {len(r.failed)} check(s) FAILED: {r.failed}")
        print(f"  wrote {out}")
        return 1
    print(f"  all {len(r.checks)} checks passed")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
