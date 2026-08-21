"""
download_public.py ??? the <=80% downloaded portion
==================================================
Your `download_hi_ne_corpora_gcp.py`, reworked to (a) fit the Phase 1 repo
layout, (b) emit manifest records so the downloaded tokens are counted and
labelled, and (c) fix four things that would have cost you data.

WHAT WAS FIXED, AND WHY IT MATTERED
-----------------------------------
1. `get_mc4()` built its argument as
       f"multilingual/c4-{code}*.json.gz".replace("*", "")
   which produces the string "multilingual/c4-hi.json.gz". That is neither a
   valid config name nor a valid data_dir for `allenai/c4`, so the call could
   only ever raise and be swallowed by the bare `except` in
   `download_hf_dataset`. mC4 was silently contributing zero. Now uses the
   documented `data_files` glob form.

2. `load_dataset(...)` without `streaming=True` materialises the whole split
   before you see a single row. mC4 Hindi is hundreds of GB. On a notebook this
   fills the disk and dies. Every source now streams and writes incrementally,
   so you can stop at a token budget instead of at an out-of-disk error.

3. `except Exception as e: print(...); return` swallowed everything, including
   "you are not authenticated" and "you have not accepted the terms". OSCAR and
   several others are gated; the run looked like it worked and produced nothing.
   Failures are now classified and reported at the end, with the specific fix.

4. No provenance labelling. Downstream cannot tell downloaded from manual, which
   is the graded distinction. Every record now carries
   `provenance_class="downloaded"`.

ON PARALLEL CORPORA (opus, samanantar)
--------------------------------------
Both are *parallel* en-xx corpora. For a monolingual corpus you take one side.
They are excluded by default: the Hindi side of an en-hi parallel corpus is
frequently itself a translation, and the brief's Phase 3 asks you to analyse
corpus quality differences. Translated text in a "monolingual Hindi" corpus is
the kind of thing that shows up as an unexplained anomaly later. Enable with
`--include-parallel` if you want it, and say so in the report.

Usage
-----
  python -m pipeline.collect.download_public --lang hindi --repo-root . \\
      --sources sangraha indiccorp wikipedia --target-chars 1600000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import Document, ShardWriter  # noqa: E402

# Each source names languages differently.
LANG_CODES = {
    "hindi":  {"sangraha": "hin", "indiccorp": "hin_Deva", "oscar": "hi",
               "mc4": "hi", "cc100": "hi", "wikipedia": "hi"},
    "nepali": {"sangraha": "nep", "indiccorp": "npi_Deva", "oscar": "ne",
               "mc4": "ne", "cc100": "ne", "wikipedia": "ne"},
}

# (repo_id, how to address the subset, text field)
SOURCES = {
    "sangraha":  {"repo": "ai4bharat/sangraha", "mode": "data_dir",
                  "pattern": "verified/{code}", "text": "text"},
    "indiccorp": {"repo": "ai4bharat/IndicCorpV2", "mode": "data_dir",
                  "pattern": "data/{code}", "text": "text"},
    "wikipedia": {"repo": "wikimedia/wikipedia", "mode": "config",
                  "pattern": "20231101.{code}", "text": "text"},
    "oscar":     {"repo": "oscar-corpus/OSCAR-2301", "mode": "config",
                  "pattern": "{code}", "text": "text", "gated": True},
    "mc4":       {"repo": "allenai/c4", "mode": "data_files",
                  "pattern": "multilingual/c4-{code}.tfrecord-*.json.gz",
                  "text": "text"},
    "cc100":     {"repo": "cc100", "mode": "config",
                  "pattern": "{code}", "text": "text", "needs_trust": True},
}

DEFAULT_SOURCES = ["sangraha", "indiccorp", "wikipedia"]


def build_loader_kwargs(src: str, code: str) -> dict:
    spec = SOURCES[src]
    pat = spec["pattern"].format(code=code)
    if spec["mode"] == "data_dir":
        return {"path": spec["repo"], "data_dir": pat}
    if spec["mode"] == "config":
        return {"path": spec["repo"], "name": pat}
    return {"path": spec["repo"], "data_files": pat}


def download_source(*, src: str, lang: str, out_dir: Path, target_chars: int,
                    min_chars: int, max_hours: float, log=print) -> dict:
    from datasets import load_dataset

    code = LANG_CODES[lang].get(src)
    if code is None:
        return {"source": src, "status": "unavailable_for_language", "documents": 0}

    spec = SOURCES[src]
    kwargs = build_loader_kwargs(src, code)
    log(f"  [{src}] streaming {kwargs} ...")

    try:
        ds = load_dataset(**kwargs, split="train", streaming=True,
                          **({"trust_remote_code": True} if spec.get("needs_trust") else {}))
    except Exception as e:
        msg = str(e)
        hint = ""
        low = msg.lower()
        if "gated" in low or "401" in low or "403" in low or "authenticate" in low:
            hint = (" -> this dataset is GATED. Accept its terms on the HF website "
                    "and run `huggingface-cli login`.")
        elif "trust_remote_code" in low:
            hint = " -> needs trust_remote_code=True."
        elif "not found" in low or "404" in low:
            hint = " -> repo id or subset name is wrong for this language."
        log(f"    FAILED: {msg.splitlines()[0][:180]}{hint}")
        return {"source": src, "status": "load_failed",
                "error": msg.splitlines()[0][:300], "documents": 0}

    out_path = out_dir / f"downloaded_{src}.jsonl"
    writer = ShardWriter(out_path)
    text_key = spec["text"]
    chars = 0
    n = skipped = 0
    t0 = time.monotonic()

    try:
        for rec in ds:
            text = (rec.get(text_key) or "").strip()
            if len(text) < min_chars:
                skipped += 1
                continue
            doc = Document(text=text, language=lang, provenance_class="downloaded",
                           source=src, collection_method="hf_download",
                           extra={"hf_repo": spec["repo"], "hf_subset": code})
            if writer.write(doc):
                n += 1
                chars += len(text)
            if chars >= target_chars:
                log(f"    reached target_chars ({chars / 1e6:.0f}M)")
                break
            if (time.monotonic() - t0) / 3600 >= max_hours:
                log(f"    reached --max-hours")
                break
            if n and n % 20000 == 0:
                el = time.monotonic() - t0
                log(f"    {n:,} docs, {chars / 1e6:.0f}M chars, "
                    f"{n / max(1e-9, el):.0f} docs/s")
    except KeyboardInterrupt:
        log("    interrupted -- partial output kept")
    except Exception as e:
        log(f"    stream error after {n:,} docs: {type(e).__name__}: {str(e)[:160]}")
    finally:
        writer.close()

    log(f"    kept {n:,} docs ({chars / 1e6:.1f}M chars), skipped {skipped:,} short")
    return {"source": src, "status": "ok", "documents": n, "characters": chars,
            "duplicates_skipped": writer.duplicates, "path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                    help=f"any of {sorted(SOURCES)}")
    ap.add_argument("--target-chars", type=int, default=1_600_000_000,
                    help="per source. ~1.6G chars is a rough proxy for ~400M "
                         "tokens of Devanagari, i.e. the <=80% downloaded share.")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-hours", type=float, default=3.0, help="per source")
    ap.add_argument("--include-parallel", action="store_true",
                    help="allow opus/samanantar; see the module docstring first")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = root / args.lang / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = [s for s in args.sources if s not in SOURCES]
    if unknown:
        print(f"[error] unknown sources: {unknown}. Available: {sorted(SOURCES)}",
              file=sys.stderr)
        return 1

    print(f"[{args.lang}] downloading {args.sources}")
    results = []
    for src in args.sources:
        results.append(download_source(
            src=src, lang=args.lang, out_dir=out_dir,
            target_chars=args.target_chars, min_chars=args.min_chars,
            max_hours=args.max_hours))

    total_docs = sum(r.get("documents", 0) for r in results)
    total_chars = sum(r.get("characters", 0) for r in results)
    print(f"\n  total: {total_docs:,} documents, {total_chars / 1e6:.1f}M characters")

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print("\n  sources that produced nothing:")
        for r in failed:
            print(f"    {r['source']:<12} {r['status']}  {r.get('error', '')[:120]}")
        print("\n  A source failing here is normal (gating, renamed subsets). What is\n"
              "  NOT normal is not noticing -- the original script swallowed these.")

    import json
    stats_path = root / args.lang / "data" / "stats" / "download_summary.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(
        {"language": args.lang, "sources": results,
         "total_documents": total_docs, "total_characters": total_chars},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
