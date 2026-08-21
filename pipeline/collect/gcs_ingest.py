"""
gcs_ingest.py ??? stream the downloaded side straight out of Cloud Storage
========================================================================
Reads the corpora you already downloaded and parked in GCS, converts them to
manifest records, and stops when the token budget is met.

    gs://lma-01-hi-ne-corpus/raw/hi/sangraha/verified/data.jsonl
    gs://lma-01-hi-ne-corpus/raw/hi/sangraha/unverified/data.jsonl
    gs://lma-01-hi-ne-corpus/raw/hi/wikipedia/data.jsonl
    gs://lma-01-hi-ne-corpus/raw/ne/...

Only Sangraha and Wikipedia are wired in. OpenSubtitles and the mixed
`hi_ne_corpus.zip` are deliberately absent: OpenSubtitles is the English side
of a translation pair (machine- and human-translated subtitle text, which is
exactly the register you do not want in a monolingual LM), and a zip that mixes
both languages violates "do not share documents across corpora" unless it is
split first. Both exclusions, with reasons, are recorded under
`excluded_sources:` in <lang>/configs/data_config.yaml; add a source back by
adding it to the `gcs.sources:` list there.

WHY THIS STREAMS INSTEAD OF DOWNLOADING
---------------------------------------
Sangraha unverified is tens of GB per language. `gsutil cp` then read means you
pay for the whole file before you know whether you needed the last 90% of it --
and with a 400M-token cap you usually do not. `blob.open("rt")` pulls chunks
over HTTP and this module stops reading the moment the budget is hit, so the
bytes you pay to transfer are roughly the bytes you keep.

The cost of streaming is that a restart re-reads from the top. A state file
(<lang>/data/stats/gcs_ingest_state.json, written after every source and on
interrupt) records how many lines of each blob were consumed, so a restart
skips forward without re-parsing JSON. `--no-resume` ignores it.

THE BUDGET, AND WHY IT OVERSHOOTS ON PURPOSE
--------------------------------------------
--target-tokens is the budget for the FINAL corpus. Cleaning, quality gates and
deduplication in `build` remove a large minority of what comes in, so ingesting
exactly 400M tokens' worth here lands well under 400M after build. The
--overcollect multiplier (default 1.45) compensates. The authoritative trim to
exactly 400M happens in build_corpus, after dedup, where the numbers are real.

Tokens are estimated here as characters / --chars-per-token (default 4.0, a
Devanagari-on-SentencePiece rule of thumb). That is a PROXY. After the
tokenizer exists, count_corpus_tokens.py prints the measured ratio; feed it
back on the second pass.

SOURCE PRIORITY
---------------
Sources fill in priority order, so the budget is spent on the best data first:

    wikipedia           (1) edited prose, taken in full -- it is small
    sangraha verified   (2) AI4Bharat's human-verified split
    sangraha unverified (3) the elastic filler that absorbs whatever is left

If Wikipedia and verified Sangraha alone exceed the budget, unverified is never
touched and that is the correct outcome.

Usage
-----
  python -m pipeline.collect.gcs_ingest --lang hindi --repo-root . \\
      --target-tokens 400000000

  # inspect the layout and per-source sizes without writing anything
  python -m pipeline.collect.gcs_ingest --lang hindi --dry-run

  # point at a local directory with the same tree (offline testing)
  python -m pipeline.collect.gcs_ingest --lang hindi \\
      --bucket-root /tmp/fake-bucket --target-tokens 1000000
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.manifest import Document, ShardWriter  # noqa: E402

DEFAULT_BUCKET_ROOT = "gs://lma-01-hi-ne-corpus/raw"

LANG_CODE = {"hindi": "hi", "nepali": "ne"}

# (key, relative path under <bucket-root>/<code>/, priority, hf repo it came from)
SOURCES = [
    ("wikipedia",           "wikipedia/data.jsonl",           1, "wikimedia/wikipedia"),
    ("sangraha_verified",   "sangraha/verified/data.jsonl",   2, "ai4bharat/sangraha"),
    ("sangraha_unverified", "sangraha/unverified/data.jsonl", 3, "ai4bharat/sangraha"),
]

TEXT_KEYS = ("text", "content", "raw_content", "body", "article", "sentence")

DEV_RE = re.compile(r"[???-???]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------
# blob access: gs:// via google-cloud-storage, anything else via the filesystem
# ---------------------------------------------------------------------------

def _split_gs(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


class Blob:
    """One remote or local JSONL file, opened lazily as a text stream."""

    def __init__(self, uri: str):
        self.uri = uri
        self.is_gs = uri.startswith("gs://")

    def exists(self) -> bool:
        if not self.is_gs:
            return Path(self.uri).exists()
        try:
            from google.cloud import storage
        except ImportError:
            return True          # cannot check; let open() produce the real error
        bucket, key = _split_gs(self.uri)
        return storage.Client().bucket(bucket).blob(key).exists()

    def size(self) -> int | None:
        if not self.is_gs:
            p = Path(self.uri)
            return p.stat().st_size if p.exists() else None
        try:
            from google.cloud import storage
        except ImportError:
            return None
        bucket, key = _split_gs(self.uri)
        b = storage.Client().bucket(bucket).blob(key)
        b.reload()
        return b.size

    def open_text(self):
        if not self.is_gs:
            p = Path(self.uri)
            if p.name.endswith(".gz"):
                return gzip.open(p, "rt", encoding="utf-8", errors="replace")
            return open(p, "rt", encoding="utf-8", errors="replace")

        try:
            from google.cloud import storage
        except ImportError as e:
            raise SystemExit(
                "google-cloud-storage is not installed. Either\n"
                "    pip install google-cloud-storage\n"
                "or pass --bucket-root pointing at a local copy.") from e

        bucket, key = _split_gs(self.uri)
        blob = storage.Client().bucket(bucket).blob(key)
        if key.endswith(".gz"):
            raw = blob.open("rb")
            return io.TextIOWrapper(gzip.GzipFile(fileobj=raw),
                                    encoding="utf-8", errors="replace")
        # chunk_size controls how much is buffered per HTTP range request.
        return blob.open("rt", encoding="utf-8", errors="replace",
                         chunk_size=8 * 1024 * 1024)


def load_config(repo_root: Path, lang: str) -> dict:
    """
    <lang>/configs/data_config.yaml, if present and readable.

    The config is authoritative for bucket layout and budgets; CLI flags
    override it. Falling back to the SOURCES constant keeps this runnable in a
    bare checkout, but the YAML is where the layout is meant to be edited.
    """
    p = repo_root / lang / "configs" / "data_config.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"  [warn] could not read {p}: {e}")
        return {}


def resolve_sources(bucket_root: str, lang: str, only: list[str] | None,
                    cfg: dict) -> list[dict]:
    gcs_cfg = cfg.get("gcs") or {}
    code = gcs_cfg.get("lang_code") or LANG_CODE[lang]
    root = bucket_root.rstrip("/")

    entries = gcs_cfg.get("sources")
    if entries:
        spec = [(e["key"], e["path"], e.get("priority", 9),
                 e.get("hf_repo", "?")) for e in entries]
    else:
        spec = SOURCES

    out = []
    for key, rel, prio, repo in spec:
        if only and key not in only:
            continue
        out.append({"key": key, "priority": prio, "hf_repo": repo,
                    "uri": f"{root}/{code}/{rel}"})
    out.sort(key=lambda s: s["priority"])
    return out


# ---------------------------------------------------------------------------
# record handling
# ---------------------------------------------------------------------------

def extract_text(rec: dict) -> str:
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def cheap_reject(text: str, min_chars: int, min_dev_ratio: float) -> str | None:
    """
    A deliberately shallow gate. build_corpus does the real filtering; this only
    exists to stop obviously unusable text from being written to disk and read
    back again. Anything expensive belongs downstream, not in the hot loop of a
    30 GB stream.
    """
    if len(text) < min_chars:
        return "too_short"
    letters = LETTER_RE.findall(text[:2000])
    if not letters:
        return "no_letters"
    dev = sum(1 for c in letters if DEV_RE.match(c))
    if dev / len(letters) < min_dev_ratio:
        return "not_devanagari"
    return None


# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[2],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    # None means "take it from data_config.yaml"; a flag means override.
    ap.add_argument("--bucket-root", default=None,
                    help=f"default from data_config.yaml, else "
                         f"{DEFAULT_BUCKET_ROOT}; may be a local path")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="subset of source keys; default all, in priority order")
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="token budget for the FINAL downloaded corpus")
    ap.add_argument("--overcollect", type=float, default=None,
                    help="ingest this multiple of the target, because build "
                         "removes duplicates and low-quality documents")
    ap.add_argument("--chars-per-token", type=float, default=None,
                    help="proxy for token counting before a tokenizer exists; "
                         "replace with the measured value on pass 2")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-devanagari-ratio", type=float, default=0.60)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the state file and re-read every blob from the top")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the resolved URIs and blob sizes, write nothing")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    lang = args.lang
    raw_dir = root / lang / "data" / "raw"
    state_path = root / lang / "data" / "stats" / "gcs_ingest_state.json"

    cfg = load_config(root, lang)
    targets = cfg.get("targets") or {}

    def pick(cli, section, key, fallback):
        return cli if cli is not None else section.get(key, fallback)

    bucket_root = pick(args.bucket_root, cfg.get("gcs") or {},
                       "bucket_root", DEFAULT_BUCKET_ROOT)
    target_tokens = pick(args.target_tokens, targets,
                         "downloaded_target_tokens", 400_000_000)
    overcollect = pick(args.overcollect, targets, "overcollect", 1.45)
    chars_per_token = pick(args.chars_per_token, targets,
                           "chars_per_token_proxy", 4.0)

    sources = resolve_sources(bucket_root, lang, args.sources, cfg)
    if not sources:
        print(f"[error] no sources resolved. --sources {args.sources} matched "
              f"nothing in data_config.yaml.", file=sys.stderr)
        return 1

    char_budget = int(target_tokens * overcollect * chars_per_token)

    print(f"=== gcs-ingest: {lang} ===")
    print(f"  bucket root   : {bucket_root}")
    print(f"  target tokens : {target_tokens:,}  "
          f"(x{overcollect} overcollect = {int(target_tokens * overcollect):,} ingested)")
    print(f"  char budget   : {char_budget:,}  (at {chars_per_token} chars/token)")
    print(f"  sources       : {[s['key'] for s in sources]}\n")

    # ---- resolve and report ------------------------------------------------
    missing = []
    for s in sources:
        b = Blob(s["uri"])
        ok = b.exists()
        size = b.size() if ok else None
        s["_size"] = size
        print(f"  [{s['priority']}] {s['key']:<22} {s['uri']}")
        print(f"      {'exists' if ok else 'MISSING':<8}"
              + (f"  {size / 1e9:.2f} GB" if size else ""))
        if not ok:
            missing.append(s["key"])
    print()

    if missing:
        print(f"  [warn] not found: {missing}")
        print(f"         check with:  gsutil ls -r {bucket_root.rstrip('/')}/"
              f"{LANG_CODE[lang]}/\n")
    if len(missing) == len(sources):
        print("[error] no readable sources. Nothing to do.", file=sys.stderr)
        return 1

    if args.dry_run:
        total = sum(s["_size"] or 0 for s in sources)
        print(f"  total bytes available: {total / 1e9:.2f} GB")
        print(f"  budget would read roughly {min(char_budget, total) / 1e9:.2f} GB")
        print("  (dry run ??? nothing written)")
        return 0

    # ---- ingest in priority order -----------------------------------------
    state = {} if args.no_resume else load_state(state_path)
    deadline = time.monotonic() + args.max_hours * 3600

    chars_total = 0
    per_source: dict[str, dict] = {}
    rejects: Counter[str] = Counter()
    stopped_early = False

    for s in sources:
        if s["key"] in missing:
            continue
        if chars_total >= char_budget:
            print(f"  [{s['key']}] skipped ??? budget already met by "
                  f"higher-priority sources")
            per_source[s["key"]] = {"documents": 0, "characters": 0,
                                    "skipped": "budget_met"}
            continue

        out_path = raw_dir / f"downloaded_{s['key']}.jsonl"
        writer = ShardWriter(out_path)
        skip_lines = int(state.get(s["key"], {}).get("lines_read", 0)) if not args.no_resume else 0
        if skip_lines:
            print(f"  [{s['key']}] resuming: skipping {skip_lines:,} lines "
                  f"already consumed")

        n_lines = 0
        kept = chars = 0
        t0 = time.monotonic()
        try:
            with Blob(s["uri"]).open_text() as f:
                for line in f:
                    n_lines += 1
                    if n_lines <= skip_lines:
                        continue
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        rejects["bad_json"] += 1
                        continue
                    if not isinstance(rec, dict):
                        rejects["not_object"] += 1
                        continue

                    text = extract_text(rec).strip()
                    why = cheap_reject(text, args.min_chars,
                                       args.min_devanagari_ratio)
                    if why:
                        rejects[why] += 1
                        continue

                    doc = Document(
                        text=text, language=lang,
                        provenance_class="downloaded",
                        source=s["key"],
                        collection_method="hf_download",
                        url=rec.get("url") or rec.get("URL"),
                        title=rec.get("title"),
                        extra={"public_corpus": s["hf_repo"],
                               "gcs_uri": s["uri"]},
                    )
                    if writer.write(doc):
                        kept += 1
                        chars += len(text)
                        chars_total += len(text)

                    if kept and kept % 20000 == 0:
                        rate = kept / max(1e-6, time.monotonic() - t0)
                        print(f"      {kept:,} docs, {chars / 1e6:.0f}M chars "
                              f"({rate:.0f} docs/s), budget "
                              f"{chars_total / char_budget:.0%}", flush=True)

                    if chars_total >= char_budget:
                        print(f"      budget reached inside {s['key']}")
                        stopped_early = True
                        break
                    if time.monotonic() > deadline:
                        print(f"      [stop] --max-hours reached")
                        stopped_early = True
                        break
        except KeyboardInterrupt:
            print("\n  [interrupted] flushing and recording position")
            stopped_early = True
        finally:
            writer.close()
            state[s["key"]] = {"lines_read": n_lines, "documents": kept,
                               "characters": chars, "uri": s["uri"]}
            save_state(state_path, state)

        per_source[s["key"]] = {"documents": kept, "characters": chars,
                                "duplicates_in_file": writer.duplicates,
                                "lines_read": n_lines,
                                "hf_repo": s["hf_repo"], "uri": s["uri"]}
        print(f"  [{s['key']}] {kept:,} documents, {chars / 1e6:.1f}M chars "
              f"-> {out_path.name}")
        if stopped_early and chars_total >= char_budget:
            break

    # ---- summary -----------------------------------------------------------
    est_tokens = chars_total / chars_per_token
    print(f"\n  ingested {chars_total / 1e6:.1f}M characters "
          f"= ~{est_tokens / 1e6:.0f}M tokens (proxy)")
    if rejects:
        print("  rejected at ingest:")
        for k, v in rejects.most_common():
            print(f"    {k:<18} {v:,}")

    summary = {
        "language": lang,
        "bucket_root": bucket_root,
        "target_tokens": target_tokens,
        "overcollect": overcollect,
        "chars_per_token_proxy": chars_per_token,
        "characters_ingested": chars_total,
        "estimated_tokens_ingested": int(est_tokens),
        "stopped_on_budget": chars_total >= char_budget,
        "sources": per_source,
        "rejected_at_ingest": dict(rejects),
        "excluded_deliberately": {
            "opus/OpenSubtitles_en-hi": "translated subtitle text; wrong register "
                                        "for a monolingual LM",
            "hi_ne_corpus.zip": "mixes both languages; the brief forbids sharing "
                                "documents across corpora",
        },
        "note": ("Token figures here are a character proxy. The authoritative "
                 "count comes from pipeline/tokenizer/count_corpus_tokens.py "
                 "after the tokenizer is trained."),
    }
    sp = root / lang / "data" / "stats" / "gcs_ingest_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"  wrote {sp}")

    # ---- what this implies for the requirement -----------------------------
    need_manual = est_tokens * 0.25
    print(f"\n  --- the >=20% constraint ---")
    print(f"  This is the DOWNLOADED side ({est_tokens / 1e6:.0f}M tokens before dedup).")
    print(f"  It needs >= {need_manual / 1e6:.0f}M manual tokens beside it, or the")
    print(f"  downloaded side gets trimmed in `build` until the ratio holds.")
    print(f"  Next:  python run_phase1.py --lang {lang} --stage discover,plan,scrape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
