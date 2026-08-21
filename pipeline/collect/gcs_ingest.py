"""
gcs_ingest.py — stream the downloaded side straight out of Cloud Storage
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
Hindi's verified Sangraha is 173 GB. `gsutil cp` then read means you pay for the
whole file before you know whether you needed the last 98% of it -- and with a
400M-token cap you do not. This reads over HTTP and stops the moment the budget
is met, so the bytes you pay to transfer are roughly the bytes you keep.

WHY IT SAMPLES WINDOWS INSTEAD OF READING FROM THE TOP
-------------------------------------------------------
That budget covers about 2% of a 173 GB blob, and the first 2% of a file
assembled by concatenating sources is not a sample of that file -- it is
whichever source the authors wrote first. A tokenizer trained on it learns one
publisher's vocabulary and calls it Hindi.

So instead of a prefix, the reader divides the blob into --windows evenly
spaced byte ranges (200 by default) and takes an equal share of the budget from
each. Byte offsets never land on line boundaries, so the fragment at the start
of each window is discarded -- 200 lost records out of millions, which is the
entire cost of the scheme. Transfer volume is unchanged; coverage goes from 2%
of the file to all of it.

Sequential reading is still used when the budget would cover most of the blob
anyway (Wikipedia), and `--sampling sequential` forces it if you actually want
a prefix.

A state file (<lang>/data/stats/gcs_ingest_state.json) records lines consumed
and windows completed, so a restart resumes rather than re-reading.
`--no-resume` ignores it.

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

DEV_RE = re.compile(r"[ऀ-ॿ]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------
# blob access: gs:// via google-cloud-storage, anything else via the filesystem
# ---------------------------------------------------------------------------

def _split_gs(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


class Blob:
    """
    One remote or local JSONL file. Supports whole-file streaming and reads of
    arbitrary byte ranges, which is what strided sampling needs.

    The storage Client is cached per instance: constructing one per call adds a
    credential lookup to every window, and strided sampling makes hundreds of
    window reads per source.
    """

    def __init__(self, uri: str):
        self.uri = uri
        self.is_gs = uri.startswith("gs://")
        self._blob = None

    # -- gs:// plumbing ----------------------------------------------------
    def _gs_blob(self):
        if self._blob is None:
            try:
                from google.cloud import storage
            except ImportError as e:
                raise SystemExit(
                    "google-cloud-storage is not installed. Either\n"
                    "    pip install google-cloud-storage\n"
                    "or pass --bucket-root pointing at a local copy.") from e
            bucket, key = _split_gs(self.uri)
            self._blob = storage.Client().bucket(bucket).blob(key)
        return self._blob

    # -- metadata ----------------------------------------------------------
    def exists(self) -> bool:
        if not self.is_gs:
            return Path(self.uri).exists()
        try:
            return self._gs_blob().exists()
        except SystemExit:
            return True          # no library; let the read produce the real error

    def size(self) -> int | None:
        if not self.is_gs:
            p = Path(self.uri)
            return p.stat().st_size if p.exists() else None
        try:
            b = self._gs_blob()
        except SystemExit:
            return None
        b.reload()
        return b.size

    @property
    def is_compressed(self) -> bool:
        return self.uri.endswith(".gz")

    # -- whole-file streaming ---------------------------------------------
    def open_text(self):
        if not self.is_gs:
            p = Path(self.uri)
            if self.is_compressed:
                return gzip.open(p, "rt", encoding="utf-8", errors="replace")
            return open(p, "rt", encoding="utf-8", errors="replace")

        blob = self._gs_blob()
        if self.is_compressed:
            return io.TextIOWrapper(gzip.GzipFile(fileobj=blob.open("rb")),
                                    encoding="utf-8", errors="replace")
        # chunk_size controls how much is buffered per HTTP range request.
        return blob.open("rt", encoding="utf-8", errors="replace",
                         chunk_size=8 * 1024 * 1024)

    # -- byte-range reading, for strided sampling -------------------------
    def read_range(self, start: int, end: int) -> bytes:
        """Bytes in [start, end). Empty bytes at or past EOF."""
        if end <= start:
            return b""
        if not self.is_gs:
            with open(self.uri, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        # GCS ranges are inclusive of `end`.
        return self._gs_blob().download_as_bytes(start=start, end=end - 1)

    def iter_range_lines(self, start: int, end: int, *,
                         skip_partial: bool, chunk: int = 16 * 1024 * 1024):
        """
        Yield complete text lines from the byte range [start, end).

        A byte offset chosen arithmetically almost never lands on a line
        boundary, so the first line of a window is a fragment of the record
        that straddles the offset. `skip_partial` drops it. The final fragment
        is dropped too -- it belongs to the next window, which will read it.

        Losing one record per window is the entire cost of this scheme: at 200
        windows that is 200 documents out of millions.
        """
        pos = start
        carry = b""
        dropped_partial = not skip_partial
        while pos < end:
            stop = min(pos + chunk, end)
            data = self.read_range(pos, stop)
            if not data:
                break
            buf = carry + data
            parts = buf.split(b"\n")
            carry = parts.pop()          # incomplete; may complete next chunk
            for line in parts:
                if not dropped_partial:
                    dropped_partial = True
                    continue
                yield line.decode("utf-8", errors="replace")
            pos = stop


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


def _install_signal_handlers() -> None:
    """
    Turn SIGTERM and SIGHUP into KeyboardInterrupt so the `finally` blocks run.

    This module writes its resume state in a `finally`. By default SIGHUP kills
    the process outright, skipping it -- which is exactly what happens when you
    start a six-hour ingest in a terminal and that terminal later closes. The
    run dies having transferred tens of gigabytes and recorded nothing about
    where it got to. Raising instead lets the state file be written.

    Run long jobs under `tmux` or `nohup` anyway; this only limits the damage
    when you forget.
    """
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass             # not the main thread, or not supported here


def main() -> int:
    # Line-buffer stdout. When stdout is a pipe rather than a terminal -- a
    # notebook cell, `tee`, a redirect -- Python block-buffers it, so a running
    # job prints nothing for minutes and looks hung. Progress output is only
    # useful if it arrives while there is still a decision to make.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    _install_signal_handlers()

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
    ap.add_argument("--sampling", choices=["strided", "sequential"],
                    default=None,
                    help="strided (default) samples windows spread across the "
                         "whole blob; sequential reads from the start. Use "
                         "sequential only if you intend a prefix.")
    ap.add_argument("--windows", type=int, default=None,
                    help="number of windows for strided sampling (default 200)")
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
    gcs_cfg = cfg.get("gcs") or {}
    args.sampling = pick(args.sampling, gcs_cfg, "sampling", "strided")
    args.windows = pick(args.windows, gcs_cfg, "sampling_windows", 200)

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
        print("  (dry run — nothing written)")
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
            print(f"  [{s['key']}] skipped — budget already met by "
                  f"higher-priority sources", flush=True)
            per_source[s["key"]] = {"documents": 0, "characters": 0,
                                    "skipped": "budget_met"}
            continue

        blob = Blob(s["uri"])
        out_path = raw_dir / f"downloaded_{s['key']}.jsonl"
        writer = ShardWriter(out_path)

        remaining = char_budget - chars_total
        size = s.get("_size") or blob.size()

        # ---- decide how to read this blob ---------------------------------
        # Sequential is right when we will read most of the file anyway.
        # Striding is right when the budget covers a small fraction of it:
        # Hindi's verified Sangraha is 173 GB and a 400M-token budget touches
        # about 2% of it, and the FIRST 2% of a file assembled by concatenating
        # sources is not a sample of that file -- it is whichever source the
        # authors happened to write first. Windows spread the same number of
        # bytes across the whole blob.
        #
        # ~4 bytes per kept character: Devanagari is 3 bytes in UTF-8, plus
        # JSON structure and the records we reject.
        est_bytes_needed = remaining * 4
        nwin = 1
        if (args.sampling == "strided" and size and not blob.is_compressed
                and size > est_bytes_needed * 1.5):
            nwin = max(2, args.windows)

        resume = {} if args.no_resume else state.get(s["key"], {})
        first_window = int(resume.get("windows_done", 0))
        skip_lines = int(resume.get("lines_read", 0)) if nwin == 1 else 0

        if nwin > 1:
            print(f"  [{s['key']}] {size / 1e9:.1f} GB, budget covers "
                  f"~{est_bytes_needed / size:.1%} of it -> sampling "
                  f"{nwin} windows spread across the whole file", flush=True)
            if first_window:
                print(f"      resuming from window {first_window}", flush=True)
        else:
            print(f"  [{s['key']}] reading sequentially", flush=True)
            if skip_lines:
                print(f"      resuming: skipping {skip_lines:,} lines already "
                      f"consumed", flush=True)

        win_size = (size // nwin) if (size and nwin > 1) else 0
        per_win_chars = max(1, remaining // nwin)

        n_lines = 0
        kept = chars = 0
        windows_done = first_window
        t0 = time.monotonic()

        def line_source(wi: int):
            """Lines for window `wi`, or the whole blob when nwin == 1."""
            if nwin == 1:
                return blob.open_text()
            w_start = wi * win_size
            w_end = min(size, w_start + win_size)
            return blob.iter_range_lines(w_start, w_end, skip_partial=wi > 0)

        try:
            for wi in range(first_window, nwin):
                if chars_total >= char_budget or time.monotonic() > deadline:
                    break
                win_chars = 0
                src = line_source(wi)
                try:
                    for line in src:
                        n_lines += 1
                        if nwin == 1 and n_lines <= skip_lines:
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
                                   "window": wi if nwin > 1 else None},
                        )
                        wrote_this_doc = writer.write(doc)
                        if wrote_this_doc:
                            kept += 1
                            chars += len(text)
                            win_chars += len(text)
                            chars_total += len(text)

                        # Gate on `wrote_this_doc`, not on `kept` alone: a
                        # duplicate arriving right after a multiple of 20000
                        # leaves `kept` unchanged, so the condition would still
                        # hold and the same line would print twice.
                        if wrote_this_doc and kept % 20000 == 0:
                            rate = kept / max(1e-6, time.monotonic() - t0)
                            print(f"      {kept:,} docs, {chars / 1e6:.0f}M chars "
                                  f"({rate:.0f} docs/s), budget "
                                  f"{chars_total / char_budget:.0%}"
                                  + (f", window {wi + 1}/{nwin}" if nwin > 1 else ""),
                                  flush=True)

                        if chars_total >= char_budget:
                            print(f"      budget reached inside {s['key']}",
                                  flush=True)
                            stopped_early = True
                            break
                        if time.monotonic() > deadline:
                            print(f"      [stop] --max-hours reached", flush=True)
                            stopped_early = True
                            break
                        if nwin > 1 and win_chars >= per_win_chars:
                            break            # this window has paid its share
                finally:
                    if hasattr(src, "close"):
                        src.close()
                windows_done = wi + 1
                if stopped_early:
                    break
        except KeyboardInterrupt:
            print("\n  [interrupted] flushing and recording position", flush=True)
            stopped_early = True
        finally:
            writer.close()
            state[s["key"]] = {"lines_read": n_lines, "documents": kept,
                               "characters": chars, "uri": s["uri"],
                               "windows": nwin, "windows_done": windows_done}
            save_state(state_path, state)

        per_source[s["key"]] = {"documents": kept, "characters": chars,
                                "duplicates_in_file": writer.duplicates,
                                "lines_read": n_lines,
                                "sampling": "strided" if nwin > 1 else "sequential",
                                "windows": nwin,
                                "windows_completed": windows_done,
                                "blob_bytes": size,
                                "hf_repo": s["hf_repo"], "uri": s["uri"]}
        print(f"  [{s['key']}] {kept:,} documents, {chars / 1e6:.1f}M chars "
              f"-> {out_path.name}", flush=True)
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
