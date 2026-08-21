"""
make_reports.py — generate every Phase 1 deliverable report from the artifacts
===============================================================================
Reads the JSON that each pipeline stage already writes and emits the eight
required markdown reports plus an updated README.

    python tools/make_reports.py --repo-root .

WHY THIS GENERATES RATHER THAN TEMPLATES
-----------------------------------------
Every number here has exactly one source: a stats file written by the stage
that measured it. Nothing is typed in by hand, so nothing can drift out of
sync with the corpus, and re-running after a rebuild refreshes everything.

MEASURED vs NOT MEASURED
------------------------
The brief is explicit: "Clearly distinguish measured values from
estimates/proxies. Do not report estimated token counts as final token counts."

So this never invents a plausible-looking number. A field whose source file
does not exist yet renders as

    ⚠ NOT YET MEASURED — run `<the command that produces it>`

which is safe to commit: it says what is missing and how to get it, instead of
a figure a reader would take as real. Values that ARE proxies (character-based
token estimates from before the tokenizer existed) are labelled ESTIMATE inline
every time they appear.

Run it early for the skeleton, run it again when the pipeline finishes.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LANGS = ("hindi", "nepali")
MISSING = "⚠ **NOT YET MEASURED**"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def need(cmd: str) -> str:
    return f"{MISSING} — run `{cmd}`"


def num(v, fmt: str = ",") -> str:
    return f"{v:{fmt}}" if isinstance(v, (int, float)) else MISSING


def dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=20).stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def dir_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class Ctx:
    """Everything the reports read, loaded once."""

    def __init__(self, root: Path):
        self.root = root
        self.now = datetime.now(timezone.utc)
        self.commit = sh(["git", "-C", str(root), "rev-parse", "HEAD"])
        self.branch = sh(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
        self.dirty = sh(["git", "-C", str(root), "status", "--porcelain"])
        self.verification = load(root / "report" / "verification.json")
        self.d = {}
        for lang in LANGS:
            s = root / lang / "data" / "stats"
            self.d[lang] = {
                "corpus": load(s / "corpus_stats.json"),
                "tokens": load(s / "token_accounting.json"),
                "gcs": load(s / "gcs_ingest_summary.json"),
                "download": load(s / "download_summary.json"),
                "ocr": load(s / "ocr_summary.json"),
                "scrape": load(s / "scrape_summary.json"),
                "vocab": load(root / lang / "tokenizer" / "analysis" / "vocab_selection.json"),
                "tokstats": load(root / lang / "tokenizer" / "analysis" / "token_stats.json"),
                "raw_bytes": dir_bytes(root / lang / "data" / "raw"),
                "split_bytes": dir_bytes(root / lang / "data" / "splits"),
            }

    def header(self, title: str) -> list[str]:
        return [
            f"# {title}", "",
            f"| | |", "|---|---|",
            f"| Generated (UTC) | {self.now:%Y-%m-%d %H:%M:%S} |",
            f"| Git commit | `{self.commit}` |",
            f"| Branch | `{self.branch}` |",
            f"| Working tree | {'**dirty** — uncommitted changes present' if self.dirty else 'clean'} |",
            "",
            "> Values marked **ESTIMATE** are character-based proxies computed "
            "before a tokenizer existed. Final token counts are measured by "
            "encoding the final corpus with the final tokenizer "
            "(`count_corpus_tokens.py`).", "",
        ]


# ---------------------------------------------------------------------------
# 1. data collection
# ---------------------------------------------------------------------------

def r_collection(c: Ctx) -> str:
    L = c.header("Phase 1 — Data Collection Report")
    for lang in LANGS:
        d = c.d[lang]
        L += [f"## {lang.title()}", ""]

        L += ["### Downloaded sources (streamed from GCS)", ""]
        g = d["gcs"]
        if not g:
            L += [need(f"python run_phase1.py --lang {lang} --stage gcs-ingest"), ""]
        else:
            L += [f"Bucket root: `{g.get('bucket_root')}`  ",
                  f"Sampling: windowed byte-range sampling "
                  f"(see `gcs.sampling` in `{lang}/configs/data_config.yaml`)", "",
                  "| Source | HF origin | Documents | Characters | Sampling | Windows done |",
                  "|---|---|--:|--:|---|--:|"]
            for k, v in (g.get("sources") or {}).items():
                L.append(f"| `{k}` | {v.get('hf_repo','—')} | {num(v.get('documents'))} | "
                         f"{num(v.get('characters'))} | {v.get('sampling', v.get('skipped','—'))} | "
                         f"{num(v.get('windows_completed'))} |")
            L += ["",
                  f"- Characters ingested: **{num(g.get('characters_ingested'))}**",
                  f"- Token estimate at ingest: **{num(g.get('estimated_tokens_ingested'))}** "
                  f"(ESTIMATE, {g.get('chars_per_token_proxy')} chars/token proxy)",
                  f"- Stopped on budget: {g.get('stopped_on_budget')}", ""]
            if g.get("rejected_at_ingest"):
                L += ["Rejected at ingest:", "", "| Reason | Documents |", "|---|--:|"]
                for k, v in sorted(g["rejected_at_ingest"].items(), key=lambda kv: -kv[1]):
                    L.append(f"| {k} | {num(v)} |")
                L.append("")
            if g.get("excluded_deliberately"):
                L += ["Sources present in the bucket and deliberately excluded:", ""]
                for k, why in g["excluded_deliberately"].items():
                    L.append(f"- `{k}` — {why}")
                L.append("")

        L += ["### Manual sources", ""]
        raws = sorted((c.root / lang / "data" / "raw").glob("manual_*.jsonl"))
        if not raws:
            L += [need(f"python run_phase1.py --lang {lang} --stage scrape"), ""]
        else:
            L += ["| File | Documents | Characters | Size (MB) |", "|---|--:|--:|--:|"]
            for p in raws:
                n = ch = 0
                for line in p.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ch += len(json.loads(line).get("text") or "")
                        n += 1
                    except json.JSONDecodeError:
                        continue
                L.append(f"| `{p.name}` | {num(n)} | {num(ch)} | "
                         f"{p.stat().st_size / 1e6:,.1f} |")
            L.append("")

        L += ["### Raw data size", "",
              f"- `{lang}/data/raw/` — **{d['raw_bytes'] / 1e9:.2f} GB**",
              f"- `{lang}/data/splits/` — **{d['split_bytes'] / 1e9:.2f} GB**", ""]

        L += ["### Token counts", ""]
        t = d["tokens"]
        if not t:
            L += [need(f"python pipeline/tokenizer/count_corpus_tokens.py --lang {lang} --repo-root ."), ""]
        else:
            tot = t["totals"]
            L += ["| Metric | Value | Status |", "|---|--:|---|",
                  f"| Corpus tokens (all splits) | {num(tot['corpus_tokens_all_splits'])} | **MEASURED** |",
                  f"| Training tokens | {num(tot['train_tokens'])} | **MEASURED** |",
                  f"| Target | {num(tot['target_tokens'])} | — |",
                  f"| % of target | {tot['pct_of_target']}% | **MEASURED** |", "",
                  f"Measured with `{Path(t['tokenizer']['model_file']).name}` "
                  f"(vocab {t['tokenizer']['vocab_size']:,}). Counts are valid only "
                  f"for this tokenizer.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 2. provenance
# ---------------------------------------------------------------------------

def r_provenance(c: Ctx) -> str:
    L = c.header("Phase 1 — Data Provenance Report")
    L += ["## Source → local → processed mapping", "",
          "Every record carries `provenance_class`, `source`, "
          "`collection_method` and `collected_at`; the mapping below is "
          "therefore inspectable per document, not only in aggregate "
          "(see `pipeline/manifest.py`).", ""]
    for lang in LANGS:
        g = c.d[lang]["gcs"]
        L += [f"### {lang.title()}", "",
              "| GCS object | Local raw file | Processed into | provenance_class |",
              "|---|---|---|---|"]
        if g:
            for k, v in (g.get("sources") or {}).items():
                L.append(f"| `{v.get('uri','—')}` | `{lang}/data/raw/downloaded_{k}.jsonl` "
                         f"| `{lang}/data/splits/*.jsonl` | `downloaded` |")
        for p in sorted((c.root / lang / "data" / "raw").glob("manual_*.jsonl")):
            method = "scrape" if "scrape" in p.name else "ocr" if "ocr" in p.name else "manual"
            src = ("seed_domains.txt → sitemaps → article URLs" if method == "scrape"
                   else "pdf_sources.txt → pdf_harvest → PDFs" if method == "ocr"
                   else "—")
            L.append(f"| {src} | `{lang}/data/raw/{p.name}` | "
                     f"`{lang}/data/splits/*.jsonl` | `manual` |")
        L.append("")

    L += ["## Dataset / licence information", "",
          "| Source | Origin | Licence | Notes |", "|---|---|---|---|",
          "| Sangraha (verified, unverified) | `ai4bharat/sangraha` | CC-BY-4.0 "
          "(verify at source) | Built from web crawls of Indic publishers |",
          "| Wikipedia | `wikimedia/wikipedia` | CC-BY-SA-3.0 / GFDL | Edited prose |",
          "| Manual scrape | publisher sites in `<lang>/configs/seed_domains.txt` "
          "| per-site terms; `robots.txt` honoured incl. `Crawl-delay` | "
          "Collected by this project |",
          "| Manual OCR | government publications in `<lang>/configs/pdf_sources.txt` "
          "| government publications, generally permissive — verify per source | "
          "Text-layer extraction, OCR fallback |", "",
          "> Licence column records the commonly stated licence for each upstream "
          "dataset. Confirm against the source before redistribution; this "
          "project consumes the data rather than republishing it.", "",
          "## Version", "",
          f"- Commit: `{c.commit}`",
          f"- Branch: `{c.branch}`",
          f"- Generated: {c.now:%Y-%m-%d %H:%M:%S} UTC",
          f"- Working tree: {'dirty' if c.dirty else 'clean'}", ""]
    if c.dirty:
        L += ["", "```", c.dirty[:2000], "```", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 3. cleaning
# ---------------------------------------------------------------------------

def r_cleaning(c: Ctx) -> str:
    L = c.header("Phase 1 — Data Cleaning Report")
    L += ["## Methodology", "",
          "Seven ordered stages in `pipeline/process/build_corpus.py`:", "",
          "1. **Normalise** — NFC, nukta recomposition, zero-width strip, "
          "whitespace collapse, danda spacing",
          "2. **Validate** — length, Devanagari ratio, Latin ratio, "
          "Hindi-vs-Nepali marker discrimination",
          "3. **Quality** — Gopher-style repetition, mean word length, symbol "
          "ratio, duplicate-line ratio",
          "4. **Exact dedup** — BLAKE2b over the aggressively normalised form "
          "(case-folded, punctuation and danda stripped)",
          "5. **Near dedup** — MinHash over word 5-grams with banded LSH, "
          "global across every source and both provenance classes",
          "6. **Budget trim** — to the manual/downloaded token targets",
          "7. **Split** — train/val/test stratified by "
          "`(provenance_class, source)`", "",
          "### Why dedup runs before the split, and why it is global", "",
          "If a document appears in both Sangraha and the manual scrape and "
          "dedup ran after splitting, the two copies could land in train and "
          "test. Phase 2 perplexity would then measure memorisation rather than "
          "language modelling, and that is not recoverable after the fact. "
          "Sangraha is itself built from crawls of the same publishers the "
          "scraper visits, so this overlap is expected rather than hypothetical.", "",
          "### Deduplication tie-break", "",
          "`iter_language_docs` yields manual files first. Deduplication keeps "
          "whichever copy it sees first, so when the same article arrives from "
          "both sides the **manual** copy survives and the downloaded duplicate "
          "is discarded — same corpus, same document count, but the provenance "
          "credit lands on the manually collected side.", ""]

    for lang in LANGS:
        cs = c.d[lang]["corpus"]
        L += [f"## {lang.title()}", ""]
        if not cs:
            L += [need(f"python -m pipeline.process.build_corpus --lang {lang} --repo-root ."), ""]
            continue
        raw, kept = cs["raw_documents"], cs["kept_documents"]
        final = cs.get("documents_in_final_corpus")
        L += ["### Before / after", "",
              "| Stage | Documents | Retention |", "|---|--:|--:|",
              f"| Raw collected | {num(raw)} | — |",
              f"| After cleaning + dedup | {num(kept)} | {kept / max(1, raw):.1%} |",
              f"| In final corpus (after budget trim) | {num(final)} | "
              f"{(final / max(1, raw)) if isinstance(final, int) else 0:.1%} |", "",
              "### Removed, and why", "",
              "| Reason | Documents |", "|---|--:|"]
        for k, v in sorted(cs["counters"].items(), key=lambda kv: -kv[1]):
            if k.startswith("drop:"):
                L.append(f"| {k[5:].replace('_', ' ')} | {num(v)} |")
        for k in ("empty", "wrong_language_field", "unlabelled_provenance"):
            if cs["counters"].get(k):
                L.append(f"| {k.replace('_', ' ')} | {num(cs['counters'][k])} |")
        L += ["",
              f"Boilerplate lines removed in place: "
              f"**{num(cs['counters'].get('boilerplate_lines_removed'))}**", "",
              "### Characters by split and provenance", "",
              "| Split | Documents | Manual chars | Downloaded chars |",
              "|---|--:|--:|--:|"]
        for s in ("train", "val", "test"):
            cc = cs["split_characters_by_class"].get(s, {})
            L.append(f"| {s} | {num(cs['splits'].get(s))} | "
                     f"{num(cc.get('manual', 0))} | {num(cc.get('downloaded', 0))} |")
        L.append("")
        b = cs.get("budget")
        if b:
            L += ["### Budget trim", "",
                  f"- Targeted **{b.get('effective_fraction_targeted', 0):.1%}** manual "
                  f"(requirement {b.get('min_manual_fraction', 0):.0%} + "
                  f"{b.get('margin', 0):.1%} margin for character/token drift)",
                  f"- Manual kept: {num(b['kept_chars']['manual'])} of "
                  f"{num(b['available_chars']['manual'])} chars available",
                  f"- Downloaded kept: {num(b['kept_chars']['downloaded'])} of "
                  f"{num(b['available_chars']['downloaded'])} chars available",
                  f"- Binding constraint: **{b.get('downloaded_budget_binding_constraint')}**", ""]
        if cs["config"].get("near_dup_threshold"):
            L += [f"MinHash Jaccard threshold: "
                  f"**{cs['config']['near_dup_threshold']}**, word 5-grams.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 4. manual data
# ---------------------------------------------------------------------------

def r_manual(c: Ctx) -> str:
    L = c.header("Phase 1 — Manual Data Report")
    L += ["Manual collection means text this project gathered itself: pages "
          "scraped and cleaned, and PDFs whose text was extracted. It is "
          "distinguished from downloaded public corpora by the "
          "`provenance_class` field, set at ingest and never inferred "
          "afterwards.", ""]
    for lang in LANGS:
        L += [f"## {lang.title()}", ""]
        cfg_seed = c.root / lang / "configs" / "seed_domains.txt"
        cfg_pdf = c.root / lang / "configs" / "pdf_sources.txt"
        for label, p in (("Scraping seed domains", cfg_seed),
                         ("PDF seed pages", cfg_pdf)):
            if p.exists():
                doms = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                        if l.strip() and not l.startswith("#")]
                L += [f"**{label}:** {len(doms)} (`{p.relative_to(c.root)}`)", ""]

        urlf = c.root / lang / "data" / "raw" / "article_urls.txt"
        if urlf.exists():
            urls = [l for l in urlf.read_text(encoding="utf-8").splitlines() if l.strip()]
            from urllib.parse import urlsplit
            hosts = {urlsplit(u).netloc for u in urls}
            L += [f"**URLs discovered:** {len(urls):,} across {len(hosts)} hosts", ""]

        L += ["### Documents retained", "", "| Source file | Documents | Characters |",
              "|---|--:|--:|"]
        tot_docs = tot_ch = 0
        for p in sorted((c.root / lang / "data" / "raw").glob("manual_*.jsonl")):
            n = ch = 0
            for line in p.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    ch += len(json.loads(line).get("text") or "")
                    n += 1
                except json.JSONDecodeError:
                    continue
            tot_docs += n
            tot_ch += ch
            L.append(f"| `{p.name}` | {num(n)} | {num(ch)} |")
        L += [f"| **total** | **{num(tot_docs)}** | **{num(tot_ch)}** |", ""]

        t = c.d[lang]["tokens"]
        if t:
            m = t["manual_vs_downloaded"]
            L += [f"**Manual tokens (MEASURED, final tokenizer):** "
                  f"{num(m['manual_tokens'])} "
                  f"({m['manual_fraction_of_tokens']:.2%} of corpus tokens)", "",
                  f"Requirement ≥{m['requirement']:.0%} — "
                  f"**{'MET' if m['requirement_met'] else 'NOT MET'}**", ""]
        else:
            L += [need(f"python pipeline/tokenizer/count_corpus_tokens.py --lang {lang} --repo-root ."), ""]

        L += ["### Failures", ""]
        if c.d[lang]["scrape"]:
            L += ["```", json.dumps(c.d[lang]["scrape"], indent=2, ensure_ascii=False)[:2000], "```", ""]
        else:
            L += ["Scrape and OCR failure counts are printed by their stages "
                  "(robots-denied, HTTP errors, extraction failures, wrong "
                  "language, too short, and for OCR `SKIP:no_text_layer`). "
                  "Paste the run summary here — it is not currently written to "
                  "a JSON file.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5. tokenizer
# ---------------------------------------------------------------------------

def r_tokenizer(c: Ctx) -> str:
    L = c.header("Phase 1 — Tokenizer Report")
    L += ["## Algorithm and configuration", "",
          "SentencePiece, trained **from scratch** on the training split only. "
          "No pretrained tokenizer or model is used anywhere in this project.", "",
          "Key settings (full config in `<lang>/configs/tokenizer_config.yaml`):", "",
          "| Setting | Value | Why |", "|---|---|---|",
          "| `model_type` | `unigram` | Kudo (2018); probabilistic segmentation |",
          "| `byte_fallback` | `true` | Makes UNK structurally zero — unseen "
          "characters decompose to UTF-8 byte pieces |",
          "| `normalization_rule_name` | `identity` | Normalisation happens once, "
          "in `build_corpus.normalize()`. Two normalisers in sequence means "
          "stored text and tokenizer-visible text diverge |",
          "| `split_digits` | `true` | Stops numerals consuming merge budget |",
          "| `character_coverage` | `0.9999` | Devanagari plus incidental Latin |", "",
          "### Why byte-fallback rate replaces UNK rate", "",
          "With `byte_fallback=True` the UNK rate is structurally zero, so it "
          "cannot discriminate between candidate vocabularies. The successor "
          "metric is the **byte-fallback rate**: the fraction of emitted tokens "
          "that are raw `<0xNN>` pieces. On Devanagari this is the quantity that "
          "tracks token inflation — stranded single bytes are what make "
          "English-centric tokenizers expensive on Indic scripts.", ""]

    for lang in LANGS:
        v, ts, t = c.d[lang]["vocab"], c.d[lang]["tokstats"], c.d[lang]["tokens"]
        L += [f"## {lang.title()}", ""]
        if not v:
            L += [need(f"python pipeline/tokenizer/train_tokenizer.py --lang {lang} --repo-root ."), ""]
            continue
        L += ["### Vocabulary sweep", "",
              "| Vocab | Algorithm | Fertility | Δ vs prev | chars/token | "
              "byte-fallback | Embedding share | Selected |",
              "|--:|---|--:|--:|--:|--:|--:|:-:|"]
        for cand in v["candidates"]:
            g = cand["relative_fertility_gain_vs_prev"]
            L.append(f"| {cand['vocab_size']:,} | {cand['model_type']} | "
                     f"{cand['fertility']} | {(f'{g:.2%}' if g != '' else '—')} | "
                     f"{cand['chars_per_token']} | {cand['byte_fallback_rate']:.2e} | "
                     f"{cand['embedding_share']:.1%} | "
                     f"{'**yes**' if cand['selected'] else ''} |")
        L += ["",
              f"### Selected: **{v['selected_vocab_size']:,}**", "",
              f"{v['selection_reason']}", "",
              f"Scaling-law reference (Tao et al. 2024, "
              f"N_v ∝ N_nv^0.83): **{v['scaling_law_reference_vocab']:,}** "
              f"— selected size is {v['ratio_to_scaling_law_reference']}× that.", "",
              "Selection is an elbow on marginal fertility gain, not "
              "minimum fertility. Fertility falls monotonically with vocabulary "
              "size, so 'lowest fertility' would always return the largest size "
              "swept and the sweep would be decorative.", ""]
        if ts:
            e, u, vv = ts["efficiency"], ts["unknown_tokens"], ts["vocabulary"]
            L += ["### Measurements on the held-out test split", "",
                  "| Metric | Value |", "|---|--:|",
                  f"| Fertility (tokens/word) | {e['fertility_tokens_per_word']} |",
                  f"| Characters per token | {e['avg_chars_per_token']} |",
                  f"| Bytes per token | {e['avg_bytes_per_token']} |",
                  f"| UNK rate | {u['unk_rate']:.2e} |",
                  f"| Byte-fallback rate | {u['byte_fallback_rate']:.2e} |",
                  f"| Vocabulary utilisation | {vv['vocab_utilisation']:.1%} |",
                  f"| Pieces never used | {num(vv['unused_pieces'])} |", ""]
            if ts.get("fertility_by_provenance"):
                L += ["#### Fertility by provenance", "", "| Provenance | Fertility |",
                      "|---|--:|"]
                for k, val in ts["fertility_by_provenance"].items():
                    L.append(f"| {k} | {val} |")
                L += ["", "_If manual fertility is materially worse than "
                      "downloaded, the tokenizer is under-serving the data "
                      "collected by hand._", ""]
        if t:
            L += [f"### Training corpus size", "",
                  f"- Training tokens: **{num(t['totals']['train_tokens'])}** (MEASURED)",
                  f"- Measured chars/token: **{dig(t, 'measured_chars_per_token', 'overall')}** "
                  f"— by provenance: {dig(t, 'measured_chars_per_token', 'by_provenance')}", ""]
        ex = c.root / lang / "tokenizer" / "analysis" / "examples.md"
        L += ["### Example tokenizations", ""]
        if ex.exists():
            L += [ex.read_text(encoding="utf-8")[:4000], ""]
        else:
            L += [need(f"python pipeline/tokenizer/tokenizer_report.py --lang {lang} --repo-root . --split test"), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 6. final statistics
# ---------------------------------------------------------------------------

def r_final_stats(c: Ctx) -> str:
    L = c.header("Phase 1 — Final Corpus Statistics")
    L += ["## Summary", "",
          "| | Hindi | Nepali |", "|---|--:|--:|"]
    rows = [
        ("Final corpus tokens (MEASURED)",
         lambda l: num(dig(c.d[l]["tokens"], "totals", "corpus_tokens_all_splits"))),
        ("Training tokens (MEASURED)",
         lambda l: num(dig(c.d[l]["tokens"], "totals", "train_tokens"))),
        ("Manual tokens (MEASURED)",
         lambda l: num(dig(c.d[l]["tokens"], "manual_vs_downloaded", "manual_tokens"))),
        ("Downloaded tokens (MEASURED)",
         lambda l: num(dig(c.d[l]["tokens"], "manual_vs_downloaded", "downloaded_tokens"))),
        ("Manual share of tokens",
         lambda l: (f"{dig(c.d[l]['tokens'], 'manual_vs_downloaded', 'manual_fraction_of_tokens'):.2%}"
                    if dig(c.d[l]["tokens"], "manual_vs_downloaded", "manual_fraction_of_tokens") is not None
                    else MISSING)),
        ("Documents in final corpus",
         lambda l: num(dig(c.d[l]["corpus"], "documents_in_final_corpus"))),
        ("Raw documents collected",
         lambda l: num(dig(c.d[l]["corpus"], "raw_documents"))),
        ("Selected vocabulary size",
         lambda l: num(dig(c.d[l]["vocab"], "selected_vocab_size"))),
        ("Fertility (test split)",
         lambda l: str(dig(c.d[l]["tokstats"], "efficiency", "fertility_tokens_per_word") or MISSING)),
        ("Raw data on disk",
         lambda l: f"{c.d[l]['raw_bytes'] / 1e9:.2f} GB"),
    ]
    for label, fn in rows:
        L.append(f"| {label} | {fn('hindi')} | {fn('nepali')} |")
    L.append("")

    for lang in LANGS:
        cs, t = c.d[lang]["corpus"], c.d[lang]["tokens"]
        L += [f"## {lang.title()} — splits", ""]
        if not cs:
            L += [need(f"python -m pipeline.process.build_corpus --lang {lang} --repo-root ."), ""]
            continue
        L += ["| Split | Documents | Manual chars | Downloaded chars | Tokens (MEASURED) |",
              "|---|--:|--:|--:|--:|"]
        for s in ("train", "val", "test"):
            cc = cs["split_characters_by_class"].get(s, {})
            tok = dig(t, "per_split", s, "total_tokens")
            L.append(f"| {s} | {num(cs['splits'].get(s))} | {num(cc.get('manual', 0))} | "
                     f"{num(cc.get('downloaded', 0))} | {num(tok)} |")
        L.append("")
        if t and dig(t, "per_split", "train", "tokens_by_source"):
            L += ["### Tokens by source (train split)", "", "| Source | Tokens |", "|---|--:|"]
            for k, v in list(dig(t, "per_split", "train", "tokens_by_source").items())[:25]:
                L.append(f"| `{k}` | {num(v)} |")
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 7. reproducibility
# ---------------------------------------------------------------------------

def r_repro(c: Ctx) -> str:
    L = c.header("Phase 1 — Reproducibility Report")
    L += ["## Environment", "", "| | |", "|---|---|",
          f"| Python | {platform.python_version()} |",
          f"| Platform | {platform.platform()} |",
          f"| Processor | {platform.processor() or 'unknown'} |",
          f"| Git commit | `{c.commit}` |",
          f"| Git branch | `{c.branch}` |", "",
          "### Installed packages", "", "```"]
    L += [sh([sys.executable, "-m", "pip", "freeze"])[:6000], "```", "",
          "## Configuration files", ""]
    for lang in LANGS:
        for f in sorted((c.root / lang / "configs").glob("*")):
            if f.is_file():
                L.append(f"- `{f.relative_to(c.root)}`")
    L += ["", "## GCS bucket structure", "", "```",
          "gs://lma-01-hi-ne-corpus/",
          "└── raw/",
          "    ├── hi/",
          "    │   ├── wikipedia/data.jsonl",
          "    │   ├── sangraha/verified/data.jsonl",
          "    │   ├── sangraha/unverified/data.jsonl",
          "    │   ├── opus/OpenSubtitles_en-hi.zip      (excluded — see configs)",
          "    │   └── hi_ne_corpus.zip                  (excluded — see configs)",
          "    └── ne/",
          "        ├── wikipedia/data.jsonl",
          "        ├── sangraha/verified/data.jsonl",
          "        └── sangraha/unverified/data.jsonl",
          "```", "",
          "## Commands to reproduce", "", "```bash",
          "git clone <repo> && cd <repo>",
          f"git checkout {c.commit}",
          "pip install -r requirements.txt",
          "gcloud auth application-default login",
          "gcloud config set project lma-01",
          "",
          "for L in hindi nepali; do",
          "  python run_phase1.py --lang $L --stage gcs-ingest",
          "  python run_phase1.py --lang $L --stage discover,plan",
          "  python run_phase1.py --lang $L --stage scrape --scrape-hours 6",
          "  python -m pipeline.collect.pdf_harvest --lang $L --out-dir ~/${L}_pdfs \\",
          "      --depth 4 --user-agent 'corpus-research/1.0 (+you@example.com)'",
          "  python -m pipeline.collect.ocr_collect --lang $L \\",
          "      --input-dir ~/${L}_pdfs --text-layer-only",
          "  python -m pipeline.process.build_corpus --lang $L --repo-root . --workers 7",
          "  python run_phase1.py --lang $L --stage tokenizer,count,report",
          "done",
          "python -m pipeline.process.verify_corpora --repo-root .",
          "python tools/make_reports.py --repo-root .",
          "```", "",
          "### Determinism", "",
          "- Split assignment: seeded (`--seed`, default 20260820)",
          "- MinHash permutations: seeded from the same constant",
          "- URL shuffle in the scraper: seeded",
          "- SentencePiece: `shuffle_input_sentence` with a fixed sample size", "",
          "Collection stages are **not** deterministic — the live web changes "
          "between runs. Reproducing the exact corpus requires the collected "
          "raw JSONL, not just the code.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 8. validation
# ---------------------------------------------------------------------------

def r_validation(c: Ctx) -> str:
    L = c.header("Phase 1 — Validation Report")
    L += ["## Automated checks", "",
          "Produced by `python -m pipeline.process.verify_corpora --repo-root .`", ""]
    v = c.verification
    if not v:
        L += [need("python -m pipeline.process.verify_corpora --repo-root ."), ""]
    else:
        L += ["| Check | Result | Detail |", "|---|---|---|"]
        for ch in v["checks"]:
            L.append(f"| {ch['name']} | {'✅ PASS' if ch['passed'] else '❌ FAIL'} | "
                     f"{ch['detail'] or '—'} |")
        L += ["", f"**Overall: {'ALL PASSED' if v['all_passed'] else 'FAILURES PRESENT'}**", ""]

    L += ["## What each check defends", "",
          "| Check | Requirement it enforces |", "|---|---|",
          "| C1 | \"Do not share documents across corpora\" — spans both "
          "languages, so no single-language stage can see it |",
          "| C2 | Train/test leakage by `doc_id` |",
          "| C3 | Train/test leakage by identical text under a different id |",
          "| C4 | Every document carries a provenance class, so the manual "
          "fraction is measured rather than lower-bounded |",
          "| C5 | The ≥20% manual token requirement |",
          "| C6 | Tokenizer trained on the train split only — held-out text in "
          "the vocabulary invalidates every downstream fertility and perplexity "
          "number |",
          "| C7 | Identical text across languages, usually a language-ID failure |", "",
          "## JSONL integrity", ""]
    for lang in LANGS:
        L += [f"### {lang.title()}", "", "| File | Lines | Parse failures |", "|---|--:|--:|"]
        for s in ("train", "val", "test"):
            p = c.root / lang / "data" / "splits" / f"{s}.jsonl"
            if not p.exists():
                L.append(f"| `{s}.jsonl` | {MISSING} | — |")
                continue
            n = bad = 0
            for line in p.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
            L.append(f"| `{s}.jsonl` | {num(n)} | {num(bad)} |")
        L.append("")

    L += ["## Tokenizer round-trip", "",
          "`tokenizer_report.py` encodes and decodes held-out text and records "
          "whether the decoded string matches the input exactly. With "
          "`byte_fallback=True` this should hold for every input, since no "
          "character is unrepresentable. See "
          "`<lang>/tokenizer/analysis/examples.md`.", ""]
    for lang in LANGS:
        ex = c.root / lang / "tokenizer" / "analysis" / "examples.md"
        L.append(f"- {lang}: {'`' + str(ex.relative_to(c.root)) + '`' if ex.exists() else MISSING}")
    L += ["", "## Phase 2 consumability", "",
          "The final corpus is newline-delimited JSON with one document per "
          "line and a stable schema (`doc_id`, `text`, `language`, "
          "`provenance_class`, `source`, `collection_method`, `collected_at`). "
          "The tokenizer is a standard SentencePiece `.model` loadable by "
          "`sentencepiece.SentencePieceProcessor`. No custom deserialisation is "
          "required.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def r_readme(c: Ctx) -> str:
    hi, ne = c.d["hindi"], c.d["nepali"]

    def tok(l, *k):
        return num(dig(c.d[l]["tokens"], *k))

    L = ["# Phase 1 — Hindi and Nepali corpora and tokenizers", "",
         f"_Generated {c.now:%Y-%m-%d %H:%M} UTC from commit `{c.commit[:12]}`._", "",
         "## Executive summary", "",
         "Two independent monolingual corpora and two from-scratch SentencePiece "
         "tokenizers, one per language. No pretrained tokenizer or model is used "
         "anywhere. Every document carries its provenance, so the manual versus "
         "downloaded split is measured rather than asserted.", "",
         "The downloaded side streams from Cloud Storage with **windowed byte-range "
         "sampling** rather than reading each blob from the top: at a 400M-token "
         "budget against a 173 GB Sangraha file, a prefix read would have covered "
         "2% of the file and delivered whichever source happened to be written "
         "first. The manual side comes from scraping (robots-respecting, "
         "`Crawl-delay` honoured) and from text extraction over government PDFs.", "",
         "## Statistics", "",
         "| | Hindi | Nepali |", "|---|--:|--:|",
         f"| Corpus tokens (MEASURED) | {tok('hindi','totals','corpus_tokens_all_splits')} | "
         f"{tok('nepali','totals','corpus_tokens_all_splits')} |",
         f"| Training tokens | {tok('hindi','totals','train_tokens')} | "
         f"{tok('nepali','totals','train_tokens')} |",
         f"| Manual tokens | {tok('hindi','manual_vs_downloaded','manual_tokens')} | "
         f"{tok('nepali','manual_vs_downloaded','manual_tokens')} |",
         f"| Downloaded tokens | {tok('hindi','manual_vs_downloaded','downloaded_tokens')} | "
         f"{tok('nepali','manual_vs_downloaded','downloaded_tokens')} |",
         f"| Vocabulary size | {num(dig(hi['vocab'],'selected_vocab_size'))} | "
         f"{num(dig(ne['vocab'],'selected_vocab_size'))} |",
         f"| Fertility (test) | {dig(hi['tokstats'],'efficiency','fertility_tokens_per_word') or MISSING} | "
         f"{dig(ne['tokstats'],'efficiency','fertility_tokens_per_word') or MISSING} |",
         "",
         "## Dataset locations", "", "```",
         "<lang>/data/raw/       collected documents, one JSONL per source",
         "<lang>/data/interim/   cleaned pool before the budget trim",
         "<lang>/data/splits/    train.jsonl / val.jsonl / test.jsonl",
         "<lang>/data/stats/     corpus_stats.json, token_accounting.json, ...",
         "<lang>/tokenizer/vocab/    <lang>_tokenizer.model / .vocab / .json",
         "<lang>/tokenizer/analysis/ sweep_results.csv, token_stats.json, examples.md",
         "report/                the eight Phase 1 reports",
         "```", "",
         "## Reports", "",
         "| File | Contents |", "|---|---|",
         "| `phase1_data_collection_report.md` | Sources, counts, sizes, failures |",
         "| `phase1_provenance_report.md` | GCS paths, source→local→processed mapping, licences |",
         "| `phase1_cleaning_report.md` | Filters, dedup methodology, before/after |",
         "| `phase1_manual_data_report.md` | Scraping and OCR sources, retention |",
         "| `phase1_tokenizer_report.md` | Sweep, selection, fertility, examples |",
         "| `phase1_final_statistics.md` | Final measured counts per language |",
         "| `phase1_reproducibility.md` | Commit, environment, commands |",
         "| `phase1_validation_report.md` | All checks, integrity, round-trip |", "",
         "## Known limitations", "", "_Edit this section before submitting._", "",
         "- **Token counts** are only valid for the tokenizer that produced them. "
         "Retraining the tokenizer invalidates every token figure.",
         "- **Collection is not reproducible** from code alone; the live web "
         "changes. The raw JSONL is the reproducible artifact.",
         "- **OCR** used `--text-layer-only` where noted, so scanned PDFs without "
         "an embedded text layer were skipped rather than rasterised. State the "
         "count if you used it.",
         "- **Nepali is the lower-resource language** and the brief anticipates a "
         "shortfall: report the exact token count and justify it here.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------

REPORTS = [
    ("phase1_data_collection_report.md", r_collection),
    ("phase1_provenance_report.md", r_provenance),
    ("phase1_cleaning_report.md", r_cleaning),
    ("phase1_manual_data_report.md", r_manual),
    ("phase1_tokenizer_report.md", r_tokenizer),
    ("phase1_final_statistics.md", r_final_stats),
    ("phase1_reproducibility.md", r_repro),
    ("phase1_validation_report.md", r_validation),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default="report")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("loading artifacts ...")
    c = Ctx(root)

    missing_total = 0
    for name, fn in REPORTS:
        text = fn(c)
        (out / name).write_text(text, encoding="utf-8")
        n = text.count(MISSING)
        missing_total += n
        print(f"  wrote {args.out_dir}/{name:<42} "
              f"{'(' + str(n) + ' fields not yet measured)' if n else '(complete)'}")

    (root / "README.md").write_text(r_readme(c), encoding="utf-8")
    print(f"  wrote README.md")

    print()
    if missing_total:
        print(f"  {missing_total} field(s) still unmeasured. Each one names the "
              f"command that fills it.")
        print(f"  Re-run this after the pipeline completes and they resolve "
              f"themselves.")
    else:
        print("  every field populated from measured artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
