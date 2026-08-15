"""
quality_filters.py  (v2)
-------------------------
Stage 8: Rule + Metric Quality Filtering.

v2 changes:
  - Full 9-feature set (not just 2-3 heuristics).
  - All thresholds read from filtering_thresholds.yaml with null-detection
    and conservative fallback.
  - Logs per-feature drop counts (to diagnose which filter is most aggressive).
  - Updates manifest drop_stage/drop_reason.
  - For Nepali, logs a WARNING if any single filter drops > 30% of input.

Features computed per document:
  1.  char_count             ??? total characters
  2.  word_count             ??? whitespace-delimited words
  3.  devanagari_ratio       ??? fraction of Devanagari script chars (not generic alpha)
  4.  whitespace_ratio       ??? fraction of whitespace chars
  5.  punct_symbol_ratio     ??? fraction of punctuation + symbol chars
  6.  digit_ratio            ??? fraction of digit chars
  7.  url_ratio              ??? fraction of tokens that look like URLs
  8.  repeated_line_ratio    ??? fraction of non-empty lines that are duplicated within doc
  9.  repeated_ngram_ratio   ??? fraction of 5-grams repeated within doc
  10. mean_line_length       ??? mean chars per non-empty line
  11. paragraph_count        ??? number of paragraphs (split on ???2 newlines)

Usage
-----
  python quality_filters.py \\
      --lang hindi \\
      --input hindi/data/cleaned/lang_filtered.jsonl \\
      --output hindi/data/cleaned/quality_filtered.jsonl \\
      --repo-root .
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import update_rows

_DEVA_RE  = re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]")
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
_PARA_SEP = re.compile(r"\n{2,}")


# ????????? Feature extraction ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def extract_features(text: str) -> dict:
    """
    Compute the full 9-feature quality-filter feature set for a document.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        Keys: char_count, word_count, devanagari_ratio, whitespace_ratio,
              punct_symbol_ratio, digit_ratio, url_ratio,
              repeated_line_ratio, repeated_ngram_ratio,
              mean_line_length, paragraph_count.
    """
    chars  = len(text)
    words  = text.split()
    lines  = text.split("\n")
    nelines = [l.strip() for l in lines if l.strip()]
    paras  = [p.strip() for p in _PARA_SEP.split(text) if p.strip()]

    import unicodedata
    deva  = len(_DEVA_RE.findall(text))
    sp    = sum(1 for c in text if c.isspace())
    dig   = sum(1 for c in text if c.isdigit())
    # Use Unicode P* (Punctuation) and S* (Symbol) categories only.
    # This correctly excludes Devanagari combining marks (Mc, Mn, Cf)
    # which the old subtraction method wrongly counted as punctuation.
    punct_sym = sum(1 for c in text if unicodedata.category(c)[0] in ("P", "S"))
    urls  = len(_URL_RE.findall(text))

    # Repeated lines within document
    lc = Counter(nelines)
    rep_lines = sum(v - 1 for v in lc.values() if v > 1)
    rep_line_r = rep_lines / len(nelines) if nelines else 0.0

    # Repeated 5-grams
    if len(words) >= 5:
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        ngc = Counter(ngrams)
        rep_ng = sum(v - 1 for v in ngc.values() if v > 1)
        rep_ng_r = rep_ng / len(ngrams)
    else:
        rep_ng_r = 0.0

    mean_ll = sum(len(l) for l in nelines) / len(nelines) if nelines else 0.0

    return {
        "char_count":           chars,
        "word_count":           len(words),
        "devanagari_ratio":     deva / chars if chars else 0.0,
        "whitespace_ratio":     sp   / chars if chars else 0.0,
        "punct_symbol_ratio":   punct_sym / chars if chars else 0.0,
        "digit_ratio":          dig  / chars if chars else 0.0,
        "url_ratio":            urls / len(words) if words else 0.0,
        "repeated_line_ratio":  rep_line_r,
        "repeated_ngram_ratio": rep_ng_r,
        "mean_line_length":     round(mean_ll, 1),
        "paragraph_count":      len(paras),
    }


# ????????? Threshold loader ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _get(cfg: dict, *keys, fallback=None):
    """
    Navigate a nested dict and return the leaf value, or fallback if null/missing.

    Parameters
    ----------
    cfg : dict
    *keys : str
    fallback : any

    Returns
    -------
    tuple[any, bool]
        (value, is_fallback)
    """
    node = cfg
    for k in keys:
        node = node.get(k, {}) if isinstance(node, dict) else {}
    val = node if not isinstance(node, dict) else None
    if val is None:
        return fallback, True
    return val, False


def load_thresholds(thresh_cfg: dict, lang: str) -> tuple[dict, bool]:
    """
    Load all quality-filter thresholds from config, using fallbacks where null.

    Parameters
    ----------
    thresh_cfg : dict
        Parsed filtering_thresholds.yaml content.
    lang : str
        Used for contextual warning messages.

    Returns
    -------
    tuple[dict, bool]
        (thresholds dict, any_fallback_used)
    """
    qf = thresh_cfg.get("quality_filters", {})
    any_fb = False

    def _t(key, fb_key="conservative_fallback"):
        node = qf.get(key, {})
        val  = node.get("value")
        fb   = node.get(fb_key, 0.0)
        if val is None:
            print(
                f"[quality_filters] WARNING: quality_filters.{key}.value is null "
                f"in filtering_thresholds.yaml. Using fallback: {fb}",
                file=sys.stderr,
            )
            return fb, True
        return val, False

    results = {}
    for name, fb in [
        ("min_char_count",          150),
        ("min_devanagari_ratio",    0.45),
        ("max_whitespace_ratio",    0.35),
        ("max_punct_symbol_ratio",  0.20),
        ("max_digit_ratio",         0.15),
        ("max_url_ratio",           0.10),
        ("max_repeated_line_ratio", 0.25),
        ("max_repeated_ngram_ratio",0.20),
        ("min_mean_line_length",    25.0),
        ("min_paragraph_count",     1),
    ]:
        v, is_fb = _t(name)
        if is_fb:
            any_fb = True
        results[name] = v

    return results, any_fb


# ????????? Document filter ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def apply_filters(feats: dict, thresholds: dict) -> tuple[bool, str]:
    """
    Apply all quality filters to a precomputed feature dict.

    Parameters
    ----------
    feats : dict
        Output of extract_features().
    thresholds : dict
        Output of load_thresholds().

    Returns
    -------
    tuple[bool, str]
        (keep, reason) ??? reason is 'ok' if kept, else the failed filter name.
    """
    t = thresholds
    if feats["char_count"] < t["min_char_count"]:
        return False, f"min_char_count:{feats['char_count']}"
    if feats["devanagari_ratio"] < t["min_devanagari_ratio"]:
        return False, f"min_devanagari_ratio:{feats['devanagari_ratio']:.3f}"
    if feats["whitespace_ratio"] > t["max_whitespace_ratio"]:
        return False, f"max_whitespace_ratio:{feats['whitespace_ratio']:.3f}"
    if feats["punct_symbol_ratio"] > t["max_punct_symbol_ratio"]:
        return False, f"max_punct_symbol_ratio:{feats['punct_symbol_ratio']:.3f}"
    if feats["digit_ratio"] > t["max_digit_ratio"]:
        return False, f"max_digit_ratio:{feats['digit_ratio']:.3f}"
    if feats["url_ratio"] > t["max_url_ratio"]:
        return False, f"max_url_ratio:{feats['url_ratio']:.3f}"
    if feats["repeated_line_ratio"] > t["max_repeated_line_ratio"]:
        return False, f"max_repeated_line_ratio:{feats['repeated_line_ratio']:.3f}"
    if feats["repeated_ngram_ratio"] > t["max_repeated_ngram_ratio"]:
        return False, f"max_repeated_ngram_ratio:{feats['repeated_ngram_ratio']:.3f}"
    if feats["mean_line_length"] < t["min_mean_line_length"]:
        return False, f"min_mean_line_length:{feats['mean_line_length']:.1f}"
    if feats["paragraph_count"] < t["min_paragraph_count"]:
        return False, f"min_paragraph_count:{feats['paragraph_count']}"
    return True, "ok"


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 8: Full 9-feature quality filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    """Apply full quality filter suite, log per-filter drop counts."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    thresh_path = repo_root / args.lang / "configs" / "filtering_thresholds.yaml"
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    thresholds, any_fallback = load_thresholds(thresh_cfg, args.lang)

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    drop_by_filter: dict[str, int] = {}
    drop_updates: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"[quality_filters] {args.lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text   = doc.get(args.text_key, "")
            feats  = extract_features(text)
            keep, reason = apply_filters(feats, thresholds)

            doc_id = str(doc.get("doc_id", ""))

            if keep:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
            else:
                filter_name = reason.split(":")[0]
                drop_by_filter[filter_name] = drop_by_filter.get(filter_name, 0) + 1
                drop_updates[doc_id] = {
                    "status":     "dropped",
                    "drop_stage": "quality_filter",
                    "drop_reason": reason,
                }

    # Nepali safety warning
    if args.lang == "nepali" and total > 0:
        for fname, cnt in drop_by_filter.items():
            if cnt / total > 0.30:
                print(
                    f"[quality_filters] NEPALI WARNING: Filter '{fname}' dropped "
                    f"{cnt/total:.1%} of documents (> 30% threshold). "
                    f"Manually review a sample of dropped documents before proceeding.",
                    file=sys.stderr,
                )

    if drop_updates:
        n = update_rows(args.lang, drop_updates, str(repo_root))
        print(f"[quality_filters] Updated {n} manifest rows.")

    print(
        f"\n[quality_filters] Done.  Input: {total:,} | Kept: {kept:,} | "
        f"Dropped: {total - kept:,}"
        + (" [USING FALLBACK THRESHOLDS]" if any_fallback else "")
    )
    print("  Drop breakdown by filter:")
    for fname, cnt in sorted(drop_by_filter.items(), key=lambda x: -x[1]):
        print(f"    {fname:<30s}: {cnt:,}")


if __name__ == "__main__":
    main()
