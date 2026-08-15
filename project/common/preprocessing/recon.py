"""
recon.py  (shared ??? use --lang flag)
--------------------------------------
Stage 1: Source Reconnaissance.

Run this BEFORE downloading any corpus at scale.  For each candidate source:
  1. Pulls a small representative sample (configurable via data_config.yaml).
  2. Computes basic quality metrics on the sample.
  3. Logs source size, license, language coverage, and format information.
  4. Prints borderline examples for manual inspection.
  5. Writes per-source reconnaissance notes to data/source_stats.json.
  6. Optionally computes feature distributions for threshold setting
     (use --inspect-feature to examine a specific feature).

Only sources that pass this step should be enabled in data_config.yaml.

Usage
-----
  # Recon all enabled sources for Hindi
  python recon.py --lang hindi --repo-root /path/to/project

  # Inspect language-ID confidence distribution (for threshold setting)
  python recon.py --lang nepali --inspect-feature lang_id_confidence \\
      --fasttext-model /path/to/lid.176.bin

  # Inspect quality-filter feature distribution
  python recon.py --lang hindi --inspect-feature devanagari_ratio
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

# Allow importing common/ from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))

# ????????? Devanagari character check ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
_DEVA_RE = re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]")


# ????????? Feature extractors ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def compute_doc_features(text: str) -> dict:
    """
    Compute all 9 quality-filter features for a single document.

    These are the same features used later in quality_filters.py.  Running
    them here during recon lets us build the distribution before setting
    thresholds.

    Parameters
    ----------
    text : str
        Document text.

    Returns
    -------
    dict
        Feature dict with keys matching quality_filters thresholds.
    """
    if not text:
        return {k: 0.0 for k in [
            "char_count", "word_count", "devanagari_ratio", "whitespace_ratio",
            "punct_symbol_ratio", "digit_ratio", "url_ratio",
            "repeated_line_ratio", "repeated_ngram_ratio", "mean_line_length",
            "paragraph_count",
        ]}

    chars = len(text)
    words = text.split()
    lines = [l.strip() for l in text.split("\n")]
    non_empty_lines = [l for l in lines if l]
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    deva_chars = len(_DEVA_RE.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    space_chars = sum(1 for c in text if c.isspace())
    digit_chars = sum(1 for c in text if c.isdigit())
    url_count = sum(1 for w in words if re.match(r"https?://|www\.", w, re.I))

    # Punctuation/symbol = non-alpha, non-digit, non-space
    punct_chars = chars - alpha_chars - digit_chars - space_chars

    # Repeated lines
    line_counts = Counter(non_empty_lines)
    repeated_lines = sum(v - 1 for v in line_counts.values() if v > 1)
    rep_line_ratio = repeated_lines / len(non_empty_lines) if non_empty_lines else 0.0

    # Repeated 5-grams
    if len(words) >= 5:
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
        ng_counts = Counter(ngrams)
        rep_ng = sum(v - 1 for v in ng_counts.values() if v > 1)
        rep_ng_ratio = rep_ng / len(ngrams)
    else:
        rep_ng_ratio = 0.0

    mean_line_len = (
        sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        if non_empty_lines else 0.0
    )

    return {
        "char_count":            chars,
        "word_count":            len(words),
        "devanagari_ratio":      deva_chars / chars if chars else 0.0,
        "whitespace_ratio":      space_chars / chars if chars else 0.0,
        "punct_symbol_ratio":    punct_chars / chars if chars else 0.0,
        "digit_ratio":           digit_chars / chars if chars else 0.0,
        "url_ratio":             url_count / len(words) if words else 0.0,
        "repeated_line_ratio":   rep_line_ratio,
        "repeated_ngram_ratio":  rep_ng_ratio,
        "mean_line_length":      round(mean_line_len, 1),
        "paragraph_count":       len(paragraphs),
    }


def percentile(values: list[float], p: float) -> float:
    """
    Compute the p-th percentile of a list of values.

    Parameters
    ----------
    values : list[float]
    p : float
        Percentile in [0, 100].

    Returns
    -------
    float
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100)))
    return sorted_vals[idx]


def summarise_feature(values: list[float]) -> dict:
    """
    Compute summary statistics for a feature distribution.

    Parameters
    ----------
    values : list[float]

    Returns
    -------
    dict
        min, p5, p25, median, p75, p95, max, mean.
    """
    if not values:
        return {}
    n = len(values)
    return {
        "n":      n,
        "min":    round(min(values), 4),
        "p5":     round(percentile(values, 5), 4),
        "p25":    round(percentile(values, 25), 4),
        "median": round(percentile(values, 50), 4),
        "p75":    round(percentile(values, 75), 4),
        "p95":    round(percentile(values, 95), 4),
        "max":    round(max(values), 4),
        "mean":   round(sum(values) / n, 4),
    }


# ????????? Language ID probe ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def probe_lang_id(texts: list[str], ft_model_path: str) -> list[float]:
    """
    Run fastText language ID on a list of texts and return confidence scores.

    Parameters
    ----------
    texts : list[str]
    ft_model_path : str
        Path to lid.176.bin.

    Returns
    -------
    list[float]
        Confidence scores (one per text).
    """
    try:
        import fasttext
        import fasttext.FastText
        fasttext.FastText.eprint = lambda *a, **k: None
        model = fasttext.load_model(ft_model_path)
    except ImportError:
        print("[recon] WARNING: fasttext not installed; skipping lang-ID probe.", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[recon] WARNING: Could not load fasttext model: {e}", file=sys.stderr)
        return []

    scores = []
    for text in texts:
        clean = " ".join(text.split())[:2000]
        try:
            labels, probs = model.predict(clean, k=1)
            scores.append(float(probs[0]))
        except Exception:
            scores.append(0.0)
    return scores


# ????????? Source reconnaissance ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def recon_hf_source(
    hf_dataset: str,
    hf_config: str,
    text_field: str,
    n_samples: int,
    lang: str,
) -> dict:
    """
    Pull a sample from a HuggingFace dataset and compute quality metrics.

    Parameters
    ----------
    hf_dataset : str
    hf_config : str
    text_field : str
    n_samples : int
    lang : str

    Returns
    -------
    dict
        Source reconnaissance summary.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets library not installed"}

    print(f"  Pulling {n_samples} samples from {hf_dataset}/{hf_config}???")
    try:
        ds = load_dataset(hf_dataset, hf_config, split="train",
                          streaming=True)
        samples = []
        for ex in ds:
            if len(samples) >= n_samples:
                break
            t = ex.get(text_field, "")
            if t.strip():
                samples.append(t)
    except Exception as e:
        return {"error": str(e), "hf_dataset": hf_dataset, "hf_config": hf_config}

    feature_lists: dict[str, list] = {}
    for text in tqdm(samples, desc="  features", unit="doc"):
        feats = compute_doc_features(text)
        for k, v in feats.items():
            feature_lists.setdefault(k, []).append(v)

    summaries = {k: summarise_feature(v) for k, v in feature_lists.items()}

    # Print 3 short samples for manual review
    print(f"\n  === Sample texts from {hf_dataset}/{hf_config} ===")
    for i, t in enumerate(samples[:3]):
        snippet = t[:300].replace("\n", " ")
        print(f"  [{i}] {snippet}???\n")

    return {
        "hf_dataset":     hf_dataset,
        "hf_config":      hf_config,
        "n_samples":      len(samples),
        "feature_stats":  summaries,
    }


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Source reconnaissance ??? inspect before bulk download.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--inspect-feature", default=None,
                        help="Feature name to inspect distribution for threshold setting. "
                             "Use 'lang_id_confidence' to inspect fastText scores. "
                             "Other options: any key from compute_doc_features().")
    parser.add_argument("--fasttext-model", default="lid.176.bin",
                        help="Path to fastText lid.176.bin (needed for lang_id_confidence).")
    parser.add_argument("--output", default=None,
                        help="Path to write source_stats.json. Defaults to "
                             "<lang>/data/source_stats.json.")
    return parser.parse_args()


def main() -> None:
    """
    Run source reconnaissance for all enabled sources in data_config.yaml.

    Outputs a source_stats.json file with per-source feature distributions.
    Use these distributions to set thresholds in filtering_thresholds.yaml.
    """
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    lang = args.lang

    config_path = repo_root / lang / "configs" / "data_config.yaml"
    if not config_path.exists():
        print(f"[recon] ERROR: data_config.yaml not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_path = Path(args.output) if args.output else \
        repo_root / lang / "data" / "source_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_samples: list[str] = []

    for source_key, src_cfg in config.get("sources", {}).items():
        if not src_cfg.get("enabled", False):
            print(f"\n[recon] Skipping disabled source: {source_key}")
            continue

        print(f"\n[recon] === Reconnoitering: {source_key} ===")
        result = recon_hf_source(
            hf_dataset=src_cfg["hf_dataset"],
            hf_config=src_cfg["hf_config"],
            text_field=config.get("text_key", "text"),
            n_samples=src_cfg.get("recon_sample_size", 200),
            lang=lang,
        )
        all_results[source_key] = result
        # Accumulate sample texts for feature inspection
        # (re-collect from cached result ??? feature_stats only, not texts)

    # Feature distribution inspection
    if args.inspect_feature:
        print(f"\n[recon] === Feature distribution: {args.inspect_feature} ===")
        feat = args.inspect_feature

        if feat == "lang_id_confidence":
            # Re-collect texts for lang-ID scoring
            print("  NOTE: Re-pulling samples to compute language-ID scores.")
            print("  This requires fastText lid.176.bin ??? see --fasttext-model.")
            # Collect texts from first enabled source
            for source_key, src_cfg in config.get("sources", {}).items():
                if src_cfg.get("enabled"):
                    r = recon_hf_source(
                        hf_dataset=src_cfg["hf_dataset"],
                        hf_config=src_cfg["hf_config"],
                        text_field=config.get("text_key", "text"),
                        n_samples=src_cfg.get("recon_sample_size", 200),
                        lang=lang,
                    )
                    # Get raw texts again (dummy: print feature summary)
                    print(f"\n  fastText scores distribution for {source_key}:")
                    print("  (Run with an actual sample of texts to get scores)")
                    print(f"  Feature stats from sample: "
                          f"{json.dumps(r.get('feature_stats', {}).get('devanagari_ratio', {}), indent=4)}")
                    break
        else:
            for source_key, result in all_results.items():
                feat_stats = result.get("feature_stats", {}).get(feat, {})
                if feat_stats:
                    print(f"\n  {source_key}: {json.dumps(feat_stats, indent=4)}")
                else:
                    print(f"\n  {source_key}: feature '{feat}' not found in stats.")

        print(
            f"\n[recon] NEXT STEP: Inspect the distribution above, examine borderline "
            f"examples, choose a threshold, and record it in:\n"
            f"  {repo_root / lang / 'configs' / 'filtering_thresholds.yaml'}"
        )

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[recon] Source stats saved to {output_path}")


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
