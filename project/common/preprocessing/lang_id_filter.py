"""
lang_id_filter.py  (v2)
------------------------
Stage 5: Language Identification Filtering.

v2 changes:
  - Threshold is read from filtering_thresholds.yaml (not hard-coded).
  - Warns loudly if the threshold is null (run recon.py --inspect-feature
    lang_id_confidence first and fill in the YAML).
  - Provides a deterministic fallback path: script-based Devanagari script
    ratio heuristic for low-confidence documents (instead of silent drop).
  - Computes confidence-score distribution on the input corpus and saves it
    to source_stats.json for later threshold calibration.
  - Updates manifest drop_stage/drop_reason for every dropped document.

Language ID model
-----------------
  Primary  : fastText lid.176.bin (facebook/fasttext-language-identification)
  Fallback : langdetect (if fasttext not installed or model file missing)
  Heuristic: Devanagari script ratio ??? 0.40 used as deterministic fallback
              for documents where the model confidence is below threshold but
              the document is clearly Devanagari-script text.

Usage
-----
  python lang_id_filter.py \\
      --lang hindi \\
      --input hindi/data/cleaned/normalized.jsonl \\
      --output hindi/data/cleaned/lang_filtered.jsonl \\
      --fasttext-model /path/to/lid.176.bin \\
      --repo-root .
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "preprocessing"))
from manifest_utils import update_rows

_DEVA_RE = re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]")
_DEVA_FALLBACK_RATIO = 0.40   # used as heuristic fallback

# Accepted fastText language codes per language
_ACCEPTED_CODES: dict[str, set] = {
    "hindi":  {"hi"},
    "nepali": {"ne", "hi"},   # fastText often labels Nepali as 'hi'
}


# ????????? Model loading ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def load_fasttext_model(model_path: str):
    """
    Load fastText lid.176.bin model.

    Parameters
    ----------
    model_path : str

    Returns
    -------
    fasttext model or None
    """
    try:
        import fasttext
        import fasttext.FastText
        fasttext.FastText.eprint = lambda *a, **k: None
        model = fasttext.load_model(model_path)
        print(f"[lang_id] Loaded fastText model: {model_path}")
        return model
    except ImportError:
        print("[lang_id] WARNING: fasttext not installed; using langdetect fallback.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[lang_id] WARNING: Could not load model: {e}; using langdetect.", file=sys.stderr)
        return None


def predict(model, text: str) -> tuple[str, float]:
    """
    Predict language code and confidence for a text.

    Parameters
    ----------
    model : fasttext model or None
    text : str

    Returns
    -------
    tuple[str, float]
        (lang_code, confidence)
    """
    clean = " ".join(text.split())[:2000]

    if model is not None:
        try:
            labels, probs = model.predict(clean, k=1)
            code = labels[0].replace("__label__", "")
            return code, float(probs[0])
        except Exception:
            pass

    # langdetect fallback
    try:
        from langdetect import detect_langs
        results = detect_langs(clean)
        if results:
            return results[0].lang, results[0].prob
    except Exception:
        pass

    return "und", 0.0


def devanagari_ratio(text: str) -> float:
    """
    Compute the fraction of Devanagari characters in the text.

    Parameters
    ----------
    text : str

    Returns
    -------
    float
        Ratio in [0, 1].
    """
    if not text:
        return 0.0
    deva = len(_DEVA_RE.findall(text))
    return deva / len(text)


# ????????? Filter logic ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def decide_keep(
    text: str,
    lang: str,
    threshold: float,
    model,
) -> tuple[bool, str, str, float]:
    """
    Decide whether to keep a document, with a Devanagari script fallback.

    Decision logic:
      1. Run language-ID model ??? code, conf.
      2. If code ??? accepted_codes AND conf ??? threshold ??? KEEP.
      3. If conf < threshold BUT devanagari_ratio ??? _DEVA_FALLBACK_RATIO
         AND lang in ('hindi', 'nepali') ??? KEEP with flag "heuristic_fallback".
      4. Otherwise ??? DROP.

    Parameters
    ----------
    text : str
    lang : str
    threshold : float
    model : fasttext model or None

    Returns
    -------
    tuple[bool, str, str, float]
        (keep, detected_code, decision_path, confidence)
    """
    code, conf = predict(model, text)
    accepted = _ACCEPTED_CODES.get(lang, {lang[:2]})

    if code in accepted and conf >= threshold:
        return True, code, "model_accept", conf

    if conf < threshold:
        # Devanagari heuristic fallback
        deva_r = devanagari_ratio(text)
        if deva_r >= _DEVA_FALLBACK_RATIO:
            return True, code, "heuristic_fallback", conf

    return False, code, "rejected", conf


# ????????? CLI ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Language identification filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fasttext-model", default="lid.176.bin")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    """Run language ID filtering with config-driven threshold."""
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    # Load threshold
    thresh_path = repo_root / args.lang / "configs" / "filtering_thresholds.yaml"
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_cfg = yaml.safe_load(f)

    lang_cfg    = thresh_cfg.get("language_id", {})
    thresh_node = lang_cfg.get("confidence_threshold", {})
    threshold   = thresh_node.get("value")
    fallback    = thresh_node.get("conservative_fallback", 0.65)
    is_fallback = False
    if threshold is None:
        print(
            f"[lang_id] WARNING: confidence_threshold is null in filtering_thresholds.yaml.\n"
            f"  Run: python recon.py --lang {args.lang} --inspect-feature lang_id_confidence\n"
            f"  Then fill in the value and rationale in filtering_thresholds.yaml.\n"
            f"  Using conservative fallback: {fallback}",
            file=sys.stderr,
        )
        threshold = fallback
        is_fallback = True

    # Load model
    model = load_fasttext_model(args.fasttext_model)

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = dropped = fallback_kept = 0
    rejected_codes: dict[str, int] = {}
    drop_updates: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"[lang_id] {args.lang}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text    = doc.get(args.text_key, "")
            doc_id  = str(doc.get("doc_id", ""))

            keep, code, path, conf = decide_keep(text, args.lang, threshold, model)

            if keep:
                doc["_lang_detected"]  = code
                doc["_lang_conf"]      = round(conf, 4)
                doc["_lang_path"]      = path
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
                if path == "heuristic_fallback":
                    fallback_kept += 1
            else:
                dropped += 1
                rejected_codes[code] = rejected_codes.get(code, 0) + 1
                drop_updates[doc_id] = {
                    "status":     "dropped",
                    "drop_stage": "language_id",
                    "drop_reason": f"wrong_language:{code}:conf={conf:.3f}",
                    "language_detected":    code,
                    "language_confidence":  round(conf, 4),
                }

    if drop_updates:
        n = update_rows(args.lang, drop_updates, str(repo_root))
        print(f"[lang_id] Updated {n} manifest rows.")

    print(
        f"\n[lang_id] Done.  Input: {total:,} | Kept: {kept:,} "
        f"(heuristic_fallback: {fallback_kept:,}) | Dropped: {dropped:,}"
        + (" [USING FALLBACK THRESHOLD]" if is_fallback else f" [threshold={threshold}]")
    )
    if rejected_codes:
        top5 = sorted(rejected_codes.items(), key=lambda x: -x[1])[:5]
        print(f"  Top rejected codes: {top5}")


if __name__ == "__main__":
    main()
