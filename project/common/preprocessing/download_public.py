"""
download_public.py
------------------
Stage 2: Raw Data Acquisition for Downloaded Public Corpora.

Downloads public datasets (Wikipedia, IndicCorpV2, OSCAR-2301, CC100, etc.)
as specified in <lang>/configs/data_config.yaml, formats them into JSONL,
and registers every downloaded document in the per-language manifest CSV.

Usage
-----
  python common/preprocessing/download_public.py --lang hindi --source wikipedia --max-docs 1000 --repo-root project/
  python common/preprocessing/download_public.py --lang nepali --source all --repo-root project/
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_utils import init_manifest, make_row, append_rows, LANG_EXPECTED

SOURCE_PREFIXES = {
    "wikipedia": "wiki",
    "indiccorp_v2": "indiccorp",
    "oscar_2301": "oscar",
    "cc100": "cc100",
}


def download_source(
    lang: str,
    source_name: str,
    source_cfg: dict,
    max_docs: int | None,
    repo_root: Path,
) -> None:
    """Download a single public dataset source, format as JSONL, and update manifest."""
    hf_dataset = source_cfg.get("hf_dataset")
    hf_config = source_cfg.get("hf_config")
    license_note = source_cfg.get("license", "Unknown")

    if not hf_dataset:
        print(f"[download_public] Skipping {source_name}: missing hf_dataset in config.")
        return

    print(f"\n[download_public] Downloading {lang} source '{source_name}' ({hf_dataset}/{hf_config})???")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[download_public] ERROR: `datasets` library not installed. Run `pip install datasets`.", file=sys.stderr)
        sys.exit(1)

    try:
        ds = load_dataset(hf_dataset, hf_config, split="train", streaming=True)
    except Exception as e:
        print(f"[download_public] ERROR loading dataset {hf_dataset}/{hf_config}: {e}", file=sys.stderr)
        return

    prefix = SOURCE_PREFIXES.get(source_name, source_name)
    iso2 = LANG_EXPECTED.get(lang.lower(), lang[:2])
    source_id = f"{iso2}_{prefix}"

    out_dir = repo_root / lang / "data" / "raw" / "downloaded"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{source_id}.jsonl"

    init_manifest(lang, str(repo_root))

    docs_written = 0
    manifest_rows = []

    with open(out_file, "w", encoding="utf-8") as f_out:
        pbar = tqdm(desc=f"  {source_id}", unit="doc")
        for idx, ex in enumerate(ds):
            if max_docs is not None and docs_written >= max_docs:
                break

            text = ex.get("text", "").strip()
            if not text:
                continue

            doc_id = f"{source_id}_{idx:06d}"
            url_or_path = ex.get("url") or ex.get("id") or hf_dataset

            doc_obj = {
                "doc_id": doc_id,
                "text": text,
                "source": source_id,
                "collection_method": "downloaded",
                "url": url_or_path,
            }

            f_out.write(json.dumps(doc_obj, ensure_ascii=False) + "\n")
            docs_written += 1
            pbar.update(1)

            row = make_row(
                doc_id=doc_id,
                source_id=source_id,
                source_name=f"{source_name.title()} ({hf_dataset})",
                source_type="wikipedia" if source_name == "wikipedia" else "hf_corpus",
                collection_method="downloaded",
                url_or_path=str(url_or_path),
                license_note=license_note,
                lang=lang,
                raw_char_count=len(text),
                raw_word_count=len(text.split()),
            )
            manifest_rows.append(row)

            if len(manifest_rows) >= 5000:
                append_rows(lang, manifest_rows, str(repo_root))
                manifest_rows = []

        pbar.close()

    if manifest_rows:
        append_rows(lang, manifest_rows, str(repo_root))

    print(f"[download_public] Downloaded {docs_written} docs to {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public datasets for Phase 1 pipeline.")
    parser.add_argument("--lang", choices=["hindi", "nepali"], help="Language to download.")
    parser.add_argument("--source", default="all", help="Source name ('wikipedia', 'indiccorp_v2', 'oscar_2301', etc.) or 'all'.")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to download per source.")
    parser.add_argument("--repo-root", default=".", help="Path to project repository root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    lang = args.lang
    if not lang:
        # Try to infer lang from current working directory or repo structure
        if (repo_root / "configs" / "data_config.yaml").exists():
            # Executed from inside project/hindi or project/nepali
            lang = repo_root.name
            repo_root = repo_root.parent

    if not lang or lang not in ["hindi", "nepali"]:
        print("[download_public] ERROR: Please specify --lang (hindi or nepali).", file=sys.stderr)
        sys.exit(1)

    config_path = repo_root / lang / "configs" / "data_config.yaml"
    if not config_path.exists():
        print(f"[download_public] ERROR: Config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sources_cfg = cfg.get("sources", {})

    if args.source == "all":
        target_sources = [name for name, sc in sources_cfg.items() if sc.get("enabled", False)]
    else:
        target_sources = [args.source]

    for s_name in target_sources:
        s_cfg = sources_cfg.get(s_name)
        if not s_cfg:
            print(f"[download_public] Source '{s_name}' not configured in {config_path}")
            continue
        max_d = args.max_docs if args.max_docs is not None else (s_cfg.get("max_docs") or 1000)
        download_source(lang, s_name, s_cfg, max_d, repo_root)


if __name__ == "__main__":
    main()
