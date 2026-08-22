"""
make_gcs_fixture.py — a fake bucket with the real tree, for offline testing.

Builds

    <out>/hi/wikipedia/data.jsonl
    <out>/hi/sangraha/verified/data.jsonl
    <out>/hi/sangraha/unverified/data.jsonl
    <out>/ne/...

so `gcs_ingest --bucket-root <out>` exercises exactly the code path that runs
against gs://lma-01-hi-ne-corpus/raw, minus the network. Blob.open_text falls
back to the filesystem for any root that is not a gs:// URI, so nothing is
stubbed or monkey-patched -- the same reader runs both times.

The fixture deliberately contains, in each source:
  * clean documents
  * near-duplicates (so MinHash has something to find)
  * documents that are ALSO in the manual set (so the cross-set dedup and the
    manual-wins tie-break are exercised, which is the whole point)
  * wrong-language and too-short junk

Usage
  python tests/make_gcs_fixture.py /tmp/fake-bucket
"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.make_fixture import HI_ROOT, HI_SUF, HI_V, NE_ROOT, NE_SUF, NE_V, prose, vocab  # noqa: E402

random.seed(23)

LANGS = {
    "hi": (HI_ROOT, HI_SUF, HI_V),
    "ne": (NE_ROOT, NE_SUF, NE_V),
}

# (relative path, number of documents). Unverified is the biggest, as in the
# real bucket, so the budget runs out inside it and the early-stop path runs.
LAYOUT = [
    ("wikipedia/data.jsonl", 300),
    ("sangraha/verified/data.jsonl", 700),
    ("sangraha/unverified/data.jsonl", 4000),
]


def write_source(path: Path, n: int, words, verbs, shared: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        prev = None
        for i in range(n):
            r = random.random()
            if r < 0.05 and shared:
                # a document that also exists in the manual set
                text = random.choice(shared)
            elif r < 0.10 and prev:
                text = prev + " यह अतिरिक्त वाक्य है।"
            elif r < 0.13:
                text = "abc " * 30                      # junk
            elif r < 0.16:
                text = prose(words, verbs, 1)           # too short
            else:
                text = prose(words, verbs)
                prev = text
            f.write(json.dumps({"text": text, "url": f"https://example/{i}",
                                "title": f"doc {i}"}, ensure_ascii=False) + "\n")


def main(out: Path, manual_overlap_dir: Path | None):
    for code, (roots, sufs, verbs) in LANGS.items():
        words = vocab(roots, sufs, 2500)
        shared: list[str] = []
        if manual_overlap_dir:
            lang = "hindi" if code == "hi" else "nepali"
            for p in (manual_overlap_dir / lang / "data" / "raw").glob("manual_*.jsonl"):
                for line in p.read_text(encoding="utf-8").splitlines()[:60]:
                    try:
                        shared.append(json.loads(line)["text"])
                    except Exception:
                        pass
        for rel, n in LAYOUT:
            path = out / code / rel
            write_source(path, n, words, verbs, shared)
            print(f"  {path.relative_to(out)}: {n} documents")
    print(f"fake bucket ready at {out}"
          + (f" (with {len(shared)} documents duplicated from the manual set)"
             if manual_overlap_dir else ""))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake-bucket").resolve()
    repo = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    main(out, repo)
