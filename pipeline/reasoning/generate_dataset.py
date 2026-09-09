"""Synthetic comparative/transitive-reasoning dataset generator (Phase 3,
section 3.1). Template-driven, not scraped/downloaded, so every label is
exactly known at generation time.

    python -m pipeline.reasoning.generate_dataset --lang hindi --repo-root .
    python -m pipeline.reasoning.generate_dataset --lang nepali --repo-root .

Leakage avoidance (two independent mechanisms, both applied every time):
  1. Entity names: the name pool for each language is split into disjoint
     train / val / test subsets (see `split_names`). An entity name that
     appears in a test example was NEVER used to generate a train or val
     example, and vice versa.
  2. Relation patterns: for every template family with more than one
     phrasing, the last phrasing is reserved exclusively for the test
     split; train/val only ever sample from the remaining phrasings. So
     some exact sentence structures in the test set were never seen during
     training either.

Output schema (one JSON object per line):
    {
      "id": "hindi_train_000042",
      "family": "A_more" | "A_less" | "B_most" | "B_least" |
                 "C_endpoints_more" | "C_endpoints_less" | "C_most" | "C_least" |
                 "D_equal",
      "attribute": "height" | "age" | "price" | "weight",
      "n_entities": 2 | 3,
      "entities": ["राम", "श्याम"],
      "values": [172, 165] | null,           # null for the pure-relational family C
      "prompt": "...question text, ending right before the answer delimiter...",
      "answer": "राम",                        # exact-match gold label
    }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pipeline.reasoning.lexicon import LEXICONS, Lexicon

FAMILIES_2E = ["A_more", "A_less"]              # 2-entity numeric
FAMILIES_3E_NUM = ["B_most", "B_least"]          # 3-entity numeric superlative
FAMILIES_3E_REL = ["C_endpoints_more", "C_endpoints_less", "C_most", "C_least"]  # 3-entity pure-relational
FAMILY_EQ = "D_equal"                             # 2-entity equality check


def split_names(names: list[str], seed: int, train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, list[str]]:
    """Disjoint train/val/test partition of a name pool. This is the
    primary leakage-avoidance mechanism: a name in one split never appears
    in either of the others."""
    rng = random.Random(seed)
    pool = list(dict.fromkeys(names))  # de-dup while preserving order; a
    # duplicate name in the source list would otherwise be free to land in
    # two different splits after shuffling, silently breaking the very
    # leakage guarantee this function exists to provide.
    rng.shuffle(pool)
    n = len(pool)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": pool[:n_train],
        "val": pool[n_train : n_train + n_val],
        "test": pool[n_train + n_val :],
    }


def templates_for_split(lex: Lexicon, family: str, split: str) -> list[str]:
    all_templates = lex.templates[family]
    if len(all_templates) < 2:
        return all_templates  # nothing to hold out; entity-name disjointness still protects this family
    return all_templates[-1:] if split == "test" else all_templates[:-1]


def _fmt(template: str, attr, **kw) -> str:
    return template.format(
        attr=attr.noun, unit=attr.unit, whose=attr.whose, poss=attr.poss, will_be=attr.will_be, **kw
    )


def gen_two_entity(rng: random.Random, lex: Lexicon, split: str, names: list[str], family: str, ex_id: str) -> dict:
    attr_key = rng.choice(list(lex.attributes))
    attr = lex.attributes[attr_key]
    pool = lex.people_names if attr.domain == "people" else lex.object_names
    pool = [n for n in pool if n in names] or names  # fall back if domain pool doesn't intersect this split's names
    a, b = rng.sample(pool, 2) if len(pool) >= 2 else rng.sample(names, 2)
    va = rng.randint(attr.low, attr.high)
    vb = rng.randint(attr.low, attr.high)
    while vb == va:
        vb = rng.randint(attr.low, attr.high)

    template = rng.choice(templates_for_split(lex, family, split))
    prompt = _fmt(template, attr, A=a, B=b, va=va, vb=vb)
    if family == "A_more":
        answer = a if va > vb else b
    else:  # A_less
        answer = a if va < vb else b

    return {
        "id": ex_id, "family": family, "attribute": attr_key, "n_entities": 2,
        "entities": [a, b], "values": [va, vb], "prompt": prompt, "answer": answer,
    }


def gen_equality(rng: random.Random, lex: Lexicon, split: str, names: list[str], ex_id: str) -> dict:
    attr_key = rng.choice(list(lex.attributes))
    attr = lex.attributes[attr_key]
    pool = lex.people_names if attr.domain == "people" else lex.object_names
    pool = [n for n in pool if n in names] or names
    a, b = rng.sample(pool, 2) if len(pool) >= 2 else rng.sample(names, 2)
    va = rng.randint(attr.low, attr.high)
    equal = rng.random() < 0.5
    vb = va if equal else rng.randint(attr.low, attr.high)
    if not equal:
        while vb == va:
            vb = rng.randint(attr.low, attr.high)

    template = rng.choice(templates_for_split(lex, FAMILY_EQ, split))
    prompt = _fmt(template, attr, A=a, B=b, va=va, vb=vb)
    answer = lex.yes if equal else lex.no

    return {
        "id": ex_id, "family": FAMILY_EQ, "attribute": attr_key, "n_entities": 2,
        "entities": [a, b], "values": [va, vb], "prompt": prompt, "answer": answer,
    }


def gen_three_entity_numeric(rng: random.Random, lex: Lexicon, split: str, names: list[str], family: str, ex_id: str) -> dict:
    attr_key = rng.choice(list(lex.attributes))
    attr = lex.attributes[attr_key]
    pool = lex.people_names if attr.domain == "people" else lex.object_names
    pool = [n for n in pool if n in names] or names
    a, b, c = rng.sample(pool, 3) if len(pool) >= 3 else rng.sample(names, 3)
    vals = set()
    while len(vals) < 3:
        vals.add(rng.randint(attr.low, attr.high))
    va, vb, vc = list(vals)

    template = rng.choice(templates_for_split(lex, family, split))
    prompt = _fmt(template, attr, A=a, B=b, C=c, va=va, vb=vb, vc=vc)
    entities, values = [a, b, c], [va, vb, vc]
    if family == "B_most":
        answer = entities[values.index(max(values))]
    else:  # B_least
        answer = entities[values.index(min(values))]

    return {
        "id": ex_id, "family": family, "attribute": attr_key, "n_entities": 3,
        "entities": entities, "values": values, "prompt": prompt, "answer": answer,
    }


def gen_transitive_chain(rng: random.Random, lex: Lexicon, split: str, names: list[str], family: str, ex_id: str) -> dict:
    """Pure-relational: only 'A > B' and 'B > C' are stated (no numbers).
    Answering requires chaining the two inequalities -- the model is never
    told the A-vs-C relation directly."""
    attr_key = rng.choice(list(lex.attributes))
    attr = lex.attributes[attr_key]
    pool = lex.people_names if attr.domain == "people" else lex.object_names
    pool = [n for n in pool if n in names] or names
    a, b, c = rng.sample(pool, 3) if len(pool) >= 3 else rng.sample(names, 3)
    # narrative order is the ranking itself: A > B > C

    template = rng.choice(templates_for_split(lex, family, split))
    prompt = _fmt(template, attr, A=a, B=b, C=c)
    if family == "C_endpoints_more":
        answer = a
    elif family == "C_endpoints_less":
        answer = c
    elif family == "C_most":
        answer = a
    else:  # C_least
        answer = c

    return {
        "id": ex_id, "family": family, "attribute": attr_key, "n_entities": 3,
        "entities": [a, b, c], "values": None, "prompt": prompt, "answer": answer,
    }


ALL_FAMILIES = FAMILIES_2E + FAMILIES_3E_NUM + FAMILIES_3E_REL + [FAMILY_EQ]
# rough weighting: keep 2-entity numeric comparisons (the simplest case) most
# common, transitive chains and equality as substantial but smaller slices
FAMILY_WEIGHTS = {
    # Numeric-magnitude-comparison families (A_*, B_*) are weighted up:
    # reasoning_eval showed these are the weak spot (5-25% accuracy vs
    # 80-95%+ for the pure-relational C_* families even at matched
    # data-sparsity conditions) -- sparse coverage of the (value_a, value_b)
    # space, not tokenization (digits already get one token each), is the
    # bottleneck, so these get proportionally more generated examples.
    "A_more": 5, "A_less": 5,
    "B_most": 4, "B_least": 4,
    "C_endpoints_more": 2, "C_endpoints_less": 2, "C_most": 1, "C_least": 1,
    "D_equal": 2,
}


def generate_split(lex: Lexicon, split: str, names: list[str], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    families = list(FAMILY_WEIGHTS)
    weights = [FAMILY_WEIGHTS[f] for f in families]
    seen_prompts = set()
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        family = rng.choices(families, weights=weights, k=1)[0]
        ex_id = f"{lex.lang}_{split}_{len(out):06d}"
        if family in FAMILIES_2E:
            ex = gen_two_entity(rng, lex, split, names, family, ex_id)
        elif family == FAMILY_EQ:
            ex = gen_equality(rng, lex, split, names, ex_id)
        elif family in FAMILIES_3E_NUM:
            ex = gen_three_entity_numeric(rng, lex, split, names, family, ex_id)
        else:
            ex = gen_transitive_chain(rng, lex, split, names, family, ex_id)

        if ex["prompt"] in seen_prompts:
            continue  # dedup within split
        seen_prompts.add(ex["prompt"])
        out.append(ex)
    return out


def dataset_stats(examples: list[dict]) -> dict:
    from collections import Counter
    fam_counts = Counter(e["family"] for e in examples)
    attr_counts = Counter(e["attribute"] for e in examples)
    n_ent_counts = Counter(e["n_entities"] for e in examples)
    prompt_lens = [len(e["prompt"]) for e in examples]
    return {
        "n_examples": len(examples),
        "by_family": dict(fam_counts),
        "by_attribute": dict(attr_counts),
        "by_n_entities": {str(k): v for k, v in n_ent_counts.items()},
        "prompt_char_len_min": min(prompt_lens) if prompt_lens else 0,
        "prompt_char_len_max": max(prompt_lens) if prompt_lens else 0,
        "prompt_char_len_mean": sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0,
    }


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def generate_all(lang: str, repo_root: Path, n_train: int, n_val: int, n_test: int, seed: int) -> dict:
    lex = LEXICONS[lang]
    people_split = split_names(lex.people_names, seed=seed)
    object_split = split_names(lex.object_names, seed=seed + 1)
    name_split = {
        s: people_split[s] + object_split[s] for s in ("train", "val", "test")
    }

    out_dir = repo_root / lang / "data" / "reasoning"
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    stats = {"lang": lang, "seed": seed, "splits": {}}
    for split, n in sizes.items():
        examples = generate_split(lex, split, name_split[split], n, seed=seed + hash(split) % 10_000)
        write_jsonl(out_dir / f"{split}.jsonl", examples)
        stats["splits"][split] = dataset_stats(examples)
        stats["splits"][split]["names_used"] = sorted(set(name_split[split]))

    stats["name_pool_sizes"] = {s: len(name_split[s]) for s in name_split}
    stats["leakage_avoidance"] = (
        "Entity names are partitioned into disjoint train/val/test pools "
        "before any example is generated (split_names, seed={seed}); no "
        "name in the test set was ever used to generate a train or val "
        "example. Additionally, for every template family with more than "
        "one phrasing, the last phrasing is reserved exclusively for the "
        "test split (templates_for_split) -- some test sentence structures "
        "were never seen during training either."
    ).format(seed=seed)

    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    stats = generate_all(args.lang, root, args.n_train, args.n_val, args.n_test, args.seed)
    print(json.dumps({k: v for k, v in stats.items() if k != "splits"}, indent=2, ensure_ascii=False))
    for split, s in stats["splits"].items():
        print(f"{split}: {s['n_examples']} examples, families={s['by_family']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
