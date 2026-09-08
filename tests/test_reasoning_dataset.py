"""Correctness checks for the Phase 3 synthetic reasoning-dataset generator:
every generated label is re-derived independently from the example's own
entities/values and must match, and the leakage-avoidance guarantees
(disjoint entity names, held-out template patterns) must actually hold.

    python -m pytest tests/test_reasoning_dataset.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.reasoning.generate_dataset import (
    FAMILY_EQ,
    generate_all,
    split_names,
    templates_for_split,
)
from pipeline.reasoning.lexicon import LEXICONS


def test_split_names_are_disjoint():
    for lex in LEXICONS.values():
        for pool in (lex.people_names, lex.object_names):
            s = split_names(pool, seed=20260909)
            train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
            assert not (train & val)
            assert not (train & test)
            assert not (val & test)
            assert train | val | test == set(pool)


def test_held_out_template_never_used_outside_test():
    for lex in LEXICONS.values():
        for family, all_templates in lex.templates.items():
            if len(all_templates) < 2:
                continue
            held_out = all_templates[-1]
            train_pool = templates_for_split(lex, family, "train")
            val_pool = templates_for_split(lex, family, "val")
            assert held_out not in train_pool
            assert held_out not in val_pool
            assert templates_for_split(lex, family, "test") == [held_out]


def _check_examples(lang: str, tmp_path: Path):
    stats = generate_all(lang, tmp_path, n_train=300, n_val=60, n_test=60, seed=1)
    assert stats["splits"]["train"]["n_examples"] == 300
    assert stats["splits"]["val"]["n_examples"] == 60
    assert stats["splits"]["test"]["n_examples"] == 60

    import json
    all_names_by_split = {}
    for split in ("train", "val", "test"):
        path = tmp_path / lang / "data" / "reasoning" / f"{split}.jsonl"
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        names_here = set()
        for ex in examples:
            names_here.update(ex["entities"])
            _assert_answer_correct(ex)
        all_names_by_split[split] = names_here

    # cross-split leakage check on the ACTUAL generated data, not just the pools
    assert not (all_names_by_split["train"] & all_names_by_split["test"])
    assert not (all_names_by_split["val"] & all_names_by_split["test"])


def _assert_answer_correct(ex: dict):
    entities, values, answer, family = ex["entities"], ex["values"], ex["answer"], ex["family"]

    if family == FAMILY_EQ:
        assert answer in (LEXICONS[ex_lang_of(ex)].yes, LEXICONS[ex_lang_of(ex)].no)
        va, vb = values
        lex = LEXICONS[ex_lang_of(ex)]
        expected = lex.yes if va == vb else lex.no
        assert answer == expected
        return

    if values is None:
        # pure-relational transitive-chain family: narrative order IS the
        # ranking (A > B > C), independent of any lexicon
        a, b, c = entities
        if family == "C_endpoints_more":
            assert answer == a
        elif family == "C_endpoints_less":
            assert answer == c
        elif family == "C_most":
            assert answer == a
        elif family == "C_least":
            assert answer == c
        return

    if family in ("A_more", "B_most"):
        assert answer == entities[values.index(max(values))]
    elif family in ("A_less", "B_least"):
        assert answer == entities[values.index(min(values))]


def ex_lang_of(ex: dict) -> str:
    return ex["id"].split("_")[0]


def test_hindi_examples_are_internally_consistent(tmp_path):
    _check_examples("hindi", tmp_path)


def test_nepali_examples_are_internally_consistent(tmp_path):
    _check_examples("nepali", tmp_path)


def test_no_duplicate_prompts_within_a_split(tmp_path):
    stats = generate_all("hindi", tmp_path, n_train=200, n_val=50, n_test=50, seed=7)
    import json
    for split in ("train", "val", "test"):
        path = tmp_path / "hindi" / "data" / "reasoning" / f"{split}.jsonl"
        prompts = [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(prompts) == len(set(prompts))
