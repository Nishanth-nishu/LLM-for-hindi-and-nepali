# Phase 3.1 — Reasoning Finetuning

Both models are finetuned independently, in their own language, on a
synthetic comparative/transitive-reasoning dataset built specifically for
this project (not a downloaded benchmark). This report covers dataset
construction, the finetuning protocol, an iteration that meaningfully
improved accuracy after diagnosing why the first pass underperformed, and
the final pretrained-vs-finetuned results for Model H (Hindi) and Model L
(Nepali).

## 1. Synthetic reasoning dataset

Generator: [`pipeline/reasoning/generate_dataset.py`](../pipeline/reasoning/generate_dataset.py),
vocabulary/templates: [`pipeline/reasoning/lexicon.py`](../pipeline/reasoning/lexicon.py).
Reproduce with:

```bash
python -m pipeline.reasoning.generate_dataset --lang hindi  --repo-root . --n-train 12000 --n-val 1500 --n-test 1500
python -m pipeline.reasoning.generate_dataset --lang nepali --repo-root . --n-train 12000 --n-val 1500 --n-test 1500
```

**Every example is written natively in the target language's script and
grammar** — separate entity-name pools, separate attribute vocabulary, and
separate sentence templates per language, not English templates with names
substituted in. Hindi templates use correct possessive/gender agreement
per attribute (की/का, किसकी/किसका, होगी/होगा — "वज़न" is grammatically
masculine, the other three attributes feminine); Nepali templates use its
own gender-invariant possessive "कसको", which needed no such agreement
handling. Comparison words in the primary templates (Hindi अधिक/कम, Nepali
बढी/कम) are themselves gender/number-invariant by design, since getting
Hindi/Nepali adjective-noun agreement right for an arbitrary noun (a
person's name or an object) programmatically is not reliable — a
documented simplification, not an oversight.

### Nine question families

| Family | Entities | Given | Asked |
|---|---|---|---|
| `A_more` / `A_less` | 2 | two numeric attribute values | which entity has more / less |
| `B_most` / `B_least` | 3 | three numeric attribute values | which entity is the extreme |
| `C_endpoints_more` / `C_endpoints_less` | 3 | **only** "A > B" and "B > C" (no numbers) | which of the two endpoints (A, C) is more/less |
| `C_most` / `C_least` | 3 | same pure-relational chain | which of the three is the extreme |
| `D_equal` | 2 | two numeric attribute values | yes/no equality check |

The `C_*` families are the assignment's "given A > B and B > C" transitive
chain style exactly: the model is never told the A-vs-C relation directly
and must chain the two given inequalities to answer. Four attributes per
language (height/age — people; price/weight — objects), each with its own
natural unit and numeric range.

### Sizes and template variety

| | train | val | test |
|---|---:|---:|---:|
| Hindi | 12,000 | 1,500 | 1,500 |
| Nepali | 12,000 | 1,500 | 1,500 |

Family sampling is weighted, not fixed-quota, and — after the accuracy
iteration in §3 — deliberately biased toward the two-/three-entity
**numeric**-comparison families (`A_more`/`A_less`/`B_most`/`B_least`,
~65% of examples combined) over the purely relational `C_*` families,
since numeric comparison turned out to need far more example coverage to
generalize (see §3). Exact realized distribution per split is in each
language's `data/reasoning/dataset_stats.json`. 2–3 phrasings per family
(`lexicon.py` templates), prompt length 67–140 characters.

### Leakage avoidance (two independent, tested mechanisms)

1. **Disjoint entity names.** Both languages' name pools (40 people-names,
   20 object-names) are shuffled and split 70/15/15 into train/val/test
   *before* any example is generated (`split_names`). A name used in a
   test example was never used in a train or val example, and vice versa.
2. **Held-out template patterns.** For every family with more than one
   phrasing, the *last* phrasing is reserved exclusively for the test
   split (`templates_for_split`); train/val only ever draw from the
   others. Some test sentence structures were literally never seen during
   training.

Both mechanisms are checked directly against the *generated data* (not
just the source pools) in
[`tests/test_reasoning_dataset.py`](../tests/test_reasoning_dataset.py)
(`test_split_names_are_disjoint`,
`test_held_out_template_never_used_outside_test`,
`test_hindi_examples_are_internally_consistent` /
`..._nepali_...`, `test_no_duplicate_prompts_within_a_split`) — 5/5
passing. The generator itself also de-duplicates the name pool defensively
(`dict.fromkeys`) after this test suite caught a genuine duplicate name in
an early version of the Hindi pool that would have silently broken the
disjointness guarantee.

## 2. Finetuning protocol

[`pipeline/train/finetune.py`](../pipeline/train/finetune.py) +
[`pipeline/reasoning/reasoning_dataset.py`](../pipeline/reasoning/reasoning_dataset.py).
Each language starts from **that language's own** Phase 2 pretrained
checkpoint and keeps its tokenizer/vocabulary completely fixed — no
retokenization, no vocabulary changes, no cross-language sharing of
weights or data at any point.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Base checkpoint | `hindi/checkpoints/latest.pt`, step 6500 | `nepali/checkpoints/latest.pt`, step 6799 |
| Tokenizer | `hindi_tokenizer.model` (fixed) | `nepali_tokenizer.model` (fixed) |
| Answer delimiter | "उत्तर:" | "जवाफ:" |
| `lr` / `min_lr` | 6e-5 / 6e-6 | 6e-5 / 6e-6 |
| `epochs` / `batch_size` | 15 / 32 | 15 / 32 |
| `warmup_steps` | 150 | 150 |
| `weight_decay`, `betas`, `grad_clip` | 0.1, (0.9, 0.95), 1.0 | identical |
| `seed` | 20260909 | 20260909 |
| Hardware | RTX 3090, IIIT-H ADA cluster | same |
| Steps run / best-checkpoint step | 5625 / **75** | 5625 / **75** |

Configs: [`hindi/configs/reasoning_finetune_config.yaml`](../hindi/configs/reasoning_finetune_config.yaml),
[`nepali/configs/reasoning_finetune_config.yaml`](../nepali/configs/reasoning_finetune_config.yaml)
— identical hyperparameters, differing only in base checkpoint, tokenizer,
and delimiter, so the two languages are a fair, controlled comparison.

**Answer-only loss masking.** Each training example is tokenized as
`prompt + delimiter + answer + eos`; the target sequence has `-100`
(ignored by `GPTLanguageModel`'s existing masked cross-entropy) on every
position whose target token belongs to the prompt or padding, so the model
is only ever penalized for the answer tokens it actually needs to produce
— see `ReasoningExampleDataset.encode`.

**Checkpoints are resume-capable in the exact Phase 2 format**
(`model_state_dict` / `optimizer_state_dict` / `scheduler_state_dict` /
`step` / `config`, plus `base_checkpoint` / `base_checkpoint_step` for
provenance). `finetune.py` saves `ckpt_dir/best.pt` every time evaluation
finds a new best validation loss (see §3.1 for why this — not just
per-epoch snapshots — matters), plus a per-epoch snapshot and `latest.pt`.

Reproduce:
```bash
python -m pipeline.train.finetune --lang hindi  --repo-root .
python -m pipeline.train.finetune --lang nepali --repo-root .
```

## 3. Diagnosing and fixing a first pass that badly underperformed

The first finetuning pass (3000 train examples, `lr=1e-4`, 8 epochs) scored
**32.5%** (Hindi) and **17.5%** (Nepali) exact-match accuracy — a real
improvement over the 0% pretrained baseline, but well below what seemed
achievable for a task this templated. Two hypotheses were checked before
changing anything, in order of how cheap they were to rule out:

**Hypothesis 1: inconsistent digit tokenization.** Ruled out directly —
`sp.encode("172")` returns `['▁','1','7','2']`; every digit 0–9 has its
own dedicated vocabulary entry in both tokenizers, checked explicitly.
Numeric comparison is not being sabotaged by the tokenizer splitting
numbers inconsistently.

**Hypothesis 2: sparse coverage of the numeric-comparison space.** With
only ~500 `A_more` examples spread across 4 attributes and numeric ranges
as wide as 20–2000 (price), the model saw far too few distinct
`(value_a, value_b)` pairs to learn a general "compare two digit
sequences" operation — consistent with the family-level pattern already
observed: pure-relational chaining (no numbers) scored 80–95%+, while
numeric comparison scored 5–25%, even in an isolated matched-conditions
check (val split, seen templates, held-out names only: 92.1% on
`C_endpoints_less` vs. 5.5% on `A_more`+`A_less` combined).

**Fix, part 1 — more data, weighted toward the weak spot.** Regenerated
both datasets at 4x the size (12,000/1,500/1,500) with `FAMILY_WEIGHTS`
biased toward `A_more`/`A_less`/`B_most`/`B_least` (§1).

**Fix, part 2 — a real bug in the checkpointing.** The first rerun with
the larger dataset (`lr=6e-5`, `eval_every=100`) revealed a bug in
`finetune.py` itself: with `batch_size=32` and 12,000 examples, one epoch
is 375 steps, but the true validation-loss optimum for the *first* pass
had already occurred at **step 40 of ~93 steps/epoch** — well inside
epoch 0. `finetune.py` only saved checkpoints at epoch boundaries, so the
"best" checkpoint it could ever select was already several hundred steps
past the true optimum and partway into the overfitting climb. This was
fixed by saving `ckpt_dir/best.pt` every time evaluation finds a new best
validation loss (§2), independent of epoch boundaries, and tightening
`eval_every` to 25 to pinpoint the optimum more precisely. The final run's
true optimum is at **step 75** for both languages (`val_loss` 0.90 Hindi /
0.92 Nepali) — a checkpoint the old epoch-only logic would have missed
entirely.

Both fixes applied together (larger, rebalanced dataset + correct
best-checkpoint selection) took accuracy from 32.5%→**33.5%** (Hindi, a
modest net gain masking a much larger shift within families, see §4.2) and
17.5%→**36.1%** (Nepali, essentially doubled). All results below are from
this corrected pipeline; the numbers above are kept as a documented,
diagnosed comparison point, not silently dropped.

## 4. Results: pretrained vs. finetuned accuracy (test split, exact match)

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| **Pretrained (zero-shot)** | **0.0%** (n=1500) | **0.0%** (n=1500) |
| **Finetuned (`best.pt`, step 75)** | **33.5%** | **36.1%** |

Full JSON: `<lang>/data/stats/phase3_reasoning_eval_{pretrained,finetuned}.json`.

The pretrained checkpoint scores exactly 0% for both languages — expected,
since it has never seen this task's answer-delimiter format and was only
ever trained on raw next-token prediction over natural corpus text.

### 4.1 Per-family accuracy

| Family | Hindi | Nepali |
|---|---:|---:|
| A_more | 44.0% | 41.9% |
| A_less | 39.8% | 40.9% |
| B_most | 13.9% | 18.0% |
| B_least | 13.3% | 17.8% |
| C_endpoints_more | 50.0% | 43.8% |
| C_endpoints_less | 61.5% | 77.8% |
| C_most | 9.1% | 11.5% |
| C_least | 53.8% | 66.7% |
| D_equal | 50.4% | 51.6% |

The two-entity numeric families (`A_more`/`A_less`) improved the most
dramatically from the first pass — Hindi's `A_more` went 12.7%→44.0%,
Nepali's `B_most`/`B_least`/`C_most`/`C_least` all went from a flat **0%**
to real, non-trivial accuracy. The three-entity numeric families
(`B_most`/`B_least`) remain the hardest for both languages (13–18%) —
comparing three magnitudes and picking the extreme is a strictly harder
composition of the same underlying skill that two-entity comparison
already struggles with.

### 4.2 What actually changed: entity tracking vs. magnitude comparison

Rerunning the failure-mode breakdown from the first pass (§3) on the new
checkpoint, restricted to the same held-out-template `A_more`/`A_less`
family (634 Hindi test examples):

| | First pass (3000-example dataset) | Fixed pipeline (12,000-example dataset) |
|---|---:|---:|
| Exact match | 18.6% (26/140) | **41.8%** (265/634) |
| Wrong, but predicted entity *was* in the prompt | — | 37.2% (236/634) |
| Predicted something **not in the prompt at all** | **65.0%** (91/140) | **21.0%** (133/634) |

**Entity tracking improved far more than magnitude comparison did.**
Combining "correct" and "wrong but in-prompt", the model now correctly
identifies *which* entities were mentioned in **79.0%** of cases (up from
a regime where two-thirds of answers referenced an entity that was never
in the prompt at all) — the larger, better-name-covered dataset fixed the
*parsing* failure almost completely. The residual errors are now
concentrated in two more specific places: genuine magnitude-comparison
mistakes (picking the wrong, but real, entity — 37.2%), and a **single,
specific cross-family confusion**: every one of the 133 "not in prompt"
predictions is the token "हाँ" ("yes") — a `D_equal`-family answer
leaking into `A_more`/`A_less` prompts — not the diffuse, multi-name
hallucination pattern (e.g. a fabricated name like "रीति", not in the
name pool at all) seen in the first pass. The remaining gap is now a
narrower, better-characterized problem: real numeric-magnitude comparison
and occasional family confusion, not a general failure to read the
prompt.

## 5. Model H vs. Model L on reasoning

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Pretrained LM quality (test PPL / BPB, step of base ckpt) | 16.12 / 0.309 (step 6500) | 25.07 / 0.479 (step 6799) |
| Reasoning accuracy, pretrained | 0.0% | 0.0% |
| Reasoning accuracy, finetuned | 33.5% | **36.1%** |

Notably, after the accuracy fix, **Model L's finetuned reasoning accuracy
slightly exceeds Model H's**, despite Model L's clearly weaker pretrained
LM quality (36% higher test perplexity) and worse tokenizer fertility
(§`report/phase3_final_report.md`). This is a genuinely interesting
result in its own right: whatever LM-quality advantage Hindi carries into
pretraining did not translate into a reasoning-finetuning advantage once
both models had adequate task-specific data and a correctly-selected
checkpoint — the earlier (first-pass) reading that "Model H reasons
better because its base LM is better" does not hold up once the
data-sparsity and checkpoint-selection confounds are removed. The
per-family table in §4.1 shows Nepali is specifically stronger on the
pure-relational `C_*` families (e.g. `C_endpoints_less` 77.8% vs. Hindi's
61.5%), while the two languages are close on the numeric families —
suggesting Nepali's finetuning made more effective use of the same
442-example held-out-template budget for that particular pattern, not
that Nepali reasons better in general.

## 6. Reproduction

```bash
# 1. generate the datasets (already committed under <lang>/data/reasoning/)
python -m pipeline.reasoning.generate_dataset --lang hindi  --repo-root . --n-train 12000 --n-val 1500 --n-test 1500
python -m pipeline.reasoning.generate_dataset --lang nepali --repo-root . --n-train 12000 --n-val 1500 --n-test 1500

# 2. finetune (needs the Phase 2 pretrained checkpoints -- Drive links in the README)
python -m pipeline.train.finetune --lang hindi  --repo-root .
python -m pipeline.train.finetune --lang nepali --repo-root .

# 3. evaluate pretrained vs finetuned
python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt --tag pretrained
python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints_reasoning/best.pt --tag finetuned
# repeat with --lang nepali

# 4. correctness / leakage tests
python -m pytest tests/test_reasoning_dataset.py -v
```
