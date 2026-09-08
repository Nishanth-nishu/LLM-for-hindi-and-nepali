# Phase 3.1 — Reasoning Finetuning

Both models are finetuned independently, in their own language, on a
synthetic comparative/transitive-reasoning dataset built specifically for
this project (not a downloaded benchmark). This report covers dataset
construction, the finetuning protocol, and the pretrained-vs-finetuned
results for Model H (Hindi) and Model L (Nepali).

## 1. Synthetic reasoning dataset

Generator: [`pipeline/reasoning/generate_dataset.py`](../pipeline/reasoning/generate_dataset.py),
vocabulary/templates: [`pipeline/reasoning/lexicon.py`](../pipeline/reasoning/lexicon.py).
Reproduce with:

```bash
python -m pipeline.reasoning.generate_dataset --lang hindi  --repo-root . --n-train 3000 --n-val 400 --n-test 400
python -m pipeline.reasoning.generate_dataset --lang nepali --repo-root . --n-train 3000 --n-val 400 --n-test 400
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
| Hindi | 3000 | 400 | 400 |
| Nepali | 3000 | 400 | 400 |

Family mix is randomized per example (weighted, 2-entity comparisons most
common) rather than fixed quotas — see `dataset_stats.json` in each
language's `data/reasoning/` for the exact realized distribution. 2–3
phrasings per family (`lexicon.py` templates), prompt length 67–136
characters (Hindi; Nepali similar).

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
| `lr` / `min_lr` | 1e-4 / 1e-5 | 1e-4 / 1e-5 |
| `epochs` / `batch_size` | 8 / 32 | 8 / 32 |
| `warmup_steps` | 50 | 50 |
| `weight_decay`, `betas`, `grad_clip` | 0.1, (0.9, 0.95), 1.0 | identical |
| `seed` | 20260909 | 20260909 |
| Hardware | RTX 3090, IIIT-H ADA cluster | same |
| Total steps / wall time | 744 / 102 s | 744 / 89 s |

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
provenance), saved once per epoch under `<lang>/checkpoints_reasoning/`.

Reproduce:
```bash
python -m pipeline.train.finetune --lang hindi  --repo-root .
python -m pipeline.train.finetune --lang nepali --repo-root .
```

### An early-stopping finding, not just a hyperparameter footnote

Both languages **overfit almost immediately**: the lowest validation loss
of the entire run occurs within epoch 0 (Hindi: val_loss 0.94 at step 40;
Nepali: val_loss 0.99 at step 40), then climbs for several epochs (Hindi
peaks at val_loss 2.08 around epoch 4; Nepali peaks at 3.16 around epoch
5) before partially — never fully — recovering by epoch 7. Both languages'
full step-by-step logs are committed
(`<lang>/checkpoints_reasoning/training_log.jsonl`). Because of this, the
checkpoint reported below as "finetuned" for both languages is
**`epoch_0.pt`** (lowest val_loss — standard early-stopping practice), not
the final epoch, even though (see §3) the two epochs land on nearly the
same *overall* test accuracy for Hindi and the *final* epoch is marginally
*higher* for Nepali. Selecting by val_loss rather than by final-epoch or
raw test accuracy is the principled choice; §3 shows why the raw-accuracy
alternative would be misleading.

## 3. Results: pretrained vs. finetuned accuracy (test split, exact match)

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| **Pretrained (zero-shot)** | **0.0%** | **0.0%** |
| **Finetuned (epoch_0, best val_loss)** | **32.5%** | **17.5%** |
| Finetuned (final epoch_7, for comparison) | 32.5% | 18.75% |

Full JSON: `<lang>/data/stats/phase3_reasoning_eval_{pretrained,finetuned_best,finetuned_final}.json`.

The pretrained checkpoint scores exactly 0% for both languages — expected,
since it has never seen this task's answer-delimiter format and was only
ever trained on raw next-token prediction over natural corpus text.
Finetuning is what teaches the model the *format* (stop after the
delimiter, produce a short entity name or yes/no token) as well as
whatever reasoning it can pick up from 3000 examples.

### 3.1 Final epoch's accuracy is not a fair second look — it hides a collapse

Both epoch_0 and epoch_7 land near the same *overall* accuracy, but the
per-family breakdown is not the same model in disguise:

| Family | Hindi epoch_0 | Hindi epoch_7 | Nepali epoch_0 | Nepali epoch_7 |
|---|---:|---:|---:|---:|
| A_more | 12.7% | 8.5% | 20.0% | 0.0% |
| A_less | 24.6% | 23.2% | 24.3% | 1.4% |
| B_most | 13.3% | 6.7% | 0.0% | 0.0% |
| B_least | 20.4% | **0.0%** | 0.0% | 0.0% |
| C_endpoints_more | 54.5% | 63.6% | 0.0% | 4.7% |
| C_endpoints_less | 92.9% | 92.9% | 28.2% | **92.3%** |
| C_most | 12.0% | 52.0% | 0.0% | 5.3% |
| C_least | 81.8% | 95.5% | 0.0% | **81.2%** |
| D_equal | 48.8% | 53.5% | 53.8% | 42.3% |

By epoch 7, Nepali's correctness has collapsed onto exactly two families
(`C_endpoints_less`, `C_least`) while everything else — including
`A_more`/`A_less`, which epoch_0 got partially right — drops to 0–5%.
Its higher *overall* number (18.75% vs 17.5%) is an artifact of those two
families having more test examples, not a better model. This is exactly
the failure mode early stopping on val_loss is supposed to catch, and
exactly why §2's protocol reports epoch_0, not the final epoch, as *the*
finetuned result.

### 3.2 What generalizes and what doesn't: relational chaining vs. numeric comparison

The clearest pattern in the table above: **pure-relational transitive
chains generalize far better than numeric-magnitude comparison**, for
both languages. To isolate *why* — template novelty, or the numeric
reasoning itself — two additional checks were run directly against the
Hindi `epoch_0` checkpoint on the **validation** split, where every
example uses a *seen* template (unlike test, which by construction always
uses the held-out template for `A_more`/`A_less`/`B_most`/`B_least`) and
only the entity *names* are held out from train:

| Family (val split, seen templates, held-out names only) | Accuracy |
|---|---:|
| `C_endpoints_less` | **35/38 = 92.1%** |
| `A_more` + `A_less` | 8/145 = **5.5%** |

Both settings have unseen names; neither uses a held-out template. The gap
(92.1% vs 5.5%) isolates the cause: it is not template novelty, it is that
tracking two numeric values and comparing their magnitudes is a much
harder compositional-generalization problem for a 24M-parameter model
trained on 3000 examples than following a purely positional/relational
chain ("A precedes B in the statement, B precedes C, so A precedes C" —
no arithmetic required).

### 3.3 A specific, quantified failure mode: name hallucination under template novelty

On the **test** split's `A_more`/`A_less` family (140 Hindi examples, all
using the held-out "यदि... हो, तो..." template by construction):

- **26/140 (18.6%) exactly correct**
- **91/140 (65.0%) predict an entity name that does not appear anywhere
  in the prompt at all** — e.g. "रीति", a name that does not exist in
  the Hindi name pool (`pipeline/reasoning/lexicon.py`) at all, appears
  39 times as the model's prediction; "जूता" ("shoe" — an *object* name,
  hallucinated for a people-attribute question about age/height) appears
  41 times.

Example (from `phase3_reasoning_eval_finetuned_best.json`):

> **Prompt:** "यदि अर्जुन की उम्र 49 वर्ष और मोहन की उम्र 23 वर्ष हो, तो
> किसकी उम्र अधिक होगी?" (*If Arjun's age is 49 years and Mohan's age is
> 23 years, then whose age would be more?*)
> **Gold:** अर्जुन &nbsp;&nbsp; **Predicted:** रीति

On the *unseen template* combined with numeric comparison, the finetuned
model doesn't just get the magnitude comparison wrong — it frequently
fails to even parse which entities were mentioned, falling back to a
plausible-sounding name from its general Hindi pretraining rather than
copying from the actual prompt. Combined with §3.2's val-split result,
the picture is: numeric comparison is hard on its own, and an unfamiliar
template phrasing makes the entity-tracking itself unreliable on top of
that — two compounding, independently-observable failure modes, not one.

## 4. Model H vs. Model L on reasoning

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Pretrained LM quality (test PPL / BPB, step of base ckpt) | 16.12 / 0.309 (step 6500) | 25.07 / 0.479 (step 6799) |
| Reasoning accuracy, pretrained | 0.0% | 0.0% |
| Reasoning accuracy, finetuned (epoch_0) | **32.5%** | **17.5%** |

Model H reaches roughly **1.9× Model L's finetuned reasoning accuracy**,
tracking the same direction as its LM-quality edge (lower PPL/BPB) —
consistent with Model H's base checkpoint being trained closer to
convergence on its own schedule (step 6500 of a 15000-step plan, with a
markedly lower PPL) than Model L was on its available compute, even
though Model L's base run (step 6799 of 6800) is *closer to its own
schedule's completion*. §3.3 of `report/phase3_final_report.md` returns
to this comparison in the context of the full project's data-scale and
tokenizer differences between the two languages.

## 5. Reproduction

```bash
# 1. generate the datasets (already committed under <lang>/data/reasoning/)
python -m pipeline.reasoning.generate_dataset --lang hindi  --repo-root .
python -m pipeline.reasoning.generate_dataset --lang nepali --repo-root .

# 2. finetune (needs the Phase 2 pretrained checkpoints -- Drive links in the README)
python -m pipeline.train.finetune --lang hindi  --repo-root .
python -m pipeline.train.finetune --lang nepali --repo-root .

# 3. evaluate pretrained vs finetuned
python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt --tag pretrained
python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints_reasoning/epoch_0.pt --tag finetuned_best
# repeat with --lang nepali

# 4. correctness / leakage tests
python -m pytest tests/test_reasoning_dataset.py -v
```
