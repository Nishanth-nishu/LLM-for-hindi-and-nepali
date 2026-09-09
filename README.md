# Language Models and Agents — Individual Project

Two from-scratch, decoder-only, GPT-style Transformer language models —
**Model H (Hindi)** and **Model L (Nepali)** — built end to end: data
collection and cleaning, tokenizer training, pretraining, evaluation, and
reasoning finetuning with attention analysis. No `nn.Transformer*`, no
HuggingFace model classes, no pretrained weights or tokenizers, no shared
multilingual model — every component is implemented from primitive
PyTorch layers (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`)
and each language has its own independent tokenizer, model instance, and
training run.

**`phase-3` is the complete project snapshot** (per the assignment's
submission policy: "Phase 3 / branch `phase-3` at the final deadline is
your complete project snapshot"). `phase-1` and `phase-2` remain frozen as
they were graded; all Phase 3 work happened only on this branch.

## Repository layout

```
pipeline/
  model/        GPTConfig, GPTLanguageModel -- causal self-attention (from
                scratch), pre-norm blocks, tied embeddings, optional
                positional embeddings (ablation switch)
  train/        dataset.py (streaming tokenization + block dataset),
                train.py (AdamW, warmup+cosine LR, checkpointing),
                finetune.py (Phase 3.1: reasoning finetuning, answer-only
                loss masking, reuses train.py's optimizer/LR-schedule code)
  eval/         lm_metrics.py (PPL/BPB), generation_eval.py (BLEU-4/chrF/
                ROUGE-L, distinct-n, repetition rate), attention_analysis.py
                (heatmaps, entropy, mean attention distance, causal-mask
                verification), reasoning_eval.py (Phase 3.1: exact-match
                accuracy, pretrained vs finetuned), reasoning_attention_compare.py
                (Phase 3.2: pretrained-vs-finetuned attention comparison,
                reuses attention_analysis.py's toolkit unchanged)
  reasoning/    lexicon.py + generate_dataset.py -- the Phase 3.1 synthetic
                comparative/transitive-reasoning dataset generator (native
                Hindi/Nepali templates, not translated English), plus
                reasoning_dataset.py (tokenization + loss masking for
                finetuning)
<lang>/configs/                  architecture + training + finetuning hyperparameters
<lang>/tokenizer/vocab/          trained SentencePiece tokenizer (Phase 1 output)
<lang>/data/reasoning/           synthetic reasoning dataset (train/val/test, committed)
<lang>/checkpoints/              pretraining checkpoints + training_log.jsonl
<lang>/checkpoints_reasoning/    reasoning-finetuning checkpoints + training_log.jsonl
<lang>/checkpoints_ablation_*/   bonus: no-positional-embeddings ablation (Hindi only)
<lang>/data/stats/phase{2,3}_*.json   measured eval outputs (never invented/estimated)
report/phase1_*.md               Phase 1: data collection, cleaning, tokenizer
report/phase2_*.md               Phase 2: architecture, training, LM eval,
                                  generation eval, attention, ablation (bonus)
report/phase3_*.md               Phase 3: reasoning finetuning, attention
                                  comparison, final consolidated report
report/figures/                  loss curves, attention heatmaps, dataset figures
report/phase2_checkpoint_links.json   Google Drive links for every checkpoint
tests/                           unit tests (model correctness, reasoning
                                  dataset leakage/consistency checks)
```

Checkpoints (`.pt` files) and raw corpus splits are **not** committed to
git (see `.gitignore`) — they are large binaries, uploaded to Google Drive
instead (links below). Everything needed to *regenerate* them (code,
configs, tokenizers, the synthetic reasoning dataset itself) is committed.

## Google Drive links

See [`report/phase2_checkpoint_links.json`](report/phase2_checkpoint_links.json)
for the authoritative, up-to-date registry (which checkpoint is which run,
what it superseded, and why). Summary:

| Checkpoint | Status | Notes |
|---|---|---|
| **Hindi pretrained** | [Uploaded](https://drive.google.com/file/d/1M2jZQ_e_o0mh1WwGEYoXGOqqE0VgpcK1/view?usp=drive_link) | step 6500/15000, test PPL 16.12 / BPB 0.309 |
| **Nepali pretrained** | *pending upload* | step 6799/6800, test PPL 25.07 / BPB 0.479 — see recovery note in the JSON above |
| **Hindi reasoning-finetuned** | *pending upload* | best.pt (lowest val_loss, step 75), 33.5% test exact-match |
| **Nepali reasoning-finetuned** | *pending upload* | best.pt (lowest val_loss, step 75), 36.1% test exact-match |

Shared Drive folder (all checkpoints, viewable by anyone with the link):
https://drive.google.com/open?id=18GnBNUb7v0GlF5d1Ob7imOP4QaGElM1Q

## Results at a glance

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Parameters | 24.4M | 24.4M |
| Tokenizer fertility (tokens/word) | 1.6522 | 1.8457 |
| Real pretraining corpus (train tokens) | 474,751,658 | 471,622,290 |
| Test perplexity / BPB (pretrained) | 16.12 / 0.309 | 25.07 / 0.479 |
| Generation quality, greedy (BLEU-4 / chrF / ROUGE-L) | 0.032 / 0.159 / 0.119 | 0.010 / 0.099 / 0.051 |
| Reasoning accuracy, pretrained → finetuned | 0.0% → 33.5% | 0.0% → **36.1%** |

Full analysis and the evidence chain behind these numbers:
[`report/phase3_final_report.md`](report/phase3_final_report.md).

## Reproducing

```bash
pip install -r requirements.txt
```

### Phase 1 — data & tokenizer
Already produced; tokenizer files are committed under `<lang>/tokenizer/vocab/`.
See `report/phase1_*.md` for the collection/cleaning/tokenizer-training
pipeline and its own reproduction steps.

### Phase 2 — pretraining & evaluation
```bash
python -m pipeline.train.train --lang hindi --repo-root .
python -m pipeline.eval.lm_metrics --lang hindi --checkpoint hindi/checkpoints/latest.pt --split test
python -m pipeline.eval.generation_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt
python -m pipeline.eval.attention_analysis --lang hindi --checkpoint hindi/checkpoints/latest.pt
# repeat with --lang nepali

python tools/make_phase2_figures.py --repo-root .
python tools/make_phase2_reports.py --repo-root .
```
Bonus ablation (no positional embeddings, Hindi):
```bash
python -m pipeline.train.train --lang hindi --config model_config_ablation_baseline.yaml
python -m pipeline.train.train --lang hindi --config model_config_ablation_nopos.yaml
# then lm_metrics / generation_eval / attention_analysis with the matching --config
```

### Phase 3 — reasoning finetuning & attention comparison
```bash
# 1. synthetic dataset (already committed under <lang>/data/reasoning/, regenerate with:)
python -m pipeline.reasoning.generate_dataset --lang hindi  --repo-root .
python -m pipeline.reasoning.generate_dataset --lang nepali --repo-root .

# 2. finetune from each language's own pretrained checkpoint (needs the Drive checkpoints above)
python -m pipeline.train.finetune --lang hindi  --repo-root .
python -m pipeline.train.finetune --lang nepali --repo-root .

# 3. pretrained vs finetuned reasoning accuracy
python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt --tag pretrained
python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints_reasoning/best.pt --tag finetuned
# repeat with --lang nepali

# 4. pretrained vs finetuned attention comparison
python -m pipeline.eval.reasoning_attention_compare --lang hindi \
    --pretrained-checkpoint hindi/checkpoints/latest.pt \
    --finetuned-checkpoint hindi/checkpoints_reasoning/best.pt
# repeat with --lang nepali
```

### Tests
```bash
python -m pytest tests/ -v
```
14/14 passing: causal-mask correctness, parameter budget, checkpoint
round-trip, the no-positional-embeddings ablation's permutation-invariance
proof, and the reasoning dataset's leakage-avoidance + label-correctness
guarantees (disjoint entity-name pools, held-out template patterns,
independently re-derived answers, no duplicate prompts within a split).

## Compute

All pretraining, reasoning finetuning, and attention-comparison GPU runs
in this project ran on the IIIT-Hyderabad ADA cluster (SLURM, RTX 3090)
and/or Kaggle's free GPU tier — see `report/phase2_resource_comparison.md`
and `report/phase2_checkpoint_links.json`'s `training_runs` for the full,
honest account of what ran where (including the runs that failed or had
to be redone, e.g. Nepali's pretrained checkpoint being lost and
regenerated from scratch on ADA — documented rather than hidden).
