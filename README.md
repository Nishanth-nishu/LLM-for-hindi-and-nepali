# Phase 2 — From-Scratch GPT-Style Transformer (Model H / Model L)

This branch contains **only Phase 2**: the decoder-only Transformer
implementation, training, and evaluation. Phase 1 (data collection,
cleaning, tokenizer training) lives on the `phase-1` branch — this branch
consumes its two outputs (the SentencePiece tokenizer models under
`<lang>/tokenizer/vocab/`) but not its pipeline code, configs, or reports.

## What's here

```
pipeline/
  model/    GPTConfig, GPTLanguageModel — causal self-attention, positional
            embeddings, pre-norm blocks, tied embeddings, all built from
            nn.Linear / nn.Embedding / nn.LayerNorm / nn.Dropout only
  train/    dataset.py (parallel SentencePiece tokenization + block dataset),
            train.py (AdamW, warmup+cosine LR, checkpointing)
  eval/     lm_metrics.py (PPL/BPB), generation_eval.py (BLEU-4/chrF/ROUGE-L,
            distinct-n, repetition rate), attention_analysis.py (heatmaps,
            entropy, mean attention distance, causal-mask verification)
<lang>/configs/model_config.yaml   architecture + training hyperparameters
<lang>/tokenizer/vocab/            trained tokenizer (Phase 1 output, consumed here)
<lang>/checkpoints/training_log.jsonl
<lang>/data/stats/phase2_*.json    measured eval outputs
report/phase2_*.md                 the six Phase 2 deliverable reports
report/figures/phase2/             loss curves, attention heatmaps
docs/PHASE2_GCP_TRAINING.md        full-run time/cost plan, GPU path
tests/test_phase2_model.py         causal-mask, param-count, checkpoint tests
run_phase2.py                      stage orchestrator (train / evaluate-lm /
                                    evaluate-generation / attention / report)
```

## Status

Both models: ~24.4M parameters, verified architecture (8 passing tests).
A real bounded pilot (200 steps each) has been trained and evaluated on
the actual Phase 1 corpora — see `report/phase2_training_report.md` for
loss curves and `report/phase2_lm_evaluation_report.md` /
`report/phase2_generation_report.md` / `report/phase2_attention_report.md`
for measured PPL/BPB, generation quality, and attention analysis.

**This is a pilot, not a converged model** (0.5% of the planned 40,000
steps) — CPU-only hardware in this project makes a full run take days per
model; `docs/PHASE2_GCP_TRAINING.md` has the measured throughput, the
full-run time estimate, and the GPU path to bring that down to hours.

## Reproducing

```bash
pip install -r requirements.txt

python -m pipeline.train.train --lang hindi --repo-root .
python -m pipeline.eval.lm_metrics --lang hindi --checkpoint hindi/checkpoints/latest.pt --split test
python -m pipeline.eval.generation_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt
python -m pipeline.eval.attention_analysis --lang hindi --checkpoint hindi/checkpoints/latest.pt
# repeat with --lang nepali

python tools/make_phase2_figures.py --repo-root .
python tools/make_phase2_reports.py --repo-root .
```

Checkpoints are not committed (large binaries) — Drive links are in
`report/phase2_checkpoint_links.json` and inlined in
`report/phase2_training_report.md`.
