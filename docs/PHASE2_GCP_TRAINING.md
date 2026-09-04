# Phase 2 — Full-Budget Training Plan

This documents what it takes to go from the pilot checkpoints in
`report/phase2_training_report.md` (199 steps, ~0.5% of `max_steps`) to a
fully-trained Model H / Model L, and why the pilot stopped where it did.

## Hardware actually available

| Resource | Type | Notes |
|---|---|---|
| `nishanth-corpus-500m` | Vertex AI Workbench VM, `n2-highmem-8` (8 vCPU, 64GB RAM, Cascade Lake) | No GPU. This is what the pilot ran on. |
| GPU quota | `compute.googleapis.com/gpus_all_regions` | **0** in every region of project `lma-01` at the time of writing — a GPU VM cannot be created until Google approves a quota increase (Console → IAM & Admin → Quotas → "GPUs (all regions)" → Edit Quotas; typically minutes to ~2 business days on a billing-enabled project). |

Billing is enabled with a real budget, so **cost is not the constraint** —
even a multi-day CPU run for both models costs on the order of a few
hundred dollars against an $8K credit. **Wall-clock time is the
constraint.**

## Measured throughput (this hardware, this model)

Benchmarked directly on `nishanth-corpus-500m` with the real
`GPTConfig(vocab_size=4000, d_model=512, n_layer=7, n_head=8, d_ff=2048)`
model, batch 32 x block 512:

- ~1,100 tokens/sec forward+backward, single job, 8 threads.
- Tokenizing the full corpus once (SentencePiece, parallelized across 8
  processes) takes ~90-115s for ~500-560M tokens — no longer a bottleneck
  after `pipeline/train/dataset.py`'s parallel tokenizer fix (the original
  single-process version took 20+ minutes and was the reason the first
  pilot attempt was killed and restarted).

## Estimated time to one full epoch (CPU-only, this VM)

```
time = train_tokens / throughput

Hindi:  559,913,515 tokens / 1,100 tok/s ≈ 509,000 s ≈ 5.9 days
Nepali: 501,226,336 tokens / 1,100 tok/s ≈ 456,000 s ≈ 5.3 days
```

Running both concurrently (as the pilot did) roughly halves each job's
share of the 8 cores, so wall-clock for both finishing together is close
to the same ~6 days, not 11.

**One epoch is also the compute-optimal budget here** — see the Phase
1→2 handoff, section 5: 25M params x 20 tokens/param ≈ 500M tokens, and
both corpora sit just above that. So "one epoch, CPU-only" is a coherent
(if slow) target, not an arbitrary partial run.

## Path to hours instead of days: GPU

Once GPU quota is approved:

```bash
gcloud compute instances create nishanth-phase2-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB
```

A T4 (~$0.35-0.55/hr on-demand) at even a conservative 15-20x the measured
CPU throughput would bring one epoch down to roughly 8-12 hours per
model — well within a single extended session, at a cost of a few dollars
per full run.

## What "resuming" looks like

Every checkpoint (`pipeline/model/checkpoint.py`) carries model weights,
optimizer state, the LR-schedule step, and the config, so a full run is
just:

```bash
python -m pipeline.train.train --lang hindi --repo-root .
python -m pipeline.train.train --lang nepali --repo-root .
```

with `training.max_steps` in `<lang>/configs/model_config.yaml` left at
its real value (40,000) rather than the pilot's `--max-steps 200`
override, and the pilot-tuned `warmup_steps`/`eval_every`/`ckpt_every`
(shrunk for a 200-step run) restored to values sane for a 40,000-step run
(e.g. `warmup_steps: 1000`, `eval_every: 500`, `ckpt_every: 1000` — the
Phase 2 spec's starting values, edited down for the pilot and left that
way in the committed config since that's what actually produced the
committed results; restore them before a full run).

## Re-running evaluation after a full run

```bash
python -m pipeline.eval.lm_metrics --lang hindi --checkpoint hindi/checkpoints/latest.pt --split test
python -m pipeline.eval.generation_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt
python -m pipeline.eval.attention_analysis --lang hindi --checkpoint hindi/checkpoints/latest.pt
# repeat for nepali
python tools/make_phase2_figures.py --repo-root .
python tools/make_phase2_reports.py --repo-root .
```

This regenerates every `report/phase2_*.md` file from the new stats —
nothing is hand-edited, same discipline as Phase 1's `make_reports.py`.
