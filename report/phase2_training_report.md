# Phase 2 — Training Report

| | |
|---|---|
| Generated (UTC) | 2026-09-03 23:42:39 |
| Git commit | `c2a82c02fa66e6395fd229e237a102fc4cb385f5` |
| Branch | `phase-1` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Training setup

AdamW (`betas`, `weight_decay` per config; weight decay applied only to >=2D parameter tensors, not biases/LayerNorm), linear warmup then cosine decay to `min_lr`, gradient clipping, gradient accumulation. Source: `pipeline/train/train.py`. Each checkpoint contains model weights, optimizer state, the LR-schedule step, and the exact config (`pipeline/model/checkpoint.py`).

## Model H (higher-resource, Hindi)

| Setting | Value |
|---|--:|
| `batch_size` | 32 |
| `grad_accum_steps` | 1 |
| `max_steps` | 40000 |
| `warmup_steps` | 20 |
| `lr` | 0.0003 |
| `min_lr` | 3e-05 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `eval_every` | 40 |
| `ckpt_every` | 100 |

Effective tokens/step: **16,384**

### Progress

| Metric | First logged step | Latest logged step |
|---|--:|--:|
| Step | 0 | 199 |
| Train loss | 8.3921 | 5.7241 |
| Val loss | 8.3034 | 5.6619 |
| Val PPL | 4037.67 | 287.71 |
| LR | 1.50e-05 | 3.00e-05 |

Best val loss so far: **5.6619**  
Training progress: **199 / 40,000 steps (0.5%)**.

**This is a partial/pilot run, not a converged model** — see `docs/PHASE2_GCP_TRAINING.md` for the full-budget run plan and estimated wall-clock time on the project's hardware.

Loss curve: `report/figures/phase2/hindi/loss_curve.png`

## Model L (lower-resource, Nepali)

| Setting | Value |
|---|--:|
| `batch_size` | 32 |
| `grad_accum_steps` | 1 |
| `max_steps` | 40000 |
| `warmup_steps` | 20 |
| `lr` | 0.0003 |
| `min_lr` | 3e-05 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `eval_every` | 40 |
| `ckpt_every` | 100 |

Effective tokens/step: **16,384**

### Progress

| Metric | First logged step | Latest logged step |
|---|--:|--:|
| Step | 0 | 199 |
| Train loss | 8.3859 | 6.2850 |
| Val loss | 8.3108 | 6.2139 |
| Val PPL | 4067.71 | 499.65 |
| LR | 1.50e-05 | 3.00e-05 |

Best val loss so far: **6.2139**  
Training progress: **199 / 40,000 steps (0.5%)**.

**This is a partial/pilot run, not a converged model** — see `docs/PHASE2_GCP_TRAINING.md` for the full-budget run plan and estimated wall-clock time on the project's hardware.

Loss curve: `report/figures/phase2/nepali/loss_curve.png`
