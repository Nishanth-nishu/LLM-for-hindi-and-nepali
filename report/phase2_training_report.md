# Phase 2 — Training Report

| | |
|---|---|
| Generated (UTC) | 2026-09-07 06:54:26 |
| Git commit | `2c72a3b797bc6a52b00fcb81d0b357ca59d72870` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Training setup

AdamW (`betas`, `weight_decay` per config; weight decay applied only to >=2D parameter tensors, not biases/LayerNorm), linear warmup then cosine decay to `min_lr`, gradient clipping, gradient accumulation. Source: `pipeline/train/train.py`. Each checkpoint contains model weights, optimizer state, the LR-schedule step, and the exact config (`pipeline/model/checkpoint.py`).

## Model H (higher-resource, Hindi)

| Setting | Value |
|---|--:|
| `batch_size` | 64 |
| `grad_accum_steps` | 1 |
| `max_steps` | 5000 |
| `warmup_steps` | 200 |
| `lr` | 0.0003 |
| `min_lr` | 3e-05 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `eval_every` | 100 |
| `ckpt_every` | 250 |

Effective tokens/step: **32,768**

### Progress

| Metric | First logged step | Latest logged step |
|---|--:|--:|
| Step | 0 | 4999 |
| Train loss | 8.3921 | 3.4781 |
| Val loss | 8.3853 | 3.0833 |
| Val PPL | 4382.23 | 21.83 |
| LR | 1.50e-06 | 3.00e-05 |

Best val loss so far: **3.0833**  
Training progress: **4,999 / 5,000 steps (100.0%)**.

Loss curve: `report\figures\phase2\hindi\loss_curve.png`

Checkpoint (Drive): https://drive.google.com/open?id=1Jb6hesJKgdeTYjpJh0SkL6IMl_ynG53P

## Model L (lower-resource, Nepali)

| Setting | Value |
|---|--:|
| `batch_size` | 64 |
| `grad_accum_steps` | 1 |
| `max_steps` | 5000 |
| `warmup_steps` | 200 |
| `lr` | 0.0003 |
| `min_lr` | 3e-05 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `eval_every` | 100 |
| `ckpt_every` | 250 |

Effective tokens/step: **32,768**

### Progress

| Metric | First logged step | Latest logged step |
|---|--:|--:|
| Step | 0 | 4999 |
| Train loss | 8.3890 | 3.6113 |
| Val loss | 8.3865 | 3.3429 |
| Val PPL | 4387.37 | 28.30 |
| LR | 1.50e-06 | 3.00e-05 |

Best val loss so far: **3.3429**  
Training progress: **4,999 / 5,000 steps (100.0%)**.

Loss curve: `report\figures\phase2\nepali\loss_curve.png`

Checkpoint (Drive): https://drive.google.com/open?id=1w8gCOZa5zHn09lQ3kMT_jRN2o_8poino
