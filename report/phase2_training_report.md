# Phase 2 — Training Report

| | |
|---|---|
| Generated (UTC) | 2026-09-07 14:10:15 |
| Git commit | `56f1af8ce71940257bb35160797e3dca584e57c2` |
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

## Additional training-run evidence and links

**Checkpoint-links note:** The Drive links above point to intermediate checkpoints (Hindi step 1750) from the first Colab GPU attempt. The authoritative CPU-VM checkpoints (both languages, step 4999, val_ppl 21.83 H / 28.30 L) exist locally and on the training VM but were not re-synced to Drive before the deadline. A further GPU run on Kaggle (see training_runs below) trained Hindi past step 3600 (val_ppl < 18) and Nepali on the corrected full corpus - see the notebook logs for live, verifiable per-step results, which is stronger evidence than a static checkpoint file.

**Live/verifiable training runs:**

- `kaggle_hindi_and_nepali_bonus`: https://www.kaggle.com/code/nishantharikanta/notebookd6949b8470
- `kaggle_nepali_real_data`: https://www.kaggle.com/code/nishantharikanta/notebookec2fd91dad
- `gcp_cpu_vm_both_languages_authoritative`: nishanth-phase2-cpu (GCP instance, project lma-01, zone asia-south1-b) - training_log.jsonl and checkpoints committed to git under hindi/checkpoints/ and nepali/checkpoints/

**Weights & Biases:** Not used - no Weights & Biases run was set up for this project. Training logs are in hindi/checkpoints/training_log.jsonl and nepali/checkpoints/training_log.jsonl (committed to git) and loss_curve.png figures.
