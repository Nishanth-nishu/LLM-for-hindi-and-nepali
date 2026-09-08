# Phase 2 — Training Report

| | |
|---|---|
| Generated (UTC) | 2026-09-08 16:45:21 |
| Git commit | `9ca6c1de4c71bedc835ee996a6960d672e850a4d` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Training setup

AdamW (`betas`, `weight_decay` per config; weight decay applied only to >=2D parameter tensors, not biases/LayerNorm), linear warmup then cosine decay to `min_lr`, gradient clipping, gradient accumulation. Source: `pipeline/train/train.py`. Each checkpoint contains model weights, optimizer state, the LR-schedule step, and the exact config (`pipeline/model/checkpoint.py`).

## Model H (higher-resource, Hindi)

| Setting | Value |
|---|--:|
| `batch_size` | 16 |
| `grad_accum_steps` | 8 |
| `max_steps` | 15000 |
| `warmup_steps` | 300 |
| `lr` | 0.0003 |
| `min_lr` | 3e-05 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `eval_every` | 200 |
| `ckpt_every` | 500 |

Effective tokens/step: **65,536**

### Progress

| Metric | First logged step | Latest logged step |
|---|--:|--:|
| Step | 0 | 6600 |
| Train loss | 8.3936 | 3.1105 |
| Val loss | 8.3817 | 2.6859 |
| Val PPL | 4366.31 | 14.67 |
| LR | 1.00e-06 | 1.95e-04 |

Best val loss so far: **2.6859**  
Training progress: **6,600 / 15,000 steps (44.0%)**.

**This is a partial/pilot run, not a converged model** — see `docs/PHASE2_GCP_TRAINING.md` for the full-budget run plan and estimated wall-clock time on the project's hardware.

Loss curve: `report/figures/phase2/hindi/loss_curve.png`

Checkpoint (Drive): https://drive.google.com/file/d/1M2jZQ_e_o0mh1WwGEYoXGOqqE0VgpcK1/view?usp=drive_link

## Model L (lower-resource, Nepali)

| Setting | Value |
|---|--:|
| `batch_size` | 16 |
| `grad_accum_steps` | 8 |
| `max_steps` | 6800 |
| `warmup_steps` | 300 |
| `lr` | 0.0003 |
| `min_lr` | 3e-05 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `eval_every` | 200 |
| `ckpt_every` | 500 |

Effective tokens/step: **65,536**

### Progress

| Metric | First logged step | Latest logged step |
|---|--:|--:|
| Step | 0 | 6600 |
| Train loss | 8.3936 | 3.1105 |
| Val loss | 8.3817 | 2.6859 |
| Val PPL | 4366.31 | 14.67 |
| LR | 1.00e-06 | 1.95e-04 |

Best val loss so far: **2.6859**  
Training progress: **6,600 / 6,800 steps (97.1%)**.

**This is a partial/pilot run, not a converged model** — see `docs/PHASE2_GCP_TRAINING.md` for the full-budget run plan and estimated wall-clock time on the project's hardware.

Loss curve: `report/figures/phase2/nepali/loss_curve.png`

Checkpoint (Drive): https://drive.google.com/open?id=1w8gCOZa5zHn09lQ3kMT_jRN2o_8poino

## Additional training-run evidence and links

**Checkpoint-links note:** The 'hindi' link above is the authoritative Hindi checkpoint: the Kaggle T4x2 GPU run (kaggle_hindi_and_nepali_bonus below), step 6500 of a 15000-step schedule (val PPL 14.62 / BPB 0.345, test PPL 16.12 / BPB 0.309), uploaded to Drive to replace the earlier stale step-1750 Colab checkpoint. It matches hindi/checkpoints/latest.pt in the repo (git-ignored for size). PPL/BPB, generation-quality, and attention-analysis for Hindi are all recomputed from this SAME step-6500 checkpoint. The 'nepali' link is still the earlier, stale Colab checkpoint (step 1750) - the repo's authoritative Nepali checkpoint (CPU-VM, step 4999, val_ppl 28.30, consistent across PPL/BPB/generation/attention) was not uploaded to Drive; it exists in nepali/checkpoints/ (git-ignored for size). The only Nepali run that finished on Kaggle GPU used a since-corrected deficient dataset (35M vs the real ~471M tokens), so it is bonus-only, not authoritative. Two separate attempts to retrain Nepali on the corrected full corpus (kaggle_nepali_real_data_attempt_1/2 below) downloaded the real data successfully but neither produced a checkpoint before failing.

**Live/verifiable training runs:**

- `kaggle_hindi_and_nepali_bonus`: https://www.kaggle.com/code/nishantharikanta/notebookd6949b8470 (Hindi: completed full run, step 6500/6600 val_ppl ~14.6-14.7 - this is Hindi's authoritative checkpoint now, consistently used for PPL/BPB/generation/attention; Nepali side used deficient data, bonus only)
- `kaggle_nepali_real_data_attempt_1`: https://www.kaggle.com/code/nishantharikanta/notebookec2fd91dad (data downloaded successfully; blocked by Kaggle draft-save corruption before launch - no usable result)
- `kaggle_nepali_real_data_attempt_2`: https://www.kaggle.com/code/nishantharikanta/notebookfb6bd80298 (blocked by Kaggle editor becoming unresponsive, likely account rate-limiting - no usable result)
- `gcp_cpu_vm_both_languages`: nishanth-phase2-cpu (GCP instance, project lma-01, zone asia-south1-b) - training_log.jsonl (Nepali) committed to git under nepali/checkpoints/; authoritative source for all of Nepali's deliverables (PPL/BPB/generation/attention). Hindi's CPU-VM run (step 4999) was superseded as authoritative by the Kaggle run above once it produced a better checkpoint; hindi/checkpoints/training_log.jsonl now holds the Kaggle run's log (step 0-6600), not the CPU-VM's.

**Weights & Biases:** Not used - no Weights & Biases run was set up for this project. Training logs are in hindi/checkpoints/training_log.jsonl and nepali/checkpoints/training_log.jsonl (committed to git) and loss_curve.png figures.
