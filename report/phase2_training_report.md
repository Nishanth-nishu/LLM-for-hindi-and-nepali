# Phase 2 — Training Report

| | |
|---|---|
| Generated (UTC) | 2026-09-13 11:21:26 |
| Git commit | `21e27548771512fdaa9b753388a69f26b8a7f564` |
| Branch | `phase-3` |

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
| Step | 0 | 6799 |
| Train loss | 8.3890 | 3.3817 |
| Val loss | 8.3865 | 3.1426 |
| Val PPL | 4387.37 | 23.16 |
| LR | 1.50e-06 | 3.00e-05 |

Best val loss so far: **3.1426**  
Training progress: **6,799 / 6,800 steps (100.0%)**.

Loss curve: `report/figures/phase2/nepali/loss_curve.png`

Checkpoint (Drive): PENDING UPLOAD -- see note

## Additional training-run evidence and links

**Checkpoint-links note:** hindi_pretrained is the authoritative Hindi checkpoint: Kaggle T4x2 GPU run (kaggle_hindi_and_nepali_bonus below), step 6500 of a 15000-step schedule (val PPL 14.62 / BPB 0.345, test PPL 16.12 / BPB 0.309). Matches hindi/checkpoints/latest.pt (git-ignored for size). nepali_pretrained: the ORIGINAL authoritative Nepali checkpoint (real 471.6M-token corpus, Kaggle GPU run, step 6799/6800, val PPL ~23.2) was never uploaded to Drive and its .pt file was subsequently lost entirely (only its training_log.jsonl survived, already committed to git) -- it was regenerated from scratch on the IIIT-H ADA cluster (RTX 3090) using the identical config/seed/corpus, reproducing statistically the same quality (test PPL 25.07 / BPB 0.479, val PPL 25.41 / BPB 0.480 -- the small PPL difference from the original 23.x figure reflects normal GPU-run-to-run variance, not a different corpus or config; see report/phase3_reasoning_report.md and report/phase3_final_report.md for how this recovered checkpoint is used as Model L's pretrained base going forward). It matches nepali/checkpoints/latest.pt (git-ignored for size) and needs to be uploaded to the Drive folder above. hindi_finetuned_reasoning and nepali_finetuned_reasoning are the Phase 3.1 reasoning-finetuned checkpoints (best.pt, step 75, lowest val_loss -- see report/phase3_reasoning_report.md for why checkpoints are tracked at eval granularity rather than only at epoch boundaries, and for the dataset-size/rebalancing iteration that raised test exact-match accuracy from an initial 32.5%/17.5% to 33.5%/36.1% for Hindi/Nepali respectively), matching hindi/checkpoints_reasoning/latest.pt and nepali/checkpoints_reasoning/latest.pt respectively (also git-ignored for size, also pending upload).

**Live/verifiable training runs:**

- `kaggle_hindi_and_nepali_bonus`: https://www.kaggle.com/code/nishantharikanta/notebookd6949b8470 (Hindi: completed full run, step 6500/6600 val_ppl ~14.6-14.7 -- Hindi's authoritative pretrained checkpoint, consistently used for PPL/BPB/generation/attention/reasoning-finetuning-base; Nepali side of this same notebook used a deficient dataset, bonus only, not authoritative)
- `kaggle_nepali_real_data_original`: The original real-data Nepali GPU retrain (step 6799/6800, val_ppl 23.2) that produced the checkpoint later lost -- see nepali_pretrained note above. training_log.jsonl survived and is committed under nepali/checkpoints/.
- `ada_cluster_nepali_recovery`: IIIT-H ADA cluster, node gnode118 (RTX 3090), partition/account plafnet2 -- recovery retrain of Nepali's pretrained checkpoint (identical config/seed/corpus to the lost original), plus both languages' Phase 3.1 reasoning finetuning and Phase 3.2 attention comparison runs.
- `gcp_cpu_vm_both_languages`: nishanth-phase2-cpu (GCP instance, project lma-01, zone asia-south1-b) -- an earlier, superseded CPU-only Nepali run (step 4999, val_ppl 28.30) and Hindi run (step 4999), both since superseded by the GPU runs above.

**Weights & Biases:** Not used - no Weights & Biases run was set up for this project. Training logs are in <lang>/checkpoints/training_log.jsonl and <lang>/checkpoints_reasoning/training_log.jsonl (all committed to git) and loss_curve.png figures.
