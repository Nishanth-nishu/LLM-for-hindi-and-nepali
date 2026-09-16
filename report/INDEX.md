# Report Index — Phase 1, 2, 3

One place to find every per-phase report in this repo. Each phase's reports
are grouped below in the order they're usually read. The consolidated,
10-page submission report (LaTeX) is at
[`final_report.tex`](final_report.tex) — see the note at the bottom.

---

## Phase 1 — Data Collection, Cleaning, Tokenization

| Report | Covers |
|---|---|
| [phase1_summary.md](phase1_summary.md) | Executive summary — start here for Phase 1 |
| [phase1_data_collection_report.md](phase1_data_collection_report.md) | Data Collection Report |
| [phase1_cleaning_report.md](phase1_cleaning_report.md) | Data Cleaning Report |
| [phase1_manual_data_report.md](phase1_manual_data_report.md) | Manual Data Report |
| [phase1_provenance_report.md](phase1_provenance_report.md) | Data Provenance Report |
| [phase1_tokenizer_report.md](phase1_tokenizer_report.md) | Tokenizer Report |
| [phase1_validation_report.md](phase1_validation_report.md) | Validation Report |
| [phase1_final_statistics.md](phase1_final_statistics.md) | Final Corpus Statistics |
| [phase1_reproducibility.md](phase1_reproducibility.md) | Reproducibility Report |

## Phase 2 — Model Architecture, Pretraining, Evaluation

| Report | Covers |
|---|---|
| [phase2_architecture_report.md](phase2_architecture_report.md) | Model Architecture Report |
| [phase2_training_report.md](phase2_training_report.md) | Training Report |
| [phase2_lm_evaluation_report.md](phase2_lm_evaluation_report.md) | Language Modeling Evaluation |
| [phase2_generation_report.md](phase2_generation_report.md) | Generation Quality and Diversity |
| [phase2_attention_report.md](phase2_attention_report.md) | Attention Analysis |
| [phase2_resource_comparison.md](phase2_resource_comparison.md) | Resource-Level Comparison (Model H vs Model L) |
| [phase2_ablation_report.md](phase2_ablation_report.md) | Bonus Ablation: No Positional Embeddings |
| [phase2_checkpoint_links.json](phase2_checkpoint_links.json) | Google Drive links for every checkpoint/training run |

## Phase 3 — Reasoning Finetuning and Analysis

| Report | Covers |
|---|---|
| [phase3_reasoning_report.md](phase3_reasoning_report.md) | 3.1 — Reasoning Finetuning (dataset, protocol, results) |
| [phase3_attention_report.md](phase3_attention_report.md) | 3.2 — Attention Analysis: Pretrained vs. Finetuned |
| [phase3_final_report.md](phase3_final_report.md) | 3.3 — Final Report: Model H vs. Model L, project-wide |

## Supporting artifacts

| File | Covers |
|---|---|
| [verification.json](verification.json) | Machine-checked verification results referenced across reports |
| [figures/](figures/) | All loss curves, attention heatmaps, corpus/tokenizer figures used above |

---

**Submission report:** [`final_report.tex`](final_report.tex) is the single,
consolidated, ≤10-page LaTeX report covering all three phases that actually
gets submitted per the assignment brief. It is a structural skeleton with
figure/table placeholders and pointers back into the reports above — the
analysis and reflections inside it must still be written by you before
compiling. This index is just for navigating the detailed per-phase source
reports it draws from.
