# Phase 3.3 — Final Report: Model H (Hindi) vs. Model L (Nepali)

This report consolidates the project: two independent, from-scratch,
decoder-only Transformer language models — Model H (Hindi, the
higher-resource of the two languages used in this project) and Model L
(Nepali, the lower-resource tier) — carried through data collection
(Phase 1), pretraining and evaluation (Phase 2), and reasoning finetuning
plus attention analysis (Phase 3). Full detail for each stage is in the
per-phase reports linked throughout; this report pulls the key numbers
into one place and answers the four questions the assignment asks for.

## 1. The master comparison table

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| **Architecture** | 7-layer, 8-head, d_model 512, 24.4M params | identical |
| **Vocabulary** | 4,000 (SentencePiece Unigram, byte-fallback) | identical size, separate tokenizer |
| **Tokenizer fertility (tokens/word, held-out test)** | **1.6522** | **1.8457** |
| **Real pretraining corpus (train tokens)** | 474,751,658 | 471,622,290 |
| **Pretraining schedule (steps run / planned)** | 6,500 / 15,000 (43%) | 6,799 / 6,800 (~100%) |
| **Test perplexity / BPB (pretrained)** | **16.12 / 0.309** | **25.07 / 0.479** |
| **Generation quality, greedy (BLEU-4 / chrF / ROUGE-L)** | **0.032 / 0.159 / 0.119** | 0.010 / 0.099 / 0.051 |
| **Reasoning accuracy, pretrained (zero-shot, exact match)** | 0.0% | 0.0% |
| **Reasoning accuracy, finetuned (best val_loss epoch)** | **32.5%** | **17.5%** |
| **Attention entropy shift after reasoning finetuning** | −0.128 nats (sharper) | −0.163 nats (sharper) |
| **Attention distance shift after reasoning finetuning** | −0.47 tokens (more local) | +0.31 tokens (less local) |

Sources: `report/phase1_tokenizer_report.md` (fertility),
`<lang>/data/stats/phase2_corpus_token_counts.json` (real token counts,
measured directly from the cached token arrays, not the Phase 1 handoff's
estimated figures — see `report/phase2_lm_evaluation_report.md` for that
discrepancy), `<lang>/data/stats/phase2_lm_metrics_test.json` /
`phase2_generation_eval.json` (Phase 2), `report/phase3_reasoning_report.md`
/ `phase3_attention_report.md` (Phase 3, this consolidation's main new
material).

## 2. How did data scale and quality differ between Model H and Model L?

**Scale was, in the end, almost identical**: 474.75M vs. 471.62M real
train tokens — a 0.7% difference, not the kind of gap that would explain
a 9-point PPL gap or a 15-point (Hindi test PPL is 36% lower) reasoning
gap on its own. This is worth stating plainly because it was *not* always
true during the project: an earlier discrepancy check found the Phase 1
handoff's claimed "final" corpus sizes (559.9M Hindi / 501.2M Nepali) were
both overstated relative to what actually made it into the trained
splits, and Nepali's usable pretraining checkpoint was lost once already
mid-project and had to be regenerated from the same real corpus on the
IIIT-H ADA cluster (see `report/phase3_reasoning_report.md`'s finetuning
protocol section and this project's git history) — the number above is
the corpus actually consumed by the checkpoints these results are
measured against, not an aspirational figure.

Where the two languages *do* differ measurably is **tokenizer
efficiency**, not raw token count (see §4).

## 3. How do language-modeling and reasoning results compare across the two resource tiers?

Every evaluation axis in this project — intrinsic LM quality (PPL/BPB),
generation quality (BLEU-4/chrF/ROUGE-L), and reasoning accuracy — points
the same direction: **Model H outperforms Model L by a wide, consistent
margin**, despite starting from near-identical corpus sizes:

- LM quality: Hindi's test perplexity (16.12) is **36% lower** than
  Nepali's (25.07); bits-per-byte (the tokenizer-agnostic metric) shows
  the same gap (0.309 vs 0.479).
- Generation quality: Hindi's greedy BLEU-4 is **3.2×** Nepali's
  (0.032 vs 0.010); chrF and ROUGE-L show the same ~1.6–2.3× gap.
- Reasoning: after identical finetuning protocols (same hyperparameters,
  same epoch count, same early-stopping criterion), Hindi reaches
  **32.5%** exact-match accuracy on the synthetic reasoning test set vs.
  Nepali's **17.5%** — roughly **1.9×**.

The reasoning gap is the widest, proportionally, of the three — consistent
with reasoning finetuning being the task most sensitive to the *quality*
of the base language model's representations (finetuning only ran 744
steps on 3000 examples; it cannot compensate for a weaker starting point
the way another 6000+ steps of pretraining might).

## 4. What tokenizer / corpus factors most affected the lower-resource model?

**Tokenizer fertility is the clearest, most directly measured factor.**
Nepali's SentencePiece Unigram tokenizer needs **1.8457 tokens per word**
on held-out text, against Hindi's **1.6522** — Nepali requires **~12%
more tokens to encode the same amount of text**. Both tokenizers were
built with the same vocabulary size (4,000, chosen for both languages by
the same scaling-law-informed sweep in Phase 1, not tuned differently per
language) and the same algorithm, so this gap reflects the underlying
language/script/corpus, not a tokenizer-configuration choice that favored
Hindi.

The practical consequence: **the two corpora being nearly equal in *token*
count does not mean they were equal in *text* content.** At a fixed
per-step token budget (same `max_seq_len`, `batch_size`, `grad_accum_steps`
for both languages' pretraining configs), Nepali's higher fertility means
each training step covers measurably less actual Nepali text than the
equivalent Hindi step covers Hindi text — a real, quantifiable resource
disadvantage hiding inside what looked like an even token-count race.

A second, harder-to-fully-isolate factor is that **Model L's checkpoint
had run 6,799 of its planned 6,800 steps (essentially complete), while
Model H's had run only 6,500 of a *3× longer* planned 15,000-step
schedule** — i.e. Hindi's reported checkpoint is an early snapshot of a
longer plan, not the end of a fully-converged run either. That Hindi still
outperforms Nepali substantially despite being the less-complete run (by
schedule fraction) argues that raw step count is not the dominant factor
here — tokenizer fertility and whatever it reflects about underlying
corpus/script complexity are doing more of the work.

## 5. What evidence explains the observed differences?

Putting §§2–4 together, the evidence chain is:

1. **Corpus token counts are nearly equal** (474.75M vs. 471.62M) —
   ruling out "Model L just saw less data" as the primary explanation.
2. **Tokenizer fertility is measurably worse for Nepali** (1.8457 vs.
   1.6522 tokens/word, Phase 1, held-out test split) — meaning the
   *effective* text coverage per training step was lower for Nepali even
   at matched token counts.
3. **Every downstream metric degrades in the same direction and by a
   similar relative magnitude** — PPL/BPB (+36%/+55%), generation quality
   (BLEU-4 –68%, chrF –38%, ROUGE-L –57%), and reasoning accuracy (–46%,
   32.5%→17.5%) — which is what you would expect if a single upstream
   factor (effective per-token text coverage, driven by fertility) were
   propagating through the whole pipeline, rather than each metric having
   an unrelated, independent cause.
4. **The reasoning-finetuning attention analysis adds a mechanistic
   detail** (`report/phase3_attention_report.md`): finetuning sharpens
   attention (lower entropy) in *both* models similarly, but Hindi's
   attention becomes more local while Nepali's becomes slightly less
   local. A plausible reading is that Hindi's stronger base
   representations let finetuning sharpen attention onto the *correct*
   nearby evidence (the specific number or entity just mentioned), while
   Nepali's weaker base representations let finetuning sharpen attention
   without it converging on the right evidence as reliably — consistent
   with Nepali's much lower reasoning accuracy despite a similar-sized
   entropy drop.
5. **The specific reasoning failure analysis** (`report/phase3_reasoning_report.md`
   §3.2–3.3) shows *both* languages struggle far more with numeric-
   magnitude comparison than with pure relational chaining, but Model L's
   struggle is broader and deeper — its finetuned checkpoint fails
   entirely (0%) on both `B_most` and `B_least`, while Model H's partially
   succeeds on both (13.3%, 20.4%) — the same qualitative failure mode
   the weaker base model, appearing more severely.

None of this rules out other contributing factors this project did not
directly measure (script complexity, morphological richness, or the
specific web-text sources each corpus drew from in Phase 1) — but
tokenizer fertility is the one factor with a direct, held-out-measured
number showing a real, non-trivial gap in the direction that matches
every downstream result.

## 6. Report index

| Phase | Report | Covers |
|---|---|---|
| 1 | `report/phase1_*.md` | data collection, cleaning, tokenizer training, corpus statistics |
| 2 | `report/phase2_architecture_report.md` | from-scratch Transformer design |
| 2 | `report/phase2_training_report.md` | pretraining logs, loss curves |
| 2 | `report/phase2_lm_evaluation_report.md` | PPL/BPB, real-vs-claimed corpus discrepancy |
| 2 | `report/phase2_generation_report.md` | BLEU-4/chrF/ROUGE-L, diversity diagnostics |
| 2 | `report/phase2_attention_report.md` | pretrained attention heatmaps/entropy/distance |
| 2 | `report/phase2_ablation_report.md` | bonus: no-positional-embeddings ablation (Hindi) |
| 2 | `report/phase2_resource_comparison.md` | GCP/Kaggle/ADA compute path comparison |
| 3 | `report/phase3_reasoning_report.md` | synthetic dataset, finetuning protocol, pretrained-vs-finetuned accuracy |
| 3 | `report/phase3_attention_report.md` | pretrained-vs-finetuned attention comparison |
| 3 | `report/phase3_final_report.md` | this report |

## 7. Reproducibility

See the top-level `README.md` for the full reproduction sequence (data →
tokenizer → pretraining → evaluation → reasoning finetuning → reasoning
evaluation → attention comparison) and Google Drive links for every
checkpoint (pretrained H/L, finetuned H/L) referenced in this report.
