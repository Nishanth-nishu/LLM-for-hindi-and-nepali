# Phase 2 Bonus Ablation: No Positional Embeddings (Hindi / Model H)

Optional bonus deliverable per the assignment spec: retrain one of the two
language models with positional embeddings removed, run the full Phase 2
evaluation suite on it, and compare against the standard model trained
under otherwise identical conditions.

## 1. Setup

**Language chosen:** Hindi (Model H). Hindi has the most complete real-data
corpus and tokenizer artifacts verified locally, making it the more
reliable base for a controlled comparison.

**Compute:** IIIT-Hyderabad ADA cluster, node `gnode118`
(NVIDIA RTX 3090, 24 GB), SLURM partition/account `plafnet2`. PyTorch
2.5.1+cu121.

**Controlled comparison.** Two runs were trained from the *same* real
Hindi corpus (474,751,658 train tokens; see
[`phase2_corpus_token_counts.json`](../hindi/data/stats/phase2_corpus_token_counts.json))
with **identical** hyperparameters, data order (same seed), and step
budget — the *only* difference is the `use_positional_embeddings` flag in
[`pipeline/model/transformer.py`](../pipeline/model/transformer.py):

| | `model_config_ablation_baseline.yaml` | `model_config_ablation_nopos.yaml` |
|---|---|---|
| `use_positional_embeddings` | `true` | `false` |
| `d_model` / `n_layer` / `n_head` / `d_ff` | 512 / 7 / 8 / 2048 | same |
| `batch_size` × `grad_accum_steps` | 16 × 8 = 128 effective | same |
| `max_steps` | 3000 | same |
| `lr` schedule (warmup 100, cosine to `min_lr`) | 3e-4 → 3e-5 | same |
| `seed` | 20260820 | same |
| Parameters | 24,377,856 | 24,115,712 |

The parameter difference is exactly 262,144 = `max_seq_len(512) ×
d_model(512)` — precisely the size of the removed positional embedding
table, confirming nothing else in the architecture changed
(`pipeline/model/transformer.py`, `GPTLanguageModel.__init__`: `self.pos_emb
= nn.Embedding(...) if cfg.use_positional_embeddings else None`).

This step budget (3000 steps) is shorter than the main Model H run (which
went to 6500+ steps on Kaggle) — the ablation's goal is an apples-to-apples
*relative* comparison between two models trained under identical
conditions, not a new state-of-the-art checkpoint for Hindi. Both runs used
the real, full local corpus and the standard train/val/test split
(`hindi/data/splits/`).

Raw training curves: [`hindi/checkpoints_ablation_baseline_training_log.jsonl`](../hindi/checkpoints_ablation_baseline_training_log.jsonl),
[`hindi/checkpoints_ablation_nopos_training_log.jsonl`](../hindi/checkpoints_ablation_nopos_training_log.jsonl).

## 2. Results

### 2.1 Perplexity / bits-per-byte (full val + test sweep, `pipeline/eval/lm_metrics.py`)

| Split | Metric | Baseline (with pos. emb.) | No positional embeddings | Relative change |
|---|---|---:|---:|---:|
| val | perplexity | 32.73 | 36.31 | **+10.9%** |
| val | bits/byte | 0.6342 | 0.6530 | +3.0% |
| test | perplexity | 32.01 | 35.56 | **+11.1%** |
| test | bits/byte | 0.6314 | 0.6506 | +3.0% |

(Full JSON: [`phase2_lm_metrics_val_ablation_baseline.json`](../hindi/data/stats/phase2_lm_metrics_val_ablation_baseline.json),
[`_test_ablation_baseline.json`](../hindi/data/stats/phase2_lm_metrics_test_ablation_baseline.json),
[`_val_ablation_nopos.json`](../hindi/data/stats/phase2_lm_metrics_val_ablation_nopos.json),
[`_test_ablation_nopos.json`](../hindi/data/stats/phase2_lm_metrics_test_ablation_nopos.json).)

Removing positional embeddings costs the model roughly **11% higher
perplexity** on held-out Hindi text, measured over the *entire* val/test
split (~4.8M tokens each) — a consistent, sizeable degradation, not noise.
(Note: the quick in-training validation numbers printed during training —
e.g. val_ppl 22.55 at step 2999 for the baseline — only sample the first
`eval_iters=20` batches of the unshuffled val split and are systematically
lower than this full sweep; the `lm_metrics.py` numbers above, computed
over the complete split, are the ones to trust.)

### 2.2 Generation quality (`pipeline/eval/generation_eval.py`, 30 held-out prompts, 48-token continuations)

| Decoding | Metric | Baseline | No pos. emb. |
|---|---|---:|---:|
| greedy | BLEU-4 | 0.0176 | 0.0219 |
| greedy | chrF | 0.1147 | 0.1383 |
| greedy | ROUGE-L | 0.0824 | 0.0968 |
| temp=1.0 | BLEU-4 | 0.0193 | 0.0152 |
| temp=1.0 | chrF | 0.1817 | 0.1769 |
| temp=1.0 | distinct-2 | 0.943 | 0.943 |
| greedy | repetition rate | 0.708 | 0.630 |

Full breakdown across all four decoding settings (greedy, temp 0.5/1.0/1.5)
is in [`phase2_generation_eval_ablation_baseline.json`](../hindi/data/stats/phase2_generation_eval_ablation_baseline.json)
and [`phase2_generation_eval_ablation_nopos.json`](../hindi/data/stats/phase2_generation_eval_ablation_nopos.json).

Unlike perplexity, these n=30-example generation metrics are **noisy and
inconsistent** between the two models — neither model is a clear winner on
BLEU/chrF/ROUGE-L or diversity, and the direction of the (small) gap even
flips between decoding settings. This is expected: BLEU/chrF/ROUGE-L
against a single reference continuation are high-variance at this sample
size, while perplexity is measured over the full ~4.8M-token split and is
the statistically reliable signal here. We report generation metrics for
completeness (as the spec requires) but the headline finding is the PPL/BPB
gap in §2.1, corroborated by the attention analysis below.

### 2.3 Attention analysis (`pipeline/eval/attention_analysis.py`, 5 held-out sentences)

| | Baseline | No pos. emb. |
|---|---:|---:|
| Causal mask verified (all sentences) | ✓ true | ✓ true |
| Mean entropy per layer (nats), layers 1→7 | 4.22, 4.54, 3.69, 2.61, 1.79, 3.07, 3.36 | 4.89, 4.83, 4.65, 4.31, 2.43, 2.30, 3.31 |
| **Overall mean entropy** | **3.322** | **3.818** |
| Mean attention distance per layer, layers 1→7 | 45.5, 68.3, 47.9, 7.1, 5.5, 33.6, 98.5 | 121.6, 121.9, 93.6, 94.5, 5.9, 13.2, 79.4 |
| **Overall mean attention distance** | **43.78 tokens** | **75.74 tokens** |

Full per-head breakdown: [`phase2_attention_analysis_ablation_baseline.json`](../hindi/data/stats/phase2_attention_analysis_ablation_baseline.json),
[`phase2_attention_analysis_ablation_nopos.json`](../hindi/data/stats/phase2_attention_analysis_ablation_nopos.json).
Heatmaps: [baseline early layer](../report/figures/phase2/hindi__model_config_ablation_baseline/heatmap_layer_early.png) /
[late layer](../report/figures/phase2/hindi__model_config_ablation_baseline/heatmap_layer_late.png),
[no-pos early layer](../report/figures/phase2/hindi__model_config_ablation_nopos/heatmap_layer_early.png) /
[late layer](../report/figures/phase2/hindi__model_config_ablation_nopos/heatmap_layer_late.png).

The causal mask is empirically verified intact for both models (perturbing
the last token never changes logits at earlier positions) — the ablation
only removes the positional *embedding*, not the causal masking that
already prevents the model from seeing the future.

The no-positional-embeddings model's attention is measurably **more
diffuse and looks further back**: overall mean entropy rises from 3.32 to
3.82 nats, and mean attention distance nearly doubles, from 43.8 to 75.7
tokens. With positional embeddings, several heads (e.g. layers 4–5) learn
sharp, short-range, near-diagonal attention patterns — the kind of "look at
the last few tokens" heads visible as bright near-diagonal bands in the
baseline heatmaps. Without positional embeddings, heads have no cheap way
to identify "nearby" versus "far" tokens by position, so attention spreads
out and relies more on content-based similarity alone, which is a strictly
weaker signal for local syntactic structure (e.g. adjacent-word agreement,
local word order) in Hindi.

## 3. What breaks without position information

**Theory, verified empirically in `tests/test_phase2_model.py`
(`test_no_positional_embeddings_breaks_order_sensitivity`):** for a
**single** causal self-attention layer with no positional embeddings, Q/K/V
are pure per-token linear maps of the token embedding — no position is
mixed in anywhere. The output at a fixed query position is therefore a
weighted sum over the *set* of preceding (key, value) pairs, and is
**provably invariant to permuting the tokens that precede the query**: two
prefixes that are anagrams of each other produce bit-identical logits at
the final position. The test confirms this holds exactly (`torch.allclose`,
atol 1e-5) for a 1-layer no-pos model, and confirms it does **not** hold
once positional embeddings are restored (permuting the prefix measurably
changes the logits).

**Why the real 7-layer model still works, just worse, rather than failing
completely.** The single-layer permutation-invariance proof does *not*
extend to a stack of causal layers: causal masking itself gives position
`i` a *different visible history* under a permutation of the tokens before
it — "3 tokens precede me, in this order" is implicit positional
information, leaked purely through the mask, even with zero explicit
position embedding anywhere. Re-running the same permutation test at
`n_layer=2` fails the invariance assertion for exactly this reason (noted
in the test's docstring), which is why the ablated model in this report —
7 layers of causal self-attention with no positional embedding — still
learns a usable, if noticeably worse, language model rather than
collapsing to bag-of-words behavior: each layer's causal mask re-injects a
weak, implicit ordering signal that later layers can exploit, just far
less directly and reliably than an explicit learned positional embedding.
This matches what the results show: perplexity is consistently ~11% worse
and attention is measurably more diffuse and longer-range (§2.3) — a real,
consistent degradation from losing the *explicit* positional signal, not a
catastrophic failure, because causal masking supplies a weaker implicit
substitute.

## 4. Reproduction

```bash
# Baseline arm (positional embeddings on)
python -m pipeline.train.train --lang hindi --config model_config_ablation_baseline.yaml
python -m pipeline.eval.lm_metrics --lang hindi --config model_config_ablation_baseline.yaml \
    --checkpoint hindi/checkpoints_ablation_baseline/latest.pt --split test
python -m pipeline.eval.generation_eval --lang hindi --config model_config_ablation_baseline.yaml \
    --checkpoint hindi/checkpoints_ablation_baseline/latest.pt
python -m pipeline.eval.attention_analysis --lang hindi --config model_config_ablation_baseline.yaml \
    --checkpoint hindi/checkpoints_ablation_baseline/latest.pt

# Ablated arm (positional embeddings off)
python -m pipeline.train.train --lang hindi --config model_config_ablation_nopos.yaml
python -m pipeline.eval.lm_metrics --lang hindi --config model_config_ablation_nopos.yaml \
    --checkpoint hindi/checkpoints_ablation_nopos/latest.pt --split test
python -m pipeline.eval.generation_eval --lang hindi --config model_config_ablation_nopos.yaml \
    --checkpoint hindi/checkpoints_ablation_nopos/latest.pt
python -m pipeline.eval.attention_analysis --lang hindi --config model_config_ablation_nopos.yaml \
    --checkpoint hindi/checkpoints_ablation_nopos/latest.pt

# Unit-level proof of the permutation-invariance property
python -m pytest tests/test_phase2_model.py -k no_positional -v
```
