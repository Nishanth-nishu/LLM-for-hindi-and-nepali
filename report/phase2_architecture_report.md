# Phase 2 — Model Architecture Report

| | |
|---|---|
| Generated (UTC) | 2026-09-04 18:20:27 |
| Git commit | `593e6e79a54d90f533b73353471ce30cd6fe237a` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## 1. Implementation

Decoder-only GPT-style Transformer, built from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only — no `nn.Transformer*`, no HuggingFace model classes, no pre-built attention block. Source: `pipeline/model/transformer.py`.

| Component | Choice | Why |
|---|---|---|
| Positional scheme | Learned absolute positional embeddings | Simple, and Phase 1 fixed a known small context length — no need for length extrapolation. Hard consequence: the model has no representation for any position >= `max_seq_len`, which is therefore a hard ceiling on sequence length, not a soft default. |
| Norm placement | Pre-norm (`x + Sublayer(LN(x))`) | Keeps an unimpeded identity path through the residual stream; post-norm routes every residual gradient back through a LayerNorm, which is harder to train at depth. |
| Attention scaling | `1/sqrt(d_head)` | Keeps dot-product variance ~O(1) regardless of `d_head`; unscaled, variance grows with `d_head` and pushes softmax into a near-one-hot, near-zero-gradient regime. |
| Causal mask | Additive upper-triangular `-inf` mask, added to scores before softmax | Implemented by hand (`CausalSelfAttention.causal_mask`), not a library flag. Verified empirically — see `tests/test_phase2_model.py::test_causal_mask_future_token_does_not_change_past_logits` and `pipeline/eval/attention_analysis.verify_causal_mask`. |
| FFN | Linear(d->4d) -> GELU -> Linear(4d->d) | Standard GPT expansion ratio. |
| Weight tying | `lm_head.weight = tok_emb.weight` | Saves `vocab_size * d_model` parameters — see `tests/test_phase2_model.py::test_weight_tying_saves_parameters`, which asserts the saving is exactly that many parameters. |

### Tensor shapes through one block (`X` of shape `(B, T, d_model)`)

```
qkv = qkv_proj(X)                          (B, T, 3*d_model)
q, k, v = split(qkv, d_model, dim=2)       each (B, T, d_model)
q, k, v -> view + transpose                each (B, n_head, T, d_head)
scores = q @ k^T / sqrt(d_head)             (B, n_head, T, T)
scores = scores + causal_mask               (B, n_head, T, T)
weights = softmax(scores, dim=-1)           (B, n_head, T, T)
out = weights @ v                           (B, n_head, T, d_head)
out -> transpose + reshape (concat heads)   (B, T, d_model)
out = out_proj(out)                         (B, T, d_model)
```

## 2. Configurations

### Model H (higher-resource, Hindi)

| Hyperparameter | Value |
|---|--:|
| `vocab_size` | 4000 |
| `max_seq_len` | 512 |
| `d_model` | 512 |
| `n_layer` | 7 |
| `n_head` | 8 |
| `d_ff` | 2048 |
| `embed_dropout` | 0.1 |
| `attn_dropout` | 0.1 |
| `resid_dropout` | 0.1 |
| `tie_weights` | True |

**Computed parameter count: 24,377,856** (embedding share 8.4%, non-embedding 22,329,856) — computed directly from the config above; matches `pipeline.model.count_parameters()` at model-build time.

### Model L (lower-resource, Nepali)

| Hyperparameter | Value |
|---|--:|
| `vocab_size` | 4000 |
| `max_seq_len` | 512 |
| `d_model` | 512 |
| `n_layer` | 7 |
| `n_head` | 8 |
| `d_ff` | 2048 |
| `embed_dropout` | 0.1 |
| `attn_dropout` | 0.1 |
| `resid_dropout` | 0.1 |
| `tie_weights` | True |

**Computed parameter count: 24,377,856** (embedding share 8.4%, non-embedding 22,329,856) — computed directly from the config above; matches `pipeline.model.count_parameters()` at model-build time.

### Depth/width trade-off

At a fixed ~25M-parameter budget, both configs use `d_model=512`, `n_layer=7`, `n_head=8` (`d_head=64`), `d_ff=2048`. Depth was favoured over extra width: doubling `d_model` roughly quadruples attention+FFN cost per layer (both scale with `d^2`) for the same parameter budget, forcing far fewer layers — and a causal-LM objective benefits from more sequential composition steps more than from a wider single step at this scale. Both models share the architecture on purpose: Phase 1 landed both tokenizers on vocab 4,000, so any H vs L difference in Phase 2 results reflects the corpora/language, not an architecture change.

### Vocabulary and embedding cost

Both tokenizers are vocab 4,000 (Phase 1). Tied embeddings: 4,000 x 512 = 2,048,000 parameters (~8.2-8.4% of the ~24-25M total). Untying would add a second such matrix, roughly doubling the embedding share — the vocabulary size was chosen in Phase 1 assuming the tied figure (see `report/phase1_tokenizer_report.md`).
