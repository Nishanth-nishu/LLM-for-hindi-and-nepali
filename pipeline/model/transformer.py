"""
Decoder-only (GPT-style) Transformer, implemented from primitive layers only
(nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout). No nn.Transformer*,
no HuggingFace model classes, no pre-built attention block.

Design choices (Phase 2 spec asks these to be stated explicitly):

- Positional scheme: LEARNED ABSOLUTE positional embeddings (an nn.Embedding
  of shape (max_seq_len, d_model), added to the token embedding). Chosen over
  sinusoidal/RoPE for simplicity and because Phase 1 fixed a small, known
  context length (see GPTConfig.max_seq_len) — the corpus does not need
  length extrapolation. Consequence: the model has no representation for any
  position >= max_seq_len, so max_seq_len is a hard ceiling on sequence
  length at both train and inference time, not just a training convenience.

- Normalization placement: PRE-NORM (LayerNorm applied to the sublayer's
  input, before attention/FFN, with the residual added around the
  un-normalized branch). Post-norm (LayerNorm after the residual add) is the
  original Transformer design but is known to be harder to train at depth —
  gradients must pass back through every LayerNorm on the residual path, and
  with post-norm that path is not identity. Pre-norm keeps an unimpeded
  identity path through the residual stream, which is why it is standard in
  GPT-2 onward.

- Weight tying: the output projection (lm_head) shares its weight matrix
  with the token embedding (tok_emb.weight). This removes a second
  vocab_size x d_model matrix, saving exactly vocab_size * d_model
  parameters. Phase 1's tokenizer vocabulary (4,000) was selected assuming
  this tied embedding cost — see report/phase1_tokenizer_report.md and the
  Phase 1->2 handoff, section 5 ("Embedding cost").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 512
    d_model: int = 512
    n_layer: int = 7
    n_head: int = 8
    d_ff: int = 2048
    embed_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    tie_weights: bool = True
    use_positional_embeddings: bool = True
    """Ablation switch (bonus deliverable). False removes the learned
    positional embedding table entirely -- the model then sees only
    token identity at every position, with position never injected
    anywhere else in the forward pass. Self-attention itself is
    permutation-equivariant (Q/K/V projections, the softmax(QK^T/sqrt(d))
    weighting, and the output projection all act identically regardless
    of how the T token vectors are ordered), and the position-wise FFN
    is applied per-position with no cross-position mixing -- so with this
    off, the only sequence-order signal reaching the model at all is the
    causal mask's admissible-key-set (position t can attend to <= t
    keys), which shrinks monotonically with t but does not distinguish
    *which* earlier position a key came from. See report/phase2_ablation_report.md."""

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles.

    Shapes, for input X of shape (B, T, d_model):
      qkv_proj(X)                    -> (B, T, 3*d_model)
      split into q, k, v             -> each (B, T, d_model)
      view + transpose (the "reshape into heads" step)
                                      -> each (B, n_head, T, d_head), d_head = d_model / n_head
      scores = q @ k^T / sqrt(d_head)-> (B, n_head, T, T)   [query position x key position]
      scores = scores + causal_mask  -> (B, n_head, T, T)   (mask broadcasts over B, n_head)
      weights = softmax(scores, -1)  -> (B, n_head, T, T)
      out = weights @ v              -> (B, n_head, T, d_head)
      transpose + reshape (concat heads)
                                      -> (B, T, d_model)
      out_proj(out)                  -> (B, T, d_model)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model

        # One linear layer producing Q, K, V together (equivalent to three
        # separate W_Q, W_K, W_V projections stacked column-wise).
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)  # W_O

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        # Additive causal mask: 0 where position j <= i (allowed), -inf where
        # j > i (future, forbidden). Built once for the maximum sequence
        # length and sliced down to the actual T at forward time.
        mask = torch.triu(
            torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, D = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=2)  # each (B, T, d_model)

        # (B, T, d_model) -> (B, T, n_head, d_head) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product attention. The 1/sqrt(d_head) scale keeps the
        # variance of the dot products ~O(1) regardless of d_head: q and k
        # entries are ~unit variance, so an unscaled dot product over d_head
        # terms has variance ~d_head, which pushes softmax into a
        # near-one-hot, near-zero-gradient regime for large d_head. Dividing
        # by sqrt(d_head) cancels that growth.
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, h, T, T)
        att_scores = att_scores + self.causal_mask[:T, :T]

        att_weights = torch.softmax(att_scores, dim=-1)  # (B, h, T, T)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v  # (B, h, T, d_head)

        # Concatenate heads: (B, h, T, d_head) -> (B, T, h, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out, (att_weights if return_weights else None)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model).

    Applied identically (same weights) at every sequence position, hence
    "position-wise" — there is no mixing across the T dimension here, all
    cross-position mixing happens in attention.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """One pre-norm Transformer decoder block:

        x = x + Attn(LN1(x))
        x = x + FFN(LN2(x))

    Both sublayers are wrapped in a residual connection around the
    *un-normalized* input, which is what makes this pre-norm.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        a, w = self.attn(self.ln1(x), return_weights=return_weights)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, w


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Ablation: when disabled, no positional embedding table is even
        # allocated (not just zeroed/unused) -- the parameter-count
        # reported for this model is honestly smaller, not padded with
        # dead weights.
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.use_positional_embeddings else None
        self.embed_dropout = nn.Dropout(cfg.embed_dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Weight tying: lm_head.weight IS tok_emb.weight (same Parameter
            # object, not a copy) — this removes a second vocab_size x
            # d_model matrix from the parameter count.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """
        idx:     (B, T) token ids
        targets: (B, T) token ids shifted by one position, or None
        returns: logits (B, T, vocab_size), loss (scalar or None),
                 attn_weights (list of (B, n_head, T, T), one per layer, or None)
        """
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(idx)  # (B, T, d_model)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.embed_dropout(x)

        attn_weights = [] if return_attn else None
        for block in self.blocks:
            x, w = block(x, return_weights=return_attn)
            if return_attn:
                attn_weights.append(w)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return logits, loss, attn_weights

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx: (B, T0) prompt token ids.

        temperature == 0.0 or greedy=True -> argmax decoding.
        Otherwise: logits / temperature -> softmax -> multinomial sample.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits, _, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size) — next-token logits
            if greedy or temperature == 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tok_emb = model.tok_emb.weight.numel() if hasattr(model, "tok_emb") else 0
    return {
        "total": total,
        "trainable": trainable,
        "embedding_params": tok_emb,
        "embedding_share": tok_emb / total if total else 0.0,
    }
