"""Correctness checks for the from-scratch Phase 2 Transformer:
causal masking, parameter budget, and a training-loop smoke test
(forward -> backward -> optimizer step -> checkpoint round trip),
all on random token ids so it needs no corpus or tokenizer.

    python -m pytest tests/test_phase2_model.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.model import GPTConfig, GPTLanguageModel, count_parameters, save_checkpoint, load_checkpoint


def make_model(vocab_size=100, max_seq_len=32, d_model=64, n_layer=2, n_head=4, d_ff=256):
    cfg = GPTConfig(
        vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model,
        n_layer=n_layer, n_head=n_head, d_ff=d_ff,
        embed_dropout=0.0, attn_dropout=0.0, resid_dropout=0.0,
    )
    return GPTLanguageModel(cfg), cfg


def test_forward_shapes():
    model, cfg = make_model()
    B, T = 3, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, attn = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None
    assert attn is None

    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, _ = model(idx, targets=targets)
    assert loss.dim() == 0
    assert loss.item() > 0

    _, _, attn = model(idx, return_attn=True)
    assert len(attn) == cfg.n_layer
    assert attn[0].shape == (B, cfg.n_head, T, T)


def test_attention_rows_sum_to_one():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    _, _, attn = model(idx, return_attn=True)
    row_sums = attn[0].sum(dim=-1)  # (B, h, T)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_causal_mask_future_token_does_not_change_past_logits():
    """The spec's own verification: changing token t+1 must not change the
    logits at position t. This is the empirical proof the mask works."""
    model, cfg = make_model()
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 12))

    with torch.no_grad():
        logits_a, _, _ = model(idx)
        idx_perturbed = idx.clone()
        idx_perturbed[:, -1] = (idx_perturbed[:, -1] + 1) % cfg.vocab_size
        logits_b, _, _ = model(idx_perturbed)

    t = idx.shape[1] - 1
    assert torch.allclose(logits_a[:, :t, :], logits_b[:, :t, :], atol=1e-5), (
        "changing the last token changed logits at an earlier position — causal mask is leaking"
    )
    # Sanity: the position that CAN see the perturbed token should usually change.
    assert not torch.allclose(logits_a[:, t, :], logits_b[:, t, :], atol=1e-5)


def test_weight_tying_saves_parameters():
    tied, _ = make_model()
    cfg_untied = GPTConfig(vocab_size=100, max_seq_len=32, d_model=64, n_layer=2, n_head=4, d_ff=256, tie_weights=False)
    untied = GPTLanguageModel(cfg_untied)

    tied_count = count_parameters(tied)["total"]
    untied_count = count_parameters(untied)["total"]
    assert untied_count - tied_count == 100 * 64  # vocab_size * d_model


def test_param_budget_near_25m():
    cfg = GPTConfig(vocab_size=4000, max_seq_len=512, d_model=512, n_layer=7, n_head=8, d_ff=2048)
    model = GPTLanguageModel(cfg)
    total = count_parameters(model)["total"]
    assert 20_000_000 <= total <= 30_000_000, f"expected ~25M params, got {total:,}"


def test_backward_and_optimizer_step_reduce_loss():
    torch.manual_seed(0)
    model, cfg = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))

    _, loss0, _ = model(idx, targets=targets)
    for _ in range(20):
        opt.zero_grad()
        _, loss, _ = model(idx, targets=targets)
        loss.backward()
        opt.step()
    _, loss1, _ = model(idx, targets=targets)
    assert loss1.item() < loss0.item()


def test_checkpoint_round_trip(tmp_path):
    model, cfg = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, opt, None, step=42, config={"model": cfg.to_dict()})

    loaded = load_checkpoint(ckpt_path)
    assert loaded["step"] == 42
    restored = GPTLanguageModel(GPTConfig(**loaded["config"]["model"]))
    restored.load_state_dict(loaded["model_state_dict"])
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        a, _, _ = model(idx)
        b, _, _ = restored(idx)
    assert torch.allclose(a, b)


def test_generate_respects_max_seq_len():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len - 2))
    out = model.generate(idx, max_new_tokens=5, greedy=True)
    assert out.shape[1] == idx.shape[1] + 5
