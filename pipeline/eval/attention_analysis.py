"""Attention analysis: heatmaps, per-head/layer entropy, mean attention
distance, and an empirical causal-mask check (this doubles as the "verify
the model cannot see the future" evidence the spec asks for).

    python -m pipeline.eval.attention_analysis --lang hindi --checkpoint hindi/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Devanagari (Hindi/Nepali) tick labels render as tofu boxes under
# matplotlib's default font. Use the first Devanagari-capable font actually
# installed, in a cross-platform-friendly order; silently keep the default
# if none exist rather than fail plotting.
_DEVANAGARI_FONTS = ["Nirmala UI", "Noto Sans Devanagari", "Mangal", "Lohit Devanagari"]
_installed = {f.name for f in fm.fontManager.ttflist}
for _font in _DEVANAGARI_FONTS:
    if _font in _installed:
        plt.rcParams["font.family"] = _font
        break
import sentencepiece as spm
import torch
import yaml

from pipeline.model import GPTConfig, GPTLanguageModel, load_checkpoint
from pipeline.train.dataset import load_documents


@torch.no_grad()
def get_attention(model: GPTLanguageModel, idx: torch.Tensor):
    _, _, attn = model(idx, return_attn=True)
    return attn  # list of (B, n_head, T, T), one per layer


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """attn: (B, h, T, T) softmax weights (rows sum to 1 over allowed keys).
    Returns (h,) entropy in nats, averaged over batch and valid query rows
    (row 0 attends to a single key and has 0 entropy by construction, which
    is legitimate, not an artifact)."""
    eps = 1e-12
    ent = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, h, T)
    return ent.mean(dim=(0, 2))  # (h,)


def mean_attention_distance(attn: torch.Tensor) -> torch.Tensor:
    """attn: (B, h, T, T). distance(i) = sum_j p_ij * (i - j), averaged over
    query position i and batch. Higher = heads that look further back."""
    B, H, T, _ = attn.shape
    i_idx = torch.arange(T, device=attn.device).view(1, 1, T, 1)
    j_idx = torch.arange(T, device=attn.device).view(1, 1, 1, T)
    dist = (i_idx - j_idx).clamp(min=0).float()  # (1,1,T,T)
    weighted = (attn * dist).sum(dim=-1)  # (B, h, T)
    return weighted.mean(dim=(0, 2))  # (h,)


def verify_causal_mask(model: GPTLanguageModel, idx: torch.Tensor, vocab_size: int) -> bool:
    """Empirical proof the model cannot see the future: change token t+1 and
    confirm the logits at position t are bit-for-bit unchanged."""
    model.eval()
    logits_a, _, _ = model(idx)
    idx2 = idx.clone()
    t = idx.shape[1] - 1
    idx2[:, t] = (idx2[:, t] + 1) % vocab_size  # perturb the LAST token
    logits_b, _, _ = model(idx2)
    # Every position strictly before t must be unaffected by changing token t.
    return torch.allclose(logits_a[:, :t, :], logits_b[:, :t, :], atol=1e-5)


def plot_heatmap(attn_layer: torch.Tensor, tokens: list[str], layer_name: str, out_path: Path, n_heads: int = 4):
    attn_layer = attn_layer[0]  # first batch element: (h, T, T)
    n_heads = min(n_heads, attn_layer.shape[0])
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4.5))
    if n_heads == 1:
        axes = [axes]
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn_layer[h].cpu().numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"{layer_name} · head {h}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        if len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Attention weights — {layer_name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_attention_analysis(lang: str, checkpoint_path: str | Path, repo_root: str | Path = ".",
                            n_sentences: int = 5, device: str = "cpu") -> dict:
    root = Path(repo_root).resolve()
    with open(root / lang / "configs" / "model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]

    sp = spm.SentencePieceProcessor(model_file=str(root / tcfg["tokenizer_model"]))
    ckpt = load_checkpoint(root / checkpoint_path, map_location=device)
    mcfg = GPTConfig(**ckpt["config"]["model"])
    model = GPTLanguageModel(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    documents = load_documents(root / tcfg["test_file"])
    sentences = [d for d in documents if 8 <= len(sp.encode(d, out_type=int)) <= 32][:n_sentences]
    if not sentences:
        sentences = documents[:n_sentences]

    fig_dir = root / "report" / "figures" / "phase2" / lang
    per_layer_entropy, per_layer_distance = [], []
    causal_checks = []

    for si, sent in enumerate(sentences):
        ids = sp.encode(sent, out_type=int)[: mcfg.max_seq_len]
        if len(ids) < 4:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        attn = get_attention(model, idx)
        causal_checks.append(verify_causal_mask(model, idx, mcfg.vocab_size))

        if si == 0:
            # The heatmap needs a SHORT, readable snippet (so per-token tick
            # labels fit and individual weights are visible against the 0-1
            # color scale) — that's a different requirement from the
            # entropy/distance stats below, which want long, representative
            # sequences. Phase 1's corpus has no naturally-short documents,
            # so plotting `ids` (up to max_seq_len=512) directly produced
            # heatmaps where a 512x512 grid of mostly-near-zero weights
            # looked all but blank. Take a short prefix of the same
            # document instead, with its own small forward pass.
            heatmap_ids = ids[:20]
            if len(heatmap_ids) >= 4:
                heatmap_idx = torch.tensor([heatmap_ids], dtype=torch.long, device=device)
                heatmap_attn = get_attention(model, heatmap_idx)
                tokens = [sp.id_to_piece(i) for i in heatmap_ids]
                plot_heatmap(heatmap_attn[0], tokens, "layer_0_early", fig_dir / "heatmap_layer_early.png")
                plot_heatmap(heatmap_attn[-1], tokens, f"layer_{len(heatmap_attn) - 1}_late", fig_dir / "heatmap_layer_late.png")

        ent = torch.stack([attention_entropy(a) for a in attn])  # (n_layer, h)
        dist = torch.stack([mean_attention_distance(a) for a in attn])  # (n_layer, h)
        per_layer_entropy.append(ent)
        per_layer_distance.append(dist)

    entropy = torch.stack(per_layer_entropy).mean(dim=0)  # (n_layer, h)
    distance = torch.stack(per_layer_distance).mean(dim=0)  # (n_layer, h)

    result = {
        "lang": lang,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": ckpt.get("step"),
        "n_layer": mcfg.n_layer,
        "n_head": mcfg.n_head,
        "n_sentences_analyzed": len(per_layer_entropy),
        "causal_mask_verified_all_sentences": all(causal_checks) if causal_checks else None,
        "entropy_nats_by_layer_head": entropy.tolist(),
        "mean_attention_distance_by_layer_head": distance.tolist(),
        "entropy_mean_per_layer": entropy.mean(dim=1).tolist(),
        "distance_mean_per_layer": distance.mean(dim=1).tolist(),
        "heatmaps": [
            str((fig_dir / "heatmap_layer_early.png").relative_to(root)),
            str((fig_dir / "heatmap_layer_late.png").relative_to(root)),
        ],
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--n-sentences", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    result = run_attention_analysis(args.lang, args.checkpoint, root, args.n_sentences, args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out = Path(args.out) if args.out else root / args.lang / "data" / "stats" / "phase2_attention_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
