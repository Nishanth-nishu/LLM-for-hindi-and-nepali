"""Phase 3.2: pretrained-vs-finetuned attention comparison on comparative-
reasoning prompts. Reuses the Phase 2 attention toolkit
(pipeline/eval/attention_analysis.py) unchanged -- same entropy/attention-
distance/heatmap/causal-mask-check functions -- run once per checkpoint
(pretrained, finetuned) over the SAME reasoning prompts, so the two are
directly comparable layer-by-layer.

Prompts are drawn preferentially from the transitive-chain families
(C_endpoints_more/less, C_most/least), since those are the multi-hop
comparative-reasoning cases the spec calls out specifically, and each
prompt includes its gold answer appended (prompt + delimiter + answer),
so the heatmap covers the position where the model actually commits to
the answer, not just the question.

    python -m pipeline.eval.reasoning_attention_compare --lang hindi \
        --pretrained-checkpoint hindi/checkpoints/latest.pt \
        --finetuned-checkpoint hindi/checkpoints_reasoning/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
import yaml

from pipeline.eval.attention_analysis import (
    attention_entropy,
    get_attention,
    mean_attention_distance,
    plot_heatmap,
    verify_causal_mask,
)
from pipeline.model import GPTConfig, GPTLanguageModel, load_checkpoint


def _load_model(checkpoint_path: Path, device: str):
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    mcfg = GPTConfig(**ckpt["config"]["model"])
    model = GPTLanguageModel(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, mcfg, ckpt.get("step")


def _pick_examples(examples: list[dict], n: int) -> list[dict]:
    chain = [e for e in examples if e["family"].startswith("C_")]
    rest = [e for e in examples if not e["family"].startswith("C_")]
    ordered = chain + rest
    return ordered[:n]


def _analyze_checkpoint(
    model: GPTLanguageModel, mcfg: GPTConfig, sp: spm.SentencePieceProcessor,
    examples: list[dict], delimiter: str, device: str, fig_dir: Path, tag: str,
) -> dict:
    per_layer_entropy, per_layer_distance, causal_checks = [], [], []
    heatmap_paths: list[str] = []

    for si, ex in enumerate(examples):
        text = f'{ex["prompt"]} {delimiter} {ex["answer"]}'
        ids = sp.encode(text, out_type=int)[: mcfg.max_seq_len]
        if len(ids) < 4:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        attn = get_attention(model, idx)
        causal_checks.append(verify_causal_mask(model, idx, mcfg.vocab_size))

        if si == 0:
            tokens = [sp.id_to_piece(i) for i in ids]
            early_path = fig_dir / f"{tag}_heatmap_layer_early.png"
            late_path = fig_dir / f"{tag}_heatmap_layer_late.png"
            plot_heatmap(attn[0], tokens, f"{tag} · layer 0 (early)", early_path)
            plot_heatmap(attn[-1], tokens, f"{tag} · layer {len(attn) - 1} (late)", late_path)
            heatmap_paths = [str(early_path), str(late_path)]

        ent = torch.stack([attention_entropy(a) for a in attn])       # (n_layer, h)
        dist = torch.stack([mean_attention_distance(a) for a in attn])  # (n_layer, h)
        per_layer_entropy.append(ent)
        per_layer_distance.append(dist)

    entropy = torch.stack(per_layer_entropy).mean(dim=0)
    distance = torch.stack(per_layer_distance).mean(dim=0)

    return {
        "n_sentences_analyzed": len(per_layer_entropy),
        "causal_mask_verified_all_sentences": all(causal_checks) if causal_checks else None,
        "entropy_mean_per_layer": entropy.mean(dim=1).tolist(),
        "distance_mean_per_layer": distance.mean(dim=1).tolist(),
        "entropy_overall_mean": entropy.mean().item(),
        "distance_overall_mean": distance.mean().item(),
        "heatmaps": heatmap_paths,
    }


def run_comparison(
    lang: str,
    pretrained_checkpoint: str | Path,
    finetuned_checkpoint: str | Path,
    repo_root: str | Path = ".",
    config_name: str = "reasoning_finetune_config.yaml",
    split: str = "test",
    device: str = "cpu",
    n_sentences: int = 8,
) -> dict:
    root = Path(repo_root).resolve()
    with open(root / lang / "configs" / config_name, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fcfg = cfg["finetune"]
    delimiter = fcfg["delimiter"]

    sp = spm.SentencePieceProcessor(model_file=str(root / fcfg["tokenizer_model"]))
    split_file = {"train": fcfg["train_file"], "val": fcfg["val_file"], "test": fcfg["test_file"]}[split]
    examples = []
    with open(root / split_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples = _pick_examples(examples, n_sentences)

    fig_dir = root / "report" / "figures" / "phase3" / lang / "attention_pretrained_vs_finetuned"

    pre_model, pre_cfg, pre_step = _load_model(root / pretrained_checkpoint, device)
    pre_result = _analyze_checkpoint(pre_model, pre_cfg, sp, examples, delimiter, device, fig_dir, "pretrained")
    pre_result["checkpoint"] = str(pretrained_checkpoint)
    pre_result["checkpoint_step"] = pre_step
    del pre_model

    ft_model, ft_cfg, ft_step = _load_model(root / finetuned_checkpoint, device)
    ft_result = _analyze_checkpoint(ft_model, ft_cfg, sp, examples, delimiter, device, fig_dir, "finetuned")
    ft_result["checkpoint"] = str(finetuned_checkpoint)
    ft_result["checkpoint_step"] = ft_step
    del ft_model

    n_layer = pre_cfg.n_layer
    delta_entropy_per_layer = [
        ft_result["entropy_mean_per_layer"][i] - pre_result["entropy_mean_per_layer"][i] for i in range(n_layer)
    ]
    delta_distance_per_layer = [
        ft_result["distance_mean_per_layer"][i] - pre_result["distance_mean_per_layer"][i] for i in range(n_layer)
    ]

    return {
        "lang": lang, "split": split, "n_layer": n_layer, "n_head": pre_cfg.n_head,
        "prompts_analyzed": [{"family": e["family"], "prompt": e["prompt"], "answer": e["answer"]} for e in examples],
        "pretrained": pre_result,
        "finetuned": ft_result,
        "delta_entropy_per_layer_finetuned_minus_pretrained": delta_entropy_per_layer,
        "delta_distance_per_layer_finetuned_minus_pretrained": delta_distance_per_layer,
        "delta_entropy_overall": ft_result["entropy_overall_mean"] - pre_result["entropy_overall_mean"],
        "delta_distance_overall": ft_result["distance_overall_mean"] - pre_result["distance_overall_mean"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--pretrained-checkpoint", required=True)
    ap.add_argument("--finetuned-checkpoint", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config", default="reasoning_finetune_config.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-sentences", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    result = run_comparison(
        args.lang, args.pretrained_checkpoint, args.finetuned_checkpoint, root,
        args.config, args.split, args.device, args.n_sentences,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "prompts_analyzed"}, indent=2, ensure_ascii=False))

    out = Path(args.out) if args.out else root / args.lang / "data" / "stats" / "phase3_attention_pretrained_vs_finetuned.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
