"""Intrinsic LM metrics: cross-entropy, perplexity, bits-per-byte.

    python -m pipeline.eval.lm_metrics --lang hindi --checkpoint hindi/checkpoints/latest.pt --split test

BPB normalizes by UTF-8 bytes of the reference text rather than by tokens,
so Hindi and Nepali stay comparable across two different tokenizers/
vocabularies with different fertility (Phase 1 measured 1.65 vs 1.83
tokens/word) — a token-normalized loss is not a fair cross-language number.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pipeline.model import GPTConfig, GPTLanguageModel, load_checkpoint
from pipeline.train.dataset import build_dataset, load_documents


@torch.no_grad()
def compute_lm_metrics(
    checkpoint_path: str | Path,
    jsonl_path: str | Path,
    tokenizer_model: str | Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    mcfg = GPTConfig(**ckpt["config"]["model"])
    model = GPTLanguageModel(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    ds = build_dataset(jsonl_path, tokenizer_model, mcfg.max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    total_loss_nats, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        n = x.numel()
        total_loss_nats += loss.item() * n
        total_tokens += n

    avg_loss = total_loss_nats / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 30))
    total_bytes = sum(len(d.encode("utf-8")) for d in load_documents(jsonl_path))
    bpb = (total_loss_nats / math.log(2)) / max(1, total_bytes)

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": ckpt.get("step"),
        "eval_file": str(jsonl_path),
        "tokens_evaluated": total_tokens,
        "bytes_in_reference_text": total_bytes,
        "cross_entropy_nats": avg_loss,
        "perplexity": ppl,
        "bits_per_byte": bpb,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    with open(root / args.lang / "configs" / "model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    split_file = root / (tcfg["val_file"] if args.split == "val" else tcfg["test_file"])

    result = compute_lm_metrics(
        root / args.checkpoint, split_file, root / tcfg["tokenizer_model"], device=args.device
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out = Path(args.out) if args.out else root / args.lang / "data" / "stats" / f"phase2_lm_metrics_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
