"""Phase 2 pretraining loop for Model H / Model L.

    python -m pipeline.train.train --lang hindi --repo-root .
    python -m pipeline.train.train --lang nepali --repo-root .

AdamW, linear warmup + cosine decay, periodic validation, periodic
checkpointing. Every checkpoint contains model weights, optimizer state,
scheduler step, and the exact config used, per the Phase 2 spec.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pipeline.model import GPTConfig, GPTLanguageModel, count_parameters, save_checkpoint
from pipeline.train.dataset import build_dataset


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at_step(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr` by `max_steps`."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]):
    """Weight decay on >=2D parameters (linear/embedding weight matrices)
    only; biases and LayerNorm gain/bias are left undecayed, since decaying
    them has no regularizing motivation and empirically hurts small models."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def evaluate(model: GPTLanguageModel, loader: DataLoader, device: torch.device, max_iters: int) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def load_config(repo_root: Path, lang: str, config_name: str = "model_config.yaml") -> dict:
    with open(repo_root / lang / "configs" / config_name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(
    lang: str,
    repo_root: str = ".",
    max_steps_override: int | None = None,
    config_name: str = "model_config.yaml",
) -> dict:
    root = Path(repo_root).resolve()
    cfg = load_config(root, lang, config_name)
    mcfg, tcfg = cfg["model"], cfg["training"]

    torch.manual_seed(tcfg["seed"])
    device = pick_device(tcfg["device"])

    model_cfg = GPTConfig(**mcfg)
    model = GPTLanguageModel(model_cfg).to(device)
    pcount = count_parameters(model)
    print(f"[{lang}] parameters: {pcount['total']:,} (embedding share {pcount['embedding_share']:.1%}) on {device}", flush=True)

    t_tok0 = time.time()
    train_ds = build_dataset(root / tcfg["train_file"], root / tcfg["tokenizer_model"], model_cfg.max_seq_len)
    print(f"[{lang}] tokenized train split: {len(train_ds.tokens):,} tokens in {time.time() - t_tok0:.1f}s", flush=True)
    t_tok1 = time.time()
    val_ds = build_dataset(root / tcfg["val_file"], root / tcfg["tokenizer_model"], model_cfg.max_seq_len)
    print(f"[{lang}] tokenized val split: {len(val_ds.tokens):,} tokens in {time.time() - t_tok1:.1f}s", flush=True)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, drop_last=True)

    optimizer = build_optimizer(model, tcfg["lr"], tcfg["weight_decay"], tuple(tcfg["betas"]))

    max_steps = max_steps_override or tcfg["max_steps"]
    ckpt_dir = root / tcfg["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.jsonl"

    class _LRState:
        """Minimal object exposing state_dict()/load_state_dict() so the LR
        schedule (a plain function of step, not a torch scheduler class) can
        still be checkpointed like one."""

        def __init__(self, step: int = 0):
            self.step = step

        def state_dict(self):
            return {"step": self.step}

        def load_state_dict(self, d):
            self.step = d["step"]

    scheduler_state = _LRState()

    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        for step in range(max_steps):
            lr = lr_at_step(step, tcfg["warmup_steps"], max_steps, tcfg["lr"], tcfg["min_lr"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(tcfg["grad_accum_steps"]):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                _, loss, _ = model(x, targets=y)
                (loss / tcfg["grad_accum_steps"]).backward()
                accum_loss += loss.item() / tcfg["grad_accum_steps"]

            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler_state.step = step

            if step % tcfg["eval_every"] == 0 or step == max_steps - 1:
                val_loss = evaluate(model, val_loader, device, tcfg["eval_iters"])
                rec = {
                    "step": step,
                    "train_loss": accum_loss,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(min(val_loss, 20)),
                    "lr": lr,
                    "elapsed_s": time.time() - t0,
                }
                print(f"[{lang}] step {step:>7}  train_loss {accum_loss:.4f}  "
                      f"val_loss {val_loss:.4f}  val_ppl {rec['val_ppl']:.2f}  lr {lr:.2e}", flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

            if step > 0 and (step % tcfg["ckpt_every"] == 0 or step == max_steps - 1):
                save_checkpoint(
                    ckpt_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    scheduler_state,
                    step,
                    {"model": mcfg, "training": tcfg},
                )
                save_checkpoint(
                    ckpt_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler_state,
                    step,
                    {"model": mcfg, "training": tcfg},
                )

    return pcount


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--max-steps", type=int, default=None, help="override training.max_steps (e.g. for a smoke run)")
    ap.add_argument("--config", default="model_config.yaml",
                     help="config filename under <lang>/configs/ (e.g. model_config_ablation_nopos.yaml)")
    args = ap.parse_args()
    train(args.lang, args.repo_root, args.max_steps, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
