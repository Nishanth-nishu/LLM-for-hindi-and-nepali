"""Phase 3.1 reasoning finetuning: start from a language's own Phase 2
pretrained checkpoint, keep its tokenizer/vocabulary fixed, and finetune on
that language's synthetic reasoning dataset with answer-only loss masking
(pipeline/reasoning/reasoning_dataset.py).

    python -m pipeline.train.finetune --lang hindi --repo-root .
    python -m pipeline.train.finetune --lang nepali --repo-root .

Reuses the pretraining loop's device selection, LR schedule, and optimizer
setup (pipeline/train/train.py) rather than duplicating them -- only the
dataset and the per-step loop (small dataset, epoch-based, no gradient
accumulation) differ from pretraining.

Checkpoints are saved in the exact same resume-capable format as
pretraining (model/optimizer/scheduler/step/config), so they can be
loaded with the same pipeline.model.load_checkpoint used everywhere else
in this repo (reasoning_eval.py, attention_analysis.py, ...).
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

from pipeline.model import GPTConfig, GPTLanguageModel, count_parameters, load_checkpoint, save_checkpoint
from pipeline.reasoning.reasoning_dataset import ReasoningExampleDataset
from pipeline.train.train import build_optimizer, lr_at_step, pick_device


def load_config(repo_root: Path, lang: str, config_name: str) -> dict:
    with open(repo_root / lang / "configs" / config_name, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _LRState:
    """Same minimal state_dict()/load_state_dict() shim train.py uses, so a
    plain step-indexed LR function can still be checkpointed like a
    scheduler object."""

    def __init__(self, step: int = 0):
        self.step = step

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, d):
        self.step = d["step"]


@torch.no_grad()
def evaluate(model: GPTLanguageModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def finetune(lang: str, repo_root: str = ".", config_name: str = "reasoning_finetune_config.yaml") -> dict:
    root = Path(repo_root).resolve()
    cfg = load_config(root, lang, config_name)
    fcfg = cfg["finetune"]

    torch.manual_seed(fcfg["seed"])
    device = pick_device(fcfg["device"])

    base_ckpt = load_checkpoint(root / fcfg["base_checkpoint"], map_location=str(device))
    model_cfg = GPTConfig(**base_ckpt["config"]["model"])
    model = GPTLanguageModel(model_cfg).to(device)
    model.load_state_dict(base_ckpt["model_state_dict"])
    pcount = count_parameters(model)
    print(
        f"[{lang}] loaded base checkpoint {fcfg['base_checkpoint']} (step {base_ckpt.get('step')}); "
        f"{pcount['total']:,} params on {device}",
        flush=True,
    )

    train_ds = ReasoningExampleDataset(
        root / fcfg["train_file"], root / fcfg["tokenizer_model"], fcfg["delimiter"], fcfg["max_len"]
    )
    val_ds = ReasoningExampleDataset(
        root / fcfg["val_file"], root / fcfg["tokenizer_model"], fcfg["delimiter"], fcfg["max_len"]
    )
    train_loader = DataLoader(train_ds, batch_size=fcfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=fcfg["batch_size"], shuffle=False, drop_last=False)
    print(f"[{lang}] reasoning train examples: {len(train_ds)}, val examples: {len(val_ds)}", flush=True)

    optimizer = build_optimizer(model, fcfg["lr"], fcfg["weight_decay"], tuple(fcfg["betas"]))
    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * fcfg["epochs"]

    ckpt_dir = root / fcfg["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.jsonl"
    scheduler_state = _LRState()

    model.train()
    step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    with open(log_path, "a", encoding="utf-8") as logf:
        for epoch in range(fcfg["epochs"]):
            for x, y in train_loader:
                lr = lr_at_step(step, fcfg["warmup_steps"], max_steps, fcfg["lr"], fcfg["min_lr"])
                for g in optimizer.param_groups:
                    g["lr"] = lr

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, loss, _ = model(x, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), fcfg["grad_clip"])
                optimizer.step()
                scheduler_state.step = step

                if step % fcfg["eval_every"] == 0 or step == max_steps - 1:
                    val_loss = evaluate(model, val_loader, device)
                    rec = {
                        "step": step, "epoch": epoch, "train_loss": loss.item(),
                        "val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20)),
                        "lr": lr, "elapsed_s": time.time() - t0,
                    }
                    print(f"[{lang}] step {step:>5}/{max_steps} (epoch {epoch})  "
                          f"train_loss {loss.item():.4f}  val_loss {val_loss:.4f}  "
                          f"val_ppl {rec['val_ppl']:.2f}  lr {lr:.2e}", flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()

                    # Track the best checkpoint by validation loss at EVAL
                    # granularity (eval_every steps), not just at epoch
                    # boundaries -- with a small finetuning set the true
                    # optimum can fall well inside the first epoch, and an
                    # epoch-only checkpoint would silently miss it.
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            ckpt_dir / "best.pt", model, optimizer, scheduler_state, step,
                            {"model": base_ckpt["config"]["model"], "finetune": fcfg},
                            extra={
                                "base_checkpoint": fcfg["base_checkpoint"],
                                "base_checkpoint_step": base_ckpt.get("step"),
                                "best_val_loss": best_val_loss,
                            },
                        )
                step += 1

            save_checkpoint(
                ckpt_dir / f"epoch_{epoch}.pt", model, optimizer, scheduler_state, step,
                {"model": base_ckpt["config"]["model"], "finetune": fcfg},
                extra={"base_checkpoint": fcfg["base_checkpoint"], "base_checkpoint_step": base_ckpt.get("step")},
            )
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer, scheduler_state, step,
                {"model": base_ckpt["config"]["model"], "finetune": fcfg},
                extra={"base_checkpoint": fcfg["base_checkpoint"], "base_checkpoint_step": base_ckpt.get("step")},
            )

    return pcount


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config", default="reasoning_finetune_config.yaml")
    args = ap.parse_args()
    finetune(args.lang, args.repo_root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
