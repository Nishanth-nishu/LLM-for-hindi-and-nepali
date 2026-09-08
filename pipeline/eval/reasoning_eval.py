"""Phase 3.1 reasoning evaluation: exact-match accuracy on the synthetic
comparative-reasoning test set, for either the pretrained (zero-shot) or
the finetuned checkpoint -- same script, just point --checkpoint at
whichever one you want scored.

    python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt --tag pretrained
    python -m pipeline.eval.reasoning_eval --lang hindi --checkpoint hindi/checkpoints_reasoning/latest.pt --tag finetuned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
import yaml

from pipeline.model import GPTConfig, GPTLanguageModel, load_checkpoint


def load_examples(jsonl_path: Path) -> list[dict]:
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@torch.no_grad()
def run_reasoning_eval(
    lang: str,
    checkpoint_path: str | Path,
    repo_root: str | Path = ".",
    config_name: str = "reasoning_finetune_config.yaml",
    split: str = "test",
    device: str = "cpu",
    max_new_tokens: int = 8,
    n_qualitative: int = 12,
) -> dict:
    root = Path(repo_root).resolve()
    with open(root / lang / "configs" / config_name, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fcfg = cfg["finetune"]

    sp = spm.SentencePieceProcessor(model_file=str(root / fcfg["tokenizer_model"]))
    ckpt = load_checkpoint(root / checkpoint_path, map_location=device)
    mcfg = GPTConfig(**ckpt["config"]["model"])
    model = GPTLanguageModel(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    split_file = {"train": fcfg["train_file"], "val": fcfg["val_file"], "test": fcfg["test_file"]}[split]
    examples = load_examples(root / split_file)
    delimiter = fcfg["delimiter"]
    eos_id = sp.eos_id()

    correct = 0
    by_family: dict[str, list[int]] = {}
    qualitative = []

    for ex in examples:
        prompt_ids = sp.encode(f'{ex["prompt"]} {delimiter} ', out_type=int)
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(prompt_t, max_new_tokens=max_new_tokens, greedy=True)
        gen_ids = out[0, len(prompt_ids):].tolist()
        if eos_id is not None and eos_id >= 0 and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]
        predicted = sp.decode(gen_ids).strip()
        gold = ex["answer"].strip()
        is_correct = predicted == gold
        correct += int(is_correct)

        fam = ex["family"]
        by_family.setdefault(fam, []).append(int(is_correct))

        if len(qualitative) < n_qualitative:
            qualitative.append({
                "family": fam, "prompt": ex["prompt"], "gold": gold,
                "predicted": predicted, "correct": is_correct,
            })

    n = len(examples)
    result = {
        "lang": lang, "checkpoint": str(checkpoint_path), "checkpoint_step": ckpt.get("step"),
        "split": split, "n_examples": n,
        "exact_match_accuracy": correct / n if n else 0.0,
        "accuracy_by_family": {
            fam: sum(flags) / len(flags) for fam, flags in by_family.items()
        },
        "n_by_family": {fam: len(flags) for fam, flags in by_family.items()},
        "qualitative_examples": qualitative,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config", default="reasoning_finetune_config.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", required=True, help="'pretrained' or 'finetuned' -- used only to name the output file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    result = run_reasoning_eval(
        args.lang, args.checkpoint, root, args.config, args.split, args.device
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out = Path(args.out) if args.out else root / args.lang / "data" / "stats" / f"phase3_reasoning_eval_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
