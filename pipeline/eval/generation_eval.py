"""Generation-quality evaluation: greedy + temperature sampling from
held-out prefixes, scored against the true continuation, plus fluency/
diversity diagnostics.

    python -m pipeline.eval.generation_eval --lang hindi --checkpoint hindi/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
import yaml

from pipeline.eval.text_metrics import (
    corpus_bleu4,
    corpus_chrf,
    corpus_repetition_profile,
    corpus_rouge_l,
    distinct_n,
    repetition_rate,
)
from pipeline.model import GPTConfig, GPTLanguageModel, load_checkpoint
from pipeline.train.dataset import load_documents

DECODING_SETTINGS = [
    ("greedy", None, True),
    ("temp_0.5", 0.5, False),
    ("temp_1.0", 1.0, False),
    ("temp_1.5", 1.5, False),
]


def load_model(checkpoint_path: str | Path, device: str):
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    mcfg = GPTConfig(**ckpt["config"]["model"])
    model = GPTLanguageModel(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, mcfg, ckpt.get("step")


def build_prompt_reference_pairs(
    sp: spm.SentencePieceProcessor,
    documents: list[str],
    n_examples: int,
    prompt_tokens: int,
    ref_tokens: int,
) -> list[tuple[list[int], list[int]]]:
    pairs = []
    for doc in documents:
        ids = sp.encode(doc, out_type=int)
        if len(ids) < prompt_tokens + 8:
            continue
        prompt = ids[:prompt_tokens]
        ref = ids[prompt_tokens : prompt_tokens + ref_tokens]
        if len(ref) < 4:
            continue
        pairs.append((prompt, ref))
        if len(pairs) >= n_examples:
            break
    return pairs


@torch.no_grad()
def run_generation_eval(
    lang: str,
    checkpoint_path: str | Path,
    repo_root: str | Path = ".",
    n_examples: int = 30,
    prompt_tokens: int = 48,
    ref_tokens: int = 48,
    device: str = "cpu",
    seed: int = 20260820,
    config_name: str = "model_config.yaml",
) -> dict:
    torch.manual_seed(seed)
    root = Path(repo_root).resolve()
    with open(root / lang / "configs" / config_name, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]

    sp = spm.SentencePieceProcessor(model_file=str(root / tcfg["tokenizer_model"]))
    model, mcfg, step = load_model(root / checkpoint_path, device)

    documents = load_documents(root / tcfg["test_file"])
    pairs = build_prompt_reference_pairs(sp, documents, n_examples, prompt_tokens, ref_tokens)

    results = {"lang": lang, "checkpoint": str(checkpoint_path), "checkpoint_step": step,
               "n_examples": len(pairs), "prompt_tokens": prompt_tokens, "ref_tokens": ref_tokens,
               "settings": {}, "examples": []}

    for name, temperature, greedy in DECODING_SETTINGS:
        hyp_word_lists, ref_word_lists = [], []
        hyp_texts, ref_texts = [], []
        gen_token_lists = []
        rep_rates = []
        examples = []

        for prompt_ids, ref_ids in pairs:
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out = model.generate(
                prompt_t, max_new_tokens=len(ref_ids), temperature=(temperature or 1.0), greedy=greedy
            )
            gen_ids = out[0, len(prompt_ids):].tolist()
            gen_text = sp.decode(gen_ids)
            ref_text = sp.decode(ref_ids)

            hyp_texts.append(gen_text)
            ref_texts.append(ref_text)
            hyp_word_lists.append(gen_text.split())
            ref_word_lists.append(ref_text.split())
            gen_token_lists.append([sp.id_to_piece(i) for i in gen_ids])
            rep_rates.append(repetition_rate([sp.id_to_piece(i) for i in gen_ids]))

            if len(examples) < 5:
                examples.append({
                    "prompt": sp.decode(prompt_ids),
                    "reference_continuation": ref_text,
                    "generated_continuation": gen_text,
                })

        rep_profile = corpus_repetition_profile(gen_token_lists, ns=(1, 2, 3, 4))
        results["settings"][name] = {
            "bleu4": corpus_bleu4(hyp_word_lists, ref_word_lists),
            "chrf": corpus_chrf(hyp_texts, ref_texts),
            "rouge_l": corpus_rouge_l(hyp_word_lists, ref_word_lists),
            "distinct_1": distinct_n(gen_token_lists, 1),
            "distinct_2": distinct_n(gen_token_lists, 2),
            "repetition_rate": sum(rep_rates) / max(1, len(rep_rates)),
            "repetition_rate_1gram": rep_profile[1],
            "repetition_rate_2gram": rep_profile[2],
            "repetition_rate_3gram": rep_profile[3],
            "repetition_rate_4gram": rep_profile[4],
        }
        if name == "temp_1.0":
            results["examples"] = examples

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="model_config.yaml")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    result = run_generation_eval(args.lang, args.checkpoint, root, n_examples=args.n_examples,
                                  device=args.device, config_name=args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out = Path(args.out) if args.out else root / args.lang / "data" / "stats" / "phase2_generation_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
