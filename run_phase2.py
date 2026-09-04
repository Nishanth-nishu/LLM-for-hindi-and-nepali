"""Phase 2 orchestrator — mirrors run_phase1.py's --stage convention.

    python run_phase2.py --lang hindi --stage train
    python run_phase2.py --lang hindi --stage evaluate-lm --checkpoint hindi/checkpoints/latest.pt
    python run_phase2.py --lang hindi --stage evaluate-generation --checkpoint hindi/checkpoints/latest.pt
    python run_phase2.py --lang hindi --stage attention --checkpoint hindi/checkpoints/latest.pt
    python run_phase2.py --lang hindi --stage all --checkpoint hindi/checkpoints/latest.pt

Each stage is also independently runnable as its own module (see the
`python -m pipeline....` command in that module's docstring) — this wrapper
just sequences them for a single language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STAGES = ["train", "evaluate-lm", "evaluate-generation", "attention", "report", "all"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--checkpoint", default=None, help="relative path, e.g. hindi/checkpoints/latest.pt")
    ap.add_argument("--max-steps", type=int, default=None, help="train stage only: override max_steps")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    stages = ["train", "evaluate-lm", "evaluate-generation", "attention", "report"] if args.stage == "all" else [args.stage]

    for stage in stages:
        print(f"\n=== [{args.lang}] stage: {stage} ===")
        if stage == "train":
            from pipeline.train.train import train
            train(args.lang, str(root), args.max_steps)
        elif stage in ("evaluate-lm", "evaluate-generation", "attention"):
            ckpt = args.checkpoint or f"{args.lang}/checkpoints/latest.pt"
            if not (root / ckpt).exists():
                print(f"  skipped — no checkpoint at {ckpt} yet (run --stage train first)")
                continue
            if stage == "evaluate-lm":
                from pipeline.eval.lm_metrics import compute_lm_metrics
                import yaml, json
                with open(root / args.lang / "configs" / "model_config.yaml", encoding="utf-8") as f:
                    tcfg = yaml.safe_load(f)["training"]
                for split, path in (("val", tcfg["val_file"]), ("test", tcfg["test_file"])):
                    result = compute_lm_metrics(root / ckpt, root / path, root / tcfg["tokenizer_model"], device=args.device)
                    out = root / args.lang / "data" / "stats" / f"phase2_lm_metrics_{split}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"  {split}: ppl={result['perplexity']:.2f} bpb={result['bits_per_byte']:.4f} -> {out}")
            elif stage == "evaluate-generation":
                from pipeline.eval.generation_eval import run_generation_eval
                import json
                result = run_generation_eval(args.lang, ckpt, root, device=args.device)
                out = root / args.lang / "data" / "stats" / "phase2_generation_eval.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  wrote {out}")
            elif stage == "attention":
                from pipeline.eval.attention_analysis import run_attention_analysis
                import json
                result = run_attention_analysis(args.lang, ckpt, root, device=args.device)
                out = root / args.lang / "data" / "stats" / "phase2_attention_analysis.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  wrote {out}")
        elif stage == "report":
            from tools.make_phase2_reports import main as make_reports_main
            make_reports_main(["--repo-root", str(root)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
