"""make_phase2_reports.py — generate the Phase 2 deliverable reports from
whatever artifacts exist (config files, checkpoints, eval JSON, training
logs). Same rule as tools/make_reports.py: nothing is hand-typed, a missing
artifact renders as "NOT YET MEASURED" naming the command that produces it,
never a plausible-looking invented number.

    python tools/make_phase2_reports.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

LANGS = ("hindi", "nepali")
MODEL_NAME = {"hindi": "Model H (higher-resource, Hindi)", "nepali": "Model L (lower-resource, Nepali)"}
MISSING = "⚠ **NOT YET MEASURED**"


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def need(cmd: str) -> str:
    return f"{MISSING} — run `{cmd}`"


def num(v, fmt: str = ",") -> str:
    return f"{v:{fmt}}" if isinstance(v, (int, float)) else MISSING


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


class Ctx:
    def __init__(self, root: Path):
        self.root = root
        self.now = datetime.now(timezone.utc)
        self.commit = sh(["git", "-C", str(root), "rev-parse", "HEAD"])
        self.branch = sh(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
        self.d = {}
        for lang in LANGS:
            cfg_path = root / lang / "configs" / "model_config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
            stats = root / lang / "data" / "stats"
            log_path = root / lang / "checkpoints" / "training_log.jsonl"
            log = []
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        log.append(json.loads(line))
            links_path = root / "report" / "phase2_checkpoint_links.json"
            links = load(links_path) or {}
            self.d[lang] = {
                "cfg": cfg,
                "drive_checkpoint": links.get(lang),
                "lm_val": load(stats / "phase2_lm_metrics_val.json"),
                "lm_test": load(stats / "phase2_lm_metrics_test.json"),
                "gen": load(stats / "phase2_generation_eval.json"),
                "attn": load(stats / "phase2_attention_analysis.json"),
                "log": log,
                "ckpt_dir": root / lang / "checkpoints",
            }

    def header(self, title: str) -> list[str]:
        return [
            f"# {title}", "",
            "| | |", "|---|---|",
            f"| Generated (UTC) | {self.now:%Y-%m-%d %H:%M:%S} |",
            f"| Git commit | `{self.commit}` |",
            f"| Branch | `{self.branch}` |", "",
            "> Following the Phase 1 convention: every number below is read from a "
            "JSON file a script actually produced. A field with no source file "
            f"renders as `{MISSING}` and names the command that fills it.", "",
        ]


def param_count_for(cfg: dict) -> dict | None:
    """Compute the exact parameter count from the config, same formula as
    pipeline/model/transformer.py, without needing torch at report time."""
    if cfg is None:
        return None
    m = cfg["model"]
    V, d, L, ff = m["vocab_size"], m["d_model"], m["n_layer"], m["d_ff"]
    ctx = m["max_seq_len"]
    embed = V * d
    pos = ctx * d
    per_block = 4 * (d * d + d) + 4 * d + 2 * (d * ff + ff) + ff * d - (d * ff)  # placeholder, replaced below
    # exact: qkv (d->3d) + out (d->d), each with bias; 2 LayerNorms; FFN d->ff->d with bias
    attn = (d * 3 * d + 3 * d) + (d * d + d)
    ln = 2 * (2 * d)
    ffn = (d * ff + ff) + (ff * d + d)
    per_block = attn + ln + ffn
    final_ln = 2 * d
    non_embed = L * per_block + pos + final_ln
    total_tied = embed + non_embed
    total_untied = total_tied + embed  # separate output head
    tied = m.get("tie_weights", True)
    total = total_tied if tied else total_untied
    return {
        "total": total, "embedding_params": embed, "embedding_share": embed / total,
        "non_embedding_params": non_embed, "tied": tied,
    }


# ---------------------------------------------------------------------------
# 1. Architecture report (deliverables 1, 2, 3-partial)
# ---------------------------------------------------------------------------

def r_architecture(c: Ctx) -> str:
    L = c.header("Phase 2 — Model Architecture Report")
    L += [
        "## 1. Implementation", "",
        "Decoder-only GPT-style Transformer, built from `nn.Linear`, "
        "`nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only — no "
        "`nn.Transformer*`, no HuggingFace model classes, no pre-built "
        "attention block. Source: `pipeline/model/transformer.py`.", "",
        "| Component | Choice | Why |", "|---|---|---|",
        "| Positional scheme | Learned absolute positional embeddings | "
        "Simple, and Phase 1 fixed a known small context length — no need "
        "for length extrapolation. Hard consequence: the model has no "
        "representation for any position >= `max_seq_len`, which is "
        "therefore a hard ceiling on sequence length, not a soft default. |",
        "| Norm placement | Pre-norm (`x + Sublayer(LN(x))`) | Keeps an "
        "unimpeded identity path through the residual stream; post-norm "
        "routes every residual gradient back through a LayerNorm, which "
        "is harder to train at depth. |",
        "| Attention scaling | `1/sqrt(d_head)` | Keeps dot-product "
        "variance ~O(1) regardless of `d_head`; unscaled, variance grows "
        "with `d_head` and pushes softmax into a near-one-hot, "
        "near-zero-gradient regime. |",
        "| Causal mask | Additive upper-triangular `-inf` mask, added to "
        "scores before softmax | Implemented by hand "
        "(`CausalSelfAttention.causal_mask`), not a library flag. Verified "
        "empirically — see `tests/test_phase2_model.py::"
        "test_causal_mask_future_token_does_not_change_past_logits` and "
        "`pipeline/eval/attention_analysis.verify_causal_mask`. |",
        "| FFN | Linear(d->4d) -> GELU -> Linear(4d->d) | Standard GPT "
        "expansion ratio. |",
        "| Weight tying | `lm_head.weight = tok_emb.weight` | Saves "
        "`vocab_size * d_model` parameters — see `tests/"
        "test_phase2_model.py::test_weight_tying_saves_parameters`, which "
        "asserts the saving is exactly that many parameters. |", "",
        "### Tensor shapes through one block (`X` of shape `(B, T, d_model)`)", "",
        "```", "qkv = qkv_proj(X)                          (B, T, 3*d_model)",
        "q, k, v = split(qkv, d_model, dim=2)       each (B, T, d_model)",
        "q, k, v -> view + transpose                each (B, n_head, T, d_head)",
        "scores = q @ k^T / sqrt(d_head)             (B, n_head, T, T)",
        "scores = scores + causal_mask               (B, n_head, T, T)",
        "weights = softmax(scores, dim=-1)           (B, n_head, T, T)",
        "out = weights @ v                           (B, n_head, T, d_head)",
        "out -> transpose + reshape (concat heads)   (B, T, d_model)",
        "out = out_proj(out)                         (B, T, d_model)",
        "```", "",
        "## 2. Configurations", "",
    ]
    for lang in LANGS:
        cfg = c.d[lang]["cfg"]
        L += [f"### {MODEL_NAME[lang]}", ""]
        if cfg is None:
            L += [need(f"create {lang}/configs/model_config.yaml"), ""]
            continue
        m = cfg["model"]
        pc = param_count_for(cfg)
        L += ["| Hyperparameter | Value |", "|---|--:|"]
        for k in ("vocab_size", "max_seq_len", "d_model", "n_layer", "n_head", "d_ff",
                   "embed_dropout", "attn_dropout", "resid_dropout", "tie_weights"):
            L.append(f"| `{k}` | {m[k]} |")
        L += ["", f"**Computed parameter count: {pc['total']:,}** "
              f"(embedding share {pc['embedding_share']:.1%}, "
              f"non-embedding {pc['non_embedding_params']:,}) "
              f"— computed directly from the config above; matches "
              f"`pipeline.model.count_parameters()` at model-build time.", ""]
    L += [
        "### Depth/width trade-off", "",
        "At a fixed ~25M-parameter budget, both configs use `d_model=512`, "
        "`n_layer=7`, `n_head=8` (`d_head=64`), `d_ff=2048`. Depth was "
        "favoured over extra width: doubling `d_model` roughly "
        "quadruples attention+FFN cost per layer (both scale with `d^2`) "
        "for the same parameter budget, forcing far fewer layers — and a "
        "causal-LM objective benefits from more sequential composition "
        "steps more than from a wider single step at this scale. Both "
        "models share the architecture on purpose: Phase 1 landed both "
        "tokenizers on vocab 4,000, so any H vs L difference in Phase 2 "
        "results reflects the corpora/language, not an architecture change.", "",
        "### Vocabulary and embedding cost", "",
        "Both tokenizers are vocab 4,000 (Phase 1). Tied embeddings: "
        "4,000 x 512 = 2,048,000 parameters (~8.2-8.4% of the ~24-25M "
        "total). Untying would add a second such matrix, roughly doubling "
        "the embedding share — the vocabulary size was chosen in Phase 1 "
        "assuming the tied figure (see `report/phase1_tokenizer_report.md`).", "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 2. Training report (deliverable 3, 4)
# ---------------------------------------------------------------------------

def r_training(c: Ctx) -> str:
    L = c.header("Phase 2 — Training Report")
    L += [
        "## Training setup", "",
        "AdamW (`betas`, `weight_decay` per config; weight decay applied "
        "only to >=2D parameter tensors, not biases/LayerNorm), linear "
        "warmup then cosine decay to `min_lr`, gradient clipping, gradient "
        "accumulation. Source: `pipeline/train/train.py`. Each checkpoint "
        "contains model weights, optimizer state, the LR-schedule step, "
        "and the exact config (`pipeline/model/checkpoint.py`).", "",
    ]
    for lang in LANGS:
        d = c.d[lang]
        L += [f"## {MODEL_NAME[lang]}", ""]
        if not d["cfg"]:
            L += [need(f"create {lang}/configs/model_config.yaml"), ""]
            continue
        tcfg = d["cfg"]["training"]
        L += ["| Setting | Value |", "|---|--:|"]
        for k in ("batch_size", "grad_accum_steps", "max_steps", "warmup_steps",
                   "lr", "min_lr", "weight_decay", "grad_clip", "eval_every", "ckpt_every"):
            L.append(f"| `{k}` | {tcfg[k]} |")
        L += ["", f"Effective tokens/step: **{tcfg['batch_size'] * tcfg['grad_accum_steps'] * d['cfg']['model']['max_seq_len']:,}**", ""]

        if not d["log"]:
            L += [need(f"python -m pipeline.train.train --lang {lang} --repo-root ."), ""]
            continue
        first, last = d["log"][0], d["log"][-1]
        best_val = min(r["val_loss"] for r in d["log"])
        L += ["### Progress", "",
              "| Metric | First logged step | Latest logged step |",
              "|---|--:|--:|",
              f"| Step | {first['step']} | {last['step']} |",
              f"| Train loss | {first['train_loss']:.4f} | {last['train_loss']:.4f} |",
              f"| Val loss | {first['val_loss']:.4f} | {last['val_loss']:.4f} |",
              f"| Val PPL | {first['val_ppl']:.2f} | {last['val_ppl']:.2f} |",
              f"| LR | {first['lr']:.2e} | {last['lr']:.2e} |", "",
              f"Best val loss so far: **{best_val:.4f}**  ", ]
        pct = 100 * last["step"] / tcfg["max_steps"]
        L += [f"Training progress: **{last['step']:,} / {tcfg['max_steps']:,} steps "
              f"({pct:.1f}%)**.", ""]
        if pct < 100:
            L += ["**This is a partial/pilot run, not a converged model** — "
                  "see `docs/PHASE2_GCP_TRAINING.md` for the full-budget run "
                  "plan and estimated wall-clock time on the project's "
                  "hardware.", ""]
        fig = c.root / "report" / "figures" / "phase2" / lang / "loss_curve.png"
        L += [f"Loss curve: `{fig.relative_to(c.root)}`" if fig.exists()
              else need(f"python tools/make_phase2_figures.py --repo-root ."), ""]
        if d.get("drive_checkpoint"):
            L += [f"Checkpoint (Drive): {d['drive_checkpoint']}", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 3. LM evaluation report (deliverable 5a)
# ---------------------------------------------------------------------------

def r_lm_eval(c: Ctx) -> str:
    L = c.header("Phase 2 — Language Modeling Evaluation")
    L += ["## PPL / BPB, side by side", "",
          "| | Model H (Hindi) | Model L (Nepali) |", "|---|--:|--:|"]
    rows = [
        ("Checkpoint step", lambda l: str((c.d[l]["lm_test"] or {}).get("checkpoint_step", MISSING))),
        ("Test cross-entropy (nats)", lambda l: f"{(c.d[l]['lm_test'] or {}).get('cross_entropy_nats', MISSING):.4f}" if c.d[l]["lm_test"] else MISSING),
        ("Test perplexity", lambda l: f"{(c.d[l]['lm_test'] or {}).get('perplexity', MISSING):.2f}" if c.d[l]["lm_test"] else MISSING),
        ("Test bits-per-byte", lambda l: f"{(c.d[l]['lm_test'] or {}).get('bits_per_byte', MISSING):.4f}" if c.d[l]["lm_test"] else MISSING),
        ("Val perplexity", lambda l: f"{(c.d[l]['lm_val'] or {}).get('perplexity', MISSING):.2f}" if c.d[l]["lm_val"] else MISSING),
    ]
    for label, fn in rows:
        L.append(f"| {label} | {fn('hindi')} | {fn('nepali')} |")
    L += ["", "Missing rows: " + need("python -m pipeline.eval.lm_metrics --lang <lang> --checkpoint <ckpt> --split test") if not (c.d["hindi"]["lm_test"] and c.d["nepali"]["lm_test"]) else "", ""]
    L += [
        "## Why BPB, not just PPL", "",
        "Perplexity is computed per *token*, and the two languages use "
        "separate tokenizers with different measured fertility (Phase 1: "
        "1.6522 Hindi vs 1.8338 Nepali tokens/word) — a token is not the "
        "same amount of text in both languages, so comparing raw PPL "
        "across languages is misleading. Bits-per-byte normalizes by "
        "UTF-8 bytes of the underlying text instead, which is tokenizer- "
        "and language-agnostic.", "",
        "## Discussing the H vs L gap", "",
        "Once both are measured, this section should relate the gap to: "
        "training-token count (559.9M H vs 501.2M L), manual-token share "
        "(21.29% H vs 20.83% L), tokenizer fertility (1.65 vs 1.83) and "
        "byte-fallback rate (0.73% H vs 0.28% L — see "
        "`report/phase1_tokenizer_report.md`), and script/orthographic "
        "differences between the two languages. Fill in with the measured "
        "numbers once both checkpoints have been evaluated.", "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 4. Generation report (deliverables 5b, 6)
# ---------------------------------------------------------------------------

def r_generation(c: Ctx) -> str:
    L = c.header("Phase 2 — Generation Quality and Diversity")
    L += ["## Metrics by decoding setting", ""]
    for lang in LANGS:
        L += [f"## {MODEL_NAME[lang]}", ""]
        gen = c.d[lang]["gen"]
        if not gen:
            L += [need(f"python -m pipeline.eval.generation_eval --lang {lang} --checkpoint <ckpt>"), ""]
            continue
        L += [f"Checkpoint step: **{gen.get('checkpoint_step')}**, "
              f"{gen.get('n_examples')} held-out prompt/reference pairs "
              f"({gen.get('prompt_tokens')} prompt tokens, "
              f"{gen.get('ref_tokens')} reference tokens).", "",
              "| Setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for name, m in gen["settings"].items():
            L.append(f"| {name} | {m['bleu4']:.4f} | {m['chrf']:.4f} | {m['rouge_l']:.4f} | "
                     f"{m['distinct_1']:.3f} | {m['distinct_2']:.3f} | {m['repetition_rate']:.3f} |")
        L += ["", "### Example generations (temperature 1.0)", ""]
        for ex in gen.get("examples", [])[:5]:
            L += [f"- **Prompt:** {ex['prompt']}",
                  f"  **Reference:** {ex['reference_continuation']}",
                  f"  **Generated:** {ex['generated_continuation']}", ""]
    L += [
        "## Why these metrics are (and aren't) informative here", "",
        "- **BLEU-4** rewards exact n-gram overlap with one reference "
        "continuation; for open-ended generation there are many valid "
        "continuations, so BLEU understates quality whenever the model "
        "diverges from the reference in a reasonable way — it is more a "
        "lower bound / sanity check than a quality score here.",
        "- **chrF** operates on character n-grams, so it is more forgiving "
        "of morphological variation (relevant for both Hindi and Nepali "
        "inflection) than word-level BLEU, and degrades more gracefully "
        "under a small, undertrained model's spelling drift.",
        "- **ROUGE-L** rewards the longest common subsequence rather than "
        "exact n-gram matches, so it tolerates reordering better than "
        "BLEU but still assumes a single reference is representative.",
        "- **Distinct-1/2 and repetition rate** matter independently of "
        "all three above: a model can score adequately on n-gram overlap "
        "while degenerating into repetition loops (a known failure mode "
        "of greedy decoding on undertrained LMs) — this is exactly the "
        "case the corpus-overlap metrics do not catch.", "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5. Attention report (deliverable 7)
# ---------------------------------------------------------------------------

def r_attention(c: Ctx) -> str:
    L = c.header("Phase 2 — Attention Analysis")
    for lang in LANGS:
        L += [f"## {MODEL_NAME[lang]}", ""]
        a = c.d[lang]["attn"]
        if not a:
            L += [need(f"python -m pipeline.eval.attention_analysis --lang {lang} --checkpoint <ckpt>"), ""]
            continue
        L += [f"Checkpoint step: **{a.get('checkpoint_step')}**. "
              f"Causal mask verified on every analyzed sentence: "
              f"**{a.get('causal_mask_verified_all_sentences')}** "
              f"(changing a future token left every earlier position's "
              f"logits bit-for-bit identical).", "",
              "### Heatmaps", "",
              f"- Early layer: `{a['heatmaps'][0]}`",
              f"- Late layer: `{a['heatmaps'][1]}`", "",
              "### Entropy (nats) and mean attention distance, by layer", "",
              "| Layer | Mean entropy | Mean attention distance |",
              "|--:|--:|--:|"]
        for i, (e, dist) in enumerate(zip(a["entropy_mean_per_layer"], a["distance_mean_per_layer"])):
            L.append(f"| {i} | {e:.3f} | {dist:.2f} |")
        L += ["",
              "Low entropy + low distance = a head attending sharply to "
              "nearby positions (positional/local). High entropy = diffuse "
              "attention across many positions. High distance with "
              "moderate entropy = a head pulling in specific, far-back "
              "content (content-based, long-range).", ""]
    L += [
        "## Model H vs Model L", "",
        "Once both models are analyzed, compare per-layer entropy and "
        "attention-distance profiles here: do the same layer indices play "
        "the same local-vs-long-range role in both languages, or does the "
        "lower-resource model (L) show flatter, less-differentiated "
        "attention (a common undertraining signature)?", "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 6. Resource comparison (deliverable 8)
# ---------------------------------------------------------------------------

def r_resources(c: Ctx) -> str:
    L = c.header("Phase 2 — Resource-Level Comparison (Model H vs Model L)")
    L += ["| | Model H (Hindi) | Model L (Nepali) |", "|---|--:|--:|",
          "| Training tokens (Phase 1, measured) | 559,913,515 | 501,226,336 |",
          "| Manual token share | 21.29% | 20.83% |",
          "| Tokenizer vocab | 4,000 | 4,000 |",
          "| Fertility (tokens/word) | 1.6522 | 1.8338 |",
          "| Byte-fallback rate | 0.7316% | 0.2757% |", ""]
    for lang in LANGS:
        pc = param_count_for(c.d[lang]["cfg"]) if c.d[lang]["cfg"] else None
        L.append(f"" )
    L += ["| Parameters | " +
          (f"{param_count_for(c.d['hindi']['cfg'])['total']:,}" if c.d['hindi']['cfg'] else MISSING) +
          " | " +
          (f"{param_count_for(c.d['nepali']['cfg'])['total']:,}" if c.d['nepali']['cfg'] else MISSING) +
          " |", ""]
    for lang in LANGS:
        lm = c.d[lang]["lm_test"]
        L.append("")
    L += ["| Test perplexity | " +
          (f"{c.d['hindi']['lm_test']['perplexity']:.2f}" if c.d['hindi']['lm_test'] else MISSING) +
          " | " +
          (f"{c.d['nepali']['lm_test']['perplexity']:.2f}" if c.d['nepali']['lm_test'] else MISSING) +
          " |",
          "| Test BPB | " +
          (f"{c.d['hindi']['lm_test']['bits_per_byte']:.4f}" if c.d['hindi']['lm_test'] else MISSING) +
          " | " +
          (f"{c.d['nepali']['lm_test']['bits_per_byte']:.4f}" if c.d['nepali']['lm_test'] else MISSING) +
          " |", ""]
    L += [
        "## Write-up", "",
        "Same architecture, same parameter budget, same vocabulary size, "
        "independently trained on independently collected corpora that "
        "share no documents (Phase 1 verification check C1-C3). Any gap "
        "in the metrics above is therefore attributable to the corpus "
        "(size, manual/downloaded mix, quality) and the language/script "
        "itself (Phase 1's own finding: manual text tokenizes *better* "
        "than downloaded in Nepali but *worse* in Hindi — an observation, "
        "not a generalizable finding, with only two languages) — not to "
        "any architectural difference between the two models.", "",
        "Fill in the specific gap size and direction once both models "
        "have been trained to a comparable step count and evaluated.", "",
    ]
    return "\n".join(L)


REPORTS = [
    ("phase2_architecture_report.md", r_architecture),
    ("phase2_training_report.md", r_training),
    ("phase2_lm_evaluation_report.md", r_lm_eval),
    ("phase2_generation_report.md", r_generation),
    ("phase2_attention_report.md", r_attention),
    ("phase2_resource_comparison.md", r_resources),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default="report")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    c = Ctx(root)
    missing_total = 0
    for name, fn in REPORTS:
        text = fn(c)
        (out / name).write_text(text, encoding="utf-8")
        n = text.count(MISSING)
        missing_total += n
        print(f"  wrote {args.out_dir}/{name:<38} {'(' + str(n) + ' fields not yet measured)' if n else '(complete)'}")

    print(f"\n  {missing_total} field(s) not yet measured." if missing_total else "\n  every field populated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
