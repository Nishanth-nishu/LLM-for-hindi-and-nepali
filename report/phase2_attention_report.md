# Phase 2 — Attention Analysis

| | |
|---|---|
| Generated (UTC) | 2026-09-07 10:06:53 |
| Git commit | `90f14547911afa81df40cb58257390d619cc8030` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Model H (higher-resource, Hindi)

Checkpoint step: **4999**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report/figures/phase2/hindi/heatmap_layer_early.png`
- Late layer: `report/figures/phase2/hindi/heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 4.220 | 48.96 |
| 1 | 4.445 | 67.35 |
| 2 | 3.854 | 61.61 |
| 3 | 2.687 | 18.54 |
| 4 | 1.942 | 9.91 |
| 5 | 2.681 | 22.62 |
| 6 | 3.292 | 98.91 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model L (lower-resource, Nepali)

Checkpoint step: **4999**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report/figures/phase2/nepali/heatmap_layer_early.png`
- Late layer: `report/figures/phase2/nepali/heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 4.237 | 48.87 |
| 1 | 4.679 | 84.42 |
| 2 | 4.153 | 78.05 |
| 3 | 2.646 | 18.73 |
| 4 | 2.266 | 6.32 |
| 5 | 2.992 | 30.93 |
| 6 | 3.406 | 97.01 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model H vs Model L

Both models share the same architecture (7 layers, 8 heads) and the same qualitative shape: entropy is highest in the early layers (broad, exploratory attention), drops through the middle layers, and per-layer mean attention distance is lowest around layer 4 (short-range/local heads) before rising sharply at the final layer (layer 6: 98.9 tokens back in H, 97.0 in L — both models' last layer aggregates information from far earlier in the sequence, consistent with preparing next-token predictions from the full context).

Layer 4 has the lowest mean entropy in Model H and layer 4 in Model L (the same layer); layer 1 has the highest in H and layer 1 in L (the same layer) — so the local-vs-long-range role is aligned by layer index across the two languages. The spread between each model's highest and lowest per-layer entropy is 2.503 nats (H) vs 2.413 nats (L); Model L (Nepali) shows the flatter, less-differentiated profile, which — given L was trained on ~0.7% fewer tokens (471.6M vs 474.8M, this run's measured corpus) — is
 consistent with, though not conclusive proof of, a mild undertraining signature relative to H.
