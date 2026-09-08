# Phase 2 — Attention Analysis

| | |
|---|---|
| Generated (UTC) | 2026-09-08 17:22:48 |
| Git commit | `95d64d64b9b122cea18c7162e68cd1e26c02c648` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Model H (higher-resource, Hindi)

Checkpoint step: **6500**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report\figures\phase2\hindi\heatmap_layer_early.png`
- Late layer: `report\figures\phase2\hindi\heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 4.143 | 58.37 |
| 1 | 4.110 | 81.81 |
| 2 | 3.042 | 60.37 |
| 3 | 2.213 | 10.40 |
| 4 | 1.896 | 17.94 |
| 5 | 2.691 | 78.05 |
| 6 | 3.519 | 84.88 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model L (lower-resource, Nepali)

Checkpoint step: **6799**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report/figures/phase2/nepali/heatmap_layer_early.png`
- Late layer: `report/figures/phase2/nepali/heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 4.083 | 45.63 |
| 1 | 4.671 | 94.31 |
| 2 | 3.664 | 73.55 |
| 3 | 2.327 | 5.54 |
| 4 | 2.249 | 8.64 |
| 5 | 2.951 | 43.58 |
| 6 | 3.236 | 113.07 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model H vs Model L

Both models share the same architecture (7 layers, 8 heads) and the same qualitative shape: entropy is highest in the early layers (broad, exploratory attention), drops through the middle layers, and per-layer mean attention distance is lowest around layer 4 (short-range/local heads) before rising sharply at the final layer (layer 6: 84.9 tokens back in H, 113.1 in L — both models' last layer aggregates information from far earlier in the sequence, consistent with preparing next-token predictions from the full context).

Layer 4 has the lowest mean entropy in Model H and layer 4 in Model L (the same layer); layer 0 has the highest in H and layer 1 in L (different layers) — so the local-vs-long-range role is broadly similar in shape but not perfectly aligned by layer index. The spread between each model's highest and lowest per-layer entropy is 2.247 nats (H) vs 2.422 nats (L); Model H (Hindi) shows the flatter, less-differentiated profile, which — given L was trained on ~0.7% fewer tokens (471.6M vs 474.8M, this run's measured corpus) — is
 consistent with, though not conclusive proof of, a mild undertraining signature relative to H.
