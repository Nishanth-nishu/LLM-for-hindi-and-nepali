# Phase 2 — Attention Analysis

| | |
|---|---|
| Generated (UTC) | 2026-09-08 16:45:21 |
| Git commit | `9ca6c1de4c71bedc835ee996a6960d672e850a4d` |
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

Checkpoint step: **4999**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report\figures\phase2\nepali\heatmap_layer_early.png`
- Late layer: `report\figures\phase2\nepali\heatmap_layer_late.png`

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

Both models share the same architecture (7 layers, 8 heads) and the same qualitative shape: entropy is highest in the early layers (broad, exploratory attention), drops through the middle layers, and per-layer mean attention distance is lowest around layer 4 (short-range/local heads) before rising sharply at the final layer (layer 6: 84.9 tokens back in H, 97.0 in L — both models' last layer aggregates information from far earlier in the sequence, consistent with preparing next-token predictions from the full context).

Layer 4 has the lowest mean entropy in Model H and layer 4 in Model L (the same layer); layer 0 has the highest in H and layer 1 in L (different layers) — so the local-vs-long-range role is broadly similar in shape but not perfectly aligned by layer index. The spread between each model's highest and lowest per-layer entropy is 2.247 nats (H) vs 2.413 nats (L); Model H (Hindi) shows the flatter, less-differentiated profile, which — given L was trained on ~0.7% fewer tokens (471.6M vs 474.8M, this run's measured corpus) — is
 consistent with, though not conclusive proof of, a mild undertraining signature relative to H.
