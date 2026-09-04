# Phase 2 — Attention Analysis

| | |
|---|---|
| Generated (UTC) | 2026-09-04 18:20:27 |
| Git commit | `593e6e79a54d90f533b73353471ce30cd6fe237a` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Model H (higher-resource, Hindi)

Checkpoint step: **199**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report/figures/phase2/hindi/heatmap_layer_early.png`
- Late layer: `report/figures/phase2/hindi/heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 4.903 | 106.41 |
| 1 | 4.708 | 103.42 |
| 2 | 4.659 | 105.26 |
| 3 | 4.499 | 106.33 |
| 4 | 4.450 | 107.09 |
| 5 | 4.565 | 108.04 |
| 6 | 4.694 | 107.97 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model L (lower-resource, Nepali)

Checkpoint step: **1250**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report/figures/phase2/nepali/heatmap_layer_early.png`
- Late layer: `report/figures/phase2/nepali/heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 4.529 | 64.95 |
| 1 | 4.636 | 74.22 |
| 2 | 4.625 | 70.14 |
| 3 | 3.008 | 13.47 |
| 4 | 2.990 | 11.84 |
| 5 | 3.390 | 29.24 |
| 6 | 4.339 | 84.82 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model H vs Model L

Once both models are analyzed, compare per-layer entropy and attention-distance profiles here: do the same layer indices play the same local-vs-long-range role in both languages, or does the lower-resource model (L) show flatter, less-differentiated attention (a common undertraining signature)?
