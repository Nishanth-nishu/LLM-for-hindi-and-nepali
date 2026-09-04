# Phase 2 — Attention Analysis

| | |
|---|---|
| Generated (UTC) | 2026-09-04 06:39:46 |
| Git commit | `d68cff126061309a911334f15371e3df0d4197c3` |
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

Checkpoint step: **199**. Causal mask verified on every analyzed sentence: **True** (changing a future token left every earlier position's logits bit-for-bit identical).

### Heatmaps

- Early layer: `report/figures/phase2/nepali/heatmap_layer_early.png`
- Late layer: `report/figures/phase2/nepali/heatmap_layer_late.png`

### Entropy (nats) and mean attention distance, by layer

| Layer | Mean entropy | Mean attention distance |
|--:|--:|--:|
| 0 | 5.177 | 126.79 |
| 1 | 4.871 | 126.15 |
| 2 | 4.041 | 128.91 |
| 3 | 4.021 | 127.41 |
| 4 | 3.701 | 127.24 |
| 5 | 4.813 | 127.62 |
| 6 | 5.120 | 127.17 |

Low entropy + low distance = a head attending sharply to nearby positions (positional/local). High entropy = diffuse attention across many positions. High distance with moderate entropy = a head pulling in specific, far-back content (content-based, long-range).

## Model H vs Model L

Once both models are analyzed, compare per-layer entropy and attention-distance profiles here: do the same layer indices play the same local-vs-long-range role in both languages, or does the lower-resource model (L) show flatter, less-differentiated attention (a common undertraining signature)?
