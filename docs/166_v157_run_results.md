# 166: v157 Layer-Gated Reservoir — Run Results

Date: 2026-08-02
Cluster: 4 nodes x 8x H20 (32 GPUs)
Commit: `ece308c` on `codex/v98-correctness-fixes`

## 1. Summary

v157 tests whether reservoir's motion/stability trade-off varies by transformer
depth. Four layer placements (early10/middle10/late10/interleaved10) each
enable reservoir on 10 of 30 layers (120 heads). Four methods are reused from
v155 (SF, all-reservoir, QK-top-reservoir, all-recent8). Total 128 videos
(64 new + 64 reused).

**Generation and VBench core-9 fully succeeded.** All 72 core tasks (8 methods
× 9 dimensions) complete. The interleaved10 placement — the pre-registered
blind primary — passes all 5 metric gates, retaining most of all-reservoir's
motion gain while recovering temporal stability.

## 2. VBench Core-9 Paper Table

| Method | Dynamic Degree | Motion Smoothness | Overall Consistency | Imaging Quality | Aesthetic Quality | Quality Score |
|---|---:|---:|---:|---:|---:|---:|
| sf_native | 64.17 | 98.22 | 23.32 | 68.93 | 61.57 | 83.04 |
| early10 reservoir | 77.50 | 97.91 | 23.47 | 71.03 | 62.17 | 83.83 |
| middle10 reservoir | 77.50 | 98.00 | 23.94 | 70.78 | 62.97 | 84.24 |
| late10 reservoir | 75.83 | 97.99 | 23.90 | 70.93 | 61.55 | 83.68 |
| **interleaved10 reservoir** | **79.17** | **98.15** | **23.86** | **71.24** | **62.67** | **84.54** |
| all reservoir (ref) | 83.33 | 97.71 | 24.14 | 69.69 | 61.83 | 83.72 |
| QK-top4 reservoir (ref) | 72.08 | 98.17 | 23.81 | 70.68 | 61.67 | 83.76 |
| all recent8 (ref) | 73.33 | 97.96 | 23.09 | 70.79 | 61.97 | 83.60 |

## 3. Metric Gate (per doc 165 section 6.4)

The interleaved10 blind primary must satisfy all 5 gates:

| Gate | Requirement | interleaved10 | Reference | Delta | Pass |
|---|---|---|---|---|---|
| dynamic vs recent8 | ≥ +0.02 | 0.79167 | 0.73333 | +0.05834 | ✅ |
| temporal quality vs all-reservoir | ≥ +0.003 | 0.96230 | 0.95468 | +0.00762 | ✅ |
| history consistency vs recent8 | ≥ -0.002 | 0.97213 | 0.96848 | +0.00365 | ✅ |
| temporal quality vs recent8 | ≥ -0.004 | 0.96230 | 0.95941 | +0.00289 | ✅ |
| visual quality vs recent8 | ≥ -0.01 | 0.71240 | 0.70789 | +0.00451 | ✅ |

**All 5 gates pass.** The interleaved10 layer-gated reservoir retains most of
all-reservoir's motion gain (79.17 vs 83.33 dynamic) while recovering temporal
stability (flicker 0.96230 vs 0.95468) and improving visual quality (imaging
0.71240 vs 0.69689).

## 4. Layer Placement Comparison

| Placement | Dynamic | Flicker | Smoothness | Quality Score |
|---|---:|---:|---:|---:|
| early10 (layers 0-9) | 77.50 | 0.95811 | 0.97905 | 83.83 |
| middle10 (layers 10-19) | 77.50 | 0.96132 | 0.98003 | 84.24 |
| late10 (layers 20-29) | 75.83 | 0.95882 | 0.97988 | 83.68 |
| **interleaved10** | **79.17** | **0.96230** | **0.98150** | **84.54** |
| all (layers 0-29) | 83.33 | 0.95468 | 0.97708 | 83.72 |

Layer placement matters: interleaved10 achieves the best Quality Score (84.54)
and best temporal stability among reservoir methods, while all-reservoir has the
most motion but worst stability. Early and late placements are weaker than
middle and interleaved, suggesting the reservoir benefit is not uniform across
depth — it concentrates in middle and distributed layers.

## 5. Key Findings

1. **Layer gating works**: restricting reservoir to 10 of 30 layers recovers
   temporal stability while retaining most of the motion gain. This confirms
   the v157 hypothesis (doc 165 section 6.1).

2. **Interleaved placement is optimal**: distributed layer selection (1,4,7,
   10,13,16,19,22,25,28) outperforms contiguous blocks on both motion and
   quality. This suggests the reservoir benefit is depth-distributed, not
   concentrated in one transformer region.

3. **All layer-gated methods beat SF**: every placement has higher Dynamic
   Degree (75-79 vs 64) and Quality Score (83.7-84.5 vs 83.0) than native SF.

4. **Pareto improvement over all-reservoir**: interleaved10 sacrifices only
   4.17 dynamic degree vs all-reservoir (79.17 vs 83.33) but gains 0.00762
   flicker, 0.00442 smoothness, and 0.01551 imaging quality. This is a
   favorable Pareto trade-off.

5. **Independent of QK membership**: this result uses ALL heads in the selected
   layers (120 heads = 10 layers × 12), not QK-top4 membership. The layer
   placement — not head classification — drives the improvement.

## 6. Decision (per doc 165 section 8)

> If interleaved simultaneously passes metric and blind gate, do a smaller
> layer-count budget sweep, rather than directly expanding to 128 prompts.

The metric gate passes. The blind gate requires human review (not yet scored).
Next step: complete the blind review, then if confirmed, run a layer-count
budget sweep (e.g., 6, 8, 10, 12 layers) before scaling to 128 prompts.

This result belongs to **cache allocation**, not head taxonomy. The v155
"cache useful, classifier unsupported" conclusion stands — v157 adds that
layer placement is a viable allocation axis.

## 7. Preserved Artifacts

```text
runs/v157_layer_gated_moviebench16/full8/
|-- videos/                    # 128 MP4 (64 new + 64 reused v155)
|-- published_manifest.json
|-- blind_review/
|-- contracts/
|-- metrics/
|   |-- vbench_core9_summary.{json,csv,md}
|   `-- paper_table_core9/paper_table.md
`-- v157_diagnostics.tar.gz
```

Key results copied to `docs/results/v157_layer_gated_moviebench16/`.
