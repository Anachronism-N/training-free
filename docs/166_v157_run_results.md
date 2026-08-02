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
× 9 dimensions) complete. The interleaved10 placement, which was the
pre-registered blind primary, passes all 5 metric gates. Middle10 and late10
also pass the same metric screen; therefore the result supports layer-gated
allocation but does not establish that interleaved10 is uniquely optimal.

The blind package contains 128 rows, but all score fields are still empty.
No human-promotion conclusion is available yet.

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
| dynamic vs recent8 | ≥ +0.02 | 0.79167 | 0.73333 | +0.05833 | PASS |
| temporal quality vs all-reservoir | ≥ +0.003 | 0.97190 | 0.96588 | +0.00603 | PASS |
| history consistency vs recent8 | ≥ -0.002 | 0.72448 | 0.72019 | +0.00429 | PASS |
| temporal quality vs recent8 | ≥ -0.004 | 0.97190 | 0.96950 | +0.00240 | PASS |
| visual quality vs recent8 | ≥ -0.01 | 0.66956 | 0.66378 | +0.00578 | PASS |

**All 5 primary gates pass.** Here `temporal_quality` is the mean of flicker
and smoothness, `history_consistency` is the mean of subject/background/overall
consistency, and `visual_quality` is the mean of aesthetic/imaging quality.
The earlier version of this table accidentally printed individual dimensions
under composite labels; the frozen analyzer and pass/fail decisions were not
affected.

For completeness, middle10 and late10 also pass all five metric gates. Early10
fails only temporal recovery versus all-reservoir (`+0.00270 < +0.003`).

## 4. Layer Placement Comparison

| Placement | Dynamic | Flicker | Smoothness | Quality Score |
|---|---:|---:|---:|---:|
| early10 (layers 0-9) | 77.50 | 0.95811 | 0.97905 | 83.83 |
| middle10 (layers 10-19) | 77.50 | 0.96132 | 0.98003 | 84.24 |
| late10 (layers 20-29) | 75.83 | 0.95882 | 0.97988 | 83.68 |
| **interleaved10** | **79.17** | **0.96230** | **0.98150** | **84.54** |
| all (layers 0-29) | 83.33 | 0.95468 | 0.97708 | 83.72 |

Layer placement matters: interleaved10 has the best point estimates for Quality
Score and temporal stability among the tested reservoir methods, while
all-reservoir has the most motion but worst stability. With only 16 prompts and
one generation seed, these point estimates do not prove that the interleaved
layout is uniquely better than middle10 or late10.

## 5. Key Findings

1. **Layer gating works**: restricting reservoir to 10 of 30 layers recovers
   temporal stability while retaining most of the motion gain. This confirms
   the v157 hypothesis (doc 165 section 6.1).

2. **Interleaved placement is the predeclared primary and best point estimate**:
   distributed layer selection (1,4,7,10,13,16,19,22,25,28) has the strongest
   observed motion/quality combination. Middle10 and late10 passing the same
   screen prevents a unique-optimum claim.

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

The metric gate passes. The blind gate requires human review and is not yet
scored. The v158 layer-count budget sweep is implemented, but its GPU launch is
hard-blocked until the frozen v157 human gate passes. This preserves the
predeclared decision sequence rather than treating automatic metrics as a
replacement for review.

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
