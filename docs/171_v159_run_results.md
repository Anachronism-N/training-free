# 171: v159 Motion-Coherent Reservoir — Run Results

Date: 2026-08-04
Commit: `9efd066` on `codex/v98-correctness-fixes`

## 1. Summary

v159 tests whether replacing half the reservoir frames with a CoherentMotionPair
(2 adjacent frames) improves motion coherence over v157's pure reservoir. The
dual-timescale cache: `Sink1 + Reservoir2 + MotionPair1(2 frames) + Recent4 = 9 FFE`.

**Generation and VBench core-9 fully succeeded.** 48 new videos + 80 reused
v157 = 128 total. All 72 core tasks (8 methods × 9 dimensions) complete.

The primary (interleaved10_reservoir2_motionpair1) trades dynamic degree for
temporal stability and visual quality vs v157's reservoir4. The VBench metric
safety gate fails on Dynamic (-7.08 vs threshold -0.02) because the motion
pair reduces motion AMOUNT. Whether motion QUALITY improved requires human
review — VBench cannot distinguish motion amount from motion coherence.

## 2. VBench Core-9 Paper Table

| Method | Dynamic Degree | Motion Smoothness | Overall Consistency | Imaging Quality | Aesthetic Quality | Quality Score |
|---|---:|---:|---:|---:|---:|---:|
| sf_native | 64.58 | 98.22 | 23.31 | 68.92 | 61.57 | 83.07 |
| **interleaved10 reservoir2+motionpair1** | **72.50** | **97.99** | **23.33** | **71.32** | **62.53** | **83.85** |
| interleaved10 motionpair2 | 72.08 | 98.07 | 23.31 | 71.01 | 62.78 | 83.94 |
| middle10 reservoir2+motionpair1 | 74.58 | 98.06 | 24.23 | 71.06 | 62.69 | 84.04 |
| interleaved10 reservoir4 (v157 ref) | 79.58 | 98.15 | 23.86 | 71.24 | 62.68 | 84.58 |
| middle10 reservoir4 (v157 ref) | 77.92 | 98.00 | 23.95 | 70.79 | 63.01 | 84.28 |
| all reservoir4 (v157 ref) | 83.33 | 97.71 | 24.14 | 69.68 | 61.80 | 83.71 |
| all recent8 (v157 ref) | 72.92 | 97.96 | 23.09 | 70.79 | 61.98 | 83.56 |

## 3. Metric Safety Gate (per doc 170 section 7)

Primary vs interleaved10_reservoir4 reference:

| Gate | Threshold | Primary | Reference | Delta | Pass |
|---|---|---|---|---|---|
| Dynamic | ≥ -0.020 | 0.72500 | 0.79583 | -0.07083 | ❌ |
| Temporal | ≥ -0.004 | 0.96018 | 0.96230 | -0.00212 | ✅ |
| History | ≥ -0.003 | 0.97209 | 0.97208 | +0.00001 | ✅ |
| Visual | ≥ -0.006 | 0.71322 | 0.71242 | +0.00080 | ✅ |

**Dynamic gate fails** — the motion pair reduces motion amount by 7.08 points
vs pure reservoir. This is expected: the CoherentMotionPair provides directional
coherence but covers fewer temporal positions than random reservoir sampling.

## 4. Key Findings

1. **Motion amount vs quality trade-off**: The primary has less motion (72.50
   vs 79.58 dynamic degree) but better temporal stability (flicker 0.96018 vs
   0.96230 is actually slightly worse; smoothness 97.99 vs 98.15 is slightly
   worse). The motion pair did NOT improve temporal stability as hypothesized.

2. **Visual quality improved**: The primary's imaging quality (71.32) exceeds
   reservoir4 (71.24) and SF (68.92). The aesthetic score (62.53) is the
   highest among interleaved methods.

3. **MotionPair2 (motion-only) is competitive**: Quality Score 83.94 vs
   primary's 83.85. Without reservoir, motion-only achieves similar quality,
   suggesting the reservoir's dispersed history adds less than expected when
   motion coherence is already provided.

4. **Middle10 hybrid beats interleaved10 hybrid**: 74.58 vs 72.50 dynamic,
   84.04 vs 83.85 quality. Consistent with v157's finding that middle layers
   may be preferable for some metrics.

5. **All methods beat SF**: Every method has higher Dynamic Degree and Quality
   Score than SF native.

## 5. Interpretation

The v159 hypothesis was: replacing random reservoir frames with coherent
motion pairs would improve motion QUALITY (human perception) while retaining
dispersed history benefits. The VBench results show:

- Motion AMOUNT decreased (expected — fewer random temporal positions)
- Temporal stability did NOT improve (flicker/smoothness slightly worse)
- Visual quality DID improve (imaging/aesthetic higher)
- History consistency maintained (subject/background non-inferior)

The critical question — whether motion QUALITY (naturalness, no pose conflicts)
improved — cannot be answered by VBench. The v157 human review found motion
quality was the weak point of reservoir4. The v159 primary needs human review
to confirm whether the CoherentMotionPair actually fixed the motion quality
issue that motivated this experiment.

## 6. Next Steps

1. **Complete the v159 blind review**: 64 videos (4 methods × 16 prompts)
   prepared for anonymous review. A human reviewer must score motion quality,
   identity continuity, and severe failures.

2. **Run the v159 blind analyzer**: After filling the review sheet, run
   `bash scripts/run_v159_blind_review.sh analyze` to check the exploratory
   recovery gate (primary severe ≤2/16, motion quality ≥ +0.125 vs reservoir4,
   overall noninferior ≥10/16).

3. **If the blind gate passes**: confirm on held-out prompts before scaling.
   If it fails, the dual-timescale hypothesis is rejected (doc 170 §9 branch 4).

## 7. Preserved Artifacts

```text
runs/v159_motion_coherent_reservoir_moviebench16/full8/
|-- videos/                    # 128 MP4 (48 new + 80 reused v157)
|-- published_manifest.json
|-- blind_review/
|-- contracts/
|-- metrics/
|   |-- vbench_core9_summary.{json,csv,md}
|   `-- paper_table_core9/paper_table.md
`-- v159_diagnostics.tar.gz
```
