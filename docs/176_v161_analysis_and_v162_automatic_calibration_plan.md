# 176: v161 Analysis and v162 Automatic Calibration Plan

Date: 2026-08-05

## 1. Decision

Do not generate another method yet. First evaluate the 96 videos already
produced by v161 and determine whether automatic metrics can reliably replace
most exploratory human review.

v162 has two goals:

1. Run prompt-correct VBench-Long core-9 on all six existing v161 methods.
2. Calibrate per-clip VBench signals against frozen human labels from v157,
   then test transfer on the independent v160 review.

The resulting review bundle contains either 3 or at most 8 videos. The old
v161 protocol required 12 videos in Wave 1 and up to 24 videos across two
waves.

## 2. What v161 Established

### 2.1 The mechanism is active

State-matched motion retrieval is not behaving like the fresh-reference
policy:

| Trace statistic | v161 result |
|---|---:|
| Reads | 640 |
| Multi-candidate reads | 445 |
| Non-newest selections | 273 |
| Negative-direction rejections | 135 |
| State/direction abstentions | 11 |
| Atomic/contract violations | 0 |
| Selected pair age p95 | 22 |
| Selected pair age max | 23 |

The mechanism gate is therefore `TRUE`. In 61.3% of multi-candidate reads,
v161 selected a compatible older pair instead of the newest pair.

### 2.2 Aggregate diagnostics are slightly positive, not decisive

| Method | Composite | DINO | First-last gap | Text | Background | Loop |
|---|---:|---:|---:|---:|---:|---:|
| v161 state-motion | 0.56378 | 0.83751 | 0.31469 | 0.28253 | 0.92642 | 0.08636 |
| v160 fresh reference | 0.56287 | 0.83416 | 0.34966 | 0.27898 | 0.92945 | 0.08620 |
| reservoir-4 reference | 0.55854 | 0.83698 | 0.33853 | 0.28306 | 0.92421 | 0.08902 |
| SF native | 0.56037 | 0.81574 | 0.38684 | 0.28252 | 0.91445 | 0.02742 |

Lower first-last gap is favorable; lower loop score is favorable. The state
method has the best aggregate composite and improves several history metrics,
but its margin over the fresh reference is only `+0.00091`.

Paired bootstrap comparisons across the 16 prompts do not establish a robust
advantage over the closest references:

- State vs fresh composite: `+0.00091`, 9/16 wins, 95% CI
  `[-0.00826, +0.01056]`.
- State vs reservoir composite: `+0.00524`, 8/16 wins, 95% CI
  `[-0.00383, +0.01621]`.
- State vs SF has significant gains in DINO, first-last gap, minimum CLIP
  alignment, and background drift, but a significant loop-score regression.

The correct conclusion is "promising but uncertain", not "better".

### 2.3 Safety remains unresolved

The automatic safety screen is `FLAGGED`:

- Prompt 7: background drift.
- Prompt 11: subject consistency drop and background drift.
- Prompt 12: late-motion collapse and temporal discontinuity.

The main mechanism-level concern is retrieval age. v160 fresh retrieval had a
pair-age p95 near 11 and max 13; v161 increases these to 22 and 23. State
compatibility may be preserving an old appearance/motion state at the cost of
freshness. This is a hypothesis, not yet a diagnosis.

## 3. v162 Experiment

### 3.1 Frozen inputs

- Source videos: v161 `full8`, 6 methods x 16 prompts = 96 videos.
- Duration: 30 seconds.
- Prompts: frozen diverse MovieBench-Qwen 16 subset used by v154-v161.
- No video generation and no new seed selection.
- VBench-Long: 15 eight-frame clips per video.

Methods:

1. `sf_native`
2. `ours_middle10_reservoir2_statemotionpair1`
3. `ours_middle10_reservoir2_freshmotionpair1_reference`
4. `ours_middle10_reservoir2_motionpair1_reference`
5. `ours_middle10_reservoir4_reference`
6. `ours_all_recent8_reference`

Core-9 dimensions:

1. subject consistency
2. background consistency
3. temporal flickering
4. motion smoothness
5. overall consistency
6. dynamic degree
7. aesthetic quality
8. imaging quality
9. temporal style

This is 54 method-dimension jobs. With four nodes and eight GPUs per node, the
static allocation is 14, 14, 13, and 13 jobs.

### 3.2 Metric-human calibration

Training labels are the 64 completed v157 blind reviews: 16 prompts x 4
methods. The four targets are identity, background, motion, and overall
preference.

Each video is represented by 18 frozen VBench features:

- mean of the eight non-duplicated core dimensions;
- minimum subject, background, and overall consistency;
- late-five minus early-five clip change for subject, background, flicker,
  motion smoothness, overall consistency, and dynamic degree;
- dynamic-degree standard deviation.

The model is a ridge regression on within-prompt method differences. Both
feature scaling and model fitting happen inside nested leave-one-prompt-out
cross-validation. Regularization is selected from
`{0.01, 0.1, 1, 10, 100}` without exposing the held-out prompt.

The final v157 model is then tested without refitting on all 24 completed v160
reviews: 8 prompts x 3 methods. The transfer set contributes 24 within-prompt
pair records per target. This transfer check matters because a model
that only memorizes v157 method artifacts is not useful for v161 triage.

The original v160 analyzer stored `prompt_indices` in sorted order but stored
the corresponding delta arrays in review-sheet insertion order. Aggregate
means were unaffected, but per-prompt calibration would have been wrong. v162
now reconstructs the frozen Wave 1 and Wave 2 insertion order from both review
sheets, checks the Wave 1 overlap exactly, and hashes both sheets. Future v160
reports write an explicit sorted `delta_prompt_order`.

Calibration passes only if all checks pass:

- v157 overall directional accuracy >= 0.60;
- v157 overall Spearman rho >= 0.20;
- every v157 target directional accuracy >= 0.55;
- v160 overall transfer accuracy >= 0.60;
- v160 motion transfer accuracy >= 0.55.

These are engineering thresholds selected before inspecting v162 predictions.

### 3.3 Automatic comparative gate

After calibration, predict state-motion minus fresh and state-motion minus
reservoir for every v161 prompt. A comparison is robust only when:

- the prompt-bootstrap 95% lower bound for predicted overall difference is
  greater than zero; and
- at least 10 of 16 prompts have a positive predicted difference.

Both reference comparisons must pass. Aggregate means alone cannot pass this
gate.

### 3.4 Minimal human review

The review selection is deterministic after the gates are computed.

**Mode A: `safety_only`**

Used only when calibration and both comparative tests pass. Review the primary
method on flagged prompts 7, 11, and 12: 3 videos total.

**Mode B: `sentinel_blind`**

Used otherwise. Blindly compare state-motion, fresh, and reservoir on:

- prompt 12, the highest-risk case;
- prompt 6, the largest metric-disagreement case.

This is 6 blind videos. Add primary-only safety checks for flagged prompts 7
and 11, which are not already represented: 8 videos total.

The automatic calibration chooses which videos need review, so this adaptive
sample is engineering evidence only. It cannot be reported as a fixed human
study in a paper. Severe corruption always retains a manual safety check.

## 4. Server Commands

Run the historical preflight before allocating GPUs. It verifies all v157
per-clip metrics and both completed review artifacts.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh history-preflight
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh prepare
```

Run split, preflight, and evaluation on each node, with `NODE_RANK` set to
`0`, `1`, `2`, and `3` respectively:

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v162_automatic_calibration.sh split
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v162_automatic_calibration.sh preflight
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v162_automatic_calibration.sh eval
```

Collect and calibrate on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh status
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh collect
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh calibrate
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh prepare-review
```

If status reports missing jobs, rank 0 can run:

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v162_automatic_calibration.sh resume-missing
```

Important outputs:

```text
runs/v162_automatic_calibration/
|-- metrics/vbench_core9_summary.{json,csv,md}
|-- analysis/v162_vbench_analysis.{json,md}
|-- analysis/v162_metric_human_calibration.{json,md}
`-- minimal_review/
    |-- review_manifest.json
    |-- reviewer/
    |   |-- REVIEW_INSTRUCTIONS.md
    |   |-- v162_review_sheet.csv
    |   `-- videos/                         # 3 or 8 links
    `-- private/v162_blind_key.json
```

## 5. Next Generation Decision

### Branch 1: calibrated comparison and safety review pass

Freeze v161 as the current candidate. Do not tune on the same 16 prompts.
Move to a held-out, capacity-matched evaluation and later a fixed human study.

### Branch 2: calibration works, but v161 is not robust or safety fails

The first v163 generation experiment should regularize retrieval age while
keeping state/direction compatibility:

- keep the v161 state and direction gates;
- retain a hard safety age cap;
- add either a soft recency penalty or a freshness band so a much older pair
  is selected only when its compatibility margin is substantial;
- target a selected-age p95 between the v160 value (about 11) and v161 value
  (22), rather than forcing either extreme.

Only one or two frozen recency settings should be generated on the same 16
prompts. Run the calibrated automatic screen first; manually inspect only the
winning sentinel pair and any severe flags.

### Branch 3: calibration does not transfer

Do not use VBench ranking to select a method. Keep automatic metrics for
failure localization only and use the fixed 6-video sentinel comparison for
engineering decisions. Improve the evaluator before launching another broad
cache search.

ABA/prompt-switch generation remains deferred until the single-prompt
retrieval/freshness tradeoff is resolved. It should not consume GPUs needed to
identify the primary long-video mechanism.

## 6. Claim Boundary

v162 is an evaluator-calibration experiment, not a method contribution. Its
purpose is to reduce repetitive manual inspection and prevent automatic metric
means from being mistaken for evidence of superiority. Any final paper still
requires a frozen held-out benchmark and a predeclared human evaluation.
