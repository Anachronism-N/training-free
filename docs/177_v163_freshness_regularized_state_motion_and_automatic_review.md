# 177: v163 Freshness-Regularized State-Motion Retrieval

Date: 2026-08-05

## 1. Result-Based Decision

The newly synchronized evidence changes the next experiment.

1. v160 fresh-motion retrieval is not promoted. Across the complete eight-prompt human review, it is tied with the old motion hybrid in overall preference, but trails reservoir-4 in overall preference (`-0.175`), identity (`-0.2125`), background (`-0.175`), and motion naturalness (`-0.125`). It also has two severe failures, so the exploratory recovery gate is false.
2. v161 state-matched retrieval is a real mechanism, not a no-op: 445 reads had multiple candidates, 273 selected a non-newest pair, 135 rejected an opposite-direction pair, and no atomic-pair contract failed.
3. v161 nevertheless reads history too aggressively. Its selected pair age has p95 `22` and maximum `23`, compared with approximately p95 `11` for v160 fresh retrieval. Prompts 7, 11, and 12 were automatically flagged.
4. The next hypothesis is therefore: **a compatible old motion state should be retrieved only when its compatibility benefit pays for its staleness**.

Do not manually review the unfilled v161 Wave 1 bundle. Complete v162 automatic calibration, then run v163.

## 2. v163 Methods

All methods use the frozen MovieBench-Qwen diverse 16 prompts, 30-second generation, the same seed, Middle10 placement, and the same attention-read budget.

### 2.1 Shared selected-layer cache

- sink: 1 frame;
- temporal reservoir: 2 frames;
- state-motion archive: 4 adjacent frame pairs stored, at most 1 pair read;
- recent: 4 frames;
- other layers: sink1 + recent8;
- maximum read: 9 full-frame equivalents;
- the explicit composition is the only dynamic-history owner.

The admission policy is unchanged from v160/v161: adjacent motion pairs enter an archive of four, the nominal admission age is 12, and stale replacement can bypass the motion quantile. State similarity must be at least `-0.25`; an available direction descriptor must have cosine similarity at least `0.0`. Reads remain atomic two-frame pairs.

### 2.2 Candidate A: hard read age 12

Method: `ours_middle10_reservoir2_stateage12motionpair1`

- only pairs with `current_t - pair_end_t <= 12` are eligible;
- among passing pairs, preserve the v161 direction/state/recency lexicographic order;
- purpose: isolate whether v161 failures come primarily from allowing age 13-24 reads.

### 2.3 Candidate B: recency-regularized compatibility

Method: `ours_middle10_reservoir2_statebalancedmotionpair1`

- retain the age-24 search horizon;
- after the same state and direction hard gates, define
  `compatibility = mean(state_similarity, direction_similarity)`;
- select by
  `compatibility - 0.25 * pair_age / 24`;
- purpose: retain genuinely useful older states while requiring a measurable compatibility margin over fresher candidates.

The coefficient `0.25` is one frozen exploratory value, not a threshold sweep. The two candidates differ only in read selection. Sink, recent, reservoir, archive admission, layer placement, and token budget are fixed.

### 2.4 Reused references

No reference video is regenerated:

- `sf_native`;
- v161 legacy state retrieval;
- v160 fresh-motion retrieval;
- reservoir-4.

The grid is six methods x 16 prompts = 96 videos, but only two methods x 16 prompts = 32 new generations. The other 64 are validated links to v161.

## 3. Debug and Correctness Contracts

Every state retrieval now logs:

- eligible count before and after age filtering;
- state and direction similarity for every candidate;
- raw compatibility and regularized selection score;
- selected, legacy-selected, and newest-passing pair;
- whether regularization changed the v161 choice;
- selected age, compatibility gain over newest, and age gap;
- selection mode and recency coefficient.

`analyze_v163_recency_trace.py` validates all 16 traces for both candidates. It rejects non-atomic reads, age escapes, stale configuration propagation, selected/read mismatches, wrong score argmax, archive overflow, and debug inconsistencies. A quality experiment must not proceed when the trace contract fails.

The v160 metric-human calibration also received a correctness fix. Its old complete report sorted `prompt_indices` but emitted per-prompt deltas in review-sheet insertion order. This did not change old aggregate means, but it would corrupt per-prompt transfer calibration. v162 now recovers the true order from both frozen review sheets and verifies the Wave 1 overlap before fitting.

## 4. Minimal-Review Protocol

Human review is conditional, not the default next step.

### Stage 1: mechanism

Both candidate trace contracts must pass. Candidate A must stay within age 12. Candidate B must actually change at least one legacy choice. At least one candidate must reduce selected-age p95 below the v161 value of 22.

### Stage 2: automatic quality and safety

Run temporal diagnostics, the comprehensive evaluator, and prompt-correct VBench-Long core-9. Apply the v162 ridge models only if they pass nested v157 validation and transfer to all 24 v160 human-reviewed videos.

For every reference, including SF native, a candidate must satisfy:

- positive predicted overall mean;
- positive overall result on at least 9/16 prompts;
- bootstrap lower bound no worse than `-0.10` on the `[-2,2]` human scale;
- predicted identity and background mean no worse than `-0.05`;
- positive predicted motion mean;
- no severe automatic corruption flag.

If no candidate passes, `manual_video_count=0`. Do not review more videos; inspect logs and automatic failure localization instead.

### Stage 3: optional review

If a candidate passes, compare the automatic winner with its strongest reference on only two prompts: its weakest predicted case and a typical case. This is four blind videos. Add at most two winner-only clips when non-severe automatic flags remain. Therefore review contains exactly 4-6 videos, never 12 or 24.

This adaptive review is engineering triage. A final paper claim still requires a separately frozen held-out benchmark and human protocol.

## 5. Server Commands

### 5.1 First complete v162 calibration

Use [176_v161_analysis_and_v162_automatic_calibration_plan.md](./176_v161_analysis_and_v162_automatic_calibration_plan.md). At minimum, verify the historical inputs after pulling this commit:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v162_automatic_calibration.sh history-preflight
```

Finish v162 `prepare`, `split`, `preflight`, `eval`, `collect`, and `calibrate` if the calibration report does not already exist. Do not use a stale pre-fix calibration JSON.

### 5.2 Generate v163 on four nodes / 32 GPUs

Run on every node with its own rank:

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v163_recency_regularized_state_motion_moviebench16.sh preflight

NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v163_recency_regularized_state_motion_moviebench16.sh generate
```

After all nodes finish, run on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_recency_regularized_state_motion_moviebench16.sh audit
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh mechanism
```

Expected generation count: 32 new, 64 reused.

### 5.3 Automatic metrics

Prepare on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh prepare
```

Split, preflight, and evaluate on all four nodes:

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v163_automatic_selection.sh split
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v163_automatic_selection.sh preflight
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v163_automatic_selection.sh eval
```

Run temporal diagnostics and comprehensive metrics on rank 0 after GPUs are free:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh temporal
NODE_RANK=0 NUM_NODES=4 EVAL_GPUS=0,1,2,3,4,5 \
  bash scripts/run_v163_automatic_selection.sh comprehensive
```

Collect, select, and conditionally package review on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh status
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh collect
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh select
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v163_automatic_selection.sh prepare-review
```

If VBench reports missing jobs:

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v163_automatic_selection.sh resume-missing
```

## 6. Outputs to Return

```text
runs/v163_recency_regularized_state_motion_moviebench16/full8/
|-- published_manifest.json
|-- contracts/experiment.json
|-- traces/*.policy.jsonl
|-- automated_selection/
|   |-- recency_trace.{json,md}
|   |-- temporal_diagnostics.csv
|   |-- comprehensive.json
|   |-- vbench_core9_summary.{json,csv,md}
|   |-- v163_vbench_analysis.{json,md}
|   `-- automatic_selection.{json,md}
`-- minimal_review/
    |-- review_manifest.json
    |-- private/v163_blind_key.json
    `-- reviewer/                       # 0 or 4-6 videos
```

The first files needed for the next analysis are `recency_trace.json`, `automatic_selection.json`, VBench summary/analysis, temporal diagnostics, and comprehensive results. Return the review sheet only when `manual_video_count` is nonzero.

## 7. Decision Tree

1. Trace contract fails: implementation/configuration issue; do not interpret video metrics.
2. Trace passes but neither method reduces age: increase freshness pressure only after inspecting score/age logs; do not sweep blindly.
3. Freshness passes but automatic quality gate has no winner: no human review; analyze prompt-localized identity, motion, and corruption failures.
4. A winner passes and 4-6 video review passes: freeze the method and move to a held-out 128-prompt generation study.
5. A winner passes metrics but fails minimal review: improve evaluator calibration and inspect the selected sentinel failures before changing cache capacity.

ABA/prompt-switch generation remains deferred until this single-prompt history-selection tradeoff is resolved.
