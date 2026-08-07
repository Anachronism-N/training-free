# v164 Direction Compatibility and Freshness Experiment

## 1. Why this experiment is next

The corrected v163 traces showed that absolute state cosine is almost
saturated (roughly 0.983-0.999), while the direction threshold rejects real
candidates. The strict selector still reads old pairs (selected-age p95 around
22), so it does not isolate whether its occasional gains come from meaningful
motion compatibility or from unstable abstention/read-budget changes.

v164 tests two narrow hypotheses:

1. **Direction compatibility is more informative than saturated absolute
   state similarity.** The query direction is the normalized descriptor delta
   from the first to the last frame of the current generated block. A stored
   pair direction is the normalized descriptor delta between its adjacent
   frames.
2. **A fixed freshness penalty can prevent stale recalls without suppressing
   the history-read budget.** It must change only ranking, not cache capacity,
   admission, or the number of recalled frames.

This is a generator-side development experiment, not held-out paper evidence.

## 2. Frozen cache and methods

The tested policy is enabled in the frozen Middle10 layers. Other layers use
`sink1 + recent8`.

| Component | Capacity/read rule |
|---|---|
| Sink | frame 0, one frame |
| Temporal reservoir | two frames |
| Motion archive | stores four adjacent pairs |
| Motion recall | reads exactly one adjacent pair when an age-eligible pair exists |
| Recent | four frames |
| Maximum read budget | nine full-frame equivalents |
| Admission freshness | stale replacement horizon 12 |
| Recall age | at most 24 frames |

The two new methods are:

1. `ours_middle10_reservoir2_directionmatch1`
   - Candidate gate: `direction_similarity >= 0.1`.
   - Ranking: highest direction similarity, then newest pair.
   - Absolute state similarity is logged but is not a gate and has zero score
     weight.
2. `ours_middle10_reservoir2_directionfresh1`
   - Uses the same gate and cache.
   - Ranking score:

     ```text
     direction_similarity - 0.25 * age / 24
     ```

If no candidate passes the direction gate, both methods read the newest
age-eligible adjacent pair. This fail-soft fallback is intentional: it keeps
the motion-history read budget equal, so a result cannot improve merely because
the method attended to less history.

The frozen six-method comparison is:

1. `sf_native` (reused from v161)
2. `ours_middle10_reservoir2_directionmatch1` (new)
3. `ours_middle10_reservoir2_directionfresh1` (new)
4. `ours_middle10_reservoir2_statemotionpair1_reference` (reused from v161)
5. `ours_middle10_reservoir4_reference` (reused from v161)
6. `ours_all_recent8_reference` (reused from v161)

There are 96 published videos but only 32 new generations. With four 8-GPU
nodes, every node receives 24 tasks: eight new and sixteen reused.

## 3. Correctness finding about old aliases

The audit found that the aliases `reservoir2_freshmotion4` and
`reservoir2_statemotion1_strict` enabled the motion-pair strategy but were
missing from the temporal-reservoir capacity set. Their old videos therefore
must not be described as the intended
`sink1 + reservoir2 + motion pair + recent4` cache.

The mapping and generic audit have now been fixed. v164 deliberately does not
reuse those ambiguous videos. It reuses only v161 methods whose cache ownership
and capacities were already explicit.

## 4. Debug and automatic checks

Every retrieval trace now records:

- all age-eligible candidates;
- state and direction similarities;
- direction-gate decisions;
- compatibility and final ranking scores;
- selected pair and newest eligible pair;
- fallback use and reason;
- whether the pair-read budget was preserved;
- whether freshness changed the direction-only choice;
- selected age and age gaps.

`analyze_v164_direction_freshness_trace.py` independently recomputes every
candidate score and expected selection. It fails on a non-atomic pair, a score
mismatch, a selected/read mismatch, age above 24, changed frozen parameters, or
loss of an available pair-read budget.

The automatic quality stage computes:

- low-motion coverage, late-motion ratio, longest low-motion run, temporal
  jump, appearance/flow outliers, and frame corruption diagnostics;
- DINO consistency, first-last gap, drift, smoothness, ArcFace when applicable,
  flicker, CLIP alignment, background consistency/drift, and loop score;
- optional prompt-correct VBench-Long core-9.

Do not inspect videos before these checks finish. The automatic report should
first identify whether any human review is informative.

## 5. Server commands

All four nodes must use the same commit, shared output directory, prompt file,
checkpoint, and environment.

```bash
git fetch origin
git checkout codex/v164-direction-freshness
git pull --ff-only

export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export V164_REUSE_V161_ROOT=${REPO_ROOT}/runs/v161_state_matched_motion_moviebench16/full8
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export NODE_RANK=0  # set 0, 1, 2, or 3 on the corresponding node

bash scripts/run_v164_direction_freshness_moviebench16.sh preflight
bash scripts/run_v164_direction_freshness_moviebench16.sh generate
```

After all four nodes finish, run on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v164_direction_freshness_moviebench16.sh audit
bash scripts/run_v164_automated_screen.sh all
```

The automatic screen uses six GPUs by default. Override with, for example,
`EVAL_GPUS=0,1,2,3,4,5`.

Only if the mechanism gate passes and the lightweight automatic screen is not
clearly negative, run VBench-Long core-9.

On node 0:

```bash
bash scripts/run_v164_vbench_long.sh prepare
```

On all four nodes, with their respective `NODE_RANK`:

```bash
bash scripts/run_v164_vbench_long.sh split
bash scripts/run_v164_vbench_long.sh preflight
bash scripts/run_v164_vbench_long.sh eval
```

After all VBench jobs finish, on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v164_vbench_long.sh collect
```

If a worker failed, use `resume-missing` on all nodes instead of restarting the
complete evaluation.

## 6. Decision logic

Evaluate in this order:

1. **Correctness:** both mechanism gates pass; zero score, atomic-pair, age,
   selected/read, and budget violations.
2. **Selector informativeness:** DirectionMatch has multi-candidate compatible
   reads. DirectionFresh changes at least one DirectionMatch ranking.
3. **Freshness effect:** DirectionFresh lowers selected-age p95 relative to
   DirectionMatch. If it does not, the fixed penalty is ineffective.
4. **No motion collapse:** late-motion ratio and dynamic degree must not fall
   materially below SF merely to improve consistency.
5. **Quality:** compare history, temporal, visual, identity, background, and
   prompt alignment rather than selecting by a single aggregate.

Interpretation branches:

- DirectionMatch better than the v161 state reference supports removing the
  saturated state term.
- DirectionFresh better than DirectionMatch while preserving consistency
  supports freshness-regularized direction recall.
- DirectionFresh reduces age but loses motion or consistency: keep
  DirectionMatch and reject the penalty.
- Neither new method improves the Pareto frontier: do not manually review all
  videos. Use the saved direction/age distributions to redesign the descriptor
  time scale or score; do not tune another arbitrary threshold first.

## 7. What to return for analysis

Push or provide these small files first:

```text
runs/v164_direction_freshness_moviebench16/full8/published_manifest.json
runs/v164_direction_freshness_moviebench16/full8/contracts/experiment.json
runs/v164_direction_freshness_moviebench16/full8/automated_screen/direction_freshness_trace.json
runs/v164_direction_freshness_moviebench16/full8/automated_screen/temporal_diagnostics.csv
runs/v164_direction_freshness_moviebench16/full8/automated_screen/comprehensive.json
runs/v164_direction_freshness_moviebench16/full8/automated_screen/automated_screen.json
runs/v164_direction_freshness_moviebench16/full8/metrics/vbench_core9_summary.json
runs/v164_direction_freshness_moviebench16/full8/analysis/v164_vbench_analysis.json
```

Also provide failed worker logs if any. Do not upload all videos or perform a
broad blind review unless the automatic analysis identifies a small unresolved
metric disagreement.

## 8. Local verification completed

- Python compilation passed for all new and modified Python entry points.
- Git Bash syntax checks passed for all three new shell entry points.
- Shared cache-audit and v111/v115/v119/v125/v159-v164 contract tests:
  `74 passed`.
- The focused v159-v164 run additionally reported `45 passed, 1 skipped`.
- The skipped test directly executes PyTorch motion-pair selection; this local
  machine has no usable PyTorch runtime. The real server trace audit is the
  required final runtime check.
