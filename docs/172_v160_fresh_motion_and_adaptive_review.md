# 172: v160 Fresh-Motion Recovery and Adaptive Review

Date: 2026-08-04

## 1. Latest repository result

The repository was updated to commit `bc1e331`, which contains the completed
v159 generation and VBench core-9 results. v159 did not crash and its cache
contract passed, but the proposed motion mechanism did not meet its intended
goal:

| Method | Dynamic Degree | Motion Smoothness | Overall Consistency | Imaging Quality | Quality Score |
|---|---:|---:|---:|---:|---:|
| SF native | 64.58 | 98.22 | 23.31 | 68.92 | 83.07 |
| v159 interleaved hybrid | 72.50 | 97.99 | 23.33 | 71.32 | 83.85 |
| v159 Middle10 hybrid | 74.58 | 98.06 | 24.23 | 71.06 | 84.04 |
| v159 Middle10 reservoir4 | 77.92 | 98.00 | 23.95 | 70.79 | 84.28 |

The Middle10 hybrid is a better starting point than the interleaved hybrid,
but it still has less motion than reservoir4. Motion smoothness/flicker also do
not support the claim that the coherent pair improved motion quality.

## 2. Trace diagnosis

The new reproducible analyzer is:

```bash
python scripts/analyze_v159_motion_pair_trace.py
```

It reads the committed v159 diagnostics archive and freezes:

```text
docs/results/v159_motion_coherent_reservoir_moviebench16/
  v159_motion_pair_trace_diagnosis.json
  v159_motion_pair_trace_diagnosis.md
```

For one representative selected layer/head per prompt:

| Route | Accepted updates / 39 | Rejected | Mean per-prompt pair-age p95 | Maximum age |
|---|---:|---:|---:|---:|
| Interleaved hybrid | 6.125 | 32.875 | 34.49 | 61 |
| MotionPair2 | 6.688 | 32.312 | 71.86 | 115 |
| Middle10 hybrid | 5.938 | 33.062 | 36.43 | 73 |

The motion strategy was active. The main rejection reason was the rolling
motion-quantile gate. The old `max_pair_age=24` only relaxed the replacement
gate; a stale candidate could still be rejected by the quantile gate. It was
therefore not an actual freshness bound.

## 3. v160 isolated change

v160 tests exactly one new method:

```text
Middle10 selected layers:
  Sink1 + Reservoir2 + FreshCoherentMotionPair1 + Recent4 = at most 9 FFE

Other layers:
  Sink1 + Recent8 = 9 FFE
```

`FreshCoherentMotionPair1` keeps the v159 semantic definition and changes only
the stale replacement behavior:

1. The pair must still consist of adjacent frames.
2. Motion must remain positive.
3. The semantic-coherence floor still applies.
4. Pair-spacing constraints still apply.
5. Normal updates still use motion quantile `0.70` and replacement margin
   `0.05`.
6. When the retained pair is at least 12 physical frames older than an eligible
   candidate, that candidate may bypass only the motion-quantile gate.

This is a freshness-aware recovery rule, not an unconditional periodic update.
The trace now records `stale_quantile_bypass`,
`stale_refresh_bypass_quantile`, `victim_stale`, `max_pair_age`, the decision
reason, pair IDs, accepted/rejected counts, and the full read-budget audit.

Only 16 videos are newly generated. The following 64 are linked from v159:

- SF native
- v159 Middle10 reservoir2 + motionpair1
- v159 Middle10 reservoir4
- all-layer recent8

## 4. Generation commands

Run the same preflight on all four nodes with the corresponding rank:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v160_fresh_motion_moviebench16.sh preflight
```

Replace `NODE_RANK=0` with `1`, `2`, and `3` on the other nodes. If all four
preflights pass, run:

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v160_fresh_motion_moviebench16.sh generate
```

After all nodes complete, run once from node 0:

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v160_fresh_motion_moviebench16.sh audit

bash scripts/run_v160_fresh_motion_moviebench16.sh package
```

If v159 lives elsewhere, set:

```bash
export V160_REUSE_V159_ROOT=/absolute/path/to/v159_motion_coherent_reservoir_moviebench16/full8
```

## 5. Automated diagnostic screen

After audit, run the complete automatic screen on one node. Five GPUs are used
in parallel for the comprehensive metrics; temporal diagnostics use CPU
workers.

```bash
EVAL_GPUS=0,1,2,3,4 TEMPORAL_WORKERS=8 \
  bash scripts/run_v160_automated_screen.sh all
```

The screen computes:

- a policy-trace gate that requires the new freshness path to change at least
  one accepted update, and separately validates/counts true below-quantile
  bypasses and 12--23-frame early stale refreshes;
- decode, duration, luminance, contrast, and edge-density failure signals;
- Farneback motion coverage, long low-motion runs, final-quarter motion ratio,
  appearance jumps, flow acceleration, and outliers;
- DINO consistency/drift, RAFT acceleration, ArcFace where detectable, LPIPS,
  CLIP alignment, background consistency, and loop diagnostics;
- paired deltas for the same prompt and seed, never an unpaired leaderboard.

Outputs are written under:

```text
runs/v160_fresh_motion_moviebench16/full8/automated_screen/
  temporal_diagnostics.csv
  comprehensive.json
  fresh_motion_trace.json
  fresh_motion_trace.md
  automated_screen.json
  automated_screen.md
  review_plan.json
```

The automatic safety result is a failure/triage screen only. It cannot promote
a method and must not be reported as an unbiased paper metric.

## 6. Adaptive blind review

The `all` command also prepares Wave 1:

```text
runs/v160_fresh_motion_moviebench16/full8/adaptive_review/wave1/reviewer/
```

Wave 1 contains 12 videos: three methods on four automatically selected
prompts. The four prompts represent:

1. highest automatic risk;
2. largest predicted gain;
3. largest metric disagreement;
4. a typical case.

The reviewer separately scores motion amount and motion naturalness, avoiding
the VBench Dynamic Degree ambiguity. After completing the CSV:

```bash
bash scripts/run_v160_automated_screen.sh analyze-wave1
```

Read `adaptive_review_analysis.json`:

- `exploratory_pass_stop_after_wave1`: stop; do not review more exploratory
  videos.
- `exploratory_reject_stop_after_wave1`: reject the freshness change.
- `continue_wave2`: prepare only the second frozen 12-video wave.

```bash
bash scripts/run_v160_automated_screen.sh review-wave2
# Fill the Wave-2 CSV.
bash scripts/run_v160_automated_screen.sh analyze-wave2
```

Thus normal review load is 12 videos and the maximum is 24, instead of blindly
reviewing 64 videos every iteration.

## 7. Decision branches

### Automatic corruption/freeze flag

Inspect the flagged prompt videos and corresponding policy/video logs first.
Do not interpret metric differences until implementation failure is excluded.

### Human rejection

Reject v160's stale-refresh change. v159 remains useful evidence that motion
pair selection alone does not solve motion quality; do not scale this route.

### Exploratory pass

The next run must use held-out prompts with a fixed, non-adaptive protocol.
Only then run standard VBench-Long and a prespecified human comparison. The
adaptive 4/8-prompt review cannot serve as paper evidence because prompt
selection observed diagnostic results.

## 8. Important claim boundary

v160 is a mechanism-recovery experiment, not the final method. A pass means the
freshness mechanism is worth held-out validation. It does not establish that
the method outperforms SF, PF, or other long-video baselines, and no new paper
claim should be made from the adaptive sample alone.
