# v119 sink3 Bug Fix and v120 Safe Launch

Date: 2026-07-27

Status: root cause audited; unsafe sink3 candidates retired; fail-closed
runtime guard and split v120 execution implemented.

## 1. Immediate conclusion

The polygon noise is confined to the two v119 sink3 cells:

- `legacy_v98_landmark4_motion1_sink3_extra`;
- `legacy_v98_landmark2_motion1_sink3_budget9`.

The three Retrieval cells are visually clean enough for continued
comparison. The bug does not invalidate:

- v116 `landmark_motion1`;
- v119 `landmark_retrieval1_age24`;
- v119 `landmark_retrieval_motion`;
- SF native or PF native.

Do not run or promote either sink3 cell.

## 2. Confirmed code-level failure

Both sink3 budget profiles changed the sink of **both** old-v98 roles:

```text
Supportive:  sink1 -> sink3
Suppressive: sink1 -> sink3
```

Therefore all 360 heads captured the complete three-frame opening block as
static sink. `_capture_sink_if_needed()` removed those frames from the dynamic
store, leaving no recent frames at `sync_t=0`.

The readout code applies one time-synchronised RoPE coordinate to every
decoupled static-sink token. Consequently, physical frames 0, 1, and 2 were
all read at one sink time instead of retaining their distinct temporal
coordinates. This is a stronger failure than a simple capacity imbalance:

```text
physical opening:  [frame0, frame1, frame2]
unsafe sink3 read: [sink@t, sink@t, sink@t]
dynamic recent:    []
```

The first malformed block then propagates through autoregressive generation.

## 3. Claim boundary

The observation does not prove that sink3 is intrinsically invalid:

- native PF uses sink3 for selected Anchor/Veil heads;
- the old-v98 Supportive set also contains 133 PF Wave heads;
- the old-v98 Suppressive set contains another 23 PF Wave heads;
- v119 changed all 360 heads simultaneously.

Thus the defensible conclusion is:

> All-head sink3 under the old-v98 304/56 explicit composition is unsafe for
> the three-frame warm start.

v119 did not isolate Supportive-only sink3 from Suppressive-only sink3, so
neither subgroup should be blamed without a separate one-video test. That
test is unnecessary for the current v120 main comparison.

## 4. Code fix

The new runtime guard checks every exclusive role composition before sink
capture:

```text
available_recent = min(recent_budget, opening_frames - sink_frames)
require available_recent > 0
```

An unsafe layout now raises a descriptive error before it changes the cache
or generates a video. Policy traces also report:

- whether sink time is synchronised;
- the mapped sink RoPE time;
- whether multiple physical sink frames collapse to that time;
- whether the opening recent window is starved.

Additional safeguards:

- the v119 sink3 cells are retained only as provenance and blocked from new
  runs;
- `landmark_motion1_sink3_budget9` was removed from v120 candidate aliases;
- sink1 Retrieval and MotionPair candidates remain unchanged.

## 5. What can run now

### 5.1 Baselines

Baselines may run while the ours branch is being reviewed:

```bash
export REPO_ROOT=/path/to/training-free
cd "$REPO_ROOT"
git pull --ff-only

export V120_BASELINE_ONLY=1
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

python scripts/run_v120_moviebench32_main.py generate --baseline-only
```

Use `NODE_RANK=1/2/3` on the other nodes. Then audit once on node 0:

```bash
python scripts/run_v120_moviebench32_main.py audit --baseline-only
```

This produces 32 SF and 32 PF videos under:

```text
runs/v120_moviebench32_main/baselines_seed0/
```

All nodes participating in one method set must use the same commit. If a
baseline contract was already frozen before this fix, finish every baseline
node at that exact commit rather than mixing revisions. The baseline and ours
manifests remain separate and record their own implementation hashes.

### 5.2 Ours without regenerating baselines

The recommended 32-prompt pair is:

1. `landmark_motion1`: previously clean and balanced v116 control;
2. `landmark_retrieval_motion`: clean v119 bounded Retrieval+Motion candidate.

```bash
unset V120_BASELINE_ONLY
export V120_OURS_ONLY=1
export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_motion1,landmark_retrieval_motion
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

python scripts/run_v120_moviebench32_main.py generate --ours-only
```

Repeat on nodes 1-3, then audit on node 0:

```bash
python scripts/run_v120_moviebench32_main.py audit --ours-only
```

This produces 64 ours videos without rerunning SF/PF:

```text
runs/v120_moviebench32_main/ours_only2_9b9ca1a08d27/
```

If compute must be reduced further, run only the established control:

```bash
export V120_CANDIDATES=landmark_motion1
```

## 6. VBench-Long without duplicate generation

Evaluate the baseline set:

```bash
export V120_SCOPE=baselines
bash scripts/run_v120_vbench_long.sh eval
bash scripts/run_v120_vbench_long.sh collect
```

Evaluate the ours set:

```bash
export V120_SCOPE=ours
export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_motion1,landmark_retrieval_motion
bash scripts/run_v120_vbench_long.sh eval
bash scripts/run_v120_vbench_long.sh collect
```

As with generation, run `eval` on all four nodes with their respective
`NODE_RANK`, then run `collect` once on node 0.

Merge the separately audited summaries:

```bash
python scripts/merge_v120_vbench_summaries.py \
  --baseline-summary \
  runs/v120_moviebench32_main/baselines_seed0/metrics/vbench_long_summary.json \
  --ours-summary \
  runs/v120_moviebench32_main/ours_only2_9b9ca1a08d27/metrics/vbench_long_summary.json \
  --output-root \
  runs/v120_moviebench32_main/comparison_motion_retrieval/metrics
```

The merged table contains SF, PF, `landmark_motion1`, and
`landmark_retrieval_motion`, while preserving the original source paths.

## 7. Current method decision

The sink experiment is now a negative implementation result, not a method
candidate. The useful decision remains:

```text
Supportive = sink1 + Landmark4 + recent4

Suppressive option A =
  sink1 + MotionPair1(two adjacent frames) + recent6

Suppressive option B =
  sink1 + age-bounded Retrieval1
  + MotionPair1(two adjacent frames) + recent5
```

v120 determines whether option B's richer retrieval/motion lifecycle improves
VBench-Long over the simpler option A and how both compare with PF.
