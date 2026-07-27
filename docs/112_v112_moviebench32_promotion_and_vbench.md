# v112 MovieGenBench-32 Promotion and VBench-Long

Date: 2026-07-27

Status: implemented but gated on the v111 one-video review.

## 1. Purpose

v112 promotes exactly one v111 candidate to 32 diverse 30-second prompts.
It does not sweep all eight one-video candidates. This prevents spending
compute on policies that have not passed artifact and visual-quality review.

The full suite contains four methods and 128 generated videos:

```text
selected role-conditioned candidate
all-Recent8 binary control
all-Landmark4 role-neutral control
all-MotionPair2 role-neutral control
```

All methods use the same old-v98 304/56 map, prompt file, seed, checkpoint,
decoded-frame contract, nine-frame read budget, and exclusive cache owner.
Existing SF, PF native, and compatible PF-binary results should be reused in
the final table after prompt/seed/checkpoint contracts are verified.

## 2. Candidate Keys

Choose the key that won v111:

```text
support_landmark_suppress_recent
support_hybrid_suppress_recent
support_recent_suppress_motion
support_landmark_suppress_motion
support_hybrid_suppress_motion
```

Do not select a candidate from metric intuition before recording the blind
v111 review.

## 3. Four-Node Generation

Set the same candidate and shared output root on all four nodes:

```bash
cd /path/to/training-free
git pull

export CANDIDATE=support_landmark_suppress_motion
export V111_PROMOTION_APPROVED=1
export PF_CHECKPOINT="$PWD/third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"
export OUT_ROOT="$PWD/runs/v112_role_event_cache_32prompt/$CANDIDATE"
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7

NODE_RANK=<0|1|2|3> \
python scripts/run_v112_role_event_cache_32prompt.py generate \
  --candidate "$CANDIDATE" --suite full
```

The deterministic task partition contains 32 tasks per node and four
sequential videos per GPU. Each prompt is reseeded independently with seed 0.
Completed tasks resume from frozen markers; mixed code, maps, prompts, or
commands are rejected.

After every node completes, audit and freeze VBench-ready directories once:

```bash
NUM_NODES=4 NODE_RANK=0 \
python scripts/run_v112_role_event_cache_32prompt.py audit \
  --candidate "$CANDIDATE" --suite full
```

The audit requires exactly `000000.mp4` through `000031.mp4` for each method
and writes:

```text
$OUT_ROOT/published_manifest.json
$OUT_ROOT/published/<method>/*.mp4
```

## 4. VBench-Long

Run one method per node. Keep the same `CANDIDATE`, `RUN_ROOT`, and
`V111_PROMOTION_APPROVED=1`:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 GPU_LIST=0 \
bash scripts/run_v112_vbench_long.sh eval "$CANDIDATE"
```

After all four nodes finish:

```bash
NODE_RANK=0 NUM_NODES=4 \
bash scripts/run_v112_vbench_long.sh collect "$CANDIDATE"
```

The frozen dimensions are:

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
motion_smoothness
dynamic_degree
```

The collector writes JSON, CSV, and Markdown summaries under
`$OUT_ROOT/metrics/`.

## 5. Required Feedback

Return or push the small artifacts below; do not push videos:

```text
contracts/
configs/
status/
diagnostics/
traces/
logs/
published_manifest.json
metrics/vbench_long_summary.json
metrics/vbench_long_summary.csv
metrics/vbench_long_summary.md
```

Also provide:

- the v111 blind scorecard and selected candidate;
- representative failure timestamps from v112 human review;
- VBench-Long per-method `results.json`;
- `diagnostics/role_event_summary.json`;
- any log containing `PyramidKVRoleEventTraceError`,
  `cache_contract_pass=false`, traceback, OOM, NaN, or polygon noise.

## 6. Interpretation

A method-selection result requires all three:

1. The candidate beats all-Recent, showing that event memory matters.
2. It beats or meaningfully trades off against role-neutral Landmark/Motion
   controls, showing that the 304/56 coupling matters.
3. It remains competitive with reused PF while being clearly distinct in
   classifier and cache mechanism.

If only all-Landmark wins, the binary routing contribution is unsupported.
If only all-Motion wins, the long-term identity story is unsupported. If the
candidate improves consistency but collapses dynamic degree, report the
tradeoff and continue cache tuning rather than declaring a final method.

Even a successful v112 result is a cache-mechanism result under a diagnostic
304/56 partition. Before the paper main table, repeat the selected route with
a shift-invariant offline binary classifier and its threshold/random/inverted
controls. Do not present the old absolute-sign map as a learned or invariant
head discovery method.
