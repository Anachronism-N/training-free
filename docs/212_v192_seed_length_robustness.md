# v192: Conditional Seed and Length Robustness

## 1. Current repository conclusion

At commit `cb15f859`, v191 code is complete but no new v191 generation, VBench,
or temporal result is present in the repository. Therefore there is currently no
new evidence that can promote or reject the frozen Head x Denoising-Phase method.
v192 is intentionally impossible to prepare unless the SHA-bound v191 decision
contains all passing confirmation gates and the exact recommendation:

```text
freeze_head_phase_method_for_seed_length_and_cross_model_replication
```

This prevents a failed v191 method from consuming another large experiment budget.

## 2. Question answered by v192

v191 asks whether the selected route works on 128 prompts excluded from profiling
and causal screening. A single prompt suite, seed, and 30-second duration is not
enough to claim robustness. v192 freezes the exact v191 method and changes one
factor at a time:

| Scope | Prompts | Seed | Latent frames | Decoded duration | Changed factor |
|---|---:|---:|---:|---:|---|
| `seed2026_30s_128` | 128 | 2026 | 120 | 29.8125 s | seed only |
| `long60_seed10000_32` | 32 | 10000 | 240 | 59.8125 s | duration only |

The long scope uses v191 prompt positions `0,4,...,124`, corresponding to source
indices `128,132,...,252`. This systematic subset is frozen before v192 metrics
and is not selected using v191 prompt-level performance.

Both scopes contain exactly three methods:

1. `sf_native`: native Self-Forcing.
2. `all_recent`: equal-budget local cache control, `sink1 + recent8`.
3. `head_phase_joint`: the exact v191-frozen Head x Phase route.

There is no PF-native baseline and no ABA experiment in v192. The
`third_party/Pyramid-Forcing` directory is only the code host for the audited
cache runtime; PF head classes and the PF baseline are not evaluated here.

## 3. Frozen cache and classification contract

v192 does not refit, merge, or relabel heads. It copies and hashes the exact v191
map and bank:

- clean denoising update: every head reads Recent;
- noisy denoising updates: each `(call, layer, head)` follows the frozen v191 map;
- Recent read: `sink1 + recent8`;
- Coverage read: `sink1 + structured-middle4 + recent4`;
- dynamic RoPE is enabled;
- maximum read budget is 9 frame equivalents for both cache methods.

The only manipulated variables are seed and output length. This makes a failure
diagnosable: seed failure indicates unstable efficacy, while long-scope failure
indicates long-horizon decay rather than a changed classifier or cache budget.

## 4. Compute budget

The generation grid contains:

- seed replication: `128 x 3 = 384` 30-second videos;
- long persistence: `32 x 3 = 96` 60-second videos;
- total: 480 videos, equivalent to 576 30-second generations.

With 4 nodes x 8 GPUs, each GPU receives 18 30-second-equivalent generations.
Methods are rotated across nodes to reduce fixed method/order coupling. Only rank
0 emits a full schedule/readout trace for each cache method; all shards retain
runtime logs.

## 5. Required execution order

First finish and collect v191 on node 0:

```bash
git pull
NODE_RANK=0 bash scripts/run_v191_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v191_vbench_long.sh decision
```

Do not proceed unless the printed recommendation is exactly the required v191
recommendation above. Then freeze v192 inputs:

```bash
NODE_RANK=0 bash scripts/run_v192_head_phase_robustness_32gpu.sh prepare
```

Run one smoke video per method before the full grid:

```bash
SCOPE=seed2026_30s_128 NODE_RANK=0 \
  bash scripts/run_v192_head_phase_robustness_32gpu.sh smoke
SCOPE=seed2026_30s_128 NODE_RANK=0 \
  bash scripts/run_v192_head_phase_robustness_32gpu.sh audit-smoke
```

For each scope, launch the same command on all four nodes with `NODE_RANK=0,1,2,3`:

```bash
SCOPE=seed2026_30s_128 NODE_RANK=<0..3> \
  bash scripts/run_v192_head_phase_robustness_32gpu.sh generate

SCOPE=long60_seed10000_32 NODE_RANK=<0..3> \
  bash scripts/run_v192_head_phase_robustness_32gpu.sh generate
```

Audit on node 0 after all nodes finish:

```bash
NODE_RANK=0 bash scripts/run_v192_head_phase_robustness_32gpu.sh status
SCOPE=seed2026_30s_128 NODE_RANK=0 \
  bash scripts/run_v192_head_phase_robustness_32gpu.sh audit
SCOPE=long60_seed10000_32 NODE_RANK=0 \
  bash scripts/run_v192_head_phase_robustness_32gpu.sh audit
```

## 6. VBench-Long and temporal evaluation

For each scope, prepare on node 0, split and evaluate on all four nodes, then
collect on node 0:

```bash
SCOPE=<scope> NODE_RANK=0 bash scripts/run_v192_vbench_long.sh prepare
SCOPE=<scope> NODE_RANK=<0..3> bash scripts/run_v192_vbench_long.sh split
SCOPE=<scope> NODE_RANK=<0..3> bash scripts/run_v192_vbench_long.sh eval
SCOPE=<scope> NODE_RANK=0 bash scripts/run_v192_vbench_long.sh collect
```

Use `resume-missing` only on one node:

```bash
SCOPE=<scope> NODE_RANK=0 NUM_NODES=1 \
  bash scripts/run_v192_vbench_long.sh resume-missing
```

After both scope reports exist:

```bash
NODE_RANK=0 bash scripts/run_v192_vbench_long.sh decision
```

The split path reads the frozen duration from the comparison manifest. It creates
15 clips per 30-second video and 30 clips per 60-second video.

## 7. Decision logic

### 7.1 New-seed scope

The candidate must:

- be confidence-bound non-inferior to `all_recent` and `sf_native` on quality,
  identity/background, temporal mechanics, and informative motion;
- reproduce at least one valid metric whose v191 paired CI was already positive;
- pass automatic temporal safety against both controls.

Dynamic Degree at an all-one ceiling is only a non-regression check and cannot
be promoted as a motion improvement.

### 7.2 Long scope

VBench is analyzed over full `[0,30)`, early `[0,15)`, and late `[15,30)` clip
windows. The candidate must:

- be non-inferior to both controls over the full and late windows;
- retain at least one positive mean direction against the equal-budget control;
- keep late-minus-early effect decay above the frozen persistence margins;
- pass automatic full-video temporal safety against both controls.

The persistence margins are development tolerances, not universal equivalence
thresholds: quality `-0.25`, identity `-0.0025`, informative motion `-0.04`, and
temporal mechanics `-0.004`.

### 7.3 Combined decision

For each v191-positive target metric, the seed-10000 and seed-2026 paired deltas
are averaged per prompt, then bootstrapped over the 128 prompts. The final method
advances only if:

1. the new-seed scope passes;
2. at least one target has positive means at both seeds and positive pooled CI;
3. the 60-second scope passes.

The passing recommendation is:

```text
freeze_within_model_head_phase_method_for_cross_model_transfer
```

Only after all automatic gates pass is manual review requested. The final queue
is capped at four prompt cases: two high-information seed cases and two
long-horizon cases. Each case contains the three aligned method videos. A failed
automatic decision requires no broad video review.

## 8. Expected artifacts

```text
runs/v192_head_phase_robustness/
  inputs/manifest.json
  scopes/seed2026_30s_128/
    published_manifest.json
    metrics/vbench_core9_summary.json
    metrics/temporal_diagnostics.csv
    analysis/v192_scope_analysis.json
  scopes/long60_seed10000_32/
    published_manifest.json
    metrics/vbench_core9_summary.json
    metrics/temporal_diagnostics.csv
    analysis/v192_scope_analysis.json
  analysis/v192_head_phase_robustness.json
```

If v192 passes, the next justified experiment is cross-model transfer with a
separately audited runtime contract. If seed replication fails, do not tune on
these 128 prompts. If only the long scope fails, return to the memory update and
long-horizon retrieval mechanism while keeping the classifier frozen.
