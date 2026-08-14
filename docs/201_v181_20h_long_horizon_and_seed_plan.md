# v181 20-Hour Long-Horizon and Seed-Replication Plan

## 1. Current state

The remote repository still has no new formal v178 or v179 metric result.
The uploaded generation grids remain incomplete:

| Experiment | Present | Missing |
|---|---:|---:|
| v178, six methods | 16/32 prompts per method | 16-31 |
| v179, two generated cells | 16/32 prompts per method | 8-23 |

v177 selected five Coverage-compatible heads under the frozen RCCP profile:
`L0H10`, `L8H6`, `L23H2`, `L5H3`, and `L6H6`. This is still a profiling
hypothesis, not a demonstrated generation method. The complete v178 causal
membership gate remains the first required decision.

The 20-hour budget is therefore allocated in this order:

1. finish v178 and compute its formal decision;
2. if v178 passes, run the fresh 128-prompt 30-second v180 confirmation;
3. run the new independent 128-prompt 60-second v181 confirmation;
4. only with remaining time, run the frozen 64-prompt second-seed replication;
5. do not rerun PF and do not spend this window on broad ABA evaluation.

v179 recovery may run concurrently with v178 generation because its missing
cells are already frozen. It becomes formal only if v178 passes.

## 2. Correctness fix required before collection

The previous paired-analysis stack had a hidden dynamic-grid bug. v178 and
v180 passed their `prompt_count` into `analyze_v174_paired_metrics.py`, but
the lower-level clip loader still used v165's global `16 prompts x 15 clips`
constants. Existing unit tests mocked that loader and did not expose the
problem. A complete 32- or 128-prompt VBench result could therefore fail at
the final collection step.

This revision makes both `prompt_count` and `clips_per_video` explicit through
the full loader stack. It also makes the v154/v175 VBench wrappers accept the
frozen duration from each comparison manifest. The resulting contracts are:

| Duration | Latent frames | Decoded frames | 2-second clips |
|---|---:|---:|---:|
| 30 seconds | 120 | 477 | 15 |
| 60 seconds | 240 | 957 | 30 |

Pull this revision before running any v178/v180 `collect` action.

## 3. Frozen method and controls

The v180/v181 comparison does not include PF. Pyramid-Forcing is only the
audited host runtime for applying per-head cache maps to the same Self-Forcing
checkpoint.

| Method | Head assignment | Purpose |
|---|---|---|
| `sf_native` | native Self-Forcing path | required backbone baseline |
| `rccp_matched` | 355 Recent + five frozen Coverage heads | proposed method |
| `all_recent` | 360 Recent heads | equal-budget local-memory control |

v180 additionally contains `all_coverage` as an all-head operator ablation.
It is not a PF baseline.

The operator budgets remain unchanged:

```text
Recent   = sink1 + recent8
Coverage = sink1 + reservoir4 + recent4
Episode  = sink1 + reservoir2 + coherent-motion-pair2 + recent4
```

Every route reads at most nine frame equivalents. v181 does not tune the map,
threshold, cache budget, or operator on its evaluation prompts.

## 4. New v181 scopes

v181 uses MovieGenVideoBench source entries 256-383. They are disjoint from
v177 calibration entries 0-127 and v180 evaluation entries 128-255. Exact
text overlap with both prior ranges is also rejected during preparation.

| Scope | Prompts | Length | Seed | Videos | Priority |
|---|---:|---:|---:|---:|---|
| `long60_seed0` | 128 | 60s | 0 + prompt index | 384 | required after v178 pass |
| `long60_seed10000_64` | first 64 of the same suite | 60s | 10000 + prompt index | 192 | optional replication |

The second scope shares prompts with the first scope intentionally. It tests
seed robustness, not additional prompt diversity. The two-seed analyzer
averages the two effects within each prompt before bootstrap inference.

For every 60-second video, VBench-Long is analyzed over:

- the full 30 clips;
- the first 15 clips (descriptive early-window effect);
- the last 15 clips (preregistered late-window effect).

The full and late-window quality, identity/background, and dynamic-degree
effects are tested against both SF and all-Recent. This prevents a positive
whole-video average from hiding a late collapse and prevents identity gains
obtained only by suppressing motion. Automatic decisions do not require
manual review. Each scope emits at most four late-window conflict cases.

## 5. Phase P0: close v178 and v179 gaps in parallel

Use relative node ranks within each two-node group. Both nodes in a group
must receive the same `PROMPT_INDICES` string.

On physical nodes 0-1, finish v178:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

PROMPT_INDICES="$(seq -s, 16 31)" NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh generate32
```

At the same time on physical nodes 2-3, finish the already prepared v179
exploratory cells:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

PROMPT_INDICES="$(seq -s, 8 23)" NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_head_attribution_32gpu.sh generate32
```

Do not rerun v179 `prepare` in its existing output directory.

After v178 generation completes, audit on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare
```

Split and evaluate on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh eval
```

Collect on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh decision
```

If `decision` exits with code 3, stop v180/v181. Do not tune on these 32
holdout prompts. The optional use of the remaining window is to preserve and
inspect v179 as exploratory attribution, then design a state-conditional
router on a new split.

## 6. Phase P1: v180 fresh 128 x 30 seconds

Only after v178 prints
`advance_rccp_membership_to_broader_generation`, prepare v180 on node 0:

```bash
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh preflight
```

Generate on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_rccp_fresh128_32gpu.sh generate128
```

Then audit and evaluate:

```bash
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v180_vbench_long.sh prepare

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_vbench_long.sh split
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v180_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v180_vbench_long.sh decision
```

v180 tests fresh 30-second content transfer. v179 is not a prerequisite and
must not be used to change the frozen five-head map midway through this run.

## 7. Phase P2: v181 fresh 128 x 60 seconds

Prepare once on node 0 after a passing v178:

```bash
NODE_RANK=0 bash scripts/run_v181_rccp_long_stress_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v181_rccp_long_stress_32gpu.sh preflight
```

Generate the required main scope on all four nodes:

```bash
SCOPE=long60_seed0 NUM_NODES=4 NODE_RANK=<0|1|2|3> \
  GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v181_rccp_long_stress_32gpu.sh generate
```

Audit on node 0:

```bash
NODE_RANK=0 bash scripts/run_v181_rccp_long_stress_32gpu.sh status
SCOPE=long60_seed0 NODE_RANK=0 \
  bash scripts/run_v181_rccp_long_stress_32gpu.sh audit
```

Prepare, split, and evaluate VBench-Long:

```bash
SCOPE=long60_seed0 NODE_RANK=0 bash scripts/run_v181_vbench_long.sh prepare

SCOPE=long60_seed0 NUM_NODES=4 NODE_RANK=<0|1|2|3> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v181_vbench_long.sh split
SCOPE=long60_seed0 NUM_NODES=4 NODE_RANK=<0|1|2|3> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v181_vbench_long.sh eval

SCOPE=long60_seed0 NODE_RANK=0 bash scripts/run_v181_vbench_long.sh status
SCOPE=long60_seed0 NODE_RANK=0 bash scripts/run_v181_vbench_long.sh collect
SCOPE=long60_seed0 NODE_RANK=0 bash scripts/run_v181_vbench_long.sh decision
```

Interrupted generation and metric jobs are resumable. Generation skips only
when both the shard marker and every assigned video exist. VBench uses the
existing `resume-missing` action on one node for isolated missing jobs.

## 8. Phase P3: optional second seed

Start this phase only if P2 generation is complete and enough wall time
remains to finish both generation and core-9 evaluation. Partial second-seed
metrics do not support a robustness claim.

Generate and audit:

```bash
SCOPE=long60_seed10000_64 NUM_NODES=4 NODE_RANK=<0|1|2|3> \
  GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v181_rccp_long_stress_32gpu.sh generate

SCOPE=long60_seed10000_64 NODE_RANK=0 \
  bash scripts/run_v181_rccp_long_stress_32gpu.sh audit
```

Evaluate with the same four-node commands, replacing the scope:

```bash
SCOPE=long60_seed10000_64 NODE_RANK=0 \
  bash scripts/run_v181_vbench_long.sh prepare
SCOPE=long60_seed10000_64 NUM_NODES=4 NODE_RANK=<0|1|2|3> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v181_vbench_long.sh split
SCOPE=long60_seed10000_64 NUM_NODES=4 NODE_RANK=<0|1|2|3> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v181_vbench_long.sh eval
SCOPE=long60_seed10000_64 NODE_RANK=0 \
  bash scripts/run_v181_vbench_long.sh collect
```

Combine the shared-prompt effects on node 0:

```bash
python scripts/analyze_v181_seed_replication.py \
  --main-scope-root runs/v181_rccp_long_stress/scopes/long60_seed0 \
  --replicate-scope-root runs/v181_rccp_long_stress/scopes/long60_seed10000_64 \
  --output runs/v181_rccp_long_stress/analysis/v181_seed_replication.json
```

Review at most the four cases in the combined report. This queue supersedes
the individual scope queues when both seeds are complete.

## 9. Resource accounting and stop rule

The generation workload, expressed as 30-second-video equivalents, is:

| Phase | New videos | 30s equivalents |
|---|---:|---:|
| v178 recovery | 96 x 30s | 96 |
| v179 recovery, concurrent with v178 | 32 x 30s | 32 |
| v180 | 512 x 30s | 512 |
| v181 main | 384 x 60s | 768 |
| v181 second seed | 192 x 60s | 384 |

Across 32 GPUs, the full list is 56 30-second equivalents per GPU, excluding
VBench. Use the measured duration of the first v180 method to decide whether
P3 fits. If fewer than four hours remain after P2 collection, skip P3 rather
than returning an incomplete second-seed grid.

The priority order is strict:

```text
v178 decision > v180 fresh30 > v181 fresh60 > v179 formal attribution
              > v181 second seed > all other ablations or ABA
```

If v179 metrics finish early, they refine whether the five-head effect is
distributed or L0H10-dominated. They do not replace the v180/v181 fresh-suite
tests and do not authorize post-hoc map changes within this campaign.

## 10. Artifacts to push

Keep raw videos and split clips on the server. Push the following small files:

```text
runs/v178_rccp_holdout_generation/{inputs,contracts,audits}/
runs/v178_rccp_holdout_generation/published_manifest.json
runs/v178_rccp_holdout_generation/metrics/vbench_core9_summary.{json,md,csv}
runs/v178_rccp_holdout_generation/analysis/v178_paired_metrics.{json,md}

runs/v180_rccp_fresh128/{inputs,contracts,audits}/
runs/v180_rccp_fresh128/published_manifest.json
runs/v180_rccp_fresh128/metrics/vbench_core9_summary.{json,md,csv}
runs/v180_rccp_fresh128/analysis/v180_fresh128_metrics.{json,md}

runs/v181_rccp_long_stress/inputs/
runs/v181_rccp_long_stress/scopes/*/{contracts,audits}/
runs/v181_rccp_long_stress/scopes/*/published_manifest.json
runs/v181_rccp_long_stress/scopes/*/metrics/vbench_core9_summary.{json,md,csv}
runs/v181_rccp_long_stress/scopes/*/analysis/v181_long_stress_metrics.{json,md}
runs/v181_rccp_long_stress/analysis/v181_seed_replication.{json,md}
```

If v178, v180, and the main v181 scope all pass, the project has a coherent
within-model paper evidence chain: representation-complete profiling,
untouched causal membership validation, fresh 30-second transfer, and fresh
60-second late-window confirmation. Cross-model transfer and scene switching
would still remain future required experiments before making those broader
claims.
