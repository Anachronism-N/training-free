# v180 Fresh128 RCCP Confirmation

## 1. Repository state and decision boundary

Remote `main` and the working branch both stop at commit `dd27f1b1`. No new
formal v178 or v179 metric result has been uploaded.

The available evidence is therefore unchanged:

- v177 strict profiling is valid and selected five Coverage-compatible heads:
  `L0H10`, `L8H6`, `L23H2`, `L5H3`, and `L6H6`;
- v178 has only prompts 0-15 for each of six methods, so its causal membership
  gate is not available;
- v179 has prompts 0-7 and 24-31 for its two generated cells, so its factorial
  attribution is also incomplete;
- no partial result may be used to claim that the five-head map improves video
  generation.

Finish v178 before preparing v180. The v180 preparer revalidates the complete
v178 media contract, paired metrics, passing decision, and all SHA-bound input
artifacts. It fails closed when any of them is missing or changed.

## 2. Why v180 is the next confirmation

v178 asks a causal question on 32 untouched prompts: did RCCP choose better
head membership than all-Recent and layer/count-matched hard negatives?

If that gate passes, v180 asks a separate transfer question:

> Does the exact frozen five-head map improve 30-second generation on 128 new
> prompt contents, relative to native Self-Forcing and equal-budget Recent,
> without obtaining identity consistency by suppressing motion?

The prompt suite is assembled from MovieGenVideoBench source entries 128-255.
v177 used Qwen rewrites corresponding to source entries 0-127. The two source
index ranges are disjoint, and v180 additionally rejects any exact text overlap.
The 128 evaluation prompts are never used to choose heads, thresholds, cache
budgets, or operators.

v179 is useful attribution but is not a prerequisite for v180. The v180 method
remains the exact five-head map validated by v178; it is not selected post hoc
from v179's top-1/remainder result.

## 3. Frozen methods

PF is not evaluated. Pyramid-Forcing is only the host repository containing
the audited adaptive-cache runtime for the three RCCP map methods.

| Method | Runtime | Purpose |
|---|---|---|
| `sf_native` | Self-Forcing repository, no head map | required native baseline |
| `rccp_matched` | 355 Recent + five frozen Coverage heads | proposed method |
| `all_recent` | 360 Recent heads | equal-budget local-memory control |
| `all_coverage` | 360 Coverage heads | all-head nonlocal-memory ablation |

The cache operators are unchanged from v177-v179:

```text
Recent   = sink1 + recent8
Coverage = sink1 + reservoir4 + recent4
Episode  = sink1 + reservoir2 + coherent-motion-pair2 + recent4
```

Every read route is at most 9 frame equivalents. v180 does not add a new cache
trick or tune the five selected heads. Its purpose is independent confirmation.

## 4. Automated correctness and decision gates

The generation audit requires:

- 4 methods x 128 prompts = 512 videos;
- exactly 477 decoded frames, 16 fps, and 832x480 for every video;
- exactly 32 shard logs per method and no runtime failure signature;
- route counts `355/5/0`, `360/0/0`, and `0/360/0` for matched, Recent,
  and Coverage respectively;
- no RCCP route or PyramidKV head map in native SF logs;
- non-identical all-Recent and mapped trajectories at the whole-grid level.

VBench-Long evaluates the same core nine dimensions used by v178. Pairwise
analysis uses prompt-level deltas, bootstrap confidence intervals, sign tests,
and one global BH correction over six preregistered tests:

```text
rccp_matched - sf_native:  quality, identity/background, dynamic degree
rccp_matched - all_recent: quality, identity/background, dynamic degree
```

The analyzer reports two explicit gates:

- `quality_identity_gate`: both quality and identity/background beat both
  primary controls with positive CI, q <= 0.10, and win fraction >= 0.55;
- `identity_motion_gate`: identity/background and dynamic degree satisfy the
  same requirements against both controls.

It also requires dynamic mean regression no worse than `-0.02` for the broad
quality decision. The possible machine decisions are:

| Decision | Interpretation |
|---|---|
| `fresh128_quality_identity_motion_confirmed` | broad quality and the identity-motion Pareto claim both pass |
| `fresh128_quality_identity_confirmed` | broad quality passes with motion non-regression |
| `fresh128_identity_motion_confirmed` | narrower identity-plus-motion contribution passes |
| `fresh128_directional_only` | means point in the desired direction, uncertainty does not pass |
| `fresh128_rccp_not_confirmed` | do not promote the static RCCP method |

No broad blind review is required. The analyzer emits at most six cases where
identity and motion deltas disagree most strongly. Review those only after the
automatic decision, and never use them to change the frozen gate.

## 5. Server execution

### 5.1 Finish the current v178/v179 gaps

Use the exact recovery commands in
`docs/199_v179_partial_recovery_and_factorial_validation.md`. The currently
missing ranges are v178 `16-31` and v179 `8-23`.

Run the full v178 audit, core-9 evaluation, collection, and decision first:

```bash
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh split
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh decision
```

Do not continue if `decision` exits with code 3.

### 5.2 Prepare v180 after a passing v178

On node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh preflight
```

Run generation on four 8-GPU nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_rccp_fresh128_32gpu.sh generate128
```

The four methods run in sequence and each GPU generates four prompts per
method without reloading the model between prompts. To obtain the primary
three methods first, all four nodes may use:

```bash
METHODS=sf_native,rccp_matched,all_recent \
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_rccp_fresh128_32gpu.sh generate128
```

Then use the otherwise idle budget for the ablation:

```bash
METHODS=all_coverage \
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_rccp_fresh128_32gpu.sh generate128
```

Audit on node 0 only after all four methods are complete:

```bash
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh status
NODE_RANK=0 bash scripts/run_v180_rccp_fresh128_32gpu.sh audit
```

### 5.3 VBench-Long and automatic decision

```bash
NODE_RANK=0 bash scripts/run_v180_vbench_long.sh prepare

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_vbench_long.sh split
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v180_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v180_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v180_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v180_vbench_long.sh decision
```

Use `resume-missing` with one node only if metric jobs were interrupted.

## 6. Artifacts to return

Keep raw videos and split clips on the server. Push these small artifacts:

```text
runs/v180_rccp_fresh128/inputs/
runs/v180_rccp_fresh128/contracts/
runs/v180_rccp_fresh128/audits/
runs/v180_rccp_fresh128/published_manifest.json
runs/v180_rccp_fresh128/metrics/vbench_core9_summary.{json,md,csv}
runs/v180_rccp_fresh128/analysis/v180_fresh128_metrics.{json,md}
```

If v178 fails, do not run v180. That result rejects the static five-head
generation rule; the next experiment must profile an online/state-conditional
operator decision on a new split rather than retune v177 thresholds.
