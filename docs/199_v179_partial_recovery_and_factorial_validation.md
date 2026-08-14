# v179 Partial Recovery and Factorial Validation

## 1. Pulled state at `34f80e50`

The commit title says that v179 generation is complete, but the uploaded
artifacts are incomplete.

| Experiment | Method count | Present prompts per method | Missing prompts |
|---|---:|---:|---|
| v178 | 6 | 16/32 | 16-31 |
| v179 | 2 | 16/32 | 8-23 |

The v179 methods contain indices `0-7` and `24-31`. The likely cause is that
the two nodes used different `SHARD_OFFSET` values. With two 8-GPU nodes,
the same offset must be supplied to both nodes because `NODE_RANK` already
contributes the node-local offset.

The 32 uploaded v179 videos are usable as generation artifacts:

- all fully decode to 477 frames at 16 fps and 832x480;
- `profile_top1_only` logs contain 359 Recent and one Coverage head;
- `profile_remainder` logs contain 356 Recent and four Coverage heads;
- no runtime failure pattern was found.

They do not yet support an attribution conclusion. v178 has no formal
VBench-Long result, and only prompts 0-7 currently have all four factorial
cells available.

## 2. Correct interpretation of v179

v179 separates the five RCCP-selected Coverage heads into a 2x2 design:

| Cell | L0H10 | Remaining four |
|---|---:|---:|
| `all_recent` | Recent | Recent |
| `profile_top1_only` | Coverage | Recent |
| `profile_remainder` | Recent | Coverage |
| `matched` | Coverage | Coverage |

Generating the two new cells before v178 finishes is allowed only as
exploratory work. It saves wall-clock time and does not itself make a claim.
Formal Shapley attribution still requires all of the following:

1. complete and audited v178 generation;
2. a passing v178 paired membership gate;
3. complete and audited v179 generation;
4. identical prompt mapping and VBench runtime fingerprints;
5. SHA-bound reuse of v178 `all_recent` and `matched` prompt metrics.

The previous remote patch silently skipped the v178 gate and provenance
checks. The new code replaces that behavior with explicit modes:

- `exploratory_before_v178_gate`: generation is permitted, but formal claims
  and attribution decisions are disabled;
- `formal_after_v178_gate`: generation inputs are bound to a passing v178;
- formal v179 audit always revalidates v178, even when the videos were
  generated under the exploratory mode.

## 3. Complete both experiments in parallel

Use two physical 8-GPU nodes for v178 and the other two for v179. Within each
pair, use relative `NODE_RANK=0` and `NODE_RANK=1`. Every node in the same
pair must receive the identical `PROMPT_INDICES` string.

On the two v178 nodes:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

PROMPT_INDICES="$(seq -s, 16 31)" NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh generate32
```

On the two v179 nodes:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

PROMPT_INDICES="$(seq -s, 8 23)" NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_head_attribution_32gpu.sh generate32
```

Do not rerun v179 `prepare` in the existing directory. Its input manifest is
now explicitly marked exploratory. For a fresh output directory, ungated
preparation requires an explicit opt-in:

```bash
V179_EXPLORATORY=1 NODE_RANK=0 \
  bash scripts/run_v179_head_attribution_32gpu.sh prepare
```

The runners retain `SHARD_OFFSET` compatibility, but `PROMPT_INDICES` is the
recommended recovery interface. It validates bounds and duplicates, and
deterministically partitions one shared index list across all workers.

## 4. Optional 8-prompt automatic factorial diagnostic

Prompts 0-7 already have all four cells. They can be evaluated without broad
manual review. This scope is isolated and can never unlock v179 or support a
paper claim.

First prepare v178 provisional metrics:

```bash
PARTIAL_COUNT=8 NODE_RANK=0 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit-partial

PROVISIONAL_COUNT=8 NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare
PROVISIONAL_COUNT=8 NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v178_vbench_long.sh split
PROVISIONAL_COUNT=8 NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v178_vbench_long.sh eval
PROVISIONAL_COUNT=8 NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
```

Then audit and evaluate the two v179 cells:

```bash
PARTIAL_COUNT=8 NODE_RANK=0 \
  bash scripts/run_v179_head_attribution_32gpu.sh audit-partial

PROVISIONAL_COUNT=8 NODE_RANK=0 bash scripts/run_v179_vbench_long.sh prepare
PROVISIONAL_COUNT=8 NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v179_vbench_long.sh split
PROVISIONAL_COUNT=8 NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v179_vbench_long.sh eval
PROVISIONAL_COUNT=8 NODE_RANK=0 bash scripts/run_v179_vbench_long.sh collect
```

Do not run either provisional `decision` action. The outputs report only a
directional pattern and a maximum six-case targeted review queue. Cases are
ranked by factor interaction or disagreement between top-1 and remainder
Shapley contributions.

## 5. Formal order after generation

Run v178 audit and full VBench first. If its decision is not
`advance_rccp_membership_to_broader_generation`, stop formal v179 analysis.
The already generated v179 videos remain exploratory diagnostics.

If v178 passes:

```bash
NODE_RANK=0 bash scripts/run_v179_head_attribution_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v179_vbench_long.sh prepare

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_vbench_long.sh split
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v179_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v179_vbench_long.sh decision
```

The next method experiment is deliberately not frozen yet:

- if v178 fails, replace the static five-head map with a state-conditional
  router and validate it on a new prompt suite;
- if v178 passes and v179 is top-1 dominated, simplify the mechanism around
  L0H10 and validate it on a fresh suite;
- if both v179 factors contribute, retain distributed RCCP and proceed to a
  128-prompt generation confirmation and cross-model profiling.

This decision order prevents post-hoc threshold tuning on the current
32-prompt holdout.
