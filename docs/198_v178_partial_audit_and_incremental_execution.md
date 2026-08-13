# v178 Partial Audit and Incremental Execution

## 1. Current status after pulling `08c872f9`

The uploaded v178 run is incomplete. It contains the first 16 of the frozen
32 generation-holdout prompts for all six methods:

| Method | Videos | Required route counts (Recent/Coverage/Episode) |
|---|---:|---:|
| `matched` | 16/32 | 355/5/0 |
| `all_recent` | 16/32 | 360/0/0 |
| `hard_negative_0..3` | 16/32 each | 355/5/0 |

The 96 present videos pass the following limited checks:

- all six methods have prompt indices 0 through 15;
- every present video fully decodes to 477 frames at 16 fps and 832x480;
- every shard log contains exactly the expected route count and no runtime
  failure pattern;
- no pair of methods produced an identical video SHA-256 for the same prompt,
  so the six maps did alter the generation trajectory;
- the matched and four hard-negative maps preserve the same five Coverage
  layers and the same per-layer route counts.

There are no VBench-Long results yet. Therefore this upload says nothing
about whether RCCP-selected membership is better than all-Recent or the
hard-negative ensemble.

The original partial audit incorrectly wrote `published_manifest.ok=true`
while all six media reports had `ok=false` and were missing indices 16-31.
The repository manifest is now corrected to incomplete. New code cannot
prepare the formal VBench scope unless all 192 videos pass strict audit.

## 2. Audit and scheduling corrections

The v178 runner now supports an explicit global `SHARD_OFFSET`. With two
8-GPU nodes, the existing first wave used indices 0-15. The second wave uses
the same two nodes with `SHARD_OFFSET=16` and produces indices 16-31.

Audit artifacts are now mutable state until the run completes. A failed or
partial audit cannot publish a formal success manifest. Existing published
videos are accepted only when they are the same file or have the same
SHA-256; a different target remains a hard error.

An optional provisional scope is isolated under `provisional_16/`:

- it uses only prompt/video indices 0-15;
- its own 16-line prompt file is SHA-bound to the comparison;
- its result always has `membership_hypothesis_gate=null`;
- its decision is always `provisional_only_no_membership_decision`;
- it cannot unlock v179, regardless of metric values.

The paired analyzer also emits at most six targeted review cases, selected
from the largest matched losses against the hard-negative ensemble. This
queue is diagnostic only and never changes the frozen gate or head map.

## 3. Recommended next run: finish v178 first

Pull the latest code on both nodes. Confirm current state on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh status
```

Run the second 16-prompt wave on both 8-GPU nodes:

```bash
NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 SHARD_OFFSET=16 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh generate32
```

The runner uses the actual prompt index as the seed and output index. Thus
`SHARD_OFFSET=16` selects prompts 16-31 and does not change the already
generated prompt 0-15 trajectories.

After both nodes complete, run on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit
```

The formal audit must print:

```text
[v178-audit] PASS methods=6 prompts=32 videos=192 membership_decision_allowed=True
```

Then run the existing full core-9 pipeline:

```bash
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare

NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh split

NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh decision
```

If an individual method fails, regenerate only that method. For example:

```bash
METHODS=matched NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 SHARD_OFFSET=16 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh generate32
```

## 4. Optional diagnostic use of the existing 96 videos

This is optional and should not delay finishing indices 16-31. First create
the isolated provisional audit on node 0:

```bash
PARTIAL_COUNT=16 NODE_RANK=0 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit-partial
```

Then run VBench-Long with `PROVISIONAL_COUNT=16` on every action:

```bash
PROVISIONAL_COUNT=16 NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare

PROVISIONAL_COUNT=16 NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v178_vbench_long.sh split

PROVISIONAL_COUNT=16 NUM_NODES=2 NODE_RANK=<0|1> \
  GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_v178_vbench_long.sh eval

PROVISIONAL_COUNT=16 NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
```

Do not run `decision` in provisional mode; the runner rejects it. Inspect
only directional deltas and the targeted review queue. Do not tune RCCP
membership from these first 16 holdout prompts.

## 5. Next experiment after the formal gate

If and only if the complete v178 decision is
`advance_rccp_membership_to_broader_generation`, run v179. It partitions the
selected set into L0H10 and the remaining four heads. The v179 runner now
supports the same two-node waves:

```bash
NODE_RANK=0 bash scripts/run_v179_head_attribution_32gpu.sh prepare

NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 SHARD_OFFSET=0 \
  bash scripts/run_v179_head_attribution_32gpu.sh generate32

NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 SHARD_OFFSET=16 \
  bash scripts/run_v179_head_attribution_32gpu.sh generate32
```

Do not start v179 when v178 fails. A failed v178 rejects the current static
five-head generation map; it does not invalidate the profiling measurements
or the Recent/Coverage operators. The next method in that case should be a
state-conditional router tested on a new prompt suite, not threshold tuning
on these 32 holdout prompts.
