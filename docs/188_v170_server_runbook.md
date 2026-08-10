# v170 Server Runbook

## 1. Purpose and outputs

v170 is a 16-prompt, 64-video matched attribution experiment. It compares two
cache selectors with two order-balanced replicas and uses all 32 GPUs. It does
not require new manual review.

Default output:

```text
runs/v170_matched_attribution_moviebench16/full8
```

The server must provide the same shared checkpoint and third-party runtime as
v169. The default checkpoint is:

```text
/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
```

## 2. Pull and smoke test

On one node, use two GPUs to generate four videos for prompt 3. Each GPU runs
one matched lane sequentially.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git fetch origin
git checkout codex/v170-matched-attribution
git pull --ff-only origin codex/v170-matched-attribution

NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1 \
  bash scripts/run_v170_matched_attribution_moviebench16.sh smoke
```

The smoke run is only a runtime guard. Check its final summary for four
successful tasks and inspect logs only if a task fails. Do not use its videos
to select a method.

## 3. Four-node preflight

Run once on every node with rank `0..3`:

```bash
NODE_RANK=${RANK} NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v170_matched_attribution_moviebench16.sh preflight
```

Each node must report eight workers and 16 tasks. Every worker must contain two
tasks for one prompt. Do not start generation if the frozen contract differs
between nodes.

## 4. Generate 64 videos

Launch the following concurrently on all four nodes:

```bash
NODE_RANK=${RANK} NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v170_matched_attribution_moviebench16.sh generate \
  2>&1 | tee runs/v170_node${RANK}.launcher.log
```

After all nodes finish, run on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v170_matched_attribution_moviebench16.sh audit

NODE_RANK=0 \
  bash scripts/run_v170_matched_attribution_moviebench16.sh mechanism

NODE_RANK=0 \
  bash scripts/run_v170_matched_attribution_moviebench16.sh replica-hash
```

Required generated methods are:

```text
ours_v170_v166_a
ours_v170_queryweighted_a
ours_v170_v166_b
ours_v170_queryweighted_b
```

The audit requires 16 videos per method. The mechanism command requires all
ten active layers in every one of the 64 policy traces.

## 5. VBench-Long core-9

Prepare links on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v170_vbench_long.sh prepare
```

Pre-split on all four nodes:

```bash
NODE_RANK=${RANK} NUM_NODES=4 \
  bash scripts/run_v170_vbench_long.sh split
```

Check the evaluation schedule on every node:

```bash
NODE_RANK=${RANK} NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v170_vbench_long.sh preflight
```

Run evaluation concurrently on all four nodes:

```bash
NODE_RANK=${RANK} NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v170_vbench_long.sh eval
```

If jobs are missing, resume once on one node:

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v170_vbench_long.sh resume-missing
```

Collect and make the automated decision on rank 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v170_vbench_long.sh collect
```

The main decision files are:

```text
runs/v170_matched_attribution_moviebench16/full8/analysis/v170_matched_metrics.json
runs/v170_matched_attribution_moviebench16/full8/analysis/v170_matched_metrics.md
runs/v170_matched_attribution_moviebench16/full8/automated_screen/full_layer_trace.json
runs/v170_matched_attribution_moviebench16/full8/automated_screen/replica_hashes.json
```

## 6. What to return

Push the small structured artifacts and launcher logs. Videos and split clips
do not need to be uploaded.

```bash
git add \
  runs/v170_matched_attribution_moviebench16/full8/contracts \
  runs/v170_matched_attribution_moviebench16/full8/status \
  runs/v170_matched_attribution_moviebench16/full8/published_manifest.json \
  runs/v170_matched_attribution_moviebench16/full8/automated_screen \
  runs/v170_matched_attribution_moviebench16/full8/metrics/vbench_core9_summary.json \
  runs/v170_matched_attribution_moviebench16/full8/analysis \
  runs/v170_node*.launcher.log
git commit -m "results: add v170 matched attribution"
git push
```

If a mechanism or contract gate fails, also package diagnostics:

```bash
bash scripts/run_v170_matched_attribution_moviebench16.sh package
```

Return the package path and the first failing task or layer. Broad video review
is not needed for this round.
