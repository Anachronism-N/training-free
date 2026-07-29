# v134 Head Discovery Runbook

## 1. What Was Implemented

New files:

- `src/lifecycle_kv/head_profile.py`: environment-gated recorder.
- `scripts/build_v134_head_discovery_suite.py`: 128 natural plus 128
  controlled prompt jobs.
- `scripts/analyze_v134_head_discovery.py`: zero-threshold classification,
  bootstrap, factor/timestep/AR/temporal analysis, and acceptance gates.
- `scripts/run_v134_head_discovery_32gpu.sh`: four-node, 32-GPU launcher.
- `scripts/package_v134_head_discovery_results.py`: creates a small result
  bundle suitable for Git.

Modified Self-Forcing files:

- `third_party/Self-Forcing/inference.py`
- `third_party/Self-Forcing/pipeline/causal_inference.py`
- `third_party/Self-Forcing/wan/modules/causal_model.py`

When `HEAD_PROFILE_ENABLE` is not `1`, the added path is inactive. v134 must
run with all other experimental cache paths disabled.

## 2. Correctness Invariants

The runner and recorder fail loudly when any invariant is violated:

1. The manifest base prompt must match the dataset row.
2. A shadow forward must immediately follow its matching base forward.
3. `global_end_index` must equal the current block end before a shadow read.
4. A shadow cannot run with sink, full-window AAR, HCP, LifeCache, or
   structured memory enabled.
5. History must contain complete latent frames.
6. Each captured call must contain exactly one record from each of 30 layers.
7. Only the base branch writes the self-attention cache.
8. Semantic and null branches use independent cross-attention caches.
9. CPU and CUDA RNG states are restored around every shadow branch, so a
   stochastic forward cannot alter later base-trajectory noise.

The profile log prints one bounded begin/end line per video:

```text
[HeadProfile] begin index=... job=... kind=... frames=120
[HeadProfile] end job=... calls=... records=... output=...
```

Expected call and record counts:

```text
observational video: 27 calls, 810 layer records
counterfactual video: 81 calls, 2430 layer records
```

## 3. Prepare Once

On node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v134_head_discovery_32gpu.sh prepare
bash scripts/run_v134_head_discovery_32gpu.sh preflight
```

The prepare action validates the Qwen MovieBench source and writes:

```text
runs/v134_head_discovery/inputs/moviebench128_observational.txt
runs/v134_head_discovery/inputs/moviebench128_observational.jsonl
runs/v134_head_discovery/inputs/controlled128_counterfactual.txt
runs/v134_head_discovery/inputs/controlled128_counterfactual.jsonl
runs/v134_head_discovery/inputs/suite_metadata.json
```

All nodes must see the same shared path after preparation.

## 4. Run Natural MovieBench Profiling

Start the following command concurrently on all four nodes, changing only
`NODE_RANK` to `0`, `1`, `2`, or `3`:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v134_head_discovery_32gpu.sh observational
```

Each GPU receives four of the 128 prompts. The model is loaded once per GPU.

After all nodes finish, node 0 can inspect:

```bash
bash scripts/run_v134_head_discovery_32gpu.sh status
```

## 5. Run Controlled Counterfactual Profiling

Start concurrently on all four nodes:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v134_head_discovery_32gpu.sh counterfactual
```

Each controlled job generates one base video. Semantic and paraphrase branches
are read-only shadow calls, so they do not create extra videos.
The manifest overrides the dataset-index seed: all eight factor jobs in one
scenario family use the same base prompt and seed. This makes factor
comparisons trajectory matched even when the jobs run on different GPUs.

The two stages are independent. If compute is immediately available, half of
the nodes may run observational and half counterfactual by setting
`NUM_NODES=2` and ranks `0,1` within each stage, but each stage must use its
own complete rank set. The default four-node sequence is less error prone.

## 6. Audit and Analyze

On node 0 after both stages:

```bash
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v134_head_discovery_32gpu.sh audit
bash scripts/run_v134_head_discovery_32gpu.sh analyze
```

The audit requires:

- 128 profiles and 128 videos in each stage;
- 128 profile-end log markers in each stage;
- no traceback, CUDA OOM, `RuntimeError`, `NaN`, or polygon marker.

Analysis outputs:

```text
runs/v134_head_discovery/analysis/analysis_summary.md
runs/v134_head_discovery/analysis/classification_report.json
runs/v134_head_discovery/analysis/head_scores.csv
runs/v134_head_discovery/analysis/head_job_scores.csv
runs/v134_head_discovery/analysis/head_map.csv
runs/v134_head_discovery/analysis/head_factor_scores.csv
runs/v134_head_discovery/analysis/head_timestep_scores.csv
runs/v134_head_discovery/analysis/head_ar_scores.csv
runs/v134_head_discovery/analysis/temporal_timestep_scores.csv
runs/v134_head_discovery/analysis/temporal_ar_scores.csv
runs/v134_head_discovery/analysis/family_base_consistency.csv
runs/v134_head_discovery/analysis/threshold_sweep.csv
runs/v134_head_discovery/analysis/layer_summary.csv
runs/v134_head_discovery/analysis/factor_layer_summary.csv
runs/v134_head_discovery/analysis/timestep_layer_summary.csv
runs/v134_head_discovery/analysis/ar_layer_summary.csv
runs/v134_head_discovery/analysis/analysis_debug.json
```

`head_map.csv` uses:

```text
1 = prompt-conditional
0 = prompt-invariant
```

Do not use this map in generation unless
`classification_report.json -> acceptance_gates.accepted` is true.
Because each scenario family repeats the same base prompt and seed across
eight factors, `family_base_consistency.csv` should also be checked for
cross-GPU trajectory drift before interpreting factor differences.

## 7. Package Results for GitHub/Codex Review

Raw `.pt` profiles and videos should not be committed. On node 0:

```bash
bash scripts/run_v134_head_discovery_32gpu.sh package
git add docs/results/v134_head_discovery
git commit -m "results: add v134 head discovery analysis"
git push
```

The package action copies only analysis tables, the controlled prompt
manifest, checksums, inventory, and bounded worker-log diagnostics to:

```text
docs/results/v134_head_discovery/
```

For the next review, provide:

1. The result commit hash.
2. Human visual notes for obvious failures or motion/identity changes.
3. Any worker log not summarized correctly by
   `worker_log_summary.json`.

## 8. Debug Triage

### Polygon noise appears in generated videos

v134 does not alter the base prediction with shadow outputs. Therefore:

1. Check whether native observational videos also contain the noise.
2. Confirm the log shows no non-native cache path enabled.
3. Compare the base video with SF native using the same prompt and seed.
4. Check for stale environment variables and verify the runner printed
   `HEAD_ROLE_ENABLE=0`, `LIFECACHE_ENABLE=0`,
   `STRUCTURED_MEMORY_ENABLE=0`, and `COMMIT_FORCING_ENABLE=0`.

If only counterfactual videos are corrupted, treat that as an implementation
bug: shadow cache mutation or GPU memory corruption would be more plausible
than a head-classification failure.

### Profile has fewer than 30 layers per call

The strict recorder raises before saving. Inspect the first traceback and the
last `[HeadProfile] begin` line. Do not merge partial profiles.

### Semantic and null interactions are both near zero

Inspect `semantic_native_response`, `semantic_query_response`, and
`semantic_current_key_response` in `head_scores.csv`:

- all near zero: alternate text conditioning probably did not enter the
  shadow path;
- native response nonzero but residual response zero: prompt changes the
  current representation but not historical memory use;
- semantic and null equally high: the score is dominated by wording or text
  encoding variation, so H1 fails.

### Scores change strongly by timestep

Use `head_timestep_scores.csv`. Do not average the effect into a static map.
The next implementation should use a timestep-conditioned continuous gate.

### Scores change strongly by factor

Use `head_factor_scores.csv`. A scene-switch method may use scene/camera
scores, while single-prompt identity extrapolation may use identity/appearance
scores. Report this specialization instead of claiming one universal head
taxonomy.

## 9. Next Causal Experiment

Only after v134 analysis:

1. Freeze the classifier or continuous score without looking at generation
   metrics.
2. Implement prompt-boundary interventions on classified heads.
3. Compare correct, inverse, random count-matched, and all-head routing.
4. Test single-prompt 30-second extrapolation first.
5. Add A-B-A scene switching after the single-prompt path is stable.

The first cache experiment should isolate prompt-boundary handling. Temporal
cache heterogeneity and motion-frame selection should be added only after the
head property itself has causal evidence.
