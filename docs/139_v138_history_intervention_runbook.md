# v138 History Intervention Runbook

## 1. Execution order

v138 must run in this order:

```text
prepare
preflight
smoke
profile
audit
analyze
package
```

Do not skip the one-video smoke. Corrected v138 maintains a pre-RoPE key
sidecar and adds four attention-level interventions, so cache alignment,
profile format, and memory must be checked before launching 128 videos.

Version-3 v138 profiles are invalid because their invert-and-reapply check
could not validate the assumed source positions. The fixed runner writes to a
new `v138_history_interventions_v2` root and accepts version 4 only.

## 2. Prepare on node 0

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v138_history_interventions_32gpu.sh prepare
bash scripts/run_v138_history_interventions_32gpu.sh preflight
```

Node 0 preflight also runs the v138 unit tests, including exact synthetic
RoPE reconstruction/reversal, repeated-frame freezing, wrong-position
detection, and the version-4 artifact contract.

Inputs:

```text
runs/v138_history_interventions_v2/inputs/
  moviebench128_history_intervention.txt
  moviebench128_history_intervention.jsonl
  suite_metadata.json
```

Every job uses seed 0 so own-history and wrong-history descriptors are
compared under matched stochastic initialization.

## 3. Mandatory one-video smoke

On node 0:

```bash
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0
bash scripts/run_v138_history_interventions_32gpu.sh smoke
```

The command generates prompt 0 and then checks:

- one video and one profile;
- profile format version 4;
- 9 captured calls;
- 270 layer records;
- complete 30-layer coverage;
- all four intervention signatures;
- projected query/history-key descriptors;
- pre-RoPE sidecar present;
- RoPE maximum/RMS reconstruction errors at most `5e-3` / `1e-3`;
- recent-value preservation error at most `1e-6`.

Expected final line:

```text
[v138-smoke] v4 sidecar, layer coverage, descriptors, RoPE, and recent preservation: PASS
```

Inspect on failure:

```text
runs/v138_history_interventions_v2/smoke/smoke.log
```

Also inspect the smoke video for polygon noise. The interventions are
read-only and must not alter the generated base trajectory.

## 4. Full four-node profile

Start concurrently on all four nodes, changing only `NODE_RANK`:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v138_history_interventions_32gpu.sh profile
```

Use ranks 0, 1, 2, and 3 exactly once. Each GPU receives four prompts.

Progress:

```bash
bash scripts/run_v138_history_interventions_32gpu.sh status
```

Expected completion:

```text
profiles=128/128 videos=128/128
```

v138 performs four additional attention reads at nine selected states. It
does not add semantic/null full-model shadow forwards.

## 5. Audit and analyze on node 0

```bash
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v138_history_interventions_32gpu.sh audit
bash scripts/run_v138_history_interventions_32gpu.sh analyze
```

If v136 has already completed, the analyzer automatically reads:

```text
runs/v134_head_discovery/analysis_multi_axis_v136/head_axes.csv
```

Override:

```bash
V136_HEAD_AXES=/path/to/head_axes.csv \
  bash scripts/run_v138_history_interventions_32gpu.sh analyze
```

v138 remains valid without v136; only cross-axis correlation columns are
omitted.

## 6. Analysis outputs

Directory:

```text
runs/v138_history_interventions_v2/analysis/
```

Primary:

```text
analysis_summary.md
analysis_report.json
head_axes.csv
```

Context:

```text
head_timestep_axes.csv
head_ar_axes.csv
head_timestep_specialization.csv
axis_diagnostics.csv
axis_correlations.csv
```

Correctness and donor assignment:

```text
profile_contract_audit.csv
donor_audit.csv
```

Detailed per-job files are retained outside Git:

```text
head_local_job_axes.csv
head_cross_job_axes.csv
```

## 7. Review order

1. Confirm profile contract and RoPE error.
2. Check self-history specificity before interpreting any head class.
3. Check split-half and bootstrap stability.
4. Inspect GMM BIC before using the order high/low split.
5. Inspect order/freeze/value-mismatch correlations.
6. Compare with v136 CPHI and temporal reach.
7. Inspect timestep specialization.

Do not call high reverse response a motion head without the grouped causal
validation described in `docs/138_causal_history_head_profiling_design.md`.

## 8. Package for Git

```bash
bash scripts/run_v138_history_interventions_32gpu.sh package

git add docs/results/v138_history_interventions
git commit -m "results: add v138 history intervention analysis"
git push
```

The package excludes raw projected descriptors, `.pt` profiles, videos,
worker logs, and per-job tables.

Provide for review:

1. result commit hash;
2. `analysis_summary.md`;
3. smoke video assessment;
4. any worker failure log;
5. whether v136 results were available during analysis.

## 9. Failure triage

### Smoke OOM

Do not launch the full run. First reduce only profiling frequency:

```text
noisy timestep 1000 only
clean frames 21, 63, 117
```

Do not reduce projection dimension or remove the reconstruction gate before
locating peak memory.

### RoPE reconstruction fails

The sidecar/cache rolling indices are misaligned, the asserted absolute
positions are wrong, or a non-native cache path is active. Check:

```text
LIFECACHE_ENABLE=0
STRUCTURED_MEMORY_ENABLE=0
HEAD_ROLE_ENABLE=0
SF_FULL_ATTN_MAX_FRAMES unset
```

Do not loosen `5e-3`.

### Base video differs from native SF

The intervention outputs are not used by the model. A visible difference
indicates state mutation, RNG contamination, or memory corruption. Stop the
run and compare prompt 0 with the same-seed native SF video.

### Self-history specificity is non-positive

The projected descriptor does not support cross-video retrieval. Do not use
it in the method. Check full-dimension descriptors on a small subset before
concluding that the underlying Q/K signal is absent.

### Order scores are continuous without a stable mixture

Retain a continuous gate. Do not manufacture a binary order class with a
quantile threshold.
