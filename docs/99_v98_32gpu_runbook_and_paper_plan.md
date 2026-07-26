# v98 4-Node / 32-GPU Runbook and Paper Plan

Date: 2026-07-26

## 1. Implemented files

Core runtime:

- `third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py`
  - neutral labels `10/11`;
  - Supportive `stride` or `stride+cyclic`;
  - Suppressive `merge`, `cyclic`, or explicit no-middle.
- `third_party/Pyramid-Forcing/inference.py`
  - `--pyramidkv_history_polarity`;
  - explicit support/suppress policy flags;
  - runtime marker with labels and policy.
- `third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py`
  - explicit no-middle compositions no longer fall through to legacy cyclic
    update/read paths.

Offline maps and analysis:

- `scripts/build_v98_history_polarity_maps.py`
- `scripts/audit_v98_policy_traces.py`
- `scripts/analyze_v98_history_polarity.py`

Server execution:

- `scripts/run_v98_history_polarity_4node_32gpu.sh`
- `scripts/postprocess_v98_history_polarity.sh`

Tests:

- `tests/test_v98_history_polarity.py`
- `tests/test_v97_policy_contract.py`
- `third_party/Pyramid-Forcing/tests/test_adaptive_cache.py`

## 2. Required models and locations

Each node must see the same repository path and the same shared output path.
The default scripts expect:

```text
third_party/Self-Forcing/
|-- wan_models/
|   `-- Wan2.1-T2V-1.3B/
`-- checkpoints/
    `-- self_forcing_dmd.pt

third_party/Pyramid-Forcing/
|-- wan_models/
|   `-- Wan2.1-T2V-1.3B/
`-- checkpoints/
    `-- self_forcing_dmd.pt
```

Symlinks are acceptable. Override paths through `SF_REPO`, `PF_REPO`,
`SF_CHECKPOINT`, and `PF_CHECKPOINT` when the server layout differs.

The prompt files must exist:

```text
third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt
third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num128.txt
```

The v97 frozen score files are already versioned:

```text
runs/v97_qk_head_scores/scores/qk_head_scores.csv
runs/v97_qk_head_scores/scores/qk_head_score_artifact.json
```

## 3. First run: 32-prompt correctness screen

Use one shared `OUT_ROOT`. Launch exactly one command on each 8-GPU node.

Node 0:

```bash
cd /path/to/training-free
git pull
NODE_RANK=0 REPO_ROOT="$PWD" \
OUT_ROOT="$PWD/runs/v98_history_polarity_screen32" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh screen32 \
  > runs/v98_screen_node0.nohup.log 2>&1 &
```

Node 1:

```bash
cd /path/to/training-free
git pull
NODE_RANK=1 REPO_ROOT="$PWD" \
OUT_ROOT="$PWD/runs/v98_history_polarity_screen32" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh screen32 \
  > runs/v98_screen_node1.nohup.log 2>&1 &
```

Node 2:

```bash
cd /path/to/training-free
git pull
NODE_RANK=2 REPO_ROOT="$PWD" \
OUT_ROOT="$PWD/runs/v98_history_polarity_screen32" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh screen32 \
  > runs/v98_screen_node2.nohup.log 2>&1 &
```

Node 3:

```bash
cd /path/to/training-free
git pull
NODE_RANK=3 REPO_ROOT="$PWD" \
OUT_ROOT="$PWD/runs/v98_history_polarity_screen32" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh screen32 \
  > runs/v98_screen_node3.nohup.log 2>&1 &
```

The global task index is:

```text
global_rank = NODE_RANK * 8 + local_gpu_slot
method      = global_rank // 4
prompt shard = global_rank % 4
```

Thus all 32 GPUs run concurrently, each of eight methods receives four prompt
shards, and every method generates all 32 prompts at 120 frames.

## 4. The eight primary cells

| Cell | Purpose |
|---|---|
| `sf_native` | Native Self-Forcing baseline |
| `pf_native` | Official PF baseline |
| `pf_explicit_parity` | Same PF labels and policies through explicit override |
| `pf_aw_hybrid_merge` | PF-derived binary oracle with the new cache composition |
| `history_polarity_hybrid_merge` | Main natural-zero method |
| `history_polarity_stride_merge` | Remove periodic branch from Supportive cache |
| `history_polarity_hybrid_merge_v78` | Add trusted write admission |
| `positive_rate_half_hybrid_merge` | Alternative sign-fraction classifier |

The parity cell is a hard implementation control. If it is noisy while native
PF is clean, do not interpret any binary result.

## 5. Post-processing

After all four files below exist:

```text
runs/v98_history_polarity_screen32/status/node0.done
runs/v98_history_polarity_screen32/status/node1.done
runs/v98_history_polarity_screen32/status/node2.done
runs/v98_history_polarity_screen32/status/node3.done
```

run on one 8-GPU node:

```bash
cd /path/to/training-free
REPO_ROOT="$PWD" \
RUN_ROOT="$PWD/runs/v98_history_polarity_screen32" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/postprocess_v98_history_polarity.sh screen32 \
  > runs/v98_screen_postprocess.nohup.log 2>&1 &
```

The postprocessor performs:

1. exact indexed-video completeness checks;
2. strict map-hash and runtime-policy trace validation;
3. v78 transition trace validation;
4. blind-review package creation;
5. VBench-Long on five dimensions;
6. DINO/comprehensive metrics;
7. temporal-jump diagnostics;
8. controlled delta and parity reports.

Freeze this file before reading metrics:

```text
runs/v98_history_polarity_screen32/blind_review/scorecard.csv
```

## 6. Runtime logs to inspect

Every PF shard must contain:

```text
[PyramidKVRuntimePolicy]
```

Proposed methods must additionally contain:

```text
[HistoryPolarityPolicy] ... support_label=10 suppress_label=11 ...
legacy_pf_labels=false
```

The parity method must contain:

```text
[BinaryPolicyOverride] stable=stride responsive=cyclic
```

Inspect:

```text
configs/*.env
traces/*.policy.jsonl
traces/*.transition.jsonl
diagnostics/*.video.json
metrics/policy_trace_audit.json
metrics/cache_transition_summary.json
metrics/v98_analysis.md
```

The policy audit verifies:

- map SHA-256 equals the frozen shard config;
- proposed traces contain only labels `10/11`;
- Supportive hybrid reads only `CyclicStrategy + StrideStrategy`;
- Suppressive reads only `MergeStrategy`;
- sink/recent are exactly `3/4`;
- hybrid union never exceeds four frames;
- all selected layers, all 12 heads, and both CFG branches appear.

## 7. Human review form

For each method, record:

```text
polygon/noise artifact: yes/no
identity retention: 1-5
background continuity: 1-5
motion continuity: 1-5
long-range drift: 1-5
repetition/looping: 1-5
startup flashback: yes/no
catastrophic failure time:
notes:
```

Any polygon/noise artifact is a hard usability failure regardless of aggregate
metric rank.

## 8. Second run: MovieGenBench-128

Run the same four commands with:

```text
screen32 -> main128
OUT_ROOT -> $PWD/runs/v98_history_polarity_main128
```

For example on node 0:

```bash
NODE_RANK=0 REPO_ROOT="$PWD" \
OUT_ROOT="$PWD/runs/v98_history_polarity_main128" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh main128 \
  > runs/v98_main_node0.nohup.log 2>&1 &
```

Repeat for node ranks `1`, `2`, and `3`. Then:

```bash
REPO_ROOT="$PWD" \
RUN_ROOT="$PWD/runs/v98_history_polarity_main128" \
GPU_LIST=0,1,2,3,4,5,6,7 \
nohup bash scripts/postprocess_v98_history_polarity.sh main128 \
  > runs/v98_main_postprocess.nohup.log 2>&1 &
```

Do not tune the zero threshold on MovieGenBench-128. The `-0.1/+0.1` maps are
robustness ablations and must remain labeled as such.

## 9. Result-dependent next step

### Main method is clean and competitive

Freeze the exact map hash and cache configuration. Start the paper main table,
then run seed replication, 60-second extrapolation, and ABA scene-return
evaluation.

### Main method is clean but trusted writes are better

Use `history_polarity_hybrid_merge_v78` as the candidate only if the trace
shows nontrivial acceptance/rejection and blind review confirms the gain.
Report the base and write-gated variants separately.

### PF oracle works but natural polarity does not

The cache composition is useful but independent discovery is weak. Do not
call the PF map ours. Analyze the 28 mismatched heads and test only the
predeclared `-0.1/+0.1` robustness thresholds before deciding whether to
abandon the classifier.

### PF parity fails

Stop all quality interpretation. Compare native/parity runtime traces and
configuration hashes first.

### All neutral binary methods fail

Record the binary direction as negative and return to PF/v78. Do not hide the
failure by using PF's three labels under a new name.

## 10. Paper writing schedule

Paper sections that can be written immediately:

1. problem definition and training-free setting;
2. history-polarity statistic;
3. Supportive/Suppressive cache composition;
4. implementation and complexity;
5. provenance and distinction from PF;
6. experimental protocol and ablations.

Sections that must wait for GPU evidence:

1. final method choice, including whether trusted writes remain;
2. quantitative main table;
3. qualitative claims about identity, motion, and scene switching;
4. final abstract and contribution wording.

The current provisional title can be:

> History-Polarity Memory Routing for Training-Free Long Autoregressive Video
> Generation

This title remains editable if trusted writes or prompt-switch lifecycle
become central after experiments.
