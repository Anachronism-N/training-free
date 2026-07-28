# v129: no-PF paper comparison, confidence-gated recall, and 10-hour runbook

Date: 2026-07-28

Status: code-complete protocol awaiting server execution. This document
supersedes the v125 execution plan for the next run. It does not invalidate
the already completed v125 videos or metrics.

## 1. Decision

The next main experiment does **not** regenerate Pyramid Forcing and does
**not** run A-B-A prompt switching. The immediate task is 30-second,
single-prompt long-video extrapolation on all 128 Qwen-rewritten MovieBench
prompts.

The current best base method is:

```text
frozen old-v98 304/56 head partition
+ Supportive: sink1 + TemporalPrototype4 + recent4
+ Suppressive: sink1 + Retrieval1(age <= 24) + recent7
+ exclusive cache ownership
+ clean K/V descriptors and original temporal positions
+ a strict 9 full-frame-equivalent read budget
```

v125 provides direct evidence for this selection. Among its eight methods,
`ours_prototype_retrieval1_age24` reached the highest Dynamic Degree
(`61.93`) while remaining close to the other Ours candidates on the five
reported quality dimensions. This is evidence for a useful base cache, not
evidence that every quality dimension is best.

v129 adds one effect-oriented mechanism:

```text
retrieve an old state only when
  top1 cosine >= 0.55
  and top1 cosine - top2 cosine >= 0.005;
otherwise abstain from that retrieval read
```

The two new candidates differ only in the context available when retrieval
abstains:

- `ours_confidence_recent`: sink1 + gated Retrieval1 + recent7;
- `ours_confidence_motion`: sink1 + gated Retrieval1 + MotionPair1 + recent5.

MotionPair1 is an always-available two-frame motion companion in the second
candidate. It is not conditionally inserted after the gate. The gate controls
only the retrieved historical frame.

## 2. What may be claimed

The intended method has three technical components:

1. **Binary functional routing.** A fixed 304/56 head map separates
   History-Supportive and History-Suppressive heads instead of using PF's
   Anchor/Wave/Veil runtime routes.
2. **Role-conditioned bounded memory.** Supportive heads receive compressed
   temporal prototypes; Suppressive heads receive content-addressed,
   age-bounded recall. Both retain an explicit sink and recent context under
   the same maximum 9-FFE budget.
3. **Uncertainty-aware recall.** A suppressive head may abstain when the best
   archived state is absolutely weak or insufficiently separated from the
   second-best state. This directly targets forced stale-state injection and
   late subject enlargement.

The first component is still the main paper risk. The committed map is
`legacy_v98_absolute_sign_304_56.csv`; its historical absolute-logit statistic
is not invariant to a common logit shift. It is acceptable as the frozen map
for selecting and evaluating the cache, but it must not be described as a
final shift-invariant discovery method. Before submission, the head section
needs the original score artifact, threshold stability, PF-overlap analysis,
and random/inverted/all-head controls. If those controls fail, the defensible
paper claim is role-conditioned memory under an empirically discovered binary
partition, not a general head-classification theorem.

The code base is derived from and must cite
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) and
[Pyramid Forcing](https://github.com/if-lab-pku/Pyramid-Forcing).
Temporal prototypes and retrieval are our adapted implementation, but the
general principles of cache compression and retrieval have clear precedents.
Relevant comparison/inspiration repositories include
[Deep Forcing](https://github.com/cvlab-kaist/DeepForcing),
[Rolling Forcing](https://github.com/TencentARC/RollingForcing),
[LongLive](https://github.com/NVlabs/LongLive),
[LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG), and
[Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing).
The paper must attribute these ideas and distinguish our exact classifier,
budget, update rule, abstention rule, and experimental evidence.

## 3. Frozen 30-second table

The comparison has exactly eight methods:

| Key | Source | New videos | Comparison class |
|---|---|---:|---|
| `sf_native` | reuse v125 | 0 | same-backbone baseline |
| `deep_forcing` | official local code | 128 | same SF checkpoint, external method |
| `rolling_forcing` | official local code | 128 | external trained system |
| `longlive` | native LongLive config from LongLive-RAG release | 128 | external trained system |
| `ours_prototype_retrieval_age24` | reuse v125 | 0 | current no-gate method |
| `ours_confidence_recent` | v129 | 128 | new method |
| `ours_prototype_retrieval_motion` | reuse v125 | 0 | current no-gate motion method |
| `ours_confidence_motion` | v129 | 128 | new method |

Thus v129 generates 640 new videos, not 1,024:

```text
2 internal candidates x 128 = 256
3 external baselines x 128 = 384
total new videos            = 640
```

PF is absent from the frozen v129 manifest. The completed v125 PF result may
still be reported as historical context if its prompt, duration, seed, and
metric protocol are explicitly matched; it is not required to execute v129.
CausVid is omitted because a validated local implementation and checkpoint
contract are not available. A-B-A is deferred until the single-prompt table
and main method are settled.

## 4. Models and locations

The following files must exist on every node.

### Internal Ours

```text
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml
third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
configs/head_maps/legacy_v98_absolute_sign_304_56.csv
```

Example download:

```bash
cd third_party/Pyramid-Forcing
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt --local-dir .
```

### Deep Forcing

```text
third_party/DeepForcing/wan_models/Wan2.1-T2V-1.3B/
third_party/DeepForcing/checkpoints/self_forcing_dmd.pt
third_party/DeepForcing/configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml
```

```bash
cd third_party/DeepForcing
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt --local-dir .
```

v129 uses Deep Sink plus Participative Compression with `Budget=16`,
`Recent=4`, and sink10. `--is_ds_only` remains false.

### Rolling Forcing

```text
third_party/RollingForcing/wan_models/Wan2.1-T2V-1.3B/
third_party/RollingForcing/checkpoints/rolling_forcing_dmd.pt
third_party/RollingForcing/configs/rolling_forcing_dmd.yaml
```

```bash
cd third_party/RollingForcing
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download TencentARC/RollingForcing \
  checkpoints/rolling_forcing_dmd.pt --local-dir .
```

### LongLive

```text
third_party/LongLive-RAG/wan_models/Wan2.1-T2V-1.3B/
third_party/LongLive-RAG/checkpoints/longlive_base.pt
third_party/LongLive-RAG/checkpoints/longlive_lora.pt
third_party/LongLive-RAG/configs/longlive_native.yaml
```

```bash
cd third_party/LongLive-RAG
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download qixinhu11/LongLive-RAG \
  --local-dir . --include "checkpoints/*"
```

The table key is `longlive`, not `longlive_rag`: the selected YAML disables
retrieval and evaluates the released native LongLive backbone.

### Prompts and VBench

```text
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json
$VBENCH_CACHE_DIR/raft_model/models/raft-things.pth
$VBENCH_CACHE_DIR/amt_model/amt-s.pth
```

All nodes must see exactly the same prompt bytes, checkpoints, repository
checkout, VBench commit, and shared output roots.

## 5. Generation commands

Use the same exports on every node and change only `NODE_RANK` from 0 to 3.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

export REPO_ROOT="$PWD"
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7
export V129_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt

bash scripts/run_v129_no_pf_10h.sh preflight
```

Start node 0 first so it freezes the experiment contracts. Other nodes verify
the same bytes before launching.

Run the two internal methods on all four nodes:

```bash
bash scripts/run_v129_no_pf_10h.sh generate-internal
```

After every node completes, audit once on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v129_no_pf_10h.sh audit-internal
bash scripts/run_v129_no_pf_10h.sh analyze-gates
```

Run external baselines on all four nodes:

```bash
bash scripts/run_v129_no_pf_10h.sh generate-external
```

Each of the 32 workers owns a contiguous four-prompt interval and loads each
external model once. Valid partial videos are resumed. A partial file that
cannot be fully decoded or violates frame/fps/resolution is deleted from that
worker's own raw directory and regenerated, with a
`[v129-resume-repair]` log line. If an interrupted worker had already
published that corrupt file, cleanup is allowed only when its frozen marker
matches the current experiment contract, worker contract, method, prompt, and
raw directory; the owned published links and marker are then removed before
regeneration. Unmarked or mixed artifacts fail closed.

Audit and assemble once on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v129_no_pf_10h.sh audit-external
bash scripts/run_v129_no_pf_10h.sh assemble
```

The assembler refuses a prompt hash mismatch, wrong method list, wrong frame
contract, source-contract hash mismatch, partial method, or PF method. It
hardlinks or symlinks exact source videos into:

```text
runs/v129_paper_comparison_30s/published/<method>/
```

## 6. VBench-Long

The primary metric profile contains the seven official quality dimensions
plus Overall Consistency:

```text
subject_consistency
background_consistency
temporal_flickering
motion_smoothness
dynamic_degree
aesthetic_quality
imaging_quality
overall_consistency
```

This is 64 method-dimension jobs. With four nodes and eight GPUs per node,
each node receives 16 jobs and each GPU receives two sequential jobs.

First pre-split on all four nodes:

```bash
export V129_METRIC_PROFILE=core
bash scripts/run_v129_no_pf_10h.sh vbench-split
```

Then preflight and evaluate on all four nodes:

```bash
bash scripts/run_v129_no_pf_10h.sh vbench-preflight
bash scripts/run_v129_no_pf_10h.sh vbench-eval
```

Collect once on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v129_no_pf_10h.sh vbench-collect
```

The old VBench launcher had a prompt-integrity hazard: in
`long_custom_input`, numeric split folders can be interpreted as prompt text.
v129 wraps the official evaluator and rewrites every full-info row from the
frozen comparison manifest. Every job stores `prompt_mapping.json`; collection
requires all indices `0..127`, the exact manifest hash, and the mapping
artifact hash.

The core profile can produce the official VBench Quality Score because all
seven quality dimensions are present. It cannot produce Semantic Score or
Total Score. Those fields remain `n/a`, never a partial average.

If generation, core metrics, and manual checks finish with time left, run the
eight missing semantic dimensions:

```bash
export V129_METRIC_PROFILE=semantic_extension
# all four nodes
bash scripts/run_v129_no_pf_10h.sh vbench-preflight
bash scripts/run_v129_no_pf_10h.sh vbench-eval

# node 0 only
bash scripts/run_v129_no_pf_10h.sh vbench-collect
```

After both profiles, the collector computes official Quality, Semantic, and
Total scores using the checked-out VBench `scripts/constant.py`. This semantic
extension has higher priority than A-B-A.

## 7. Outputs to inspect

Generation:

```text
runs/v129_moviebench128_30s_internal/contracts/experiment.json
runs/v129_moviebench128_30s_internal/.../published_manifest.json
runs/v129_moviebench128_30s_internal/.../logs/
runs/v129_moviebench128_30s_internal/.../traces/
runs/v129_moviebench128_30s_internal/.../diagnostics/
runs/v129_moviebench128_30s_external/contracts/experiment.json
runs/v129_moviebench128_30s_external/published_manifest.json
runs/v129_moviebench128_30s_external/logs/
```

Gate diagnostics:

```text
.../analysis/retrieval_gate/retrieval_gate_summary.json
.../analysis/retrieval_gate/retrieval_gate_summary.md
.../analysis/retrieval_gate/retrieval_gate_threshold_sweep.csv
```

For each gate candidate, inspect:

- selected versus similarity-gate versus margin-gate frequency, both over
  all reads and conditioned on reads with at least one scored candidate;
- top-1, top-2, margin, and selected-age quantiles;
- rates by layer, CFG branch, and prompt;
- any cache-contract violation;
- whether the gate is degenerate: almost always accepting or abstaining.

The threshold sweep is post-hoc diagnosis. It does not transform a completed
video into a valid result for another threshold.

VBench:

```text
runs/v129_paper_comparison_30s/metrics/vbench_long_summary.json
runs/v129_paper_comparison_30s/metrics/vbench_long_coverage.json
runs/v129_paper_comparison_30s/metrics/paper_table/paper_table.md
runs/v129_paper_comparison_30s/metrics/paper_table/paper_table.csv
```

## 8. Ten-hour priority

1. Preflight and start the 256 internal videos.
2. Run the 384 external-baseline videos.
3. Audit and assemble the exact eight-method table.
4. Run the 64 core VBench jobs.
5. Collect Quality Score and raw Overall Consistency.
6. If time remains, run semantic extension.
7. Do not spend the critical path on PF regeneration or A-B-A.

Do not automatically promote the confidence gate merely because it is more
complex. Select:

- no-gate Prototype+Retrieval when gating is neutral or harmful;
- confidence+recent when it improves quality/consistency without losing the
  v125 dynamics advantage;
- confidence+motion only when the motion companion gives a repeatable metric
  or human advantage large enough to justify two extra middle frames.

## 9. Required follow-up after method selection

The main paper still needs:

1. paired bootstrap confidence intervals over 128 prompts;
2. human review focused on identity, late scale drift, background stability,
   motion, repetition, and artifacts;
3. all-head, random, inverted, and cache-swap controls;
4. removal of Prototype, Retrieval, confidence gate, sink, and recent one at a
   time;
5. retrieval age and threshold sensitivity;
6. runtime, peak memory, and effective token-budget reporting;
7. a 60-second confirmation for the final method;
8. head-score provenance and a defensible classifier experiment.

A-B-A and Echo-Forcing become secondary-task experiments only after these
single-prompt requirements are secure.
