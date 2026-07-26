# v100 Final Candidate and Immediate Experiment Plan

Date: 2026-07-27

> Status update: superseded by
> `docs/104_v100_responsive_event_cache_and_aba_fast_screen.md` for the
> immediate experiment. This file records the earlier middle-relative /
> stride-cyclic proposal and should not be used as the current server command.

This document is the current authoritative method and execution plan. It
supersedes the Merge-based v98 proposal and the causal interpretation in the
first version of docs/102.

## 1. Confirmed candidate

Provisional method name:

**History-Polarity Dual-Horizon Cache (HP-Cache)**

Target:

training-free 30-second and longer autoregressive video extrapolation, with
single-prompt identity/scene persistence as the primary task and prompt/scene
switching as a secondary task.

### 1.1 PF-independent head profiling

For each head, profile a deployment-matched, shift-invariant QK margin:

```text
s_h = median_profiles(
        standardized(mean(logit to middle history)
                   - mean(logit to recent history)))
```

The profile uses balanced uniform-stride and uniform-merge probe topologies,
two seeds, and 64 observations per head. Sink and recent keys are excluded
from the middle set.

Natural binary rule:

```text
s_h >= 0 -> History-Supportive
s_h <  0 -> Recent-Responsive
```

PF Anchor/Wave/Veil labels are not used to compute `s_h`, choose zero, or set a
class quota. PF labels are used only for post-hoc analysis and oracle controls.

### 1.2 Role-conditioned dual-horizon read cache

History-Supportive heads preserve sparse global evidence:

```text
static sink:      first 3 frames
explicit middle: global stride, interval 6, capacity 4 frames
dynamic recent:  latest 4 distinct frames
```

Recent-Responsive heads preserve bounded phase-local evidence:

```text
static sink:      first frame
explicit middle: cyclic phase buckets, period 6, capacity 4 frames
dynamic recent:  latest 4 distinct frames
```

The first route is intended to stabilize long-range subject and scene
evidence. The second retains local motion phase and current appearance without
letting arbitrary old content dominate.

### 1.3 Interference-free cache composition

`HeadComposition` is the exclusive owner of all three segments:

```text
static sink + explicit middle + dynamic recent
```

The legacy dynamic-history path is disabled for these neutral-label routes.
Runtime traces record actual frame/token ids, branch, policy, owner, segment
sizes, and overlap checks. A run fails before its completion marker if:

- sink, middle, and recent overlap;
- dynamic history exceeds `recent4`;
- a head receives the wrong strategy;
- partial Merge state leaks into a non-Merge route;
- required owner metadata or trace coverage is missing.

This lifecycle contract is necessary for scientific attribution: the tested
cache must be the cache described by the paper.

### 1.4 Optional trust-conditioned update

The eighth screen cell combines the same read cache with the previously
validated v78 trust-conditioned update. It can become a method component only
if it improves the independent-map base under matched prompts without new
artifacts.

The intended role is to control when new middle history is committed after
motion/scene shocks, complementing the static per-head read horizon. It is not
part of the core claim until the paired `candidate32` result supports it.

## 2. What is different from Pyramid Forcing

PF:

- classifies three types using its own temporal QK/sign behavior;
- routes Anchor/Wave/Veil to stride/cyclic/merge;
- uses the native PF cache lifecycle.

HP-Cache:

- uses a different shift-invariant middle-vs-recent score;
- discovers two roles without PF labels or PF class-count matching;
- routes the roles to global-stride versus phase-cyclic memory;
- explicitly enforces exclusive sink/middle/recent ownership;
- tests random, inverted, threshold, and PF-oracle maps under the same cache
  topology.

Stride and cyclic primitives come from the PF codebase and must be credited.
The potential contribution is their coupling to an independently discovered
binary criterion plus a trace-verifiable lifecycle, not ownership of either
primitive.

## 3. Paper contributions, with claim gates

### Contribution 1: deployment-matched binary head discovery

Claim only if:

- raw score artifacts pass all frozen gates;
- the exact map statistics are internally consistent;
- natural zero or a nearby fixed threshold is stable;
- the learned map beats count-matched random and is not matched by inversion.

### Contribution 2: heterogeneous dual-horizon cache

Claim only if:

- independent stride/cyclic videos are artifact-free;
- PF-AR and PF-AW controls show that topology and membership both matter;
- the method improves identity/subject consistency without collapsing motion.

### Contribution 3: interference-free cache lifecycle

This implementation claim is already testable: every accepted run must pass
the ownership/overlap trace audit. Its quality value must still be shown with
the corrected-versus-old implementation history or a direct ablation.

### Conditional Contribution 4: trust-conditioned writes

Include only if `history_polarity_stride_cyclic_v78` consistently improves the
same-map base. Otherwise report it as an ablation and keep the paper focused
on profiling and read routing.

## 4. Fast one-video diagnostic

Use a clean checkout of the pushed commit. The score root must contain the raw
v98 score artifacts. Reuse directories may contain more videos because the
auditor selects the requested index.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

NODE_RANK=0 \
GPU_LIST=0 \
PROMPT_INDEX=0 \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v100_pf_aw_stride_cyclic_smoke1" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py \
  smoke1 --smoke-cell pf-aw-stride-cyclic \
  > runs/v100_pf_aw_stride_cyclic_smoke1.nohup.log 2>&1 &
```

Inspect:

```text
runs/v100_pf_aw_stride_cyclic_smoke1/
  pf_aw_neutral_stride_cyclic/
  diagnostics/
  traces/
  configs/
  experiment_contract.json
```

The log must contain `[V99MapAudit]`; the completion marker must exist; and
the video/trace audit JSON files must report success.

## 5. Eight-cell MovieBench-32 screen

`candidate32` uses all eight GPUs on each of four nodes. Every node runs all
methods on a disjoint eight-prompt shard, avoiding method/node confounding.

| Cell | Purpose |
|---|---|
| `pf_ar_neutral_stride_cyclic` | PF Anchor/rest membership control |
| `pf_aw_neutral_stride_cyclic` | PF (Anchor+Wave)/Veil control |
| `history_polarity_stride_cyclic` | proposed natural-zero map |
| `history_polarity_random_stride_cyclic` | layer-wise count-matched random |
| `history_polarity_inverted_stride_cyclic` | causal direction inversion |
| `history_polarity_tau_m0p1_stride_cyclic` | threshold robustness |
| `history_polarity_tau_p0p1_stride_cyclic` | threshold robustness |
| `history_polarity_stride_cyclic_v78` | optional update component |

Run the same command on all four nodes, changing only `NODE_RANK=0,1,2,3`.
All nodes must share the same `OUT_ROOT`.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

NODE_RANK=0 \
GPU_LIST=0,1,2,3,4,5,6,7 \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v100_hp_cache_candidate32" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py candidate32 \
  > runs/v100_hp_cache_candidate32.node0.log 2>&1 &
```

Node 0 builds/freezes the maps; other nodes wait for the identical manifest.
Use a fresh `OUT_ROOT` if code, model, prompts, score artifacts, or maps change.

## 6. Review and promotion rule

Review blind in this order:

1. PF native reuse;
2. v93 PF-binary-v78 reuse;
3. eight candidate cells with randomized method names.

For each prompt record:

- polygon/grid corruption;
- subject count and identity drift;
- clothing/object/background consistency;
- camera and object motion;
- freezing, loops, flashbacks, abrupt jumps;
- prompt compliance in early/middle/late thirds.

Do not promote based on DINO alone. Required quantitative views are:

- VBench-Long subject/background consistency;
- motion smoothness and dynamic degree;
- aesthetic and imaging quality;
- comprehensive DINO/CLIP identity and prompt alignment;
- temporal-jump diagnostics;
- per-prompt paired deltas and failure counts.

Promotion requires:

- no systematic corruption;
- independent map better than count-matched random;
- inversion worse or behaviorally distinct in the predicted direction;
- no material motion collapse relative to PF;
- improvement replicated across prompts, not only in the aggregate mean.

If random matches the learned map, retain the cache topology as an engineering
result but do not claim head discovery. If PF-AR wins clearly, the binary cache
is viable but the current independent classifier is not yet a paper
contribution.

Prepare the exact audited inputs and randomized blind-review package:

```bash
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
RUN_ROOT="$PWD/runs/v100_hp_cache_candidate32" \
bash scripts/postprocess_v100_hp_cache.sh candidate32 prepare
```

Fill `blind_review/scorecard.csv`, then freeze it without opening the private
key:

```bash
python scripts/prepare_blind_review.py \
  --run-root runs/v100_hp_cache_candidate32/metrics/eval_inputs \
  --methods \
    pf_native pf_binary_read_reference \
    pf_ar_neutral_stride_cyclic pf_aw_neutral_stride_cyclic \
    history_polarity_stride_cyclic \
    history_polarity_random_stride_cyclic \
    history_polarity_inverted_stride_cyclic \
    history_polarity_tau_m0p1_stride_cyclic \
    history_polarity_tau_p0p1_stride_cyclic \
    history_polarity_stride_cyclic_v78 \
  --prompts third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt \
  --prompt-count 32 \
  --output runs/v100_hp_cache_candidate32/blind_review \
  --private-output runs/v100_hp_cache_candidate32/blind_review_private \
  --freeze
```

Only after freezing, run metrics:

```bash
HUMAN_REVIEW_DONE=1 \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
RUN_ROOT="$PWD/runs/v100_hp_cache_candidate32" \
bash scripts/postprocess_v100_hp_cache.sh candidate32 metrics
```

The postprocessor validates the generation contract, all four shards, map
totals, traces, completion markers, and decoded videos. It stages exactly the
requested indices and schedules methods in GPU-sized waves.

## 7. MovieBench-128 confirmation

After the 32-prompt gate, `main128` generates only:

```text
pf_ar_neutral_stride_cyclic
history_polarity_stride_cyclic
history_polarity_stride_cyclic_v78
```

PF native and the historical v93 binary result are reused. The launch is the
same as `candidate32`, but the PF reuse path must contain all 128 videos:

```bash
REUSE_PF_DIR="$PWD/runs/v93_moviebench128_main/pf" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v100_hp_cache_main128" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py main128 \
  > runs/v100_hp_cache_main128.node0.log 2>&1 &
```

Again launch ranks 0-3. Each node receives 32 of the 128 prompts.

## 8. Result-dependent paper story

Best case:

> Long autoregressive video generation does not require PF's fixed three-way
> head taxonomy. A deployment-matched relative QK diagnostic identifies a
> sparse set of global-history-support heads, while the remaining heads need
> bounded phase-local evidence. Routing these two roles through an
> interference-free dual-horizon cache improves long-term identity and scene
> persistence while preserving motion, without training.

If natural zero loses but one nearby threshold wins, present zero as the
principled reference and the selected threshold as a calibration hyperparameter
chosen on a separate development split. Do not select and report it on the
same 32 prompts without disclosure.

If independent membership is not causal, the top-conference story is not yet
supported. Do not rename PF classes or hide the negative controls. The honest
fallback is a binary cache-topology engineering result plus the validated v78
update, which is weaker and should not be overstated.

## 9. Required evidence to return for analysis

Return or push:

```text
experiment_contract.json
maps/history_polarity_manifest.json
maps/head_assignments.csv
configs/*.json
diagnostics/*.json
traces/*.jsonl
status/*.done.json
logs/*.log
human review sheet
VBench-Long summary
comprehensive metrics
temporal-jump summary
```

The most important debug lines are `[V99MapAudit]`,
`[HistoryPolarityPolicy]`, and `[PyramidKVRuntimePolicy]`.
For any diagnostic Merge cell, inspect each strategy's `state` object in the
policy JSONL for stale invalid ids, more than one incomplete block, or
unexpected completed block/token counts.
