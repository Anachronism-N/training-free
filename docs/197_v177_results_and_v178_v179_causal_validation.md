# v177 Results and v178-v179 Causal Validation

## 1. Decision summary

The corrected v177 profile is valid and supports proceeding to generation.
It does **not** yet establish a generation method.

The strict run completed:

- 128/128 prompts;
- 16/16 shards;
- 184,320 records;
- 36/36 representation-level Union-superset checks per record;
- zero missing prompts and zero teacher-superset violations.

Five of 360 self-attention heads passed every frozen discovery, validation,
stability, salience, budget, call, AR, and global-FDR gate. All five prefer
the Coverage operator; no Episode-compatible head survived the gates.

| Rank | Head | Coverage gain over Recent | Discovery margin | Validation win vs Recent |
|---:|---|---:|---:|---:|
| 1 | L0H10 | 4.7828 | 0.9625 | 1.0000 |
| 2 | L8H6 | 1.3419 | 0.3211 | 1.0000 |
| 3 | L23H2 | 0.5754 | 0.1241 | 0.7500 |
| 4 | L5H3 | 0.3098 | 0.2419 | 0.9688 |
| 5 | L6H6 | 0.2147 | 0.2147 | 0.9375 |

Every selected head chose Coverage in both call halves, both AR halves, and
all 12 prompt resamples. The result is therefore not the previous
middle-layer-only effect. The selected heads occur at layers 0, 5, 6, 8,
and 23.

The effect is highly uneven: L0H10 has a much larger profiling gain than the
other four heads. v178 tests whether the selected membership transfers to
video generation. v179, only after a positive v178, tests whether the
generation gain is distributed or dominated by L0H10.

## 2. What v177 establishes

The current mechanism hypothesis is Residual Cache-Compatibility Profiling
(RCCP):

> Under a representation-complete teacher, compare equal-budget cache
> operators by how well each preserves a head's residual contribution, then
> assign only statistically stable nonlocal-compatible heads to their
> preferred operator.

The three equal-budget candidate operators are:

| Operator | Cache | Intended behavior |
|---|---|---|
| Recent | sink1 + recent8 | local continuity |
| Coverage | sink1 + reservoir4 + recent4 | broad temporal coverage |
| Episode | sink1 + reservoir2 + coherent motion pair2 + recent4 | event/motion history |

This differs from PF's temporal-QK Anchor/Wave/Veil taxonomy. The labels are
not imported from PF, and cyclic/stride/merge membership is not used. RCCP
asks which equal-budget memory operator locally approximates the full Union
teacher for each head.

The corrected v177 result supports the following limited statements:

1. Cache compatibility is head-specific: 355 heads prefer Recent and five
   pass all gates for Coverage.
2. The five selected memberships are stable across prompt resamples, calls,
   and early/late AR positions in this SF model and prompt suite.
3. A strict representation-level teacher is necessary. The old v176 result
   was invalid because physical-frame inclusion alone did not guarantee
   inclusion of saved, time-mapped, and dynamic-RoPE K representations.

It does not yet support these statements:

- the five-head map improves generated video;
- RCCP membership is better than same-budget random membership;
- all five selected heads are causally useful;
- the map transfers to another model;
- Episode memory is useful or useless in general;
- RCCP solves scene switching.

## 3. Diagnostics that matter

The reported all-head call-half agreement of 0.9750 and AR-half agreement of
0.9472 are partly inflated by the 355 Recent assignments. More informative
for the selected set are:

- all five selected heads have stability frequency 1.0;
- all five have Coverage/Coverage call-half decisions;
- all five have Coverage/Coverage AR-half decisions;
- all validation confidence intervals versus Recent are above zero;
- all validation comparisons pass global BH q <= 0.10.

The discovery candidate count is unchanged for margin thresholds 0, 0.005,
and 0.01 (19 candidates), so the primary 0.01 discovery margin is not sitting
on a fragile threshold cliff. The final five-head set is much smaller because
validation, stability, call/AR consistency, budget, salience, and global FDR
are also required.

Legacy prompt-perturbation features are only modest predictors. The strongest
within-layer Spearman correlations are approximately 0.40-0.43 for K shift;
none is sufficient to replace RCCP with a prompt-sensitivity threshold.

## 4. v178: untouched-prompt causal membership test

v178 is now the only required next generation experiment. It uses the 32
generation-holdout prompts that were never used for v177 membership and six
methods:

| Method | Coverage heads | Purpose |
|---|---:|---|
| `matched` | v177 five-head map | membership hypothesis |
| `all_recent` | 0 | operator-utility control |
| `hard_negative_0..3` | 5 each | rejected, per-layer/count-matched memberships |

Each hard negative has one Coverage head in exactly layers 0, 5, 6, 8, and
23. Therefore matched versus the hard-negative ensemble changes membership,
not the number of nonlocal heads, layer allocation, cache budget, prompts,
seed, node, or generation settings.

The run contains 6 x 32 = 192 videos at 30 seconds. Broad manual review is
not part of the decision. The pipeline performs full media decoding, runtime
route-count checks, VBench-Long core-9, prompt-paired bootstrap confidence
intervals, sign tests, global BH correction, and an explicit gate report.

Run on node 0 first:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh preflight
```

Run on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh generate32
```

Then on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare
```

Prepare splits and evaluate on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh eval
```

Collect and print the machine decision on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh decision
```

The `decision` action exits with code 3 when any frozen gate fails and prints
the exact failed checks. Do not run v179 in that case. A failed v178 means the
static RCCP membership rule is not supported for generation, even if the
profiling proxy is internally stable.

## 5. v179: incremental 2x2 head attribution

Run v179 only when v178 emits:

```text
advance_rccp_membership_to_broader_generation
```

v179 automatically ranks the five selected heads by the frozen profiling
gain and separates L0H10 from the other four. No generated video or v178
metric is used to select that head. The factorial cells are:

| Method | L0H10 | Remaining four | Source |
|---|---:|---:|---|
| `all_recent` | Recent | Recent | reuse v178 |
| `profile_top1_only` | Coverage | Recent | generate v179 |
| `profile_remainder` | Recent | Coverage | generate v179 |
| `matched` | Coverage | Coverage | reuse v178 |

Only 2 x 32 = 64 new videos and 2 x 9 = 18 new VBench jobs are required.
The analyzer reuses v178's per-prompt `all_recent` and `matched` metrics only
after verifying their SHA-bound comparison manifest, metric summary,
generation contract, prompt file, and passing decision.

v178 `collect` also records a metric-runtime fingerprint over all 54 jobs.
The fingerprint binds the VBench commit, wrapper/full-info/RAFT/AMT hashes,
long-input mode, sample count, and local-model loading mode while ignoring
machine-specific paths. v179's 18 new jobs must have the identical
fingerprint. `collect` fails instead of subtracting scores produced by a
different evaluator or model cache.

Prepare on node 0:

```bash
NODE_RANK=0 bash scripts/run_v179_head_attribution_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v179_head_attribution_32gpu.sh preflight
```

Generate on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_head_attribution_32gpu.sh generate32
```

Audit and prepare metrics on node 0:

```bash
NODE_RANK=0 bash scripts/run_v179_head_attribution_32gpu.sh status
NODE_RANK=0 bash scripts/run_v179_head_attribution_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v179_vbench_long.sh prepare
```

Split and evaluate on all nodes, then collect on node 0:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v179_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v179_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v179_vbench_long.sh decision
```

The analyzer reports paired main effects, conditional effects, interaction,
and two-player Shapley values for quality, identity/background, temporal,
semantic, visual, and dynamic metrics. The two Shapley contributions add
exactly to `matched - all_recent` for every prompt and metric.

Possible conclusions are frozen as:

| Decision | Meaning |
|---|---|
| `distributed_selected_set_confirmed` | both factors have positive primary effects with CI/FDR support |
| `distributed_selected_set_directional_only` | both are positive, but 32 prompts do not give confirmatory uncertainty |
| `profile_top1_dominated` | L0H10 carries the primary benefit; simplify the method claim |
| `profile_remainder_dominated` | the other four carry the benefit; L0H10 profiling magnitude is misleading |
| `head_attribution_inconclusive` | do not claim distributed head specialization |

Only four preregistered tests receive BH-corrected q-values: two factors by
quality and identity/background. Other metrics and the interaction are marked
descriptive.

## 6. Stop and advance rules

1. Run v178 now.
2. If v178 fails, stop static RCCP generation work. Do not tune the five-head
   map on the 32 holdout prompts. Preserve Coverage/Recent as cache operators
   for a future online or state-conditional router if useful.
3. If v178 passes, run v179. No broad manual review is needed first.
4. If v179 supports distributed contribution, the next experiment is a
   broader 128-prompt generation confirmation and then cross-model profiling.
5. If v179 is top1-dominated, use the one-head mechanism as the honest method
   hypothesis and validate it on a fresh prompt suite before writing a broad
   head-taxonomy claim.
6. ABA/AB scene switching remains postponed. It becomes relevant only after
   a single-prompt long-generation method survives v178.

## 7. Artifacts to push back

For v178, push only:

```text
runs/v178_rccp_holdout_generation/inputs/
runs/v178_rccp_holdout_generation/contracts/
runs/v178_rccp_holdout_generation/audits/
runs/v178_rccp_holdout_generation/published_manifest.json
runs/v178_rccp_holdout_generation/metrics/vbench_core9_summary.{json,md,csv}
runs/v178_rccp_holdout_generation/analysis/v178_paired_metrics.{json,md}
```

For a passing v178 followed by v179, also push:

```text
runs/v179_rccp_head_attribution/inputs/
runs/v179_rccp_head_attribution/contracts/
runs/v179_rccp_head_attribution/audits/
runs/v179_rccp_head_attribution/published_manifest.json
runs/v179_rccp_head_attribution/metrics/vbench_core9_incremental_summary.{json,md,csv}
runs/v179_rccp_head_attribution/analysis/v179_head_attribution.{json,md}
```

Raw videos and VBench split clips should remain on the server.
