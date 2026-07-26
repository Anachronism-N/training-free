# Corrected v98 Calibration, 4-Node / 32-GPU Runbook, and Gates

Date: 2026-07-26

Status: superseded for new generation by
`docs/100_v99_binary_cache_recovery_and_paper_story.md`. Keep this document
for the exact v98 protocol and artifact audit only. Do not relaunch its
eight-cell hybrid/merge matrix while diagnosing binary cache correctness.

This runbook supersedes the earlier v98 draft. Do not reuse the v97
absolute-sign score, the draft `304/56` map, a partially populated v98 output
directory, or any blind-review package created from those inputs.

## 1. Frozen protocol

All machines must see the same repository and output paths on shared storage,
at the same clean Git commit. The scripts fail closed on mixed commits,
any tracked or non-ignored untracked change, hashes, contracts, maps, shards,
or partial resume state. Ignored `runs/` outputs are allowed. Do not place
importable scratch code or local overrides in the checkout.

The corrected protocol is:

- 120 latent output frames, corresponding to 477 decoded frames;
- 16 fps, 832 x 480;
- seed 0 with per-prompt reseeding for generation;
- few-step CFG disabled, hence conditional branch only;
- Python cache-strategy and packing paths (`PYRAMIDKV_CPP_STRATEGY=0`);
- four prompt shards; `NODE_RANK` is the shard, and every node runs every
  primary method;
- canonical MovieBench prompt content (`num32` or `num128`), not merely the
  expected number of non-empty lines;
- exact score, score-artifact, raw-profile, prompt, map, config, checkpoint,
  source, and contract hashes;
- decoded video frame-count, fps, resolution, index, and hash audits;
- a public blind-review bundle separated from its private random assignment
  seed and method ledger;
- blind review frozen before automated quality metrics;
- fixed metric sampling and per-video paired statistics with prompt-level
  bootstrap intervals;
- atomic per-node generation, calibration, and postprocessing locks.

Any deliberate protocol change is a new experiment version, not a v98 resume.

## 2. Stage A: build the corrected head score

Run this once on a 16-GPU machine from a clean committed checkout:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

git status --short

OUT_ROOT="$PWD/runs/v98_middle_relative_scores" \
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
nohup bash scripts/run_v98_middle_relative_profile_16gpu.sh \
  > runs/v98_middle_relative_profile.nohup.log 2>&1 &
```

The calibration contains exactly 64 independently generated profiles:

```text
8 counterfactual prompt pairs
x 2 sides
x 2 seeds (0, 1)
x 2 uniform probe policies (stride, merge)
= 64 profiles
```

Only conditional/noisy QK records are scored. For each query, the first three
sink frames and latest four distinct historical frames are excluded from the
intervened middle group. The standardized middle-minus-recent margin is
invariant to a common logit shift. Records are aggregated within profiles,
profiles within each probe policy, and the two policy estimates receive equal
weight. Bootstrap resampling uses the counterfactual prompt pair as the
cluster.

Successful completion creates:

```text
runs/v98_middle_relative_scores/run_manifest.env
runs/v98_middle_relative_scores/scores/qk_head_scores.csv
runs/v98_middle_relative_scores/scores/qk_head_observations.json
runs/v98_middle_relative_scores/scores/qk_head_score_artifact.json
```

The artifact must have:

```text
version = 2
method = v98_middle_relative_qk_head_scores
accepted = true
score_definition.primary_field = middle_relative_logit_margin
score_definition.probe_policy_balanced = true
score_definition.bootstrap_unit = counterfactual_prompt_pair
bootstrap_protocol.rounds = 500
bootstrap_protocol.seed = 20260726
bootstrap_protocol.zero_effect_is_stable = false
acceptance_protocol.min_stable_head_fraction = 0.80
acceptance_protocol.min_head_bootstrap_agreement = 0.75
acceptance_protocol.min_topology_sign_agreement_fraction = 0.80
acceptance_protocol.min_minority_fraction = 0.05
```

`accepted=false` or a non-zero extractor exit is a scientific stop, not a
threshold-tuning invitation. An exact-zero bootstrap estimate counts as
unstable rather than positive evidence. Cross-topology agreement requires two
non-zero medians with the same sign, and the positive-rate control counts only
strictly positive profile estimates. Do not launch video generation.

Before constructing any map, every generation node reruns the calibration
validator. It parses all 64 profiles, recomputes each head's two policy
medians, balanced score, strict sign gates, positive fraction, and the frozen
paired-cluster bootstrap from `qk_head_observations.json`, then compares those
values against the CSV and artifact. A self-consistent-looking `passed=true`
field cannot bypass the observed thresholds.

## 3. Stage B: primary MovieBench-32 screen

Use a new shared `OUT_ROOT`. Launch the same command on all four nodes, changing
only `NODE_RANK` and, if necessary, the local GPU ids:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

NODE_RANK=0 \
GPU_LIST=0,1,2,3,4,5,6,7 \
OUT_ROOT="$PWD/runs/v98_history_polarity_screen32_corrected" \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh screen32 \
  > runs/v98_screen32_node0.nohup.log 2>&1 &
```

Repeat with `NODE_RANK=1`, `2`, and `3`. All nodes must use the same absolute
`OUT_ROOT`, commit, prompt file, models, configs, and calibration artifact.

The primary eight cells are:

| Method | Purpose |
|---|---|
| `sf_native` | Native Self-Forcing baseline |
| `pf_native` | Native Pyramid Forcing baseline |
| `pf_explicit_parity` | Explicit PF policy reconstruction |
| `pf_aw_hybrid_merge` | PF-derived oracle membership control |
| `history_polarity_hybrid_merge` | Proposed natural-zero classifier and dual memory |
| `history_polarity_stride_merge` | Remove the periodic Supportive branch |
| `history_polarity_zero_random_hybrid_merge` | Layer-wise count-matched random control |
| `positive_rate_half_hybrid_merge` | Per-profile sign-fraction control |

Each node runs all eight methods on its assigned prompt shard. This removes the
old method-by-node hardware confound. Within each node, primary GPU slots are
rotated by offsets `0, 2, 5, 7` for node ranks `0, 1, 2, 3`; the frozen
contract records and audits that mapping. v78 is not a primary cell.

Only one process may own a given `OUT_ROOT + NODE_RANK`. A second launch fails
on the node lock instead of writing the same logs, traces, videos, or markers.
Remove a stale lock only after confirming that its recorded process no longer
exists.

Generation is complete only when all four files exist and their embedded
manifest hashes agree:

```text
runs/v98_history_polarity_screen32_corrected/status/node0.done
runs/v98_history_polarity_screen32_corrected/status/node1.done
runs/v98_history_polarity_screen32_corrected/status/node2.done
runs/v98_history_polarity_screen32_corrected/status/node3.done
```

The runner validates every decoded video before writing a cell marker. Missing
or stale policy/transition traces, mismatched video fingerprints, changed
configs, and mixed contracts force regeneration or abort.

## 4. Stage C: audit and create the blind package

First run postprocessing with metrics disabled. This performs the full
generation/contract/trace/video audit and creates or verifies the blind
package:

```bash
RUN_ROOT="$PWD/runs/v98_history_polarity_screen32_corrected" \
RUN_VBENCH=0 RUN_COMPREHENSIVE=0 RUN_TEMPORAL=0 RUN_ANALYSIS=0 \
bash scripts/postprocess_v98_history_polarity.sh screen32
```

Reviewers score:

```text
runs/v98_history_polarity_screen32_corrected/blind_review/scorecard.csv
```

The directory `blind_review/` is the only bundle that reviewers should
receive. It contains blinded videos, `manifest_public.json`, and
`scorecard.csv`; it contains neither a deterministic seed nor a method map.
Keep the sibling directory `blind_review_private/` inaccessible to reviewers.
It contains `.complete.json`, `key_private.json`, and, after freezing,
`FROZEN.json`. Formal v98 creation omits `--seed`, so the assignment seed is
generated privately rather than being reconstructible from the runbook.

For every `*_1_to_5` field, `5` means best: stable identity/background,
natural motion/camera, artifact-free output, strong prompt alignment, no
long-range drift, and no repetition/looping. For every `*_0_or_1` field, `1`
means the named failure is present. Any startup flashback, abrupt jump,
polygon noise, identity score at most 2, artifact score at most 2,
long-range-drift score at most 2, or repetition score at most 2 makes that
video catastrophic for the usability gate.

Do not inspect or grant reviewer access to the private directory while
scoring. When the scorecard is complete, run postprocessing with metrics
enabled once; it will stop before metrics and print the exact
`prepare_blind_review.py ... --freeze` command, including both public and
private paths. Execute that command as the experiment operator. Freezing binds
every score row, blinded video, source video, prompt, public manifest, private
ledger, and source inventory by hash. A scored or frozen package is never
silently replaced; `FORCE_BLIND=1` is only for an explicitly abandoned
package.

Only one postprocessor may own a run root. A concurrent launch fails on
`.postprocess_run_lock` before creating or replacing metric/blind artifacts.

## 5. Stage D: metrics and paired analysis

After the blind package is frozen:

```bash
RUN_ROOT="$PWD/runs/v98_history_polarity_screen32_corrected" \
VBENCH_ROOT="/absolute/path/to/VBench" \
VBENCH_EXPECTED_COMMIT="<audited-vbench-commit>" \
nohup bash scripts/postprocess_v98_history_polarity.sh screen32 \
  > runs/v98_screen32_postprocess.nohup.log 2>&1 &
```

The postprocessor stages immutable metric inputs and fingerprints them before
running VBench, comprehensive metrics, or the temporal diagnostic. Resume is
allowed only when the input fingerprint and output hash both match. Before
staging, it reruns every shard video audit and requires its current video
fingerprint to equal the generation `.done` marker; replacing a valid video
after generation is therefore detected. Raw policy traces are also bound by
SHA-256 into the audit and final analysis.

The v98 metric protocol is not configurable:

```text
comprehensive sampled frames: 64
temporal diagnostic frame step: 2
VBench dimensions:
  subject_consistency, background_consistency, aesthetic_quality,
  imaging_quality, motion_smoothness, dynamic_degree
paired bootstrap rounds: 2000
paired bootstrap seed: 20260727
```

`metric_manifest.json` records these values, evaluator file hashes, the clean
VBench commit, generation contract fingerprint, every staged-video hash, and
blind-freeze verification. The analyzer validates that manifest before
reporting a hard gate.

Primary outputs include:

```text
metrics/workflow_contract_audit.json
metrics/policy_trace_audit.json
metrics/video_audits/
metrics/vbench_long_summary.json
metrics/comprehensive.json
metrics/temporal_jump.csv
metrics/blind_frozen_verification.json
metrics/metric_manifest.json
metrics/v98_analysis.json
metrics/v98_analysis.md
```

The analysis uses matched prompt/video rows. It reports paired bootstrap
confidence intervals and sample-integrity diagnostics; aggregate-only deltas
cannot satisfy a paper gate.

## 6. Stage E: MovieBench-128

Run `main128` only after the 32-prompt screen passes every hard gate. Use a
fresh shared output directory:

```bash
NODE_RANK=0 \
GPU_LIST=0,1,2,3,4,5,6,7 \
SCREEN32_RUN_ROOT="$PWD/runs/v98_history_polarity_screen32_corrected" \
OUT_ROOT="$PWD/runs/v98_history_polarity_main128_corrected" \
nohup bash scripts/run_v98_history_polarity_4node_32gpu.sh main128 \
  > runs/v98_main128_node0.nohup.log 2>&1 &
```

Repeat for nodes 1-3. Then follow the same audit, blind-review freeze, and
postprocess sequence with `main128`. Do not mix screen and main artifacts.
The runner enforces this gate in code: it requires the screen analysis to have
`hard_gate_pass=true`, rehashes every analysis input, and requires the same
commit, calibration score, configs, checkpoints, method policies, and
individual maps before it can publish a main128 contract.

## 7. Optional matched v78 follow-up

Only after the primary blind package is frozen and
`metrics/v98_analysis.json` reports `hard_gate_pass=true` may v78 be tested.
It is a fresh, separate two-cell phase under the primary root:

```text
followup_history_polarity_hybrid_merge_base
followup_history_polarity_hybrid_merge_v78
```

Launch all four nodes with the same primary `OUT_ROOT` and
`FOLLOWUP_V78=1`:

```bash
NODE_RANK=0 FOLLOWUP_V78=1 \
GPU_LIST=0,1,2,3,4,5,6,7 \
OUT_ROOT="$PWD/runs/v98_history_polarity_screen32_corrected" \
bash scripts/run_v98_history_polarity_4node_32gpu.sh screen32
```

Postprocess with:

```bash
FOLLOWUP_V78=1 \
RUN_ROOT="$PWD/runs/v98_history_polarity_screen32_corrected" \
bash scripts/postprocess_v98_history_polarity.sh screen32
```

The follow-up has its own contracts, blind package, metrics, and transition
trace summary. Its two methods exchange GPU order within node pairs. The
follow-up gate also requires its mode, commit, canonical prompt, score, and map
contract to match the completed primary run. It cannot feed the primary
go/no-go analyzer.

## 8. Hard stops

Stop the experiment if any of these occurs:

1. calibration is not explicitly accepted;
2. the two probe policies fail the frozen sign-consistency gate;
3. a map, artifact, raw profile, prompt, model, config, source, or contract hash
   changes;
4. PF native and explicit parity differ beyond their predeclared tolerances;
5. trace coverage is incomplete for any prompt/layer/head/branch Cartesian
   cell, or a strategy/parameter/budget differs from the contract;
6. any decoded video has the wrong index, frame count, fps, resolution, or
   content hash;
7. the blind package is incomplete, altered after scoring, or not frozen before
   metrics;
8. paired sample keys do not match across methods or a claimed interval crosses
   the predeclared decision boundary.

If parity fails, fix the implementation and start from a new output root. If
parity passes but neutral binary cells fail, reject the binary story rather
than silently substituting PF labels or retuning the classifier on video
results.
