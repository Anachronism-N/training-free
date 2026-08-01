# v151: Signed Policy Preference and Low-Tail Confirmation

## 1. Purpose

v150 rejected the claim that the four largest scalar policy-shift scores form
a unique functional group. It also exposed a different, post-hoc pattern:
both the top and middle quartets frequently separated from the bottom quartet.

v151 tests two hypotheses without mixing their evidence:

1. **scalar low tail**: the bottom four v145 scalar-policy heads are a stable
   low-response group, while both remaining quartets respond more strongly;
2. **signed scene-policy preference**: heads ranked by a physically aligned,
   signed `uniform-vs-recent` preference response form a stronger causal group
   than the scalar L1 ranking.

The experiment is profiling and one-step causal confirmation. It does not yet
change a complete generation trajectory or claim improved video quality.

## 2. Signed Classification Standard

The v145 causal-policy probe stores each head's projected error for five cache
policies:

- `current_only`;
- `recent4`;
- `recent_budget` (eight recent frames);
- `boundary_recent`;
- `uniform_recent`.

For head `h`, define the uniform-policy advantage as

```text
P_h(prompt) = log((error_recent_budget + eps) /
                  (error_uniform_recent + eps)).
```

Positive values mean that uniform long-history sampling is closer to native
full-history attention than the recent-eight policy. For a controlled scene
change, v151 retains the signed displacement

```text
Delta P_h = P_h(scene variant) - P_h(base).
```

The classification score is `mean(abs(Delta P_h))`. The sign, base
preference, variant preference, family, seed, frame, timestep, layer, and head
are all retained in the exported observation table. Unlike the old scalar L1
score, this statistic identifies which cache-policy contrast changed.

### 2.1 No source/confirmation leakage

- even v145 controlled-prompt families are the discovery split;
- odd v145 families are the validation split;
- only discovery scores choose the per-layer low/middle/high quartets;
- validation scores only decide whether the observational source screen passes;
- the causal run uses 32 MovieBench prompts excluded from every v150 prompt.

The frozen signed source is:

```text
factor=scene
contrast=uniform_recent vs recent_budget
frame=117
timestep=500
heads per layer=4 / 4 / 4
```

Its source screen requires all of:

- layer-residual discovery/validation Spearman >= `0.30`;
- layer-residual seed0/seed1 Spearman >= `0.30`;
- validation high/low score ratio >= `1.10`;
- positive validation high/low separation in >= `70%` of layers;
- validation scene/paraphrase magnitude ratio >= `1.05`.

If this screen fails, the signed probes may still run for diagnosis, but the
analyzer forces all signed confirmation gates to fail. The scalar branch
remains independently valid.

## 3. Head Maps

Every map selects exactly four heads in each of 30 layers.

| Family | Groups | Definition |
|---|---|---|
| scalar | low4 / middle4 / high4 | v145 `full_semantic/all_policy_shift_mean` |
| signed | low4 / middle4 / high4 | discovery `abs(Delta P_h)` |
| control | random0..7 | balanced deterministic four-head maps |

Each random map is forbidden from matching any scalar or signed fixed group in
any layer. Across eight maps, every head is used either two or three times per
layer.

## 4. Causal Probe Grid

The core plan contains 32 probes:

- six fixed groups x four interventions = 24 probes;
- eight random maps x the primary uniform contrast = 8 probes.

| Intervention | Exact operation | Role |
|---|---|---|
| `uniform` | `uniform8 - recent8` | primary aligned policy contrast |
| `boundary` | `boundary8 - recent8` | alternative policy contrast diagnostic |
| `key_shift` | rotate old K, preserve recent four | generic K control |
| `value_shift` | rotate old V, preserve recent four | generic V control |

All probes retain the current block and use projected relative RMS target
`0.02`. Four quantization-aware refinement steps calibrate against the exact
cast-and-output-projection path. The accepted error remains the preregistered
`2%`; it is not relaxed after observing results.

The one-step contexts are frame 117 at nominal timesteps:

```text
1000, 750, 500, 250
```

This explicitly tests whether policy leverage is denoising-state conditioned.

## 5. Holdout Generation Grid

- source: 128-line Qwen-rewritten MovieBench prompt file;
- prompts: 32 deterministic diverse holdouts, disjoint from v150's 32 prompts;
- seeds: two new seeds per prompt, base `151000`;
- profiles/videos: 64;
- duration: 120 latent frames, approximately 30 seconds;
- GPUs: four nodes x eight GPUs;
- jobs per GPU: two;
- downstream records per profile: `(32 + native replay) x 4 = 132`.

Saved MP4 files are native Self-Forcing trajectories. Counterfactual probes
are read-only one-step replays and do not alter these videos.

## 6. Frozen Gates

### 6.1 Scalar low-tail branch

In one intact context, susceptibility must satisfy:

- scalar high4 > scalar low4;
- scalar middle4 > scalar low4;
- random ensemble > scalar low4;
- at least six of eight individual random maps > scalar low4.

Leverage must satisfy both high4 > low4 and middle4 > low4. Policy specificity
uses the post-hoc v150 candidate `middle4 > low4`: its uniform separation must
exceed both K-shift and V-shift separation. Final confirmation requires the
susceptibility, leverage, and specificity gates in the same context.

### 6.2 Signed branch

The signed source screen must pass first. In one intact context, signed high4
must beat low4, middle4, and the eight-map random ensemble. At least six of
eight individual high4-vs-random susceptibility effects must be positive.
Uniform high4/low4 separation must also exceed both K-shift and V-shift
separation. Final confirmation requires all gates in the same context.

Every comparison additionally requires:

- native replay relative RMS <= `1e-4`;
- no clipped, degenerate, or refinement-bound-hit layer;
- calibration relative error <= `2%`;
- accepted scale in `[0.005, 50]`;
- median paired effect >= `log(1.05)`;
- positive bootstrap lower bound or positive fraction >= `0.65`;
- seed-replicate Spearman >= `0.30`.

## 7. Server Commands

All four nodes must use the same commit and shared output directory.

### 7.1 Source analysis, preparation, and smoke

Run on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v151_signed_policy_low_tail_32gpu.sh signed_analyze

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v151_signed_policy_low_tail_32gpu.sh prepare

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v151_signed_policy_low_tail_32gpu.sh smoke
```

Do not start `core64` unless smoke prints:

```text
[v151-smoke] replay/map/refined-calibration contract: PASS ...
```

`prepare` automatically runs `signed_analyze` when the signed map is absent.
The explicit command is shown so its report can be inspected before GPU use.

### 7.2 Four-node core64

Run one command on each node:

```bash
# node 0
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v151_signed_policy_low_tail_32gpu.sh core64
# node 1
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v151_signed_policy_low_tail_32gpu.sh core64
# node 2
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v151_signed_policy_low_tail_32gpu.sh core64
# node 3
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v151_signed_policy_low_tail_32gpu.sh core64
```

After all nodes finish, run on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v151_signed_policy_low_tail_32gpu.sh analyze

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v151_signed_policy_low_tail_32gpu.sh package
```

Status is read-only:

```bash
bash scripts/run_v151_signed_policy_low_tail_32gpu.sh status
```

## 8. Required Artifacts and Debug Information

Runtime root:

```text
runs/v151_signed_policy_low_tail/
|-- signed_source/
|-- inputs/
|-- smoke/
`-- core64/
    |-- profiles/
    |-- videos/
    |-- logs/
    `-- analysis/
```

Package and push the complete directory:

```text
docs/results/v151_signed_policy_low_tail/
|-- signed_source/
|-- core/
|-- v151_probe_plan.json
`-- suite_metadata.json
```

The most important files are:

- `signed_source/report.json`: source-screen decision and exact thresholds;
- `signed_source/signed_policy_feature_audit.csv`: every factor/contrast/state;
- `signed_source/signed_scene_uniform_maps.json`: frozen signed head maps;
- `core/probe_integrity.csv`: per-probe/per-context calibration status;
- `core/group_comparisons.csv`: all fixed and ensemble comparisons;
- `core/random_map_comparisons.csv`: individual random-map controls;
- `core/intervention_specificity.csv`: uniform versus K/V specificity;
- `core/contrast_diagnostics.csv`: uniform versus boundary behavior;
- `core/downstream_observations.csv.gz`: complete observation-level data;
- `core/report.json`: exact passing contexts and final gates.

If smoke or audit fails, retain the first traceback plus:

1. maximum calibration error;
2. scale range;
3. clipped, degenerate, and refinement-bound-hit counts;
4. profile/video/log counts from `status`;
5. the signed source report and generated plan;
6. the failing `.pt` profile and its shard log.

Do not relax the 2% calibration threshold. A refinement failure should be
fixed in the calibration path or tested at a separately preregistered target.

## 9. Decisions After v151

| Result | Next action |
|---|---|
| signed branch passes | build signed, timestep-conditioned trajectory routing |
| scalar branch only passes | retain low-response tail; redesign high-group policy |
| susceptibility passes but leverage fails | do not build trajectory routing from the map |
| t500/t250 only passes | use an explicit late-denoising gate |
| specificity fails | call the signal generic history susceptibility, not prompt policy |
| both branches fail | stop static per-layer maps and move to online state-conditioned routing |

Only after a branch passes should the next experiment alter 30-second
generation. The first trajectory tests should be single-prompt long video and
AB scene replacement. ABA recall remains deferred until scene replacement and
forgetting are understood.
