# v150: Policy-Group Randomized Confirmation

## 1. Purpose

v149 produced two observations that must be separated:

1. the v145 policy top-3 group did not beat the two-map random ensemble;
2. calibrated policy leverage increased at group size four.

Two random maps are not sufficient because their downstream leverage differed
substantially. v150 therefore tests whether the top-4 policy direction is a
repeatable collective mechanism or an incidental choice of four heads.

v150 does not use PF labels to define, filter, or route any head. The only
source score is the layer-residualized v145 `full_semantic/policy_shift` score.

## 2. Frozen Hypotheses

### H1: Count-matched policy-group effect

At the same projected perturbation strength, policy top-4 must beat all of:

- policy bottom-4;
- policy middle-4;
- the geometric ensemble of eight count-matched random maps.

At least six of the eight individual top-vs-random median effects must also be
positive. All conditions must hold in the same denoising context.

### H2: Intervention specificity

The top/bottom separation under `policy_contrast` must exceed the separation
under both `key_shift` and `value_shift` in the same context. A generic response
to any history perturbation is not a policy-specific mechanism.

### H3: Strength robustness

H1 must pass at two or more projected targets from `1%`, `2%`, and `5%` in the
same context. The final x0 response at `5%` must also be at least `1.2x` the
response at `1%` for top-4, bottom-4, and the random ensemble.

## 3. Head Maps

Each layer has 12 self-attention heads.

| Group | Definition | Heads per layer |
|---|---|---:|
| `top4` | four largest policy scores | 4 |
| `bottom4` | four smallest policy scores | 4 |
| `middle4` | remaining four scores | 4 |
| `random0..7` | deterministic balanced random subsets | 4 each |

The fixed rank groups are disjoint and partition every layer. Random subsets
are sampled over all 12 heads rather than only the middle group. For every
layer:

- all eight random subsets are unique;
- no random subset equals top-4, bottom-4, or middle-4;
- every head occurs either two or three times across the eight maps.

This construction prevents one unusually strong random map or an unequal head
budget from deciding the result.

## 4. Interventions

All interventions retain the current block and preserve the newest four
historical frames for K/V shifts.

| Intervention | Operation |
|---|---|
| `key_shift` | cyclically shift only older historical K by one frame |
| `value_shift` | cyclically shift only older historical V by one frame |
| `policy_contrast` | add the directional contrast `uniform8 - recent8` |

`uniform8` and `recent8` have equal frame budgets. The contrast therefore tests
history placement rather than token count.

## 5. Calibration and Integrity

Each selected four-head group is projected through the real Self-Forcing
self-attention output projection. The resulting delta is scaled independently
at every layer, profile, probe, and context.

- runtime scale range: `[0.001, 50]`;
- accepted analysis range: `[0.005, 50]`;
- maximum target-relative error: `2%`;
- native replay tolerance: `1e-4`;
- contexts: frame 117 at nominal timesteps 1000 and 500.

Unlike v149, calibration integrity is evaluated separately for every
probe/context. A bad unrelated strength cell cannot invalidate a clean core
comparison. A comparison is eligible only when every probe it uses has zero
clipped and degenerate layers in that context.

The strict four-profile smoke requires every core probe to pass calibration.
Do not start `core64` if the smoke fails.

## 6. Experiment Grid

### 6.1 `core64`

- 32 frozen diverse MovieBench Qwen-rewritten prompts;
- two seeds per prompt, identical to v149;
- 30-second native SF trajectory;
- target `0.02`;
- 11 maps x 3 interventions = 33 probes;
- two replay contexts plus native replay = 68 records per profile.

### 6.2 `strength32`

- frozen first 16 prompts from the same set;
- two seeds per prompt;
- 30-second native SF trajectory;
- policy contrast only;
- 11 maps x 3 targets = 33 probes;
- 68 records per profile.

The saved MP4 is the native SF trajectory. Counterfactual probes are read-only
one-step replays and do not modify the generated video.

## 7. Server Commands

Use the same commit on all four 8-GPU nodes. Inputs and outputs must be on the
shared filesystem.

Prepare and run the strict smoke on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v150_policy_group_profile_32gpu.sh prepare
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v150_policy_group_profile_32gpu.sh smoke_core
```

Expected terminal line:

```text
[v150-smoke] replay/map/calibration contract: PASS scale_range=[...,...]
```

Run core on the four nodes:

```bash
# node 0
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh core64
# node 1
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh core64
# node 2
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh core64
# node 3
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh core64
```

After all four nodes finish, analyze on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v150_policy_group_profile_32gpu.sh analyze_core
```

`strength32` is independent and may run after core:

```bash
# strict strength-target smoke on node 0
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v150_policy_group_profile_32gpu.sh smoke_strength
# node 0
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh strength32
# node 1
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh strength32
# node 2
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh strength32
# node 3
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v150_policy_group_profile_32gpu.sh strength32
```

Then analyze and package:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v150_policy_group_profile_32gpu.sh analyze_strength
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v150_policy_group_profile_32gpu.sh package
```

Progress can be checked without changing the run:

```bash
bash scripts/run_v150_policy_group_profile_32gpu.sh status
```

## 8. Outputs

Runtime root:

```text
runs/v150_policy_group_confirmation/
|-- inputs/
|-- smoke_core/
|-- smoke_strength/
|-- core64/
|   |-- profiles/
|   |-- videos/
|   |-- logs/
|   `-- analysis/
`-- strength32/
    |-- profiles/
    |-- videos/
    |-- logs/
    `-- analysis/
```

Packaged results:

```text
docs/results/v150_policy_group_confirmation/
|-- core/
|-- strength/
|-- v150_policy_core_plan.json
|-- v150_policy_strength_plan.json
`-- suite_metadata.json
```

Important analysis files:

| File | Purpose |
|---|---|
| `report.json` | preregistered gates and exact qualifying contexts |
| `probe_integrity.csv` | clipping, degeneracy, target error, and scale range |
| `group_comparisons.csv` | top vs bottom, middle, and random ensemble |
| `random_map_comparisons.csv` | all eight individual random controls |
| `intervention_specificity.csv` | policy contrast vs K/V-shift separation |
| `target_response.csv` | 5% vs 1% downstream response sanity |
| `downstream_observations.csv.gz` | complete observation-level measurements |

## 9. Debug Checklist

If smoke or audit fails, retain and report:

1. the first traceback or assertion from `smoke_core/*.log` or shard logs;
2. the printed calibration scale range;
3. clipped and degenerate layer counts;
4. maximum valid calibration error;
5. profile and video counts from `status`;
6. `probe_integrity.csv` and `report.json` if analysis completed.

Do not increase the `50x` calibration cap after seeing results. A target that
requires larger amplification is an invalid cell, not evidence to relax the
gate.

## 10. Decision

| Result | Decision |
|---|---|
| Core G1/G2/G3 leverage pass | policy top-4 is a count-matched, intervention-specific candidate |
| Core passes only at t500 | use a timestep-conditioned policy gate |
| Core G1 passes but G2 fails | retain generic history susceptibility; do not claim policy heads |
| Top does not beat eight-map random ensemble | stop static policy head maps |
| Strength passes at two targets | proceed to trajectory-level method testing |
| Strength or target-response sanity fails | redesign calibration/intervention before generation tests |

Only after core and strength confirmation should the method alter a full
trajectory. The first trajectory tests remain single-prompt 30-second
generation and AB scene switching. ABA recall stays deferred until forgetting
and scene replacement are understood.
