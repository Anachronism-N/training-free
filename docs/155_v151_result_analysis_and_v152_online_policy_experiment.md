# 155: v151 Result Correction and v152 Online Policy Profiling

Date: 2026-08-01

Status: v151 is rejected. v152 code is complete and ready for mandatory smoke
validation on the GPU server.

## 1. Corrected v151 audit

The packaged v151 result contains 64 complete profiles, 32 prompts, two seeds,
four denoising contexts, and exact native replay (`maximum relative RMS = 0`).
However, none of the four contexts satisfies the frozen calibration contract.

The earlier diagnosis in `docs/153` incorrectly interpreted the probe suffix
`_t020` as timestep 250. It denotes the **2% calibration target**. The actual
context field in the committed offender table gives:

| Context | Layers above 2% | Maximum relative error |
|---|---:|---:|
| timestep 1000 | 13 | 0.0287735 |
| timestep 750 | 14 | 0.0258164 |
| timestep 500 | 18 | 0.0246616 |
| timestep 250 | 25 | 0.0240994 |
| total | 70 / 245,760 | 0.0287735 |

Therefore the statement that only the late-denoising context failed is false.
The repaired analyzer correctly reports `intact_contexts=[]` and forces all
confirmation gates to fail.

This numerical failure is not the only reason to stop v151. The ungated values
are useful as diagnostics and reject both scientific branches:

1. The scalar low tail is consistently less locally susceptible, but its
   calibrated downstream leverage is not stable across seeds and its policy
   specificity fails.
2. The signed scene-policy high group is much more locally susceptible than
   its low group, but it has **lower** downstream leverage than the signed
   middle group in every context. The `high > middle` median log effects are
   approximately `-0.123, -0.083, -0.109, -0.090` across the four contexts.
3. Uniform-policy leverage does not separate cleanly from K-shift and V-shift.
   The static maps therefore identify response magnitude, not a policy-specific
   functional head class.

Conclusion: do not repair or rerun v151. More calibration iterations cannot
turn the observed static susceptibility ranking into policy-specific causal
leverage.

## 2. Why v152 changes the question

v149-v151 asked whether an offline, prompt-aggregated head map had exceptional
downstream leverage after an equal-RMS perturbation. That approach has now
failed repeatedly.

v152 instead asks:

> Given the current native state, can an online, physically interpretable QK
> score choose which heads should retain uniformly sampled history and which
> should use only recent history?

The target is no longer "largest perturbation." For the exact same selected
heads, v152 applies two actual, equal-budget cache approximations and asks which
one is closer to native full-window Self-Forcing:

```text
uniform8 = four uniformly sampled older frames + newest four frames
recent8  = newest eight frames

uniform advantage = log(error(recent8) / error(uniform8))
```

A positive value means `uniform8` is the better approximation. No perturbation
RMS calibration is used.

## 3. Frozen-native selector protocol

Dynamic classification must not be recomputed on an already perturbed path.
For every profile, timestep, and layer, v152 performs:

1. replay the native model state;
2. compute all selector scores on that native Q/K/V state;
3. freeze the selected head ids;
4. replay `uniform8` and `recent8` from the same input state;
5. verify that both policy replays used byte-identical scores and head ids.

Selections are cached only for the active context and cleared afterward. Logs
include selector type, direction, selected layer-0 heads, and score range.

This prevents a subtle feedback error in which an early-layer intervention
changes a later-layer score and makes the compared policies use different head
sets.

## 4. Selector hypotheses

Every dynamic selector chooses four of twelve heads independently in each
layer and state.

| Selector | Score | Role |
|---|---|---|
| `policy_error_margin` | sampled projected per-head `log(E_recent/E_uniform)` | non-deployable oracle upper bound |
| `qk_policy_margin` | sampled-Q log-mean-exp compatibility of uniform K minus recent K | online-computable primary candidate |
| `old_history_mass` | sampled native QK mass outside the newest eight frames | prior-work-style temporal baseline |

High and low tails produce two groups for every selector:

```text
oracle_uniform4 / oracle_recent4
qk_uniform4     / qk_recent4
mass_uniform4   / mass_recent4
```

Static controls are:

```text
v151 signed_high4 / signed_low4
four deterministic count-matched random maps
```

Each of the twelve groups receives both policies, giving 24 downstream probes.
All probes retain the current block and replace only the selected heads.

## 5. Data split and scale

v152 uses the 64 Qwen-rewritten MovieBench prompts unused by both v150 and
v151. It therefore does not select prompts after reading v151 outcomes.

```text
64 prompts x 2 new seeds = 128 native 30-second profiles
frame = 117
timesteps = 1000, 750, 500, 250
24 probes + native replay
100 downstream records per profile
32 GPUs = four jobs per GPU
```

The saved MP4 remains native Self-Forcing. Every cache-policy intervention is
a read-only one-step replay and cannot alter the generated trajectory.

## 6. Decision gates

The analyzer evaluates X0 error with paired prompt/seed statistics. A policy
effect qualifies only if:

```text
median preferred-policy ratio >= 1.03
bootstrap mean lower bound > 0 or win rate >= 0.65
seed-replicate Spearman >= 0.30
```

The QK candidate is confirmed only in one same context where all hold:

1. oracle uniform and oracle recent groups choose the correct policy;
2. QK uniform and QK recent groups choose the correct policy;
3. both QK groups beat the direction-matched random ensemble by at least 1%;
4. QK/oracle score Spearman is at least 0.30;
5. QK/oracle top-four and bottom-four Jaccard are each at least 0.30;
6. native replay and selector-freezing contracts pass.

Interpretation:

| Outcome | Decision |
|---|---|
| Oracle fails | Local policy preference does not propagate; stop this policy axis |
| Oracle passes, QK fails | An online opportunity exists but the proposed cheap score is inadequate |
| QK policy passes, random/alignment fails | QK is not a selective classifier; do not route generation |
| Full QK gate passes | Proceed to trajectory routing and AB scene-switch profiling |
| Old-mass only passes | Temporal mass is the stronger baseline; redesign novelty around state gating rather than claim a new head taxonomy |

Selection recurrence is reported separately. High seed recurrence with lower
cross-timestep recurrence supports a static propensity modulated by denoising
state; it is not forced as a pass criterion.

`qk_policy_margin` still needs a shared candidate bank containing the recent
and uniformly sampled keys before routing. Passing v152 establishes the score,
not zero-overhead deployment; trajectory experiments must report that bank's
memory and attention cost.

## 7. Server commands

Use the same commit on all four nodes. Run preparation and the mandatory smoke
on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only origin codex/v98-correctness-fixes

export NODE_RANK=0 NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export SF_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt

bash scripts/run_v152_online_policy_profile_32gpu.sh prepare
bash scripts/run_v152_online_policy_profile_32gpu.sh preflight
bash scripts/run_v152_online_policy_profile_32gpu.sh smoke
```

Do not start the full run unless the terminal contains:

```text
[v152-audit] PASS profiles=4 ... replay=0
[v152-smoke] frozen selector and equal-budget replay contract: PASS
```

Run one command on each node:

```bash
# node 0
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh core128
# node 1
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh core128
# node 2
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh core128
# node 3
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh core128
```

Analyze and package on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh audit
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh analyze
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v152_online_policy_profile_32gpu.sh package
```

Progress is read-only:

```bash
bash scripts/run_v152_online_policy_profile_32gpu.sh status
```

## 8. Required artifacts and debug review

Push the packaged directory:

```text
docs/results/v152_online_policy_profile/
|-- v152_probe_plan.json
|-- suite_metadata.json
`-- core/
    |-- report.json
    |-- report.md
    |-- profile_audit.csv
    |-- policy_pair_summary.csv
    |-- random_control_summary.csv
    |-- selector_alignment_summary.csv
    |-- selector_recurrence_summary.csv
    `-- compressed observation and selector tables
```

For any failure, retain:

1. the first traceback and complete failing shard log;
2. the failing `.pt` profile;
3. all `[HeadProfile] dynamic-selector` lines around the failure;
4. native replay error;
5. selector ids and score vectors for both policy probes;
6. actual `uniform8` and `recent8` frame indices.

## 9. Scope boundary

Native Self-Forcing exposes a 21-frame attention window. At frame 117, the
policy comparison uses older versus recent frames **inside that window**. v152
can establish state-conditioned history policy selection, but not persistent
long-range recall by itself.

If the QK branch passes, the next profiling experiment should combine it with:

1. an explicit persistent archive outside the native window;
2. AB prompt switching, where a prompt boundary supplies an online episode
   signal and recent/forgetting policies are tested near versus after the
   boundary;
3. only then, a trajectory-level cache method.

ABA recall remains later because AB forgetting must first be understood.
