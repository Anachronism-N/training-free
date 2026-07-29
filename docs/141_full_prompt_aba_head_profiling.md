# v141: Full-Prompt A-B-A Head Profiling

## 1. Motivation

v134 changes one prompt factor and uses a globally rewritten paraphrase as its
control. The semantic edit can be smaller than the rewrite in text-embedding
space, so the original CPHI gate can fail even when the model reacts to prompt
changes.

Real long-video scene control is stronger:

```text
prompt A -> prompt B -> prompt A
```

v141 profiles this event on a real A-B-A generation trajectory. It does not
infer a switch by comparing unrelated videos.

## 2. Hypotheses

### H1: switch-responsive history use

Some heads change how they interact with the same self-attention history when
the complete prompt changes. Their response should exceed a meaning-preserving
rewrite of the active prompt.

### H2: stale-history conflict

At the A->B and B->A boundaries, the native sliding window initially contains
only or mostly features generated under the previous prompt. Switch-responsive
heads should be strongest near this mismatch and weaken after current-episode
frames enter the window.

### H3: semantic specialization

Two controlled switch types may produce different head responses:

```text
scene_action:
    identity, appearance, and style remain fixed
    action, scene, objects, camera, and atmosphere change

identity_scene:
    style remains fixed
    identity, appearance, action, scene, objects, camera, and atmosphere change
```

This separates scene/action sensitivity from full subject-and-scene
sensitivity without claiming semantic labels in advance.

### H4: context specialization

The same response may vary by layer, denoising timestep, boundary direction,
and distance from the boundary. A static binary map is allowed only if the
held-out aggregate is stable.

## 3. Generation contract

Suite:

```text
32 controlled prompts
16 subject families
2 switch types per family
120 latent frames, approximately 30 seconds
seed 0
native Self-Forcing 21-frame sliding window
```

Schedule:

```text
A1: frames 0-38
B:  frames 39-77
A2: frames 78-119
```

At boundaries:

```text
cross-attention cache: reset for the new text condition
self-attention K/V: native persistence, no reset
LifeCache/structured memory/head routing: disabled
```

Logs must contain exactly two lines per video:

```text
[PromptSchedule] ... frame=39 self_cache=native_persist crossattn=reset
[PromptSchedule] ... frame=78 self_cache=native_persist crossattn=reset
```

## 4. Same-state counterfactual branches

At every selected state, the base forward has already used the active schedule
prompt. Four read-only shadow forwards reuse the exact same:

- noisy or clean latent;
- timestep;
- native self-attention history;
- RNG state;
- cache indices.

Branches:

```text
exact_a
exact_b
paraphrase_a
paraphrase_b
```

For an A state:

```text
matched parity = exact_a
full switch = exact_b
local control = paraphrase_a
```

For a B state:

```text
matched parity = exact_b
full switch = exact_a
local control = paraphrase_b
```

The matched branch must reproduce the base head signatures. This detects
shadow-cache, RNG, or conditioning errors before interpreting sensitivity.

## 5. Sampling

Frames:

```text
36, 39, 42, 75, 78, 81, 117
```

They correspond to pre-switch, switch, post-switch, and late-A2 states.

At each frame:

```text
noisy nominal timesteps: 1000, 500
clean context: timestep 0
```

Expected per video:

```text
21 base states
5 branches including base
105 captured calls
3150 layer records
profile version 5
```

All 30 layers and 12 heads are recorded.

## 6. Primary score

For residual history interaction:

```text
R_hist =
    log((distance(full_switch_residual, base) + eps)
        /(distance(local_paraphrase_residual, base) + eps))
```

For direct query displacement:

```text
R_query =
    log((distance(full_switch_query, base) + eps)
        /(distance(local_paraphrase_query, base) + eps))
```

Primary:

```text
P_switch = R_hist - R_query
```

Natural zero:

```text
P_switch > 0:
    full prompt switching changes history use beyond its direct effect on Q

P_switch <= 0:
    no excess prompt-conditioned history interaction
```

The analyzer also records:

- native-output and current-key ratios;
- temporal-attention JS and Wasserstein changes;
- mass assigned to previous-episode frames;
- response by switch type, episode, boundary phase, layer, and timestep.

Layer 0 is excluded from threshold fitting and forced stable.

## 7. Held-out classification

Discovery and validation are split by subject family:

```text
even family indices: discovery
odd family indices: validation
```

The primary zero-threshold split requires:

```text
matched exact-shadow median <= 1e-5
matched exact-shadow p99 <= 1e-3
full-switch residual median > local-paraphrase median
discovery/validation rank Spearman >= 0.60
zero-label agreement >= 0.80
validation minority fraction >= 0.05
heads within 0.1 discovery-IQR of zero <= 0.20
```

Otsu and two-component GMM thresholds are fitted on discovery only and remain
diagnostics.

## 8. Run order

First run one smoke video:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh prepare
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh preflight
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh smoke
```

The smoke is mandatory. Review:

```text
runs/v141_full_prompt_switch_profile/smoke/smoke.log
runs/v141_full_prompt_switch_profile/smoke/videos/
```

Check that A, B, and returned A are visible and that the video has no polygon
noise. The final smoke line must be:

```text
[v141-smoke] schedule, branches, states, and native-cache persistence: PASS
```

Then launch all four nodes, changing only `NODE_RANK`:

```bash
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh profile
```

Use ranks 0, 1, 2, and 3 once. Each of the 32 GPUs receives one video.

After completion on node 0:

```bash
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh audit
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh analyze
bash scripts/run_v141_full_prompt_switch_profile_32gpu.sh package
```

## 9. Outputs

Primary:

```text
runs/v141_full_prompt_switch_profile/analysis/
  analysis_summary.md
  analysis_report.json
  head_axes.csv
```

Specialization:

```text
head_switch_type_axes.csv
head_episode_axes.csv
```

Correctness:

```text
profile_contract_audit.csv
```

The large `state_observations.csv` remains outside the Git package.

## 10. Interpretation boundary

A passed v141 gate supports a prompt-switch-responsive versus switch-stable
head taxonomy that is independent of PF's temporal QK-sign classes. It does
not yet establish the final cache policy.

The next causal experiment should alter episode visibility only for the
frozen top/bottom groups and compare count-matched random and reversed groups.
For single-prompt extrapolation, prompt response should be combined with
v136/v138 temporal and history-specificity evidence rather than used as the
only cache-design axis.
