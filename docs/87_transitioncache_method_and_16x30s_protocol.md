# TransitionCache: Current Method and 16 x 30s Protocol

> Date: 2026-07-24
> Status: implementation complete; the 16-prompt causal screen is the next run.
> Primary task: training-free 30-second single-prompt video extrapolation.

## 1. The current idea in one sentence

**TransitionCache classifies attention heads by their counterfactual output
response to prompt and history interventions, then uses online diffusion
reliability to give the resulting roles different clocks for promoting clean
K/V states into an existing long-video cache.**

The method does not claim that head specialization itself is new. A
classification method can still be a contribution when its input signal,
criterion, resulting partition, intervention target, and measured effect differ
from previous classifications. The experiments must demonstrate those
differences rather than relying on wording.

The current method has two levels:

1. **v78 Trust-Conditioned Cache Transition**, already supported by server
   results, controls whether a generated clean state is reliable enough to
   replace PF's current middle-cache state.
2. **v86 Role-Conditioned Transition**, the hypothesis tested here, assigns
   different update clocks to counterfactually profiled persistent and reactive
   heads. Uncertain heads abstain and use v78 unchanged.

## 2. Cache state and lifecycle

### 2.1 What is stored

TransitionCache preserves Pyramid-Forcing's original attention read topology:

```text
sink / anchor  +  PF middle cache  +  recent native window
immutable         controlled here    always refreshed
```

- **Sink/anchor:** captures the original long-range context according to PF.
  TransitionCache never overwrites it.
- **Middle:** PF's cyclic/stride/merge state. TransitionCache intercepts only
  clean writes into this region.
- **Recent:** the newest local context. It stays on PF's original update path.
- **No direct archive:** old K/V is not injected from a separate episodic
  archive. ProbeCache showed that this can preserve identity while producing
  background flashbacks, polygonal noise, or duplicated subjects.

For each layer/head and CFG sequence, the controller also stores compact
descriptors:

```text
noisy candidate descriptor
clean candidate descriptor
last accepted clean descriptor
age since last acceptance
```

These descriptors decide an update; they are not another attention memory.

### 2.2 Online trust

For clean candidate \(z_{t,h}\), same-block noisy descriptor
\(\tilde z_{t,h}\), and last accepted descriptor \(a_{t-1,h}\):

```text
shock(t,h)   = cosine_distance(z(t,h), a(t-1,h))
denoise(t,h) = cosine_distance(z(t,h), z_noisy(t,h))
trust(t,h)   = exp(-shock(t,h) - 2 * denoise(t,h))
novelty(t,h) = shock(t,h)
```

The noisy and clean passes already exist in the generation process, so this
signal adds no model forward. A candidate can be rejected for low trust, low
novelty, minimum interval, asynchronous phase, or per-layer write budget. A
max-age rule forces stale entries to refresh.

The validated v78 base is:

```text
mode=full
min_reliability=0.55
min_novelty=0.01
max_commit_fraction=0.75
stagger_period=1
max_age_blocks=6
branches=both
denoise_weight=2
```

The write budget ranks eligible candidates using:

```text
utility = trust + 0.25 * normalized_age + 0.25 * novelty + role_bias
```

This makes cache promotion asynchronous across heads instead of replacing all
middle states at the same AR boundary.

## 3. Counterfactual head classification

### 3.1 Measurements

The profiler records paired per-head attention-output sketches under two
controlled intervention families:

1. **Prompt intervention:** identical latent/history conditions with perturbed
   prompt evidence. The median relative output change is prompt sensitivity.
2. **History intervention:** identical prompt/latent conditions with full
   versus recent-only history. The median relative output change is remote
   history utility.

For a paired output sketch \(o^L_h,o^R_h\):

```text
relative_difference(h) =
  ||o_left(h) - o_right(h)|| /
  (0.5 * (||o_left(h)|| + ||o_right(h)||) + epsilon)
```

Prompt sensitivity and remote utility are independently normalized within each
layer using robust median/MAD statistics:

```text
role_score(h) = z(remote_utility(h)) - z(prompt_sensitivity(h))
```

A deterministic binary k-means partition gives:

- **persistent (+1):** relatively stronger dependence on remote history than
  on prompt perturbation;
- **reactive (-1):** relatively stronger prompt response or weaker remote
  history dependence.

Bootstrap agreement measures within-profile stability. A fully independent
replica profile measures reproducibility. Primary/replica disagreements become
**neutral (0)** in the consensus map, which falls back to v78.

### 3.2 Role-conditioned cache clocks

The balanced policy under test is:

| Role | Novelty threshold | Forced refresh age | Budget bias |
|---|---:|---:|---:|
| persistent | `1.5 * base` | 8 blocks | 0 |
| reactive | `0.5 * base` | 4 blocks | +0.10 |
| neutral | base | 6 blocks | 0 |

Persistent heads therefore retain an accepted long-range state until stronger
evidence appears. Reactive heads admit new motion/prompt evidence sooner.
Classification changes only the lifecycle of PF's middle writes; it does not
reuse PF's labels or modify PF's per-head read policies.

## 4. What may be claimed as innovation

The paper should test and, only if supported, claim these connected points:

1. **Counterfactual functional head classification.** Heads are partitioned by
   the contrast between output-level remote-history utility and prompt
   sensitivity, not by a hand-written semantic label or only an observed
   temporal attention pattern.
2. **Diffusion-trajectory trust promotion.** Same-block noisy/clean
   disagreement and last-accepted transition shock determine whether a clean
   state is promoted into persistent cache.
3. **Role-conditioned asynchronous lifecycle.** Head roles control write
   timing, admission threshold, forced refresh, and budget priority rather than
   merely choosing a compression ratio or a read region.
4. **Uncertainty-aware abstention.** Independent profiles are combined by
   consensus; unstable labels explicitly fall back to the validated uniform
   controller.
5. **Artifact-aware cache design.** The final path deliberately excludes direct
   archive K/V retrieval after controlled experiments showed identity gains can
   coexist with non-identity hallucination.

Point 1 remains a legitimate possible innovation even though earlier work also
classifies heads. Novelty depends on the exact classification problem and
evidence, not on being the first paper to use any head categories.

## 5. Difference from related work

| Work | Existing idea | TransitionCache difference |
|---|---|---|
| [Pyramid-Forcing](https://arxiv.org/abs/2605.13111) | Three temporal head patterns and tailored sink/middle/recent read policies | Our label is computed from paired prompt/history output interventions; the resulting role controls state promotion into PF's middle cache, not PF's read composition |
| [Forcing-KV](https://arxiv.org/abs/2605.09681) | Static/dynamic heads for head-specific KV handling and efficiency | Our score contrasts remote utility with prompt sensitivity and targets long-horizon cache reliability rather than KV compression |
| [Head Forcing](https://arxiv.org/abs/2605.14487) | Local/anchor/memory heads, fast/episodic memory, novelty-based updates | We use two counterfactual output responses, noisy-clean trust, asynchronous promotion, and consensus abstention without an episodic read path |
| [Echo-Forcing](https://arxiv.org/abs/2605.16003) | Scene snapshot preserve/recall/forget | We do not retrieve scene snapshots; Echo remains an external baseline |
| [IAMFlow](https://arxiv.org/abs/2605.18733) | Identity-aware entity/state memory | We use no entity detector, VLM identity extraction, or explicit entity memory |

Required academic wording:

- do not claim the first head-specialized long-video cache;
- do claim the exact classifier, score, lifecycle intervention, and observed
  partition differences if the experiments support them;
- cite all borrowed problem formulations and baselines;
- report negative direct-recall results instead of hiding them;
- do not describe v86 as validated before the causal controls pass.

## 6. Prior experimental conclusions

### 6.1 What failed

- LifeCache/HREM side memory was active in logs but visually matched native SF
  and collapsed at the same time. The intervention was too weak.
- Commit Forcing v74 was visible but introduced simplification, freezing, and
  jumps while adding about 50% more forwards. Its multiscale v76 extension was
  worse.
- ProbeCache direct archive recall often retained identity and reduced temporal
  jump, but every reviewed variant produced non-ID hallucination. Inverse
  labels produced polygon noise; random labels duplicated subjects; learned
  labels could cause flashbacks or background corruption.

### 6.2 What currently works

- In the original v78 screen, PF DINO was `0.8317` and v78 was `0.8300`;
  temporal jump improved from `1.7228` to `1.6449` (about 4.5%) with a 58%
  acceptance rate.
- v78 seeds 2/3 scored `0.8512/0.8425`, average `0.8468`, versus the previously
  reported PF seed-0 reference `0.8263`. This is encouraging but is not a
  matched-seed comparison.
- Blind human review ranked v78 and PF as the strongest methods, without the
  severe hallucination modes seen in direct recall.

### 6.3 What remains unproven

- Primary/replica label agreement is `84.7%`, kappa `0.557`; persistent and
  reactive Jaccard are `0.476/0.823`. The signal is reproducible but the
  persistent boundary is not fully stable.
- In the v82 direct-recall control, average DINO was v78 `0.8827`,
  PF-binary `0.8506`, random `0.8515`, learned `0.8170`, inverse `0.7480`.
  Learned roles were not causally superior in that unsafe read path.
- The new question is narrower: can the same roles improve the safer cache
  **write lifecycle**? The 16-prompt screen is designed to answer it.

## 7. Immediate 16-GPU experiment

### 7.1 Scope

The main screen is intentionally the first result-producing run. No 6-second
scientific screen is required.

```text
16 methods x 16 complex single prompts x 30 seconds x seed 0
= 256 videos
```

Each GPU runs one method over the same 16 prompts:

| GPU | Method | Purpose |
|---:|---|---|
| 0 | `sf_native` | native Self-Forcing baseline |
| 1 | `pf` | official Pyramid-Forcing baseline |
| 2 | `echo_pc` | official Echo-Forcing baseline |
| 3 | `v78` | validated uniform transition |
| 4 | `learned_neutral` | labels loaded but policy neutral; wiring control |
| 5 | `learned_balanced` | primary learned roles |
| 6 | `replica_balanced` | independent-profile replication |
| 7 | `consensus_balanced` | disagreement-abstaining role map |
| 8 | `pf_binary_balanced` | strongest prior-label control |
| 9 | `inverse_balanced` | semantic-direction control |
| 10 | `random_balanced` | partition-size/randomness control |
| 11 | `learned_conservative` | smaller role contrast |
| 12 | `learned_open` | larger reactive update contrast |
| 13 | `learned_age_only` | isolates refresh-age effect |
| 14 | `learned_early` | layers `[0,15)` only |
| 15 | `learned_late` | layers `[15,30)` only |

The prompt suite is
`prompts/v86_single_long_complex_16.txt`. It covers humans, animals, vehicles,
tools, deformable motion, camera motion, repeated structures, water, smoke,
reflections, and difficult background continuity. Every line is one continuous
prompt; there is no prompt switching.

### 7.2 Server prerequisites

Expected repositories and checkpoints:

```text
training-free/third_party/Self-Forcing/
  configs/self_forcing_dmd.yaml
  checkpoints/self_forcing_dmd.pt
  wan_models/Wan2.1-T2V-1.3B/

training-free/third_party/Pyramid-Forcing/
  configs/pyramid-forcing.yaml
  checkpoints/self_forcing_dmd.pt
  wan_models/Wan2.1-T2V-1.3B/

training-free/third_party/Echo-Forcing/
  configs/self_forcing_dmd.yaml
  checkpoints/self_forcing_dmd.pt
  wan_models/Wan2.1-T2V-1.3B/
```

The learned, replica, consensus, PF-binary, inverse, and random role CSVs must
exist at the defaults recorded by `scripts/run_v86_role_transition_16gpu.sh`.
The launcher fails before generation when any required baseline or label file
is missing.

### 7.3 Commands

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
bash scripts/run_v86_role_transition_16gpu.sh screen
```

The launcher writes:

```text
runs/v86_role_transition_screen/
  run_manifest.env
  configs/
  logs/
  traces/
  diagnostics/
  <method>/*.mp4
```

Each method directory must contain exactly 16 videos. To prevent partial or
duplicate outputs from contaminating metrics, the launcher refuses a nonempty
incomplete directory. Resume completed methods normally; use a new clean
`OUT_ROOT` for any failed method rerun.

After generation, freeze and complete blind human review before metrics:

```bash
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v86_role_transition.sh screen
```

Postprocessing computes the repository's comprehensive and temporal-jump
diagnostics, then launches VBench-Long for all 16 methods in parallel on GPUs
0-15. VBench-Long is enabled by default. Override only when necessary:

```bash
VBENCH_ROOT=/path/to/VBench \
VBENCH_GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v86_role_transition.sh screen
```

VBench-Long dimensions:

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
motion_smoothness
dynamic_degree
```

## 8. What to inspect

### 8.1 Human review first

For every prompt, compare methods blind and record:

```text
identity persistence
background persistence
action continuation
camera-motion continuation
duplicate subject
polygon/geometric noise
old-scene flashback
acceleration or boundary jump
freezing or motion collapse
```

Do not accept a method because DINO improves while camera/action motion
collapses.

### 8.2 Mechanism logs

For each transition method, inspect:

```text
acceptance rate by persistent/reactive/neutral role
low_reliability and low_novelty rejection counts
forced_max_age count
budget_deferred count
mean trust, shock, and noisy-clean disagreement
effective novelty threshold and max age
per-layer acceptance
early/late layer differences
non-finite values or missing layers
```

Expected invariants:

- `learned_neutral` is numerically and visually close to v78;
- learned and replica roles change acceptance in the same direction;
- consensus has neutral heads and remains close to learned/replica behavior;
- inverse/random controls do not match a real role gain;
- all 30 layers emit traces;
- no head remains deferred beyond its max age because of budget pressure.

## 9. Decision after the run

Promote role conditioning only when:

1. learned and replica show the same qualitative visual and metric direction;
2. learned or consensus beats inverse and random controls;
3. learned or consensus is competitive with both v78 and PF-binary;
4. subject/background consistency does not come from lower dynamic degree;
5. no systematic duplicate, polygon, flashback, or freeze artifact appears;
6. traces show a nontrivial role-conditioned acceptance difference.

Interpretation:

- **Learned/consensus wins:** use all five innovation points in Section 4 and
  proceed to matched-seed confirmation.
- **PF-binary wins:** keep trust promotion as the method; head profiling becomes
  analysis unless the classification can be improved with new evidence.
- **All role policies tie v78:** publishable core remains v78; do not retain
  role conditioning merely for story complexity.
- **PF remains best:** diagnose trust and update timing from traces before
  adding another memory mechanism.

## 10. Files to return for analysis

```text
runs/v86_role_transition_screen/run_manifest.env
runs/v86_role_transition_screen/configs/
runs/v86_role_transition_screen/logs/
runs/v86_role_transition_screen/traces/
runs/v86_role_transition_screen/diagnostics/
runs/v86_role_transition_screen/metrics/
blind human review sheet
exact git commit and commands
```

The first analysis should align the first visible failure time with the nearest
transition block and compare role acceptance, trust, shock, denoise
disagreement, age, and rejection reason. Aggregate VBench-Long alone is not
sufficient to decide the mechanism.
