# Current Idea and v86 Role-Conditioned Transition Plan

> Date: 2026-07-24
> Status: v78 is the validated paper core; v86 is a falsifiable extension.
> Recommended flexible title: **Trust the Transition: Reliability-Gated Cache
> State Promotion for Training-Free Long Video Extrapolation**
>
> **Protocol update:** `docs/87_transitioncache_method_and_16x30s_protocol.md`
> supersedes the three-prompt screen below. The next result-producing run is
> 16 methods x 16 complex single prompts x 30 seconds, followed by blind review
> and VBench-Long.

## 1. Evidence after the latest server results

The project should stop treating direct archive recall as the main line.

| Observation | Evidence | Decision |
|---|---|---|
| v78 is robust | DINO 0.8512/0.8425 at seeds 2/3; average 0.8468, about +0.021 over the reported PF reference | Keep as the paper core |
| v78 is visually strong | Human review ranks v78 first and PF second; no severe hallucination or duplicated subject | Preserve its read topology |
| ProbeCache direct recall is unsafe | All ProbeCache variants show non-ID hallucinations; inverse/random controls can create polygonal noise or duplicated subjects | Stop direct archive K/V injection |
| The learned roles are measurable | Profile/replica agreement 84.7%, kappa 0.557 | Roles may be used as a control signal |
| The role boundary is not fully stable | Persistent count changes from 99 to 56; Jaccard is 0.476 | Do not make role labels a required claim yet |
| Learned labels did not beat PF labels in direct recall | PF-binary and random controls match or exceed learned DINO in v82 | Require causal label controls in every promotion experiment |
| Lower jump can hide motion loss | ProbeCache lowers temporal jump, while human review sees motion and artifact differences | Always report dynamic degree and human motion review |

The strongest current conclusion is not "retrieval solves long video." It is:

> Long-horizon AR generation benefits from controlling when a generated state is
> admitted into persistent attention memory. Directly recalling old K/V can
> preserve the subject while corrupting non-subject regions.

## 2. Current method

### 2.1 Cache composition

The PF cache topology remains unchanged:

```text
sink / anchor  +  PF middle policies  +  recent native window
immutable         cyclic/stride/merge    always updated
```

The method never adds an archive to the attention read path. It intercepts only
the clean-block write to the PF middle region. Sink capture and recent updates
remain on their original PF paths.

### 2.2 Uniform trust-conditioned transition (validated v78)

For head \(h\) at AR block \(t\), pool normalized clean K/V into descriptor
\(z_{t,h}\). Let \(\tilde z_{t,h}\) be the existing noisy-pass descriptor and
\(a_{t-1,h}\) the last admitted clean descriptor:

```text
shock(t,h)   = cosine_distance(z(t,h), a(t-1,h))
denoise(t,h) = cosine_distance(z(t,h), z_noisy(t,h))
trust(t,h)   = exp(-w_s * shock(t,h) - w_d * denoise(t,h))
```

A middle write must pass reliability, novelty, age, stagger and per-layer budget
rules. A forced max-age refresh prevents indefinite staleness. This uses
existing SF/PF passes and adds no model forward.

Validated v78 configuration:

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

### 2.3 Role-conditioned asynchronous promotion (v86 hypothesis)

v86 does not change what a head reads. It only gives different promotion
clocks to counterfactually profiled roles:

```text
persistent:
  novelty threshold = base threshold * persistent scale
  max age            = persistent max age
  budget bias        = 0

reactive:
  novelty threshold = base threshold * reactive scale
  max age            = reactive max age
  budget bias        = small positive bias
```

Balanced initial policy:

```text
persistent scale=1.5, max_age=8
reactive   scale=0.5, max_age=4, utility_bias=0.10
```

Interpretation:

- persistent heads retain a trusted middle state longer and replace it only
  with sufficiently novel evidence;
- reactive heads refresh sooner and receive bounded priority when the 75%
  write budget is contested;
- unknown labels use the uniform v78 policy;
- layer routing is optional and uses an exclusive `[start, end)` interval.

The role CSV is independent from PF's policy CSV. Loading a transition-role CSV
does not alter PF capacities, compositions, sink sizes, recent windows, or
middle read strategies. Missing or malformed role files fail immediately.

## 3. Paper story and novelty boundary

### 3.1 Defensible story

1. **Problem:** native AR history is updated with states of unequal reliability,
   so local generation errors can become persistent memory.
2. **Insight:** SF/PF already expose noisy and clean states. Their disagreement,
   together with change from the last admitted state, gives a free online
   estimate of promotion reliability.
3. **Method:** reliability-gated cache state promotion controls the lifecycle of
   the existing middle cache without an extra forward or a new read archive.
4. **Optional extension:** intervention-derived head roles define asynchronous
   promotion clocks, decoupling long-lived structure from rapidly changing
   motion evidence.
5. **Result requirement:** improve identity and/or temporal continuity without
   buying the gain through static videos, background hallucination, or direct
   retrieval artifacts.

### 3.2 Relationship to prior work

| Work | Prior mechanism | Our required distinction |
|---|---|---|
| [Pyramid-Forcing](https://arxiv.org/abs/2605.13111) | Offline head categories and per-head sink/middle/recent read policies | PF decides what history is read; we control whether a newly generated state is trusted enough to enter its middle memory |
| [Forcing-KV](https://arxiv.org/abs/2605.09681) | Static/dynamic heads and head-specific KV compression | Our classification criterion contrasts counterfactual remote-history utility with prompt sensitivity, and targets generation-state admission rather than compression |
| [Head Forcing](https://arxiv.org/abs/2605.14487) | Local/anchor/memory heads, hierarchical memory and dynamic episodic updates | Our classification signal, resulting partition, noisy-clean trust controller and write-lifecycle intervention differ; we add no episodic read path |
| [Echo-Forcing](https://arxiv.org/abs/2605.16003) | Scene memory preserve/recall/forget | v86 performs no scene snapshot retrieval |
| [IAMFlow](https://arxiv.org/abs/2605.18733) | Identity-aware entity/state memory | v86 has no entity detector or identity memory |

No code from these external methods is copied into v86. The implementation
extends this repository's existing v78 controller. All borrowed problem
formulations and head-specialization precedents must be cited.

### 3.3 Claim decision

- If learned/replica roles consistently beat v78 and PF-binary controls, the
  role-conditioned clock can be a method contribution.
- If PF-binary is equal or better, retain v78 as the method and present the
  profile only as analysis.
- If no v86 cell beats v78, do not add role conditioning to the paper method.
- Direct ProbeCache retrieval is a documented negative result, not a hidden
  component of the final method.

## 4. Code implemented for v86

| File | Purpose |
|---|---|
| `third_party/Pyramid-Forcing/pyramidkv/transition.py` | Role-specific novelty threshold, max age, budget utility and trace fields |
| `third_party/Pyramid-Forcing/pyramidkv/config.py` | Strict independent transition-role matrix loader |
| `third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py` | Pass transition labels without changing PF head labels |
| `third_party/Pyramid-Forcing/pipeline/pyramidkv_config.py` | Default-off role configuration |
| `third_party/Pyramid-Forcing/inference.py` | CLI controls and missing-path failure |
| `scripts/build_transition_role_consensus.py` | Fail-closed primary/replica consensus; disagreements become neutral v78 heads |
| `scripts/run_v86_role_transition_16gpu.sh` | Smoke, 16-cell screen, multi-seed confirm, ultralong and switch runs |
| `scripts/summarize_cache_transition_trace.py` | Acceptance, reasons and thresholds summarized by role |
| `scripts/postprocess_v86_role_transition.sh` | Review-first comprehensive, jump, ABA and optional VBench-Long metrics |

The existence of prior head classifications does not invalidate a new
classification contribution. We do not claim first use of head specialization;
we test whether a different signal, criterion, partition and cache intervention
produce a distinct causal benefit.

New trace fields:

```text
role_conditioning_active
head_labels
head_roles
effective_min_novelty
effective_max_age
utility
commit_mask
reasons
reliability
shock
denoise_disagreement
age_before
```

The debug log prints per-block role acceptance counts for layer 0. The JSONL
trace records every layer and is the authoritative diagnosis source.

## 5. Experiment funnel

### 5.1 P0 smoke

This verifies PF, v78, role-neutral and learned-balanced paths:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
bash scripts/run_v86_role_transition_16gpu.sh smoke
```

Required checks:

- four cells produce the expected videos;
- all transition traces cover 30 layers;
- `learned_neutral` reports persistent/reactive roles but matches v78 behavior;
- no `KeyError`, missing label fallback, OOM or non-finite metric.

### 5.2 P1 16-GPU causal screen

```bash
bash scripts/run_v86_role_transition_16gpu.sh screen
```

Each cell generates 16 complex single-prompt 30-second videos. The matrix
contains:

```text
native SF, PF, Echo, v78, learned-neutral,
learned/replica/consensus/PF-binary/inverse/random balanced policies,
conservative/open/age-only ablations,
learned early/late depth routes.
```

Review videos blind before metrics. For every prompt record:

```text
ID persistence; background persistence; duplicate subject; polygon/noise;
flashback; acceleration jump; camera motion; action continuation; freeze.
```

Then run:

```bash
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v86_role_transition.sh screen
```

VBench-Long is enabled by default and evaluates all 16 methods in parallel on
GPUs 0-15. See `docs/87_transitioncache_method_and_16x30s_protocol.md` for the
authoritative matrix, prerequisites, debug invariants and decision rules.

### 5.3 P2 four-seed confirmation

Only run after a screen cell passes all gates:

```bash
bash scripts/run_v86_role_transition_16gpu.sh confirm
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v86_role_transition.sh confirm
```

The fixed confirm matrix is PF, v78, learned-balanced and PF-binary-balanced at
seeds 0-3 over 12 complex prompts. If another screen policy wins, override the
corresponding role CSV or update the fixed policy only after recording the
screen decision.

### 5.4 P3 task coverage

Single-prompt ultralong is the primary task:

```bash
bash scripts/run_v86_role_transition_16gpu.sh ultralong
HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 \
  bash scripts/postprocess_v86_role_transition.sh ultralong
```

Prompt/scene switch remains secondary:

```bash
bash scripts/run_v86_role_transition_16gpu.sh switch
HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 \
  bash scripts/postprocess_v86_role_transition.sh switch
```

## 6. Predeclared promotion gates

A v86 policy is promoted only if all conditions hold:

1. role-neutral reproduces v78 within `0.003` DINO and `0.03` temporal jump;
2. learned and replica label maps have the same qualitative direction;
3. learned beats inverse and random controls on both artifact review and
   min-DINO;
4. learned is not worse than v78 by more than `0.005` average DINO or `0.01`
   min-DINO;
5. temporal jump improves without lower VBench dynamic degree or visible freeze;
6. no duplicated subject, persistent polygon noise, or repeated flashback in
   the promoted cell;
7. traces show nontrivial persistent/reactive acceptance differences and no
   repeated forced-budget deferral beyond `configured max age + 2` blocks;
8. the result holds over at least three seeds and is not driven by one prompt.

The decisive comparison is learned roles versus PF-binary roles. Beating random
or inverse labels is necessary but not sufficient.

## 7. What to return from the server

```text
runs/v86_role_transition_*/run_manifest.env
runs/v86_role_transition_*/configs/
runs/v86_role_transition_*/logs/
runs/v86_role_transition_*/traces/
runs/v86_role_transition_*/diagnostics/
runs/v86_role_transition_*/metrics/
human review sheet with frozen method-blind scores
exact git commit and command
```

For diagnosis, first compare `acceptance_by_role`, rejection reasons,
effective max age, and per-layer acceptance. Then align the first visible
artifact time with the nearest transition block. Do not tune from aggregate
DINO alone.
