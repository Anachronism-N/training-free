# v101: 128-Prompt Paper Ablation After the One-Video Screen

Status: implementation-ready, but **do not launch until v100 videos have been
manually reviewed**.

This document covers the broad experiment that follows
`docs/104_v100_responsive_event_cache_and_aba_fast_screen.md`.

## 1. Why this is a separate run

The v100 screen answers three mechanism-selection questions with one 30-second
video per cell:

1. Which cache should Responsive heads use?
2. Do the small add-ons improve that cache?
3. Does role-aware scene archive/recall help A-B-A generation?

It is wasteful to run 128 prompts for every unverified mechanism. v101 freezes
the selected v100 candidate and then runs the full MovieGenVideoBench-128
ablation. The v101 contract refuses mixed candidate settings, maps, prompts,
configs, or implementation revisions.

## 2. Candidate parameters

The broad runner accepts three decisions:

```text
CANDIDATE_SUPPORT    = stride | hybrid
CANDIDATE_SUPPRESS   = cyclic_sink3 | motion | motion_cyclic | recent8
CANDIDATE_TRANSITION = on | off
```

Current prior, before seeing v100:

```text
CANDIDATE_SUPPORT=stride
CANDIDATE_SUPPRESS=motion_cyclic
CANDIDATE_TRANSITION=on
```

This is a hypothesis, not a frozen final method. Set the parameters from the
blind/manual review of v100. Do not select a candidate from one automatic
metric alone.

## 3. Eight 128-prompt cells

Every cell uses the same 128 prompts, seed 0, 120 latent frames, and expected
477-frame/16-fps/832x480 output.

| Cell | Changed factor | Purpose |
|---|---|---|
| `ours_full` | none | selected complete method |
| `ablate_transition_toggle` | flip v78 gate | transition/update contribution |
| `ablate_support_route_toggle` | stride ↔ hybrid | Supportive cache contribution |
| `ablate_responsive_<route-1>` | replace Responsive cache | cache mechanism contribution |
| `ablate_responsive_<route-2>` | second replacement | cache mechanism robustness |
| `control_random_membership` | count-matched random map | role assignment necessity |
| `control_pf_aw_membership` | PF (Anchor+Wave)/Veil members | membership versus our classifier |
| `control_threshold_m0p1` or `p0p1` | alternate polarity threshold | classifier stability |

The two Responsive alternatives are selected deterministically from:

```text
cyclic_sink3, recent8, motion_cyclic, motion
```

The runner excludes the chosen full route, so no broad cell is duplicated.

## 4. What each control establishes

### 4.1 Transition toggle

The head membership and cache routes remain fixed. Only the validated v78
trust-conditioned update gate is flipped. This tests whether update timing is
an independent contribution rather than a hidden consequence of routing.

### 4.2 Supportive route toggle

The Responsive side remains fixed. Supportive heads switch between:

- `stride`: four long-horizon anchors;
- `hybrid`: two stride anchors plus two phase-aligned local anchors.

This tests whether pure long-history support is preferable to a mixed temporal
coverage budget.

### 4.3 Responsive route replacements

Membership remains the old v98 304/56 history-polarity map. Only the
Responsive cache changes. The relevant routes are:

- `cyclic_sink3`: four periodic local slots, with the same three-frame sink as
  the other candidates;
- `recent8`: no middle cache and eight recent frames;
- `motion`: four clean-value motion-event frames;
- `motion_cyclic`: two motion-event plus two periodic frames.

This isolates cache behavior from head classification.

### 4.4 Random membership

The random map has the same 304/56 role count as the old v98 map. The full
candidate cache is unchanged. A meaningful drop supports the claim that the
measured head role, not merely route capacity, matters.

### 4.5 PF-AW membership

This uses PF Anchor+Wave as Supportive and PF Veil as Responsive, but still
uses our two cache routes and exclusive cache owner. It does **not** run PF's
three routes. The comparison separates:

- PF-derived membership;
- our polarity-derived membership;
- the two-route cache mechanism.

### 4.6 Threshold control

The main map uses the zero threshold. The control uses either `-0.1` or
`+0.1`. Cache routes remain fixed. This tests whether the result survives a
reasonable threshold perturbation.

## 5. Four-node launch

All four nodes must share the repository and `OUT_ROOT`. Pull the same commit
on every node. Start rank 0 first because it writes the immutable experiment
contract; ranks 1-3 wait for it.

Each node runs eight methods on GPUs 0-7 and owns one prompt interval:

```text
rank 0: [0, 32)
rank 1: [32, 64)
rank 2: [64, 96)
rank 3: [96, 128)
```

Example for the current hypothesis:

```bash
export CANDIDATE_SUPPORT=stride
export CANDIDATE_SUPPRESS=motion_cyclic
export CANDIDATE_TRANSITION=on
export THRESHOLD_CONTROL=m0p1
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export OUT_ROOT="$PWD/runs/v101_paper_ablation_128"

# Override when the checkpoint is not in the repository default location.
export PF_CHECKPOINT=/absolute/path/to/self_forcing_dmd.pt
```

Node 0:

```bash
NODE_RANK=0 nohup python scripts/run_v101_paper_ablation_4node.py \
  > runs/v101_paper_ablation_128.node0.log 2>&1 &
```

Nodes 1-3:

```bash
NODE_RANK=1 nohup python scripts/run_v101_paper_ablation_4node.py \
  > runs/v101_paper_ablation_128.node1.log 2>&1 &
```

Replace the rank and log suffix on each node.

Use a new `OUT_ROOT` after changing any candidate parameter, source file,
prompt, map, config, or model. The runner intentionally rejects a mixed run.

## 6. Runtime evidence

Each method/shard writes:

```text
runs/v101_paper_ablation_128/
  contracts/experiment.json
  videos/<method>/*.mp4
  configs/<method>.shard<rank>.json
  logs/<method>.shard<rank>.log
  traces/<method>.shard<rank>.policy.jsonl
  traces/<method>.shard<rank>.motion.jsonl
  traces/<method>.shard<rank>.transition.jsonl
  diagnostics/<method>.shard<rank>.video.json
  diagnostics/<method>.shard<rank>.policy.json
  diagnostics/<method>.shard<rank>.motion.json
  status/<method>.shard<rank>.done.json
  status/node<rank>.summary.json
```

The completion marker is written only after:

1. inference exits successfully;
2. no known fatal signature appears in the log;
3. every expected video decodes and matches the frozen media contract;
4. the runtime labels match the selected CSV;
5. sink, recent, middle strategy, policy type, and exclusive owner match the
   requested route;
6. motion-event offsets, scores, and Responsive head counts pass audit when
   applicable;
7. transition traces exist when enabled.

## 7. Human review and metrics

First prepare the blind package:

```bash
bash scripts/postprocess_v101_paper_ablation.sh prepare
```

Review and freeze the blind package using the normal
`prepare_blind_review.py` workflow. Only then run metrics:

```bash
HUMAN_REVIEW_DONE=1 \
bash scripts/postprocess_v101_paper_ablation.sh metrics
```

The postprocessor:

1. verifies the generation contract and all 32 method/shard completion
   markers;
2. strictly audits all 128 videos per method;
3. creates a randomized blind-review package;
4. runs six VBench-Long dimensions;
5. runs the comprehensive metric suite, excluding ArcFace by default;
6. computes temporal-jump diagnostics;
7. writes aggregate JSON/CSV/Markdown results.

Disable expensive groups explicitly when needed:

```bash
RUN_COMPREHENSIVE=0 RUN_TEMPORAL=0 HUMAN_REVIEW_DONE=1 \
bash scripts/postprocess_v101_paper_ablation.sh metrics
```

## 8. Reusing existing baselines

Do not regenerate SF, PF, Echo-Forcing, or earlier PF-binary outputs if their
artifacts satisfy all of the following:

- identical MovieGenVideoBench-128 prompt file hash;
- identical seed and per-prompt reseeding;
- identical 120 latent-frame and media contract;
- compatible checkpoint;
- complete per-index video audit;
- known implementation commit and configuration.

The v101 eight-cell matrix is an ablation table around our candidate. The main
comparison table should combine `ours_full` with audited reusable SF/PF/EF
results. Record source paths and hashes instead of silently copying scores.

## 9. Decision rules

### If `ours_full` wins or ties PF within noise

Keep the two-role method and report the full ablations. The paper claim can be:

1. polarity-based functional head discovery;
2. role-conditioned long-history versus change-aware local caches;
3. clean-value motion-event selection;
4. optional trust-conditioned cache updates;
5. scene-episodic extension for prompt/scene switches.

### If a Responsive replacement beats the selected full route

Adopt that route and rerun only the affected full/control cells under a new
contract. Do not relabel the old output as the new method.

### If PF-AW membership consistently beats our membership

The cache mechanism may still be useful, but the polarity classifier is not
ready as a main contribution. Report it as analysis or replace it with a
better independently measured role axis.

### If random membership ties our membership

Do not claim functional classification. The result would indicate that cache
budget or route composition, rather than the measured head role, explains the
gain.

### If threshold control collapses

Treat the zero-threshold classifier as unstable. Use the saved continuous
scores to fit a preregistered threshold on a disjoint profiling set, then
evaluate once on MovieGenVideoBench-128.

## 10. ABA remains a separate task

This v101 runner is for single-prompt long extrapolation, which remains the
primary task. A-B-A results from v100 determine whether scene-episodic
archive/recall deserves a separate broad experiment. Do not fold scene-cache
results into the single-prompt main score.
