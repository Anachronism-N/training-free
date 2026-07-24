# Post-v86 Analysis and v90 Experiment

> Date: 2026-07-24
> Status: v86 is complete; v90 code is ready for the next 16-H20 run.
> Primary task: 30-second, single-prompt, training-free extrapolation.

## 1. Current conclusion

The current paper candidate is still **v78 Trust-Conditioned Cache
Transition**, not v86 role-conditioned transition.

The defensible result is:

```text
PF provides the strong sink/middle/recent read topology.
v78 controls which generated clean states may enter PF's middle cache.
The controller uses free noisy/clean trajectory evidence, bounded
asynchronous writes, and forced refresh, with no extra model forward.
```

On the matched seed-0 v86 screen over 16 prompts:

| Method | DINO | min DINO | Human result |
|---|---:|---:|---|
| v78 | **0.8536** | 0.7873 | best overall; no specific severe artifact |
| PF binary balanced | 0.8529 | **0.7912** | good ID/BG; more camera motion |
| PF | 0.8496 | 0.7576 | strong baseline |
| learned balanced | 0.8475 | 0.7651 | duplicated subject on reviewed prompt |
| SF native | 0.7848 | 0.6850 | degrades and collapses |

Thus v78 beats PF by `+0.0040` DINO under the same prompt suite and seed.
This is positive but small. It must be confirmed with matched seeds before it
is written as a robust multi-seed gain.

The older `+0.021` statement compares v78 seeds 2/3 against a reported PF
seed-0 reference. It is encouraging evidence, but it is **not** a matched-seed
estimate and must not be used as the primary statistical claim.

## 2. What v86 established

### 2.1 Supported

1. **Uniform transition remains useful.** v78 is the best DINO cell and the
   best reviewed visual cell.
2. **PF's head partition contains a useful motion prior.** PF-binary is close
   to v78 on DINO/min-DINO and appears more dynamic in human review.
3. **The learned profile is reproducible.** Primary and replica results are
   close (`0.8475` and `0.8489` DINO).
4. **Label direction affects failure type.** Inverse labels retain high DINO
   but cause rapid background change and a limb-through-wall physics failure.
5. **DINO alone is insufficient.** Inverse ranks near the top by DINO while
   being visually unsafe.

### 2.2 Rejected

The current counterfactual persistent/reactive classifier is not causally
superior for cache writes:

```text
learned 0.8475 < inverse 0.8519
learned 0.8475 ~= random 0.8465
learned 0.8475 < PF binary 0.8529
learned 0.8475 < uniform v78 0.8536
```

The classifier may still be an interesting analysis result, but it cannot be
presented as a beneficial method component. The general fact that previous
work classifies heads does not prevent a new classifier from being novel; the
specific problem here is empirical, not rhetorical: this classifier did not
improve the target intervention.

### 2.3 Failure hypothesis

Hard role clocks simultaneously change novelty thresholds, forced-refresh
ages, and budget bias. Learned, replica, random, and several learned ablations
show duplicated subjects. The working hypothesis is that aggressive
role-dependent clocks create incompatible cache-state ages across heads.

This is not yet proven. v90 adds diagnostics for:

```text
per-batch/CFG-group max-min head age
mixed commit group rate
pairwise commit disagreement within each head group
persistent/reactive age gap
persistent/reactive commit-rate gap
forced budget deferral
```

These values must be aligned with the first duplicate-subject frame before the
paper claims a cache-state coherence mechanism.

## 3. v90 hypothesis

v90 replaces hard role clocks with a **weak priority**:

```text
all heads:
  same trust threshold
  same novelty threshold
  same max age
  same forced refresh

PF reactive heads:
  small utility bonus only when competing for the existing 75% write budget
```

The hard decision remains v78's online trust controller. The static class map
only breaks ties between already eligible candidates.

This is deliberately asymmetric:

- online noisy/clean evidence determines whether a candidate is safe;
- the weak PF prior determines which safe candidate is promoted first;
- no class can bypass reliability, novelty, or max-age safety;
- no class receives a longer stale-state lifetime.

The expected benefit is v78-level ID/background retention with PF-binary's
motion behavior, without the duplicated-subject artifact caused by different
role clocks.

This is a candidate refinement, not a validated contribution.

## 4. v90 experiment matrix

Every cell uses the same 16 complex single prompts, 120 output frames
(approximately 30 seconds), and one GPU.

### 4.1 Matched-seed paper evidence

| GPU | Cell | Seed | Purpose |
|---:|---|---:|---|
| 0 | `pf_s1` | 1 | PF-v78 matched pair |
| 1 | `v78_s1` | 1 | PF-v78 matched pair |
| 2 | `pf_s2` | 2 | PF-v78 matched pair |
| 3 | `v78_s2` | 2 | PF-v78 matched pair |
| 4 | `pf_s3` | 3 | PF-v78 matched pair |
| 5 | `v78_s3` | 3 | PF-v78 matched pair |

Together with the existing matched seed-0 v86 result, this gives four paired
seeds over the same 16-prompt suite.

### 4.2 Weak-priority and mechanism cells

| GPU | Cell | Changed factor |
|---:|---|---|
| 6 | `pf_priority_b005` | PF binary labels, utility bias `0.05` only |
| 7 | `pf_priority_b010` | PF binary labels, utility bias `0.10` only |
| 8 | `learned_priority_b005` | learned labels, utility bias `0.05` only |
| 9 | `inverse_priority_b005` | inverse-label causal control |
| 10 | `random_priority_b005` | layer-balanced random-label control |
| 11 | `pf_age_only` | PF labels, age clocks `8/4`, no novelty/bias |
| 12 | `pf_novelty_only` | PF labels, novelty scales `1.5/0.5`, equal age |
| 13 | `wave_priority_b005` | only PF Wave heads receive weak priority |
| 14 | `veil_priority_b005` | only PF Veil heads receive weak priority |
| 15 | `pf_priority_late` | PF weak priority only in layers `[15,30)` |

This matrix answers five questions:

1. Is v78's improvement reproducible under paired seeds?
2. Can a weak priority recover motion without hard-clock artifacts?
3. Does learned classification beat inverse/random when used conservatively?
4. Which hard-clock factor causes duplication: age or novelty?
5. Does the useful PF prior come from Wave, Veil, or late layers?

The seed-0 `pf`, `v78`, `pf_binary_balanced`, and `learned_balanced` videos
are reused from v86. They are not regenerated.

## 5. Code added

| File | Purpose |
|---|---|
| `scripts/build_pf_transition_controls.py` | Derive audited PF-binary, Wave-only, Veil-only, Anchor-only, Wave-Anchor, and Veil-Anchor maps |
| `scripts/run_v90_priority_factorization_16gpu.sh` | Launch 6 paired-seed and 10 factorization cells |
| `scripts/postprocess_v90_priority_factorization.sh` | Combine reused v86 baselines with v90 metrics and run 16-way VBench-Long |
| `scripts/summarize_cache_transition_trace.py` | Add cache-state coherence and cross-role age/commit diagnostics |
| `scripts/analyze_v90_metrics.py` | Produce paired-seed deltas, candidate ranking, temporal jump and coherence tables |
| `tests/test_build_pf_transition_controls.py` | Validate exact PF class mappings |
| `tests/test_v90_experiment_contract.py` | Lock the 16 methods, GPU assignment, prompts, and postprocess contract |

All method behavior remains off by default. v90 uses existing transition
controller options and does not change PF inference outside explicit CLI
flags.

## 6. Server commands

### 6.1 Generation

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
bash scripts/run_v90_priority_factorization_16gpu.sh
```

Required existing seed-0 baselines:

```text
runs/v86_role_transition_screen/pf/
runs/v86_role_transition_screen/v78/
runs/v86_role_transition_screen/pf_binary_balanced/
runs/v86_role_transition_screen/learned_balanced/
```

The launcher requires exactly 16 videos in every baseline directory and fails
before consuming GPUs if a path is missing.

New outputs:

```text
runs/v90_priority_factorization_screen/
  run_manifest.env
  labels/
  configs/
  logs/
  traces/
  diagnostics/
  <method>/*.mp4
```

If a cell is incomplete, use a new clean `OUT_ROOT`; do not append duplicate
videos to the failed directory.

### 6.2 Human review

Review method names blind. At minimum score all prompts for:

```text
identity persistence
background persistence
action continuation
camera motion
dynamic degree
duplicate subject
subject disappearance/reappearance
physics violation
flashback
acceleration jump
freeze
```

The most important direct comparisons are:

```text
v78 vs PF at the same seed
pf_priority_b005/b010 vs v78 and pf_binary_balanced
learned vs inverse vs random priority
pf_age_only vs pf_novelty_only
wave-only vs veil-only priority
```

### 6.3 Metrics and VBench-Long

After freezing human review:

```bash
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v90_priority_factorization.sh
```

The postprocessor:

1. validates exactly 16 videos and clean logs for all cells;
2. recomputes comprehensive and temporal-jump metrics over v86 baselines plus
   all v90 cells;
3. recomputes trace summaries with coherence statistics;
4. writes `v90_analysis.json/md` with paired differences and candidate ranks;
5. launches VBench-Long for the 16 new cells in parallel on GPUs 0-15.

Default VBench dimensions:

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
dynamic_degree
```

`motion_smoothness` is omitted by default because the current server cannot
download the required RAFT weight (HTTP 403). Temporal jump remains the local
smoothness diagnostic. Add the dimension later only after placing the RAFT
model:

```bash
VBENCH_DIMS="subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree" \
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v90_priority_factorization.sh
```

## 7. Predeclared decisions

### 7.1 v78 paper-core gate

Treat v78 as a confirmed PF improvement only if:

1. paired DINO difference averaged over seeds 0-3 is positive;
2. at least three of four seeds are non-negative or the confidence interval is
   clearly positive;
3. min-DINO does not systematically regress;
4. dynamic degree and human motion do not fall;
5. no new severe artifact appears.

If the paired gain is inconsistent, the claim must be "PF-level quality with
lower temporal jump", not "better identity than PF".

### 7.2 weak-priority promotion gate

Promote `pf_priority_*` only if it:

1. is at least within `0.003` DINO of v78;
2. improves dynamic degree or human camera/action motion;
3. has no duplicated subject, disappearance, or physics violation;
4. has lower age spread and role age gap than hard `pf_binary_balanced`;
5. beats learned, inverse, and random weak-priority controls on combined human
   and metric assessment.

If `pf_priority_b005` and `b010` do not improve motion, keep uniform v78.

### 7.3 classification decision

- Learned weak priority beats inverse/random and replicates visually:
  classification can remain as an analysis or secondary method component.
- Learned weak priority ties controls:
  remove the classifier from the method claim.
- PF labels win:
  cite PF labels as a borrowed prior; do not rename them as our classifier.
- Wave or Veil isolation wins:
  use the result as a factorized prior and cite PF's original taxonomy.

## 8. Paper story after v90

The current safe story is:

1. Long AR generation repeatedly writes generated states of unequal quality
   into persistent cache.
2. Existing noisy and clean diffusion passes expose a free online consistency
   signal.
3. Trust-conditioned state promotion controls the lifecycle of PF's existing
   middle cache with no extra forward and no direct archive retrieval.
4. Bounded asynchronous updates reduce temporal discontinuity while retaining
   PF-level identity/background quality.
5. Optional weak motion priority is included only if v90 improves dynamics
   without cache-state incoherence.

The story does not require claiming head classification. A new classifier
could still be innovative in principle, but the current learned classifier is
not supported by the target experiments.

## 9. Return for analysis

```text
runs/v90_priority_factorization_screen/run_manifest.env
runs/v90_priority_factorization_screen/configs/
runs/v90_priority_factorization_screen/labels/
runs/v90_priority_factorization_screen/logs/
runs/v90_priority_factorization_screen/traces/
runs/v90_priority_factorization_screen/diagnostics/
runs/v90_priority_factorization_screen/metrics/
frozen blind-review sheet
exact git commit and commands
```

For duplicated-subject diagnosis, record the first bad frame and compare the
nearest trace events across v78, hard role clocks, and weak priority. Report
age spread, commit disagreement, role age gap, forced deferrals, shock, and
denoise disagreement rather than only aggregate acceptance.
