# v144 Factorized Head-Mechanism Profiling — Results

Date: 2026-07-30
Commit: `ccf15db`
Cluster: 4-node × 8×H20 (32 GPUs)

## 1. Overview

v144 is a discovery experiment that factorizes Self-Forcing attention head behavior
along separate semantic axes (identity, scene, action, camera) with a same-prompt
different-seed control. It separates Q, K, V, cache-policy, and spatial-topology
responses and reports both raw and within-layer-residual coordinates.

The goal is to determine whether a defensible static head taxonomy exists for
training-free long autoregressive video memory, or whether head behavior is better
described as a layer/stage program with an online state gate.

**Profile version**: 7 (adds `query_projection`, `history_key_projection`,
`history_value_projection`, `history_value_rms`, `spatial_topology_metrics`,
`causal_policy_metrics`)

## 2. Prompt grid

16 controlled families × 8 variants = 128 videos (each 30 seconds / 120 latent frames):

| Variant | Prompt | Seed | Purpose |
|---|---|---|---|
| `base` | canonical A fields | family seed | reference |
| `seed_control` | exactly the base text | family seed + 10000 | trajectory noise estimate |
| `paraphrase` | identical fields, rewritten template | family seed | surface-form change |
| `identity` | identity only changed | family seed | single-factor |
| `scene` | scene only changed | family seed | single-factor |
| `action` | action only changed | family seed | single-factor |
| `camera` | camera only changed | family seed | single-factor |
| `full_semantic` | all eight fields changed | family seed | maximum semantic shift |

Each manifest row stores token Jaccard and normalized token edit distance for
perturbation-magnitude diagnosis. Seeds are matched so that `base.seed + 10000 ==
seed_control.seed`.

## 3. Captured states

For each 30-second video:

- AR starts: frame 63 and frame 117
- Noisy timesteps: 1000 and 500
- Clean context at both AR starts
- Total: 6 calls, 180 layer records per video

Profile version 7 captures per layer/head:

| Field | Shape | Description |
|---|---|---|
| `query_projection` | (H, D) | Q direction at capture point |
| `history_key_projection` | (R, H, D) | historical K direction |
| `history_value_projection` | (R, H, D) | historical V direction |
| `history_value_rms` | (R, H) | V magnitude |
| `spatial_topology_metrics` | dict | entropy, diagonal mass, displacement, coherence, top1 |
| `causal_policy_metrics` | dict | cache-policy preference |

H = 12 heads per layer, R = 4 recent frames, D = 16 projection dimensions.

## 4. Results summary

### Post-hoc evidence correction

The original total of 85 split-stable features mixes two different kinds of
evidence. Seventy-seven are raw perturbation features and only eight are
`semantic - seed_control` features. The raw Q/K/V/policy/topology responses
are highly correlated with the same-prompt different-seed control and should
be interpreted as generic trajectory susceptibility, not semantic
selectivity.

Likewise, 68 context-stable features consist of 65 raw features and only three
seed-corrected features (`identity`, `camera`, and `paraphrase` query shift)
under the median-context gate. None of those three passes every captured
state at rho >= 0.30. The dominant-factor total agreement of 0.4556 includes
145 heads that are unresolved in both splits; among the 64 heads resolved in
both splits, only 19 agree (0.2969). The compatibility-loss features used for
those labels do not pass split stability, so the factor labels are not a
usable taxonomy.

These corrected counts are emitted explicitly by the updated v144 analyzer.

| Metric | Value |
|---|---|
| Profiles | 128 |
| Head count | 360 (30 layers × 12 heads) |
| State/head observations | 241,920 |
| Split-stable layer-residual features | 85 |
| Context-stable features | 68 |
| Dominant semantic-factor split agreement | 0.4556 |
| Unresolved fraction | 57.5% |
| Functional claim admissible | No |

### 4.1 Split-stable features (85)

A feature is "split-stable" if its discovery/validation Spearman correlation
exceeds a threshold after layer residualization (head score minus median of the
12 heads in its layer).

The 85 stable features break down by variant:

| Variant | Stable features | Includes `*_excess_seed` |
|---|---:|---|
| seed_control | 11 | — |
| paraphrase | 11 | 2 |
| identity | 11 | 2 |
| scene | 11 | 2 |
| action | 11 | 2 |
| camera | 11 | 2 |
| full_semantic | 8 | 1 |

Each variant produces 11 stable features covering all 11 measures (query_shift,
key_shift, value_shift, value_scale_shift, compatibility_loss, policy_shift, and
5 spatial-topology measures). The `*_excess_seed` features are the
difference-of-differences (semantic response minus seed-control response).

### 4.2 Layer residualization effect

Layer effects are large. For `seed_control.key_shift`:

- Raw split Spearman: 0.991
- Layer-residual split Spearman: 0.978
- Discovery layer eta-squared: 0.521
- Validation layer eta-squared: 0.537

This means ~52% of the variance in key-shift is explained by layer alone. The
residualized score removes this nuisance offset; a head is called a static
candidate only if its residualized score is stable.

### 4.3 Context-conditioned stability

The context audit separates frame 63/117 and noisy-1000/noisy-500/clean contexts.
A feature is a `context_stable_candidate` if its median layer-residual split
Spearman ≥ 0.50 across all 6 contexts and its median cross-context Spearman ≥ 0.30.

68 features meet this bar. The strongest examples:

| Variant | Measure | Median split ρ | Min split ρ | Median cross-ctx ρ | State |
|---|---|---:|---:|---:|---|
| action | key_shift | 0.969 | 0.964 | 0.955 | context_stable_candidate |
| action | query_shift | 0.942 | 0.909 | 0.831 | context_stable_candidate |
| action | value_shift | 0.810 | 0.806 | 0.845 | context_stable_candidate |
| scene | key_shift | 0.967 | 0.962 | 0.950 | context_stable_candidate |
| identity | key_shift | 0.958 | 0.950 | 0.909 | context_stable_candidate |

Key-shift is the most context-stable measure, suggesting that historical-K
selection is the most persistent head property across AR states and timesteps.

## 5. Dominant semantic-factor analysis

For each head, the dominant factor is the one with the largest standardized
layer-residual score (after seed-control subtraction), subject to:

- minimum standardized score ≥ 0.50
- minimum margin over runner-up ≥ 0.25

Results across 360 heads:

| Factor | Discovery count |
|---|---:|
| camera | 48 |
| action | 42 |
| identity | 32 |
| scene | 31 |
| unresolved | 207 |

- **57.5% unresolved**: the majority of heads do not show a dominant semantic
  factor above the screening threshold.
- **Split agreement: 0.4556**: the dominant factor label is moderately but not
  highly stable across discovery/validation families.
- **Functional claim admissible: false**: descriptor dominance is observational.
  A head-selective generation intervention must change the output in the
  predicted way before any functional name is assigned.

The factor scaling medians are all 0.0 (layer-residualized), with robust scales
in the range 0.017–0.019. This means the semantic-factor signal is small
relative to trajectory noise.

## 6. Interpretation

### 6.1 What the evidence supports

1. **Layer effects dominate.** More than 50% of variance in most measures is
   explained by layer. Any taxonomy that ignores layer will rediscover
   early/middle/late layers and incorrectly call them head types.

2. **Key selection is the most stable head property.** `key_shift` has the
   highest context stability and cross-context correlation. Historical-K
   selection persists across AR states and timesteps more than Q, V, policy,
   or spatial topology.

3. **Semantic-factor selectivity is weak after seed correction.** The
   `*_excess_seed` features (semantic response minus different-seed response)
   are stable in only 8–11 cases per variant, and the dominant-factor split
   agreement is 0.46. Prompt semantics alone are not a defensible head
   classifier.

4. **A two-level mechanism is more plausible than static classes.** The
   evidence is consistent with: offline head propensity (especially K-selection)
   modulated by an online prompt/episode/timestep gate.

### 6.2 What the evidence does not support

- **No motion-head claim.** Spatial-topology metrics measure attention
  correspondence, not optical flow. Calling any head a "motion head" from
  these values alone would be unjustified.
- **No CFG-head claim.** The profiled Self-Forcing path does not execute
  runtime CFG. The `noisy`/`clean` axis is a denoising/context-refresh axis,
  not a cond/uncond axis.
- **No functional head names.** `functional_claim_admissible` is false. All
  labels are observational until a head-selective intervention passes.

## 7. Relation to prior work

| Work | Idea | v144 difference |
|---|---|---|
| Pyramid Forcing | QK polarity, Anchor/Wave/Veil | PF labels are references only; v144 removes layer effects |
| Forcing-KV | Static/dynamic from attention mass | v144 studies semantic-factor, trajectory, Q/K/V, state axes |
| Head Forcing | Local/anchor/memory heads | v144 requires within-layer and cross-context stability |
| HALO | Motion heads from displacement | v144 uses sampled AR topology, no optical-flow claim |

## 8. Outcome assessment

The results correspond to a mix of **Outcome C** and **Outcome D** from the
v144 runbook:

- **Factor axes are stable but weak**: some factor axes (especially key_shift)
  survive layer residualization and cross-context comparison, but the
  semantic-factor signal after seed correction is small.
- **Context axes are partially stable**: 68 context-stable features exist,
  suggesting a state-dependent gate is a viable model.

The recommended next mechanism is a two-level design:

```
offline head propensity (K-selection) × online prompt/episode/timestep gate
```

This may be more accurate than forcing every head into a fixed binary class.

## 9. Open issues

1. **v143 ab32 persistent probe archive bug** (`head_profile.py:934`): the
   persistent-A archive is incomplete (`expected=[0,18,36,54] captured=[0,18,36]`).
   This blocks v143_hierarchical reanalysis. Requires an upstream code fix.

2. **v143_hierarchical not run**: because v143 ab32 profiles are missing, the
   raw-vs-layer-residual clustering and context-conditioned role audit could
   not be executed. This step is CPU-only and can be run immediately once ab32
   profiles are available.

3. **No head-selective intervention**: the current results are observational.
   A causal experiment (e.g., top-cluster vs bottom-cluster head routing) is
   needed before any functional claim.

## 10. File inventory

```
docs/results/v144_deep_head_profile/factorized/
├── analysis_report.json          # full machine-readable report
├── analysis_summary.md           # one-page summary
├── profile_contract_audit.csv    # 128-row per-profile contract check
├── feature_stability_audit.csv   # 143-row per-feature split stability
├── head_factor_axes.csv          # 360-row per-head factor scores
├── family_head_axes.csv          # 5760-row per-family/head scores
├── context_head_axes.csv         # 28080-row per-context/head scores
├── context_feature_audit.csv     # 859-row per-context/feature stability
└── context_feature_stability.csv # 143-row per-context feature summary
```

Raw profiles (128 × `.pt`), videos (128 × `.mp4`), and shard logs are in
`runs/v144_deep_head_profile/factorized128/` and are not committed.

## 11. Run provenance

- **Script**: `scripts/run_v144_deep_head_profile_32gpu.sh`
- **Design doc**: `docs/144_deep_head_profiling_and_factorized_mechanisms.md`
- **Checkpoint**: `/tmp/self_forcing_dmd.pt` (5.3 GB, Self-Forcing DMD)
- **Seed base**: 144000
- **AR frames**: 63, 117
- **Timesteps**: 1000, 500
- **Commit**: `ccf15db` on `codex/v98-correctness-fixes`
- **Node 1 retry**: original run was OOM-killed during model load; retried
  successfully with the same configuration.
