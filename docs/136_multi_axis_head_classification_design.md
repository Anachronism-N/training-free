# v136: Multi-Axis Functional Head Classification

## 1. Scope

v136 is an offline analysis of the frozen v134 head profiles. It adds no
generation, no cache policy, and no GPU inference. Its purpose is to determine
which head properties actually exist before a new memory design is written.

The analysis does not use:

- PF labels or PF class counts to construct a class;
- the superseded v98 304/56 absolute-sign map;
- generated-video metrics to select a threshold;
- k-means or a forced per-layer quota;
- identity, motion, or layout names unsupported by a causal intervention.

Input:

```text
runs/v134_head_discovery/profiles/observational/*.pt
runs/v134_head_discovery/profiles/counterfactual/*.pt
```

Implementation:

```text
scripts/analyze_v136_multi_axis_head_discovery.py
scripts/run_v136_multi_axis_analysis.sh
scripts/package_v136_multi_axis_results.py
```

## 2. Why a single prompt score is insufficient

v134 has one primary classification hypothesis:

```text
Does a semantic prompt intervention change a head's older-history
contribution more than a meaning-preserving paraphrase?
```

That is the right first test for prompt-conditioned memory, but it cannot by
itself answer:

1. whether the head needs old history at all;
2. whether prompt changes which historical ages are selected;
3. whether the head prefers middle history or recent history;
4. whether its behavior is stable across denoising timestep and AR age;
5. whether it selects the correct identity, scene, or motion history.

v136 therefore treats head behavior as several independent continuous axes.
A hard class is emitted only as an exploratory view of a natural zero
boundary. The continuous artifacts are the primary result.

## 3. Historical AMA evidence boundary

The earlier AMA/QACP experiments provide useful hypotheses but not reusable
semantic labels.

### 3.1 Invalid or insufficient historical classifiers

- The early absolute `|QK|` proxy assigned all 360 SF heads to identity. It
  was invalid for the FlashAttention path and caused background locking.
- The old QACP four-role table used per-layer median splits. It guaranteed
  samples in each quadrant and did not prove natural clusters.
- Permuting archive order and then taking `max(similarity)` cannot measure
  content specificity because the candidate set is unchanged.
- PF Wave/Anchor/Veil labels describe temporal QK topology, not
  identity/motion/background semantics.
- The old v98 304/56 map came from a superseded absolute-sign statistic.
  It may be used only as a post-hoc historical comparison.

### 3.2 Signals worth retaining

Earlier profiling found nontrivial variation in:

- prompt response;
- history confidence;
- retrieval top-1/top-2 margin;
- layer and denoising-stage causal response;
- middle-versus-recent history preference;
- temporal sign switching and periodicity.

These signals were not redundant, so v136 measures them separately rather
than averaging them into one score.

## 4. Axes computed from existing v134 profiles

### 4.1 Axis P: Counterfactual Prompt-History Interaction

For prompt `p`:

```text
R_h(p) = A_full_history(h, p) - A_recent4(h, p)

d_sem  = relative_distance(R_h(p_sem),  R_h(p_base))
d_null = relative_distance(R_h(p_null), R_h(p_base))

P_h = median log((d_sem + eps) / (d_null + eps))
```

Natural zero interpretation:

```text
P_h > 0  -> semantic intervention exceeds paraphrase variation
P_h <= 0 -> no evidence for prompt-conditioned history interaction
```

This remains the primary prompt axis.

### 4.2 Axis A: Prompt-Induced History-Age Redistribution

Let `pi_h(p)` be the frame-level attention distribution over historical
frames. v136 computes:

```text
JS_sem  = JS(pi_h(p_sem),  pi_h(p_base))
JS_null = JS(pi_h(p_null), pi_h(p_base))

A_h = median log((JS_sem + eps) / (JS_null + eps))
```

A normalized one-dimensional Wasserstein score over frame age is computed as
an independent corroborating measure.

This axis answers a more specific question than P:

```text
Does the prompt change which historical ages a head reads?
```

P can be positive while A is negative when prompt changes the value/output
representation without changing temporal routing. A can be positive while P
is weak when the age shift has a small output magnitude.

### 4.3 Axis R: Distant-History Reach

The recorder stores a grouped RMS signature of the old-history residual and
the native attention output:

```text
R_h = RMS(A_full_history - A_recent4)
      / RMS(A_native)
```

This is a continuous effect-size diagnostic. It has no justified universal
hard threshold. GMM and Otsu thresholds are reported only as distribution
diagnostics.

### 4.4 Axis T: Middle-versus-Recent Preference

For native SF frame-level temporal logits:

```text
T_h = (
    mean(logit_older_than_recent4)
    - mean(logit_recent4)
) / centered_RMS(all_history_logits)
```

The zero boundary is invariant to a common additive QK-logit shift:

```text
T_h >= 0 -> middle-history supportive
T_h <  0 -> recent preferred
```

This is not the old v98 map. v134 uses native SF's contiguous 21-frame
sliding window and no immutable sink. The v136 name and artifact remain
`middle_recent_margin` to prevent accidental equivalence claims.

### 4.5 Axis L: Long-Range Attention Excess

For a history containing `N` frames and a recent window containing four:

```text
L_h = observed_old_attention_mass - (N - 4) / N
```

This compares old-history attention with a uniform frame baseline. It is
combined with T only in an exploratory joint role:

```text
long_range = (T_h >= 0) and (L_h >= 0)
```

T and L are preserved separately because a high logit margin and a high
softmax mass are not identical.

### 4.6 Temporal topology diagnostics

For each native profile v136 also computes:

- positive QK-logit fraction;
- adjacent sign-switch rate;
- dominant FFT period;
- spectral peak-energy ratio;
- logit/age correlation;
- expected age, peak age, recent mass, old mass, and temporal entropy.

These properties characterize relation to PF-style temporal topology. They
are not used to assign semantic roles.

## 5. Factor, timestep, and AR specialization

Every prompt axis is reported by:

- eight controlled factors;
- clean/noisy mode and nominal denoising timestep;
- early, middle, and late AR positions.

For each head the analyzer writes:

- dominant factor/timestep/AR bin by absolute response;
- response range and standard deviation;
- positive-bin fraction;
- normalized specialization entropy;
- pairwise context Spearman and zero-threshold label agreement.

Interpretation:

- stable rank and sign support an offline static head map;
- stable timestep-specific behavior supports a timestep-conditioned gate;
- stable factor specialization supports task-specific prompt-switch handling;
- unstable behavior supports continuous online routing, not a permanent class.

## 6. Correct exclusion of the frame-3 state

At AR frame 3, native SF contains only three historical latent frames:

```text
full_history == recent4
R_h(p) == 0
```

Those four noisy-timestep observations per profile are valid negative
controls, but they contain no evidence about older-history interaction.
v136:

1. audits and preserves them;
2. reports their residual response separately;
3. excludes them from all primary P/A/R/T/L aggregation.

This correction requires no new profiling.

## 7. Exploratory roles

The analyzer emits the following views:

```text
prompt_label:
    prompt_conditional | prompt_invariant

age_routing_label:
    age_conditional | age_invariant

history_polarity_label:
    history_supportive | recent_preferred

long_range_label:
    long_range | local_or_mixed

exploratory_joint_role:
    conditional_long
    conditional_local
    invariant_long
    invariant_local
```

`exploratory_joint_role` is not a generation-ready map. It becomes admissible
only after both constituent axes pass their frozen reproducibility gates and
a later causal routing experiment beats random, reversed, and all-head
controls.

## 8. Frozen gates

Prompt axis:

- global semantic residual response exceeds paraphrase response;
- family split-half head-rank Spearman at least 0.30;
- at least 70% of heads have bootstrap sign confidence at least 0.80;
- minority zero-threshold class at least 10%.

Temporal axis:

- MovieBench prompt split-half rank Spearman at least 0.30;
- at least 70% of heads have bootstrap sign confidence at least 0.80;
- minority zero-threshold class at least 10%.

Decision:

```text
P passes, T passes, |corr(P,T)| < 0.85:
    dual prompt/temporal axes are supported

P passes only:
    prompt class plus continuous temporal diagnostics

T passes only:
    reject prompt taxonomy; retain temporal classifier

neither passes:
    no static taxonomy; use continuous scores or abandon head routing
```

GMM-1/2/3 BIC, GMM intersections, and Otsu thresholds are saved but never
silently replace the natural zero boundaries.

## 9. Additional classifiers requiring a new causal profile

v134 cannot establish the following properties because it changes prompt
conditioning while keeping the same historical K/V.

### 9.1 Correct-History Specificity

Required matched shadows:

```text
correct history
wrong identity, matched scene/action
wrong scene, matched identity/action
recent-only
```

Candidate score:

```text
C_h = effect(correct history)
      - max(effect(wrong identity), effect(wrong scene))
```

This is the required basis for retrieval admission. Merely permuting the
candidate list is not a valid control.

### 9.2 Temporal-Order Sensitivity

Required matched shadows:

```text
normal history
time-reversed history
frozen/duplicated history
phase-shifted history
```

Historical K/V must be recomputed with the counterfactual temporal positions.
Reordering already-RoPE'd paired K/V entries is set permutation and is not a
valid temporal-order intervention.

This axis is the strongest candidate for identifying motion-relevant
history without naming PF Wave heads as motion heads.

### 9.3 Final-Prediction Causal Utility

Attention-level scores must be validated after the output projection and
remaining transformer layers:

```text
U_h = distance(
    final_prediction_with_full_history,
    final_prediction_with_head_h_recent_only
)
```

The first implementation should evaluate grouped top/bottom/random/reversed
head sets rather than 360 full-video ablations. This reduces cost and tests
whether score direction has causal value.

## 10. Intended method story if the evidence passes

The strongest defensible formulation is:

1. prompt interaction determines which memory is episode scoped;
2. temporal reach determines how much distant history a head needs during
   single-prompt extrapolation;
3. correct-history specificity controls retrieval admission;
4. temporal-order sensitivity controls motion/phase evidence;
5. timestep specialization controls when each signal is active.

This differs from PF because the primary axes are prompt-history
counterfactual response and causal history use, not sign-rate/FFT classes.
The cache operators may still borrow established implementations, with
explicit citation and ablation.
