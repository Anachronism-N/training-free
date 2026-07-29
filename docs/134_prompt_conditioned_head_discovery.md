# v134: Prompt-Conditioned History Head Discovery

## 1. Scope

This round is a discovery experiment, not a cache-policy experiment.

The objective is to determine whether Self-Forcing contains a reproducible
head-wise separation between:

- **Prompt-invariant memory heads**: history use changes little under a
  semantic prompt intervention.
- **Prompt-conditional memory heads**: the same latent and the same historical
  K/V are used differently after a semantic prompt intervention.

The old v98 304/56 map is not used to construct the new labels. That map was
obtained from history-QK polarity and remains only an optional post-hoc
comparison. PF head counts, PF labels, k-means, and a per-layer quota are not
part of the v134 classifier.

The experiment uses native Self-Forcing with its 21-latent-frame sliding
window, 120 latent frames (30 seconds), and no LifeCache, structured memory,
Commit Forcing, scene reset, PF cache, or full-attention override.

## 2. Pre-Registered Hypotheses

These hypotheses must be evaluated before designing the final cache.

### H1: Prompt-history interaction is head selective

At a fixed latent state and with identical historical K/V, a semantic prompt
change produces a larger change in the history contribution of some heads
than a meaning-preserving paraphrase does.

Falsification: semantic intervention is not stronger than the paraphrase
control, the binary partition collapses to one class, or split-half and
bootstrap reproducibility fail.

### H2: Prompt-conditional heads should be episode scoped

If H1 holds, a prompt or scene switch should primarily invalidate, namespace,
or refresh the memory visible to prompt-conditional heads. Prompt-invariant
heads may preserve prompt-independent identity and geometry evidence across
episodes.

This is a causal prediction for the next experiment. v134 does not assume it
is true merely because a classifier can be produced.

### H3: Prompt role and temporal role are related but not identical

Expected direction:

- Prompt-invariant heads may have larger expected history age and old-history
  mass.
- Prompt-conditional heads may be more recent oriented or have higher temporal
  entropy because they must reconcile current text with changing history.

The experiment is two-sided. An opposite but reproducible relation is also a
valid result. No temporal threshold is used to force the prompt labels.

### H4: Prompt conditioning varies over denoising timestep

Expected factor-specific behavior:

- Scene and camera interventions may be strongest at high-noise steps where
  global layout is formed.
- Action changes may be strongest at intermediate denoising stages.
- Identity, appearance, and style changes may remain active at later steps.

If a static head label is unstable but a head-by-timestep score is
reproducible, the final method should use a continuous timestep-conditioned
gate instead of a static binary map.

### H5: Prompt conditioning varies over AR age

The prompt-history conflict may increase after the sliding window is full and
at later extrapolation positions. A useful head property should remain
rank-stable across early, middle, and late AR regions even if its magnitude
changes.

### H6: The discovered property generalizes

The map must be reproducible across controlled factors and consistent with
temporal statistics measured on natural Qwen-rewritten MovieBench-128
prompts. A result that exists only for one synthetic factor is not sufficient
for a paper claim.

## 3. Counterfactual Definition

For a head `h`, prompt `p`, fixed noisy latent `z`, and fixed historical cache
`H`, compute two read-only attention outputs:

```text
A_full(h, p)   = Attention(Q_h(p), K_H, V_H)
A_recent(h, p) = Attention(Q_h(p), K_recent4, V_recent4)
R_h(p)         = A_full(h, p) - A_recent(h, p)
```

`R_h(p)` isolates the output contribution of history older than the last four
latent frames. It avoids labeling a head as memory-sensitive merely because
its current query changes under text conditioning.

At the same trajectory state, run three branches:

```text
base:     original prompt p
semantic: one-factor semantic intervention p_sem
null:     meaning-preserving paraphrase p_null
```

Only the base branch updates the native K/V cache and sampling trajectory.
The semantic and null branches:

- receive exactly the same current latent and timestep;
- read exactly the same completed-frame historical K/V;
- replace only the current prompt conditioning;
- use separate cross-attention caches;
- do not update self-attention cache tensors or indices;
- discard their diffusion prediction.

The recorded output is a deterministic grouped mean/RMS signature `phi`.
For each sampled state:

```text
D_sem(h)  = relative_distance(phi(R_h(p_sem)),  phi(R_h(p)))
D_null(h) = relative_distance(phi(R_h(p_null)), phi(R_h(p)))
s_h       = log((D_sem(h) + 1e-4) / (D_null(h) + 1e-4))
```

The final continuous head score is the median `s_h` over prompts, AR
positions, and selected timesteps.

Confidence intervals resample the 16 subject families as clusters, rather
than treating the eight factor variants of one subject as independent
samples. Split-half reproducibility also divides subject families while
retaining all eight factors in each half.

The pre-registered zero threshold has a semantic meaning:

```text
score > 0: semantic intervention exceeds paraphrase variation
score <= 0: semantic intervention does not exceed paraphrase variation
```

It is not fitted to PF, not count matched, and not selected to produce a
preferred class size.

## 4. Prompt Suites

### 4.1 Natural temporal profiling

- 128 Qwen-rewritten MovieBench prompts.
- 30-second native SF generation.
- Base branch only.
- Purpose: measure temporal attention, timestep behavior, AR-age behavior,
  and natural-prompt stability.

Server source:

```text
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
```

### 4.2 Controlled prompt intervention

`scripts/build_v134_head_discovery_suite.py` creates 128 jobs:

- 16 coherent scenario families crossed with eight factors;
- all eight jobs in one family use the same base prompt and seed, so their
  base trajectory is matched while only the semantic intervention differs;
- 16 jobs for each of eight factors;
- identity, appearance, action, scene, object, camera, atmosphere, and style;
- one semantic field changes per pair;
- the null branch expresses the same components with a different sentence
  structure;
- prompts remain long and contain identity, motion, scene, camera, lighting,
  objects, and long-video consistency constraints.

Each controlled job generates one 30-second base video and executes semantic
and null shadow forwards only at selected profiling states.

## 5. Sampling Grid

Noisy denoising calls:

```text
AR start frames: 3, 21, 42, 63, 84, 117
Nominal timesteps: 1000, 750, 500, 250
```

Clean-context calls:

```text
AR start frames: 21, 63, 117
Nominal timestep: 0
```

This gives 27 selected states per base video. The first generated block at
frame 0 is excluded because it has no completed historical frame.

## 6. Recorded Head Evidence

For every selected state and all 30 x 12 heads:

- full-history minus recent-history output signature;
- native self-attention output signature;
- current query signature;
- current key signature;
- frame-level temporal QK logits;
- frame-level temporal attention probabilities;
- absolute history frame ids;
- layer, head, prompt factor, branch, AR frame, nominal timestep, actual
  timestep, and clean/noisy mode.

Frame-level QK is estimated at 16 deterministic shared spatial positions per
frame. It is used for temporal characterization, not for constructing the
prompt score.

Derived temporal properties include:

- expected history age;
- recent-four-frame mass;
- history-older-than-12 mass;
- normalized temporal entropy;
- peak attended age;
- positive QK-logit fraction.

## 7. Acceptance Gates

The static binary map is admissible only if all gates pass:

1. Exactly 128 observational and 128 counterfactual profiles exist.
2. Every selected base record has semantic and null records.
3. Global median semantic interaction exceeds paraphrase interaction.
4. Even/odd prompt split head-score Spearman is at least 0.30.
5. At least 70% of heads have bootstrap label confidence at least 0.80.
6. The smaller class contains at least 10% of heads.

The threshold sweep is diagnostic only. It must not replace the zero threshold
without a separately reported validation protocol.

## 8. Outcome-Dependent Method Decisions

### Case A: Static split passes and has a temporal relation

Use a two-family memory:

- Prompt-invariant heads receive long-lived identity/layout anchors and
  retrieval from older history.
- Prompt-conditional heads receive prompt-keyed episode memory, recent
  continuity, and motion-relevant frames.
- At a prompt boundary, refresh or namespace mainly the prompt-conditional
  memory.

This is the strongest version of the intended paper story.

### Case B: Static split passes but temporal relation is weak

Keep prompt-conditioned episode handling as the primary contribution. Use the
same temporal cache structure for both classes and differ only in prompt
boundary admission. Do not invent heterogeneous temporal routes unsupported
by data.

### Case C: Static split fails but timestep/factor structure is reproducible

Replace static labels with a continuous score:

```text
gate(head, timestep, prompt factor or prompt delta)
```

The paper contribution becomes online prompt-conditioned memory control,
rather than a fixed head taxonomy.

### Case D: Prompt interaction is not reproducible

Abandon prompt sensitivity as the classification claim. Retain v98 history
polarity only if it survives an independent causal experiment, or move to a
head-agnostic episode memory. A failed result must not be repackaged as a
successful binary classifier.

## 9. Relation to Prior Work

The experiment is informed by, but does not copy, these lines of work:

- Pyramid Forcing: temporal head heterogeneity and heterogeneous cache routes.
- AMA/v80 diagnostics: prompt perturbation and QK response observations.
- Forcing-KV and Head Forcing: static/dynamic or local/anchor/memory head
  distinctions and update policies.
- Echo Forcing: scene snapshot, semantic retrieval, and motion-aware history.
- LongLive-RAG and earlier repository experiments: history retrieval.

The distinguishing point being tested is the **counterfactual interaction
between prompt semantics and history contribution**, normalized by paraphrase
variation and measured on the same trajectory. PF labels and their class
counts are not supervision. Any overlap with existing maps is reported only
after the v134 map is frozen.

## 10. Paper Claims Allowed After v134

Before results, no positive taxonomy claim is allowed.

If all gates and later causal routing tests pass, defensible claims are:

1. Long-video self-attention heads differ in how prompt semantics modulate
   their use of history.
2. This property is distinct from purely temporal QK polarity.
3. Prompt-conditioned memory should be managed at the head and episode level,
   especially at prompt or scene changes.
4. Temporal and denoising-stage properties provide a second, orthogonal axis
   for deciding what history to retain and when to expose it.

The final paper must cite all borrowed mechanisms and include failed or
negative controls needed to support these distinctions.
