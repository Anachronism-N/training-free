# TransitionCache Paper Story

> Date: 2026-07-24
> Status: writing blueprint; quantitative claims remain conditional on v90.
> Working title: **Trust the Transition: Reliability-Gated Cache State
> Promotion for Training-Free Long Video Extrapolation**

## 1. One-sentence method

TransitionCache treats long-video attention memory as a state-promotion
problem: it uses free noisy/clean diffusion evidence to decide whether and when
a generated K/V state may replace the existing persistent cache state.

## 2. The paper's central question

Most long-video cache methods ask:

```text
Which historical tokens should the current query read?
```

Our paper asks a different question:

```text
Which generated states were reliable enough to become history at all?
```

Autoregressive video generation writes its own predictions back into attention
memory. Once an unreliable state is admitted, later frames repeatedly consume
it. Better retrieval cannot fully correct a contaminated history.

This gives the paper a focused problem definition:

> Long-horizon degradation is partly a cache-state lifecycle problem, not only
> a cache-capacity or retrieval problem.

## 3. Evidence-led motivation

The project history supplies a useful empirical funnel:

1. Native Self-Forcing degrades after several seconds and later collapses.
2. Side-memory fusion was active but too weak to alter the generation path.
3. Extra correction forwards delayed some failures but introduced freezing,
   style shifts, and about 50% more forwards.
4. Direct archive recall retained identity but caused background flashbacks,
   polygonal noise, duplicated subjects, or physics violations.
5. PF's sink/middle/recent read topology remains the strongest base.
6. Controlling PF middle-cache writes with v78 matches or slightly exceeds PF
   while reducing temporal jump, without another forward.

The paper should not narrate every failed prototype. Use them to motivate two
design constraints:

```text
do not add a weak side branch
do not inject old K/V directly
```

## 4. Method story

### 4.1 Existing cache

PF already provides:

```text
sink / anchor + structured middle history + recent local window
```

TransitionCache leaves this read topology unchanged and intercepts only clean
writes into the middle region.

### 4.2 Free transition evidence

For head \(h\) and AR block \(t\):

```text
shock(t,h)   = distance(clean candidate, last promoted clean state)
denoise(t,h) = distance(clean candidate, same-block noisy state)
trust(t,h)   = exp(-w_s * shock - w_d * denoise)
```

The noisy and clean states already exist in the generation trajectory. The
controller adds descriptor pooling and decisions, not a model forward.

### 4.3 State promotion

A candidate must satisfy:

```text
minimum update interval
trust threshold
novelty threshold
asynchronous phase
per-layer promotion budget
```

A max-age refresh prevents permanent staleness. Eligible heads are ranked by
trust, age, and novelty. Only the selected heads update the PF middle cache;
sink and recent states follow PF's original lifecycle.

### 4.4 Optional weak motion priority

v90 tests an optional refinement:

```text
online trust decides eligibility
PF temporal class gives a small budget tie-break only
all classes retain identical novelty thresholds and max age
```

This component belongs in the final method only if it improves dynamic degree
or human motion without duplicated subjects. PF's class map must be cited as a
borrowed prior, not claimed as our classifier.

## 5. Contribution claims

### 5.1 Claims currently supportable

1. **Cache state promotion formulation.** We identify write-side state
   admission as a distinct control point for training-free long-video
   extrapolation.
2. **Trajectory-derived trust.** We use same-block noisy/clean disagreement
   and inter-promotion shock as a zero-forward online signal.
3. **Bounded asynchronous lifecycle.** Reliability, novelty, age, phase, and
   budget jointly control middle-cache replacement while preserving PF's read
   topology.
4. **Artifact-aware analysis.** Controlled direct-recall and role-label
   experiments show that identity metrics can hide background hallucination,
   duplicated subjects, motion loss, and physics failure.

### 5.2 Conditional claim

5. **Weak motion priority.** Include only if v90 retains v78 consistency and
   measurably improves motion without increasing cache-state incoherence.

### 5.3 Claims not supportable

- first use of head specialization;
- superiority of the current counterfactual persistent/reactive classifier;
- invention of PF's Anchor/Wave/Veil categories;
- a robust `+0.021` multi-seed gain over PF;
- general scene or identity memory;
- better motion based only on lower temporal jump.

The counterfactual classifier can be reported as analysis: it is reproducible,
but neither hard clocks nor direct recall beat PF-binary and uniform v78
controls.

## 6. Difference from related work

| Work | Primary control | Our distinction |
|---|---|---|
| Pyramid-Forcing | per-head historical read composition | we control whether a newly generated state enters its middle history |
| Forcing-KV | static/dynamic KV handling and compression | we target generation-state reliability and long-horizon error accumulation |
| Head Forcing | local/anchor/memory heads and episodic memory updates | we use diffusion-trajectory trust and no episodic read path |
| Echo-Forcing | scene snapshot preserve/recall/forget | we do not retrieve scene snapshots |
| IAMFlow | explicit identity-aware entity/state memory | we use no entity detector, VLM, or identity bank |

The correct relationship to PF is:

```text
PF is the strong read-policy base.
TransitionCache is an orthogonal write-lifecycle controller.
```

Do not obscure PF's contribution. The paper becomes more credible when the
base and our intervention are separated explicitly.

## 7. Recommended paper structure

### 7.1 Introduction

1. Long AR video generation accumulates self-generated errors.
2. Existing work improves historical access, compression, or recall.
3. A missing decision is whether a generated state deserves persistence.
4. Existing diffusion passes provide a free reliability observation.
5. Introduce TransitionCache and summarize quality, temporal, and overhead
   results.

### 7.2 Related work

Organize by mechanism rather than listing papers:

```text
training-free long-video extrapolation
KV-cache read/compression policies
episodic scene/entity memory
attention-head specialization
```

### 7.3 Method

1. PF/SF autoregressive cache preliminaries.
2. Noisy/clean transition descriptors.
3. Trust and novelty.
4. Budgeted asynchronous promotion.
5. Forced refresh and fail-closed behavior.
6. Optional weak priority, if validated.

### 7.4 Experiments

Primary task:

```text
16 complex single prompts
30-second generation
matched seeds 0-3
SF, PF, Echo, v78, and promoted v90 variant
```

Report:

```text
VBench-Long subject/background/aesthetic/imaging/dynamic
DINO average and minimum
paired seed/prompt differences
temporal jump
human blind review
compute and memory overhead
```

Ablations:

```text
audit vs gate vs full lifecycle
trust/novelty/age/budget
acceptance fraction
hard clocks vs weak priority
learned/inverse/random/PF-label controls
Wave vs Veil and depth
```

### 7.5 Limitations

- v78 currently depends on PF as the read-policy base;
- the trust descriptor is head-level and may miss localized spatial failures;
- the matched gain may be small;
- temporal jump is diagnostic, not a full motion-quality metric;
- direct recall and learned role clocks remain negative results.

## 8. Figures and tables

### Figure 1: problem and insight

Show:

```text
unreliable generated block -> unconditional cache write -> persistent error
unreliable generated block -> TransitionCache reject -> trusted history
```

### Figure 2: method

One unframed pipeline:

```text
noisy pass ----\
                -> trust/novelty -> lifecycle controller -> PF middle write
clean pass ----/                         |
last promoted state --------------------/
```

Sink and recent paths should be visibly unchanged.

### Figure 3: lifecycle trace

Plot over AR blocks:

```text
trust
shock
denoise disagreement
accepted heads
age spread
visible failure time
```

### Main result table

Use paired seeds and mark extra forwards:

| Method | Subject | Background | Dynamic | DINO | min DINO | Jump | Extra forward |
|---|---:|---:|---:|---:|---:|---:|---:|

### Mechanism table

| Variant | Acceptance | Age spread | DINO | Dynamic | Duplicate rate |
|---|---:|---:|---:|---:|---:|

## 9. Abstract template

> Autoregressive video diffusion extends generation by repeatedly writing its
> own predictions into attention memory. Existing training-free methods mainly
> control how historical states are read, while unreliable generated states
> can still be promoted into persistent cache and amplify long-horizon errors.
> We introduce TransitionCache, a training-free state-promotion controller for
> long video extrapolation. TransitionCache estimates transition reliability
> from the disagreement between existing noisy and clean diffusion states and
> the change from the last promoted state, requiring no additional model
> forward. It combines reliability, novelty, age, and a bounded asynchronous
> budget to update the middle history while preserving the base model's anchor
> and recent-cache paths. On complex 30-second single-prompt generation,
> TransitionCache [insert matched-seed consistency result] and [insert temporal
> or dynamic result] relative to strong Self-Forcing and Pyramid-Forcing
> baselines. Controlled label and direct-recall studies further show that
> identity gains alone can hide duplicated subjects, background hallucination,
> and motion loss. These results position cache state promotion as a distinct
> and efficient control point for training-free long-horizon video generation.

Do not freeze numerical wording until v90 matched-seed and VBench results are
available.

## 10. Story branches after v90

### A. v78 wins matched seeds

Main story:

```text
trustworthy cache state promotion improves consistency and temporal stability
```

Use weak priority only if it adds motion.

### B. v78 ties PF but lowers jump

Narrow story:

```text
write lifecycle reduces discontinuity at PF-level quality and zero forwards
```

Do not claim better identity.

### C. weak priority improves motion

Expanded story:

```text
online trust handles safety; weak temporal prior resolves safe write conflicts
```

Classification remains borrowed from PF unless a new validated classifier
beats causal controls.

### D. matched results are inconsistent

Do not force a top-conference claim. Retain the work as a strong negative-study
and systems analysis, then redesign the trust signal using localized/token
evidence rather than adding more fixed head labels.

## 11. Reviewer questions to answer

1. Is the gain from PF itself? Report PF, PF audit, and TransitionCache with the
   read topology held fixed.
2. Is lower jump caused by static video? Report dynamic degree and blind motion
   review.
3. Does the controller actually intervene? Report acceptance, rejection
   reasons, and age distribution.
4. Why use both shock and denoise disagreement? Include isolated ablations.
5. Are head labels necessary? Uniform v78 is the main method; weak priority is
   conditional.
6. Does it generalize beyond one seed or prompt? Use the v90 paired four-seed,
   16-prompt protocol.
7. Is overhead truly zero? Say zero **additional model forward** and separately
   report descriptor/controller latency and memory.

The paper should be built around the write-lifecycle insight and matched
evidence, not around accumulating every prototype into one method.
