# LifeCache-v1 experiment plan

## 1. Goal

Validate whether token-level head-aware cache recall improves training-free long-horizon AR video generation over native sliding-window cache and existing cache policies.

The main hypothesis:

```text
A fixed sliding window loses useful long-range information.
A small token-level historical bank with query-based recall can improve long-horizon consistency while keeping cache budget bounded.
Head-aware recall and motion-specific cache reduce interference between static scene memory and dynamic motion generation.
```

---

## 2. Base models

Primary target:

```text
Self-Forcing
```

Secondary target:

```text
Causal-Forcing
```

Implementation should begin with Self-Forcing because it is the cleanest patch target for cache instrumentation and active-cache composition.

---

## 3. Baselines

### 3.1 Required baselines

```text
B0: Self-Forcing vanilla
B1: Self-Forcing + native larger/recent-only cache if memory allows
B2: RollingForcing-style fixed first anchor + recent
B3: Pyramid-Forcing-style head-aware cache if available
B4: Forcing-KV-style dynamic/static cache if available
B5: LifeCache-v1-Minimal
```

### 3.2 Optional baselines

```text
Causal-Forcing vanilla
Causal-Forcing + LifeCache-v1
DeepForcing if its inference code is runnable in our environment
Echo-Forcing / LongLive-RAG if compatible with the same base model and prompt suite
```

---

## 4. Prompt suites

The prompt suite should be designed to expose cache failure modes.

### 4.1 Scene revisit

Tests whether the method can recover old scene layout after the scene temporarily changes.

Example:

```text
A woman in a yellow coat enters a small kitchen with blue cabinets and a red cup on a wooden table. She walks outside into a garden. Later, she returns to the same kitchen and stands beside the same wooden table.
```

Expected improvements:

- background consistency;
- scene revisit correctness;
- object/layout persistence.

### 4.2 Subject recurrence

Tests whether a subject remains identifiable over long rollouts.

Example:

```text
A white dog with a red collar runs through a park, disappears behind trees, and later reappears near a fountain, still wearing the same red collar.
```

Expected improvements:

- subject consistency;
- color/accessory persistence;
- reduced identity drift.

### 4.3 Long motion continuation

Tests whether motion remains dynamic instead of freezing.

Example:

```text
A dancer performs continuous spinning movements on a stage under moving lights for a long duration.
```

Expected improvements:

- dynamic degree;
- motion smoothness;
- less repetition and freezing.

### 4.4 Camera motion

Tests whether motion cache helps global dynamics.

Example:

```text
A drone camera flies forward through a narrow canyon, passing rocks and trees while maintaining smooth forward motion.
```

Expected improvements:

- camera motion continuity;
- reduced temporal jitter;
- less motion slowdown.

### 4.5 Hard scene switch

Tests whether recall is harmful when old memory should not be used.

Example:

```text
A robot walks in a clean white laboratory. The scene suddenly cuts to a crowded night market with neon lights.
```

Expected behavior:

- avoid over-recalling old laboratory layout;
- maintain prompt adherence after scene switch.

### 4.6 Similar distractor scenes

Tests whether recall retrieves the correct old scene rather than a similar but wrong memory.

Example:

```text
A person first enters a red kitchen, then a blue kitchen, then returns to the red kitchen.
```

Expected improvements:

- correct memory retrieval;
- fewer background swaps.

---

## 5. Metrics

### 5.1 General video quality

```text
VBench / VBench-Long overall score
visual quality
imaging quality
temporal flickering
```

### 5.2 Long-horizon consistency

```text
subject consistency
background consistency
scene revisit consistency
object/color consistency
```

### 5.3 Motion metrics

```text
dynamic degree
motion smoothness
frame difference stability
motion repetition / freeze rate
```

### 5.4 Prompt alignment

```text
CLIP text-video similarity
segment-level prompt alignment
manual or VLM-assisted evaluation for scene revisit prompts, optional
```

### 5.5 Cache efficiency

```text
number of active tokens
number of bank tokens
GPU memory
CPU memory
generation time per frame / per chunk
```

---

## 6. Ablation plan

### 6.1 Compression ablation

Compare:

```text
C0: no compressed bank
C1: Attention Participation Top-k
C2: Head-group-aware Compression
C3: Video-understanding-inspired Key Token Compression
```

Expected:

```text
C1 should outperform no bank on long-range consistency.
C2 should improve consistency without harming motion.
C3 should help scene revisit and diversity-sensitive prompts.
```

### 6.2 Recall ablation

Compare:

```text
R0: no recall
R1: chunk-level recall
R2: token-level Q-K recall
R3: token-level Q-K + head-aware recall
R4: token-level Q-K + head-aware recall + prompt/scene summary
```

Expected:

```text
R2 should outperform chunk-level recall.
R3 should reduce interference between static history and motion.
R4 may improve scene revisit but should be checked for prompt-leakage or overfitting to hand-designed prompts.
```

### 6.3 Anchor ablation

Compare:

```text
A0: no anchor
A1: fixed first anchor only
A2: dynamic anchor only
A3: fixed first anchor + dynamic anchor
```

Expected:

```text
A1 should improve stability over vanilla.
A2 should adapt better when first chunk is not representative.
A3 should be strongest for subject/background consistency.
```

### 6.4 Motion ablation

Compare:

```text
M0: no motion cache
M1: latent-delta motion tokens
M2: dynamic-K motion tokens
M3: latent-delta + dynamic-K + boundary score
```

Expected:

```text
M1 helps boundary continuity.
M2 better matches internal dynamic heads.
M3 should best preserve motion while avoiding jitter/flicker.
```

### 6.5 Head-role ablation

Compare:

```text
H0: all heads share the same active cache
H1: Pyramid/Forcing-KV prior labels
H2: profiled head roles from attention statistics
H3: profiled head roles + prompt-conditioned recall heads
```

Expected:

```text
H1 should already reduce interference.
H2 should improve over raw priors.
H3 may improve scene revisit prompts.
```

### 6.6 Historical usage ablation

Compare:

```text
U0: naive concatenation
U1: region budget only
U2: region budget + region bias
```

Expected:

```text
U1 controls memory and compute cost.
U2 may improve active-cache utility, but beta should be small to avoid destabilizing attention.
```

---

## 7. Head profiling experiments

### 7.1 Attention mass profiling

For each head h, compute:

```text
M_anchor^h = attention mass to anchor tokens
M_recent^h = attention mass to recent tokens
M_recall^h = attention mass to recalled tokens
L_temp^h = temporal locality
```

Use these scores to classify heads into:

```text
layout/anchor
motion/wave
recall/semantic
generic
```

### 7.2 Functional ablation profiling

For each candidate head group, run:

```text
remove anchor tokens
remove motion tokens
remove recall tokens
remove recent tokens
```

Measure:

```text
background consistency change
subject consistency change
dynamic degree change
prompt alignment change
```

### 7.3 Prompt-conditioned profiling

Use paired prompts:

```text
A: The woman returns to the same kitchen.
B: The woman enters a completely new room.
```

A head is prompt-conditioned recall-like if it attends more to old scene memory in A than in B.

---

## 8. Implementation phases

### Phase 0: Cache instrumentation

Add hooks to record:

```text
layer_id
head_id
current K/V shape
recent window span
attention mass to historical regions
clean-refresh K/V boundaries
```

Deliverable:

```text
cache_trace.jsonl or cache_trace.pt
```

### Phase 1: Compression-only bank

Implement:

```text
RecentCache eviction
AP Top-k compression
CompressedBank TokenSet storage
```

No recall yet. Verify memory and token counts.

### Phase 2: Token-level recall

Implement:

```text
Q-summary extraction
TokenSet-level retrieval
token-level Q-K top-k recall
active K/V concat
```

Start with all-head shared recall.

### Phase 3: Anchor and motion cache

Implement:

```text
fixed first anchor
dynamic anchor score
latent-delta motion score
dynamic-K motion score
MotionCache
```

### Phase 4: Head-aware active cache

Implement:

```text
Pyramid/Forcing-KV head label loader
head-specific region budget
head-specific active cache composition
optional region bias
```

### Phase 5: Evaluation and ablation

Run prompt suites and produce tables for:

```text
main comparison
compression ablation
recall ablation
anchor ablation
motion ablation
head-role ablation
memory/time cost
```

---

## 9. Main comparison table template

| Method | Long length | Subject Consistency | Background Consistency | Dynamic Degree | Prompt Alignment | GPU Mem | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self-Forcing | 60s | | | | | | |
| RollingForcing | 60s | | | | | | |
| Pyramid/Forcing-KV | 60s | | | | | | |
| LifeCache-v1 | 60s | | | | | | |

---

## 10. Recommended first run

Start with the smallest meaningful experiment:

```text
Base: Self-Forcing
Length: 30s or 60s
Prompt types: scene revisit + long motion continuation
Compression: AP Top-k
Recall: token-level Q-K recall
Anchor: fixed first + dynamic anchor
Motion: latent delta only
Head: all-head shared, then Pyramid/Forcing-KV prior labels
```

The first success criterion is not SOTA performance. It is:

```text
LifeCache can run without memory explosion;
active recalled tokens are actually attended to;
scene revisit or subject consistency improves over vanilla;
motion does not collapse compared to vanilla.
```

---

## 11. Risk checklist

| Risk | Symptom | Mitigation |
|---|---|---|
| Recall noise | wrong old scene leaks into current scene | reduce recall budget, add prompt/Q gating |
| Motion freeze | video becomes stable but static | isolate motion heads; reduce static recall for motion heads |
| Memory explosion | GPU OOM | CPU bank, smaller recall top-k, region budget |
| RoPE mismatch | repetition or temporal artifacts after far recall | limit far recall in v1; add RoPE-safe mode later |
| Head labels wrong | head-aware cache underperforms all-head | run profiling and fallback to generic |
| Compression loses useful tokens | recall ineffective | compare AP top-k, head-aware, and key-token compression |
