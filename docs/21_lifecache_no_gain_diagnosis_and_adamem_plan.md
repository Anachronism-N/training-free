# LifeCache no-gain diagnosis and AdaMem-inspired optimization plan

> Purpose: explain why the current LifeCache experiments show almost no gain, define the diagnostic experiments needed to locate the failure point, and extract useful design ideas from the uploaded AdaMem proposal without over-expanding the current implementation.

---

## 1. Current symptom

Observed behavior:

```text
LifeCache trace-only: no output change, expected.
LifeCache compression-only: no output change, expected.
LifeCache union recall / v2 variants: little or no measurable improvement; some variants may darken, freeze, or hallucinate.
```

Interpretation:

```text
The current failure is not caused by the absence of more memory components.
The failure is more likely caused by invalid or low-utility recalled K/V.
```

The four most likely causes are:

```text
1. Recalled K is RoPE-invalid.
2. Compression uses a poor query source.
3. Captured memory is not clean or not accumulated correctly.
4. Recalled memory receives either too little attention or the wrong heads attend to it.
```

---

## 2. Why no improvement can happen even if recall is enabled

### 2.1 Trace-only and compression-only should not improve output

If `trace_only=True`, output should be identical to native Self-Forcing. If compression is enabled but recall is disabled, output should also stay unchanged.

Therefore, lack of improvement in these modes is not a failure.

What must be checked:

```text
trace-only output == native output
compression-only output == native output
bank_total_tokens increases
compressed_tokens_added > 0
```

If output changes in either trace-only or compression-only, the integration is already intrusive and must be fixed first.

---

### 2.2 Near-only recall may be too similar to native recent cache

If `max_frame_distance <= local_attn_size`, the recalled tokens are close to the native recent window. Then recall may be redundant with recent K/V.

Expected result:

```text
near-only recall is safe but may provide little gain.
```

This is still useful because it verifies whether post-RoPE near recall is stable.

Diagnostic:

```text
If near-only recall is stable but long recall degrades, RoPE mismatch is confirmed.
If near-only recall has no gain, that does not disprove LifeCache; it only means near memory is not the hard case.
```

---

### 2.3 Long post-RoPE recall is likely harmful

If evicted K is captured from `kv_cache["k"]`, it may already contain old absolute RoPE phase. Reusing it far later makes the query-key phase difference exceed the model's training temporal range.

Symptom:

```text
attention receives historical tokens, but visual output darkens, freezes, or hallucinated bright artifacts appear.
```

Required fix:

```text
Store pre-RoPE K in the bank.
Apply a relative-clamp RoPE remap before using recalled K in attention.
Disable far post-RoPE recall by default.
```

---

### 2.4 Compression query is currently too weak if it comes from evicted K

If compression uses:

```python
q_proxy = evicted_k.mean(dim=0, keepdim=True)
```

then Q-K proxy compression is actually self-similarity over evicted K, not a query from the current generation step.

Symptom:

```text
bank grows normally but selected tokens are not useful.
recall retrieves tokens that are technically similar but not useful for the current query.
```

Required fix:

```text
Capture real q_pre_rope and q_post_rope in attention forward.
Use q_pre_rope with evicted_k_pre_rope.
Use q_post_rope only for near post-RoPE fallback.
```

---

### 2.5 Recalled tokens may not be attended to

Even if `recalled_tokens > 0`, the model may ignore them.

Trace must record:

```text
recalled_tokens
active_tokens
region_counts
attention_mass_to_recall, if attention maps or SDPA weights are available
qk_score_mean / qk_score_top / qk_score_gap
source_set_ids
source_frame_distance
```

Diagnosis:

```text
attention_mass_to_recall ≈ 0:
  recalled tokens are ignored; need better retrieval or small positive bias.

attention_mass_to_recall high but quality worse:
  recalled tokens are invalid/noisy; fix RoPE and filtering.

attention_mass_to_recall high and quality unchanged:
  recalled tokens are redundant; retrieval is not targeting hard long-range cases.
```

---

## 3. Minimal diagnostics before adding new modules

Run these in order.

### D0: native equivalence

```text
native Self-Forcing
LifeCache trace-only
LifeCache compression-only
```

Acceptance:

```text
trace-only and compression-only produce no output difference except negligible nondeterminism.
```

---

### D1: compression quality check

Run compression-only with two query sources:

```text
A: evicted_k.mean query
B: actual q_pre_rope query
```

Compare:

```text
selected token indices overlap
selected frame distribution
bank descriptor diversity
qk score distribution
```

Expected:

```text
actual q_pre_rope should select a different and more concentrated set of tokens.
If A and B are nearly identical, compression is not query-sensitive.
```

---

### D2: near-only recall

Config:

```yaml
recall_enabled: true
recall_top_sets: 2
recall_top_tokens: 128
max_frame_distance: 21
anchor_enabled: false
motion_enabled: false
region_bias_beta: 0.0
enable_last_n_layers: 6
```

Goal:

```text
Verify that bounded-distance post-RoPE recall does not catastrophically degrade quality.
```

Interpretation:

```text
Stable but no gain -> near memory is redundant.
Degrades -> even integration/attention composition is wrong.
```

---

### D3: post-RoPE far recall stress test

Run the same config with `max_frame_distance: null` and far post-RoPE recall allowed.

Interpretation:

```text
If this degrades while D2 is stable, RoPE mismatch is the main issue.
```

---

### D4: pre-RoPE remapped recall

After implementing pre-RoPE bank and relative-clamp remap, compare:

```text
near-only post-RoPE recall
far post-RoPE recall
far pre-RoPE + remap recall
```

Expected:

```text
far pre-RoPE + remap should be more stable than far post-RoPE.
If it is still useless, memory selection/forgetting is the next target.
```

---

## 4. AdaMem: what is useful for our idea

The uploaded AdaMem proposal is useful, but it should not be adopted wholesale in the next coding iteration.

### 4.1 Directly useful ideas

#### Idea A: Store raw K and apply RoPE at read time

AdaMem stores unrotated Key/Value and applies relative RoPE during reading. This directly matches the biggest current LifeCache problem.

LifeCache action:

```text
Add rope_mode to TokenSet.
Prefer pre-RoPE K in CompressedBank.
Remap recalled K at read time.
```

#### Idea B: Add soft retention score rho

AdaMem uses a soft retention score `rho` and injects it into attention via `log rho`.

LifeCache action:

```text
Add retention_score/rho to TokenSet.
Use log(rho) as memory bias for recalled tokens.
Use rho for pruning instead of only quality + importance.
```

#### Idea C: Content-addressed recall

AdaMem argues for content-based memory lookup rather than fixed position-based recall.

LifeCache action:

```text
Keep two-stage Q-K recall.
Add a descriptor z for fast TokenSet-level retrieval.
Do not rely on age or fixed anchor slots as the primary recall criterion.
```

#### Idea D: Write/merge instead of always appending

AdaMem merges redundant low-motion memory into existing slots and writes new slots only for novel/high-motion content.

LifeCache action:

```text
Add write_or_merge() to TokenSetBank.
Use descriptor cosine + motion score to decide merge vs new write.
```

#### Idea E: Soft forget instead of hard stale/flush

AdaMem continuously decays rho based on content drift.

LifeCache action:

```text
Add update_retention(current_descriptor) once per clean context step.
Do not implement full stale/invalid state yet.
```

#### Idea F: Head routing as bias first

AdaMem proposes soft head routing. For current engineering, implement this as per-head region bias, not per-head ragged K/V.

LifeCache action:

```text
Use Pyramid/Forcing-KV labels to initialize head roles.
Implement region_bias_by_head [H,K].
```

---

### 4.2 What not to adopt immediately

Do not implement these yet:

```text
fully differentiable learnable gates
LoRA/fine-tuning
VLM/LLM semantic verification
single bank replacing native recent window
per-token Python memory slots
full online head-routing learning
```

Reason:

```text
The current bottleneck is not lack of differentiability.
The current bottleneck is invalid K/V position and weak memory selection.
```

---

## 5. AdaMem-inspired LifeCache-v2.5 design

Once RoPE-safe recall is working, refine LifeCache into a simpler scoring-based memory instead of many hard regions.

### 5.1 TokenSet fields

Extend TokenSet:

```python
rope_mode: str = "pre_rope"  # preferred for long recall
frame_positions: torch.Tensor | None = None
z: torch.Tensor | None = None          # descriptor, e.g. mean pooled raw K
rho: float = 1.0                       # soft retention
age: int = 0
capture_step: int = -1
motion_score_set: float = 0.0
semantic_drift: float = 0.0
```

`region` can remain for debugging:

```text
RECENT / COMPRESSED / ANCHOR / MOTION / RECALL
```

but recall should primarily use scores, not hard region semantics.

---

### 5.2 Descriptor z

For a TokenSet:

```python
z = normalize(k_raw.float().mean(dim=(0, 1)))
```

If k is `[T,H,D]`, this gives `[D]`.

For better stability:

```python
z = normalize(k_raw.float().mean(dim=0).mean(dim=0))
```

Keep it cheap. Do not add CLIP/VLM yet.

---

### 5.3 Write or merge

For a new compressed TokenSet `s_new`, find the most similar existing memory set:

```python
j = argmax cosine(s_new.z, s_j.z)
sim = cosine(s_new.z, s_j.z)
```

Merge condition:

```text
sim >= theta_merge and motion_score_set <= motion_low_threshold
```

Merge update:

```python
alpha = sigmoid(w_m * (motion_low_threshold - motion_score_set))
old.k = (old.rho * old.k + alpha * new.k_aligned) / (old.rho + alpha)
old.v = (old.rho * old.v + alpha * new.v_aligned) / (old.rho + alpha)
old.z = normalize((old.rho * old.z + alpha * new.z) / (old.rho + alpha))
old.rho = min(1.0, old.rho + alpha)
old.capture_step = current_step
```

If token counts differ, do not merge K/V tensors directly in v2.5. Instead merge only descriptor/rho or keep the higher-importance token subset. Direct tensor merge should be optional.

Write condition:

```text
if not merged:
  add new TokenSet
  prune lowest rho/priority if budget exceeded
```

---

### 5.4 Soft forget

At each clean context update, compute current descriptor:

```python
z_current = normalize(q_pre_rope.float().mean(dim=(0,1)))
```

For each memory set `s`:

```python
drift = 1 - cosine(s.z, z_current)
mu = mu0 + beta * max(0, drift - xi)
s.rho *= exp(-mu)
```

Clamp:

```python
s.rho = clamp(s.rho, min=rho_min, max=1.0)
```

Prune if:

```text
rho < rho_drop_threshold and bank over budget
```

Do not hard delete every low-rho token immediately. Low rho should first reduce recall score.

---

### 5.5 Recall scoring with rho and RoPE safety

Set-level score:

```text
S_set(s) =
    0.45 * cos(Q_bar, K_bar_s)
  + 0.20 * cos(z_current, z_s)
  + 0.15 * log(rho_s + eps)
  + 0.10 * usage_s
  + 0.10 * distance_score_s
  - 1.00 * rope_risk_s
```

Token-level score:

```text
S_token(i) =
    0.70 * QK(i)
  + 0.20 * importance(i)
  + 0.10 * log(rho_s + eps)
```

Attention bias:

```text
memory_bias_i = lambda_rho * log(rho_source(i) + eps)
```

This is the practical training-free version of AdaMem's `+ log rho` attention term.

---

## 6. Immediate code plan

### Step 1: Add diagnostics before changing method

Add trace fields:

```text
q_source
rope_mode
num_pre_rope_sets
num_post_rope_sets
recalled_source_distance_mean
recalled_source_distance_max
qk_score_mean
qk_score_top
memory_bias_mean
attention_mass_to_recall if available
```

Add a trace summarizer table:

```text
step | layer | bank_tokens | recalled_tokens | rope_mode | mean_distance | qk_top | qk_mean | fallback
```

---

### Step 2: Fix clean-only capture and real query

Implement:

```text
runtime.begin_capture/end_capture
_lifecache_evicted_list
payload dict
q_pre_rope/q_post_rope capture
pipeline uses real q instead of evicted_k.mean
```

Run:

```text
compression-clean-only
```

Acceptance:

```text
output unchanged
bank grows
q_source=actual_q_pre_rope
```

---

### Step 3: Add RoPE metadata and near-only safety

Implement:

```text
TokenSet.rope_mode
TokenSet.frame_positions
post-RoPE far recall filter
lifecache_recall_near_only.yaml
```

Run:

```text
near-only recall
```

Acceptance:

```text
recalled_tokens > 0
no catastrophic darkening
```

---

### Step 4: Pre-RoPE bank + relative-clamp remap

Implement:

```text
capture evicted_k_pre_rope
store rope_mode=pre_rope
SelfForcingRopeAdapter.remap_recalled_k()
attention uses remapped K
```

Run:

```text
pre-rope-remap recall
```

Acceptance:

```text
far recalled tokens allowed
no post-RoPE phase collapse
quality >= near-only recall
```

---

### Step 5: Add rho scoring from AdaMem

Implement:

```text
TokenSet.rho
TokenSet.z
TokenSetBank.update_retention()
TokenSetBank.write_or_merge()
recall score + log(rho)
attention memory bias + log(rho)
```

Run:

```text
pre-rope-remap recall
pre-rope-remap + rho
pre-rope-remap + rho + merge/forget
```

Acceptance:

```text
bank diversity improves
wrong old scene recall decreases
scene revisit recall remains possible
```

---

## 7. Experiment matrix

| Experiment | Purpose | Expected outcome |
|---|---|---|
| Native SF | baseline | reference |
| Trace-only | integration safety | same output |
| Compression-clean-only | memory capture safety | same output, bank grows |
| Near-only recall | safe post-RoPE recall | stable, maybe little gain |
| Far post-RoPE recall | RoPE failure confirmation | likely degradation |
| Pre-RoPE + remap recall | core v2 test | stable long recall |
| Pre-RoPE + remap + rho | AdaMem-inspired test | less wrong recall |
| Pre-RoPE + remap + rho + merge | budget efficiency | better quality/VRAM tradeoff |

---

## 8. Recommendation

Do not add more memory pools now.

The next useful optimization is:

```text
RoPE-safe raw-K memory + real-query compression + rho-based recall bias.
```

This combines the current LifeCache engineering path with the strongest AdaMem ideas:

```text
raw K at write time
relative RoPE at read time
content-addressed recall
soft retention score
soft forget / merge later
```

If this version still shows no gain, the conclusion should be that Self-Forcing's local autoregressive cache is not bottlenecked by missing old K/V under this prompt suite, and the project should pivot to either:

```text
1. stronger benchmark prompts with explicit A-B-C-A scene revisit;
2. head-specific region bias;
3. prompt/semantic descriptors for memory recall;
4. Causal-Forcing integration where memory gaps may be larger.
```
