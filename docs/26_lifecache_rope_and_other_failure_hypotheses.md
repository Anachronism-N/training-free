# LifeCache: Why RoPE is the top suspect, and what else must be checked

> Purpose: clarify why RoPE is currently treated as the highest-priority failure hypothesis, while explicitly avoiding premature conclusion. This document also lists other plausible causes of no improvement and defines controlled checks for each.

---

## 1. Current factual status

LifeCache has passed the first major integration barrier:

```text
Bank grows.
Recall candidates exist.
recalled_tokens > 0.
Recalled tokens enter the active attention window.
```

Recent trace evidence reported:

```text
297 eviction events on layers 27-29
1680 recall candidate events
1680 compose events with recalled_tokens > 0
active cache = 32760 recent tokens + 512 recalled tokens
bank tokens = 32768
```

So the current problem is no longer:

```text
Why is recall not happening?
```

The current problem is:

```text
Recall happens, but why does it not improve generation quality?
```

This distinction is important. It means we should stop focusing only on bank growth and recalled token count. We must now check whether recalled tokens are correct, position-safe, and actually useful.

---

## 2. Why RoPE is the top suspect

RoPE is not proven to be the only cause. It is the strongest current hypothesis because it satisfies four criteria:

```text
1. Clear causal mechanism.
2. Direct code-level risk.
3. Consistency with observed failure modes.
4. Testable with controlled ablations.
```

---

## 3. RoPE failure mechanism

LifeCache recalls historical K/V tokens and inserts them into current self-attention.

In Wan/Self-Forcing style DiT attention, K is not just a content vector. It is usually rotated by temporal/spatial RoPE. The attention score depends on the phase relationship between query and key.

If old K was already rotated at its original old frame position and is later recalled at a much later frame, the effective query-key phase gap may exceed the temporal range seen during training.

In that case, the model is not simply attending to useful old content. It may be attending to a key with an invalid position phase.

Potential symptoms:

```text
recalled_tokens > 0 but no gain
video darkens after recall starts
motion becomes frozen or static
bright/hallucinated artifacts appear
attention seems to use memory but quality gets worse
```

---

## 4. Code evidence for RoPE risk

The current attention integration has two relevant paths:

### 4.1 Native KV cache path

In normal Self-Forcing paths, `kv_cache["k"]` can store post-RoPE keys.

LifeCache added `kv_cache["k_pre_rope"]`, which stores the raw pre-RoPE key `k`. This is necessary because long-range recall should prefer raw K and apply RoPE at read time.

### 4.2 Current capture payload

The capture payload contains both:

```python
evicted_k_pre_rope
evicted_k_post_rope
q_pre_rope
q_post_rope
frame_positions
current_start_frame
```

This is the right direction.

However, the current attention-side recall still has two important risks:

```text
1. retrieval may compare roped_query against pre-RoPE memory K;
2. remap currently pads sparse recalled tokens and applies RoPE with start_frame=0, which is closer to zero remap than true relative-clamp remap.
```

These risks are enough to make RoPE a top-priority debugging target.

---

## 5. Important clarification: RoPE is not fully confirmed yet

The correct statement is:

```text
RoPE is the strongest current hypothesis, not a final conclusion.
```

We should not say:

```text
The problem is definitely RoPE.
```

We should say:

```text
Because recall now happens, the next most important question is whether the recalled K/V enters attention in a valid RoPE coordinate system.
```

The final confirmation requires controlled experiments.

---

## 6. Controlled experiments to confirm or reject the RoPE hypothesis

### Experiment R0: native baseline

Run native Self-Forcing without LifeCache.

Purpose:

```text
Establish baseline quality, dynamics, identity drift, and scene revisit behavior.
```

---

### Experiment R1: near-only post-RoPE recall

Config idea:

```yaml
recall_enabled: true
allow_post_rope_recall: true
max_frame_distance: 21
recall_top_tokens: 128
anchor_enabled: false
motion_enabled: false
region_bias_beta: 0.0
```

Expected interpretation:

```text
Stable but no gain:
  near memory is probably redundant with native recent cache.

Unstable or worse:
  active cache composition or attention path may be wrong even for safe distances.
```

---

### Experiment R2: far post-RoPE recall

Config idea:

```yaml
recall_enabled: true
allow_post_rope_recall: true
max_frame_distance: null
recall_top_tokens: 128 or 256
```

Expected interpretation:

```text
If this degrades compared to R1, far post-RoPE recall is unsafe.
This supports the RoPE mismatch hypothesis.
```

---

### Experiment R3: far pre-RoPE recall + zero remap

Current optimized behavior is approximately:

```text
pre-RoPE bank
recalled K padded to full frame length
causal_rope_apply(..., start_frame=0)
```

Expected interpretation:

```text
If stable but no gain:
  zero remap may avoid catastrophic phase mismatch but may collapse temporal structure.

If worse:
  remap, retrieval, or captured memory quality is still wrong.
```

---

### Experiment R4: far pre-RoPE recall + true relative-clamp remap

Required implementation:

```text
propagate frame_positions through recall result and recall:view TokenSet;
use frame_positions to map historical tokens into legal temporal positions near the current window.
```

Mapping:

```python
rel = (current_start_frame - frame_positions).clamp(0, TR - 1)
t_mapped = (TR - 1) - rel
```

Motion-safe version:

```python
is_recent = rel < split_recent
rel_mapped = torch.where(
    is_recent,
    rel,
    torch.full_like(rel, TR - 1),
)
t_mapped = (TR - 1) - rel_mapped
```

Expected interpretation:

```text
If R4 is more stable than R2 and more useful than R3, RoPE-safe read-time remap is necessary.
If R4 still gives no gain, inspect non-RoPE causes.
```

---

## 7. Required code changes for RoPE validation

### 7.1 Use pre-RoPE query for pre-RoPE bank retrieval

Current risk:

```python
q_for_life = roped_query[0]
```

If bank stores pre-RoPE K, retrieval should use:

```python
q_for_life = q[0] if q.ndim == 4 else q
```

Add trace:

```text
retrieval_query_space = pre_rope / post_rope
bank_rope_mode = pre_rope / post_rope / mixed
recall_rope_mode = pre_rope / post_rope / mixed
```

### 7.2 Propagate frame_positions through recall

Current issue:

```text
TokenSet has frame_positions,
but RecallResult and recall:view do not preserve them.
```

Required:

```python
RecallResult.frame_positions: torch.Tensor | None
RecallResult.rope_mode: str | None
```

In `recall_tokens()`:

```python
all_frame_positions = torch.cat([...])
selected_frame_positions = all_frame_positions.index_select(0, positions)
```

In `ActiveCacheComposer`:

```python
TokenSet(
    set_id="recall:view",
    ...,
    rope_mode=recall_result.rope_mode,
    frame_positions=recall_result.frame_positions,
)
```

### 7.3 Implement real remap policy switch

Config:

```yaml
rope_remap_policy: zero
rope_remap_policy: relative_clamp
```

Trace:

```text
rope_remap_policy
recalled_frame_pos_min / max / mean
mapped_t_pos_min / max / mean
num_sparse_recalled_tokens
padded_len
```

---

## 8. Other plausible causes besides RoPE

RoPE is the highest-priority suspect, but not the only possibility. The following hypotheses must be checked if RoPE fixes do not improve results.

---

## 9. Hypothesis A: retrieval query / memory key space mismatch

This overlaps with RoPE but is slightly different.

Failure mode:

```text
Compression stores pre-RoPE memory using q_pre_rope,
but recall ranks memory using roped_query.
```

Result:

```text
Recall occurs, but selected tokens are not semantically or visually useful.
```

Check:

```text
Compare top recalled token indices under q_pre_rope vs roped_query.
Measure overlap ratio.
Trace qk_score_mean/top for both.
```

Action:

```text
Use q_pre_rope for pre-RoPE bank retrieval.
Use q_post_rope only for post-RoPE bank retrieval.
```

---

## 10. Hypothesis B: memory captured from noisy denoising states

Recent implementation notes indicate that clean context refresh may not trigger eviction; eviction may occur during denoising. Therefore memory may be captured from noisy intermediate states.

Failure mode:

```text
Bank grows and recall works,
but memory K/V represents noisy or unstable denoising states rather than clean visual memory.
```

Check:

Add trace fields:

```text
capture_reason
capture_timestep
capture_loop_index
capture_noise_level
current_start_frame
```

Run ablation:

```text
capture all denoising evictions
capture only final denoising step
capture only low-noise timesteps
capture only every Nth block
```

Expected interpretation:

```text
If low-noise capture improves quality, memory quality was the bottleneck.
If all policies are the same, capture timestep is less likely the main issue.
```

Action:

```text
Implement capture_timestep_policy.
Prefer lower-noise or final-step evictions.
```

---

## 11. Hypothesis C: recalled tokens are redundant with recent cache

Failure mode:

```text
recalled_tokens > 0,
but they come from frames close to native recent window or similar content.
```

Check:

Trace:

```text
recalled_source_distance_mean
recalled_source_distance_max
recalled_frame_pos_min/max
recent_window_frame_min/max
```

Run A-B-A prompts instead of ordinary continuation prompts.

Expected interpretation:

```text
If recall is near-only and no gain, that is expected.
If far recall also no gain on A-B-A prompts, selection or model bottleneck may be the issue.
```

Action:

```text
Use prompts requiring scene revisit.
Increase minimum recall distance only after RoPE-safe remap works.
```

---

## 12. Hypothesis D: selected memory tokens are low-utility

Failure mode:

```text
Compression and recall happen,
but selected tokens are background-noise, uninformative patches, or stale distractors.
```

Check:

Trace:

```text
qk_score_mean
qk_score_top
qk_score_gap = top - mean
selected_frame_distribution
selected_token_positions
source_set_ids
```

Compare selection methods:

```text
qk_proxy
random same-budget recall
recent-only pseudo recall
high-importance only
hybrid qk + diversity
```

Expected interpretation:

```text
If qk_proxy is no better than random, retrieval scoring is weak.
If qk_proxy selects overly concentrated tokens, add diversity.
```

Action:

```text
Add descriptor z for TokenSet-level retrieval.
Add diversity penalty.
Add AdaMem-style rho later.
```

---

## 13. Hypothesis E: attention ignores recalled tokens

Failure mode:

```text
recalled_tokens > 0,
but attention mass to recall is near zero.
```

Check:

If attention weights can be obtained, log:

```text
attention_mass_to_recall
attention_mass_to_recent
attention_mass_to_anchor
```

If attention weights cannot be obtained, log proxy:

```text
mean QK score for recall tokens
mean QK score for recent tokens
ratio recall_qk / recent_qk
```

Expected interpretation:

```text
If recall mass is near zero, memory is ignored.
If recall mass is high but quality worse, memory is harmful.
```

Action:

```text
For ignored memory:
  add small region bias beta=0.03 or 0.05;
  reduce recall tokens to top 64/128 for higher precision.

For harmful memory:
  fix RoPE, reduce budget, or restrict layers/heads.
```

---

## 14. Hypothesis F: recall pollutes motion heads

Current head-aware routing is layer-level majority. It is not true per-head routing.

Failure mode:

```text
LAYOUT recall helps spatial consistency but hurts motion heads,
causing freezing, staticness, or dynamics degradation.
```

Check:

Run layer ablation:

```text
layer29 only
layer28 only
layer27 only
layers27-29
```

If possible, add per-head diagnostic:

```text
head_role distribution per enabled layer
n_motion_heads
n_layout_heads
layer_routing_role
```

Expected interpretation:

```text
If one layer is safe but last3 is worse, recall is over-injected.
If motion-dominated layers are worse, routing is too coarse.
```

Action:

```text
Short term: use layer ablation and only enable safest layer.
Medium term: add per-head region bias [H, K].
Long term: split-head K/V only if necessary.
```

---

## 15. Hypothesis G: active cache ordering or masking is wrong

Current active cache order is effectively:

```text
[recalled tokens, recent tokens]
```

Failure mode:

```text
attention implementation or mask assumes native chronological order;
putting recalled tokens before recent tokens may interact badly with masking or implementation assumptions.
```

Check:

Run ordering ablation:

```text
order = recall_first  # [recall, recent]
order = recent_first  # [recent, recall]
```

Trace:

```text
active_order
active_tokens
recent_tokens
recalled_tokens
```

Expected interpretation:

```text
If order changes quality significantly, active attention assembly or mask assumptions must be reviewed.
```

Action:

```text
Keep the safer order.
Ensure attention mask does not impose invalid causality assumptions over recalled tokens.
```

---

## 16. Hypothesis H: recall budget is too large

Current active cache can add 512 recalled tokens to a large recent window.

Failure mode:

```text
Too many recalled tokens dilute or distort softmax.
Quality drops even if tokens are valid.
```

Check:

Run budget ablation:

```text
recall_top_tokens = 32
recall_top_tokens = 64
recall_top_tokens = 128
recall_top_tokens = 256
recall_top_tokens = 512
```

Expected interpretation:

```text
If small budgets help more than large budgets, precision matters more than coverage.
```

Action:

```text
Default to 64 or 128 until token quality is proven.
```

---

## 17. Hypothesis I: prompt suite does not require long memory

Failure mode:

```text
LifeCache works technically,
but prompts do not require recovering old scenes or identities.
Native Self-Forcing already performs similarly.
```

Check:

Use explicit memory-demanding prompts:

```text
A-B-A scene revisit
same object disappears and returns
red kitchen -> blue kitchen -> red kitchen
hard switch without return
```

Expected interpretation:

```text
If gains appear only on A-B-A prompts, LifeCache is a long-memory method, not a general quality enhancer.
If no gains appear even there, memory content or model bottleneck is likely the issue.
```

Action:

```text
Use a targeted benchmark before claiming method failure.
```

---

## 18. Recommended next diagnostic sequence

Run in this order.

### Stage 1: Correctness checks

```text
C1. Verify recalled_tokens > 0.
C2. Verify enabled layers only.
C3. Verify retrieval_query_space matches bank_rope_mode.
C4. Verify frame_positions are preserved into recall:view.
C5. Verify remap policy actually matches config.
```

### Stage 2: RoPE ablation

```text
R1. near-only post-RoPE recall
R2. far post-RoPE recall
R3. pre-RoPE + zero remap
R4. pre-RoPE + relative-clamp remap
```

### Stage 3: Non-RoPE ablation

```text
N1. recall_top_tokens = 32/64/128/256/512
N2. layer29 / layer28 / layer27 / last3
N3. q_pre_rope retrieval vs roped_query retrieval
N4. recall_first vs recent_first ordering
N5. capture timestep policies
N6. A-B-A prompt suite vs generic continuation prompts
```

---

## 19. Decision table

| Observation | Likely cause | Next action |
|---|---|---|
| far post-RoPE worse than near-only | RoPE mismatch | use pre-RoPE + remap |
| zero remap stable but no gain | temporal structure collapsed | implement relative-clamp |
| relative-clamp stable but no gain | retrieval/content issue or weak benchmark | add descriptor, A-B-A prompts |
| recall mass near zero | memory ignored | small region bias, lower budget |
| recall mass high but quality worse | harmful memory | reduce budget, fix RoPE, restrict layers |
| layer29 works but last3 worse | over-injection | enable only safest layer |
| low-noise capture works better | noisy memory | timestep-filter capture |
| random recall similar to qk recall | retrieval weak | improve scoring/diversity/rho |
| A-B-A improves, generic prompts not | expected long-memory behavior | report as targeted benefit |
| no A-B-A improvement after all fixes | old K/V may not be bottleneck | pivot to semantic descriptors / Causal-Forcing |

---

## 20. Concrete next coding checklist

```text
[ ] Change recall retrieval query to q_pre_rope when bank uses pre_rope K.
[ ] Add retrieval_query_space trace.
[ ] Extend RecallResult with frame_positions and rope_mode.
[ ] Propagate frame_positions into recall:view TokenSet.
[ ] Add actual rope_remap_policy switch: zero vs relative_clamp.
[ ] Implement relative_clamp mapping from recalled frame_positions.
[ ] Add remap trace: source frame range and mapped t_pos range.
[ ] Add capture_timestep / capture_loop_index if available.
[ ] Add qk proxy comparison: recall qk vs recent qk.
[ ] Add recall budget ablation configs: 32/64/128/256/512.
[ ] Add layer ablation configs: 27, 28, 29, 27-29.
[ ] Add active order config: recall_first / recent_first.
[ ] Add A-B-A prompt file.
[ ] Run RoPE ablations first, then non-RoPE ablations.
```

---

## 21. Final recommendation

Current priority should be:

```text
1. Treat RoPE as the top hypothesis, not the final answer.
2. Fix q/k retrieval space consistency.
3. Preserve frame_positions into recall:view.
4. Implement true relative-clamp remap.
5. In parallel, prepare non-RoPE ablations: budget, layer, capture timestep, active order, prompt suite.
```

Do not add VLM/entity memory yet. If the current recalled K/V path is not position-safe or not selected in the right query space, higher-level memory modules will only make the failure harder to diagnose.
