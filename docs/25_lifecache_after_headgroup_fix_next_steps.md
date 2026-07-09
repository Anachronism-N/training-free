# LifeCache after head_group fix: next debugging and experiment plan

> Purpose: record the next project actions after the latest `head_group` / `advance_step` fix.  
> Current stage: LifeCache is no longer a pure prototype. It is integrated into Self-Forcing and has an optimized v2 path, but the next step must be trace-driven verification rather than adding more memory modules.

---

## 1. Current repository status

Recent work has already implemented several important pieces:

```text
1. LifeCache v2 optimized configuration exists.
2. Self-Forcing integration exists.
3. Pre-RoPE K cache path has been introduced.
4. Evicted payloads are moved to CPU to avoid GPU OOM.
5. A run script entry for `optimized` exists.
6. The most recent critical fix changes bank storage head_group from `generic` to `layout` and calls `advance_step()`.
```

The latest critical fix addresses a real blocking bug:

```text
Bank had compressed tokens, but recall returned zero tokens.
Root cause: compressed TokenSets were stored with head_group="generic",
while attention recall queried head_group="layout".
```

After this fix, we need to rerun the optimized experiment and verify whether recall is finally active.

---

## 2. Main conclusion

The next project step is not to add new memory types.

The next step is to verify this minimal chain:

```text
clean-context eviction
  -> compressed TokenSets in bank
  -> recall candidates exist
  -> recalled_tokens > 0
  -> recalled K is RoPE-safe
  -> active attention uses recalled tokens
  -> A-B-A / scene-revisit prompts benefit
```

If this chain is not proven, any additional modules such as VLM anchors, entity memory, motion cache, or AdaMem-style rho will be difficult to interpret.

---

## 3. Immediate experiment: rerun latest optimized

### 3.1 Command

Use the existing optimized run entry:

```bash
bash scripts/run_experiments.sh optimized
```

Or manually:

```bash
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/Self-Forcing/scripts:${PYTHONPATH:-}"
export LIFECACHE_ENABLE=1
export LIFECACHE_CONFIG="$REPO_ROOT/configs/lifecache/lifecache_v2_optimized.yaml"

cd "$REPO_ROOT/third_party/Self-Forcing"

python inference.py \
  --config_path "$SF_CONFIG" \
  --output_folder "$REPO_ROOT/runs/sf_lifecache_v2_optimized_120f_after_fix" \
  --checkpoint_path "$SF_CHECKPOINT" \
  --data_path "$PROMPTS" \
  --num_output_frames 120 \
  --seed 0 \
  --num_samples 1 \
  --use_ema \
  --save_with_index
```

### 3.2 Do not inspect video first

Before watching the output, inspect the trace.

The first goal is to answer:

```text
Did recall actually happen after the head_group fix?
```

Required trace checks:

```text
bank_total_tokens > 0
recall_candidate_sets > 0
recall_candidate_tokens > 0
recalled_tokens > 0
active_tokens > recent_tokens
step is not always 0
enabled layers are the intended last 3 layers only
```

If `recalled_tokens` is still zero, stop video evaluation and debug the recall path.

---

## 4. Post-rerun trace checklist

### 4.1 Required event-level checks

Check these event types:

```text
on_kv_evicted
recall_candidates
compose_active_cache
```

For `on_kv_evicted`, verify:

```text
layer_id is only in enabled layers
head_group == layout
compressed_tokens_added > 0
rope_mode == pre_rope when pre-RoPE K exists
q_source is actual q, not evicted_k.mean fallback
```

For `recall_candidates`, verify:

```text
recall_candidate_sets > 0
recall_candidate_tokens > 0
candidate layer_id matches attention layer_id
candidate head_group matches recall query head_group
```

For `compose_active_cache`, verify:

```text
recalled_tokens > 0
recent_tokens > 0
active_tokens = recent_tokens + recalled_tokens + optional anchors/motion
region_counts contains recall
```

### 4.2 Required run-level checks

Aggregate over the whole run:

```text
max(bank_total_tokens)
mean(recalled_tokens) over enabled layers
max(recalled_tokens)
num compose events with recalled_tokens > 0
num compose events with recalled_tokens == 0
bank tokens by layer
```

Expected after the latest fix:

```text
recalled_tokens should no longer be always 0.
```

If it is still always 0, the next likely failure is candidate filtering or rope-mode filtering.

---

## 5. If recall is still zero

If the rerun still gives:

```text
bank_total_tokens > 0
but recalled_tokens == 0
```

then debug in this order.

### 5.1 Check layer filtering

The pipeline may still process all 30 layers even if recall is enabled only for the last 3 layers.

Add or verify this in the eviction-processing loop:

```python
if not rt.should_enable_layer(layer_id):
    continue
```

Expected trace after the fix:

```text
on_kv_evicted events only for enabled layers, e.g. 27/28/29.
bank_tokens_by_layer only contains those layers.
```

### 5.2 Check head_group alignment

Verify all stored TokenSets used for recall have:

```text
head_group == layout
```

and attention compose uses:

```text
head_group == layout
role == HeadRole.LAYOUT
```

If one side is `generic` and the other side is `layout`, recall scoring becomes weak or empty.

### 5.3 Check candidate filtering

Add temporary debug trace inside `retrieve_token_sets()`:

```text
candidate_sets_before_filter
candidate_sets_after_frame_filter
candidate_sets_after_rope_filter
score_min
score_max
score_mean
selected_sets
```

Most likely filters that can remove all candidates:

```text
max_frame_distance
rope_safe_recall
allow_post_rope_recall
layer_id mismatch
head_group mismatch
```

### 5.4 Check q/k shape and NaN

Trace:

```text
q_shape
k_summary_shape
qk_score_mean
qk_score_top
has_nan_score
```

If scores are NaN or extremely low, inspect whether recall scoring compares incompatible spaces:

```text
post-RoPE query vs pre-RoPE K
pre-RoPE query vs post-RoPE K
```

---

## 6. If recall is active but quality still does not improve

Once trace proves:

```text
recalled_tokens > 0
```

then the question changes from “why no recall” to “why recalled memory does not help”.

There are three cases.

### Case A: recalled tokens are ignored

Symptoms:

```text
recalled_tokens > 0
but attention_mass_to_recall is near zero
or Q-K proxy mass to recall is very low
```

Actions:

```text
1. Reduce recall_top_tokens to improve precision.
2. Add small recall/region bias, e.g. beta=0.05.
3. Improve token scoring with q_pre_rope and set descriptors.
```

Do not increase recall budget first. More bad tokens will not help.

### Case B: recalled tokens are used but video gets worse

Symptoms:

```text
attention uses recall
but video darkens, freezes, flickers, or hallucinates
```

Likely causes:

```text
RoPE remap is wrong.
Motion/WAVE heads are polluted by layout recall.
Too many recalled tokens contaminate softmax.
Recalled tokens are not from the right source frames.
```

Actions:

```text
1. Switch RoPE remap from zero-position remap to relative-clamp remap.
2. Reduce recall_top_tokens from 256 to 64 or 128.
3. Run layer ablation: layer 29 only, 28 only, 27 only, last3.
4. Add per-head region bias or disable recall for motion-like heads.
```

### Case C: recalled tokens are used but video is unchanged

Symptoms:

```text
recalled_tokens > 0
quality is close to native
no obvious improvement
```

Likely causes:

```text
Prompt does not require long-term recall.
Near recall is redundant with native recent cache.
Self-Forcing's current bottleneck is not old-K/V loss for this prompt.
```

Actions:

```text
1. Switch to A-B-A scene revisit prompts.
2. Use explicit object/scene return prompts.
3. Test 60s/120s only after 30s trace is stable.
4. Compare native vs optimized only on prompts where memory is required.
```

---

## 7. Fix RoPE remap next

The current optimized implementation reportedly re-ropes recalled K to position 0. This is better than unbounded stale RoPE, but it is still too coarse.

### 7.1 Problem with zero-position remap

Zero-position remap has several risks:

```text
1. Sparse recalled tokens do not form complete frames.
2. All recalled tokens lose their relative temporal spacing.
3. Different source frames collapse to the same temporal position.
4. It may behave like a static global anchor rather than real memory recall.
```

### 7.2 Required policy: relative-clamp remap

Use `TokenSet.frame_positions` to map recalled tokens into legal temporal positions.

For each recalled token:

```python
rel = (current_start_frame - frame_positions).clamp(0, TR - 1)
t_mapped = (TR - 1) - rel
```

For a motion-safe split:

```python
is_recent = rel < split_recent
rel_mapped = torch.where(
    is_recent,
    rel,
    torch.full_like(rel, TR - 1),
)
t_mapped = (TR - 1) - rel_mapped
```

Suggested defaults:

```text
TR = 21
split_recent = 4
```

### 7.3 Add remap modes

Add a config switch:

```yaml
rope_remap_policy: zero          # baseline
rope_remap_policy: relative_clamp
```

Run both:

```text
optimized_zero_remap
optimized_relative_clamp
```

Expected:

```text
relative_clamp should be at least as stable as zero remap.
If zero remap is unchanged but relative_clamp improves A-B-A prompts, the method direction is validated.
```

### 7.4 Required trace for remap

Add trace fields:

```text
rope_remap_policy
recalled_frame_pos_min
recalled_frame_pos_max
recalled_frame_pos_mean
mapped_t_pos_min
mapped_t_pos_max
mapped_t_pos_mean
num_recalled_sparse_tokens
num_recalled_full_frames_est
```

---

## 8. Fix retrieval query space

If bank stores pre-RoPE K, retrieval should use pre-RoPE query.

### 8.1 Current risk

If recall uses:

```python
q_for_life = roped_query[0]
```

while bank uses:

```text
rope_mode = pre_rope
```

then Q-K retrieval compares different spaces.

### 8.2 Required change

Use two query paths:

```python
q_for_retrieval = q[0] if q.ndim == 4 else q
q_for_attention = roped_query[0]
```

Then:

```python
active_k, active_v, view = rt.compose_active_cache(
    layer_id=block_index,
    q=q_for_retrieval,
    ...
)
```

After recall, re-rope recalled pre-RoPE K before attention.

### 8.3 Add trace

```text
retrieval_query_space: pre_rope / post_rope
bank_rope_mode: pre_rope / post_rope / mixed
recall_rope_mode: pre_rope / post_rope / mixed
```

---

## 9. Layer ablation plan

Since all heads are currently treated as layout heads, injecting recall into multiple layers can harm motion or texture.

Run these configs:

```text
optimized_layer29
optimized_layer28
optimized_layer27
optimized_last3
```

Suggested matrix:

| Run | Enabled layers | Purpose |
|---|---:|---|
| L29 | 29 only | test final-layer recall |
| L28 | 28 only | test near-final layer |
| L27 | 27 only | test deeper layer recall |
| L27-29 | 27,28,29 | current last3 baseline |

Expected interpretation:

```text
If one layer works better than last3, recall is useful but over-injected.
If no layer helps, memory selection or prompt suite may be the issue.
If layer29 is safest, start from only layer29 for further experiments.
```

---

## 10. Prompt suite must target memory

Do not judge LifeCache with generic continuation prompts only.

Use explicit A-B-A prompts.

### 10.1 Scene revisit prompts

```text
B1. A woman in a yellow coat stands in a small kitchen with blue cabinets and a red cup on a wooden table. She walks outside into a garden. Later she returns to the same kitchen and stands beside the same red cup.

B2. A white dog with a red collar runs through a park, disappears behind trees, then reappears near the same fountain still wearing the red collar.

B3. A robot walks in a clean white laboratory, moves into a dark hallway, then returns to the same laboratory with the same blue control panel.
```

### 10.2 Distractor prompts

```text
D1. A person enters a red kitchen, then a blue kitchen, then returns to the red kitchen.

D2. A cat sits on a striped sofa, later appears on a similar striped bed, then returns to the original sofa.
```

### 10.3 Hard switch prompts

```text
C1. A robot walks in a clean white laboratory. The scene suddenly cuts to a crowded night market with neon signs.

C2. A red kitchen scene changes to a blue ocean beach, with no return to the kitchen.
```

Interpretation:

```text
Scene revisit improves -> recall helps.
Hard switch worsens -> forgetting/rho needed.
Distractor wrong recall -> descriptor/rho needed.
```

---

## 11. Concrete next-agent checklist

Give the next coding agent this checklist:

```text
[ ] Rerun optimized after commit 5d3da335 and summarize trace.
[ ] Verify recalled_tokens > 0.
[ ] If recalled_tokens == 0, inspect recall_candidates and retrieve_token_sets filters.
[ ] Add should_enable_layer filter in pipeline eviction processing if missing.
[ ] Ensure on_kv_evicted events only appear for enabled layers.
[ ] Change LifeCache retrieval query from roped_query to q_pre_rope when bank rope_mode is pre_rope.
[ ] Add trace field retrieval_query_space.
[ ] Add trace fields bank_rope_mode and recall_rope_mode.
[ ] Add trace fields recalled_frame_positions min/max/mean.
[ ] Add rope_remap_policy switch: zero / relative_clamp.
[ ] Implement relative-clamp remap using TokenSet.frame_positions.
[ ] Add trace fields mapped_t_pos min/max/mean.
[ ] Run optimized_zero_remap vs optimized_relative_clamp.
[ ] Add layer ablation configs: layer_27, layer_28, layer_29, last3.
[ ] Add A-B-A scene revisit prompt file.
[ ] Run native / optimized_zero / optimized_relative_clamp on A-B-A only.
[ ] Only after recall is active and stable, consider rho/z soft retention.
```

---

## 12. Success criteria for the next iteration

This iteration is successful if the following table can be filled:

| Check | Expected |
|---|---|
| Bank grows | yes |
| Recall candidates exist | yes |
| recalled_tokens > 0 | yes |
| step increments | yes |
| only enabled layers store memory | yes |
| retrieval query space matches bank rope space | yes |
| relative-clamp trace exists | yes |
| A-B-A prompt has at least one qualitative improvement | target |
| hard-switch prompt does not collapse | target |

If the first seven are not satisfied, do not claim method-level failure.

If the first seven are satisfied but A-B-A still shows no gain, then the next explanation is more scientific:

```text
Either the selected old K/V is not the missing bottleneck for Self-Forcing,
or the memory needs a stronger descriptor/rho/forgetting mechanism.
```

---

## 13. Final recommendation

The current priority should be:

```text
1. Verify recall is no longer zero after head_group fix.
2. Filter bank storage to enabled layers only.
3. Use pre-RoPE query for pre-RoPE bank retrieval.
4. Replace zero-position remap with relative-clamp remap.
5. Run layer ablation.
6. Evaluate on A-B-A scene revisit prompts.
```

Do not yet add VLM, entity memory, or more cache pools.

The project is now bottlenecked by trace-proven correctness and RoPE-safe attention usage, not by missing high-level memory concepts.
