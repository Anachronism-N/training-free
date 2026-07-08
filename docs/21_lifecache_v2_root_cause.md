# LifeCache-v2 Root Cause Analysis — Why No Improvement

> 2026-07-09 | Deep investigation of why v2 still shows no improvement over native SF

## Executive Summary

LifeCache-v2 shows no improvement because **no evicted tokens are ever captured during clean context refresh**. The `capture_enabled` flag is only True during the clean context pass, but the KV cache indices (`global_end_index`, `local_end_index`) have already been advanced by the denoising steps. Since `current_end > global_end_index` is False during clean context, the eviction condition is never met, and `_lifecache_evicted_list` stays empty.

Additionally, the pre-RoPE K capture at `causal_model.py:285` is incorrect — it captures tokens from the **new block's K** (`k[:, :num_evicted_tokens]`), not from the **evicted history** (`kv_cache["k"][:, :num_evicted_tokens]`).

## Detailed Analysis

### Bug 1 (CRITICAL): Eviction Timing Mismatch

**The Self-Forcing pipeline flow:**

```
For each block (frames 0..119):
  Step 3.1: Spatial denoising loop (4 steps)
    self.generator(noisy_input, timestep, kv_cache, ...)
    → attention forward: writes K/V, advances global_end_index
    → Block 1-7: no eviction (cache < 21 frames)
    → Block 8+: EVICTION HAPPENS HERE
    → capture_enabled=False → NO CAPTURE

  Step 3.2: Record output

  Step 3.3: Clean context refresh
    begin_capture()  ← capture_enabled=True
    self.generator(denoised_pred, context_timestep, kv_cache, ...)
    → attention forward: writes K/V, BUT global_end_index already = current_end
    → current_end > global_end_index is FALSE
    → NO EVICTION, _lifecache_evicted_list stays empty
    end_capture()

  Pipeline: processes _lifecache_evicted_list
  → List is empty → NO TOKENS STORED IN BANK
```

**Result**: The LifeCache bank never receives any tokens. Compression, recall, and anchor promotion are all dead code.

### Bug 2 (HIGH): Wrong pre-RoPE K Slice

At `causal_model.py:285`:
```python
evicted_k_pre_rope = k[:, :num_evicted_tokens].clone()
```

`k` is the current block's pre-RoPE key, shape `[B, num_new_tokens, H, D]`. Taking `[:num_evicted_tokens]` gives the **first `num_evicted_tokens` tokens of the new block**, which are NOT the evicted history tokens.

The evicted tokens are at `kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens]` — these are the oldest tokens being rolled out of the cache. But these are post-RoPE.

### Bug 3 (MEDIUM): k.shape vs num_evicted_tokens

`k` shape is `[B, num_new_tokens, H, D]` where `num_new_tokens = num_frames_per_block * 1560 = 4680`.

When `num_evicted_tokens = 4680` (typical), `k[:, :4680]` gives all new tokens — not evicted history.

When `num_evicted_tokens > num_new_tokens` (shouldn't happen in normal operation), this would fail with an index error.

## Why Near-Only Recall Shows Same Output

The near-only recall experiment produced output files with identical sizes to native SF (7.0M/5.9M/9.8M). This is consistent with the analysis above — if no tokens are stored in the bank, `compose_active_cache()` finds empty compressed/anchor lists, recall returns empty, and the active K/V is identical to native.

## What Needs to Change

### Fix 1: Capture During Denoising (or Reset Indices)

Option A: Capture evicted tokens during ALL forward passes (remove `capture_enabled` gate):
```python
# Remove: if not rt.config.capture_clean_only or rt.capture_enabled:
# Always capture when lifecache_manager is present
```

This would capture noisy denoising K/V into the bank, which the design doc says to avoid. But it would at least populate the bank.

Option B: Reset `global_end_index` before clean context refresh so eviction triggers again:
```python
# Before clean context refresh:
for cache in self.kv_cache1:
    cache["global_end_index"] = cache["local_end_index"].clone()
    cache["local_end_index"].fill_(0)
```
This would cause the clean context pass to re-populate the cache and trigger fresh evictions. But it would also lose the attention window context.

Option C (Recommended): Capture eviction during ALL forward passes, but only compress/store during clean context. Buffer the evicted payloads and process them all after the clean context refresh:
```python
# In attention forward: always capture (no capture_enabled gate)
kv_cache.setdefault("_lifecache_evicted_list", []).append(payload)

# In pipeline: always process the list after EACH generator call
for cache in self.kv_cache1:
    evicted_list = cache.pop("_lifecache_evicted_list", [])
    # Only compress and store if this was clean context
    for payload in evicted_list:
        if is_clean_context or not config.capture_clean_only:
            runtime.on_kv_evicted(...)
```

### Fix 2: Correct pre-RoPE K Capture

The pre-RoPE K for evicted tokens is not directly available because only post-RoPE K is stored in the cache. Options:

A) Store pre-RoPE K in the cache alongside post-RoPE K:
```python
kv_cache["k_pre_rope"][:, local_start:local_end] = k
```
Then capture from `kv_cache["k_pre_rope"]`.

B) Accept post-RoPE K for now and use the near-only distance filter:
```python
evicted_k = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
evicted_k_pre_rope = None  # not available
```
With `max_post_rope_frame_distance=21`, only recently-evicted tokens would be recalled, and their RoPE positions would be within the training range.

### Fix 3: Store frame_positions Correctly

Current code:
```python
frame_positions = token_indices // frame_seqlen
```
`token_indices` are global cache positions (e.g., 32760, 37440...). Dividing by `frame_seqlen=1560` gives frame positions like 21, 24... which are the absolute frame positions of the evicted tokens. This is correct.

## Implementation Plan

### Immediate Fix (highest priority)

1. Remove `capture_enabled` gate in attention forward — always capture evicted tokens
2. Fix pre-RoPE K capture — use post-RoPE from cache as fallback (pre-RoPE = None)
3. In pipeline, only compress/store captured tokens during clean context (check `capture_reason == "clean_context"`)
4. Fix `k[:, :num_evicted_tokens]` → use `kv_cache["k"][:, sink_tokens:sink_tokens+num_evicted_tokens]`

### Secondary Fixes

5. Store `frame_positions` correctly (current implementation is correct, just verify)
6. Add trace logging to verify tokens are being captured and stored

## References
- `causal_model.py:273-303` — eviction condition and capture
- `causal_inference.py:207-235` — denoising loop (Step 3.1)
- `causal_inference.py:245-254` — clean context refresh (Step 3.3)
- `pipeline/causal_inference.py:148-151` — cache index reset
