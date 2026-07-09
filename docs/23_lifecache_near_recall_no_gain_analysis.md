# LifeCache Near-Only Recall — Why No Gain Analysis

> 2026-07-09 | Deep analysis of why near-only recall and pre-RoPE remap show no improvement

## Executive Summary

Near-only recall and pre-RoPE remap show no improvement because **recalled tokens are never actually different from the native recent window**. The core design issue is that LifeCache compresses evicted tokens from the sliding window and then recalls them back into attention — but the sliding window already contains the most recent 21 frames of K/V. Since evicted tokens are always older than 21 frames and the near-only filter rejects them (or when unfiltered, they carry stale RoPE positions), the recalled tokens either:
- Don't exist (filtered out by max_frame_distance)
- Are identical in content to what's already in the window (near-only passes because frame_ids is wrong)
- Produce garbage attention scores (post-RoPE with wrong positions)

## Root Cause 1: frame_ids = current block frames, not evicted frame positions

**File**: `pipeline/causal_inference.py:266-269`

```python
frame_ids = list(range(
    current_start_frame,
    current_start_frame + current_num_frames,
))
```

This sets `frame_ids` to the frames of the **current block being generated** (e.g., `[0, 1, 2]` for block 1, `[117, 118, 119]` for block 40). These are passed to `on_kv_evicted()` and stored in the TokenSet.

When `max_frame_distance=21` is checked in `recall.py:83-92`:
```python
center = sum(token_set.frame_ids) / len(token_set.frame_ids)
if abs(center - current_frame) <= config.max_frame_distance:
    filtered.append(token_set)
```

The `center` of frame_ids is always the center of the current block (e.g., `1.0`), and `current_frame` is the query's frame (~118). The distance `118 - 1 = 117 >> 21` — so **ALL tokens are filtered out** when `max_frame_distance=21` is set.

But this filter shouldn't even be checking `frame_ids` — it should be checking the **evicted tokens' actual frame positions**, which are stored in `frame_positions` from the attention forward. The `frame_positions` field on TokenSet is correctly set from the payload's `frame_positions` but is never used in filtering.

## Root Cause 2: frame_positions is computed from token_indices (wrong)

**File**: `causal_model.py:294`

```python
frame_positions = token_indices // frame_seqlen
```

`token_indices` are set to `[sink_tokens, sink_tokens + num_evicted_tokens]` which are indices in the **KV cache buffer**, not absolute frame indices. When sink_tokens=0:
- Block 8 eviction: token_indices = `[0, 4680]`, frame_positions = `[0, 1, 2]`
- Block 9 eviction: token_indices = `[4680, 9360]`, frame_positions = `[3, 4, 5]`

These are the **correct** absolute frame positions of the evicted tokens! So `frame_positions` is actually correct.

But `frame_ids` (from pipeline) is also `[0, 1, 2]` for block 1, which happens to match the evicted frame positions for block 8. This is a coincidence — the current block's frame_ids happen to match the evicted tokens' frame positions because eviction happens at a fixed offset (21 frames behind current).

## Root Cause 3: The distance filter passes but with wrong intent

Because `frame_ids` matches the evicted tokens' actual frames (coincidentally for the first eviction), the filter passes. But for later evictions (block 20+), `frame_ids` is `[57, 58, 59]` while the evicted tokens are from frames 36-38. The center distance is `(118 - 58) = 60 > 21`, so **later evictions are filtered out**.

This means near-only recall only works for tokens evicted in blocks 8-14 (frames 21-42), not for tokens evicted later. And since the model needs long-range recall (frames 0-20 recalled at frame 100+), the near-only filter prevents exactly the long-range recall that would provide benefit.

## Root Cause 4: All heads get the same LAYOUT role treatment

**File**: `causal_model.py:449`

```python
role=HeadRole.LAYOUT,
```

All 12 heads per layer receive `LAYOUT` role treatment (anchor=256, recall=512). WAVE/MOTION heads (156/360) should NOT receive layout/anchor tokens — they should receive motion/recent tokens only. By forcing layout tokens on motion heads, we may be degrading motion quality.

## Root Cause 5: Recall tokens are concatenated to the attention window

**File**: `runtime.py:compose_active_cache()` → `active_cache.py:compose()` → `torch.cat([s.k for s in selected], dim=0)`

Recalled tokens are concatenated BEFORE the recent window. The attention window becomes:
```
[recalled_K | recent_K]
```

The recalled K carries post-RoPE from old positions (e.g., frame 0-2 roped at position 0). The query is roped at position 118. The RoPE angle difference is `freqs * 118`, which is far outside the training range. This produces **noisy attention scores** that contaminate the softmax.

Even if the recalled tokens are semantically correct, the RoPE mismatch makes them unusable.

## Root Cause 6: Bank tokens are stored per-layer but queried by layer_id

The bank stores tokens with `layer_id=27/28/29` (the enabled layers). The `compose_active_cache` queries with `layer_id=block_index` which is also 27/28/29. This should match. But the compressed tokens are stored with `set_id=f"compressed:L{layer_id}:C{chunk_id}:S{step}"` — if `layer_id` matches, the query should find them.

## Summary of Why No Gain

| Issue | Impact |
|---|---|
| near-only filter uses wrong frame_ids | Later evictions filtered out, only early tokens pass |
| Recalled K has stale RoPE positions | Attention scores are noisy, contaminating softmax |
| All heads get LAYOUT treatment | Motion/WAVE heads forced to attend to layout tokens |
| Recalled tokens are prepended to recent | Extra tokens increase memory usage without benefit |
| No per-head routing | One-size-fits-all recall, no specialization |

## What Would Make It Work

### Fix 1: Use frame_positions instead of frame_ids for distance filtering
The `retrieve_token_sets` function should check `frame_positions` (from TokenSet v2 field), not `frame_ids` (from pipeline parameter).

### Fix 2: RoPE remap for recalled K
Before attention, recalled K must be re-roped to a legal relative position. Options:
- Store pre-RoPE K in bank → re-rope at recall time using `causal_rope_apply_pos`
- Or: remap post-RoPE K using complex rotation (expensive but works)

### Fix 3: Head-aware routing
- WAVE/MOTION heads: skip recall, only use recent window
- LAYOUT/ANCHOR heads: use recall + anchors
- GENERIC heads: recent only

### Fix 4: Better eviction frame tracking
Use `frame_positions` from the attention forward payload, not `frame_ids` from the pipeline.

## References
- `docs/21_lifecache_v2_root_cause.md` — Previous root cause (eviction timing)
- `docs/20_lifecache_v2_code_level_design.md` — v2 design spec
- `0623/06_实验记录与发现_LOG.md` Round 41 — RoPE position extrapolation confirmed
- `Pyramid-Forcing/pyramidkv/rope.py` — PF's dynamic_rope implementation
