# LifeCache v2 Optimized — Why No Gain (Deep Trace Analysis)

> 2026-07-09 | Root cause from JSONL trace analysis

## Executive Summary

v2 optimized shows no improvement because **recall never actually retrieves any tokens**. The bank stores 30,720 compressed tokens across all 30 layers, but `recall_tokens()` always returns empty. The issue is in the recall pipeline: tokens are stored with `head_group="generic"` but queried with `head_group="layout"`, causing `group_match=0` for all candidates. Combined with other scoring issues, the top-k selection returns no viable token sets.

## Trace Evidence

```
Total events: 41,314
Event types: {'on_kv_evicted': 5,940, 'compose_active_cache': 35,374}
Max bank tokens: 49,152

WARN_NO_RECALL: compose_active_cache events exist but recalled_tokens is always 0
WARN_ALL_LAYERS_ENABLED: 30 layers seen (but enable_layers=(27,28,29) is set)
```

Key findings:
1. **Bank has tokens**: 5,940 eviction events × 512 compressed tokens = 3,041,280 stored across all layers
2. **Recall runs**: 35,374 compose events across layers 27-29
3. **Recall returns empty**: `recalled_tokens=0` for ALL compose events
4. **Bank visible**: `bank_total_tokens` grows from 0 to 30,720, proving bank is populated
5. **advance_step() never called**: All events have step=0

## Root Cause Chain

### 1. head_group mismatch

Pipeline stores tokens with `head_group="generic"`:
```python
rt.on_kv_evicted(layer_id=layer_id, head_group="generic", ...)
```

Attention forward queries with `head_group="layout"`:
```python
rt.compose_active_cache(layer_id=block_index, head_group="layout", role=HeadRole.LAYOUT, ...)
```

In `score_token_sets`:
```python
group_match = 1.0 if token_set.head_group == head_group else 0.0
```

All bank tokens get `group_match=0.0`. While this doesn't directly cause empty results (query_weight still contributes), it reduces scores significantly.

### 2. Pipeline doesn't filter by should_enable_layer

The pipeline processes evicted tokens for ALL 30 layers, not just the enabled 3. This wastes bank capacity on non-recall layers. Fix: add `should_enable_layer` check in the pipeline loop.

### 3. advance_step() never called

The runtime's step counter stays at 0 throughout the entire generation. This means `last_used_step` tracking is non-functional, and `usage_weight` scoring is meaningless.

### 4. TokenSetBank stores with head_group="generic"

When recall queries `bank.list_sets(layer_id=27)`, it gets tokens with `head_group="generic"`. The `head_group_weight=0.25` in scoring is wasted because `group_match=0` for all candidates.

### 5. No per-head routing

All 12 heads per layer get the same LAYOUT treatment. WAVE/MOTION heads should skip recall entirely (recall=0 in their budget), but they receive the same active cache as LAYOUT heads.

## Concrete Fixes Needed

1. **Fix head_group**: Store tokens with `head_group="layout"` in pipeline, or remove head_group filter from recall scoring
2. **Filter pipeline by should_enable_layer**: Only process evicted tokens for enabled layers
3. **Call advance_step()**: After each block's processing in the pipeline
4. **Add recall trace**: Log number of candidates, selected sets, and scores in trace
5. **Verify recall_tokens**: Add debug trace to `retrieve_token_sets` to log candidate count and selection count

## Immediate Action

The simplest fix: change the pipeline's `on_kv_evicted` call to use `head_group="layout"` instead of `"generic"`, matching the attention forward query. This ensures `group_match=1.0` for all tokens.

Second fix: add `should_enable_layer` check in pipeline's eviction processing loop to only store tokens for layers 27-29.

Third fix: call `rt.advance_step()` after processing each block's evicted tokens.
