# LifeCache-v1 Quality Analysis — Why All Versions Degrade After 10s

> 2026-07-08 | Root cause analysis of late-video degradation

## Observed Symptoms

All LifeCache variants (trace-only, compression-only, union recall) show:
- Significant darkening after ~10s
- Subject nearly frozen / identity lost in final seconds
- Bright hallucination artifacts in background of last few frames
- No significant improvement over SF base

## Root Cause Chain

### Issue 1 (CRITICAL): RoPE Position Mismatch in Recalled Tokens

**The fundamental problem**: evicted K/V tokens stored in the bank carry absolute RoPE positions from when they were generated. When recalled later (e.g., tokens from frame 0-7 recalled at frame 118), the query-key angle difference is `freqs * (118 - 0)` — far exceeding the 21-frame training domain.

This is the same "absolute position extrapolation" identified in the 0623 experiments (Round 41). Pyramid-Forcing solves this with `dynamic_rope` (`relative_clamp` in `pyramidkv/rope.py`).

**Impact**: Recalled tokens produce noisy attention scores, contaminating the softmax and driving the model toward dark/static output.

### Issue 2 (CRITICAL): Recall Never Activated for GENERIC Role

In `runtime.py:compose_active_cache()`, the role is hardcoded as `HeadRole.GENERIC`:

```python
role: HeadRole = HeadRole.GENERIC,
```

`DEFAULT_BUDGETS[HeadRole.GENERIC]` has `recall=0`, `motion=0`, `anchor=128`. Since recall=0, the condition `if budget.recall > 0` is False, and **recall_tokens() is never called**. The compose just takes anchors (if any) and returns — essentially a no-op for GENERIC.

**Impact**: Even when `recall_enabled=True` in config, the union recall experiment is functionally equivalent to compression-only.

### Issue 3 (HIGH): `enable_last_n_layers` Not Applied

The config specifies `enable_last_n_layers: 6`, but `LifeCacheRuntimeConfig` has no `enable_last_n_layers` field. The field exists only as a YAML key, not as a dataclass attribute. In `LifecycleCacheManager.__init__`:

```python
last_n = getattr(runtime.config, "enable_last_n_layers", 0) or 0
```

`getattr` returns 0 because the dataclass doesn't have this field. The fallback `or 0` keeps it at 0. So `enable_layers` stays `None`, meaning **all 30 layers are enabled** — opposite of the intended "last 6 only".

**Impact**: Recall computation runs on all 30 layers instead of just 6, causing unnecessary overhead without benefit.

### Issue 4 (MEDIUM): All Heads Use Same Generic Treatment

In `causal_model.py`, the attention forward passes `head_group="generic"` to `compose_active_cache()`. There's no per-head routing — all 12 heads per layer receive the same treatment regardless of their profiled roles (WAVE/LAYOUT/GENERIC).

**Impact**: WAVE/MOTION heads are forced to attend to layout/anchor tokens; LAYOUT heads can't specialize in long-range recall.

### Issue 5 (MEDIUM): Q-K Proxy Compression Uses Poor Query

In `causal_inference.py`, the query passed to `on_kv_evicted()` is:

```python
q_proxy = evicted_k.mean(dim=0, keepdim=True)  # [1, heads, dim]
```

This is the mean of the evicted K itself, not the actual query. Using evicted K as a proxy for Q gives meaningless similarity scores.

**Impact**: Compression selects essentially random tokens — no better than uniform sampling.

## Current Architecture Flow (with bugs annotated)

```
Per generation block:
  1. Attention forward: evict old tokens → capture to _lifecache_evicted
     BUG: tokens carry outdated RoPE positions (Issue 1)

  2. Pipeline: read _lifecache_evicted → on_kv_evicted()
     → qk_proxy_scores(mean(evicted_k), evicted_k)  # BUG: self-similarity (Issue 5)
     → compress to 512 tokens
     → store in bank (COMPRESSED)
     → every 4 chunks: promote top → ANCHOR

  3. Each attention call:
     → compose_active_cache(role=GENERIC)  # BUG: recall=0 (Issue 2)
     → budget.recall=0 → skip recall
     → return native recent K/V unchanged
```

## Fix Plan

### Fix 1: Enable Recall for Union Mode
Change `compose_active_cache()` to use a role that has recall > 0. For union mode, use `HeadRole.LAYOUT` (recall=512, anchor=256) or create a "union" role with appropriate budgets.

### Fix 2: Fix enable_last_n_layers
Add `enable_last_n_layers: int = 0` to `LifeCacheRuntimeConfig`, and properly compute `enable_layers` from it.

### Fix 3: RoPE Remap for Recalled Tokens
Before feeding recalled K/V to attention, re-rope them to a position adjacent to the current query window. This requires:
- Storing K **un-roped** in the bank (capture `k` before `causal_rope_apply`)
- Or storing frame positions and re-roping at recall time

### Fix 4: Head-Aware Routing
Use Pyramid CSV head roles to route different heads to different token subsets:
- WAVE/MOTION heads → motion tokens + recent
- LAYOUT/ANCHOR heads → anchors + recall + recent
- GENERIC heads → recent only

### Fix 5: Better Q-K Proxy
Pass the actual query from the attention forward instead of using evicted K as proxy.

## Experiment Results Reference

| Run | Method | Time | Output change? |
|---|---|---|---|
| sf_native_120f | SF baseline | 6m 05s | — |
| sf_pyramid_120f | SF + Pyramid | 5m 36s | Yes |
| sf_lifecache_trace_120f | trace-only | 6m 07s | No |
| sf_lifecache_compression_120f | compression-only | 6m 17s | No |
| sf_lifecache_recall_120f | union recall | 10m 10s | Bug: no recall |

## References
- 0623 experiments: `/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/0623/`
- Round 41 log: absolute position extrapolation confirmed
- PF dynamic RoPE: `Pyramid-Forcing/pyramidkv/rope.py:map_dynamic_pos_time`
- Issue doc: `docs/18_lifecache_v1_issues.md`
