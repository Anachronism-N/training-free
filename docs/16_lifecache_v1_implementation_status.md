# LifeCache-v1 implementation status

This note records the current prototype state after adopting the v1 design in
`docs/11_lifecache_v1_design.md`.

## Implemented modules

The first-pass v1 code lives under `src/lifecycle_kv/`.

| Module | Status | Purpose |
|---|---|---|
| `tokenset.py` | implemented | Defines `TokenSet` and `CacheRegion`; this is the v1 memory-bank payload abstraction. |
| `bank.py` | implemented | Bounded `TokenSetBank` with per-region set/token budgets and simple pruning. |
| `compression.py` | implemented | Attention Participation top-k compression from evicted K/V and attention maps. |
| `recall.py` | implemented | Two-stage recall: TokenSet scoring followed by token-level Q-K top-k. |
| `anchor.py` | implemented | Fixed/dynamic anchor promotion helpers with simple v1 scores. |
| `motion.py` | implemented | Latent-delta, dynamic-K, boundary score, and motion TokenSet construction. |
| `head_roles.py` | implemented | Pyramid/Forcing-KV style CSV/JSON head-role prior loader. |
| `active_cache.py` | implemented | Head-role-specific active cache composition with region budgets and optional bias. |
| `instrumentation.py` | implemented | JSONL trace event writer and attention-region mass helper. |

The earlier `CacheEntry` / `LifecycleKVCache` classes are left in place as a
legacy prototype layer, but v1 implementation should use `TokenSet` first.

## Current integration boundary

The prototype is intentionally not wired into `third_party/Self-Forcing` yet.
The first integration point remains:

```text
third_party/Self-Forcing/wan/modules/causal_model.py
```

Specifically, LifeCache should be called in `CausalWanSelfAttention.forward`
after query/key RoPE has been applied and before the final attention call. At
that point the code has current Q, current K/V, and access to the native rolling
cache tensors. This is enough to:

1. trace K/V shape and region mass;
2. convert evicted recent K/V into `TokenSet`;
3. recall compressed/anchor/motion tokens using current Q;
4. concatenate the active K/V view for attention.

## Minimal next patch

The next implementation step should be Phase 0 plus Phase 1 in Self-Forcing:

```text
1. Add a disabled-by-default LifeCache config flag.
2. Instantiate a per-pipeline TokenSetBank and CacheTraceWriter.
3. In attention, emit trace events for layer/head/KV shape.
4. At recent-cache eviction, run AP top-k compression and add a compressed TokenSet.
5. Do not alter generated outputs until trace and bank token counts are verified.
```

Only after those checks should Phase 2 enable recalled tokens in the active
attention K/V path.

## Verification status

`python -m compileall src` passes in the current workspace.

The local Python environment does not have `torch` installed, so runtime tensor
smoke tests could not be executed here. They should be run inside the same
environment used for Self-Forcing inference.
