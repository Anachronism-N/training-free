# LifeCache-Forcing implementation plan

## Architecture

Use `LifecycleKVCache` as the KV control plane and keep tensor payloads in normal K/V storage. The active attention cache for each layer/head is composed as:

```text
anchor + recall_view(compressed/history) + motion + recent + current
```

`recall_view` is a selection result, not a persistent pool.

## File-level changes

1. `src/lifecycle_kv/cache_types.py`
   - Keep `CacheEntry` as the metadata unit.
   - Track `rope_mode`, `rope_range`, and `position_map_ptr`.
   - Use `STALE`/`INVALID` as hard-blocked states.

2. `src/lifecycle_kv/index.py`
   - Metadata index by scene, entity, layer/head, head role, and lifecycle state.

3. `src/lifecycle_kv/head_profiler.py`
   - Offline first-pass profiler from attention maps.
   - Seed roles from Forcing-KV static/dynamic JSON or Pyramid labels when available.

4. Self-Forcing or Causal-Forcing `pipeline/causal_inference.py`
   - Instantiate `LifecycleKVCache`.
   - During clean context refresh, register chunk K/V as `current -> recent`.
   - After each chunk, run promotion/compression/stale transitions.

5. Self-Forcing or Causal-Forcing `wan/modules/causal_model.py`
   - Add optional `lifecycle_cache`, `layer_id`, `chunk_id`, and prompt state.
   - In `CausalWanSelfAttention.forward`, compose active K/V before attention.
   - Preserve old `kv_cache` path behind a flag for ablations.

## Answers to key questions

1. `recall` should be a temporary view. Code reason: active attention only needs a concatenated K/V tensor for the current query; duplicating recalled K/V into another persistent pool creates stale state and invalidation bugs.
2. `compressed` entries should be selected from evicted recent entries. v1: keep anchor/sink frames, compress or offload older chunks, score by trust, scene/entity match, access count, motion score, and RoPE safety.
3. Motion heads should start from Forcing-KV dynamic heads, then refine using locality, temporal mask degradation, and latent-delta sensitivity.
4. Stale/invalid entries should stay in the index but be hard-filtered from active attention unless an ablation explicitly enables stale recall.
5. Minimal Self-Forcing modification: in `CausalWanSelfAttention.forward`, replace `attention(roped_query, kv_cache["k"][...], kv_cache["v"][...])` with `compose_active_cache(...)`, then append current `roped_key/v`.
6. Ablations: recent-only, anchor+recent, compressed recall on/off, post-RoPE vs position-map recall, head-aware vs global policy, stale hard-block vs soft decay, motion slots on/off.

## Experiment uncertainties

- Whether post-RoPE historical K is good enough for recall beyond the training frame window.
- How many anchor tokens per head are needed before identity/layout drift improves.
- Whether dynamic K similarity or latent-delta better identifies motion heads.
- Whether stale hard-blocking hurts deliberate scene return prompts.
- Interaction between RollingForcing windows and clean-cache trust scoring.

