# Reference code synthesis for LifeCache

This document summarizes the third-party code notes already read in this repository and turns them into implementation constraints for LifeCache-Forcing.

## 1. Main implementation target

Self-Forcing should be the first patch target. Its AR loop refreshes a clean KV cache after each generated chunk, and the attention module already exposes the exact point where post-RoPE K and raw V are inserted into the native cache. LifeCache should intercept active K/V composition at this self-attention point rather than changing the diffusion scheduler or VAE.

Causal-Forcing should be the second target. It shares the same Wan-style cache layout but may require symmetric handling of positive and negative CFG caches.

## 2. Cache layout shared by the main baselines

The common Wan-style cache is one dictionary per transformer block. Each cache stores K and V tensors shaped like `[B, cache_tokens, heads, head_dim]`, and for Wan2.1-T2V-1.3B the common shape is `[B, cache_tokens, 12, 128]`. A latent frame corresponds to 1560 tokens in the 480x832 path.

LifeCache entries should therefore point to spans inside this K/V tensor layout rather than creating an independent external memory payload.

## 3. Lessons from each reference implementation

### Self-Forcing

- Provides the cleanest integration point.
- Native policy is sink plus rolling recent/local cache.
- Cache stores post-RoPE K and raw V.
- LifeCache v1 can be post-RoPE compatible.

### Causal-Forcing

- Same basic cache layout as Self-Forcing.
- CFG means positive and negative caches must be handled carefully.
- The safest first version applies the same lifecycle policy to both caches.

### RollingForcing

- Confirms that active cache can be assembled as anchor plus working cache plus current K/V.
- LifeCache generalizes this to anchor plus recall view plus motion plus recent plus current.

### Pyramid-Forcing

- Confirms that cache policy should be per-head rather than global.
- Its head labels can be reused as a prior for LifeCache head roles.
- LifeCache should add lifecycle validity and typed recall on top of head-aware cache composition.

### Forcing-KV

- Provides the most concrete implementation basis for motion memory.
- Splits grouped attention into spatial/static and temporal/dynamic branches.
- Dynamic temporal K/V should become LifeCache motion slots.

### MemRoPE

- Motivates the RoPE-safe advanced version.
- v1 can use post-RoPE K for compatibility.
- v2 should store pre-RoPE K or temporal/spatial position maps and reapply relative RoPE at active-cache composition time.

### LongLive-RAG

- Confirms that recall is a temporary active view over compressed/offloaded history.
- LifeCache should retrieve K/V entry pointers, not only latent/frame memories.

### Echo-Forcing

- Closest conceptual reference for preserve, recall, and forget.
- Confirms that compressed history and recall are intertwined.
- Difference-aware decay motivates LifeCache stale/invalid metadata.

## 4. Refined LifeCache states

Persistent storage states:

- recent
- anchor
- compressed
- motion

Validity metadata states:

- valid
- stale
- invalid
- dropped

Runtime role:

- normal
- recall

Important distinction: compressed is where a KV entry is stored; recall is whether that entry is temporarily selected into the active cache for the current chunk.

## 5. Active cache composition

Default per-head composition:

- layout or anchor heads: anchor + scene recall + recent + current
- entity heads: anchor + entity recall + recent + current
- motion or wave heads: motion + recent + tiny anchor + current
- generic heads: anchor + recent + current

Compressed entries should not be visible by default. They become visible only when selected as recall entries.

## 6. Immediate code changes

1. Split `SlotState` into `StorageState`, `ValidityState`, and `RuntimeRole`.
2. Change `retrieve` to support layer, head, head role, storage-state, validity, scene, entity, and motion filters.
3. Change `compose_active_cache` into base-entry selection plus recall-entry selection.
4. Add a motion module based on dynamic temporal K/V entries.
5. Add stale filters for scene conflict, entity or state conflict, quality invalidity, and RoPE risk.
