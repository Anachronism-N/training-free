# Code Reading Findings for LifeCache-Forcing

This note summarizes concrete implementation details observed from the current `training-free` scaffold and reference repositories.

## 1. Current training-free scaffold

The repository README already frames the idea correctly: LifeCache-Forcing is lifecycle-aware KV-cache management, and the memory bank is a cache index/control plane rather than an independent external generator condition.

Current code status:

- `src/lifecycle_kv/cache_types.py` defines `SlotState` with `CURRENT`, `RECENT`, `ANCHOR`, `COMPRESSED`, `MOTION`, `STALE`, `INVALID`, and `DROPPED`.
- `CacheEntry` stores layer/head/chunk/token span, `kv_ptr`, head role, scene/entity/state metadata, trust/motion/stale/conflict scores, and RoPE metadata.
- `src/lifecycle_kv/index.py` provides a metadata-only index by scene, entity, head, role, and slot state.
- `src/lifecycle_kv/lifecycle_cache.py` already implements the right high-level abstraction: store tensor payloads in `kv_store`, select entries by metadata, then compose active K/V tensors for one layer/head.

Important issue: `recall` should not be added as a persistent `SlotState` unless we explicitly need debugging labels. It should remain a temporary readout view from `COMPRESSED`, `ANCHOR`, or other stored entries.

## 2. Causal-Forcing implementation details

Repository: `thu-ml/Causal-Forcing`.

Important files:

- `pipeline/causal_inference.py`
- `wan/modules/causal_model.py`
- `long_video/pipeline/rolling_forcing_inference.py`
- `long_video/wan/modules/causal_model.py`

Key observations:

1. The inference loop initializes KV cache once and then repeatedly calls the generator block by block.
2. During denoising, the generator is called with `kv_cache=self.kv_cache1` and `current_start=current_start_frame * frame_seq_length`.
3. After each generated block, Causal-Forcing runs a clean/context pass to update the KV cache using the denoised prediction.
4. `_initialize_kv_cache` creates one cache dictionary per transformer block:

```python
{
    "k": torch.zeros([batch_size, kv_cache_size, 12, 128]),
    "v": torch.zeros([batch_size, kv_cache_size, 12, 128]),
    "global_end_index": ...,
    "local_end_index": ...,
}
```

5. In `wan/modules/causal_model.py`, the normal causal attention path writes roped K and unroped V into the cache, then attends over a local slice.
6. The long-video rolling version keeps a first anchor block, extracts a working cache, dynamically re-RoPEs the first anchor block, then attends over `[anchor_cache, working_cache, current]`.

Implication for LifeCache:

- The minimal patch point is `CausalWanSelfAttention.forward` after `q,k,v` are projected and before `attention(...)` is called.
- A lightweight LifeCache prototype should first operate on post-RoPE K, because that matches the existing cache path.
- A stronger version should follow MemRoPE/Pyramid-style pre-RoPE or dynamic-RoPE readout.

## 3. Rolling-Forcing / Causal long-video path

Causal-Forcing's `long_video/pipeline/rolling_forcing_inference.py` uses rolling windows over denoising steps and then updates a clean KV cache. The key line is that only the first block of the window is cached during the clean update pass.

Implication:

- Rolling-style long-video inference already separates generation window and cache update window.
- LifeCache should hook the clean update pass, because that is where persistent cache entries should be registered.
- During noisy denoising steps, entries should be tentative/current and should not be committed into long-term `anchor` or `compressed` storage.

## 4. Pyramid-Forcing implementation details

Repository: `IF-LAB-PKU/Pyramid-Forcing`.

Important files:

- `README.md`
- `configs/head_configs/best_labels.csv`
- `wan/modules/causal_model.py`
- `pyramidkv/cache.py`
- `pyramidkv/adaptive_cache.py`

Key observations:

1. Pyramid Forcing explicitly frames itself as a head-aware pyramidal KV cache framework.
2. It classifies 30 x 12 heads into behavior groups and stores labels in `configs/head_configs/best_labels.csv`.
3. The README states the cache composition is `[sink + middle + recent]`, with different policy per head.
4. `PyramidKVCache` owns per-layer/per-batch cache state and stores per-head `static_k/v` and `dynamic_k/v` lists.
5. `AdaptiveKVCache` extends it with per-head composition, sink-grid decoupling, dynamic RoPE remapping, and double-pass semantics.
6. `wan/modules/causal_model.py` routes the attention call to `pyramidkv_attention` if `kv_cache` is a `PyramidKVCache` instance.

Implication for LifeCache:

- Do not reinvent a heavy attention kernel first. Reuse the idea of per-head readout composition.
- The practical design should be:

```text
storage state: recent / anchor / compressed / motion
runtime view: recall
head role: anchor / layout / entity / motion / wave / veil / unknown
active readout: per head, gather entries and concatenate K/V
```

- Pyramid labels can be used as initial head roles:
  - `-1` oscillating -> wave / motion candidate
  - `1` stable compact -> anchor/layout candidate
  - `2` stable sparse -> sparse recall / semantic/layout candidate

## 5. Forcing-KV implementation details

Repository: `zju-jiyicheng/Forcing-KV`.

Important file:

- `wan/modules/causal_model_forcingkv.py`

Key observations:

1. Forcing-KV implements separate cache groups for spatial and temporal heads.
2. It builds grouped active caches:

```python
k1 = [group_sink_spatial_k, spatial_cache_k, cur_spatial_k]
k2 = [group_sink_temporal_k, dynamic_k, temporal_cache_k, cur_temporal_k]
```

3. It separates spatial heads and temporal heads via `extract_heads_triton`.
4. Dynamic temporal compression computes candidate patch/chunk similarity over old and new temporal K, then selects low-similarity chunks to preserve dynamic information.
5. Dynamic temporal cache is intentionally short-range and patch-level.

Implication for LifeCache:

- Motion should not be an abstract external vector only.
- Motion cache should primarily be dynamic/temporal-head KV plus optional latent-delta metadata.
- The first MVP can borrow Forcing-KV's idea:
  - split heads into spatial/layout and temporal/motion;
  - feed motion heads with `[motion + temporal_recent + current]`;
  - feed layout heads with `[anchor + compressed/recall + recent + current]`.

## 6. MemRoPE implementation details

Repository: `YoungRaeKimm/MemRoPE`.

Important files:

- `README.md`
- `wan/modules/causal_model.py`

Key observations:

1. MemRoPE uses a three-tier fixed cache:

```text
[Sink Tokens] + [Long EMA + Short EMA Memory Tokens] + [Local Window] + [Current Chunk]
```

2. The README explicitly states keys are stored without RoPE and RoPE is applied at attention time using block-relative indices.
3. In code, `compression_method='ema'` builds long-term and short-term EMA memories when rolling mode evicts tokens.
4. Long-term EMA captures distant stable content; short-term EMA captures recent dynamics.
5. It supports adaptive alpha based on token motion magnitude.

Implication for LifeCache:

- LifeCache should support two implementation levels:
  - v0: post-RoPE compatible, minimal patch to Causal/Self attention;
  - v1: pre-RoPE storage + dynamic RoPE readout, inspired by MemRoPE and Pyramid.
- Motion memory can be upgraded from recent dynamic KV to short-term EMA / adaptive EMA memory.
- Compressed entries should store whether they are post-RoPE or pre-RoPE.

## 7. LongLive-RAG implementation details

Repository: `qixinhu11/LongLive-RAG`.

Important file:

- `README.md`

Key observations:

1. LongLive-RAG inserts retrieved historical entries between sink and local windows:

```text
Sliding window:   [C_sink || C_loc]
LongLive-RAG:     [C_sink || M_t || C_loc]
```

2. It indexes completed latent blocks and retrieves top-K relevant memory for each current block.
3. It trains a retrieval autoencoder, while the base generator remains frozen.

Implication for LifeCache:

- The most useful part is the active-context layout `[sink || retrieved || local]`.
- For strict training-free work, avoid training a latent encoder in the first version.
- Replace trained latent retrieval with:
  - prompt/scene/entity metadata;
  - CLIP/frame embedding if available;
  - internal query/KV summary similarity;
  - state/stale validity checks.

## 8. Refined direction

The refined method should be called:

```text
LifeCache-Forcing: Lifecycle-aware Active KV Views for Training-Free Long Video Generation
```

Do not frame it as an external memory bank. Frame it as an active KV view composer:

```text
persistent KV storage + metadata index -> per-head active KV view -> attention
```

The main novelty should be:

1. `compressed` is storage; `recall` is a temporary view.
2. per-head active cache composition inherits Pyramid/Forcing-KV but adds semantic/state validity.
3. motion cache is dynamic-head KV plus short-term temporal signals.
4. stale/invalid metadata prevents wrong or obsolete history from entering attention.

## 9. Immediate next implementation tasks

1. Add `PromptState` / `CacheQuery` dataclasses.
2. Add explicit `ActiveKVView` dataclass rather than treating recall as a slot state.
3. Add `CacheEntry.valid_for_roles` or role-specific stale/conflict scores.
4. Add `HeadRoleProfiler` that can import Pyramid's `best_labels.csv` as bootstrap labels.
5. Add a Causal-Forcing patch sketch targeting `CausalWanSelfAttention.forward`.
6. Add an MVP experiment mode:

```text
baseline: vanilla Causal/Self cache
+ recent/anchor only
+ compressed recall view
+ motion-head routing
+ stale filtering
```
