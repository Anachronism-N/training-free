# MemRoPE

## Reading notes

1. Autoregressive loop: `pipeline/causal_inference.py`, similar to Self-Forcing.
2. KV lifecycle: cache adds `q`, `token_temporal_indices`, `token_spatial_indices`, and selected-token metadata, preparing for position-aware compression.
3. Tensor layout: K/V/Q cache tensors are `[B, cache_tokens, 12, 128]`.
4. RoPE: `wan/modules/causal_model.py` provides explicit temporal/spatial index RoPE and block-relative RoPE. This is the most relevant code for RoPE-safe historical recall.
5. Hooks: attention forward returns cache update info separately before applying updates, which is safer than mutating cache during each block.
6. Existing policy: sink + compressed + recent + new composition, with temporal/spatial indices for compressed tokens. It also explores EMA long/short memory.
7. Reuse for LifeCache: `position_map_ptr`, pre/post-RoPE distinction, and delayed cache update protocol.
8. Required changes: connect compressed token indices to `CacheEntry` metadata and expose retrieval-selected compressed entries as a temporary recall view.

## LifeCache conclusion

MemRoPE is the main reference for the question “should we store pre-RoPE K?” The answer for v1 can be post-RoPE compatibility, but RoPE-safe recall needs stored temporal/spatial maps or pre-RoPE K.

