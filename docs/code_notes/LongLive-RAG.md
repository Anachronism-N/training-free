# LongLive-RAG

## Reading notes

1. Autoregressive loop: `pipeline/causal_inference.py`.
2. KV lifecycle: cache stores normal GPU K/V plus `cpu_k_frames` and `cpu_v_frames` for evicted memory.
3. Tensor layout: GPU cache is `[B, tokens, 12, 128]`; CPU memory is frame-level K/V lists.
4. RoPE: follows causal Wan cache semantics; recall is driven by latent descriptors, not KV-native metadata.
5. Hooks: memory selection happens once per chunk before denoising, then `memory_indices` are passed to the generator.
6. Existing policy: latent descriptors are built from generated frames by average pooling or a small latent AE; cosine similarity retrieves evicted frames while excluding the most recent evictions.
7. Reuse for LifeCache: query construction, CPU offload pattern, retrieval logs, latent descriptor as one metadata score for compressed entries.
8. Required changes: retrieve `CacheEntry` pointers rather than frame-only CPU lists; add stale/conflict filters before top-k similarity.

## LifeCache conclusion

LongLive-RAG confirms `recall` should be a temporary view over compressed/offloaded memory, not a new persistent tensor pool.

