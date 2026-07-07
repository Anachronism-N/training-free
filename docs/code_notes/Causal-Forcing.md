# Causal-Forcing

## Reading notes

1. Autoregressive loop: `pipeline/causal_inference.py`. It is Self-Forcing-compatible but adds first-chunk schedule/timing options.
2. KV lifecycle: `wan/modules/causal_model.py` mirrors Self-Forcing: cache dict per block, post-RoPE K, raw V, `global_end_index`, `local_end_index`.
3. Tensor layout: `[B, tokens, 12, 128]` for K/V cache in the public 1.3B path.
4. RoPE: post-RoPE K at the active cache point. Pre-RoPE K can be captured only inside `qkv_fn`.
5. Hooks: same as Self-Forcing. Add optional return of reduced attention stats before using full attention maps.
6. Existing policy: local-attention rolling cache with optional sink preservation.
7. Reuse for LifeCache: same as Self-Forcing, with a slightly cleaner inference loop for timing ablations.
8. Required changes: identical attention-forward patch; pass `lifecycle_cache`, `layer_id`, and chunk/prompt metadata through `_forward_inference`.

## LifeCache conclusion

Use Causal-Forcing as the second integration target after Self-Forcing, mainly to test whether lifecycle cache policies survive schedule variations.

