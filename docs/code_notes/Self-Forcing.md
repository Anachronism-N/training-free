# Self-Forcing

## Reading notes

1. Autoregressive loop: `pipeline/causal_inference.py`, `CausalInferencePipeline.inference`. It iterates chunks, runs denoising steps, writes `denoised_pred`, then reruns the generator with `context_noise` to refresh the clean KV cache.
2. KV lifecycle: `_initialize_kv_cache` creates one dict per transformer block. `wan/modules/causal_model.py::CausalWanSelfAttention.forward` inserts current post-RoPE K and raw V, then rolls the local window when full.
3. Tensor layout: self-attn `q/k/v` are `[B, L, 12, 128]`; cache tensors are `[B, cache_tokens, 12, 128]`. At 480x832, one latent frame is `1560` tokens.
4. RoPE: cache stores post-RoPE K (`causal_rope_apply(k, start_frame=current_start_frame)`) and unrotated V. Raw pre-RoPE K is available only inside the attention forward before cache insertion.
5. Hooks: per-layer hooks are easiest in `CausalWanSelfAttention.forward`; attention maps are not returned by flash attention, so profiling needs a debug SDPA path or saved reduced attention statistics.
6. Existing policy: sink + recent/local rolling window. No compressed, recall, or stale metadata.
7. Reuse for LifeCache: generation loop, clean-cache refresh pass, cache shape, sink/recent rolling logic, `current_start` frame indexing.
8. Required changes: replace raw `kv_cache["k"/"v"]` composition with `LifecycleKVCache.compose_active_cache(...)`; register new clean K/V segments after the context refresh pass; add pre/post-RoPE mode metadata.

## LifeCache conclusion

Self-Forcing is the best minimal patch target. The smallest implementation should intercept after `roped_key` and before `attention(...)`, because current cache semantics are already post-RoPE.

