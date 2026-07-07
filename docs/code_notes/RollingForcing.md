# RollingForcing

## Reading notes

1. Autoregressive loop: `pipeline/rolling_forcing_inference.py`. It builds overlapping rolling windows over chunks, denoises a window jointly, updates a noisy cache, then refreshes the clean cache.
2. KV lifecycle: upstream Wan causal model keeps a first-block attention sink and rolling local cache. Forcing-KV also includes a RollingForcing port under `pipeline/causal_inference_rollingforcing.py`.
3. Tensor layout: Wan K/V layout remains `[B, tokens, heads, head_dim]`.
4. RoPE: active K is post-RoPE; RollingForcing has special anchor handling where the first block may be stored unroped and re-roped for current relative position.
5. Hooks: hook at Wan causal self-attention; window-level denoising means cache update timing differs from Self-Forcing.
6. Existing policy: rolling denoising window plus long-term attention sink.
7. Reuse for LifeCache: rolling-window schedule is valuable as a stress test for motion continuity and stale invalidation during overlapping updates.
8. Required changes: LifeCache must distinguish `current` vs `clean_refresh` writes to avoid registering transient noisy-window KV as trusted memory.

## LifeCache conclusion

Treat RollingForcing as an evaluation baseline and later integration target, not the first patch target.

