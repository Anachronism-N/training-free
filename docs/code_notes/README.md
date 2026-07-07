# Repository reading report

## Repositories inspected

- `Self-Forcing`: cloned and inspected.
- `Causal-Forcing`: cloned and inspected.
- `RollingForcing`: script URL was wrong; corrected to `TencentARC/RollingForcing`, cloned and inspected.
- `Pyramid-Forcing`: cloned and inspected.
- `Forcing-KV`: cloned; Windows checkout is partially blocked by invalid `:` filenames under `evaluation/`, but source was inspected through Git object reads.
- `MemRoPE`: cloned and inspected.
- `LongLive-RAG`: cloned and inspected.
- `Echo-Forcing`: verified via public README/GitHub and cloned from `mingqiangWu/Echo-Forcing`.

## Cross-repo conclusions

1. The minimal LifeCache patch point is `CausalWanSelfAttention.forward`, after Q/K/V projection and RoPE, before `attention(...)`.
2. Current Self/Causal K cache is post-RoPE, while V is unrotated. RoPE-safe long recall needs either pre-RoPE K or explicit temporal/spatial position maps.
3. `recall` should be a temporary active view from `compressed`, `anchor`, `recent`, and `motion` entries. It should not be a persistent tensor pool in v1.
4. Head-aware policy is mandatory. Forcing-KV and Pyramid-Forcing both show that static/anchor and dynamic/motion heads need different history.
5. Stale/invalid should be metadata that blocks retrieval/attention, not a tensor pool.
6. The safest v1 implementation stores full recent K/V, promotes selected sink/static heads to anchor, offloads evicted history as compressed entries, and composes active K/V per layer/head before attention.

