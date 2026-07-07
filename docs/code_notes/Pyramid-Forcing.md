# Pyramid-Forcing

## Reading notes

1. Autoregressive loop: follows Self-Forcing-style inference, with a public `configs/pyramid-forcing.yaml` controlling adaptive KV behavior.
2. KV lifecycle: the config describes per-head policies rather than one global cache policy.
3. Tensor layout: inherited Wan K/V layout `[B, tokens, heads, head_dim]`.
4. RoPE: config includes dynamic RoPE remapping (`window_clamp`) for stale/historical positions.
5. Hooks: reuse head labels from `configs/head_configs/best_labels.csv` when available.
6. Existing policy: three head classes: oscillatory heads use `sink1 + cyclic + recent4`; stable-positive heads use `sink3 + stride + recent4`; stable-negative heads use `sink3 + merge + recent4`.
7. Reuse for LifeCache: head-aware policy dispatch, cyclic/stride/merge compressed middle segments, dynamic RoPE as a RoPE-safe recall ablation.
8. Required changes: map Pyramid labels to LifeCache roles: oscillatory -> `motion/wave`, stable -> `anchor/layout`, merged middle -> `compressed`.

## LifeCache conclusion

Pyramid-Forcing supports the idea that lifecycle policies should be per-head, not only per-layer or global.

