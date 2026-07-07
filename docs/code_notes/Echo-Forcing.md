# Echo-Forcing

## Reading notes

1. Autoregressive loop: `pipeline/causal_inference.py` and `pipeline/causal_diffusion_inference.py`.
2. KV lifecycle: README describes hierarchical temporal memory with stable anchors, compressed history, recent windows, scene recall frames, and difference-aware decay.
3. Tensor layout: implementation inherits Wan/Self-Forcing K/V layout `[B, tokens, heads, head_dim]`.
4. RoPE: README claims relative RoPE for hierarchical memory. Implementation should be checked deeper before porting.
5. Hooks: interactive prompts and recall/hard-cut scripts make this a useful scenario benchmark.
6. Existing policy: preserve, recall, and forget scene memories.
7. Reuse for LifeCache: scene-level recall/forget semantics, interactive state changes, difference-aware stale marking.
8. Required changes: translate scene recall frames into KV entries tagged by `scene_id`, `entity_ids`, `state_version`, and `stale_score`.

## LifeCache conclusion

Echo-Forcing is the closest conceptual match. LifeCache should keep Echo’s preserve/recall/forget behavior but implement it as KV index/control, not an independent frame-memory generator.

