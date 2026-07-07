# Forcing-KV

## Reading notes

1. Autoregressive loop: `pipeline/causal_inference_forcingkv.py` follows the Self-Forcing loop and refreshes clean KV after each chunk.
2. KV lifecycle: `wan/modules/causal_model_forcingkv.py` splits head groups after a cache switch. It composes static/spatial heads separately from dynamic/temporal heads.
3. Tensor layout: still `[B, tokens, 12, 128]`, but grouped head tensors are extracted by `extract_heads_triton` and attended separately.
4. RoPE: uses Triton RoPE over concatenated Q/K, then stores post-RoPE K and raw V.
5. Hooks: `configs_head/head_profile.py` profiles frame-wise attention maps; JSON files list `static_head` and `dynamic_head` for each layer.
6. Existing policy: static heads use sink + spatial/local cache; dynamic heads use sink + dynamic temporal cache + temporal cache + current tokens. Dynamic compression selects low-similarity patch chunks using cosine similarity between adjacent K chunks.
7. Reuse for LifeCache: offline head classification, grouped head attention, dynamic compression, per-head static/dynamic policy files.
8. Required changes: rename static/dynamic groups into LifeCache roles; persist dynamic/compressed chunks with metadata instead of anonymous grouped tensors.

## Checkout note

On Windows, full checkout is blocked by result files with `:` in their names under `evaluation/`. Source was still readable through `git show HEAD:path`.

