# LifeCache-v1 Integration Issues & Analysis

> 2026-07-08 | Analysis of why current LifeCache-v1 underperforms native Self-Forcing

## Current Architecture

### Data Flow

```
Per generation block:
  1. Attention forward: evict old tokens → capture to _lifecache_evicted
  2. Pipeline: read _lifecache_evicted → on_block_complete()
     → compress_attention_participation (AP-topk to 512 tokens)
     → store in TokenSetBank (COMPRESSED region)
     → every 4 chunks: promote top compressed → ANCHOR
  3. Each attention call:
     → compose_active_cache() → recall_tokens() → select top sets + top tokens
     → return (extra_k, extra_v)
     → get_active_kv_for_attention() → torch.cat([extra_k, recent_k])
     → attention(roped_query, active_k, active_v)
```

## Identified Issues

### Issue 1 (CRITICAL): RoPE Position Mismatch

**Severity: Fatal**

Evicted tokens are captured from `kv_cache["k"]` which is already RoPE-rotated at their absolute frame positions (e.g., position 0-7). When later recalled and attended to by a query at position ~118, the query-key angle difference is `freqs * (118 - 0)`, far exceeding the training domain (21 frames). This causes abnormal attention scores.

This is the same "absolute position extrapolation" problem identified in 0623 experiments. Pyramid-Forcing solves this with `dynamic_rope` (remapping key positions back to [0, 21]).

**Fix needed**: Add RoPE remap — re-rope recalled K/V tokens to a position adjacent to the current query window.

### Issue 2 (HIGH): All Heads Share Same Tokens

**Severity: High**

`get_active_kv_for_attention()` calls `compose_active_cache()` for each head (0..11), but TokenSets are stored as all-head (12 heads in one tensor). All 12 heads do redundant recall computations, and the first head's result is expanded to all heads.

This means:
- WAVE/MOTION heads are forced to attend to layout/anchor tokens
- No head-aware routing
- 12x redundant cosine similarity computation

**Fix needed**: Store tokens per-head or per-head-group, and route different heads to different token subsets.

### Issue 3 (HIGH): Uniform Attention Proxy for Compression

**Severity: High**

Compression uses `attn = torch.ones(...)` (uniform proxy) instead of real attention maps. AP-topk with uniform attention is equivalent to random sampling — compressed tokens carry no meaningful information.

**Fix needed**: Capture real attention maps from the attention forward pass and use them for AP-topk compression.

### Issue 4 (MEDIUM): Eviction Capture Timing

**Severity: Medium**

`_lifecache_evicted` is captured in the attention forward but only processed in the pipeline after the clean context refresh (Step 3.3). During the spatial denoising loop (Step 3.1), multiple denoising steps each trigger eviction, but only the last step's evicted tokens survive (subsequent steps overwrite `_lifecache_evicted`).

**Fix needed**: Accumulate evicted tokens across all denoising steps, or only capture during the clean context pass.

### Issue 5 (MEDIUM): Redundant Per-Head Recall Computation

**Severity: Medium (Performance)**

Each of 12 heads independently runs `compose_active_cache()` → `recall_tokens()` → `token_qk_scores()` with full cosine similarity over all stored token sets. This is extremely slow (~250s per 120-frame prompt vs ~120s native).

**Fix needed**: Run recall once per layer, then route tokens to heads based on head roles.

## Experiment Alignment Status

### Config Alignment ✅
- Model: Self-Forcing DMD checkpoint (`self_forcing_dmd.pt`)
- Config: `self_forcing_dmd.yaml` (`local_attn_size=21`, `sink_size=0`)
- Frames: 120 latent frames (~30s)
- Seed: 0
- Prompts: 3 review prompts from 0623 experiments

### Baseline Comparison

| Method | p00 (woman) | p01 (parkour) | p02 (cafe) | Time |
|---|---|---|---|---|
| Native SF (ours) | 7.0M | 5.9M | 9.8M | 6m |
| Native SF (official) | 7.0M | 6.5M | 12M | - |
| SF + Pyramid (ours) | 8.5M | 4.9M | 7.2M | 6m |
| SF + Pyramid (official) | 8.2M | 4.9M | 7.5M | - |
| SF + LifeCache-v1 | 7.3M | 6.6M | 11M | 14m |

Baseline results are consistent with official outputs (file sizes within expected range).

## References

- 0623 experiments: `/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/0623/`
- Official baselines: `0623/assets/long_extrapolation_30s/official/`
- Experiment log: `0623/06_实验记录与发现_LOG.md` (Round 41: absolute position extrapolation confirmed)
- PF dynamic RoPE: `Pyramid-Forcing/pyramidkv/rope.py:map_dynamic_pos_time`
