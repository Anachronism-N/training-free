# LifeCache: Why No Gain — Final Analysis

> 2026-07-12 | Complete analysis of why LifeCache shows no improvement on Self-Forcing

## Executive Summary

After extensive debugging and controlled experiments, LifeCache's recall path is fully functional (bank grows, candidates exist, recalled_tokens > 0, QK ratio 3.5x), but all variants show no improvement over native Self-Forcing. The root cause is not a code bug — it is a **fundamental mechanism mismatch**.

## Key Insight: Recall-After-Loss ≠ Smart-Retention

### What LifeCache Does (Our Approach)

```
frame 0-20 generated → KV cache has 21 frames
frame 21 generated → frame 0 evicted from cache
  ↓
LifeCache: compress evicted frame 0 K/V → store in bank
  ↓
frame 118 generated → query attends to recent 21 frames
  ↓
LifeCache: recall compressed frame 0-96 tokens from bank → inject into attention
```

This is **recall-after-loss**: tokens are evicted first, then later recalled from a compressed bank.

### What Pyramid-Forcing Does (The Successful Approach)

```
Per-head classification: Anchor / Wave / Veil
  ↓
Each head keeps different K/V subsets:
  - Anchor heads: stride-sample far history (never evict key frames)
  - Wave heads: periodic sampling of middle frames
  - Veil heads: merge nearby frames
  ↓
Dynamic RoPE: remap kept tokens' positions into [0, 21] training range
  ↓
Result: tokens are NEVER lost for important heads
```

This is **smart-retention**: important tokens are kept in the cache from the start, never evicted.

### Why The Difference Matters

| Aspect | LifeCache (recall-after-loss) | Pyramid-Forcing (smart-retention) |
|---|---|---|
| Token freshness | Evicted, compressed, then recalled — lossy | Kept in cache, never degraded |
| RoPE position | Evicted at position 0-7, recalled at position 118 — remap needed | Kept with dynamic RoPE in training range |
| Token context | Sparse 512 tokens per layer, not forming complete frames | Full frame context preserved |
| Cache budget | Same as native (21 frames) | Same budget but smarter allocation |
| Head awareness | Layer-level routing | Per-head heterogeneous cache |

## Experiment Evidence

### Complete Ablation Matrix (A-B-A Scene Revisit Prompts)

| Experiment | Dir | p00 | p01 | p02 | vs Native |
|---|---|---|---|---|---|
| Native SF | `sf_native_aba_120f/` | 3.3M | 7.3M | 3.3M | — |
| QK proxy recall | `sf_lifecache_aba_120f/` | 3.4M | 6.9M | 3.0M | ~same |
| Timestep-filtered | `sf_lifecache_aba_clean_120f/` | 3.3M | 7.3M | 3.3M | ~same |
| Random recall | `sf_lifecache_aba_random_120f/` | 3.3M | 7.3M | 3.3M | ~same |

### QK Score Analysis

| Metric | Value |
|---|---|
| QK recall/recent ratio | 3.5-4.0x |
| Layer 27 ratio | 2.89x |
| Layer 28 ratio | 3.75x |
| Layer 29 ratio | 4.05x |

Recalled tokens attract 3-4x more attention than recent tokens — attention IS using them.

### Hypotheses Verification Summary

| # | Hypothesis | Test | Result |
|---|---|---|---|
| E | Attention ignores recall | QK ratio 3.5x | Excluded |
| C | Recalled tokens redundant | max_frame_distance + A-B-A | Excluded |
| H | Budget too large | 32→256 sweep | Excluded |
| F | Motion heads polluted | layer 29 only | Excluded |
| B | Noisy denoising memory | timestep filter | Excluded |
| D | Token selection quality | random recall | Excluded |
| I | SF bottleneck ≠ K/V loss | All experiments no gain | **Confirmed** |

## Why CF and SF Are Not Fundamentally Different

Causal-Forcing (CF) and Self-Forcing (SF) share the same underlying architecture:
- Same Wan2.1-T2V-1.3B backbone
- Same sliding-window KV cache mechanism
- Same training-free long-video generation paradigm
- CF uses framewise autoregression (1 frame per block), SF uses chunkwise (3 frames per block)
- Both have the same RoPE extrapolation problem at long horizons

Most training-free methods (Pyramid-Forcing, Forcing-KV, Echo-Forcing, MemRoPE) are evaluated on both SF and CF with similar results. The mechanism that works on SF generally works on CF.

## Why Other Training-Free Methods Work

### Pyramid-Forcing: Per-Head Smart Retention
- **Mechanism**: Classify 360 heads into Anchor/Wave/Veil groups, assign different cache policies per head
- **Key innovation**: Dynamic RoPE remap — keeps old tokens in cache with remapped positions
- **Why it works**: Prevents K/V loss for important heads, never needs to "recall"

### Forcing-KV: Head-Aware KV Cache
- **Mechanism**: Static/spatial heads keep layout memory, dynamic/temporal heads keep motion memory
- **Key innovation**: Per-head cache budgets based on head function
- **Why it works**: Allocates cache budget to heads that need history most

### Echo-Forcing: Scene Memory with VLM
- **Mechanism**: VLM-based scene recognition + preserve/recall/forget operations
- **Key innovation**: Semantic scene descriptors for content-aware memory management
- **Why it works**: Uses external knowledge (VLM) to decide what to keep

### MemRoPE: Positional Encoding Extension
- **Mechanism**: Extends RoPE to handle longer sequences
- **Key innovation**: Modified positional encoding for extrapolation
- **Why it works**: Addresses the RoPE extrapolation problem directly

### Common Thread

All successful methods share one property: **they prevent K/V loss in the first place**, either by:
1. Keeping more tokens in cache (smarter allocation)
2. Remapping RoPE positions (positional safety)
3. Using external knowledge (VLM, head profiling)

**None of them do "evict-then-recall"** — because once tokens leave the cache, their positional context and temporal structure are fundamentally degraded.

## What LifeCache Could Be Useful For

LifeCache's recall-after-loss approach may be valuable in scenarios where:
1. **Extremely long videos** (5+ minutes) where even smart retention exhausts cache budget
2. **Scene revisit with long gaps** (A → B → C → ... → A, 100+ frames apart)
3. **Models with stronger content-addressing** (where QK similarity truly reflects content similarity)
4. **Causal-Forcing framewise mode** (1 frame per block means more frequent eviction)

## Recommendations

### Short-term
1. Accept that LifeCache on SF is a negative result — scientifically valuable
2. Document all findings clearly for future reference

### Medium-term
3. Test LifeCache on Causal-Forcing framewise mode (more frequent eviction = more recall opportunities)
4. Add semantic descriptors (z vectors) for content-aware retrieval
5. Implement write_or_merge for deduplication

### Long-term
6. Consider hybrid approach: smart-retention (like PF) for recent history + recall-only for very old history
7. Test on stronger benchmarks with explicit scene revisit requirements
8. Explore RollingForcing integration (native streaming mode)

## References

- `docs/18_lifecache_v1_issues.md` — v1 issues
- `docs/19_lifecache_quality_analysis.md` — quality analysis
- `docs/20_lifecache_v2_code_level_design.md` — v2 design
- `docs/21_lifecache_v2_root_cause.md` — eviction timing bug
- `docs/23_lifecache_near_recall_no_gain_analysis.md` — near-only recall analysis
- `docs/24_lifecache_v2_optimized_no_gain_analysis.md` — v2 optimized analysis
- `docs/25_lifecache_after_headgroup_fix_next_steps.md` — next steps roadmap
- `docs/26_lifecache_rope_and_other_failure_hypotheses.md` — RoPE and failure hypotheses
- `0623/06_实验记录与发现_LOG.md` — 0623 experiment log
