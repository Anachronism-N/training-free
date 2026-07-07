# LifeCache mechanism map

This document maps LifeCache components to the papers/codebases that motivate them. The goal is feasibility: every major design choice should have a concrete third-party implementation reference.

## LifeCache overview

LifeCache-Forcing upgrades the native AR KV cache into a typed lifecycle cache. The base video generator remains frozen. The memory bank is not an independent generation module; it is the control plane for K/V entries.

Core statement:

```text
K/V tensors are the payload.
CacheEntry metadata is the memory bank.
Active-cache composition is how memory affects generation.
```

## Component-to-reference map

| LifeCache component | What it does | Main code references | Why these references matter |
|---|---|---|---|
| Recent cache | Keeps newest clean K/V chunks for local continuity | Self-Forcing, Causal-Forcing | Native AR cache and clean-context refresh already exist |
| Anchor cache | Maintains long-range stable sinks/anchors | RollingForcing, DeepForcing, Pyramid-Forcing, MemRoPE | Existing code proves anchor/sink K/V can be concatenated with local cache |
| Compressed cache | Stores evicted history compactly | DeepForcing, Forcing-KV, LongLive-RAG, Echo-Forcing, FreePCA | Existing code covers compression, offloading, token selection, and low-rank ideas |
| Recall view | Temporarily selects compressed/offloaded history into active attention | LongLive-RAG, Echo-Forcing, SWIFT | Recall should be runtime visibility, not a persistent tensor pool |
| Motion slots | Preserves dynamic temporal K/V and motion continuity | Forcing-KV, MotionCache, FlowCache, DeepForcing | Dynamic temporal heads/cache are the most feasible motion memory implementation |
| Head-role routing | Gives different heads different active K/V mixtures | Pyramid-Forcing, Forcing-KV | Per-head cache policy is already implemented and should be reused |
| Scene/entity metadata | Tags K/V entries by scene, entity, state version | IAMFlow, Echo-Forcing, LongLive-RAG, SWIFT | Semantic memory should become CacheEntry metadata rather than external generation condition |
| Stale/invalid metadata | Blocks or decays conflicting K/V before recall | Echo-Forcing, IAMFlow | Preserve/recall/forget and VLM/LLM validation motivate validity gates |
| RoPE-safe recall | Makes recalled historical K/V positionally valid | MemRoPE, infinity-rope, DiT-Extrapolation, FreeLOC, MIGA | Long-horizon recall can fail if old K carries invalid RoPE phase |
| Sparse/budgeted attention | Controls inference cost after recall | LongVideoSparseAttention, Forcing-KV | Active cache can still grow; sparse attention/budgeting is needed |

## Detailed design references

### 1. Recent cache

References:

- Self-Forcing
- Causal-Forcing

Design decision:

```text
current clean K/V -> recent
recent keeps the last R clean chunks
recent is always visible to most heads
```

Reasoning:

Recent cache is responsible for chunk-boundary continuity, short-term texture continuity, and local motion. It should be updated only after the clean/context refresh pass. Do not register noisy denoising-step K/V as long-term memory.

### 2. Anchor cache

References:

- RollingForcing: fixed first-block sink plus rolling local cache
- DeepForcing: deep sink variants
- Pyramid-Forcing: head-aware sink/middle/recent policy
- MemRoPE: sink plus long/short memory plus local window

Design decision:

```text
anchor is a high-trust long-range K/V state, not necessarily the first chunk.
```

Update policy:

```text
recent/compressed entry -> anchor if trust high, quality high, stale low, and repeatedly useful
```

### 3. Compressed cache

References:

- DeepForcing: participative compression direction
- Forcing-KV: static/dynamic compression over grouped heads
- LongLive-RAG: offloaded history with retrieval
- Echo-Forcing: compressed history and scene recall
- FreePCA: low-rank/PCA decomposition for long/short consistency

Design decision:

```text
evicted recent -> compressed
compressed is dormant by default
```

Compression modes should be staged:

1. v1: stride or top-k token selection.
2. v2: Forcing-KV-style dynamic temporal selection.
3. v3: MemRoPE-style EMA long/short memory.
4. v4: FreePCA-style low-rank summary for global consistency.

### 4. Recall view

References:

- LongLive-RAG
- Echo-Forcing
- SWIFT

Design decision:

```text
recall is a runtime role, not a storage pool.
```

A compressed entry can become visible as recall for the current chunk. After the chunk, it returns to its persistent storage state.

Recall score:

```text
score = semantic/scene/entity relevance
      + trust
      + head-role compatibility
      + motion need if applicable
      - stale/conflict penalty
      - RoPE risk penalty
```

### 5. Motion slots

References:

- Forcing-KV
- MotionCache
- FlowCache
- DeepForcing

Design decision:

```text
motion memory should be dynamic temporal K/V first, not external flow first.
```

Motion slots contain:

- dynamic-head K/V spans;
- chunk-level latent delta metadata;
- optional frame-difference or flow summary;
- short-lived motion score.

Use:

```text
motion/wave heads: motion + recent + tiny anchor + current
layout/entity heads: do not consume motion slots by default
```

### 6. Head-role routing

References:

- Pyramid-Forcing
- Forcing-KV

Design decision:

Use existing head labels as priors before building a new profiler.

Initial policy:

```text
layout/anchor heads: anchor + scene recall + recent + current
entity heads:        anchor + entity recall + recent + current
motion/wave heads:   motion + recent + tiny anchor + current
generic heads:       anchor + recent + current
```

### 7. Scene/entity/state metadata

References:

- IAMFlow
- Echo-Forcing
- LongLive-RAG
- SWIFT

Design decision:

Scene/entity/state should be metadata attached to K/V entries:

```python
scene_id
entity_ids
state_version
trust_score
stale_score
conflict_score
```

This keeps LifeCache grounded in K/V cache management while still benefiting from semantic memory.

### 8. Stale and invalid metadata

References:

- Echo-Forcing
- IAMFlow

Design decision:

Stale and invalid are validity states, not storage states.

Examples:

```text
cup intact -> cup broken: old cup K/V becomes stale for entity recall
kitchen -> beach hard cut: old kitchen layout K/V has scene conflict unless recall cue exists
low-quality frozen chunk: K/V is invalid and cannot become anchor or compressed memory
```

### 9. RoPE-safe recall

References:

- MemRoPE
- infinity-rope
- DiT-Extrapolation
- FreeLOC
- MIGA

Design decision:

LifeCache v1 can use post-RoPE K for compatibility. LifeCache v2 should support:

```text
pre-RoPE K storage
position map metadata
temporal/spatial relative reindexing
RoPE-risk penalty for post-RoPE far-history recall
```

### 10. Sparse and budgeted attention

References:

- LongVideoSparseAttention
- Forcing-KV

Design decision:

Even after recall, the active cache must stay bounded. Add per-head budget controls:

```text
max_anchor_tokens
max_recall_tokens
max_motion_tokens
max_recent_tokens
```

Sparse/block attention can later replace naive concat if active views become too large.
