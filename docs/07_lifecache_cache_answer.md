# LifeCache cache organization and memory-bank mapping

This document answers four implementation questions: how cache is organized, how each part is obtained and updated, how it maps to the memory bank, and how the cache content is used.

## 1. How is cache organized?

LifeCache should not use six independent token pools. It should use one KV storage system with typed entries.

Each `CacheEntry` points to a K/V payload and owns three kinds of state:

```python
storage_state: current | recent | anchor | compressed | motion
validity_state: valid | stale | invalid | dropped
runtime_role: normal | recall
```

The physical payload remains K/V tensors with the Wan-style layout:

```text
K/V: [B, tokens, heads, head_dim]
```

For the common Wan2.1-1.3B path:

```text
K/V: [B, tokens, 12, 128]
```

The memory bank is an index over these entries. It stores metadata and pointers; it should not act as an independent generation condition.

## 2. How is each part obtained and updated?

### Current

Obtained from the current self-attention forward:

```python
q, k, v = qkv_fn(x)
roped_q = rope(q)
roped_k = rope(k)
```

In LifeCache v1, current K is post-RoPE for compatibility with Self-Forcing and Causal-Forcing. Current V is raw V.

After the clean refresh pass, current entries are registered into the cache.

### Recent

Recent entries are the newest clean K/V chunks. They preserve local continuity and chunk-boundary coherence.

Update rule:

```text
current clean K/V -> recent
if recent budget is exceeded: evict oldest recent entries
```

Evicted recent entries become compressed candidates instead of being immediately discarded.

### Anchor

Anchor entries are high-trust long-range K/V anchors. They replace the idea of a permanently fixed first-block sink.

Promotion rule:

```text
recent or compressed entry -> anchor if trust is high, quality is high, stale is low, and access/usefulness is high
```

Anchor should be sparse and stable.

### Compressed

Compressed entries are evicted history stored compactly. They are not visible to attention by default.

Possible compression modes:

- stride or frame sampling
- top-k token selection
- patch-level dynamic selection
- merge or average
- later: EMA long/short memory tokens

Update rule:

```text
evicted recent -> compressed
compressed entry remains dormant until selected as recall
```

### Recall

Recall is not persistent storage. It is a temporary active view.

Update rule per chunk:

```text
retrieve valid compressed or anchor entries according to prompt, scene, entity, head role, motion need, and trust
mark selected entries as runtime_role = recall only for active-cache composition
```

After the chunk, entries return to their persistent storage state.

### Motion

Motion entries are short-lived dynamic K/V entries. They should be grounded in dynamic temporal heads, not only external optical-flow features.

Obtained from:

- dynamic temporal K/V heads
- recent latent delta
- frame-difference or motion score metadata

Update rule:

```text
current dynamic-head K/V -> motion
keep only a short window of motion entries
if motion score is too low, do not promote to motion
```

Motion entries mainly serve motion or wave heads.

### Stale and invalid

Stale and invalid are validity metadata, not physical token pools.

A stale entry is blocked or down-weighted when it conflicts with current scene, entity, state, quality, or RoPE validity.

## 3. How does cache correspond to memory bank?

The correspondence is one-to-one at the entry level.

```text
KV storage: actual K/V tensors
Memory bank / index: metadata and pointers for those tensors
```

A bank entry should contain:

```python
entry_id
kv_ptr
storage_state
validity_state
runtime_role
layer_id
head_id
chunk_id
token_start
token_end
head_role
scene_id
entity_ids
state_version
trust_score
motion_score
stale_score
conflict_score
rope_mode
rope_range
position_map_ptr
```

The bank answers selection questions:

- should this entry remain recent?
- should it be promoted to anchor?
- should it be compressed?
- should it be recalled now?
- should it be blocked as stale or invalid?
- which head roles can use it?

The bank does not directly generate video. It only controls which K/V payloads enter the active cache.

## 4. How is cache content used?

Before each attention call, LifeCache composes an active K/V view.

General layout:

```text
active K/V = anchor + recall + motion + recent + current
```

But actual composition is head-role-specific:

```text
layout or anchor heads:
  anchor + scene recall + recent + current

entity heads:
  anchor + entity recall + recent + current

motion or wave heads:
  motion + recent + tiny anchor + current

generic heads:
  anchor + recent + current
```

Then the model uses the active cache exactly like native attention:

```python
out = attention(roped_query, active_k, active_v)
```

Therefore LifeCache changes inference-time K/V composition, not model weights.

## 5. Minimal implementation sequence

1. Register clean current K/V after the clean refresh pass.
2. Maintain recent entries with a small rolling budget.
3. Promote high-trust entries to anchor.
4. Compress evicted recent entries.
5. Retrieve selected compressed entries as recall view.
6. Add motion entries for dynamic heads.
7. Add stale and invalid filters before active-cache composition.
8. Add RoPE-safe recall as a later advanced module.
