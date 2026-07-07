# Refined LifeCache-Forcing Design

## 1. Core correction

The memory bank should not be an independent external memory module.

Correct formulation:

```text
KV payload storage + metadata index + per-head active readout policy
```

The lifecycle happens inside the KV cache system. The bank/index stores only metadata, pointers, scores, and validity states needed to update and compose active KV views.

## 2. Final cache state taxonomy

Use four persistent storage states, one metadata state, and one temporary view.

### Persistent storage states

1. `RECENT`
   - Full or near-full KV of the most recent chunks.
   - Purpose: local continuity, chunk boundary coherence, short-range motion.

2. `ANCHOR`
   - Stable high-trust sink/anchor entries.
   - Purpose: identity, layout, global style, stable reference.

3. `COMPRESSED`
   - Evicted history stored compactly.
   - Purpose: searchable long history under bounded memory.
   - Can be raw selected KV, EMA memory, patch-level compressed KV, quantized KV, or latent-linked KV.

4. `MOTION`
   - Motion-specific dynamic/temporal-head KV and optional latent-delta metadata.
   - Purpose: prevent motion slowdown, repetition, and boundary discontinuity.

### Metadata states

5. `STALE` / `INVALID`
   - Not a token pool.
   - A validity label or score on existing entries.
   - Used to block or down-weight entries during active readout.

### Temporary active view

6. `RECALL_VIEW`
   - Not persistent storage.
   - A per-step selected view from `COMPRESSED`, `ANCHOR`, and other stored entries.
   - It enters attention only for the current chunk.

## 3. Storage state vs runtime role

A cache entry has two different concepts:

```python
entry.slot_state      # where/how it is stored: RECENT, ANCHOR, COMPRESSED, MOTION
entry.runtime_role    # why it is currently used: recall, local, sink, motion, current
```

`recall` should be implemented as runtime role / active view, not as a separate storage pool.

Example:

```text
entry.slot_state = COMPRESSED
entry.scene_id = kitchen_01
entry.entity_ids = [woman_01, cup_01]

Current prompt: "the woman returns to the same kitchen"
=> entry selected into RECALL_VIEW for this chunk
=> physical storage remains COMPRESSED
```

## 4. Active KV composition

The active attention input should be composed per head.

General order:

```text
[ANCHOR] + [RECALL_VIEW] + [MOTION] + [RECENT] + [CURRENT]
```

But each head receives a different subset.

### Anchor / layout heads

```text
anchor + scene/layout recall + recent + current
```

### Entity / identity heads

```text
anchor + entity recall + recent + current
```

### Motion / temporal heads

```text
motion + temporal recent + tiny anchor + current
```

### Wave / oscillating heads

```text
phase/periodic recall + recent + current
```

### Unknown heads

```text
anchor + recent + current
```

## 5. Motion module refinement

Motion should be implemented at three levels.

### Level 1: latent/frame motion metadata

After each generated chunk:

```python
latent_delta = z[:, 1:] - z[:, :-1]
motion_score = latent_delta.abs().mean()
```

Also compute simple frame difference after VAE decode if available.

### Level 2: temporal/dynamic-head KV

Use head classification from Pyramid/Forcing-KV as bootstrap. Dynamic/temporal heads write their recent KV into `MOTION` storage.

MVP active view:

```text
motion heads: motion + temporal recent + current
```

### Level 3: adaptive compressed motion memory

Optional advanced version inspired by MemRoPE:

```text
short-term EMA = recent dynamics
long-term EMA = stable global content
adaptive alpha = function of token motion magnitude
```

## 6. Head role initialization

Do not start with fully learned/classified roles. Use available references:

1. Import Pyramid `best_labels.csv`.
2. Map labels:
   - `-1`: wave / oscillating / possible motion
   - `1`: anchor/layout / stable compact
   - `2`: sparse semantic/layout recall
3. Add Forcing-KV style spatial/temporal split if the implementation supports head groups.
4. Later run offline profiling to refine labels.

Suggested internal enum:

```python
class HeadRole(str, Enum):
    UNKNOWN = "unknown"
    ANCHOR = "anchor"
    LAYOUT = "layout"
    ENTITY = "entity"
    MOTION = "motion"
    WAVE = "wave"
    SPARSE_RECALL = "sparse_recall"
```

## 7. Stale / invalid logic

Stale/invalid should be role-specific when possible.

Example:

```text
old cache entry:
  scene = kitchen
  entity = cup_01
  state_version[cup_01] = intact_v1

new state table:
  cup_01 = broken_v2

Then:
  stale_for.entity = 1.0
  stale_for.layout = 0.1
  stale_for.motion = 0.8
```

This means:

- It should not be used as entity recall for `cup_01`.
- Its kitchen background/layout may still be partially useful.
- Its motion is likely obsolete.

## 8. Scoring function

Use a two-stage retrieval.

### Stage A: candidate filtering

Filter by:

```text
layer/head compatibility
slot_state in allowed states
not invalid
token budget
```

### Stage B: score candidates

```python
score = (
    w_trust * trust_score
    + w_scene * scene_match
    + w_entity * entity_match
    + w_motion * motion_need * motion_score
    + w_access * access_count_score
    - w_stale * stale_for_role
    - w_conflict * conflict_for_role
    - w_rope * rope_risk
)
```

## 9. Minimal integration with Causal/Self-Forcing

The cleanest first patch point is inside `CausalWanSelfAttention.forward`.

Current pattern:

```python
q, k, v = qkv_fn(x)
roped_query = causal_rope_apply(q, ...)
roped_key = causal_rope_apply(k, ...)
# update kv_cache
x = attention(roped_query, active_k, active_v)
```

LifeCache insertion:

```python
q, k, v = qkv_fn(x)
roped_query = ...
roped_key = ...

if isinstance(kv_cache, LifecycleKVCache):
    kv_cache.register_current(layer_id, roped_key, v, current_start, metadata)
    active_k, active_v, entries = kv_cache.compose_active_view(
        layer_id=layer_id,
        query={...},
        current_k=roped_key,
        current_v=v,
    )
    x = attention(roped_query, active_k, active_v)
else:
    # original path
```

For MVP, use post-RoPE K. For stronger version, store pre-RoPE K and apply dynamic RoPE at readout.

## 10. Implementation roadmap

### MVP-0: Read-only instrumentation

- Log K/V shapes.
- Log layer/head cache lengths.
- Log generated chunk boundaries.
- Compute latent/frame motion scores.

### MVP-1: LifeCache active view with no semantic metadata

- Support `RECENT`, `ANCHOR`, `COMPRESSED`, `MOTION`.
- Use Pyramid/Forcing-KV head labels.
- Add per-head active view composition.

### MVP-2: compressed-as-storage and recall-as-view

- Evicted recent entries become compressed entries.
- Top-k compressed entries can be recalled by prompt/scene/entity placeholders.
- Keep recall as temporary view.

### MVP-3: motion-aware routing

- Dynamic heads use `MOTION + RECENT + CURRENT`.
- Layout heads use `ANCHOR + RECALL + RECENT + CURRENT`.
- Compare with all-head same cache.

### MVP-4: stale-aware filtering

- Add simple prompt/state parser.
- Add state versions.
- Block or decay stale entries during retrieval.

### MVP-5: RoPE-safe readout

- Store pre-RoPE K.
- Apply dynamic/block-relative RoPE when composing active view.

## 11. Recommended first ablations

1. Vanilla Causal/Self-Forcing.
2. Recent-only.
3. Anchor + recent.
4. Anchor + recent + compressed recall.
5. Anchor + recent + compressed recall + motion routing.
6. Full LifeCache with stale filtering.
7. Post-RoPE recall vs pre-RoPE dynamic readout.

## 12. Main paper claim after refinement

LifeCache-Forcing is not an external memory bank. It is a lifecycle-aware active KV view composer that turns AR video cache management from a temporal FIFO policy into a typed, head-aware, motion-aware, and validity-aware cache system.
