# LifeCache-Forcing v2 specification

## 1. Goal

LifeCache-Forcing is a training-free inference-time KV-cache manager for long-horizon autoregressive video generation. It targets Self-Forcing first and Causal-Forcing second.

It does not train or modify the base generator weights. It changes only how historical K/V entries are stored, indexed, selected, and composed into the active attention cache.

## 2. Core principle

```text
Do not build an independent memory generator.
Build a lifecycle-aware KV cache.
```

The memory bank is the metadata/index/control plane. The actual generation effect must happen through active K/V tensors passed to self-attention.

## 3. Data model

Replace the old single `SlotState` with three orthogonal states.

```python
class StorageState(str, Enum):
    CURRENT = "current"
    RECENT = "recent"
    ANCHOR = "anchor"
    COMPRESSED = "compressed"
    MOTION = "motion"

class ValidityState(str, Enum):
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"
    DROPPED = "dropped"

class RuntimeRole(str, Enum):
    NORMAL = "normal"
    RECALL = "recall"
```

A cache entry should look like:

```python
@dataclass
class CacheEntry:
    entry_id: str
    storage_state: StorageState
    validity_state: ValidityState
    runtime_role: RuntimeRole

    layer_id: int
    head_id: int
    chunk_id: int
    token_start: int
    token_end: int
    kv_ptr: str | None

    head_role: HeadRole
    scene_id: str | None
    entity_ids: list[str]
    state_version: dict[str, str]

    trust_score: float
    motion_score: float
    stale_score: float
    conflict_score: float
    rope_risk_score: float
    access_count: int
    last_accessed_chunk: int

    rope_mode: str  # post_rope or pre_rope
    rope_range: tuple[int, int] | None
    position_map_ptr: str | None
```

## 4. Storage states

### 4.1 Current

Current entries are created inside the current attention forward.

For v1, current K is post-RoPE and V is raw V, matching Self-Forcing/Causal-Forcing. For v2 advanced mode, store pre-RoPE K and position metadata.

### 4.2 Recent

Recent entries are clean K/V chunks from the latest generation blocks.

Update:

```text
current clean K/V -> recent
if recent exceeds budget -> evict oldest recent
```

Evicted recent entries become compressed candidates.

### 4.3 Anchor

Anchor entries are high-trust long-range K/V. They generalize fixed sink tokens.

Promotion:

```text
entry is valid
trust high
stale low
usefulness/access high
quality high
```

Anchor should be sparse and persistent.

### 4.4 Compressed

Compressed entries are evicted historical K/V stored compactly.

They are invisible by default. They only enter attention through a recall view.

Compression modes:

1. `stride`: simple frame/token stride.
2. `topk`: select high-participation tokens.
3. `dynamic_temporal`: Forcing-KV-style dynamic temporal selection.
4. `ema`: MemRoPE-style long/short memory tokens.
5. `low_rank`: FreePCA-style low-rank global summary.

### 4.5 Motion

Motion entries are short-lived dynamic temporal K/V entries.

Source:

```text
dynamic temporal heads
recent latent delta
optional frame-difference/flow metadata
```

Motion entries serve motion/wave heads. They should not be globally injected into layout/entity heads.

## 5. Validity states

Validity controls whether an entry may enter active attention.

### valid

Default usable state.

### stale

The entry may encode outdated scene/entity/state information.

Examples:

- old cup K/V says intact; current state says broken;
- old room layout conflicts with a hard-cut new scene;
- old entity appearance conflicts with a later verified entity state.

### invalid

The entry comes from a low-quality, frozen, prompt-mismatched, or corrupt chunk. It should not be compressed, anchored, or recalled.

### dropped

Payload can be removed from storage. Metadata may remain for logging.

## 6. Runtime recall

Recall is not persistent storage.

A compressed entry can be selected as `runtime_role = RECALL` for the current chunk. After active-cache composition, it returns to `NORMAL`.

Recall scoring:

```text
score(entry, query, head_role) =
    semantic_or_scene_match
  + entity_match
  + trust_score
  + head_role_compatibility
  + motion_need * motion_score
  - stale_penalty
  - conflict_penalty
  - rope_risk_penalty
```

Compressed entries are not visible unless selected as recall.

## 7. Head-role active-cache composition

Use Pyramid-Forcing and Forcing-KV head labels as initial priors.

### Layout / anchor heads

```text
anchor + scene recall + recent + current
```

### Entity heads

```text
anchor + entity recall + recent + current
```

### Motion / wave heads

```text
motion + recent + tiny anchor + current
```

### Generic heads

```text
anchor + recent + current
```

## 8. Update algorithm

After generating a clean chunk:

```text
1. Register current clean K/V as recent.
2. Compute chunk metadata: scene, entities, state version, trust, motion score, RoPE range.
3. Promote high-trust entries to anchor.
4. Evict old recent entries if budget is exceeded.
5. Compress evicted recent entries.
6. Register dynamic-head entries as motion slots.
7. Update entity/scene/state table.
8. Mark conflicting entries stale.
9. Drop invalid or over-budget entries.
```

Before each attention call:

```text
1. Determine layer/head role.
2. Select base entries: recent, anchor, motion according to role.
3. Retrieve recall entries from compressed/anchor history according to query.
4. Filter stale/invalid/dropped entries.
5. Concatenate K/V payloads into active K/V.
6. Call attention(query, active_k, active_v).
```

## 9. First implementation target: Self-Forcing

Patch location:

```text
third_party/Self-Forcing/wan/modules/causal_model.py
```

Minimal patch logic:

```python
if lifecache is not None:
    active_k, active_v, entries = lifecache.compose_active_cache(
        layer_id=layer_idx,
        head_id=head_idx,
        query=lifecache_query,
        current_k=roped_key,
        current_v=v,
        native_cache=kv_cache,
    )
    x = attention(roped_query, active_k, active_v)
else:
    x = attention(roped_query, native_k, native_v)
```

For MVP, implement all-head shared composition first. Then add per-head composition.

## 10. Second implementation target: Causal-Forcing

Causal-Forcing has positive and negative CFG caches. Use identical lifecycle policies for both caches in the safe version.

Advanced ablation:

```text
positive cache: full LifeCache
negative cache: recent + anchor only
```

## 11. Evaluation plan

Baselines:

- Self-Forcing vanilla
- RollingForcing
- DeepForcing
- Pyramid-Forcing
- Forcing-KV
- LifeCache-v1

Ablations:

- no anchor
- fixed first-block anchor
- no compressed recall
- no motion slots
- no stale filtering
- no head-role routing
- post-RoPE recall vs RoPE-safe recall

Prompt suites:

- long motion continuation
- scene revisit
- entity recurrence
- state change and stale recall
- hard scene cut
- distractor entities

## 12. High-feasibility development order

1. Refactor cache states.
2. Implement recent + dynamic anchor.
3. Implement compressed-as-storage and recall-as-view.
4. Add Pyramid/Forcing-KV head-role prior.
5. Add motion slots from dynamic temporal heads.
6. Add stale/invalid metadata and hard filtering.
7. Add RoPE-risk score.
8. Add pre-RoPE/relative-RoPE advanced mode.
