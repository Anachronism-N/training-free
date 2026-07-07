# Compressed vs Recall

`compressed` and `recall` can overlap. In fact, they should overlap.

- `compressed`: where old history lives after eviction from recent/full cache.
- `recall`: a temporary active view assembled from compressed/anchor/history entries for the current chunk.

This matches the intuition from Echo-Forcing-style systems: scene recall frames/tokens can be stored inside a compressed historical memory and only become active when recalled.

## Recommended design

Do not implement `recall` as an independent persistent storage pool in v1. Implement it as:

```python
recall_entries = index.retrieve(query)
active_kv = gather(recall_entries.kv_ptr)
```

The entry may physically live in:

```text
compressed KV storage
anchor storage
recent storage
motion storage
```

but its current role in the active cache is `recall`.

## Lifecycle transition

```text
current -> recent -> {anchor | compressed | dropped}
compressed -> recall_view -> active attention for current chunk
compressed -> stale metadata if state conflict is detected
```

## Main research angle

The novelty is not that recall has a separate tensor store. The novelty is that compressed historical KV entries are indexed by lifecycle metadata:

- scene id
- entity id
- head role
- motion score
- trust score
- stale/conflict score
- RoPE status

so retrieval can be validity-aware rather than merely similarity-aware.
