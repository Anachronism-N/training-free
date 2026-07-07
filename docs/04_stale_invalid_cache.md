# Stale / Invalid KV entries

`stale` or `invalid` should be metadata, not a separate active KV token type.

## When to mark stale

1. State transition conflict
   - Old entry says `cup_01=intact`, current state table says `cup_01=broken`.

2. Scene conflict
   - Old kitchen layout is retrieved during a hard-cut beach scene with no return cue.

3. Entity conflict
   - Retrieved entity id conflicts with current entity state, appearance version, or disambiguation.

4. Quality/trust failure
   - A generated chunk is low quality, motion-frozen, or prompt-inconsistent and should not be promoted.

5. RoPE/position invalidity
   - Post-RoPE K from a far historical position is not safe to reuse without reindexing or decay.

## How to use stale metadata

Do not pass stale entries into active attention unless intentionally testing ablations.

Retrieval score:

```text
score = relevance + trust + future_need + motion_need - stale_penalty - conflict_penalty
```

Hard filtering:

```python
if entry.stale_score > tau_stale or entry.conflict_score > tau_conflict:
    skip(entry)
```

Soft decay:

```python
entry_weight = sigmoid(score) * (1 - stale_score)
```

## Important distinction

Stale does not always mean delete. It may still be useful as a historical fact for evaluation or state reasoning, but it should not be used as active generation memory unless explicitly requested.
