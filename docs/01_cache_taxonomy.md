# Cache taxonomy: persistent storage vs active views

The first correction is that we should avoid treating the six names as six equally persistent token pools. A cleaner design is:

## Persistent KV storage states

1. `recent`: full KV from the last R chunks. Used by nearly all heads.
2. `anchor`: small high-trust long-range KV anchors/sinks. Used mainly by anchor/layout/identity heads.
3. `compressed`: evicted historical KV stored as compressed payloads and metadata. Not always active.
4. `motion`: recent dynamic-head KV and/or latent-delta summaries. Used mainly by motion/dynamic heads.
5. `invalid/stale`: not a payload pool; a metadata state marking entries that should be blocked or down-weighted.

## Temporary active views

6. `recall`: a temporary view constructed before each chunk by selecting entries from `compressed`, `anchor`, and sometimes `recent`/`motion`. It should not necessarily be a separate persistent pool.

Therefore the actual minimal design is closer to **4 persistent payload pools + 1 invalid metadata state + 1 recall view**.

## Default active-cache order

For each layer/head, construct the active context in the following logical order:

```text
[ANCHOR] + [RECALL_FROM_COMPRESSED] + [MOTION] + [RECENT] + [CURRENT]
```

But different head types should receive different subsets:

```text
Anchor/layout heads:   anchor + scene/layout recall + recent + current
Identity/entity heads: anchor + entity recall + recent + current
Motion/dynamic heads:  motion + recent + tiny anchor + current
Generic heads:         anchor + recent + current
```

## Why not just six independent pools?

Because `recall` overlaps with `compressed`: compressed history is the storage form, recall is the current-step active selection. Similarly `stale` is not a token type; it is a validity state attached to an indexed KV entry.
