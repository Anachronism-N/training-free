# LifeCache design changelog

This document records the design corrections made after discussion.

## Correction 1: task setting

Old wording:

```text
chunk-by-chunk long-video generation
```

Correct wording:

```text
AR / sliding-window long-horizon video generation under KV-cache budget
```

Self-Forcing and Causal-Forcing are rolling AR systems. A chunk/window is an implementation unit for cache update and clean refresh, not an independent generation unit.

## Correction 2: memory bank granularity

Old design:

```text
Each historical token or block has a heavy CacheEntry with StorageState, ValidityState, RuntimeRole, scene/entity/state metadata, and RoPE metadata.
```

Corrected v1 design:

```text
The memory bank stores selected token-level K/V TokenSets.
It does not store all historical tokens.
It does not create one Python object per token.
It does not use only block-level labels.
```

## Correction 3: StorageState / ValidityState / RuntimeRole

Old design:

```text
StorageState / ValidityState / RuntimeRole are core first-version abstractions.
```

Corrected v1 design:

```text
Use cache regions and selected recall views instead.
RecentCache, AnchorCache, CompressedBank, MotionCache, RecallView are enough for v1.
Validity and stale metadata are deferred to later versions.
```

## Correction 4: compression

Old design included stride as one of the main compression options.

Corrected v1 design:

```text
Stride is too weak for the main method and should only be a trivial sanity baseline.
Main compression strategies:
1. Attention Participation Top-k.
2. Head-group-aware Compression.
3. Video-understanding-inspired Key Token Compression.
```

## Correction 5: recall granularity

Old design risked chunk/block-level recall.

Corrected v1 design:

```text
Recall must be token-level.
Use two-stage retrieval:
1. TokenSet-level coarse retrieval.
2. Token-level Q-K fine retrieval.
```

## Correction 6: motion

Old design over-relied on latent/frame differences.

Corrected v1 design:

```text
Motion should be grounded in dynamic temporal K/V.
Use latent delta + dynamic-K change + boundary score.
```

## Correction 7: historical information usage

Old design used simple active-cache concatenation.

Corrected v1 design:

```text
Use head-specific active cache, region budget, and optional region bias.
Historical K/V is still accessed through self-attention, but different heads see different historical regions.
```

## Current v1 definition

```text
LifeCache-v1 =
  token-level compressed memory bank
  + token-level Q-K recall
  + fixed and dynamic anchors
  + motion-specific token cache
  + head-aware active-cache composition
  + region budget / optional region bias
```
