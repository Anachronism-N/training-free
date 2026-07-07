# Motion memory

Motion memory should not be a generic external feature. It should be a cache-maintenance rule for dynamic heads and recent temporal deltas.

## What to store

1. Dynamic-head KV entries
   - Identify heads whose attention patterns are local/temporal and sensitive to inter-frame continuity.
   - Store a small window of their KV as `motion` slots.

2. Latent delta summaries
   - `delta_z = z_t - z_{t-1}` within a chunk or across chunk boundaries.
   - Store pooled delta magnitude/direction as metadata and optional payload.

3. Motion scores
   - frame difference
   - optical-flow magnitude if available
   - latent-delta magnitude
   - attention temporal locality

## How to update

After each chunk:

```text
current KV from dynamic heads -> motion slots
latent/frame deltas -> motion metadata
if motion score decays, increase motion/recent weight for next chunk
if repetition score rises, reduce long-range static recall for motion heads
```

## How to classify heads

Start with an offline profiling pass over several prompts and chunks. For each layer/head, compute:

- `locality`: fraction of attention mass on adjacent/nearby frames.
- `sink_mass`: attention mass on first/anchor tokens.
- `periodicity`: whether attention peaks at regular temporal intervals.
- `motion_sensitivity`: output/attention change when recent frames are perturbed or masked.
- `mask_degradation`: generation metric drop when this head's history is masked.

Initial rules:

```text
motion/dynamic head: high locality + high mask_degradation on temporal continuity
anchor head: high sink_mass + broad long-range dependency
periodic/wave head: high periodicity
layout/static head: stable attention to scene/background tokens
```

Reference anchors:
- Pyramid Forcing: Anchor/Wave/Veil head roles.
- Forcing-KV: static vs dynamic heads; dynamic heads govern inter-frame motion and consistency.
- MemRoPE: dual long/short memory streams for identity and recent dynamics.
