# Reference code reading plan

## Priority 1: Self-Forcing / Causal-Forcing

Read:
- AR generation loop
- past_key_values format
- attention forward implementation
- chunk-level latent/frame generation
- whether attention maps and KV tensors can be returned

Questions:
- Where is KV cache concatenated?
- How many layers/heads and token dimensions?
- Can we intercept per-layer/head K/V before or after RoPE?

## Priority 2: Rolling Sink / RollingForcing

Read:
- sink selection
- recent-window maintenance
- how sink+recent are passed to attention
- any recache / frame-selection logic

Goal:
- implement baseline: `anchor + recent`.

## Priority 3: Pyramid Forcing

Read:
- head profiling
- Anchor/Wave/Veil detection
- ragged-cache attention
- cache policy per head

Goal:
- borrow head-aware active-cache composition.

## Priority 4: Forcing-KV

Read:
- static/dynamic head split
- static pruning
- dynamic segment-wise pruning
- cache compression interface

Goal:
- implement compressed and motion slots.

## Priority 5: MemRoPE

Read:
- unrotated key storage
- online RoPE indexing
- long/short memory token streams

Goal:
- add RoPE-safe recall ablation.

## Priority 6: LongLive-RAG / Echo-Forcing

Read:
- retrieval query construction
- history latent store
- scene recall / decay
- compressed memory update

Goal:
- translate latent/scene retrieval into KV-entry retrieval.
