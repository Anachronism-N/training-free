# Codex prompt for detailed reference-code reading

You are helping me build a research prototype for **training-free long-horizon video generation** on top of **Self-Forcing** and **Causal-Forcing**.

The working method is **LifeCache-Forcing: Lifecycle-aware KV Cache Management**.

Important conceptual constraint:
- Do **not** treat the memory bank as an independent external memory module that directly generates video.
- The memory bank is the **index/control plane** of the KV cache: it stores metadata, scores, pointers, compression state, head roles, scene/entity/state labels, and validity flags.
- The actual generation effect must happen through the active KV cache used by transformer attention.

## Research goal

Read the reference repositories and help refine a concrete implementation where the KV cache has typed lifecycle states:

1. `recent`: full KV of recent chunks for local continuity.
2. `anchor`: high-trust long-range sink/anchor KV.
3. `compressed`: evicted historical KV stored compactly.
4. `motion`: dynamic-head KV / latent-delta entries for motion continuity.
5. `stale/invalid`: metadata state blocking conflicting or low-trust entries.
6. `recall`: not necessarily persistent storage; a temporary active view gathered from compressed/anchor/history entries for the current chunk.

## Repositories to inspect

Please inspect the following repositories if available in `third_party/`:

- Self-Forcing
- Causal-Forcing
- RollingForcing / Rolling Sink
- Pyramid-Forcing
- Forcing-KV
- MemRoPE
- LongLive-RAG
- Echo-Forcing

## Concrete tasks

For each repository, produce a markdown note under `docs/code_notes/REPO_NAME.md` answering:

1. Where is the autoregressive generation loop?
2. Where is KV cache created, updated, concatenated, or evicted?
3. What is the shape/layout of KV tensors?
4. Are K/V tensors pre-RoPE or post-RoPE at the interception point?
5. Can per-layer/head K/V and attention maps be returned or hooked?
6. How does the method choose sink/recent/compressed/recalled history?
7. What can be directly reused for LifeCache-Forcing?
8. What code changes are needed to implement typed KV slots?

## Implementation target

Create a prototype design under `src/lifecycle_kv/` with:

- `CacheEntry`: metadata for one KV segment.
- `LifecycleKVCache`: stores tensor payloads and metadata.
- `CacheIndex`: retrieves entries by scene/entity/head-role/state/trust.
- `HeadRoleProfiler`: offline profiling for anchor/motion/layout/entity-like heads.
- `compose_active_cache(layer_id, head_id, prompt_state)`: returns active K/V for attention.

## Specific questions to answer

1. Should `recall` be a persistent pool or a temporary view from `compressed`? Give code-level reasoning.
2. How should `compressed` entries be selected and stored?
3. How should motion heads be identified and updated?
4. How should stale/invalid entries be marked and blocked from active attention?
5. What is the minimal modification to Self-Forcing attention forward to support active KV composition?
6. What ablations should be added?

## Desired output

Produce:

1. A repository-reading report.
2. A concrete implementation plan with file-level modifications.
3. A minimal patch skeleton for Self-Forcing or Causal-Forcing.
4. A list of uncertainties that require running experiments.
