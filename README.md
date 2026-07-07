# training-free

Research scaffold for training-free long-horizon video generation on Self-Forcing / Causal-Forcing.

Working idea: **LifeCache-Forcing** — lifecycle-aware KV-cache management for autoregressive video diffusion. The memory bank is not an independent external generator condition; it is a cache index/control plane that manages KV slots inside the active KV cache.

## Core hypothesis
Long video generation should not only retain a longer history. The AR generator needs a typed KV cache lifecycle:

- **Recent**: short-term local continuity and chunk-boundary coherence.
- **Anchor**: long-range stable sink/identity/layout anchors.
- **Compressed**: evicted history stored in compact KV/latent form.
- **Recall view**: a temporary active view constructed from compressed/anchor/history entries for the current chunk.
- **Motion**: dynamic-head KV and latent-delta signals for motion continuity.
- **Invalid/Stale metadata**: entries blocked or down-weighted due to state conflicts or low trust.

Important: `recall` should usually be a **view/state**, not a separate persistent storage class.

## Repository layout

```text
training-free/
├── README.md
├── docs/
│   ├── 01_cache_taxonomy.md
│   ├── 02_compressed_vs_recall.md
│   ├── 03_motion_memory.md
│   ├── 04_stale_invalid_cache.md
│   ├── 05_reference_code_reading_plan.md
│   └── codex_prompt.md
├── scripts/
│   └── bootstrap_repos.sh
└── src/
    └── lifecycle_kv/
        ├── cache_types.py
        └── lifecycle_cache.py
```

## Quick start

```bash
bash scripts/bootstrap_repos.sh
```

Then ask Codex to inspect the reference implementations using `docs/codex_prompt.md`.
