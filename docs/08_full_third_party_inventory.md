# Full third-party inventory

This repository contains a broader `third_party/` collection than the initial LifeCache notes covered. This document records the intended role of each codebase for LifeCache-Forcing.

## Coverage status

| Directory | Role for LifeCache | Code-reading priority | Canonical URL status |
|---|---|---:|---|
| Self-Forcing | Primary AR baseline and first patch target | S | verified |
| Causal-Forcing | Secondary AR baseline; CFG-cache compatibility test | S | verified |
| RollingForcing | Anchor/sink + rolling recent cache implementation | S | verified |
| DeepForcing | Deep sink and participative compression baseline | S | verified |
| Pyramid-Forcing | Head-aware cache policy and head labels | S | verified |
| Forcing-KV | Static/dynamic head split; dynamic temporal cache; motion slot reference | S | canonical URL verified; local directory empty |
| MemRoPE | RoPE-safe memory, pre/post-RoPE distinction, temporal/spatial indexing | S | verified |
| LongLive-RAG | Compressed/offloaded history and temporary recall view | S | verified |
| Echo-Forcing | Preserve/recall/forget scene memory and decay/stale semantics | S | verified |
| IAMFlow | Entity/state memory, LLM/VLM verification, agentic memory table | A | verified |
| infinity-rope | RoPE/position extrapolation reference for recall legality | A | verified |
| FreePCA | Low-rank/PCA-style compressed memory reference | B | verified |
| DiT-Extrapolation | Positional/DiT extrapolation reference | B | verified: `thu-ml/DiT-Extrapolation` |
| FreeLOC | Layer-adaptive OOD/position correction reference | B | verified: `Westlake-AGI-Lab/FreeLOC` |
| MIGA | Infinite-frame / alignment / consistency reference | B | verified: `XiaokunFeng/MIGA` |
| LongVideoSparseAttention | Sparse attention and budgeted long-context reference | B | verified: `JiusiServe/LongVideoSparseAttention` |
| MotionCache | Motion-aware cache/reuse reference | B | verified: `MAC-AutoML/MotionCache` |
| FlowCache | Flow or motion-guided cache reference | B | verified: `mikeallen39/FlowCache` |
| SWIFT | Semantic injection cache / prompt-adaptive memory reference | B | verified: `ShanwenTan/SWIFT` |

## How to read the inventory

LifeCache does not need to directly absorb every mechanism. The repository should be read through the following questions:

1. Does the method modify the AR KV cache itself?
2. Does it build an active K/V view before attention?
3. Does it compress or evict historical K/V?
4. Does it retrieve historical information only when needed?
5. Does it classify heads or use head-specific cache policies?
6. Does it handle motion, flow, or dynamic temporal continuity?
7. Does it handle RoPE or positional legality for long-horizon recall?
8. Does it maintain scene/entity/state metadata that can become CacheEntry metadata?
9. Does it validate, decay, or invalidate stale memory?

## Immediate reading plan

### Stage 1: patch target and strongest baselines

Read these first and keep line-level notes:

- `third_party/Self-Forcing/pipeline/causal_inference.py`
- `third_party/Self-Forcing/wan/modules/causal_model.py`
- `third_party/Causal-Forcing/long_video/pipeline/causal_diffusion_inference.py`
- `third_party/RollingForcing/wan/modules/causal_model.py`
- `third_party/DeepForcing/wan/modules/causal_model_DS.py`
- `third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv`
- `third_party/Forcing-KV/wan/modules/causal_model_forcingkv.py`

### Stage 2: memory and recall semantics

- `third_party/LongLive-RAG/README.md`
- `third_party/Echo-Forcing/pipeline/causal_inference.py`
- `third_party/IAMFlow/iamflow/agents/memory_bank.py`
- `third_party/IAMFlow/iamflow/agents/llm_agent.py`
- `third_party/IAMFlow/iamflow/agents/vlm_agent.py`

### Stage 3: RoPE, sparse, spectral, and motion extensions

- `third_party/MemRoPE/wan/modules/causal_model.py`
- `third_party/infinity-rope/wan/modules/causal_model.py`
- `third_party/FreePCA/lvdm/modules/attention.py`
- `third_party/LongVideoSparseAttention/*`
- `third_party/MotionCache/*`
- `third_party/FlowCache/*`
- `third_party/SWIFT/*`

## Provenance and license status

Canonical URLs for the current inventory were rechecked on 2026-07-22 and are
recorded in `docs/64_related_work_code_provenance_and_claims.md`. That document
also distinguishes direct code ports, idea-level influence, unused candidates,
and missing top-level licenses.

`third_party/Forcing-KV/` is currently empty. Forcing-KV must remain a paper and
upstream-code reference until its source and license are actually vendored and
audited. Do not infer local review or reproduction from the directory name.
