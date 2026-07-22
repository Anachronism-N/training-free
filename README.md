# training-free

Research scaffold for training-free long-horizon video generation on
Self-Forcing / Causal-Forcing style autoregressive video diffusion.

The current method candidate is **HREM-v2: Head-Role Evidence-gated Episodic
Memory for Training-free Long AR Video Generation**. It keeps a bounded,
episode-balanced sidecar of clean pre-RoPE K/V frames, admits only non-recent
episodes supported by both prompt and visual-query evidence, and routes the
independent memory-attention output with online per-head persistence and motion
evidence. Uncertain recall returns the native Self-Forcing output unchanged.
The paper-facing contribution is the factorized two-stage admission decision,
not the archive or top-k retrieval components in isolation. The exact borrowed
components, claim boundary, and falsifiable hypotheses are recorded in
`docs/60_hrem_v2_novelty_and_debug_protocol.md`.
For a single review entry covering the idea, paper story, code changes, server
commands, expected logs, and go/no-go criteria, start with
`docs/61_hrem_v2_review_and_runbook.md`.

## Current Hypothesis

Long AR video generation needs to answer two different questions before using
history:

- **Which episode?** Reject the current and immediately previous scene, then
  require semantic and visual-query evidence to agree on a historical scene.
- **Which heads?** Favor heads with persistent K/V and stable queries while
  suppressing heads that show value variation or query drift.
- **How to fuse?** Use a separate bounded memory-attention branch; never write
  recalled K/V into the native cache.

LifeCache-v1 and CEMR remain in the repository as prior prototypes and
ablation infrastructure.

## Repository Layout

```text
training-free/
|-- README.md
|-- configs/
|   `-- lifecache-v1-minimal.yaml
|-- docs/
|   |-- 15_lifecache_doc_index.md
   |   |-- 59_hrem_v2_evidence_gated_episodic_memory.md
   |   |-- 60_hrem_v2_novelty_and_debug_protocol.md
   |   |-- 61_hrem_v2_review_and_runbook.md
   |   `-- 62_aaai_provisional_title_abstract.md
|-- prompts/
|   `-- hrem_v2_aba_complex_3.txt
|-- scripts/
|   |-- bootstrap_repos.sh
   |   |-- analyze_hrem_v2_debug.py
   |   `-- run_hrem_v2_evidence.sh
|-- src/
|   `-- lifecycle_kv/
`-- third_party/
    |-- Self-Forcing/
    |-- Pyramid-Forcing/
    `-- ...
```

## Implementation Status

The HREM-v2 path is connected end-to-end in Self-Forcing:

- `episodic_archive.py`: bounded episode-aware K/V archive.
- `role_episodic.py`: dual-evidence episode admission and online head routing.
- `attention_fusion.py`: query-conditioned readout and fail-closed fusion.
- `third_party/Self-Forcing/.../causal_model.py`: pre-RoPE capture and the
  independent memory-attention branch.
- `third_party/Self-Forcing/pipeline/causal_inference.py`: episode lifecycle and
  clean-context archive commits.
- `scripts/analyze_hrem_v2_debug.py`: structural diagnosis for archive,
  admission, head routing, fusion strength, and causal invariants.

The current machine has no configured PyTorch/GPU runtime, so the code has been
syntax-checked but the new CUDA path still requires the server Stage-1 run.

## Experiment Entry Points

The current causal comparison is documented in:

```text
docs/61_hrem_v2_review_and_runbook.md
docs/59_hrem_v2_evidence_gated_episodic_memory.md
```

It defines five 120-frame inference cells using three complex A-B-A prompts:

```text
prompts/hrem_v2_aba_complex_3.txt
```

The cells are:

1. raw native Self-Forcing schedule;
2. native Self-Forcing with the shared scene-boundary reset control;
3. oracle episode-0 memory;
4. dual-evidence episode selection with all heads;
5. full HREM-v2 with online head-role evidence.

## Third-Party Code

The `third_party/` directory is vendored source code in this repository, not
Git submodules. Large model checkpoints, generated videos, logs, and Python
cache files should stay out of Git.

The original repositories referenced by this project are:

| Local directory | Original repository | Role in this project |
|---|---|---|
| `third_party/Self-Forcing` | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | Primary AR baseline and first patch target. |
| `third_party/Causal-Forcing` | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) | Secondary AR baseline and cache-compatibility target. |
| `third_party/RollingForcing` | [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) | Rolling-window and sink/anchor cache reference. |
| `third_party/DeepForcing` | [cvlab-kaist/DeepForcing](https://github.com/cvlab-kaist/DeepForcing) | Deep sink and participative compression reference. |
| `third_party/Pyramid-Forcing` | [if-lab-pku/Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | Head-aware cache policy and head labels. |
| `third_party/Forcing-KV` | [zju-jiyicheng/Forcing-KV](https://github.com/zju-jiyicheng/Forcing-KV) | Static/dynamic head split and motion-oriented K/V cache reference. |
| `third_party/MemRoPE` | [YoungRaeKimm/MemRoPE](https://github.com/YoungRaeKimm/MemRoPE) | RoPE-safe memory and temporal/spatial position indexing reference. |
| `third_party/LongLive-RAG` | [qixinhu11/LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG) | Offloaded history and temporary recall-view reference. |
| `third_party/Echo-Forcing` | [mingqiangWu/Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) | Preserve/recall/forget scene memory reference. |
| `third_party/IAMFlow` | [Eddie0521/IAMFlow](https://github.com/Eddie0521/IAMFlow) | Entity/state memory and VLM/LLM validation reference. |
| `third_party/infinity-rope` | [yesiltepe-hidir/infinity-rope](https://github.com/yesiltepe-hidir/infinity-rope) | RoPE and positional extrapolation reference. |
| `third_party/FreePCA` | [JosephTiTan/FreePCA](https://github.com/JosephTiTan/FreePCA) | Low-rank / PCA-style compressed memory reference. |
| `third_party/DiT-Extrapolation` | [thu-ml/DiT-Extrapolation](https://github.com/thu-ml/DiT-Extrapolation) | DiT positional extrapolation reference. |
| `third_party/FreeLOC` | [Westlake-AGI-Lab/FreeLOC](https://github.com/Westlake-AGI-Lab/FreeLOC) | Layer-adaptive OOD / position correction reference. |
| `third_party/MIGA` | [XiaokunFeng/MIGA](https://github.com/XiaokunFeng/MIGA) | Infinite-frame / alignment / consistency reference. |
| `third_party/LongVideoSparseAttention` | [JiusiServe/LongVideoSparseAttention](https://github.com/JiusiServe/LongVideoSparseAttention) | Sparse attention and long-context budget reference. |
| `third_party/MotionCache` | [MAC-AutoML/MotionCache](https://github.com/MAC-AutoML/MotionCache) | Motion-aware cache reuse reference. |
| `third_party/FlowCache` | [mikeallen39/FlowCache](https://github.com/mikeallen39/FlowCache) | Flow-guided or motion-guided cache reference. |
| `third_party/SWIFT` | [ShanwenTan/SWIFT](https://github.com/ShanwenTan/SWIFT) | Semantic injection cache and prompt-adaptive memory reference. |

## Model Files

Model weights are intentionally not committed. For the current HREM-v2
experiment, place the required files as described in
`docs/59_hrem_v2_evidence_gated_episodic_memory.md`:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
```

## Quick Start

To clone or refresh reference repositories in a fresh workspace:

```bash
bash scripts/bootstrap_repos.sh
```

To run the first HREM-v2 matrix:

```bash
bash scripts/run_hrem_v2_evidence.sh 0 1 2 3
```

Then audit routing and evaluate A-B-A return:

```bash
python scripts/summarize_hrem_v2_trace.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl --strict
python scripts/analyze_hrem_v2_debug.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl \
  --strict --json-output runs/hrem_v2_evidence_s0/traces/hrem_v2_diagnosis.json
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hrem_v2.py \
  --run-root runs/hrem_v2_evidence_s0
```
