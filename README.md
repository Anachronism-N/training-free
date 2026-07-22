# training-free

Research scaffold for training-free long-horizon video generation on
Self-Forcing / Causal-Forcing style autoregressive video diffusion.

The current method candidate is **Scope-Conditioned Evidence-Gated Historical
Recall for training-free long AR video generation**. It keeps a bounded sidecar
of clean pre-RoPE K/V frames and supports two explicit recall scopes:
intra-episode continuity recall for ordinary single-prompt long video, and
cross-episode return recall for A-B-A style prompt switching. Both use an
independent memory-attention branch, evidence-based abstention, optional online
head routing, and native fallback. These mechanisms are hypotheses pending GPU
experiments, not established quality improvements.
The paper/code provenance ledger, license audit, high-overlap related work,
and claim-safety rules are recorded in
`docs/64_related_work_code_provenance_and_claims.md`.
The latest single-prompt correction, server matrix, debug invariants, and
go/no-go rules are in `docs/68_single_prompt_continuity_recall.md`.
The current paper alignment, canonical PF/Echo baselines, review-first protocol,
and server commands are in `docs/69_paper_alignment_canonical_experiments.md`.
Use `docs/67_post_sweep_optimization_and_server_protocol.md` for the separate
prompt-switch/return-recall branch.

## Current Hypothesis

Long AR video generation needs to answer three questions before using history:

- **Which scope?** For continuous generation, read only sufficiently old frames
  from the current episode. For scene return, select a supported older episode
  and reject the immediately previous scene.
- **Which content?** Use current Q-K evidence and explicit abstention to choose
  a bounded set of full spatial K/V frames.
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
   |   |-- 62_aaai_provisional_title_abstract.md
   |   |-- 62_hrem_v2_results_and_iteration.md
   |   |-- 63_hrem_v2_p0_role_calibration.md
   |   |-- 64_related_work_code_provenance_and_claims.md
   |   |-- 65_swift_collision_audit.md
   |   |-- 66_gate_sweep_results_and_review.md
   |   |-- 67_post_sweep_optimization_and_server_protocol.md
   |   |-- 68_single_prompt_continuity_recall.md
   |   `-- 69_paper_alignment_canonical_experiments.md
|-- prompts/
|   |-- hrem_v2_aba_complex_3.txt
|   |-- hrem_v2_single_long_complex_3.txt
|   |-- paper_single_long_echo_3.txt
|   |-- paper_scene_switch_sf_3.txt
|   `-- paper_scene_switch_echo_3.txt
|-- scripts/
|   |-- bootstrap_repos.sh
   |   |-- analyze_hrem_v2_debug.py
   |   |-- prepare_blind_review.py
   |   |-- validate_echo_prompts.py
   |   |-- run_paper_single_prompt_30s.sh
   |   |-- run_paper_scene_switch_30s.sh
   |   `-- run_paper_metrics.sh
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
  admission, selected frame age, head routing, fusion strength, and causal
  invariants, factorized by layer and denoising call.
- `scripts/run_paper_single_prompt_30s.sh`: canonical native SF, official PF,
  official Echo, all-head recall, and role-routed recall matrix.
- `scripts/run_paper_scene_switch_30s.sh`: canonical segmented SF, official Echo
  recall, and our two mechanism cells for A-B-A return.
- `scripts/prepare_blind_review.py` and `scripts/run_paper_metrics.sh`: enforce
  blind human review before comprehensive and VBench-Long metrics.
- `scripts/run_hrem_v2_single_prompt.sh`: controlled native, capture-only,
  intra-all-head, and intra-role 30-second single-prompt matrix.
- `scripts/run_hrem_v2_role_ablation.sh`: single-GPU P0 matrix for absolute,
  relative, and hybrid head-role calibration.
- `scripts/compare_hrem_role_ablation.py`: joins video metrics with mechanism
  diagnostics without conflating retrieval acceptance and role selection.

The current machine has no configured PyTorch/GPU runtime, so the code has been
syntax-checked but the new CUDA path still requires the server Stage-1 run.

## Experiment Entry Points

The canonical paper experiments are documented in:

```text
docs/69_paper_alignment_canonical_experiments.md
```

The primary single-prompt matrix contains:

```text
sf_native
sf_pyramid_forcing
sf_echo_forcing
ours_all_heads
ours_role
```

The first pass is always three complex prompts, one seed, and 120 latent frames
(approximately 30 seconds), followed by blind human review, metrics, and trace
analysis in that order.

## Third-Party Code

Most directories under `third_party/` are vendored source code, not Git
submodules. `third_party/Forcing-KV/` is currently an empty placeholder and
must not be described as a local reproduction. Large model checkpoints,
generated videos, logs, and Python cache files should stay out of Git.

The detailed paper, code-path, license, and claim-boundary audit is in
`docs/64_related_work_code_provenance_and_claims.md`. A repository link in the
table below does not imply that its code is used by HREM-v2.

The original repositories referenced by this project are:

| Local directory | Original repository | Role in this project |
|---|---|---|
| `third_party/Self-Forcing` | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | Primary AR baseline and first patch target. |
| `third_party/Causal-Forcing` | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) | Secondary AR baseline and cache-compatibility target. |
| `third_party/RollingForcing` | [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) | Rolling-window and sink/anchor cache reference. |
| `third_party/DeepForcing` | [cvlab-kaist/DeepForcing](https://github.com/cvlab-kaist/DeepForcing) | Deep sink and participative compression reference. |
| `third_party/Pyramid-Forcing` | [if-lab-pku/Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | Head-aware cache policy and head labels. |
| `third_party/Forcing-KV` | [zju-jiyicheng/Forcing-KV](https://github.com/zju-jiyicheng/Forcing-KV) | Empty local placeholder; paper-level static/dynamic head reference only. |
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

Model weights are intentionally not committed. For the current Self-Forcing
HREM-v2.1 experiment, place:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

The older Pyramid-Forcing experiments additionally use:

```text
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
```

Echo-Forcing uses the same model family and checkpoint at:

```text
third_party/Echo-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Echo-Forcing/checkpoints/self_forcing_dmd.pt
```

## Quick Start

To keep existing vendored directories and clone only missing references:

```bash
bash scripts/bootstrap_repos.sh
```

To run the canonical primary matrix on five GPUs:

```bash
bash scripts/run_paper_single_prompt_30s.sh 0 1 2 3 4
```

For one GPU, run the same cells sequentially:

```bash
PARALLEL=0 bash scripts/run_paper_single_prompt_30s.sh 0 0 0 0 0
```

To run the canonical scene-switch/return matrix:

```bash
bash scripts/run_paper_scene_switch_30s.sh 0 1 2 3
```

Both generation scripts prepare a randomized blind-review directory. Freeze its
`scorecard.csv`, then run metrics:

```bash
HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh single 0
HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh scene 0
```
