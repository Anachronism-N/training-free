# training-free

Research scaffold for training-free long-horizon video generation on
Self-Forcing / Causal-Forcing style autoregressive video diffusion.

The newest method candidate is **Commit Forcing: Reliability-Gated Multiscale
State Commit for Training-Free Long Video Extrapolation**. It treats long
autoregressive generation as a state-admission problem: denoising-trajectory
disagreement decides which clean states enter a bounded
origin/compressed/recent commit bank, motion controls consolidation and
readout, and selected references correct the main sampling path while the
native recent cache remains unchanged.
Single-prompt 30s+ extrapolation is the primary task; prompt switching is
secondary.

This reset follows the completed LifeCache-v3 screen: the side-memory
intervention was measurable but all variants were visually equivalent to
native SF and collapsed at the same time. The result is recorded in
`docs/73_lifecache_v3_screen_results.md`. Commit Forcing v1 produced a visible
but limited improvement over native SF; its metrics, human review, and three
remaining failure modes are in `docs/75_commit_forcing_v74_screen_results.md`.
The multiscale v2 lifecycle, trajectory-coupled re-noising, provenance
boundary, 16-GPU matrix, logs, and go/no-go rules are in
`docs/76_multiscale_commit_bank_design_and_server_plan.md`. The v2 additions
remain unvalidated.
The paper/code provenance ledger, license audit, high-overlap related work,
and claim-safety rules are recorded in
`docs/64_related_work_code_provenance_and_claims.md`.
The latest single-prompt correction, server matrix, debug invariants, and
go/no-go rules are in `docs/68_single_prompt_continuity_recall.md`.
The current paper alignment, canonical PF/Echo baselines, review-first protocol,
and server commands are in `docs/69_paper_alignment_canonical_experiments.md`.
The superseded v3 cache lifecycle and intervention-routing design remain in
`docs/70_lifecache_v3_typed_memory_intervention_routing.md`.
Use `docs/67_post_sweep_optimization_and_server_protocol.md` for the separate
prompt-switch/return-recall branch.

## Current Hypothesis

Long AR video generation needs to answer three questions before using a state
as long-term context:

- **Should it be committed?** Estimate frame reliability from disagreement
  among the clean predictions already produced along the denoising path.
- **Where does it live?** Keep immutable episode origins, motion-compatible
  multiscale summaries, exact recent commits, and the native SF recent cache
  under separate update rules.
- **How does it intervene?** At selected nominal diffusion timesteps, denoise
  with the reference bank, re-noise to the same actual scheduler timestep, and
  then denoise with the native recent context. This changes the sampling path
  instead of weakly blending a side-memory attention output.

Manual head/layer classes are not part of P0. They may return only if measured
counterfactual evidence is stable across prompts and seeds.

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
|   |-- 64_related_work_code_provenance_and_claims.md
|   |-- 68_single_prompt_continuity_recall.md
|   |-- 69_paper_alignment_canonical_experiments.md
|   |-- 70_lifecache_v3_typed_memory_intervention_routing.md
|   |-- 71_human_review_and_code_alignment.md
|   |-- 72_lifecache_v3_post_review_optimization.md
|   |-- 73_lifecache_v3_screen_results.md
|   |-- 74_commit_forcing_research_reset.md
|   |-- 75_commit_forcing_v74_screen_results.md
|   `-- 76_multiscale_commit_bank_design_and_server_plan.md
|-- prompts/
|   |-- lifecache_v3_calibration_complex_12.txt
|   |-- lifecache_v3_single_long_complex_12.txt
|   `-- ...
|-- scripts/
|   |-- bootstrap_repos.sh
|   |-- analyze_hrem_v2_debug.py
|   |-- build_intervention_profile.py
|   |-- compute_temporal_jump_diagnostic.py
|   |-- run_v69_typed_cache_16gpu.sh
|   |-- run_v74_commit_forcing_16gpu.sh
|   |-- run_v76_multiscale_commit_16gpu.sh
|   |-- summarize_commit_forcing_trace.py
|   `-- ...
|-- src/
|   `-- lifecycle_kv/
`-- third_party/
    |-- Self-Forcing/
    |-- Pyramid-Forcing/
    `-- ...
```

## Implementation Status

Commit Forcing is integrated into the Self-Forcing inference path:

- `commit_forcing.py`: denoising-path reliability, online latent motion,
  bounded FIFO or origin/summary/recent banks, motion-compatible multiscale
  consolidation, pre-RoPE reference reconstruction, trajectory-coupled
  re-noising, and JSONL trace.
- `third_party/Self-Forcing/.../causal_model.py`: optional pre-RoPE K capture
  for clean state commits.
- `third_party/Self-Forcing/pipeline/causal_inference.py`: nominal versus
  warped timestep handling, reference-conditioned forward, re-noise,
  normal-context forward, and clean-state commit.
- `tests/test_commit_forcing.py`: CPU tests for reliability, admission,
  reference-cache construction, episode reset, multiscale compaction, motion
  gating, and fresh/trajectory noise behavior.
- `scripts/run_v74_commit_forcing_16gpu.sh`: three-GPU smoke, 16-cell
  single-prompt screen, and four-seed native/PF/fixed-origin/hybrid confirm.
- `scripts/run_v76_multiscale_commit_16gpu.sh`: official SF/PF/Echo baselines
  plus re-noise, cache lifecycle, motion gate, and merge-policy ablations.
- `scripts/summarize_commit_forcing_trace.py` and `scripts/v74_postprocess.sh`:
  strict mechanism checks and post-review evaluation.

All new behavior is off by default. Commit Forcing cannot be enabled together
with LifeCache or Structured Memory in the P0 implementation.

The superseded LifeCache-v3 candidate remains connected to the HREM-v2
side-memory path for negative controls and historical ablations:

- `typed_memory.py`: exact-anchor and temporal-summary state machine with
  explicit budgets, admission, replacement, merge and scope rules.
- `episodic_archive.py`: bounded episode-aware K/V archive and typed sidecars.
- `intervention_router.py`: offline profile lookup, online rank calibration,
  safety constraints, top-budget routing and abstention.
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
- `scripts/build_intervention_profile.py`: converts paired native/intervention
  video metrics into a reliability-weighted layer/head/call profile.
- `scripts/compute_temporal_jump_diagnostic.py`: paired appearance/flow jump
  diagnostics for the PF discontinuity and smooth-activation ablation.
- `scripts/run_v69_typed_cache_16gpu.sh`: historical 12-prompt, 16-GPU
  LifeCache-v3 screen and confirmation phases.

The current machine has no configured PyTorch/GPU runtime, so only tests that do
not import PyTorch and static compilation can run here. The Commit Forcing
tensor/CUDA path still requires the server `smoke` phase.

## Experiment Entry Points

The current v2 experiments are documented in:

```text
docs/76_multiscale_commit_bank_design_and_server_plan.md
```

Run the mandatory smoke test first, then the 16-GPU single-prompt screen:

```bash
SMOKE_FRAMES=12 GPU_LIST=0,1,2,3 \
bash scripts/run_v76_multiscale_commit_16gpu.sh smoke

GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v76_multiscale_commit_16gpu.sh screen
```

The screen separates official baselines, v74 controls, trajectory re-noising,
multiscale history, motion-gated readout, and summary merge policies. Promote
only cells that improve identity without worsening freeze, style shift, or
temporal jumps.

## Third-Party Code

Most directories under `third_party/` are vendored source code, not Git
submodules. `third_party/Forcing-KV/` is currently an empty placeholder and
must not be described as a local reproduction. Large model checkpoints,
generated videos, logs, and Python cache files should stay out of Git.

The detailed paper, code-path, license, and claim-boundary audit is in
`docs/64_related_work_code_provenance_and_claims.md`. A repository link in the
table below does not imply that its code is used by the current method.

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
| Not vendored | [xbxsxp9/Pathwise_TTC](https://github.com/xbxsxp9/Pathwise_TTC) | Closest paper-level prior for fixed-reference pathwise correction; no source code copied. |

[Future Forcing](https://arxiv.org/abs/2605.30083) is also tracked as a
paper-level collision for future-query cache policy; no local source tree or
unverified repository link is recorded.

## Model Files

Model weights are intentionally not committed. For the current Self-Forcing
Commit Forcing experiment, place:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

The four-seed confirmation and older Pyramid-Forcing experiments additionally
use:

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

To run the current Commit Forcing v2 smoke and screen:

```bash
SMOKE_FRAMES=12 GPU_LIST=0,1,2,3 \
bash scripts/run_v76_multiscale_commit_16gpu.sh smoke

GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v76_multiscale_commit_16gpu.sh screen
```

After blind review:

```bash
HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 GPU=0 \
RUN_ROOT="$PWD/runs/v76_multiscale_commit_screen" \
bash scripts/v74_postprocess.sh
```

Historical LifeCache/HREM paper matrices remain available:

```bash
bash scripts/run_paper_single_prompt_30s.sh 0 1 2 3 4
bash scripts/run_paper_scene_switch_30s.sh 0 1 2 3
bash scripts/run_v69_typed_cache_16gpu.sh baselines
bash scripts/run_v69_typed_cache_16gpu.sh screen
```

Their historical review/metric commands are:

```bash
HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh single 0
HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh scene 0
```
