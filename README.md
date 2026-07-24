# training-free

Research scaffold for training-free long-horizon video generation on
Self-Forcing / Causal-Forcing style autoregressive video diffusion.

The validated method candidate is **v78 Trust-Conditioned Cache Transition**.
It controls whether a clean autoregressive block may update Pyramid-Forcing's
middle cache, using free noisy/clean K/V disagreement, transition shock,
novelty, age and an asynchronous write budget. It does not alter PF's
sink/middle/recent read topology and adds no model forward. Across the latest
matched seed-0 screen, v78 beats PF by 0.004 DINO and is ranked best by human
review. Earlier seeds 2/3 also score well, but their reported PF comparison is
not seed-matched; v90 supplies the missing matched seeds 1-3.

The v86 screen is complete. Uniform v78 is best (`0.8536` DINO versus PF
`0.8496`) on the matched 16-prompt seed-0 screen. Hard learned role clocks are
not causally superior and can duplicate subjects; they are no longer a method
claim. The next experiment runs PF-v78 matched seeds 1-3 and tests a safer weak
motion priority that changes only budget ordering among trusted candidates.
Current analysis, code, 16-GPU commands and decision gates are in
`docs/90_post_v86_analysis_and_v90_experiment.md`. The claim-safe paper story,
section structure, figures and abstract template are in
`docs/91_transitioncache_paper_story.md`.

ProbeCache direct archive recall is now a negative branch. It retained identity
and often reduced temporal jump, but consistently introduced non-ID
hallucinations; inverse/random controls also produced polygon noise or
duplicated subjects. Results and human review are in `docs/83`-`docs/85`.
Single-prompt 30s+ extrapolation remains the primary task; prompt switching is
secondary.

This reset follows the completed LifeCache-v3 screen: the side-memory
intervention was measurable but all variants were visually equivalent to
native SF and collapsed at the same time. The result is recorded in
`docs/73_lifecache_v3_screen_results.md`. Commit Forcing v1 produced a visible
but limited improvement over native SF; its metrics, human review, and three
remaining failure modes are in `docs/75_commit_forcing_v74_screen_results.md`.
The multiscale Commit Forcing v2 screen was a negative result: trajectory
re-noising and compressed summaries were worse than native SF, while official
PF was the clear quality leader. Results are in
`docs/77_commit_forcing_v76_screen_results.md`. The current implementation,
16-GPU matrix, logs, and go/no-go rules are in
`docs/78_cache_transition_implementation_and_experiment_plan.md`.
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

The current hypothesis is that long AR generation fails partly because
unequally reliable generated states are promoted into persistent attention
memory. PF already provides a strong cache read topology; the method therefore
controls state admission instead of adding another recall path.

- **Trust signal:** noisy/clean descriptor disagreement plus shock from the
  last admitted clean state.
- **Uniform transition:** reliability, novelty, age and budget gate PF middle
  writes; this is the validated v78 method.
- **Optional weak priority:** v90 tests whether PF temporal classes can break
  ties among trusted candidates without receiving different stale-state ages.
- **Fail-closed scope:** no direct archive read, no extra forward, and all role
  behavior is off by default.

Head specialization and novelty-based memory updates have prior art. This does
not preclude a different classifier from being innovative, but the current
counterfactual classifier did not improve the target cache-write intervention.
The supported contribution candidate is noisy-clean trust-conditioned state
promotion and bounded asynchronous cache writes. PF labels are cited as a
borrowed prior in v90, not renamed as our classification.

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
|   |-- 76_multiscale_commit_bank_design_and_server_plan.md
|   |-- 77_commit_forcing_v76_screen_results.md
|   |-- 78_cache_transition_implementation_and_experiment_plan.md
|   |-- 80_v78_method_and_ama_veil_head_history.md
|   |-- 81_probecache_method_implementation_and_server_plan.md
|   |-- 82_probecache_10h_followup_experiment_plan.md
|   |-- 83_probecache_v81_screen_results.md
|   |-- 84_probecache_v81_v82_comprehensive_results.md
|   |-- 85_comprehensive_human_review_v81_v82.md
|   |-- 86_current_idea_and_role_transition_plan.md
|   |-- 87_transitioncache_method_and_16x30s_protocol.md
|   |-- 88_transitioncache_v86_partial_results.md
|   |-- 89_v86_human_review_and_combined_analysis.md
|   |-- 90_post_v86_analysis_and_v90_experiment.md
|   `-- 91_transitioncache_paper_story.md
|-- prompts/
|   |-- lifecache_v3_calibration_complex_12.txt
|   |-- lifecache_v3_single_long_complex_12.txt
|   |-- v86_single_long_complex_16.txt
|   `-- ...
|-- scripts/
|   |-- bootstrap_repos.sh
|   |-- analyze_hrem_v2_debug.py
|   |-- build_intervention_profile.py
|   |-- compute_temporal_jump_diagnostic.py
|   |-- run_v69_typed_cache_16gpu.sh
|   |-- run_v74_commit_forcing_16gpu.sh
|   |-- run_v76_multiscale_commit_16gpu.sh
|   |-- run_v77_commit_closure_16gpu.sh
|   |-- run_v78_cache_transition_16gpu.sh
|   |-- run_v81_probecache_profile_16gpu.sh
|   |-- run_v81_probecache_16gpu.sh
|   |-- postprocess_v81_probecache.sh
|   |-- run_v82_probecache_10h.sh
|   |-- postprocess_v82_probecache.sh
|   |-- build_probecache_head_profile.py
|   |-- build_probecache_control_labels.py
|   |-- compare_probecache_head_profiles.py
|   |-- audit_probecache_experiment_runs.py
|   |-- summarize_probecache_trace.py
|   |-- summarize_cache_transition_trace.py
|   |-- build_transition_role_consensus.py
|   |-- run_v86_role_transition_16gpu.sh
|   |-- postprocess_v86_role_transition.sh
|   |-- build_pf_transition_controls.py
|   |-- run_v90_priority_factorization_16gpu.sh
|   |-- postprocess_v90_priority_factorization.sh
|   |-- analyze_v90_metrics.py
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

ProbeCache remains integrated as a completed negative-result branch:

- `pyramidkv/probecache.py`: shared clean archive, prompt segments,
  persistent/reactive selection, abstention, PF fallback, and JSONL trace.
- `pyramidkv/adaptive_cache.py`: clean archive commits, role-specific recent
  tails, query-dependent packed-readout invalidation, and direct PF middle-slot
  replacement.
- `wan/modules/attention/core.py`: pre-RoPE query handoff and compact
  counterfactual output capture.
- `scripts/run_v81_probecache_profile_16gpu.sh`: 48 paired profile jobs and
  robust binary label generation.
- `scripts/run_v81_probecache_16gpu.sh`: 16-cell single/switch/smoke matrix.
- `scripts/run_v82_probecache_10h.sh`: deadline-aware profile replication,
  classification controls, multi-seed confirmation, 60-second extrapolation,
  switch follow-up, and blind-review preparation.
- `tests/test_build_probecache_head_profile.py` and
  `third_party/Pyramid-Forcing/tests/test_probecache.py`: CPU mechanism tests.

All ProbeCache behavior is off by default. The v81/v82 GPU runs confirmed
strong identity retention but systematic non-ID hallucinations from direct
archive readout.

Trust-conditioned transition control is integrated into both PF inference
pipelines:

- `pyramidkv/transition.py`: descriptor state, reliability/shock metrics,
  gate/stagger/full decisions, forced refresh, role-conditioned clocks, and
  per-head JSONL diagnostics.
- `pyramidkv/adaptive_cache.py`: intercepts clean middle writes while leaving
  sink and recent updates on PF's original path; transition-role labels are
  independent from PF read-policy labels.
- `pipeline/pyramidkv_config.py` and `inference.py`: default-off YAML/CLI
  controls with strict role-CSV validation.
- `scripts/run_v78_cache_transition_16gpu.sh`: SF/PF/Echo baselines, audit,
  gate, stagger, full, threshold, budget, age, and CFG-branch cells.
- `scripts/run_v86_role_transition_16gpu.sh`: native SF/PF/Echo baselines plus
  role-neutral, learned, replica, consensus, PF-binary, inverse/random controls,
  policy/depth ablations, multi-seed, ultralong and switch experiments.
- `scripts/postprocess_v86_role_transition.sh`: review-first DINO, temporal
  jump, ABA and 16-GPU parallel VBench-Long evaluation.
- `scripts/run_v90_priority_factorization_16gpu.sh`: matched PF-v78 seeds 1-3,
  weak-priority label controls, lifecycle factorization, PF class isolation,
  and depth routing over 16 prompts.
- `scripts/postprocess_v90_priority_factorization.sh` and
  `scripts/analyze_v90_metrics.py`: reused seed-0 baselines, paired-seed
  deltas, coherence diagnostics, temporal jump and parallel VBench-Long.
- `scripts/summarize_cache_transition_trace.py`: strict layer/head mechanism
  validation and reason/label/role statistics.

Commit Forcing remains integrated into the Self-Forcing inference path as a
completed secondary branch:

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
- `scripts/run_v77_commit_closure_16gpu.sh`: lower-frequency, trigger,
  correction-strength, per-timestep, and ramp controls requested by docs/77.
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

The current workstation has no usable PF CUDA runtime. Static compilation,
trace tests and shell validation run here; actual PF FlashAttention inference
and the torch-dependent transition tests must also run in the server
environment.

## Experiment Entry Points

The current main-line experiment is documented in:

```text
docs/86_current_idea_and_role_transition_plan.md
```

Run the role-transition smoke and 16-GPU causal screen:

```bash
bash scripts/run_v86_role_transition_16gpu.sh smoke
bash scripts/run_v86_role_transition_16gpu.sh screen
```

Freeze blind human review before metrics:

```bash
HUMAN_REVIEW_DONE=1 \
  bash scripts/postprocess_v86_role_transition.sh screen
```

The screen separates PF, uniform v78, role-neutral parity, learned/replica/
consensus/PF-binary/inverse/random labels, schedule components and depth
routes. Promote a role-conditioned policy only if it beats v78 and PF-binary
under the predeclared gates; otherwise v78 remains the final method.

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
| Not vendored | [Head Forcing project page](https://jiahaotian-sjtu.github.io/headforcing.github.io/) | Local/anchor/memory head and episodic-update prior; public code is still marked coming soon. |
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

Model weights are intentionally not committed. Native Self-Forcing baselines
use:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

The current ProbeCache/Pyramid-Forcing experiments use:

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

Build the ProbeCache head profile, run the smoke matrix, then run the main
single-prompt screen:

```bash
bash scripts/run_v81_probecache_profile_16gpu.sh

HEAD_CSV="$PWD/runs/v81_probecache_profile/labels/probecache_binary_labels.csv" \
bash scripts/run_v81_probecache_16gpu.sh smoke

HEAD_CSV="$PWD/runs/v81_probecache_profile/labels/probecache_binary_labels.csv" \
FRAMES=120 \
bash scripts/run_v81_probecache_16gpu.sh single
```

Prepare the blind review, freeze its scorecard, then run metrics:

```bash
bash scripts/postprocess_v81_probecache.sh prepare single

HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 GPU=0 \
bash scripts/postprocess_v81_probecache.sh metrics single
```

Historical Commit Forcing, LifeCache, and HREM matrices remain available:

```bash
bash scripts/run_v76_multiscale_commit_16gpu.sh smoke
bash scripts/run_v76_multiscale_commit_16gpu.sh screen
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
