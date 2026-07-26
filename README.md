# training-free

Research scaffold for training-free long-horizon video generation on
Self-Forcing / Causal-Forcing style autoregressive video diffusion.

The current pre-result candidate is **v100 History-Polarity Event Cache
(HP-Event)**. It uses the reproducible old-v98 two-role map: 304
History-Supportive heads and 56 Recent-Responsive heads. The exact post-hoc PF
overlap is Anchor `169/3`, Wave `133/23`, and Veil `2/30`
(Supportive/Responsive). PF labels are not used to form this map.

Supportive heads use `sink3 + stride4 + recent4`. The primary Responsive
hypothesis uses `sink3 + motion-event2 + cyclic2 + recent4`: one frame per
clean block is selected from a layer-shared normalized V-change score, while
two phase slots preserve short periodic evidence. `cyclic4` with both sink1
and sink3, `motion4`, and budget-matched `recent8` are explicit controls.
Merge is diagnostic-only until its binary-route artifact risk is resolved.

For A-B-A prompts, an optional scene episode archives/restores only Supportive
stride banks, clears Responsive local middle state, and keeps zero or one
recent bridge frame. IDF prompt matching selects the recalled scene; every
similarity, cache action, selected motion frame, and ownership check is written
to JSONL.

The authoritative method, 16-cell one-video screen, four-node commands,
debug contract, deferred broad ablations, provenance boundaries, and
result-dependent paper story are in
`docs/104_v100_responsive_event_cache_and_aba_fast_screen.md`. The runner is
`scripts/run_v100_fast_selection_1video.py`. `docs/103` is now the historical
middle-relative/stride-cyclic proposal.

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

The current hypothesis is that net historical QK support exposes two
functional groups that need different bounded temporal evidence:

- **Frozen old-v98 classification:** signed historical QK mass `>= 0` is
  History-Supportive and `< 0` is Recent-Responsive; PF labels are used only
  for post-hoc analysis.
- **Supportive memory:** sink3 + stride(interval 6, capacity 4) + recent4.
- **Responsive memory hypothesis:** sink3 + clean-V motion events(capacity 2)
  + phase cyclic(capacity 2) + recent4.
- **Equal-budget controls:** cyclic4/sink1, cyclic4/sink3, motion4, and recent8.
- **Exclusive cache ownership:** explicit composition owns sink, middle, and
  recent; the legacy dynamic-history owner is disabled.
- **Scene episodes:** Supportive stride anchors are archived by matched scene;
  Responsive middle memory resets and rebuilds locally.
- **Trust promotion:** v78 is optional and retained only if it improves the
  selected read cache.
- **Fail-closed scope:** every experimental component is default-off, requires
  a frozen map/config, and must pass decoded-video and trace audits.

PF's stride/cyclic/merge operators and head-aware caching have prior art.
Echo-Forcing's scene pool and recall lifecycle also have prior art. They are
borrowed components, not contribution claims. The proposed claims are limited
to the two-role criterion, generated-motion event memory, role-specific
composition, auditable ownership, and role-aware scene episodes, and only
survive if the documented controls support them.

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
|   |-- 91_transitioncache_paper_story.md
|   |-- 92_prompt_contrastive_binary_cache_and_uniqueness_plan.md
|   |-- 93_moviebench_10h_128_and_head32_plan.md
|   |-- 95_post_v93_dual_axis_phase_cache.md
|   |-- 96_qk_threshold_binary_cache_method_and_experiment.md
|   |-- 97_score_artifact_threshold_pf_merge_experiment.md
|   |-- 98_history_polarity_dual_memory_method.md
|   |-- 99_v98_32gpu_runbook_and_paper_plan.md
|   |-- 100_v99_binary_cache_recovery_and_paper_story.md
|   |-- 103_v100_final_candidate_and_immediate_plan.md
|   `-- 104_v100_responsive_event_cache_and_aba_fast_screen.md
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
|   |-- build_prompt_contrastive_head_maps.py
|   |-- run_v92_prompt_binary_cache_16gpu.sh
|   |-- postprocess_v92_prompt_binary_cache.sh
|   |-- analyze_v92_metrics.py
|   |-- run_v92_echo_unique_snapshot_4gpu.sh
|   |-- run_v93_moviebench_main_16gpu.sh
|   |-- run_v93_moviebench_head32_16gpu.sh
|   |-- postprocess_v93_moviebench.sh
|   |-- analyze_v93_moviebench.py
|   |-- run_v93_moviebench_10h.sh
|   |-- run_v95_dual_axis_warmup_16gpu.sh
|   |-- postprocess_v95_dual_axis.sh
|   |-- summarize_prompt_warmup_trace.py
|   |-- analyze_v95_dual_axis.py
|   |-- run_v96_qk_head_profile_16gpu.sh
|   |-- build_v96_qk_head_thresholds.py
|   |-- run_v96_binary_cache_16gpu.sh
|   |-- postprocess_v96_binary_cache.sh
|   |-- analyze_v96_binary_cache.py
|   |-- run_v96_10h.sh
|   |-- extract_v97_qk_head_scores.py
|   |-- classify_v97_qk_head_scores.py
|   |-- run_v97_qk_head_profile_16gpu.sh
|   |-- run_v97_threshold_pf_merge_16gpu.sh
|   |-- summarize_v97_policy_traces.py
|   |-- postprocess_v97_threshold_pf_merge.sh
|   |-- analyze_v97_threshold_pf_merge.py
|   |-- run_v97_10h.sh
|   |-- run_v98_middle_relative_profile_16gpu.sh
|   |-- extract_v98_middle_relative_scores.py
|   |-- build_v98_history_polarity_maps.py
|   |-- run_v98_history_polarity_4node_32gpu.sh
|   |-- run_v100_fast_selection_1video.py
|   |-- audit_v98_policy_traces.py
|   |-- postprocess_v98_history_polarity.sh
|   |-- analyze_v98_history_polarity.py
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

The v100 candidate path and its fail-closed diagnostics are implemented:

- `pyramidkv/motion_event.py`: bounded clean-V motion-event memory with
  layer-shared selection and per-head frame storage.
- `pyramidkv/stride.py` and `pipeline/causal_inference.py`: scene-specific
  Supportive stride banks, IDF A-B-A matching, Responsive local reset, and a
  configurable zero/one-frame recent bridge.
- `run_v100_fast_selection_1video.py`: frozen 16-cell, one-video 30-second
  screen for Responsive policies, small add-ons, and A-B-A episodes, including
  policy/motion/scene/video audits.

- `run_v98_middle_relative_profile_16gpu.sh` and
  `extract_v98_middle_relative_scores.py`: frozen 64-profile, two-topology
  calibration with dependency hashes and acceptance gates.
- `build_v98_history_polarity_maps.py`: fail-closed map construction from the
  accepted artifact, including natural-zero, random, sign-fraction, and PF
  oracle controls.
- `run_v99_binary_cache_recovery_4node_32gpu.py`: one-video diagnostics,
  eight-cell `candidate32`, three-cell `main128`, immutable cross-node
  contract, decoded-video audit, map cross-tab validation, and exclusive cache
  trace checks.
- `postprocess_v100_hp_cache.sh`: v100 contract/video audit, exact-index
  staging, blind-review preparation/freeze verification, resumable
  GPU-batched VBench-Long/comprehensive metrics, and temporal-jump evaluation.
- `audit_v98_policy_traces.py` and
  `postprocess_v98_history_polarity.sh`: historical v98 auditing/evaluation
  tools. Their hard-coded v98 method contract must not be used as a v100
  postprocessor.

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
- `scripts/run_v93_moviebench_main_16gpu.sh`: eight-method MovieBench-128
  comparison with two global-index shards per method and matched per-prompt
  reseeding.
- `scripts/run_v93_moviebench_head32_16gpu.sh`: 16-cell MovieBench-32 causal
  screen for PF-binary and prompt-contrastive head classifications.
- `scripts/postprocess_v93_moviebench.sh`: parallel comprehensive and
  VBench-Long evaluation, temporal-jump diagnostics, strict count checks and
  blind-review packaging for both v93 matrices.
- `scripts/run_v93_moviebench_10h.sh`: resumable generation-and-metrics queue
  for one 16-GPU node.
- `scripts/summarize_cache_transition_trace.py`: strict layer/head mechanism
  validation and reason/label/role statistics.
- `pyramidkv/prompt_warmup.py`: default-off prompt-role history shielding,
  deterministic per-head release, and JSONL mechanism traces.
- `scripts/run_v95_dual_axis_warmup_16gpu.sh`: 16-cell MovieBench-32 causal
  screen for semantic update priority and phase-limited history exposure.
- `scripts/postprocess_v95_dual_axis.sh`: full per-cell VBench-Long,
  comprehensive metrics, temporal-jump diagnostics, and blind review.
- `scripts/summarize_prompt_warmup_trace.py` and
  `scripts/analyze_v95_dual_axis.py`: strict runtime audit and predeclared
  semantic-versus-random/inverse decision gates.

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
docs/103_v100_final_candidate_and_immediate_plan.md
```

After the one-video PF-AW stride/cyclic diagnostic, launch the same no-Merge
MovieBench-32 command on four eight-GPU nodes, changing only `NODE_RANK`:

```bash
NODE_RANK=0 GPU_LIST=0,1,2,3,4,5,6,7 \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v100_hp_cache_candidate32" \
python scripts/run_v99_binary_cache_recovery_4node_32gpu.py candidate32
```

The eight cells occupy all eight GPUs per node and every node runs every method
on a disjoint prompt shard. Review videos blind before metrics. Run
`main128` only after the learned map beats its count-matched random and
inverted controls without motion collapse.

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
| Not vendored | [NVlabs/LongLive](https://github.com/NVlabs/LongLive) | Training-time long causal video and KV recaching comparison. |
| Not vendored | [KlingAIResearch/MemFlow](https://github.com/KlingAIResearch/MemFlow) | Prompt-conditioned memory retrieval comparison. |
| Not vendored | [csguoh/DummyForcing](https://github.com/csguoh/DummyForcing) | Head/context allocation comparison. |

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
