# training-free

Research scaffold for training-free long-horizon video generation on
Self-Forcing / Causal-Forcing style autoregressive video diffusion.

The current task is the **v129 no-PF MovieBench-128 paper comparison**. v125 is
complete and selected Prototype4 plus age-bounded Retrieval1 as the strongest
effect-oriented base candidate, with the highest observed Dynamic Degree.
v129 reuses validated v125 SF and Ours videos, adds confidence/margin-gated
retrieval, and generates Deep Forcing, Rolling Forcing, and LongLive
comparators. It does not regenerate PF and defers A-B-A prompt switching.

The frozen table contains eight methods and 128 Qwen-rewritten prompts at
30 seconds. The primary VBench-Long pass uses all seven official quality
dimensions plus Overall Consistency. A prompt-aware evaluator maps numeric
split directories back to the exact frozen prompt text; official composite
scores are emitted only when every required component is present. The exact
four-node commands, model paths, reuse contracts, debug outputs, and claim
boundaries are in
`docs/129_no_pf_paper_comparison_and_10h_runbook.md`.

v119 kept the frozen old-v98 `304/56` diagnostic partition and tested five
new 30-second videos:

- Retrieval top-1 instead of top-2;
- Retrieval top-1 with a 24-latent-frame age bound;
- age-bounded Retrieval1 plus one coherent high-motion adjacent pair;
- sink3 with an expanded 11-frame-equivalent budget;
- sink3 with a budget-matched 9-frame reallocation.

The three Retrieval cells were clean. Both sink3 cells produced polygon noise:
all 360 heads captured the complete three-frame opening block as a
time-synchronised static sink, leaving no dynamic recent frames. The sink3
cells are retired, a runtime guard now rejects this layout before generation,
and v120 no longer exposes a sink3 candidate.

The default cache remains sink1 plus a content-selected middle bank plus
recent frames. MotionPair1 occupies two adjacent latent frames, not one.
Runtime audits freeze the exact sink/middle/recent allocation, exclusive cache
ownership, Retrieval ages and selections, motion-pair lifecycle, original
position sidecars, and decoded video properties.

The v120 aggregate results show Ours above SF in DINO consistency, drift,
raw long-range clip2clip, aesthetic quality, and imaging quality. Against PF,
Ours is close in imaging and drift but lower in DINO and raw long-range
consistency. Human review reports better visual behavior than the aggregate
ranking suggests. This is currently treated as a metric-alignment question,
not as evidence that Ours quantitatively beats PF.

The standard VBench subject/background fusion compresses high-range
clip2clip differences, while consistency and smoothness can reward static or
repetitive behavior. The missing dynamic-degree result and conflicting
auxiliary loop tables must be resolved before the paper story is frozen.

The latest v116 interpretation is in `docs/118_v116_review_results.md`.
The exact v119 allocations, four-node commands, v120 SF/PF/ours runner, model
paths, VBench commands, and decision rules are in
`docs/119_candidate_refinement_and_moviebench32_runbook.md`. The document
index is `docs/15_lifecache_doc_index.md`. The corrected historical-trick
ledger, post-selection ablation queue, and conditional paper story are in
`docs/120_post_selection_trick_ledger_and_paper_story.md`.
The sink3 code audit, fail-closed fix, safe v120 commands, and split VBench
merge procedure are in
`docs/121_v119_sink3_bugfix_and_v120_safe_launch.md`.
The v120 aggregate tables are in `docs/122` and `docs/123`. Their corrected
interpretation, integrity blockers, no-regeneration follow-up, and claim
boundaries are in `docs/124_v120_metric_human_alignment_audit.md`.
The final 10-hour generation/evaluation matrix, exact four-node commands,
model locations, frozen comparison assembly, and paper decision rule are in
`docs/125_v125_moviebench128_final_candidate_runbook.md`.
The evidence and compute rationale for the eight-method expansion are in
`docs/126_v125_eight_method_quality_expansion.md`.
The completed v125 six-dimension results are in
`docs/128_v125_vbench_long_results.md`.

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

- **Diagnostic binary roles:** the frozen 304/56 map captures Anchor-like and
  Veil-like extremes while leaving Wave mixed. It is useful for cache search
  but comes from an absolute QK-sign statistic that is not shift invariant;
  it is not yet a paper-ready classifier.
- **Supportive memory:** sink1 + TemporalPrototype4 + recent4 is the current
  reference. It compresses older context into four online temporal
  prototypes; the unsafe all-head sink3 warm start remains retired.
- **Suppressive memory:** sink1 + bounded Retrieval1 + recent7 is the selected
  base. v129 tests whether absolute similarity and top-1/top-2 margin can
  abstain from uncertain old-state reads; a second candidate keeps an
  always-available MotionPair1 while gating only Retrieval1.
- **v125 evidence:** Prototype4 + Retrieval1(age<=24) achieved the highest
  Dynamic Degree in the completed 128-prompt screen while remaining close on
  the other measured quality dimensions. v129 tests the gate as a conditional
  improvement, not as an assumed promotion.
- **Current comparison:** reuse validated v125 SF and no-gate Ours videos,
  generate the two gated candidates plus Deep/Rolling/LongLive, and rerun
  prompt-correct VBench-Long. PF regeneration and A-B-A are outside the
  critical path.
- **Deferred ablations:** all-head, random/inverted labels, role count and
  capacity curves are run only after the main cache is selected.
- **Exclusive ownership:** explicit composition is the only owner of sink,
  middle, and recent. The legacy PF dynamic-history path is disabled.
- **Fail-closed evidence:** frozen hashes, exact map counts, runtime routes,
  actual frame ids, overlap/budget checks, role-event features, and decoded
  video properties must pass before a result is accepted.
- **Paper gate:** a successful cache must later be paired with an independent
  shift-invariant binary classifier and threshold/random/inverted controls.
  The old absolute-sign map must not be presented as the final discovery
  method.

PF/SF provide the inference base, cache composition, and dynamic-RoPE
infrastructure and must be cited. EF motivates coherent snapshot selection;
LongLive-RAG motivates bounded non-recent retrieval; the internal
Flash-VAReason notes motivate sparse core-token coverage. These sources must
be cited as inspiration. The possible contribution is a validated binary
role criterion plus role-conditioned cache lifecycle under one budget, not
ownership of prior snapshot, retrieval or token-compression components.

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
|   |-- 104_v100_responsive_event_cache_and_aba_fast_screen.md
|   |-- 105_v101_paper_ablation_after_fast_screen.md
|   |-- 106_v100_fast_screen_review_results.md
|   |-- 107_polygon_noise_rootcause_recovery_and_paper_gate.md
|   |-- 109_legacy_v98_suppressive_cache_1video_screen.md
|   |-- 110_v109_review_results.md
|   |-- 111_nonperiodic_role_event_cache_screen.md
|   |-- 112_v112_moviebench32_promotion_and_vbench.md
|   |-- 114_v111_motion_pair2_rerun_review.md
|   |-- 115_v115_role_memory_design_and_1video_screen.md
|   |-- 116_v115_review_results.md
|   |-- 116_v116_moviebench16_evaluation_runbook.md
|   |-- 117_post_v115_targeted_candidate_plan.md
|   |-- 118_v116_review_results.md
|   |-- 119_candidate_refinement_and_moviebench32_runbook.md
|   |-- 120_post_selection_trick_ledger_and_paper_story.md
|   |-- 120_v119_review_and_sink3_bug.md
|   |-- 121_v119_sink3_bugfix_and_v120_safe_launch.md
|   |-- 122_v120_moviebench32_results.md
|   |-- 123_v120_vbench_analysis_dino_and_pf_alignment.md
|   |-- 124_v120_metric_human_alignment_audit.md
|   |-- 125_v125_moviebench128_final_candidate_runbook.md
|   `-- 126_v125_eight_method_quality_expansion.md
|-- prompts/
|   |-- lifecache_v3_calibration_complex_12.txt
|   |-- lifecache_v3_single_long_complex_12.txt
|   |-- moviegenbench_diverse16.json
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
|   |-- run_v119_candidate_refinement_1video.py
|   |-- run_v120_moviebench32_main.py
|   |-- run_v120_vbench_long.sh
|   |-- merge_v120_vbench_summaries.py
|   |-- analyze_v120_paired_metrics.py
|   |-- run_v125_moviebench128_main.py
|   |-- prepare_v125_moviebench128_comparison.py
|   |-- prepare_v125_vbench_splits.py
|   |-- run_v125_moviebench128_10h.sh
|   |-- run_v125_vbench_long.sh
|   |-- merge_v125_vbench_long_parts.py
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
|   |-- run_v100_fast_selection_1video.py
|   |-- run_v101_paper_ablation_4node.py
|   |-- run_v107_polygon_rootcause_1video.py
|   |-- run_v111_role_event_cache_1video.py
|   |-- run_v112_role_event_cache_32prompt.py
|   |-- run_v112_vbench_long.sh
|   |-- analyze_v111_role_event_traces.py
|   |-- postprocess_v101_paper_ablation.sh
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

The current v125 paper-scale path is implemented:

- `run_v125_moviebench128_main.py`: eight-method, 1,024-video generation with
  frozen Qwen-rewrite prompt, map, implementation, and decoded-video contracts.
- `prepare_v125_moviebench128_comparison.py`: fail-closed eight-method
  comparison assembly with exact prompt and generation-contract checks.
- `prepare_v125_vbench_splits.py`: one-time atomic two-second clip splitting,
  preventing concurrent VBench dimensions from mutating the same input tree.
- `run_v125_vbench_long.sh`: 48 method-by-dimension jobs, one process per GPU,
  RAFT/AMT preflight, resumable markers, and complete dynamic-degree coverage.
- `merge_v125_vbench_long_parts.py`: strict dimension merge, provenance table,
  and machine-readable/CSV/Markdown summaries.

The supporting v119/v120 path and result-analysis safeguards are implemented:

- `run_v119_candidate_refinement_1video.py`: bounded Retrieval1,
  Retrieval1+MotionPair1, and retired sink3 provenance cells.
- `run_v120_moviebench32_main.py`: split SF/PF/Ours generation, frozen
  32-prompt contracts, exact resume, publication, and decoded-video audit.
- `run_v120_vbench_long.sh` and `merge_v120_vbench_summaries.py`:
  six-dimension evaluation and isolated-result merge.
- `analyze_v120_paired_metrics.py`: fail-closed per-prompt VBench and
  comprehensive analysis with confidence intervals and W/T/L counts.

The v111/v112 non-periodic cache path and fail-closed diagnostics are
implemented:

- `pyramidkv/role_event.py`: semantic-coverage landmark memory and coherent
  high-motion pair memory with bounded content-driven replacement.
- `pyramidkv/factory.py` and `pyramidkv/policy_overrides.py`: per-role
  landmark/motion compositions, shared layer-wide contexts for same-route
  controls, sink1/recent4 or matched recent8 budgets, and exclusive dynamic
  ownership.
- `pyramidkv/adaptive_cache.py`: clean-K/V descriptor and motion calculation,
  role-event traces, bank-state traces, and reset handling.
- `run_v111_role_event_cache_1video.py`: eight frozen 30-second cache cells
  under the exact old-v98 304/56 diagnostic map.
- `analyze_v111_role_event_traces.py`: acceptance, reason, feature, occupancy,
  and period-collapse diagnostics.
- `run_v112_role_event_cache_32prompt.py`: human-gated four-node/32-GPU
  promotion, exact task resume, publication markers, and VBench-ready audit.
- `run_v112_vbench_long.sh`: distributed six-dimension VBench-Long evaluation
  and result collection.
- `run_v107_polygon_rootcause_1video.py`,
  `run_v100_fast_selection_1video.py`, and
  `run_v101_paper_ablation_4node.py`: historical recovery/reproduction
  infrastructure; see docs/107 and docs/111 for their historical commands.

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
docs/124_v120_metric_human_alignment_audit.md
```

Preserve all v120 videos. Push only the small metric JSON and frozen contracts,
then run paired analysis over the existing results:

```bash
python scripts/analyze_v120_paired_metrics.py \
  --vbench sf_native=/path/sf_native/results.json \
  --vbench pf_native=/path/pf_native/results.json \
  --vbench ours_landmark_retrieval_motion=/path/ours/results.json \
  --comprehensive /path/all_results_summary.json \
  --references sf_native pf_native \
  --candidates ours_landmark_retrieval_motion \
  --output-json /path/v120_paired_analysis.json \
  --output-md /path/v120_paired_analysis.md
```

The analyzer fails closed unless every requested metric covers prompt indices
0 through 31. Final cache selection requires paired statistics plus blind
review; MovieGenVideoBench-128 remains the subsequent paper-scale confirmation.

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
