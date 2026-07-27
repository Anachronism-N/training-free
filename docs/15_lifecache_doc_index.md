# LifeCache / CEMR / HREM document index

## Current method (read first)

- `docs/119_candidate_refinement_and_moviebench32_runbook.md`: authoritative
  experiment definition, now corrected by docs/121 after the sink3 failure.
- `docs/121_v119_sink3_bugfix_and_v120_safe_launch.md`: authoritative current
  execution plan. It audits the sink3 opening-block bug, retires unsafe
  candidates, defines fail-closed runtime diagnostics, and gives separate
  SF/PF and ours MovieBench-32 plus VBench-Long commands.
- `docs/120_v119_review_and_sink3_bug.md`: server-side v119 human review and
  first sink3 diagnosis. Use docs/121 for the stricter causal boundary and
  repaired commands.
- `docs/118_v116_review_results.md`: latest 16-prompt evidence.
  Landmark4/MotionPair1 is the most balanced candidate; Retrieval2 is
  competitive but has possible late scale enlargement; the next experiment
  must isolate retrieval strength, age and motion support.
- `docs/117_post_v115_targeted_candidate_plan.md`: authoritative current
  pre-v116 decision. It corrects the interpretation of one-prompt Suppressive
  viability, defines the nine-method MovieBench-16 matrix, defers all-head
  controls to ablation, adds paired metric analysis, and prioritizes the
  remaining capacity/lifecycle/sink experiments.
- `docs/116_v115_review_results.md`: completed human review of all 16 v115
  cells. Landmark4 is the most stable Supportive cache, Prototype4 is the
  strongest alternative, and Suppressive routes remain visually distinct but
  unresolved by one prompt.
- `docs/115_v115_role_memory_design_and_1video_screen.md`: authoritative
  completed one-video cache search design. It keeps the old-v98 304/56 map,
  tests Supportive and Suppressive caches separately, adds temporal
  prototypes, coherent snapshots, bounded retrieval, sparse snapshots and a
  compact motion-pair route, and freezes a 16-cell budget-matched matrix with
  strict runtime traces.
- `docs/114_v111_motion_pair2_rerun_review.md`: completed v111 review.
  Landmark4 is the strongest observed Supportive cache; Motion-pair2 is clean
  only when paired with Landmark support and has no demonstrated gain over
  Suppressive Recent8. This evidence motivates the broader v115 search.
- `docs/116_v116_moviebench16_evaluation_runbook.md`: gated promotion after
  v115 human review. It evaluates seven Suppressive routes under fixed
  Landmark4 support plus two Prototype-Supportive candidates on a frozen
  diverse MovieGenBench-16 subset, publishes separately audited VBench and
  indexed-metric views, and runs paired diagnostics on four nodes.
- `docs/113_v111_review_results_and_bug_record.md`: first v111 server review.
  It records four clean Landmark/Recent outputs, identifies the exact
  Motion-pair2 second-slot fill crash, documents the reviewed fix and targeted
  four-cell rerun, and prevents one-prompt viability from being overstated as
  a role-conditioned gain.
- `docs/111_nonperiodic_role_event_cache_screen.md`: authoritative current
  one-video experiment. It freezes the old-v98 304/56 diagnostic partition,
  removes stride/cyclic/Merge from every candidate, introduces semantic
  landmark and coherent motion-pair memories, defines eight budget-matched
  cells, strict cache/trace audits, human-review gates, and claim boundaries.
- `docs/112_v112_moviebench32_promotion_and_vbench.md`: gated follow-up after
  v111 review. It promotes one selected candidate to MovieGenBench-32 against
  three role-neutral controls on four nodes/32 GPUs, publishes audited video
  directories, and runs six VBench-Long dimensions.
- `docs/110_v109_review_results.md`: corrected v109 human review. All five
  old-v98-map routes are artifact-free and visibly but modestly different;
  the result establishes viability but does not select a final cache.
- `docs/109_legacy_v98_suppressive_cache_1video_screen.md`: historical
  cyclic-carrier diagnostic that made v111 possible. Its cyclic carrier is
  explicitly not the final Supportive mechanism.
- `docs/107_polygon_noise_rootcause_recovery_and_paper_gate.md`:
  earlier recovery diagnosis. It corrects the causal
  overclaim in docs/106, identifies the v100 old-map provenance error,
  rebuilds the intended 33/327 map from frozen artifacts, defines direct
  PF-AR/PF-AW one-video controls, preserves cyclic4 while screening one
  additive motion-event slot, and gates all broad experiments.
- `docs/106_v100_fast_screen_review_results.md`: v100 human-review record.
  Every non-native old-304/56-map cell had polygon noise. Its original
  Wave-to-stride and 33/327 overlap interpretation is corrected by docs/107.
- `docs/104_v100_responsive_event_cache_and_aba_fast_screen.md`:
  historical failed-screen protocol. It freezes the
  tracked old-v98 `304/56` two-role map, factorizes Responsive cache choices,
  adds clean-V motion-event memory and role-aware A-B-A scene episodes, and
  defines the 16-cell one-video screen, traces, commands, review gates,
  deferred broad ablations, provenance, and paper story. Disabled by default.
- `docs/105_v101_paper_ablation_after_fast_screen.md`: superseded old-map
  broad matrix. It is disabled by default and must not be used for paper
  results.
- `docs/103_v100_final_candidate_and_immediate_plan.md`: superseded
  middle-relative/stride-cyclic proposal. Retained as design history; do not
  use its server command as the current experiment.
- `docs/102_v99_smoke_results_and_decision.md`: corrected smoke interpretation.
  Binary stride/cyclic is artifact-free on prompt 0; stride/merge failed, but
  Merge and loss of Wave cyclic were confounded. It defines the remaining
  one-video PF-AW diagnostic.
- `docs/101_v98_middle_relative_profiling_results.md`: profiling protocol,
  reported score statistics, hard gates, and the unresolved impossible
  `33/327` versus `169/172` PF-overlap claim. Raw artifacts, not prose, are
  required for the final cross-tab.
- `docs/100_v99_binary_cache_recovery_and_paper_story.md`: recovery
  protocol. It records the duplicate cache-owner bug, restores the
  quality-tested stride/cyclic binary topology, adds a one-prompt human gate,
  and reuses existing videos. Its pre-result Merge decision tree is superseded
  by docs/102 and docs/103.
- `docs/99_v98_experiment_results.md`: historical v98 measurements. The
  videos/metrics remain valid for those runs, but the causal claim that binary
  heads were cleanly disproved is superseded by docs/100.
- `docs/98_history_polarity_dual_memory_method.md`: current v98 paper
  hypothesis record. It replaces prompt sensitivity as the static classifier with a
  natural-zero history-polarity split, uses neutral labels `10/11`, defines
  Supportive hybrid stride+cyclic memory and Suppressive compressed memory,
  and records the exact difference from PF.
- `docs/99_v98_32gpu_runbook_and_paper_plan.md`: four-node/eight-GPU-per-node
  MovieGenBench-32 and MovieGenBench-128 commands, model paths, strict debug
  contracts, post-processing, review gates, and result-dependent paper plan.
- `docs/97_v97_experiment_results.md`: v97 result record. Its corrected head
  captures remain diagnostic, but its absolute-sign score is not a valid input
  to corrected v98; human review also found polygon noise in all binary
  generation cells.
- `docs/97_score_artifact_threshold_pf_merge_experiment.md`: current v97
  correctness reset and 10-hour protocol. It fixes the v96 layer-index
  aliasing bug, saves immutable unthresholded per-head scores, applies manual,
  GMM/Otsu, sign-rate, and PF-derived binary classifications offline, and
  runs a 16-cell MovieBench-32 screen with strict map and runtime-policy
  traces.
- `docs/96_qk_threshold_binary_cache_method_and_experiment.md`: current v96
  binary-head candidate, now superseded for learned-map evidence by v97. It
  corrects the PF QK/sign interpretation, discovers
  Prompt-Stable/Prompt-Responsive heads with a data-derived GMM threshold
  rather than a PF Anchor quota, reports post-hoc PF overlap and Wave sign
  statistics, and factorizes binary membership from merge/cyclic/recent cache
  policies in a 16-GPU MovieGenBench-32 screen.
- `docs/93_moviebench_10h_128_and_head32_plan.md`: current execution protocol.
  It turns the partial v90/v92 evidence into a MovieBench-128 main comparison
  and a separate MovieBench-32 causal head-classification matrix, with
  per-prompt matched reseeding, parallel VBench/comprehensive metrics, blind
  review, debug checks and result-dependent decision branches.
- `docs/92_prompt_contrastive_binary_cache_and_uniqueness_plan.md`: current
  differentiable follow-up. It corrects the interpretation of v86
  `pf_binary_balanced`, defines an actual Anchor-versus-(Wave+Veil) read
  topology and a prompt-intervention partition, and provides the 16-GPU
  factorized screen, coherent-snapshot ablation, debug contract and paper
  decision branches.
- `docs/90_post_v86_analysis_and_v90_experiment.md`: authoritative post-v86
  conclusion, corrected claim boundary, v78 matched-seed requirement, v90 weak
  priority hypothesis, 16-GPU matrix, VBench commands and promotion gates.
- `docs/91_transitioncache_paper_story.md`: paper-writing blueprint with the
  central question, contribution claims, related-work boundary, figures,
  experiment structure, abstract template and result-dependent story branches.
- `docs/89_v86_human_review_and_combined_analysis.md`: v86 human review;
  v78/PF-binary are strongest, learned-role cells duplicate subjects, and
  inverse labels create physics violations.
- `docs/88_transitioncache_v86_partial_results.md`: complete 16-prompt DINO
  table appended after the original partial report; learned role conditioning
  is reproducible but not causally superior.
- `docs/87_transitioncache_method_and_16x30s_protocol.md`: authoritative
  pre-result v86 method definition and protocol; superseded by docs/90 for
  current claims and experiments.
- `docs/86_current_idea_and_role_transition_plan.md`: authoritative current
  idea as of the previous three-prompt plan; its experimental protocol is
  superseded by docs/87.
- `docs/85_comprehensive_human_review_v81_v82.md`: latest human review. v78 and
  PF are best; ProbeCache direct archive retrieval causes non-ID hallucinations;
  PF-binary preserves substantial camera motion.
- `docs/84_probecache_v81_v82_comprehensive_results.md`: quantitative v81/v82
  results, multi-seed v78 confirmation, label controls and profile replication.
- `docs/83_probecache_v81_screen_results.md`: first ProbeCache screen metrics
  and incomplete-cell record.
- `docs/80_v78_method_and_ama_veil_head_history.md`: v78 mechanism and the
  historical evidence against semantic/manual head labels.
- `docs/79_cache_transition_v78_screen_results.md`: validated v78 screen,
  temporal-jump result, human review and the selected `full_budget075_p1`
  configuration.
- `docs/78_cache_transition_implementation_and_experiment_plan.md`: original
  transition implementation and server matrix.
- `docs/64_related_work_code_provenance_and_claims.md`: authoritative
  paper/code provenance ledger, collision audit, license status and safe claim
  language.

Recommended reading order:

1. `docs/121_v119_sink3_bugfix_and_v120_safe_launch.md`
2. `docs/120_v119_review_and_sink3_bug.md`
3. `docs/119_candidate_refinement_and_moviebench32_runbook.md`
4. `docs/120_post_selection_trick_ledger_and_paper_story.md`
5. `docs/118_v116_review_results.md`
6. `docs/117_post_v115_targeted_candidate_plan.md`
7. `docs/116_v116_moviebench16_evaluation_runbook.md`
8. `docs/116_v115_review_results.md`
9. `docs/115_v115_role_memory_design_and_1video_screen.md`
10. `docs/114_v111_motion_pair2_rerun_review.md`
11. `docs/113_v111_review_results_and_bug_record.md`
12. `docs/111_nonperiodic_role_event_cache_screen.md`
13. `docs/110_v109_review_results.md`
14. `docs/109_legacy_v98_suppressive_cache_1video_screen.md`
15. `docs/107_polygon_noise_rootcause_recovery_and_paper_gate.md`
16. `docs/100_v99_binary_cache_recovery_and_paper_story.md`
17. `docs/101_v98_middle_relative_profiling_results.md`
18. `docs/98_history_polarity_dual_memory_method.md`
19. `docs/97_v97_experiment_results.md`
20. `docs/64_related_work_code_provenance_and_claims.md`

The immediate step is split v120 execution: finish the isolated SF/PF
baselines, then run `landmark_motion1` and `landmark_retrieval_motion` as an
ours-only 32-prompt set. Evaluate both manifests with six-dimensional
VBench-Long and merge their summaries. Historical tricks remain deferred
until the base cache is selected.
The corrected evidence tiers and the exact one-prompt -> 16-prompt ->
ablation order for those historical tricks are recorded in
`docs/120_post_selection_trick_ledger_and_paper_story.md`.
ProbeCache, Commit Forcing, LifeCache-v3, HREM and CEMR remain negative-result
or design history. LifeCache-v1 below is retained as the original snapshot.

## Historical method checkpoints

- `docs/81_probecache_method_implementation_and_server_plan.md` and
  `docs/82_probecache_10h_followup_experiment_plan.md`: direct archive recall
  branch tested by docs/83-85.
- `docs/74_commit_forcing_research_reset.md` through
  `docs/77_commit_forcing_v76_screen_results.md`: pathwise correction branch
  and its negative multiscale result.
- `docs/68_single_prompt_continuity_recall.md` through
  `docs/73_lifecache_v3_screen_results.md`: typed-memory branch and the
  visually native-equivalent result.
- `docs/59_hrem_v2_evidence_gated_episodic_memory.md` through
  `docs/67_post_sweep_optimization_and_server_protocol.md`: HREM prompt-switch
  branch, results and collision audits.
- `docs/47_current_idea_full_description.md` through
  `docs/58_hrem_v1_results.md`: CEMR/HREM-v1 design history.
- `docs/62_aaai_provisional_title_abstract.md`: modification-friendly AAAI
  registration title and abstract templates.

## Original LifeCache-v1 documents

The original design documents are:

- `docs/11_lifecache_v1_design.md`: concrete LifeCache-v1 method design.
- `docs/12_lifecache_experiment_plan.md`: experiment and ablation plan.
- `docs/13_lifecache_codex_implementation_prompt.md`: implementation prompt
  for Codex or another coding agent.
- `docs/14_lifecache_design_changelog.md`: record of design corrections from
  earlier over-complex versions.

The original reading order is:

1. `docs/14_lifecache_design_changelog.md`
2. `docs/11_lifecache_v1_design.md`
3. `docs/12_lifecache_experiment_plan.md`
4. `docs/13_lifecache_codex_implementation_prompt.md`

The original LifeCache-v1 snapshot was:

```text
LifeCache-v1 =
  token-level compressed memory bank
  + token-level Q-K recall
  + fixed and dynamic anchors
  + motion-specific token cache
  + head-aware active-cache composition
  + region budget / optional region bias
```
