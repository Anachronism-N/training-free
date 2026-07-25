# LifeCache / CEMR / HREM document index

## Current method (read first)

- `docs/96_qk_threshold_binary_cache_method_and_experiment.md`: current v96
  binary-head candidate. It corrects the PF QK/sign interpretation, discovers
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

1. `docs/96_qk_threshold_binary_cache_method_and_experiment.md`
2. `docs/95_post_v93_dual_axis_phase_cache.md`
3. `docs/93_moviebench_10h_128_and_head32_plan.md`
4. `docs/92_prompt_contrastive_binary_cache_and_uniqueness_plan.md`
5. `docs/90_post_v86_analysis_and_v90_experiment.md`
6. `docs/91_transitioncache_paper_story.md`
7. `docs/89_v86_human_review_and_combined_analysis.md`
8. `docs/88_transitioncache_v86_partial_results.md`
9. `docs/64_related_work_code_provenance_and_claims.md`

The current GPU-pending candidate is the v96 QK-threshold binary cache.
It replaces the PF-count-matched Prompt-Stable definition with a data-derived
threshold and tests Veil-style merge, Wave-style cyclic and recent-only
Responsive policies independently. The v95 Dual-Axis Phase Cache remains the
fallback if the v96 binary taxonomy gates fail. v78
Trust-Conditioned Cache Transition remains one of its optional lifecycle
components. Matched seeds 0-2 show v78 and PF are effectively tied in mean
DINO, so a consistent DINO improvement is not currently supportable. v86 counterfactual
role-conditioned clocks are a negative result: the labels are reproducible but
do not beat v78, PF-binary, inverse and random controls. Remaining v90 cells
test lower-risk weak priorities and lifecycle factors. v92 tests a new
hypothesis: prompt response may be useful for choosing the read timescale even
though the earlier remote-minus-prompt labels were not useful as hard write
clocks. This hypothesis does not replace v78 until inverse/random/replica
controls and human review pass. v93 scales the main comparison to 128
MovieBench prompts and the classifier controls to 32 MovieBench prompts. The
complete head32 results support prompt-intervention scores as an orthogonal
signal, but do not support permanently replacing PF's three-class read
topology. `docs/95` defines the current Dual-Axis Phase Cache candidate and
v95 causal screen: PF steady-state reads, prompt-guided startup history
exposure, trust-qualified middle-state promotion, and weak semantic priority
after trust gates. ProbeCache, Commit Forcing, LifeCache-v3, HREM and CEMR remain as
negative-result or design history. LifeCache-v1 below is retained as the
original design snapshot.

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
