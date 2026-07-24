# LifeCache / CEMR / HREM document index

## Current method (read first)

- `docs/86_current_idea_and_role_transition_plan.md`: authoritative current
  idea, v78 evidence, v86 role-conditioned write-clock implementation, paper
  story, related-work boundary, 16-GPU commands, traces and promotion gates.
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

1. `docs/85_comprehensive_human_review_v81_v82.md`
2. `docs/84_probecache_v81_v82_comprehensive_results.md`
3. `docs/79_cache_transition_v78_screen_results.md`
4. `docs/86_current_idea_and_role_transition_plan.md`
5. `docs/64_related_work_code_provenance_and_claims.md`

The current validated candidate is v78 Trust-Conditioned Cache Transition.
v86 role-conditioned transition is an experimental extension and must not be
called the final method unless it beats v78 and PF-binary controls. ProbeCache,
Commit Forcing, LifeCache-v3, HREM and CEMR remain as negative-result or design
history. LifeCache-v1 below is retained as the original design snapshot.

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
