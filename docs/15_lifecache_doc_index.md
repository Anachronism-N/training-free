# LifeCache / CEMR / HREM document index

## Current method (read first)

- `docs/67_post_sweep_optimization_and_server_protocol.md`: latest post-sweep corrections, optional episode-local ramp, controlled server matrix, required logs, and go/no-go rules.
- `docs/66_gate_sweep_results_and_review.md`: absolute-threshold sweep metrics and human-review observations; superseded by doc 67 for next-step decisions.
- `docs/65_swift_collision_audit.md`: mechanism-level SWIFT overlap audit and remaining claim boundary.
- `docs/64_related_work_code_provenance_and_claims.md`: authoritative paper/code provenance ledger, related-work collision audit, license status, safe claim language, and future-use priorities.
- `docs/63_hrem_v2_p0_role_calibration.md`: current P0 implementation, role-gate ablation matrix, trace fields, server commands, and decision rules.
- `docs/62_hrem_v2_results_and_iteration.md`: first HREM-v2 server results and iteration priorities.
- `docs/62_aaai_provisional_title_abstract.md`: modification-friendly AAAI registration title, result-free abstract, and post-experiment replacement templates.
- `docs/61_hrem_v2_review_and_runbook.md`: single review entry for the current idea, paper story, implementation map, server usage, log interpretation, and go/no-go criteria.
- `docs/60_hrem_v2_novelty_and_debug_protocol.md`: paper-facing novelty boundary, falsifiable story, debug fields, and server diagnosis loop.
- `docs/59_hrem_v2_evidence_gated_episodic_memory.md`: current method, implementation, causal matrix, server commands, and promotion criteria.
- `docs/58_hrem_v1_results.md`: HREM-v1 result and failure analysis.
- `docs/55_cemr_ceg_full_idea_spec.md`: strongest CEMR/CEG evidence and unresolved episode-selection failure.
- `docs/47_current_idea_full_description.md`: earlier structured-memory direction and experiment history.

The current candidate is HREM-v2. Review `docs/61_hrem_v2_review_and_runbook.md` first. LifeCache-v1 below is retained as the original design history, not the recommended first experiment.

## Original LifeCache-v1 documents

The latest design documents are:

- `docs/11_lifecache_v1_design.md`: concrete LifeCache-v1 method design.
- `docs/12_lifecache_experiment_plan.md`: experiment and ablation plan.
- `docs/13_lifecache_codex_implementation_prompt.md`: implementation prompt for Codex or another coding agent.
- `docs/14_lifecache_design_changelog.md`: record of design corrections from earlier over-complex versions.

The recommended reading order is:

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
