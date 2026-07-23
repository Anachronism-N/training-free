# LifeCache / CEMR / HREM document index

## Current method (read first)

- `docs/76_multiscale_commit_bank_design_and_server_plan.md`: current
  Commit Forcing v2 design. Defines the explicit origin/compressed/recent
  lifecycle, reliability-weighted motion-compatible consolidation,
  motion-adaptive readout, trajectory-coupled re-noising, 16-GPU screen,
  debug checks, provenance, and predeclared promotion gates.
- `docs/75_commit_forcing_v74_screen_results.md`: completed Commit Forcing v1
  metrics and human review. Reliability-gated hybrid correction visibly
  improves over native and fixed origin, but style simplification, motion
  freezing, and acceleration-like jumps remain.
- `docs/74_commit_forcing_research_reset.md`: current research reset after the
  visually negative LifeCache-v3 result. Defines reliability-gated state
  admission, the bounded origin/trusted/native-recent lifecycle, pathwise
  reference correction, independent TTC/PF baselines, implementation files,
  16-GPU commands, trace invariants, and paper go/no-go rules.
- `docs/73_lifecache_v3_screen_results.md`: completed 16-cell result and human
  review. The intervention is mechanically active but visually equivalent to
  native SF; this is the evidence for stopping side-output fusion tuning.
- `docs/72_lifecache_v3_post_review_optimization.md`: newest response to the
  first 30-second human review; historical design that was tested by docs/73
  and is no longer the recommended method.
- `docs/71_human_review_and_code_alignment.md`: historical first-run evidence:
  native SF degrades, PF preserves identity but can jump, old ours cells were
  functionally native, and Echo OOMed on a partially occupied H20.
- `docs/70_lifecache_v3_typed_memory_intervention_routing.md`: proposed
  half-life-aware anchor/summary/recent cache, non-handcrafted offline/online
  intervention routing, explicit update rules, 16-GPU matrix, diagnostics and
  promotion criteria. This is the newest implementation candidate and is not
  yet an experimentally validated replacement for HREM-v2.
- `docs/69_paper_alignment_canonical_experiments.md`: current top-paper alignment, single-prompt-first method story, PF/Echo distinction, canonical 3x30s generation/blind-review/metric commands, model locations, debug return package, and academic-integrity rules.
- `docs/68_single_prompt_continuity_recall.md`: latest correction that makes single-prompt long video a primary task, adds explicit intra-episode temporal recall, a 30-second controlled matrix, trace invariants, provenance, and go/no-go rules.
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

The current candidate is Commit Forcing v2. Review docs/75 for the validated v1
result, then docs/76 for the current design and server protocol. Docs/74 and 73
record the research reset and its negative experimental motivation.
Docs/69-72 describe the superseded LifeCache/HREM route and remain necessary
for baseline history and negative-result reporting.
LifeCache-v1 below is retained as the original design history, not the
recommended first experiment.

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
