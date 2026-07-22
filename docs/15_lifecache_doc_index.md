# LifeCache / CEMR / HREM document index

## Current method (read first)

- `docs/60_hrem_v2_novelty_and_debug_protocol.md`: paper-facing novelty boundary, falsifiable story, debug fields, and server diagnosis loop.
- `docs/59_hrem_v2_evidence_gated_episodic_memory.md`: current method, implementation, causal matrix, server commands, and promotion criteria.
- `docs/58_hrem_v1_results.md`: HREM-v1 result and failure analysis.
- `docs/55_cemr_ceg_full_idea_spec.md`: strongest CEMR/CEG evidence and unresolved episode-selection failure.
- `docs/47_current_idea_full_description.md`: earlier structured-memory direction and experiment history.

The current candidate is HREM-v2. LifeCache-v1 below is retained as the original design history, not the recommended first experiment.

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
