# v142 Output-Causal Head Profiling

- Recommendation: `online_context_conditioned_output_causal_routing`
- Natural profiles: `128`
- Correctness gate: `True`
- Static policy gate: `False`
- Online-policy opportunity: `False`
- Policy split Spearman / label agreement: `0.9661` / `0.9167`
- Validation static-policy regret median: `0.000000`
- Prompt-policy modulation gate: `True`
- Exact-switch / paraphrase policy distance: `0.00192069` / `0.0012503`
- Persistent-A selectivity gate: `False`
- Persistent content selectivity median: `0.000158519`
- Persistent content split Spearman: `0.3395`

## Interpretation Boundary

This experiment measures per-head attention-output approximation error and a
read-only sampled A-episode archive. It does not modify the generated
trajectory and cannot by itself establish a generation-quality improvement.
