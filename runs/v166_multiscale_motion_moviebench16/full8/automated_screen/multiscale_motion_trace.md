# v166 Multi-scale Motion Trace Audit

Overall mechanism gate: **True**

| Method | Gate | Changed vs legacy | Direction p50 | Magnitude p50 | Selected age p95 | Fallbacks | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_multiscaledir1 | True | 148 | 0.275404654443264 | 0.9039352834440315 | 22.0 | 38 | 0 |
| ours_middle10_reservoir2_multiscalemotion1 | True | 145 | 0.30103421956300735 | 0.9047552766492211 | 22.0 | 37 | 0 |

This audit recomputes magnitude matches, aggregate scores,
gates, counterfactual/final selections, and atomic reads from
the logged cosine and norm primitives. It proves execution,
not quality.
