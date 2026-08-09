# v167 State-conditioned Motion Trace Audit

Overall mechanism gate: **True**

| Method | Gate | Changed vs motion-only | State-filter changes | Deficit triggers/ready | State residual p50 | Age p95 | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_staterankmotion1 | True | 65 | 65 | 0/560 | 0.8474826812744141 | 20.0 | 0 |
| ours_middle10_reservoir2_deficitstaterankmotion1 | True | 147 | 70 | 194/560 | 0.8447918891906738 | 19.0 | 0 |

This audit recomputes motion scores, candidate-relative state
ranks, top-half membership, two-scale deficit decisions,
counterfactual/final selections, and atomic reads from logged
primitives. It proves execution, not quality.
