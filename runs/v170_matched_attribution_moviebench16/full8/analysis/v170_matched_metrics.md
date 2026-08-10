# v170 Matched Attribution

Decision: **reject_query_weighting_without_additional_manual_review**
Attribution gate: **False**

| Metric | Lane A delta | Lane B delta | Matched effect | Replica noise | Effect/noise | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| official_quality_score | 0.171500 | 0.215567 | 0.193533 | 0.061557 | 3.144 | [-0.222998, 0.628403] |
| identity_background | -0.000840 | -0.000669 | -0.000754 | 0.000268 | -2.810 | [-0.002223, 0.000393] |
| temporal_mechanics | -0.000553 | -0.000536 | -0.000544 | 0.000014 | -39.175 | [-0.002806, 0.001359] |
| semantic_alignment | -0.000736 | -0.001092 | -0.000914 | 0.000299 | -3.056 | [-0.003010, 0.000964] |
| visual_quality | 0.002008 | 0.002158 | 0.002083 | 0.000436 | 4.773 | [-0.001456, 0.005711] |
| dynamic_degree | 0.025000 | 0.029167 | 0.027083 | 0.006250 | 4.333 | [-0.016667, 0.075000] |

The two lanes run each policy sequentially on the same GPU with opposite order. The matched effect is the per-prompt average of the two within-GPU policy deltas.

The v169 blind review preferred v166 in both selected pairs; v170 therefore requests no additional manual review.
