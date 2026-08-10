# v169 Offline Counterfactual

Controlled-change gate: **True**

| Method | Passing | Changed vs v166 | Change rate | Old recall | Conflict changes | Agreement changes | Age median |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_multiscalequeryweighted1 | 518 | 23 | 0.0444 | 177 | 23 | 0 | 10.0 |
| ours_middle10_reservoir2_multiscalebottleneck1 | 518 | 57 | 0.1100 | 188 | 57 | 0 | 10.0 |

The gate only proves that both selectors make bounded, genuine changes while preserving old-history reads. It does not predict video quality.
