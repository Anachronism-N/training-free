# v183 Recovered v180 Paired Analysis

Evidence scope: `exploratory_recovered_generation`
Recommendation: `stop_static_strict5_and_revisit_operator`
Formal RCCP membership claim allowed: `False`

| Comparison role | Metric | Mean delta | CI95 | Win | q (exploratory) |
|---|---|---:|---:|---:|---:|
| end_to_end | official_quality_score | 0.378313 | [0.024032, 0.723459] | 0.586 | 0.135 |
| end_to_end | identity_background | -0.000115 | [-0.001287, 0.001115] | 0.414 | 1 |
| end_to_end | dynamic_degree | 0.091667 | [0.053125, 0.130221] | 0.445 | 1.144e-05 |
| strict5_increment | official_quality_score | -0.164313 | [-0.362118, 0.033699] | 0.398 | 1 |
| strict5_increment | identity_background | -0.000772 | [-0.001315, -0.000233] | 0.406 | 1 |
| strict5_increment | dynamic_degree | 0.005729 | [-0.016667, 0.028646] | 0.242 | 1 |
| all_head_operator | official_quality_score | 0.657568 | [0.337824, 0.988470] | 0.594 | 0.1041 |
| all_head_operator | identity_background | -0.001686 | [-0.002517, -0.000858] | 0.344 | 1 |
| all_head_operator | dynamic_degree | 0.109896 | [0.071875, 0.148958] | 0.461 | 2.022e-07 |
| equal_budget_host_control | official_quality_score | 0.542625 | [0.188057, 0.892008] | 0.594 | 0.1041 |
| equal_budget_host_control | identity_background | 0.000657 | [-0.000608, 0.001908] | 0.492 | 1 |
| equal_budget_host_control | dynamic_degree | 0.085938 | [0.044792, 0.128125] | 0.406 | 0.0005613 |
| sparse_vs_dense_coverage | official_quality_score | -0.821880 | [-1.111420, -0.531749] | 0.344 | 1 |
| sparse_vs_dense_coverage | identity_background | 0.000914 | [0.000198, 0.001677] | 0.570 | 0.2487 |
| sparse_vs_dense_coverage | dynamic_degree | -0.104167 | [-0.140625, -0.070312] | 0.062 | 1 |

The 128-prompt result can compare generated videos, cache operators, and the frozen strict-five candidate against SF. It cannot establish that RCCP chose better heads than count/layer-matched alternatives because the recorded v178 gate is not a real paired metric artifact. All confidence intervals and q-values in this report are exploratory.
