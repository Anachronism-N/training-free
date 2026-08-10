# v171 Offline Demand-gated Counterfactual

Offline gate: **True**

The frozen v170 trace contains 6400 retrieval decisions. Motion deficit triggered 1890 times. Full Query weighting changed 258 v166 choices.

| Candidate | Changed | Deficit changed | Healthy changed | Change rate | Age delta mean |
|---|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_deficitquery1 | 86 | 86 | 0 | 1.3438% | 1.477 |
| ours_middle10_reservoir2_deficitbaseline1 | 180 | 180 | 0 | 2.8125% | 0.956 |

For changed baseline-calibrated decisions, selected historical motion magnitude increased by 0.007984 (local) and 0.005531 (context) on average.

Counterfactual replay proves branch activity and selector sparsity, not video quality; autoregressive generation must be rerun.
