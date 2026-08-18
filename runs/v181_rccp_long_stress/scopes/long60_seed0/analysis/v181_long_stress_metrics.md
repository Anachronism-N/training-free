# v181 Long-Stress Paired Analysis

Scope: `long60_seed0`
Decision: `long_horizon_rccp_not_confirmed`
Quality + identity gate: `False`
Identity + motion gate: `False`
Late identity gate: `False`
Dynamic non-regression: `True`

| Window | Control | Metric | Mean delta | CI95 | Win | q |
|---|---|---|---:|---:|---:|---:|
| full | sf_native | official_quality_score | 1.145967 | [0.543282, 1.744926] | 0.664 | 0.0003878 |
| full | sf_native | identity_background | -0.001804 | [-0.003623, 0.000079] | 0.430 | 1 |
| full | sf_native | dynamic_degree | 0.224219 | [0.177083, 0.271094] | 0.695 | 1.774e-12 |
| full | all_recent | official_quality_score | -0.070648 | [-0.591035, 0.350135] | 0.547 | 0.2482 |
| full | all_recent | identity_background | -0.002059 | [-0.003354, -0.001126] | 0.352 | 1 |
| full | all_recent | dynamic_degree | 0.050260 | [0.018750, 0.082031] | 0.484 | 0.0005255 |
| late_half | sf_native | official_quality_score | 1.638129 | [0.899320, 2.523438] | 0.664 | 0.0003878 |
| late_half | sf_native | identity_background | -0.003850 | [-0.006171, -0.001202] | 0.312 | 1 |
| late_half | sf_native | dynamic_degree | 0.319271 | [0.256250, 0.384375] | 0.680 | 9.92e-13 |
| late_half | all_recent | official_quality_score | 0.042036 | [-0.501414, 0.528823] | 0.562 | 0.1583 |
| late_half | all_recent | identity_background | -0.002571 | [-0.003908, -0.001396] | 0.336 | 1 |
| late_half | all_recent | dynamic_degree | 0.061979 | [0.020820, 0.105729] | 0.406 | 0.01983 |

The review queue is capped at four late-window metric-conflict cases.

v181 tests the exact frozen v177 five-head RCCP map on unseen 60-second prompts. Full and late-half endpoints are frozen; this scope does not establish cross-model or scene-switch transfer.
