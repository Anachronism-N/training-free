# v181 Long-Stress Paired Analysis

Scope: `long60_seed10000_64`
Decision: `long_horizon_rccp_not_confirmed`
Quality + identity gate: `False`
Identity + motion gate: `False`
Late identity gate: `False`
Dynamic non-regression: `True`

| Window | Control | Metric | Mean delta | CI95 | Win | q |
|---|---|---|---:|---:|---:|---:|
| full | sf_native | official_quality_score | 1.039776 | [0.443735, 1.766643] | 0.656 | 0.02529 |
| full | sf_native | identity_background | -0.000298 | [-0.002186, 0.002056] | 0.328 | 0.9997 |
| full | sf_native | dynamic_degree | 0.148958 | [0.082292, 0.214596] | 0.609 | 0.001605 |
| full | all_recent | official_quality_score | 0.057606 | [-0.266215, 0.380945] | 0.547 | 0.5323 |
| full | all_recent | identity_background | -0.000815 | [-0.001417, -0.000168] | 0.297 | 0.9997 |
| full | all_recent | dynamic_degree | 0.033854 | [-0.008854, 0.079167] | 0.484 | 0.1427 |
| late_half | sf_native | official_quality_score | 1.624201 | [0.905360, 2.433088] | 0.719 | 0.001605 |
| late_half | sf_native | identity_background | -0.001705 | [-0.003963, 0.000890] | 0.328 | 0.9997 |
| late_half | sf_native | dynamic_degree | 0.234375 | [0.153125, 0.316693] | 0.609 | 9.133e-05 |
| late_half | all_recent | official_quality_score | 0.145949 | [-0.376036, 0.654064] | 0.516 | 0.6755 |
| late_half | all_recent | identity_background | -0.000740 | [-0.001528, 0.000075] | 0.359 | 0.9997 |
| late_half | all_recent | dynamic_degree | 0.032292 | [-0.035417, 0.101042] | 0.375 | 0.5585 |

The review queue is capped at four late-window metric-conflict cases.

v181 tests the exact frozen v177 five-head RCCP map on unseen 60-second prompts. Full and late-half endpoints are frozen; this scope does not establish cross-model or scene-switch transfer.
