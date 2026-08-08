# v165 Direction Stale-Tie Trace Audit

Overall mechanism gate: **True**

| Method | Gate | Tie uses | Changed | Direction loss p95 | Age gain mean | Selected age p95 | Fallbacks | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_dirstaletie003 | True | 41 | 36 | 0.028898999999999994 | 8.555555555555555 | 22.0 | 44 | 0 |
| ours_middle10_reservoir2_dirstaletie005 | True | 68 | 57 | 0.04627560000000001 | 8.982456140350877 | 22.0 | 40 | 0 |

The audit independently recomputes the direction maximum, stale
gate, near-equivalent candidate set, newest tie choice, and read.
It establishes mechanism execution only, not video quality.
