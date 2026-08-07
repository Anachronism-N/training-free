# v164 Direction/Freshness Trace Audit

Overall mechanism gate: **True**

| Method | Gate | Compatible reads | Direction rejects | Fallbacks | Freshness changes | Age p95 | Budget violations | Contract failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_directionmatch1 | True | 517 | 347 | 39 | 0 | 22.0 | 0 | 0 |
| ours_middle10_reservoir2_directionfresh1 | True | 505 | 343 | 46 | 95 | 19.0 | 0 | 0 |

The audit recomputes every candidate score and selected pair. 
It establishes mechanism execution only, not video-quality improvement.
