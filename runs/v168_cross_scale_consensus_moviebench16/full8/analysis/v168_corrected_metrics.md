# v168 Corrected VBench Decision

Mechanism gate: **False**
Recommendation: **do_not_scale; inspect_the_automatic_failure_cases**
Selected candidate: `None`

| Method | Quality | Identity/background | Temporal mechanics | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_multiscalemotion1 | 84.4361 | 0.968516 | 0.971133 | 0.234877 | 0.669155 | 0.783333 |
| ours_middle10_reservoir2_staterankmotion1 | 84.2455 | 0.968233 | 0.970784 | 0.237240 | 0.668456 | 0.766667 |
| ours_middle10_reservoir2_directionmatch1 | 84.1935 | 0.966810 | 0.970755 | 0.236760 | 0.667663 | 0.770833 |
| ours_middle10_reservoir2_multiscalepareto1 | 83.9481 | 0.967246 | 0.971060 | 0.236716 | 0.667636 | 0.733333 |
| ours_middle10_reservoir2_multiscaleconsensus1 | 83.8487 | 0.967382 | 0.970587 | 0.236183 | 0.668699 | 0.720833 |
| sf_native | 83.0370 | 0.964832 | 0.975111 | 0.233162 | 0.652428 | 0.641667 |

The decision uses mutually exclusive diagnostic groups and the official VBench Quality Score. Human review is capped at two automatically selected prompts.
