# v167 Corrected VBench Metrics

This report uses mutually exclusive diagnostic groups and the
official VBench Quality Score.

Duplicate clip pairs checked: 1440
Exact duplicate: **True**

| Method | Quality Score | Identity/background | Temporal mechanics | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_multiscalemotion1 | 84.4697 | 0.968511 | 0.971128 | 0.234871 | 0.669228 | 0.787500 |
| ours_middle10_reservoir2_staterankmotion1 | 84.3436 | 0.968273 | 0.970784 | 0.237193 | 0.668478 | 0.779167 |
| ours_middle10_reservoir2_deficitstaterankmotion1 | 84.2618 | 0.967772 | 0.971181 | 0.238005 | 0.668402 | 0.766667 |
| ours_middle10_reservoir2_statemotionpair1_reference | 84.2318 | 0.968029 | 0.971307 | 0.238389 | 0.666877 | 0.766667 |
| ours_middle10_reservoir2_directionmatch1 | 84.2283 | 0.966929 | 0.970755 | 0.236816 | 0.667607 | 0.775000 |
| sf_native | 83.1087 | 0.964873 | 0.975109 | 0.233134 | 0.652623 | 0.650000 |

Paired comparisons use the same 16 prompts and bootstrap the
prompt-level mean. This remains development evidence, not a
held-out paper result.
