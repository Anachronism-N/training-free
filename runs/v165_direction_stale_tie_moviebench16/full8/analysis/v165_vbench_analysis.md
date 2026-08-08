# v165 VBench-Long Core-9

| Method | History | Temporal | Visual | Dynamic |
|---|---:|---:|---:|---:|
| sf_native | 0.72094 | 0.72780 | 0.65245 | 0.64167 |
| ours_middle10_reservoir2_directionmatch1 | 0.72356 | 0.72610 | 0.66764 | 0.77083 |
| ours_middle10_reservoir2_dirstaletie003 | 0.72371 | 0.72627 | 0.66747 | 0.75000 |
| ours_middle10_reservoir2_dirstaletie005 | 0.72372 | 0.72618 | 0.67211 | 0.76667 |
| ours_middle10_reservoir2_directionfresh1 | 0.72454 | 0.72660 | 0.66919 | 0.76250 |
| ours_middle10_reservoir2_statemotionpair1_reference | 0.72478 | 0.72700 | 0.66708 | 0.76250 |

## Frozen development gates

| Gate | Metric | Delta | Minimum | Pass |
|---|---|---:|---:|---:|
| history_vs_directionmatch | history_consistency | +0.00016 | -0.00300 | True |
| temporal_vs_directionmatch | temporal_quality | +0.00008 | +0.00100 | False |
| visual_vs_directionmatch | visual_quality | +0.00447 | -0.00600 | True |
| dynamic_vs_directionmatch | dynamic_degree | -0.00417 | -0.02000 | True |
| history_vs_sf | history_consistency | +0.00278 | +0.00200 | True |
| dynamic_vs_sf | dynamic_degree | +0.12500 | +0.02000 | True |
| temporal_vs_sf | temporal_quality | -0.00162 | -0.00400 | True |
| visual_vs_sf | visual_quality | +0.01966 | -0.00600 | True |

Development candidate gate: **False**

The metric_promotion_gate is a compatibility field for the shared collector. It selects a v165 development candidate only; the 16 prompts are not a held-out paper comparison.
