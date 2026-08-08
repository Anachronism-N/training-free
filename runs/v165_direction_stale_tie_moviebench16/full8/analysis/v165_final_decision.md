# v165 Final Development Decision

Primary candidate: `ours_middle10_reservoir2_dirstaletie005`

Mechanism gate: **True**  
Aggregate gate: **False**  
Paired-support gate: **False**  
Candidate safety gate: **False**  
Development candidate gate: **False**

Recommendation: `targeted_review_before_keep_or_reject`

## Frozen aggregate gates

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

## Paired VBench deltas

| Reference | Metric | Mean | 95% CI | Positive prompts |
|---|---|---:|---:|---:|
| ours_middle10_reservoir2_directionmatch1 | history_consistency | +0.00016 | [-0.00113, +0.00132] | 7/16 |
| ours_middle10_reservoir2_directionmatch1 | temporal_quality | +0.00008 | [-0.00140, +0.00136] | 7/16 |
| ours_middle10_reservoir2_directionmatch1 | visual_quality | +0.00447 | [-0.00080, +0.01061] | 12/16 |
| ours_middle10_reservoir2_directionmatch1 | dynamic_degree | -0.00417 | [-0.02083, +0.01250] | 2/16 |
| sf_native | history_consistency | +0.00278 | [+0.00003, +0.00552] | 11/16 |
| sf_native | temporal_quality | -0.00162 | [-0.00510, +0.00152] | 7/16 |
| sf_native | visual_quality | +0.01966 | [+0.01251, +0.02787] | 16/16 |
| sf_native | dynamic_degree | +0.12500 | [+0.00417, +0.26250] | 7/16 |
| ours_middle10_reservoir2_dirstaletie003 | history_consistency | +0.00001 | [-0.00142, +0.00135] | 10/16 |
| ours_middle10_reservoir2_dirstaletie003 | temporal_quality | -0.00009 | [-0.00132, +0.00110] | 9/16 |
| ours_middle10_reservoir2_dirstaletie003 | visual_quality | +0.00464 | [-0.00114, +0.01125] | 11/16 |
| ours_middle10_reservoir2_dirstaletie003 | dynamic_degree | +0.01667 | [-0.00417, +0.03750] | 4/16 |

## Candidate-specific safety

- Prompt 1: edge_density_failure
- Prompt 10: late_motion_collapse
- Prompt 11: background_drift

## Minimal review

- Prompt 10: candidate_safety_flag
- Prompt 11: candidate_safety_flag

This adaptive 16-prompt analysis chooses the next development step. It is not held-out evidence, and its selected review clips cannot be reported as an unbiased human comparison.
