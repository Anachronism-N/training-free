# v169 Corrected VBench Decision

Mechanism gate: **True**
Recommendation: **review_two_prompts_before_decision**
Selected candidate: `None`
Review candidate: `ours_middle10_reservoir2_multiscalequeryweighted1`

| Method | Quality | Identity/background | Temporal mechanics | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| ours_middle10_reservoir2_multiscalequeryweighted1 | 84.5851 | 0.967723 | 0.970587 | 0.233964 | 0.671420 | 0.804167 |
| ours_middle10_reservoir2_multiscalemotion1 | 84.4386 | 0.968583 | 0.971130 | 0.234806 | 0.669164 | 0.783333 |
| ours_middle10_reservoir2_directionmatch1 | 84.2321 | 0.966958 | 0.970756 | 0.236824 | 0.667686 | 0.775000 |
| ours_middle10_reservoir2_multiscalebottleneck1 | 84.1977 | 0.966613 | 0.970348 | 0.237493 | 0.667106 | 0.779167 |
| ours_middle10_reservoir2_multiscalepareto1 | 83.9089 | 0.967220 | 0.971063 | 0.236694 | 0.667424 | 0.729167 |
| sf_native | 83.0347 | 0.964817 | 0.975108 | 0.233165 | 0.652376 | 0.641667 |

The decision counts duplicate ViCLIP output once. When neither candidate reaches the frozen near-frontier, no manual review is requested.
