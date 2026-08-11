# v172 Relative-Depth Paired Analysis

No universal layer count is selected from this development suite.

| Method | Quality | Identity/background | Temporal | Semantic | Visual | Dynamic | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours_depth_center_1of3_multiscalemotion_reference | 84.4063 | 0.968570 | 0.971127 | 0.234805 | 0.669181 | 0.779167 | True |
| ours_depth_late_1of3_multiscalemotion | 84.0612 | 0.963562 | 0.970148 | 0.235328 | 0.666921 | 0.779167 | True |
| ours_depth_all_multiscalemotion | 84.0310 | 0.962429 | 0.967142 | 0.241129 | 0.660533 | 0.841667 | True |
| ours_depth_center_1of4_multiscalemotion | 83.9644 | 0.966653 | 0.971080 | 0.236338 | 0.663778 | 0.754167 | True |
| ours_depth_center_1of2_multiscalemotion | 83.8376 | 0.966872 | 0.969867 | 0.237949 | 0.671219 | 0.720833 | True |
| ours_depth_center_1of6_multiscalemotion | 83.8298 | 0.967070 | 0.971257 | 0.234394 | 0.667631 | 0.716667 | True |
| ours_depth_interleaved_1of3_multiscalemotion | 83.6556 | 0.965964 | 0.969914 | 0.235494 | 0.665247 | 0.725000 | True |
| ours_depth_early_1of3_multiscalemotion | 83.5038 | 0.962348 | 0.967861 | 0.239031 | 0.666686 | 0.741667 | True |
| sf_native | 83.0694 | 0.964809 | 0.975110 | 0.233090 | 0.652468 | 0.645833 | False |

The center dose curve is 1/6, 1/4, 1/3, and 1/2 of model depth. 
Early/center/late/interleaved placement uses exactly one-third 
of layers. Pairwise prompt bootstrap results are stored in the JSON.

The 16 prompts are adaptive development evidence. A favorable fraction or placement is an operator-specific hypothesis, not a semantic layer class or cross-model result.
