# v225 fixed-input replay

Diagnostic repeated evaluation of existing videos, not independent efficacy evidence. One repeat estimates only realized variation. Input contrasts include any remaining evaluation noise; they do not prove all original drift came from encoding. All dimensions/windows retained.

All values below use 0-100 points, full-video window; paired CIs are descriptive, not an acceptance gate.

| Metric | Original-input effect | Resplit-input effect | Repeat effect | Input change in effect | Repeat change in effect |
|---|---:|---:|---:|---:|---:|
| subject_consistency | +0.009623 | +0.001525 | +0.003178 | -0.004875 | -0.006445 |
| background_consistency | -0.022959 | -0.015720 | -0.023261 | +0.007390 | -0.000302 |
| temporal_flickering | +0.018799 | +0.018812 | +0.018799 | +0.000013 | +0.000000 |
| motion_smoothness | +0.008706 | +0.008813 | +0.008707 | +0.000107 | +0.000000 |
| overall_consistency | +0.064425 | +0.058862 | +0.064425 | -0.005563 | +0.000000 |
| dynamic_degree | -1.458333 | -1.041667 | -1.458333 | +0.416667 | +0.000000 |
| aesthetic_quality | -0.037980 | -0.021947 | -0.037980 | +0.016033 | +0.000000 |
| imaging_quality | +0.122324 | +0.120611 | +0.122324 | -0.001712 | +0.000000 |
| temporal_style | +0.064425 | +0.058862 | +0.064425 | -0.005563 | +0.000000 |
| official_quality_score | -0.089855 | -0.055491 | -0.091080 | +0.034977 | -0.001224 |

Do not pick the best replay or merge repeated scores into a larger prompt count.
No generation or additional manual review is authorized by this diagnostic.
