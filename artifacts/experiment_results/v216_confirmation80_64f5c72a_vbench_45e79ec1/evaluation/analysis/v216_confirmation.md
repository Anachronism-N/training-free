# v216 frozen-candidate confirmation

Primary confirmation uses 80 prompts excluded from v215 selection; these are not historically unseen. One endpoint frozen before generation. Other metrics are descriptive. No automatic publication verdict.

Selected v215 method: headwise_correct
Primary: full/imaging_quality
Paired delta (80 prompts): +0.001231; 95% CI [-0.002066445837033298, 0.004588990992406983]

| Full-video metric | SF | Ours | Ours - SF |
|---|---:|---:|---:|
| quality_without_dynamic_degree | 85.947903 | 86.024550 | +0.076647 |
| official_quality_score | 81.544057 | 81.633525 | +0.089468 |
| identity_background | 0.967286 | 0.967274 | -0.000011 |
| temporal_mechanics | 0.980526 | 0.980748 | +0.000222 |
| semantic_alignment | 0.235660 | 0.236658 | +0.000998 |
| visual_quality | 0.638764 | 0.640589 | +0.001825 |
| dynamic_degree | 0.427500 | 0.429167 | +0.001667 |
| subject_consistency | 0.970766 | 0.970672 | -0.000094 |
| imaging_quality | 0.685255 | 0.686486 | +0.001231 |

The 128-prompt table includes the 48 selection prompts and is NOT an independent confirmation.
Quality uses percentage points; other metrics retain their native scale. core-9 is not complete official Total/Semantic.
At most 5 diagnostic review pairs; not a user preference study.
