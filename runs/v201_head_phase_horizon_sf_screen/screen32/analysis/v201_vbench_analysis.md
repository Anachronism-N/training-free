# v201 Head x Phase x AR-Horizon VBench-Long

| Method | Quality | Quality w/o Dynamic | Identity/background | Temporal | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|
| sf_native | 86.5227 | 86.5227 | 0.96865 | 0.97927 | 0.23071 | 0.65948 | 1.00000 |
| landmark_all_recent | 86.3627 | 86.3627 | 0.97053 | 0.97340 | 0.24098 | 0.66887 | 1.00000 |
| landmark_all_coverage | 85.8638 | 85.8638 | 0.96775 | 0.97014 | 0.24111 | 0.66579 | 1.00000 |
| landmark_static_top10 | 86.0319 | 86.0319 | 0.96955 | 0.97167 | 0.24286 | 0.66466 | 1.00000 |
| landmark_horizon_top10 | 85.9581 | 85.9581 | 0.96957 | 0.97079 | 0.23916 | 0.66468 | 1.00000 |
| landmark_horizon_shift_top10 | 85.9722 | 85.9722 | 0.96943 | 0.97066 | 0.23957 | 0.66581 | 1.00000 |
| retrieval_all_recent | 86.3714 | 86.3714 | 0.97066 | 0.97340 | 0.24100 | 0.66898 | 1.00000 |
| retrieval_all_coverage | 85.7731 | 85.7731 | 0.96746 | 0.96973 | 0.23684 | 0.66458 | 1.00000 |
| retrieval_static_top10 | 86.2913 | 86.2913 | 0.96988 | 0.97362 | 0.24038 | 0.66678 | 1.00000 |
| retrieval_horizon_top10 | 86.3406 | 86.3406 | 0.97069 | 0.97313 | 0.23906 | 0.66883 | 1.00000 |
| retrieval_horizon_shift_top10 | 86.2772 | 86.2772 | 0.97028 | 0.97333 | 0.23924 | 0.66667 | 1.00000 |

Aggregate metrics alone do not validate AR-horizon routing. The paired v201 decision first tests efficacy against canonical SF; static-top10 and horizon-shift then provide separate mechanism attribution. Dynamic Degree is never used for promotion.
