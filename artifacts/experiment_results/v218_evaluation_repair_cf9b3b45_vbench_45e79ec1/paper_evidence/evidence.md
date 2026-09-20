# ICASSP LPHC evidence packet

Status: development_only_wait_for_manual_method_freeze

No automatic publication decision; primary claims follow the frozen endpoint and observed tradeoffs.

| Development method | Control | Window / metric | Delta | 95% CI |
|---|---|---|---:|---|
| fifo_correct | sf_fifo21 | full / imaging_quality | -0.001209 | [-0.006277, 0.004575] |
| fifo_correct | fifo_random | full / imaging_quality | -0.000970 | [-0.004340, 0.002251] |
| fifo_correct | sf_fifo21 | full / official_quality_score | -0.114860 | [-0.383992, 0.149240] |
| fifo_correct | fifo_random | full / official_quality_score | -0.140557 | [-0.338594, 0.047463] |
| fifo_correct | sf_fifo21 | late_half / subject_consistency | -0.000392 | [-0.002571, 0.001571] |
| fifo_correct | fifo_random | late_half / subject_consistency | -0.001168 | [-0.003135, 0.000594] |
| headwise_correct | sf_fifo21 | full / imaging_quality | +0.004358 | [-0.002571, 0.012183] |
| headwise_correct | fifo_random | full / imaging_quality | +0.004597 | [0.000429, 0.009109] |
| headwise_correct | sf_fifo21 | full / official_quality_score | -0.000727 | [-0.229148, 0.216252] |
| headwise_correct | fifo_random | full / official_quality_score | -0.026424 | [-0.244371, 0.175070] |
| headwise_correct | sf_fifo21 | late_half / subject_consistency | +0.001090 | [-0.000988, 0.003117] |
| headwise_correct | fifo_random | late_half / subject_consistency | +0.000315 | [-0.001228, 0.001698] |
| centered_correct | sf_fifo21 | full / imaging_quality | +0.001270 | [-0.006245, 0.010088] |
| centered_correct | fifo_random | full / imaging_quality | +0.001509 | [-0.002682, 0.005909] |
| centered_correct | sf_fifo21 | full / official_quality_score | -0.111248 | [-0.326265, 0.068690] |
| centered_correct | fifo_random | full / official_quality_score | -0.136944 | [-0.290075, 0.006579] |
| centered_correct | sf_fifo21 | late_half / subject_consistency | +0.000340 | [-0.001602, 0.002043] |
| centered_correct | fifo_random | late_half / subject_consistency | -0.000436 | [-0.002317, 0.000956] |
| fifo_full_a002 | sf_fifo21 | full / imaging_quality | +0.001482 | [-0.004009, 0.006919] |
| fifo_full_a002 | fifo_full_random | full / imaging_quality | -0.002156 | [-0.006889, 0.002492] |
| fifo_full_a002 | sf_fifo21 | full / official_quality_score | +0.045842 | [-0.216438, 0.295737] |
| fifo_full_a002 | fifo_full_random | full / official_quality_score | -0.077379 | [-0.300056, 0.145405] |
| fifo_full_a002 | sf_fifo21 | late_half / subject_consistency | -0.000005 | [-0.002026, 0.001956] |
| fifo_full_a002 | fifo_full_random | late_half / subject_consistency | -0.002924 | [-0.005389, -0.000754] |

Review queue: 4 new pairs; 0 already reviewed, total budget six. This is diagnosis, not a user study.

- Compact receipts and arithmetic checked; raw media/tensors/VBench parts not re-evaluated.
- v215 and selection_included128 are not independent confirmation.
- Joint seeds use the same 64 prompts with two seeds, not 128 independent observations; no extrapolation to unseen seeds.
- CI spanning zero does not forbid submission but does not establish a reliable improvement.
- Raw temporal_style and overall_consistency are retained as reported, not counted twice in a new score.
- Subject consistency is not person re-identification accuracy; core-9 is not complete Semantic/Total.
