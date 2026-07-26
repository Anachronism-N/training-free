# v97 Policy Trace Audit

- strict pass: `True`
- methods: `16`
- events: `38400`

| Method | Status | Events | Layers | Strategies | Failures |
|---|---|---:|---|---|---:|
| prompt_tau_0p0_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:1200, StrideStrategy:1200 | 0 |
| prompt_tau_0p5_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:680, StrideStrategy:1720 | 0 |
| prompt_tau_1p0_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:400, StrideStrategy:2000 | 0 |
| prompt_tau_1p5_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:360, StrideStrategy:2040 | 0 |
| prompt_tau_2p0_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:240, StrideStrategy:2160 | 0 |
| prompt_tau_1p0_cyclic | nominal | 2400 | [0, 7, 15, 23, 29] | CyclicStrategy:400, StrideStrategy:2000 | 0 |
| prompt_tau_1p0_recent | nominal | 2400 | [0, 7, 15, 23, 29] | StrideStrategy:2000 | 0 |
| prompt_tau_1p0_random_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:400, StrideStrategy:2000 | 0 |
| prompt_tau_1p0_reversed_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:400, StrideStrategy:2000 | 0 |
| sign_rpos_0p5_stride_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:240, StrideStrategy:2160 | 0 |
| pf_ar_stride_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:1160, StrideStrategy:1240 | 0 |
| pf_aw_stride_merge | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:200, StrideStrategy:2200 | 0 |
| pf_native | nominal | 2400 | [0, 7, 15, 23, 29] | CyclicStrategy:960, MergeStrategy:200, StrideStrategy:1240 | 0 |
| pf_anchor_extended_recent | nominal | 2400 | [0, 7, 15, 23, 29] | CyclicStrategy:960, MergeStrategy:200 | 0 |
| pf_wave_extended_recent | nominal | 2400 | [0, 7, 15, 23, 29] | MergeStrategy:200, StrideStrategy:1240 | 0 |
| pf_veil_extended_recent | nominal | 2400 | [0, 7, 15, 23, 29] | CyclicStrategy:960, StrideStrategy:1240 | 0 |
