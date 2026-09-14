# v200 Head x Denoising Phase x AR-Horizon Audit

- Recommendation: `advance_head_phase_horizon_to_runtime_design`
- Horizon candidates: `['landmark', 'retrieval']`
- New videos required: `False`
- Manual review required: `False`
- The frozen v189 map is unchanged.

| Operator | Horizon gate | Interaction rho | Primary delta | CI95 | Win | Time-perm p |
|---|---:|---:|---:|---:|---:|---:|
| landmark | True | 0.8852 | 0.368739 | [0.334631, 0.400795] | 1.000 | 0.0002 |
| retrieval | True | 0.9355 | 0.406086 | [0.390259, 0.421656] | 1.000 | 0.0002 |

v200 is a cross-fit shadow-readout audit. It can authorize a new runtime design, but it cannot establish generated-video quality or replace a causal equal-exposure generation screen.
