# v204 Existing-Video SF Motion Audit

- Recommendation: `no_existing_method_has_sf_motion_quality_signal`
- Operator candidates for v201: `[]`
- Manual review required: `False`
- Paper claim ready: `False`

| Candidate | Residual motion delta | 95% CI | Directional | Strong | Corrected quality NI | Safety |
|---|---:|---|---:|---:|---:|---:|
| all_recent | +0.004795 | [+0.002824, +0.006794] | False | False | False | False |
| rccp_matched | +0.005573 | [+0.003810, +0.007216] | False | False | False | False |
| all_coverage | +0.012408 | [+0.010379, +0.014495] | False | False | False | False |

v204 reuses exploratory v183 videos to test whether a cache operator changes camera-compensated local motion without violating corrected quality tolerances versus SF. It can guide v201 but cannot validate a new classifier or serve as fresh confirmatory paper evidence.
