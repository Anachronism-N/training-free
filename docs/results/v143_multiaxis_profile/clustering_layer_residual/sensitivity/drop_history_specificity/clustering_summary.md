# v143 Multi-axis Head Taxonomy

- Accepted split-stable features: `22`
- Coordinate system: `layer_residual`
- Selected k: `None`
- Status: `no_stable_k`
- PF and other published labels are post-hoc references only.
- Functional role names remain blocked until causal cache routing passes.

## k diagnostics

| k | split agreement | split ARI | silhouette | bootstrap ARI | min class | layer-band NMI | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9972 | 0.9685 | 0.4552 | 0.0955 | 0.0500 | 0.0170 | 0 |
| 3 | 0.9583 | 0.7555 | 0.4559 | 0.4134 | 0.0417 | 0.0444 | 0 |
| 4 | 0.9194 | 0.7755 | 0.1315 | 0.5099 | 0.0222 | 0.0389 | 0 |
| 5 | 0.9333 | 0.8413 | 0.1758 | 0.5264 | 0.0028 | 0.0345 | 0 |
| 6 | 0.9028 | 0.7600 | 0.1232 | 0.4549 | 0.0028 | 0.0317 | 0 |
