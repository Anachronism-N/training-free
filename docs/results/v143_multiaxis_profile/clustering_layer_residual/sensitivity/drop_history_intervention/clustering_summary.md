# v143 Multi-axis Head Taxonomy

- Accepted split-stable features: `19`
- Coordinate system: `layer_residual`
- Selected k: `None`
- Status: `no_stable_k`
- PF and other published labels are post-hoc references only.
- Functional role names remain blocked until causal cache routing passes.

## k diagnostics

| k | split agreement | split ARI | silhouette | bootstrap ARI | min class | layer-band NMI | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9667 | 0.8645 | 0.2196 | 0.0942 | 0.2639 | 0.0119 | 0 |
| 3 | 0.9472 | 0.8480 | 0.2436 | 0.7412 | 0.0222 | 0.0297 | 0 |
| 4 | 0.8889 | 0.6927 | 0.1626 | 0.5990 | 0.0222 | 0.0309 | 0 |
| 5 | 0.8889 | 0.7040 | 0.1546 | 0.3206 | 0.0028 | 0.0326 | 0 |
| 6 | 0.9028 | 0.7408 | 0.1384 | 0.4375 | 0.0028 | 0.0460 | 0 |
