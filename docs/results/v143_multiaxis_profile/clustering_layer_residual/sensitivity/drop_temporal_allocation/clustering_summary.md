# v143 Multi-axis Head Taxonomy

- Accepted split-stable features: `11`
- Coordinate system: `layer_residual`
- Selected k: `None`
- Status: `no_stable_k`
- PF and other published labels are post-hoc references only.
- Functional role names remain blocked until causal cache routing passes.

## k diagnostics

| k | split agreement | split ARI | silhouette | bootstrap ARI | min class | layer-band NMI | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9722 | 0.5164 | 0.5881 | -0.0049 | 0.0222 | 0.0481 | 0 |
| 3 | 0.9694 | 0.7874 | 0.4748 | 0.2648 | 0.0250 | 0.0368 | 0 |
| 4 | 0.8694 | 0.6132 | 0.1507 | 0.4126 | 0.0222 | 0.0280 | 0 |
| 5 | 0.8722 | 0.6192 | 0.1542 | 0.3350 | 0.0028 | 0.0304 | 0 |
| 6 | 0.8444 | 0.6537 | 0.1607 | 0.5089 | 0.0028 | 0.0434 | 0 |
