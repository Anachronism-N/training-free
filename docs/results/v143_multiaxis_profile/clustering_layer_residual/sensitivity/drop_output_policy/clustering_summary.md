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
| 2 | 0.9722 | 0.5529 | 0.5723 | 0.4087 | 0.0250 | 0.0360 | 0 |
| 3 | 0.8750 | 0.6231 | 0.1590 | 0.6570 | 0.0222 | 0.0183 | 0 |
| 4 | 0.8917 | 0.7065 | 0.1274 | 0.5514 | 0.0222 | 0.0388 | 0 |
| 5 | 0.8528 | 0.6850 | 0.1258 | 0.5387 | 0.0167 | 0.0346 | 0 |
| 6 | 0.8139 | 0.6145 | 0.1206 | 0.5009 | 0.0139 | 0.0529 | 0 |
