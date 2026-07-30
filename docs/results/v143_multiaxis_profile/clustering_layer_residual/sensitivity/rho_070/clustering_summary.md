# v143 Multi-axis Head Taxonomy

- Accepted split-stable features: `20`
- Coordinate system: `layer_residual`
- Selected k: `None`
- Status: `no_stable_k`
- PF and other published labels are post-hoc references only.
- Functional role names remain blocked until causal cache routing passes.

## k diagnostics

| k | split agreement | split ARI | silhouette | bootstrap ARI | min class | layer-band NMI | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9778 | 0.5751 | 0.6603 | 0.7586 | 0.0222 | 0.0481 | 0 |
| 3 | 0.9333 | 0.7933 | 0.2382 | 0.7359 | 0.0194 | 0.0368 | 0 |
| 4 | 0.9083 | 0.7434 | 0.2093 | 0.6036 | 0.0194 | 0.0382 | 0 |
| 5 | 0.8583 | 0.6934 | 0.2012 | 0.4795 | 0.0139 | 0.0341 | 0 |
| 6 | 0.8500 | 0.6971 | 0.1484 | 0.3422 | 0.0139 | 0.0350 | 0 |
