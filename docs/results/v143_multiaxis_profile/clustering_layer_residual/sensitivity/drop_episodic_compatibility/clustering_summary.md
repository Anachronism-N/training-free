# v143 Multi-axis Head Taxonomy

- Accepted split-stable features: `23`
- Coordinate system: `layer_residual`
- Selected k: `None`
- Status: `no_stable_k`
- PF and other published labels are post-hoc references only.
- Functional role names remain blocked until causal cache routing passes.

## k diagnostics

| k | split agreement | split ARI | silhouette | bootstrap ARI | min class | layer-band NMI | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9556 | 0.8268 | 0.2183 | 0.0914 | 0.3139 | 0.0142 | 0 |
| 3 | 0.9472 | 0.8480 | 0.2343 | 0.6942 | 0.0222 | 0.0319 | 0 |
| 4 | 0.8889 | 0.6784 | 0.1406 | 0.4710 | 0.0222 | 0.0269 | 0 |
| 5 | 0.9111 | 0.7693 | 0.1375 | 0.3893 | 0.0028 | 0.0299 | 0 |
| 6 | 0.8972 | 0.7281 | 0.1328 | 0.4288 | 0.0028 | 0.0422 | 0 |
