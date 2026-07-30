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
| 2 | 0.9639 | 0.8531 | 0.2329 | 0.1108 | 0.2556 | 0.0134 | 0 |
| 3 | 0.9444 | 0.8380 | 0.2579 | 0.6820 | 0.0222 | 0.0318 | 0 |
| 4 | 0.9056 | 0.7178 | 0.1613 | 0.4603 | 0.0194 | 0.0299 | 0 |
| 5 | 0.9028 | 0.7111 | 0.1830 | 0.3249 | 0.0028 | 0.0344 | 0 |
| 6 | 0.9083 | 0.7575 | 0.1590 | 0.4964 | 0.0028 | 0.0520 | 0 |
