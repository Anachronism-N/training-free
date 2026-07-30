# v143 Multi-axis Head Taxonomy

- Accepted split-stable features: `21`
- Coordinate system: `layer_residual`
- Selected k: `None`
- Status: `no_stable_k`
- PF and other published labels are post-hoc references only.
- Functional role names remain blocked until causal cache routing passes.

## k diagnostics

| k | split agreement | split ARI | silhouette | bootstrap ARI | min class | layer-band NMI | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9694 | 0.8749 | 0.2651 | 0.6861 | 0.2556 | 0.0122 | 0 |
| 3 | 0.9250 | 0.7411 | 0.1611 | 0.5943 | 0.0444 | 0.0101 | 0 |
| 4 | 0.9222 | 0.7434 | 0.1644 | 0.3773 | 0.0028 | 0.0154 | 0 |
| 5 | 0.9472 | 0.8479 | 0.1562 | 0.5592 | 0.0028 | 0.0374 | 0 |
| 6 | 0.9222 | 0.7930 | 0.1381 | 0.4980 | 0.0028 | 0.0386 | 0 |
