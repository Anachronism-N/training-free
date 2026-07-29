# v132 binary head-partition audit

PF labels are used only for post-hoc interpretation, never to form the proposed partition.

- Heads: 360
- Counts: `{'supportive': 304, 'suppressive': 56}`
- Frozen-map mismatches: 0

## Post-hoc PF cross-tab

| PF class | Heads | Supportive | Suppressive |
|---|---:|---:|---:|
| wave | 156 | 133 | 23 |
| anchor | 172 | 169 | 3 |
| veil | 32 | 2 | 30 |

## Threshold stability

| Threshold | Supportive | Suppressive | Changed vs 0 | Jaccard vs 0 | PF-AW agreement (post-hoc) |
|---:|---:|---:|---:|---:|---:|
| -0.250 | 315 | 45 | 11 | 0.8036 | 0.9361 |
| -0.100 | 309 | 51 | 5 | 0.9107 | 0.9306 |
| -0.050 | 306 | 54 | 2 | 0.9643 | 0.9278 |
| 0.000 | 304 | 56 | 0 | 1.0000 | 0.9222 |
| 0.050 | 302 | 58 | 2 | 0.9655 | 0.9167 |
| 0.100 | 301 | 59 | 3 | 0.9492 | 0.9139 |
| 0.250 | 293 | 67 | 11 | 0.8358 | 0.8917 |
