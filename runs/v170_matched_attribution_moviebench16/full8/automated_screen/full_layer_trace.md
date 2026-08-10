# v170 Full Active-layer Trace Audit

Overall mechanism gate: **True**

| Method | Gate | Reads | Retrievals | Multi-candidate | Changed selector | Old recalls | Age p95 | Budget errors | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours_v170_v166_a | True | 6400 | 6400 | 3968 | 1328 | n/a | 22.0 | 0 | 0 |
| ours_v170_queryweighted_a | True | 6400 | 6400 | 3628 | 258 | 1986 | 22.0 | 0 | 0 |
| ours_v170_v166_b | True | 6400 | 6400 | 3968 | 1328 | n/a | 22.0 | 0 | 0 |
| ours_v170_queryweighted_b | True | 6400 | 6400 | 3628 | 258 | 1986 | 22.0 | 0 | 0 |

## ours_v170_v166_a

| Layer | Gate | Retrievals | Changed | Old recalls | Age median | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | True | 640 | 167 | n/a | 10.0 | 0 |
| 11 | True | 640 | 137 | n/a | 10.0 | 0 |
| 12 | True | 640 | 138 | n/a | 10.0 | 0 |
| 13 | True | 640 | 146 | n/a | 10.0 | 0 |
| 14 | True | 640 | 121 | n/a | 10.0 | 0 |
| 15 | True | 640 | 145 | n/a | 10.0 | 0 |
| 16 | True | 640 | 135 | n/a | 10.0 | 0 |
| 17 | True | 640 | 118 | n/a | 10.0 | 0 |
| 18 | True | 640 | 116 | n/a | 10.0 | 0 |
| 19 | True | 640 | 105 | n/a | 10.0 | 0 |

## ours_v170_queryweighted_a

| Layer | Gate | Retrievals | Changed | Old recalls | Age median | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | True | 640 | 28 | 209 | 10.0 | 0 |
| 11 | True | 640 | 24 | 227 | 10.0 | 0 |
| 12 | True | 640 | 27 | 203 | 10.0 | 0 |
| 13 | True | 640 | 22 | 189 | 10.0 | 0 |
| 14 | True | 640 | 30 | 218 | 10.0 | 0 |
| 15 | True | 640 | 26 | 209 | 10.0 | 0 |
| 16 | True | 640 | 25 | 190 | 10.0 | 0 |
| 17 | True | 640 | 34 | 192 | 10.0 | 0 |
| 18 | True | 640 | 23 | 161 | 10.0 | 0 |
| 19 | True | 640 | 19 | 188 | 10.0 | 0 |

## ours_v170_v166_b

| Layer | Gate | Retrievals | Changed | Old recalls | Age median | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | True | 640 | 167 | n/a | 10.0 | 0 |
| 11 | True | 640 | 137 | n/a | 10.0 | 0 |
| 12 | True | 640 | 138 | n/a | 10.0 | 0 |
| 13 | True | 640 | 146 | n/a | 10.0 | 0 |
| 14 | True | 640 | 121 | n/a | 10.0 | 0 |
| 15 | True | 640 | 145 | n/a | 10.0 | 0 |
| 16 | True | 640 | 135 | n/a | 10.0 | 0 |
| 17 | True | 640 | 118 | n/a | 10.0 | 0 |
| 18 | True | 640 | 116 | n/a | 10.0 | 0 |
| 19 | True | 640 | 105 | n/a | 10.0 | 0 |

## ours_v170_queryweighted_b

| Layer | Gate | Retrievals | Changed | Old recalls | Age median | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | True | 640 | 28 | 209 | 10.0 | 0 |
| 11 | True | 640 | 24 | 227 | 10.0 | 0 |
| 12 | True | 640 | 27 | 203 | 10.0 | 0 |
| 13 | True | 640 | 22 | 189 | 10.0 | 0 |
| 14 | True | 640 | 30 | 218 | 10.0 | 0 |
| 15 | True | 640 | 26 | 209 | 10.0 | 0 |
| 16 | True | 640 | 25 | 190 | 10.0 | 0 |
| 17 | True | 640 | 34 | 192 | 10.0 | 0 |
| 18 | True | 640 | 23 | 161 | 10.0 | 0 |
| 19 | True | 640 | 19 | 188 | 10.0 | 0 |

Per-layer gates are diagnostics only; the frozen mechanism gate requires complete coverage and a valid aggregate execution for each replica. Selector changes need not occur in every layer.
