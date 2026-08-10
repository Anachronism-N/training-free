# v170 Replica Hash Audit

Coverage gate: **True**

| Comparison | Pairs | Exact hashes | Different | Mean absolute byte delta |
|---|---:|---:|---:|---:|
| v166_replica_a_vs_b | 16 | 10 | 6 | 1384.44 |
| query_replica_a_vs_b | 16 | 13 | 3 | 222.94 |
| lane_a_v166_vs_query | 16 | 0 | 16 | 66682.44 |
| lane_b_v166_vs_query | 16 | 0 | 16 | 65685.31 |

Hash equality is only an exact reproducibility diagnostic. Different hashes are expected to be resolved by paired metrics; they do not by themselves indicate a failed run.
