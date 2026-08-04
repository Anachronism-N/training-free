# v159 Motion-Pair Trace Diagnosis

This is a mechanism audit, not a quality comparison.

| Method | Accept/prompt | Reject/prompt | Pair-age p95 | Max age | Union mean |
|---|---:|---:|---:|---:|---:|
| ours_interleaved10_reservoir2_motionpair1 | 6.125 | 32.875 | 34.494 | 61 | 3.392 |
| ours_interleaved10_motionpair2 | 6.688 | 32.312 | 71.862 | 115 | 3.294 |
| ours_middle10_reservoir2_motionpair1 | 5.938 | 33.062 | 36.431 | 73 | 3.400 |

## Frozen conclusion

The motion route executed and respected its read budget, but the motion-quantile gate was the dominant rejection path. The existing `max_pair_age=24` only relaxed replacement; it did not bypass the quantile gate, so retained pairs could remain substantially older than 24 frames.

v160 therefore changes only the selected Middle10 route: stale pairs use a 12-frame refresh horizon and may bypass the motion quantile gate. Positive-motion and semantic-coherence eligibility, pair adjacency, spacing, sink/recent allocation, and the 9-FFE read budget remain unchanged.
