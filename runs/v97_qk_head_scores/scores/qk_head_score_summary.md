# Unthresholded QK Head Scores

- profiles: 32
- heads: 360 (30 x 12)
- classification applied: no
- layer source: `kv_cache.layer_idx`
- scores: `qk_head_scores.csv`
- score artifact: `qk_head_score_artifact.json`

All thresholds and alternative classifiers must consume this immutable score artifact. Profiling inference must not be rerun to change labels.
