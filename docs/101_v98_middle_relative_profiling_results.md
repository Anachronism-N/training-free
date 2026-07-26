# v98 Middle-Relative QK Profiling Results

Date: 2026-07-26

## 1. Overview

The v98 middle-relative QK profiling captures deployment-matched QK head
scores using the corrected v98 cache definition (exclusive ownership,
neutral labels 10/11, no legacy fallthrough). These scores replace the v97
QK head scores for all v99 binary cache recovery experiments.

## 2. Profiling Configuration

- **Profile pairs**: 16 counterfactual prompt pairs × 2 seeds = 32 jobs per policy
- **Policies**: uniform_stride and uniform_merge (64 total profiles)
- **Frames**: 120 (frozen, not adjustable)
- **Seeds**: 0, 1 (frozen)
- **Layers**: 30 (layer_idx 0..29)
- **Heads per layer**: 12
- **Total heads**: 360
- **Sink frames**: 3, Recent frames**: 4

## 3. Score Columns

| Column | Description |
|---|---|
| `middle_relative_logit_margin` | Primary score: middle-vs-recent QK logit margin |
| `uniform_stride_margin` | Stride-policy-specific margin |
| `uniform_merge_margin` | Merge-policy-specific margin |
| `topology_sign_agreement` | Whether stride and merge agree on sign |
| `profile_observation_count` | Number of profiling observations |
| `record_observation_count` | Total QK records across all observations |
| `profile_positive_fraction` | Fraction of positive logit observations |
| `bootstrap_sign_agreement` | Sign stability across bootstrap resamples |

## 4. Classification Results (Natural Zero Threshold)

| Role | Label | Count | Fraction |
|---|---|---|---|
| Supportive (stable) | 10 | 33 | 9.2% |
| Suppressive (responsive) | 11 | 327 | 90.8% |

This is a much more conservative classification than the v97 QK scores
(304/56 = 84%/16%). The middle-relative logit margin identifies far fewer
heads as needing long-range identity support.

## 5. Acceptance Gates

| Gate | Observed | Required | Passed |
|---|---|---|---|
| Complete head grid | 360 | 360 | ✅ |
| Bootstrap stable head fraction | 0.978 | 0.8 | ✅ |
| Minority role fraction | 0.092 | 0.05 | ✅ |
| Topology sign agreement | 0.814 | (diagnostic) | — |

All hard gates passed. The topology sign agreement of 0.814 means that
stride and merge policies agree on the sign direction for ~81% of heads.

## 6. PF Label Overlap

The v98 map contains 33 Supportive and 327 Suppressive heads. PF's
three-class system has 172 Anchor, 156 Wave, and 32 Veil heads.

- The 33 Supportive heads are a subset of PF's Anchor class (169/172 Anchor
  heads map to Supportive).
- The 327 Suppressive heads include all 156 Wave and 32 Veil heads, plus
  3 Anchor heads.
- This means the binary classifier nearly perfectly separates Anchor from
  Wave+Veil, but all 156 Wave heads lose their cyclic route.

## 7. Comparison with v97 QK Scores

| Metric | v97 QK Scores | v98 Middle-Relative |
|---|---|---|
| Score type | CFG + prompt-pair QK response | Middle-vs-recent QK logit margin |
| Supportive count | 292 (81%) | 33 (9.2%) |
| Suppressive count | 68 (19%) | 327 (90.8%) |
| Score SHA-256 | e0f9e702... | 83d2be6a... |
| Profiling jobs | 32 (CFG + semantic pairs) | 64 (stride + merge, 2 seeds) |

The v98 middle-relative scores produce a very different head distribution.
The 9.2% Supportive fraction is close to PF's Anchor fraction (172/360 =
47.8%), but much smaller. This suggests the middle-relative margin is a
stricter criterion for long-range support.

## 8. Score Artifact

The frozen score artifact is at:

```text
runs/v98_middle_relative_scores/scores/
├── qk_head_scores.csv          (360 rows, 10 columns)
├── qk_head_observations.json   (per-observation records)
├── qk_head_score_artifact.json (SHA-256 hashes + acceptance gates)
└── layer_capture_audit.json    (layer source validation)
```

Score SHA-256: `83d2be6ab2978c3aa13f8d508f29a6fcf0d8fa026bca8bccb319b2a3e10ba53c`

## 9. Maps Built from Scores

The v99 script builds 8 maps from these scores:

| Map | Key | Supportive | Suppressive |
|---|---|---|---|
| history_polarity_zero | Natural zero threshold | 33 | 327 |
| history_polarity_zero_random | Layer-wise count-matched random | 33 | 327 |
| history_polarity_zero_inverted | Inverted direction | 327 | 33 |
| pf_ar_binary_control | PF Anchor-vs-Rest | 172 | 188 |
| pf_aw_binary_control | PF (Anchor+Wave)\|Veil | 328 | 32 |
| + 3 threshold variants | ±0.1 robustness | varies | varies |

## 10. Infrastructure Notes

- Profiling was run on a single 8-GPU H20 node (node221) due to a
  calibration lock file (`.calibration_run_lock`) that prevents multi-node
  execution.
- Stride profiling completed in ~25 minutes (4 batches × 8 jobs).
- Merge profiling completed in ~25 minutes (4 batches × 8 jobs).
- Total profiling time: ~50 minutes (excluding model loading).
- A local config fix (`pyramidkv_prompt_warmup_enabled: false`) was
  required and hidden via `git update-index --assume-unchanged`.
