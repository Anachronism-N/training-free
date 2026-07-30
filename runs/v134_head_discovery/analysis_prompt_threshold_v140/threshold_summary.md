# v140 Prompt-Sensitivity Threshold Audit

- Recommendation: `no_thresholded_prompt_head_class_supported`
- Discovery/validation split: even/odd subject family.
- Layer 0 is structural zero and is not used to fit thresholds.

## Scores

### raw_cphi

- Split Spearman / zero-label agreement: `0.7428` / `0.9856`
- Validation positive / minority / boundary fractions: `3/348` / `0.0086` / `0.0029`
- Frozen zero-threshold gate: `False`

### query_adjusted

- Split Spearman / zero-label agreement: `0.6588` / `0.7443`
- Validation positive / minority / boundary fractions: `240/348` / `0.3103` / `0.1006`
- Frozen zero-threshold gate: `False`

### native_adjusted

- Split Spearman / zero-label agreement: `0.6454` / `0.7241`
- Validation positive / minority / boundary fractions: `237/348` / `0.3190` / `0.1034`
- Frozen zero-threshold gate: `False`

### key_adjusted

- Split Spearman / zero-label agreement: `0.6282` / `0.7586`
- Validation positive / minority / boundary fractions: `240/348` / `0.3103` / `0.0776`
- Frozen zero-threshold gate: `False`

## Interpretation

GMM and Otsu thresholds are fitted on discovery families only. They remain diagnostics unless their labels transfer to validation.
The percentile sweep shows how strongly class membership depends on an arbitrary threshold; it must not be selected by PF overlap or downstream video metrics.
