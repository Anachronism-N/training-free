# v134 Head Discovery Analysis

- Acceptance gates: **FAIL**
- Profiles: 128 observational, 128 counterfactual
- Zero-threshold partition: 0 prompt-conditional, 360 prompt-invariant
- Semantic/paraphrase median interaction: 0.001803 / 0.002936
- Split-half Spearman / label agreement: 0.8105 / 1.0000
- Bootstrap-reliable head fraction: 0.9972
- Matched-base residual/native drift: 0.000000 / 0.000000

## Prompt-Temporal Relations

- CPHI vs `expected_age` Spearman: 0.0922
- CPHI vs `recent4_mass` Spearman: -0.0759
- CPHI vs `old12_mass` Spearman: 0.1186
- CPHI vs `temporal_entropy` Spearman: 0.1051
- CPHI vs `positive_logit_fraction` Spearman: -0.0089

## Representation Relations

- CPHI vs `native_log_ratio` Spearman: 0.7372
- CPHI vs `query_log_ratio` Spearman: 0.8090
- CPHI vs `current_key_log_ratio` Spearman: 0.7629

## Interpretation Rule

The static binary map is admissible only when the acceptance gates pass. If the global map fails but timestep/factor tables are reproducible, use a continuous timestep-conditioned gate instead of forcing a binary partition.
