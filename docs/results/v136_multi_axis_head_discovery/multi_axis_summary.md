# v136 Multi-Axis Head Analysis

This report analyzes frozen v134 profiles. It does not use video metrics, PF labels, or legacy class counts to select a map.

## Decision

- Recommendation: `temporal_axis_only_prompt_taxonomy_rejected`
- Prompt-axis gate: `False`
- Temporal-axis gate: `True`
- CPHI/temporal Spearman: `0.1075`

## Class Counts

- `prompt_label`: prompt_conditional=1, prompt_invariant=359
- `age_routing_label`: age_invariant=360
- `history_polarity_label`: history_supportive=49, recent_preferred=311
- `long_range_label`: local_or_mixed=314, long_range=46
- `exploratory_joint_role`: conditional_local=1, invariant_local=313, invariant_long=46

## Reproducibility

- CPHI split-half Spearman: `0.8163`
- CPHI bootstrap-reliable fraction: `0.9917`
- Middle/recent split-half Spearman: `0.9959`
- Middle/recent bootstrap-reliable fraction: `0.9917`

## Negative Control

- Ineligible states are excluded from primary scores because their full history contains no frame older than recent4.
- Median semantic/null residual responses: `0.000000` / `0.000000`

## Interpretation Boundary

- `prompt_label` is the zero-threshold CPHI hypothesis.
- `age_routing_label` asks whether prompt semantics change which history ages receive attention.
- `history_polarity_label` is a native-window middle-vs-recent diagnostic, not the superseded v98 304/56 map.
- `exploratory_joint_role` must not be used for generation until the corresponding prompt and temporal gates pass.
- GMM and Otsu thresholds are diagnostics only.
