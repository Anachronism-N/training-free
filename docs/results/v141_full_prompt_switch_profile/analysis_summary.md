# v141 Full-Prompt A-B-A Head Profiling

- Recommendation: `retain_continuous_diagnostic_or_redesign_prompt_axis`
- Full prompt-switch gate: `False`
- Exact-shadow parity median / p99: `0` / `0`
- Switch / local-paraphrase residual median: `0.00632642` / `0.00369758`
- Discovery-validation Spearman / label agreement: `0.5558` / `0.6695`
- Validation positive / active heads: `189` / `348`

## Interpretation

The base trajectory executes A-B-A and preserves native self-attention history. Exact-A/B and local-paraphrase shadows change only current conditioning on the same latent and history.
The primary score is residual switch/paraphrase log-ratio minus the corresponding query log-ratio. Zero therefore asks whether prompt switching changes history use beyond its direct effect on Q.
Layer 0 is structurally prompt-blind at self-attention and is forced to the stable class. Otsu/GMM thresholds are diagnostics only.
