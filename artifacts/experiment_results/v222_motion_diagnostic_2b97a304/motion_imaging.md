# Fixed-seed motion/imaging diagnostics

Status: v220_pending

All differences below use a 0-100 scale. Descriptive diagnostics, not causal evidence.

| Cohort | Candidate - control | Imaging | DD | Median imaging | Trimmed imaging | Imaging wins without DD drop |
|---|---|---:|---:|---:|---:|---:|
| v219 | ours_correct - sf_fifo21 | +0.3790 | -1.2500 | +0.0627 | +0.2473 | 27/64 |
| v219 | ours_correct - ours_random | +0.0446 | -1.6667 | +0.0255 | +0.0779 | 28/64 |
| v219 | ours_correct - pooled_correct | +0.1327 | +0.1042 | +0.0208 | +0.0487 | 29/64 |
| v219 | ours_random - sf_fifo21 | +0.3344 | +0.4167 | +0.0042 | +0.1451 | 27/64 |
| v219 | pooled_correct - sf_fifo21 | +0.2463 | -1.3542 | +0.1183 | +0.2326 | 27/64 |
| v219 | pooled_correct - ours_random | -0.0881 | -1.7708 | +0.0571 | +0.0804 | 29/64 |

## v219 Ours - SF

| Observed DD change | Prompts | Mean imaging delta | Contribution to full imaging delta |
|---|---:|---:|---:|
| dd_down | 16 | +0.0097 | +0.0024 |
| dd_tied | 39 | +0.3712 | +0.2262 |
| dd_up | 9 | +1.0695 | +0.1504 |
- Top 1 positive prompts: sources [113], 20.4% of positive-delta mass. They are NOT removed from the main estimate.
- Top 3 positive prompts: sources [113, 114, 85], 41.1% of positive-delta mass. They are NOT removed from the main estimate.

## Interpretation boundaries

- Posthoc diagnostics; the frozen full-video imaging endpoint is unchanged.
- Motion groups condition on observed treatment outcomes, not randomized subgroups or causal controls.
- Dynamic-degree ties do not establish equal motion: the metric is a coarse clip-level threshold statistic.
- Subgroup intervals are descriptive, unadjusted for multiple comparisons, and not new primary tests.
- Trimmed/leave-out means diagnose concentration only; every prompt remains in the main table.
- Correlations and a positive tied-motion subgroup do not prove that motion is preserved.
- Keep SF, random and pooled comparisons, including negative effects. No seed/method winner is selected.
- No new video, metric-model inference, or human review is requested; no automatic acceptance verdict.
