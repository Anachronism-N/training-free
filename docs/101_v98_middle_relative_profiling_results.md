# v98 Middle-Relative QK Profiling Results

Date: 2026-07-26

Status: profiling completed; PF-overlap claims require artifact verification.

## 1. What was measured

The profiler estimates, for every attention head, whether its current query
places relatively more QK logit mass on non-recent middle history or on the
latest distinct history frames. For one captured query:

```text
margin = standardized(mean(middle logits) - mean(recent logits))
```

The first three sink frames and the latest four distinct recent frames are
excluded from `middle`. Because both terms use the same query, subtracting them
is invariant to a common additive logit shift. Scores are aggregated within
each independently generated profile and then across profiles by a median.

This is not a prompt-sensitivity classifier and does not use PF labels to make
the primary split.

## 2. Frozen profiling protocol

- 8 counterfactual prompt pairs, both sides, 2 seeds
- 32 profiles under uniform stride and 32 under uniform merge
- 64 observations per head in total
- 120 latent frames, conditional branch, noisy update mode
- 30 layers x 12 heads = 360 heads
- score field: `middle_relative_logit_margin`
- natural threshold: zero

The map builder applies:

```text
score >= 0  -> label 10, History-Supportive
score < 0   -> label 11, Recent-Responsive
```

The zero threshold means equal middle/recent preference and is not selected to
match a PF class count. Thresholds `-0.1` and `+0.1` are robustness ablations.

## 3. Reported natural-zero counts

The server-side result report states:

| Role | Label | Heads | Fraction |
|---|---:|---:|---:|
| History-Supportive | 10 | 33 | 9.2% |
| Recent-Responsive | 11 | 327 | 90.8% |

These counts are plausible as an independent sparse-global split, but the raw
CSV and generated map manifest have not been committed. They must therefore be
treated as reported values until the artifacts below are available.

## 4. Acceptance gates

| Gate | Observed | Required | Status |
|---|---:|---:|---|
| complete head grid | 360 | 360 | pass |
| bootstrap-stable head fraction | 0.978 | 0.80 | pass |
| minority role fraction | 0.092 | 0.05 | pass |
| topology sign agreement | 0.814 | 0.80 | pass |

Topology sign agreement is a hard gate in
`extract_v98_middle_relative_scores.py`; it is not merely diagnostic.

Passing these gates establishes score reproducibility under the frozen
profiling protocol. It does not establish that the resulting membership
improves video quality.

## 5. PF-overlap inconsistency in the original report

The first version of this document simultaneously claimed:

- only 33 heads are History-Supportive; and
- 169 of 172 PF Anchor heads are History-Supportive.

Both cannot be true: if there are 33 Supportive heads in total, at most 33 PF
Anchor heads can be Supportive. The associated Wave/Veil row totals were also
inconsistent with 360 heads. Therefore no PF-overlap number from that prose
report may be cited in a paper or used to name the classes.

The v99 runner now independently reloads every 30x12 CSV, recomputes:

- exact label counts;
- the Anchor/Wave/Veil x Supportive/Responsive cross-tab; and
- total-head consistency;

then compares those values with the map manifest. Any discrepancy aborts the
run. It also prints one `[V99MapAudit]` JSON line per selected map and freezes
the same statistics in the experiment contract and per-cell config.

## 6. Evidence boundary

Established:

- a shift-invariant middle-vs-recent score was collected for all heads;
- all frozen profiling stability gates reportedly passed;
- the score produces a non-degenerate binary split;
- one independent-map stride/cyclic video was visually free of polygon noise.

Not yet established:

- the exact PF class overlap;
- that the natural-zero map is better than count-matched random or inverted;
- that zero is the best quality threshold;
- that the binary method matches or exceeds native PF on 32/128 prompts.

## 7. Artifacts that must be preserved

Push or archive these small files before making quantitative head-overlap
claims:

```text
runs/v98_middle_relative_scores/scores/qk_head_scores.csv
runs/v98_middle_relative_scores/scores/qk_head_score_artifact.json
runs/v98_middle_relative_scores/scores/layer_capture_audit.json
runs/v99_binary_cache_recovery_*/maps/history_polarity_manifest.json
runs/v99_binary_cache_recovery_*/maps/history_polarity_zero.csv
runs/v99_binary_cache_recovery_*/maps/head_assignments.csv
```

The reported score CSV SHA-256 is:

```text
83d2be6ab2978c3aa13f8d508f29a6fcf0d8fa026bca8bccb319b2a3e10ba53c
```
