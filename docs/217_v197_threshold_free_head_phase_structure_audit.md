# v197: Threshold-Free Head x Denoising-Phase Structure Audit

> Date: 2026-08-26
> Status: code ready; runs only after v189 analysis exists
> Compute: CPU only, no video generation, no human review

## 1. Current synchronized state

The GitHub experiment branch remains at `2bc495a9`. No v189-v195 result artifact
has been pushed, so the verifiable campaign frontier is still v189. The server may
contain newer unpushed files; `scripts/run_v196_campaign_frontier.sh show` remains
the authoritative server-side check.

The next GPU action is therefore still v189 profiling. v197 does not bypass that
experiment and does not authorize v190. It is a small automatic audit that runs as
soon as v189 writes `analysis.json` and `cell_scores.csv`.

## 2. Question

v189 defines a primary map with fixed gain, validation, budget, and residual-energy
thresholds. It already reports a threshold-count grid. Cell counts alone do not
answer whether the underlying continuous signal contains reproducible head and
denoising-phase structure.

v197 asks:

> If discovery scores choose a head or denoising phase without using validation
> scores, does that choice remain enriched on validation prompts, and is the joint
> Head x Phase residual reproducible before any generated video is inspected?

This addresses threshold arbitrariness without choosing a new threshold. The v189
compatible map remains byte-identical and v190 remains the only generated-video
causal gate.

## 3. Automatic analyses

For each frozen Landmark and Retrieval profile, v197 performs four analyses.

### 3.1 Continuous split reproducibility

It reports discovery-versus-validation Pearson and Spearman correlations, sign
agreement, and fixed top-1%, top-5%, and top-10% rank overlap across all 1,440
`(call, layer, head)` cells.

### 3.2 Threshold-free variance decomposition

Within each layer, the continuous gain tensor is decomposed into:

```text
layer mean
+ denoising-phase main effect
+ head main effect
+ Head x Phase interaction residual
```

The report records each component's fraction of total score variation and the
discovery-validation correlation of the head, phase, and interaction components.
No binary cache class is needed for this calculation.

### 3.3 Cross-fit randomization diagnostics

Selections are made only from discovery scores and evaluated only with validation
scores:

- head identity: within every `(call, layer)`, select discovery top-1, top-3, or
  top-6 heads;
- phase identity: within every `(layer, head)`, select the discovery top-1 noisy
  denoising call;
- global ranking: select discovery top-1%, top-5%, or top-10% cells.

Each validation contrast is compared with 10,000 deterministic count-matched random
selections. These are correlated-cell diagnostics, not independent-head hypothesis
tests and not formal paper p-values.

### 3.4 Frozen-map topology and threshold neighborhood

The existing v189 compatible map is audited against `cell_scores.csv`, including
its SHA, dimensions, operator, per-call counts, and exact membership. v197 then
reports:

- cells by denoising call and layer;
- heads selected in zero, one, two, three, or four calls;
- phase-varying versus call-invariant heads;
- membership Jaccard against the fixed 4 x 3 gain/win threshold neighborhood.

The grid is descriptive. It cannot be used to replace the v189 map after observing
v190 generation metrics.

## 4. Interpretation

The diagnostic label uses the predeclared discovery top-3 head test, top-1 phase
test, and continuous interaction correlation:

| Label | Profiling interpretation |
|---|---|
| `joint_head_phase_structure` | head and phase cross-fit enrichments are positive and the interaction residual replicates |
| `additive_head_and_phase_structure` | both main structures replicate but the joint residual does not |
| `head_structure_only` | only exact head ranking transfers |
| `phase_structure_only` | only denoising-phase ranking transfers |
| `operator_level_gain_only` | no stable classifier structure, but mean Coverage gain is positive |
| `unsupported` | no positive diagnostic structure |

These labels refine the mechanism story but never claim generated-video benefit.
A passing v190 comparison against all-Coverage, Head-only, Phase/Layer-only,
membership-shift, phase-shift, and dense controls is still required.

## 5. Server commands

After v189 `analyze` completes on node 0:

```bash
git pull
bash scripts/run_v197_head_phase_structure.sh show
bash scripts/run_v197_head_phase_structure.sh package
```

The commands write:

```text
runs/v197_head_phase_structure/analysis/analysis.json
runs/v197_head_phase_structure/analysis/analysis.md
runs/v197_head_phase_structure/analysis/threshold_grid.csv
runs/v197_head_phase_structure/analysis/crossfit_tests.csv
runs/v197_head_phase_structure/v197_small_artifacts.tar.gz
```

Push the four files under `analysis/`. The archive is for direct transfer and does
not need to be committed.

## 6. Decision order

1. Run v196 `show` on the server.
2. If the frontier is v189, finish v189 profiling and analysis.
3. Run v197 automatically; no video review is needed.
4. If v189 itself rejects both operators, stop regardless of v197's descriptive
   label.
5. If v189 advances an operator, run the already implemented v190 generation-side
   causal screen. Do not tune v189 thresholds from v197 or v190 outcomes.
