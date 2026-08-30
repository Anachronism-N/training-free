# v200: Head x Denoising Phase x AR-Horizon Audit

> Date: 2026-08-31
> Status: code complete; waits for v189 profiles
> GPU generation: none
> Manual video review: none

## 1. What changed after syncing the repository

The latest `main` commit uploads complete 60-second VBench-Long artifacts for
v186.  The directly reported aggregate scores are:

| Dimension | SF native | all-Recent | all-head Retrieval | PF context |
|---|---:|---:|---:|---:|
| Subject consistency | 0.96994 | 0.97089 | 0.97070 | 0.97537 |
| Background consistency | 0.96294 | 0.96250 | 0.96179 | 0.96618 |
| Temporal flickering | 0.97705 | 0.96895 | 0.96901 | 0.97131 |
| Motion smoothness | 0.98415 | 0.98295 | 0.98371 | 0.98443 |
| Overall consistency | 0.21199 | 0.21945 | 0.22502 | 0.23285 |
| Aesthetic quality | 0.56354 | 0.57161 | 0.59699 | 0.60740 |
| Imaging quality | 0.67618 | 0.68567 | 0.69523 | 0.69286 |

`Dynamic Degree=1.0` for Retrieval and PF is an evaluator failure/ceiling and
must not be treated as motion evidence or included in a claimed quality gain.
The strict v198 unified evaluation remains necessary before a paper claim.

Using the uploaded per-clip details only as an exploratory diagnostic, the
correctly rescaled Retrieval-minus-all-Recent effects are strongly
horizon-dependent:

| Window | Semantic alignment | Visual quality | Identity/background |
|---|---:|---:|---:|
| First 30 seconds | +0.00054 | +0.00376 | -0.00035 |
| Last 30 seconds | +0.01058 | +0.03118 | -0.00054 |

The late visual/semantic effect is much larger, while identity/background is
approximately neutral.  This motivates testing AR horizon as a routing axis;
it does not prove such a mechanism.

## 2. Hypothesis

v189 estimates a fixed map over:

```text
operator x noisy denoising call x layer x head
```

but averages its 12 sampled AR readout positions.  If long-history usefulness
changes as local context becomes insufficient, this averaging can erase a
real structure.  v200 tests the stricter hypothesis:

```text
Coverage compatibility depends jointly on head, denoising call,
and current AR horizon.
```

`current_frame` is the AR generation location.  It is not the diffusion
denoising timestep; denoising call remains a separate axis.

## 3. Data and leakage control

v200 reuses the v189 representation-complete shadow profiles:

- 128 Qwen-rewritten MovieGen prompts;
- 4 noisy calls, 30 layers, 12 heads;
- 12 AR locations: `12, 21, ..., 111`;
- Recent and Coverage both read at most 9 FFE;
- Union is a verified representation-complete teacher of at most 13 FFE;
- active generation always used Recent, so shadow measurements did not change
  the latent trajectory.

The frozen v189 split is unchanged:

- 64 discovery prompts choose selectors;
- 32 validation prompts test them;
- 32 generation-holdout prompts are never read by v200.

No threshold is tuned on videos, VBench, or the generation holdout.

## 4. Primary test

For each record and head:

```text
gain = log(error Recent -> Union) - log(error Coverage -> Union)
```

At each sparsity level, v200 compares two selectors:

1. **Static Head x Phase**: rank call/layer/head cells by discovery gain
   averaged over all 12 AR locations, then use the same membership everywhere.
2. **Horizon-conditioned**: rank cells independently at each AR location.

Both select exactly the same number of cells at every location.  Thus any
validation difference cannot come from more Coverage exposure.

The primary sparsity is 10% of the 1,440 call/layer/head cells per AR location.
The 1%, 5%, and 20% results are sensitivity analyses.  The automatic gate is:

- primary paired validation CI lower bound is strictly positive;
- primary prompt win fraction is at least 0.55;
- shuffling the 12 horizon memberships gives permutation `p <= 0.05`;
- at least one adjacent sparsity, 5% or 20%, also has nonnegative CI lower
  bound and positive mean.

These four checks answer effect, prompt prevalence, correct temporal alignment,
and local sparsity robustness.  v200 intentionally does not add another gate
requiring a globally positive AR slope: different cells may help early and
late, and that interaction is exactly the hypothesis.

The report also includes discovery/validation correlations, per-cell linear
horizon slopes, membership turnover, budget coverage, and residual energy.
They diagnose the mechanism but do not add post-hoc promotion conditions.

## 5. Decisions

| Recommendation | Meaning |
|---|---|
| `advance_head_phase_horizon_to_runtime_design` | At least one operator has reproducible equal-exposure horizon structure; implement a separate causal generation screen next. |
| `retain_v189_head_phase_without_ar_horizon` | v189 has candidates, but AR horizon does not add cross-fit predictive value. |
| `no_reproducible_classifier_structure_do_not_generate` | Neither v189 nor v200 supports selective routing; keep only operator-level work. |

Even a pass does not establish video quality.  It only authorizes implementing
the v201 runtime with static, horizon-conditioned, horizon-shift, all-Recent,
and all-Coverage generation controls.  The frozen v189 map is never rewritten.

## 6. Server commands

First complete v189 through `audit` and `analyze`.  Then on any CPU node:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

bash scripts/run_v200_head_phase_horizon_audit.sh preflight
bash scripts/run_v200_head_phase_horizon_audit.sh analyze
bash scripts/run_v200_head_phase_horizon_audit.sh show
bash scripts/run_v200_head_phase_horizon_audit.sh package
```

This step does not use a GPU, decode a video, or require manual review.  It can
run while v198/v199 evaluation jobs use the GPU nodes.

Push only these small outputs:

```text
runs/v200_head_phase_horizon_audit/analysis/analysis.json
runs/v200_head_phase_horizon_audit/analysis/analysis.md
runs/v200_head_phase_horizon_audit/analysis/horizon_curves.csv
runs/v200_head_phase_horizon_audit/analysis/cell_horizon_slopes.csv
runs/v200_head_phase_horizon_audit/analysis/selector_tests.csv
```

Do not upload the v189 `.pt` profiles unless the analyzer reports a contract
failure that cannot be diagnosed from the small report.

## 7. Execution priority

1. Run v198 to finish the strict reused-video comparison.
2. Run v189 profiling and its existing v197 threshold-free audit.
3. Run v200 immediately after v189; it is CPU-only.
4. Run v199 if v198 does not reject Retrieval, to identify the minimum archive
   storage capacity.
5. Implement/run v201 generation only if v200 passes.  Otherwise continue with
   the existing Head x Phase v190 screen or stop selective classification,
   according to the v189/v197 result.

This ordering prevents another broad generation campaign before the new
classification axis has cross-fit evidence.
