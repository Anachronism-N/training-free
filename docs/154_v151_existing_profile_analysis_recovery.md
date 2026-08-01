# 154: v151 Existing-Profile Analysis Recovery

Date: 2026-08-01

Status: code complete. This recovery analyzes the existing 64 v151 profiles;
it does not regenerate videos and does not relax the frozen 2% calibration
threshold.

## 1. Why this recovery is needed

The v151 GPU run completed all 64 trajectories and all 132 downstream records
per profile. The old runner stopped before analysis because 70 of 245,760
calibrated layers exceeded 2% relative error. Every failure was at nominal
timestep 250; timesteps 1000, 750, and 500 were intact.

Two independent implementation problems also blocked the result:

1. the generated plan and profiles used eight refinement steps, while the
   analyzer still required four;
2. the runner applied one global maximum-error assertion before the analyzer,
   although the preregistered analyzer already gates each probe/context
   independently.

The second issue discarded three valid denoising contexts because one context
failed. The repaired path keeps the strict gate and separates two audits:

```text
audit:
    strict artifact gate; all four contexts must pass

audit_analysis:
    structural/replay gate; at least one context must be intact

analyzer:
    every probe/context still requires error <= 2%, valid scale,
    no clipped/degenerate/bound-hit layer, and the exact target
```

An invalid context remains in diagnostic CSV files but cannot satisfy any
scientific confirmation gate.

## 2. Code changes

- `scripts/analyze_v151_signed_policy_low_tail_profiles.py`
  - reads the refinement-step count from the frozen probe plan;
  - verifies every profile used the same count;
  - reports intact and invalid contexts explicitly;
  - allows invalid cells to be gated locally instead of aborting all analysis.
- `scripts/audit_v151_signed_policy_profiles.py`
  - performs full profile, prompt/seed, replay, probe-grid, layer, target,
    scale, and calibration checks;
  - writes every offending layer to CSV;
  - supports strict and analysis-ready modes.
- `scripts/run_v151_signed_policy_low_tail_32gpu.sh`
  - adds `audit_analysis`;
  - makes `analyze` call `audit_analysis` rather than the strict audit;
  - restores the smoke calibration threshold to the same 2% used by analysis.
- tests now connect the suite builder to the real analyzer and verify that one
  bad probe/context does not invalidate unrelated contexts.

## 3. Server commands

Run only on node 0. Keep the existing `runs/v151_signed_policy_low_tail/`
directory and do not run `prepare`, `smoke`, or `core64` again.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only origin codex/v98-correctness-fixes

export NODE_RANK=0
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export SF_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt

bash scripts/run_v151_signed_policy_low_tail_32gpu.sh audit_analysis
bash scripts/run_v151_signed_policy_low_tail_32gpu.sh analyze
bash scripts/run_v151_signed_policy_low_tail_32gpu.sh package
```

These commands are CPU-only. The GPU occupier may remain active.

The strict audit is retained for provenance and is expected to fail on the
current profiles:

```bash
bash scripts/run_v151_signed_policy_low_tail_32gpu.sh audit
```

That failure means the four-context artifact is not globally clean. It does
not invalidate independently intact contexts reported by `analyze`.

## 4. Expected audit output

For the profiles described in `docs/153`, `audit_analysis` should report the
following structure:

```text
[v151-context-audit] context=noisy_t1000 pass=1 ...
[v151-context-audit] context=noisy_t750  pass=1 ...
[v151-context-audit] context=noisy_t500  pass=1 ...
[v151-context-audit] context=noisy_t250  pass=0 ... error_fail=70 ...
[v151-audit-summary] mode=analysis accepted=1 ...
```

Do not continue if replay is nonzero, the profile grid is incomplete, or any
of the first three contexts unexpectedly fails.

## 5. Required result artifacts

The package command creates:

```text
docs/results/v151_signed_policy_low_tail/
|-- signed_source/
|-- v151_probe_plan.json
|-- suite_metadata.json
`-- core/
    |-- pre_analysis_audit_analysis.json
    |-- calibration_offenders_analysis.csv
    |-- profile_audit.csv
    |-- probe_integrity.csv
    |-- probe_effect_summary.csv
    |-- group_comparisons.csv
    |-- random_map_comparisons.csv
    |-- intervention_specificity.csv
    |-- contrast_diagnostics.csv
    |-- downstream_observations.csv.gz
    |-- report.json
    `-- report.md
```

Push this directory and the node-0 analysis log. Do not push videos, model
weights, split clips, or the 64 raw `.pt` profiles.

## 6. Frozen decision rule

Read `core/report.json` in this order:

1. `intact_contexts` must contain the contexts used by a claim;
2. `g0_native_replay` must pass;
3. scalar low-tail uses `g1` through `g4`;
4. signed policy uses source gate `g5` and causal gates `g6` through `g8`;
5. a branch is confirmed only when its final gate passes in one same intact
   context, including susceptibility, leverage, intervention specificity,
   random controls, and seed reproducibility.

Possible outcomes:

| Result | Decision |
|---|---|
| Neither `g4` nor `g8` passes | Reject v151; do not spend GPU time on t250 |
| Only `g4` passes | Retain scalar low-tail, reject signed taxonomy |
| Only `g8` passes | Retain signed policy axis, reject scalar ranking |
| Both pass | Compare effect size and stability; do not combine automatically |

Even a passing branch is a one-step causal result. It must still be converted
to a trajectory-level cache policy and evaluated on long videos before it can
be a method contribution.

## 7. Timestep-250 follow-up boundary

Do not increase refinement iterations before reading the three intact
contexts. The current profiles do not save the full native Q/K/V and raw
replacement tensors, so timestep-250 cannot be recalibrated offline from the
committed `.pt` summaries.

If and only if a branch passes at another timestep, the t250 follow-up should:

1. generate to frame 117 but execute downstream probes only at timestep 250;
2. test a quantization-aware bracket or neighboring-scale search rather than
   assuming additional multiplicative iterations will converge;
3. save per-iteration scale, achieved RMS, and error for every failed layer;
4. retain the same 2% target and acceptance threshold;
5. rerun a small frozen smoke set before the 64-profile confirmation.

This follow-up is a separate experiment contract. It must not overwrite the
current v151 plan or profiles.
