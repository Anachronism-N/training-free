# v136 Multi-Axis Analysis Runbook

## 1. When to run

Run v136 after both v134 stages finish and pass the v134 audit:

```text
128 observational profiles
128 counterfactual profiles
```

v136 is CPU-only. It does not generate videos and does not modify raw
profiles. It may run on node 0 while GPUs are used by another experiment,
provided the v134 profile files are complete and no worker is still writing
them.

Do not run against a partial directory. The script requires exact profile,
state, branch, layer, head, and tensor-shape contracts.

## 2. Input paths

Default:

```text
runs/v134_head_discovery/profiles/observational
runs/v134_head_discovery/profiles/counterfactual
```

The frozen sampling contract is:

```text
128 profiles per stage
27 base states per profile
30 layers
12 heads
base branch for observational
base + semantic + null branches for counterfactual
```

## 3. Status

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

bash scripts/run_v136_multi_axis_analysis.sh status
```

Expected:

```text
[v136-status] observational=128/128 counterfactual=128/128
```

## 4. Analyze and package

Foreground:

```bash
bash scripts/run_v136_multi_axis_analysis.sh all
```

Recommended background command:

```bash
mkdir -p runs/v134_head_discovery/logs
nohup bash scripts/run_v136_multi_axis_analysis.sh all \
  > runs/v134_head_discovery/logs/v136_multi_axis.nohup.log 2>&1 &
```

Run analysis and packaging separately:

```bash
bash scripts/run_v136_multi_axis_analysis.sh analyze
bash scripts/run_v136_multi_axis_analysis.sh package
```

Useful smoke override for synthetic or partial developer fixtures:

```bash
EXPECTED_COUNT=2 EXPECTED_STATES=2 BOOTSTRAP_ROUNDS=20 \
  V134_OUT_ROOT=/path/to/fixture \
  V136_OUT_DIR=/tmp/v136_fixture \
  bash scripts/run_v136_multi_axis_analysis.sh analyze
```

Do not use those overrides for a research result.

## 5. Analysis outputs

Default analysis directory:

```text
runs/v134_head_discovery/analysis_multi_axis_v136/
```

Primary:

```text
multi_axis_summary.md
multi_axis_report.json
head_axes.csv
```

Specialization:

```text
head_factor_axes.csv
head_timestep_axes.csv
head_ar_axes.csv
head_factor_specialization.csv
head_timestep_specialization.csv
head_ar_specialization.csv
context_stability.csv
```

Natural temporal characterization:

```text
head_natural_timestep_axes.csv
head_natural_ar_axes.csv
axis_diagnostics.csv
axis_correlations.csv
```

Correctness:

```text
profile_contract_audit.csv
state_eligibility_audit.csv
```

Detailed per-job tables remain in the run directory and are not copied into
the Git review bundle:

```text
head_prompt_job_axes.csv
head_temporal_job_axes.csv
```

## 6. Review order

1. Confirm `profile_contract_passed=true`.
2. Read `multi_axis_summary.md`.
3. Inspect prompt and temporal gates independently.
4. Check `axis_correlations.csv`; prompt and temporal scores should not be
   treated as two contributions if they are nearly identical.
5. Inspect factor and timestep specialization before selecting a static map.
6. Confirm frame-3 negative-control residuals are zero or numerically tiny.
7. Inspect GMM/Otsu diagnostics only after the frozen zero-threshold result.

Do not use `exploratory_joint_role` in generation merely because all four
role names are present.

## 7. Package for Git review

The package action writes:

```text
docs/results/v136_multi_axis_head_discovery/
```

It contains only bounded CSV/JSON/Markdown analysis artifacts and SHA-256
inventory. Raw `.pt` files, videos, and worker logs are excluded.

After review:

```bash
git add docs/results/v136_multi_axis_head_discovery
git commit -m "results: add v136 multi-axis head analysis"
git push
```

Provide:

1. result commit hash;
2. `multi_axis_summary.md`;
3. any failure line from the v136 nohup log;
4. whether v134 videos showed native-SF corruption.

## 8. Failure interpretation

### Contract audit fails

Do not weaken the audit. Check the first reported job for:

- incomplete worker output;
- missing semantic/null branch;
- fewer than 30 layers;
- stale profile from another commit;
- malformed temporal tensor shape.

### Negative-control response is nonzero

At frame 3, full history and recent4 should be identical. A material residual
means the recorder's history slicing or signature construction is wrong.
Stop classification work and inspect the raw profile.

### Prompt axis fails but age redistribution passes

Prompt may change temporal routing without a large head-output residual.
Use the age axis as a causal hypothesis, but validate it through grouped
history visibility ablations before generation.

### Prompt axis passes but age redistribution fails

Prompt affects history values/output without changing the selected ages.
Scene-switch handling may still use prompt-keyed namespace or write
admission, but there is no evidence for a different temporal sampling pattern.

### Static labels vary strongly by timestep

Do not average them into one map. Use `head_timestep_axes.csv` and test a
timestep-conditioned continuous gate.

### No axis has a stable split

Report the negative result. Keep continuous retrieval confidence or abandon
head classification. Do not tune a threshold against VBench until a desired
class count appears.
