# Fixed-seed final diagnostics and experiment freeze

Date: 2026-09-23. Input repository commit: `2b97a304`, branch `worktree-v210-lphc`.

## Decision

No more generation seeds or broad method search for this submission. Keep the existing per-prompt seed `21600 + source_index` for the v219/v220 comparison. This reuses the paired videos, not a retrospective best-seed selection. Earlier seed results remain available; a selected illustration must not be described as a randomly sampled example.

The latest remote contains completed v219 and the v221 reporting utility, but no uploaded v220 package. This does not establish the state of server jobs. Finish the existing v220 run without changing its method, configuration, source checkout or output directory.

The next experiment is a **zero-generation diagnostic of existing results**. It addresses two remaining questions:

1. Is the positive imaging estimate concentrated in videos whose measured motion decreases?
2. Is that estimate supported only by one or two exceptionally favorable prompts?

This is not a new video-generation method, not a new primary metric, and not a way to filter the evaluation set. No GPU inference or additional manual review is needed.

## New code

`scripts/analyze_lphc_motion_tradeoff.py` reads the verified v219 package and optionally the completed v220 package. It uses the existing strict horizon loader, including frozen-method, parent-selection, model, prompt, seed, gate, evaluator and completion-receipt checks. When both durations are supplied, their prompt/seed/configuration bindings must match.

It preserves all six existing v219 method contrasts and all three windows (full, early, late). v220 contributes only its actual Ours/SF contrast, with the same three windows. Nothing is inferred about missing random/pooled 60s arms.

For each contrast/window:

- Partition prompts by the observed paired Dynamic Degree difference: negative, numerically tied, positive. The fixed `1e-9` tolerance handles numerical zero, not a tuned physical motion threshold.
- Report each group's prompt count, imaging difference, original Quality/subject differences, and contribution to the full imaging mean.
- Each contribution is `sum(group imaging differences) / all_prompt_count`. The three contributions must reconstruct the original full mean; group means must not be averaged with equal weights.
- Retain the original full-cohort differences and confidence intervals unchanged.
- Report median imaging difference, symmetric approximately 10% trimmed mean, leave-one-prompt-out sensitivity and top-positive contribution concentration. These are diagnostics, not alternative reported cohorts.
- Export every prompt's identity, effective seed and paired differences to CSV. Preserve all automatic risk flags separately.

Subgroup confidence intervals resample prompts, not frames or clips. Empty groups have no invented mean/interval; singleton groups have no interval. The program requests zero new review pairs and never launches generation or VBench models.

## Actual v219 findings

The new script was run on:

```text
artifacts/experiment_results/v219_mechanism64_38a06347_vbench_45e79ec1
```

All following differences are on a 0-100 scale. CSV/JSON keep raw Imaging/DD/Subject differences on 0-1 and official Quality on 0-100.

| Observed Ours - SF DD change | Prompts | Mean imaging difference | Contribution to full imaging difference |
|---|---:|---:|---:|
| Decreases | 16 | +0.0097 | +0.0024 |
| Tied | 39 | +0.3712 | +0.2262 |
| Increases | 9 | +1.0695 | +0.1504 |
| All | 64 | +0.3790 | +0.3790 |

The observed imaging gain is not concentrated in the measured-DD-decrease group. This is useful descriptive evidence against the simplest explanation that all imaging benefit comes from reduced DD. It does **not** establish causal independence from motion: groups are defined by post-treatment outcomes, and unchanged DD can hide changes in flow magnitude or motion quality.

Imaging sensitivity, Ours minus SF:

- Mean +0.3790 points, median +0.0627 points, positive on 36/64 prompts.
- Removing six values from each tail for a diagnostic trimmed mean gives +0.2473 points. All 64 remain in the main result.
- Diagnostic removal of the single strongest positive prompt leaves +0.2278 points; removal of the three strongest leaves +0.0712 points. This does not make either reduced cohort a confirmation experiment.
- Sources 113, 114 and 85 contribute 41.1% of positive-difference mass (not 41.1% of net gain). The positive mean is not solely due to one prompt, but its magnitude is noticeably influenced by favorable cases.
- Source 85 also has an automatic temporal-discontinuity flag. High imaging scores must not be treated as proof of a good video or used to bypass that flag.

The original v219 full imaging CI still includes zero. Overall Quality remains slightly negative, and DD is -1.25 points on average. Neither this diagnostic nor its subgroup intervals replaces those results. Random and pooled controls are retained; headwise retrieval's independent advantage remains unresolved.

The positive two-seed imaging result from docs/242 may be reported with its original scope. This batch does not add seeds, pool horizons, or choose a better-performing seed per method.

## Minimal remaining server work

### A. Finish only the existing v220 jobs

Use the original frozen v220 checkout and its existing environment/output root. Do not pull new analysis changes into an active generation checkout.

```bash
: "${V220_OUT_ROOT:?Use the EXISTING v220 run directory}"
bash scripts/run_v220_experiment.sh status
bash scripts/run_v220_experiment.sh eval-status
```

If generation is complete but evaluation is incomplete, use the original documented publish/split workflow from docs/241. Resume `eval-missing` once on each configured node with its original `NODE_RANK=0..7` and GPU list. It reuses validated completed metric jobs; do not launch eight ranks on one host or start a second worker over an already running rank.

Once all metric jobs have completed, run once on rank 0 in that original checkout:

```bash
bash scripts/run_v220_experiment.sh collect
bash scripts/run_v220_experiment.sh analyze
bash scripts/run_v220_experiment.sh package
```

If generation is incomplete, resume only the existing `generate64` workers on their original nodes/devices. Do not create a new run or new seed to replace missing videos.

### B. Export the final tables and new diagnostics

Use a separate analysis worktree or the local clone containing uploaded results. Both commands are CPU-only and run on one node, not all eight nodes.

```bash
export V219_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64
: "${V220_OUT_ROOT:?Use the EXISTING completed v220 run directory}"

python scripts/export_lphc_horizon_evidence.py \
  --v219-root "$V219_OUT_ROOT" --v220-root "$V220_OUT_ROOT" \
  --output-root "$V220_OUT_ROOT/paper_closure_v222"

python scripts/analyze_lphc_motion_tradeoff.py \
  --v219-root "$V219_OUT_ROOT" --v220-root "$V220_OUT_ROOT" \
  --output-root "$V220_OUT_ROOT/motion_diagnostic_v222"
```

While v220 is pending, omit `--v220-root` and use a different output directory. A supplied but incomplete v220 package fails validation; it is not silently ignored. Outputs are immutable, so the completed report must not overwrite a different pending report.

Return the original v220 small-artifact package plus these two output directories. Also return the original v216 `decisions/sf_upstream_gate.json` and `decisions/gate0.json` when available. No video regeneration is needed to fill those missing receipts.

### C. Stop expanding the experiment matrix

Keep 30s/60s, SF/Ours, and the already completed random/pooled controls as the short-paper evidence set. CF transfer, Deep Forcing fusion, new head routing, ABA, additional seeds and new thresholds are not scheduled in this batch. This avoids changing the method while its main experiment is finishing.

Begin the methods/protocol/results draft now, with unfilled 60s cells explicitly pending. If v220 supports a useful imaging gain with acceptable tradeoffs, finalize a narrow local-preserving historical correction story. If it does not, report the limited result or revise the submission plan rather than search seeds until one wins. This is an experimental decision, not an automatic acceptance rule.

## Outputs and checks

Already produced locally, without inference:

```text
artifacts/experiment_results/v222_motion_diagnostic_2b97a304/
  motion_imaging.json
  motion_imaging.md
  motion_groups.csv
  paired_prompt_deltas.csv
```

There are 18 method/window analyses, 54 motion groups, and 1,152 per-prompt contrast rows. The rows are repeated measurements on 64 prompts, not 1,152 independent observations.

CPU verification: 62 tests passed across the new diagnostic, horizon exporter, paper exporter, v219 and v220 suites. The new tests cover subgroup accounting, numerical ties, empty/singleton groups, invalid contrasts, outlier sensitivity, prompt/seed identity, immutable outputs, preservation of controls/risks and separate duration rows. Real v219 artifacts passed the strict loader. The real v220 path still awaits its upload.

No inference implementation, cache policy, source manifest, frozen endpoint, active v220 configuration or original result file was changed.
