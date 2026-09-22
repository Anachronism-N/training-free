# Fixed-seed paper closure: no new GPU sweep

Date: 2026-09-23, approximately 03:58 Asia/Shanghai.

The requested three-hour follow-up pulled `worktree-v210-lphc` successfully. The remote still contained `3fce4726` (v219 analysis); no v220 result upload was available. This document does not report any v220 result or infer whether its server jobs have completed.

## Decision and scope

Stop adding seeds. Keep `21600 + source_index` for the existing v219/v220 comparison, with the same seed for every method on a prompt. This choice preserves the already generated 30s/60s pairing, not a retrospective best-seed search. Previously evaluated seeds remain evidence; development-selected seeds must not be described as independent confirmation.

Finish the already running 64-prompt SF/Ours 60s experiment. Reuse every valid completed video. No extra GPU generation, new prompt suite, broader manual review, or new seed is requested in this batch.

New code: `scripts/export_lphc_horizon_evidence.py`. It closes the reporting gap between the completed v219 study and pending v220, without importing a GPU model or rerunning VBench. The existing v220 generator/evaluator and frozen method are unchanged.

## What the exporter checks and reports

- Verifies each campaign's generation-completion identities, same-GPU pairing records, trace-audit receipts, baseline/zero gates, summary and diagnostic hashes, and frozen method/endpoint.
- Uses the embedded v216 selection and development evidence. It does not require the full v216 server directory or manufacture v216's missing gate receipts. The standalone v219/v220 report does not certify those missing v216 checks.
- Checks the same prompt text/hash/order, seed, model, cache operator, method configuration and evaluator across horizons.
- Verifies that v220 really represents 240 latent frames, 957 decoded frames, and all 30 evaluation clips. A 15-clip report cannot be relabeled as 60s.
- Preserves all four v219 methods and both v220 methods. Produces all nine raw VBench dimensions and official Quality in CSV; does not fabricate full Semantic/Total scores.
- Retains full/early/late paired comparisons and their available confidence intervals. v219 did not store paired raw contrasts for every dimension, so missing intervals are not invented from aggregate means.
- Computes the descriptive paired interaction `(Ours60 - SF60) - (Ours30 - SF30)`, resampling 64 prompts. It does not average the two durations into a new primary score, count clips as independent samples, or assume the two rollouts share an identical prefix.
- Preserves all automatic risk flags, memory and wall-time summaries. Flags are not confirmed visual failures; wall time under shared load is not isolated DiT throughput.
- Requests zero new human-review pairs by default. `--review-pair-limit 2` optionally exports at most two already selected diagnostic pairs, with reviewed-ID deduplication and the existing total-six cap.

## Run without disturbing v220

Do not pull analysis changes into an actively running generation checkout: jobs bind to its original source commit. Use a separate worktree after fetching, or run this exporter locally on the uploaded compact artifacts.

```bash
# Run from a repository clone; this does not change the active checkout's HEAD.
git fetch origin
git worktree add --detach ../training-free-closure-v221 origin/worktree-v210-lphc
cd ../training-free-closure-v221

export V219_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64
# V220_OUT_ROOT must point to the existing v220 run, not a new output directory.
: "${V220_OUT_ROOT:?Set this to the completed v220 run directory}"

python scripts/export_lphc_horizon_evidence.py \
  --v219-root "$V219_OUT_ROOT" \
  --v220-root "$V220_OUT_ROOT" \
  --output-root "$V220_OUT_ROOT/paper_closure_v221"
```

This runs on one CPU node; do not launch it on all eight nodes. It only reads the run artifacts and creates a new output directory. It never launches generation, edits manifests, replaces metric results, or chooses a method/seed winner.

While v220 is pending, omit `--v220-root` and use a different output directory:

```bash
python scripts/export_lphc_horizon_evidence.py \
  --v219-root "$V219_OUT_ROOT" \
  --output-root "$V219_OUT_ROOT/paper_closure_v221_pending"
```

The pending report explicitly says `v220_pending`. Outputs are immutable: use a fresh output directory when adding v220, instead of overwriting the pending report.

## Files to return

- `horizon_evidence.json`: primary results, all paired contrasts, risks, costs and source hashes.
- `horizon_evidence.md`: concise summary and limitations.
- `main_tables.csv`: raw metric table, with 30s and 60s rows kept separate.
- `paired_contrasts.csv`: all available Ours-versus-control comparisons, including unfavorable ones.
- `horizon_interactions.csv`: descriptive duration interactions, produced only when v220 is present.
- `review_queue.json`, `review_queue_ids.json`: empty by default; optional diagnostic pairs only.

Return the v220 compact package described in docs/241 as well. Existing packaging remains:

```bash
# In the original frozen run checkout, after its evaluation has completed:
bash scripts/run_v220_experiment.sh collect
bash scripts/run_v220_experiment.sh analyze
bash scripts/run_v220_experiment.sh package
```

Export failures are actionable: missing gates/metrics require uploading or completing those existing artifacts; a hash/model/config mismatch requires inspection, not changing the expected hash or rerunning all videos automatically.

## Minimal remaining experimental plan

1. Complete v220 and run the exporter once. Reconcile the two original missing v216 gate receipts without regeneration.
2. If 60s supports a useful imaging benefit with acceptable tradeoffs, freeze the local-preserving correction method and proceed to the four-page draft. Keep random/pooled as concise mechanism controls; do not claim they have already been beaten reliably.
3. If the 60s estimate remains uncertain, use qualified effect estimates and existing evidence rather than launching more seeds until an interval turns positive. A confidence threshold is not an automatic acceptance rule.
4. If clear degradation or a real artifact appears, diagnose its corresponding logs first. Only fix and repeat the affected experiment when there is a concrete implementation defect. Do not remove its prompt from the reported cohort.
5. Additional participation-based archive selection, CF transfer, Deep Forcing comparisons, new head routing and ABA are not prerequisites for this closure. They remain separate follow-up work, not changes to the running method or guaranteed improvements.

The paper can emphasize the favorable imaging result and local-preserving design without presenting every abandoned historical experiment. It must still retain directly relevant controls and distinguish selected qualitative examples from population-level evidence. No new claim of identity, motion, retrieval, speed or overall-quality superiority follows automatically from the imaging endpoint.

## Verification and current output

- CPU verification: 45 tests passed across `test_lphc_horizon_evidence.py`, `test_lphc_paper_evidence.py`, `test_v219_closure.py` and `test_v220_long_horizon.py`.
- Covered missing gates, mismatched prompts/seeds/configurations, incomplete 60s metadata, preservation of controls/flags, prompt-paired duration interactions and bounded optional review.
- Ran the exporter on the actual uploaded v219 compact artifacts. It produced four method rows and 81 paired comparisons, with status `v220_pending` and zero requested review pairs.
- Outputs: `artifacts/experiment_results/v221_closure_pending_3fce4726/`. This is an analysis of existing v219 results, not a new v221 generation experiment.
- No GPU inference or VBench recomputation was performed locally; the new 60s export path has synthetic-contract tests but still awaits the real v220 upload.
