# v220 results and final v223 core ablation

Date: 2026-09-23. Input: `25949a44`, branch `worktree-v210-lphc`.

The user authorized at most one more fixed-seed core ablation. This document supersedes the pending-v220 status in docs/243-244. No new seed search, broad method sweep, CF transfer, Deep Forcing fusion or ABA experiment is scheduled.

## Current conclusion

v220 is complete: 64 fixed prompts, 60 seconds, the same `21600 + source_index` seed assignment and frozen method as v219. The strict horizon loader passed on the uploaded package.

All numbers in this table use a 0-100 scale:

| Duration | Method | Imaging | Quality | Subject consistency | Dynamic degree |
|---|---|---:|---:|---:|---:|
| 30s | SF | 69.0480 | 81.6178 | 96.9634 | 42.6042 |
| 30s | Ours | 69.4270 | 81.5518 | 96.9252 | 41.3542 |
| 60s | SF | 66.4866 | 79.3809 | 96.6096 | 31.1458 |
| 60s | Ours | 66.7157 | 79.4773 | 96.6811 | 30.7812 |

- 60s Imaging: +0.2290 points, 95% CI [-0.4213, +0.8958]; 36/64 prompts improve.
- 60s Quality: +0.0964 points, CI [-0.3935, +0.6529]. Subject: +0.0715, CI [-0.1460, +0.3655].
- 60s DD: -0.3646 points, CI [-3.8021, +2.8125]. Overall consistency: -0.1724 points, CI [-0.5731, +0.2105].
- The positive imaging direction persists, but neither this cohort nor its comparison against 30s establishes increasing benefit at longer durations. Two durations are not independent prompt sets.
- The two-seed 30s result in docs/242 remains valid within its stated scope. It is not replaced by selecting the better seed. For all remaining work, keep `21600 + source_index` to reuse existing videos.
- Automatic flags: 10/64 at 30s, 18/64 at 60s. They are heuristic risk flags, not confirmed visual failures, and all flagged prompts remain in the main evaluation.
- v220 contains 48 recovered media receipts without elapsed timing. Existing cost reports exclude them rather than impute timing. Do not claim acceleration or use incomplete/shared-load wall times as pure DiT throughput.

The 60s diagnostic median imaging difference is +0.1436 points; the symmetric trimmed diagnostic mean is +0.1696. Neither diagnostic replaces the full-cohort result. Unlike 30s, the DD-decrease group contributes materially to the net 60s imaging difference. We cannot claim that imaging improvements are independent of motion reduction.

CPU-only reports have already been generated locally:

```text
artifacts/experiment_results/v223_final_closure_25949a44/horizon/
artifacts/experiment_results/v223_final_closure_25949a44/motion/
```

They contain all four 30s methods, both 60s methods, negative contrasts, confidence intervals, risk flags, source hashes and motion diagnostics. No local inference or VBench model execution was performed.

## Paper scope

Begin writing now in parallel with this last ablation. The present narrow claim is a modest observed imaging benefit from local-preserving historical readout correction, with mixed secondary metrics. Do not claim robust overall superiority, improved identity at both durations, validated head classes, retrieval superiority, or absence of failures. No statistical gate is an acceptance guarantee.

The core method remains native FIFO21 plus an auxiliary clean-history archive, retrieved history readout, and a bounded residual correction at phase 0 with alpha 0.02. Random and pooled v219 controls remain in the mechanism table. New ablations may explain the chosen dose/schedule; they do not retroactively establish clipping's separate contribution or the novelty of borrowed components.

## One final batch: v223

| Arm | Alpha | Noisy phases | Descriptor | Action |
|---|---:|---|---|---|
| `sf_fifo21` | disabled | native | native | Reuse v219 video |
| `ours_correct` | 0.02 | phase 0 (`e1`) | headwise | Reuse v219 video |
| `strong_e1` | 0.10 | phase 0 (`e1`) | headwise | Generate once |
| `phase_full` | 0.02 | phases 0,1,2,3 (`full`) | headwise | Generate once |

Use 32 prompts and approximately 30s per video (120 latent frames, 477 decoded frames, 16 fps). Sources are selected by taking every other eight-prompt row in the original 64-prompt v219 placement, not by scores or visual examples. The seed is always `21600 + source_index`, identical across all methods.

Total: **64 new videos + 64 reused videos**. No SF/Ours regeneration. Inference/cache implementation, descriptor, retrieval mode, archive capacity, history budget, local window, model weights and RoPE remain unchanged. Only alpha or schedule changes in the new arms.

Questions answered:

1. Does the smaller correction outperform a moderate five-times-higher correction? This is a dose comparison, not removal of clipping and not direct union attention.
2. Does applying the correction only at the first noisy step compare favorably with applying it at every noisy step? Full-phase has more intervention exposure and compute; this is not a dose/FLOP-matched timing experiment.

Neither ablation is an automatic method-selection tournament. If an ablation wins, report that outcome and qualify the corresponding design claim; do not launch another seed/sweep or silently replace the frozen main results. This subset has already been evaluated and is not a new independent confirmation set.

## Placement, reuse and logs

- Keep all eight original v219 nodes. Each selected prompt stays on its original physical GPU; new jobs reject a different hostname/GPU UUID before launching inference.
- Four selected GPU slots per node (`0,2,4,6`) each run two new methods sequentially in alternating order. Thus 32 of the available 64 GPUs generate the 64 videos in two waves. The other slots are intentionally idle during generation, preserving the established same-device pairing rather than duplicating controls. Evaluation can use all eight slots per node.
- Original SF/Ours media and completion receipts are read in place and hash-checked. They are not restamped as new generation. The new comparison explicitly records cross-batch reference reuse; it is not a latency benchmark.
- Frozen prompts, model/config hashes, inference-operator hashes, evaluator fingerprint, source gates and source completions must match. A missing video or changed operator fails before the new generation batch.
- The existing passed source gates are reused only because the inference operator is unchanged. There is no fabricated v223 `gate0.json`, no additional short smoke video, and no new baseline generation. Each new video still receives the full phase/dose/cache trace audit.
- Invocation environment, stdout/stderr, `trace.jsonl`, `trace_audit`, timestamps and media hashes are preserved for both new arms. `[v223-start]`, `[v223-done]` and `[v223-skip]` identify progress/resume.
- Zero new manual-review pairs are requested by default. Inspect targeted logs only if a new trace/media check fails; automatic flags remain available for later diagnosis.

## Commands

Use a clean analysis/generation checkout for v223, not a checkout still running old jobs. Freeze its commit after preparation.

```bash
git fetch origin
git worktree add --detach ../training-free-v223 origin/worktree-v210-lphc
cd ../training-free-v223

export V223_V219_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64
export V223_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v223_runs/v223_core32
export GPU_LIST=0,1,2,3,4,5,6,7
export NODE_RANK=0

# Run once on rank 0. These checks do not generate videos.
bash scripts/run_v223_experiment.sh freeze
bash scripts/run_v223_experiment.sh prepare
bash scripts/run_v223_experiment.sh schedule
```

The wrapper uses the established `CONDA_SH`, `CONDA_ENV`, `SHARED_CHECKPOINT`, `WAN_MODEL`, prompt and VBench defaults. Retain the same actual environment and model files as v219; set those variables explicitly if your deployment uses different paths.

On each of the eight original nodes, set its rank `0..7` and corresponding frozen IP. Reading the IP from the manifest does not bypass the local-interface check:

```bash
# Set V223_OUT_ROOT and the SAME checkout path/environment on every node.
# NODE_RANK must be that node's ORIGINAL v219 rank, not zero on every node.
export V223_NODE_ADDRESS=$(python -c 'import json,os; from pathlib import Path; d=json.loads((Path(os.environ["V223_OUT_ROOT"])/"inputs/selection.json").read_text()); print(d["authorized_nodes"][int(os.environ["NODE_RANK"])])')
bash scripts/run_v223_experiment.sh generate32
```

No `baseline`, `gate0` or `smoke` action is needed or permitted for v223. Resume the same command on the same node/GPU; validated finished jobs are skipped. Do not update the generation checkout after preparing it.

After all new jobs finish:

```bash
# Once on rank 0:
bash scripts/run_v223_experiment.sh status
bash scripts/run_v223_experiment.sh publish

# On every rank 0..7, with the same V223_NODE_ADDRESS binding:
bash scripts/run_v223_experiment.sh split
bash scripts/run_v223_experiment.sh eval-missing

# Once on rank 0, after all metric parts finish:
bash scripts/run_v223_experiment.sh collect
bash scripts/run_v223_experiment.sh analyze
bash scripts/run_v223_experiment.sh package
```

For simplicity and consistent subset indexing, VBench re-evaluates all four arms on these 32 prompts; **only the two ablation arms regenerate videos**. Existing v219/v220 scores are not overwritten. The evaluator must remain the repaired pinned version/checkpoint; no new metric or seed selection is introduced.

Return `v223_small_artifacts.tar.gz`, especially `evaluation/analysis/v223_core_ablation.{json,md,csv}`, `inputs/selection.json`, `decisions/reference_reuse.json`, comparison manifest, metric receipts and new generation logs/traces. Keep the original v219 package available because reused completion paths point there.

## Stop rule

This is the last authorized new-generation batch. Main tables use the completed 64-prompt 30s/60s cohorts; the 32-prompt ablation table is separate. Write the four-page paper concurrently, leaving only the two ablation entries pending. Do not expand to additional seeds or components after inspecting v223.

If resources or remaining time cannot accommodate the two generation waves and evaluation, omit this optional ablation and narrow the design claims. Do not block the already available main-results draft on new experiments.

## Verification

CPU checks: 76 tests passed, one skipped because Windows lacks the symlink privilege needed for the publication-link test. Covered fixed membership/seed, single-factor specifications, original-device placement, environment cleanup, source-gate failures, changed media/receipts, forbidden regeneration, operator/config drift, and retention of unfavorable controls. Existing v212/v219/v220 and reporting tests were included.

Python compilation and Bash syntax checks passed. GPU generation and VBench execution were not run locally; the server remains the execution environment. No files under `src/` or the Self-Forcing inference implementation were edited.
