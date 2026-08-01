# 153: v151 Run Results and Analyze-Gate Failure Diagnosis

Date: 2026-08-01
Cluster: 4 nodes x 8x H20 (32 GPUs)
- node0: 29.232.229.115
- node1: 29.119.98.254
- node2: 29.119.98.54
- node3: 29.127.36.158

## 1. Summary

v151 was run end-to-end under a supervisor that releases and re-acquires the
GPU occupy layer around each stage, restarts failed stages up to three times,
and re-occupies all 32 GPUs when the experiment ends. The generation phase
(`signed_analyze`, `prepare`, `smoke`, `core64`) completed successfully on the
first attempt. The `analyze` stage failed its `audit` gate on all three
attempts because 70 of 245,760 calibration layers (0.028%) exceeded the frozen
2% relative-error threshold. The failure is a deterministic late-denoising
calibration-convergence limit, not a process crash and not a head-group
scientific signal. The 64 profiles and 64 videos are preserved.

This document records the run, the diagnosis, and the options. It does not
relax the 2% threshold (see `docs/152` section 8).

## 2. Configuration Actually Run

The working tree at run time used `CALIBRATION_REFINEMENT_STEPS = 8` (raised
from the committed 4 in `scripts/build_v151_signed_policy_low_tail_suite.py`,
`scripts/run_v151_signed_policy_low_tail_32gpu.sh`, and
`tests/test_v151_signed_policy_suite.py`). All other frozen contracts from
`docs/152` are unchanged:

- calibration target: projected relative RMS `0.02`;
- scale range `[0.005, 50]`;
- 30 layers, 4 heads per map per layer;
- 32 probes (6 fixed groups x 4 interventions + 8 random maps x uniform);
- 4 contexts at nominal timesteps `1000, 750, 500, 250`;
- 64 profiles = 32 prompts x 2 seeds, 120 latent frames;
- checkpoint: `/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt`;
- conda env: `longlive` (torch 2.10.0+cu128).

The smoke check in the runner uses a relaxed `error <= 0.025` for its
diagnostic pass; the `audit` gate invoked by `analyze` still requires
`max_error <= 0.02`. This is the gate that failed.

## 3. Run Timeline

Supervisor log: `/tmp/v151_supervisor/supervisor.log`.

| Stage | Node | Start | End | rc | Note |
|---|---|---|---|---|---|
| 1 signed_analyze | 0 (CPU) | 16:07:18 | 16:08:41 | 0 | signed source screen PASS |
| 2 prepare | 0 (CPU) | 16:08:41 | 16:08:45 | 0 | 64 jobs / 32 prompts / 32 probes |
| 3 smoke | 0 (GPU 0-3) | 16:08:53 | 16:16:09 | 0 | 4 videos, replay/map/calibration PASS |
| 4 core64 | 0 | 16:16:44 | 16:27:54 | 0 | 8 shards |
| 4 core64 | 1 | 16:16:44 | 16:27:59 | 0 | 8 shards |
| 4 core64 | 2 | 16:16:44 | 16:28:09 | 0 | 8 shards |
| 4 core64 | 3 | 16:16:44 | 16:28:00 | 0 | 8 shards |
| 5 analyze | 0 (CPU) | 16:28:10 | 16:30:03 | 1 | audit AssertionError |
| 5 analyze (retry 2) | 0 | 16:30:08 | 16:32:24 | 1 | identical |
| 5 analyze (retry 3) | 0 | 16:32:29 | 16:34:22 | 1 | identical |

`core64` produced all 64 profiles and 64 videos in about 12 minutes across the
four nodes. After `core64` the supervisor re-occupied all 32 GPUs before
attempting `analyze`, so utilization was maintained throughout the CPU-only
analysis phase.

## 4. Analyze-Gate Failure

`analyze` invokes `audit` first. `audit` loads all 64 profiles and asserts
per-layer calibration integrity. The failing assertion is the audit heredoc
line corresponding to `assert max_error <= 0.02` (reported as `<stdin>:48`
because the heredoc's first Python line is `import sys`). The traceback is a
bare `AssertionError` with no message, repeated identically across all three
attempts.

Aggregate calibration statistics over all 245,760 non-native-replay layers
(64 profiles x 128 probe rows x 30 layers):

| Metric | Value | Gate |
|---|---|---|
| `max_replay` | 0 | <= 1e-4 PASS |
| clipped count | 0 | == 0 PASS |
| degenerate count | 0 | == 0 PASS |
| refinement_bound_hit count | 0 | == 0 PASS |
| `max_error` | 0.0287735 | <= 0.02 **FAIL** |
| 99th percentile error | 0.0093339 | - |
| layers with error > 0.02 | 70 / 245,760 (0.028%) | - |

The worst layer is prompt slot 27, seed 0, probe
`signed_high4_key_shift_t020`, policy `key_shift`, layer 0, error 0.0287735,
scale 3.4962.

## 5. Root Cause: Late-Denoising Convergence Limit

The 70 offending layers share three properties that identify a generic
numerical limit rather than a scientific signal:

1. **All offending probes are at the `_t020` context (nominal timestep 250).**
   No offending layer appears at t1000, t750, or t500. The late-denoising
   state is the hardest to calibrate to 2% within 8 refinement steps.

2. **Offending layers span every head group.** The 70 layers distribute across
   scalar_low4, scalar_middle4, scalar_high4, signed_low4, signed_middle4,
   signed_high4, and the random0..7 maps. The per-group counts:

   | probe (t020) | policy | offending layers | probe max err |
   |---|---|---|---|
   | scalar_low4_uniform | policy_contrast | 8 | 0.0235866 |
   | scalar_high4_key_shift | key_shift | 6 | 0.0249681 |
   | scalar_high4_value_shift | value_shift | 5 | 0.0240994 |
   | signed_middle4_uniform | policy_contrast | 5 | 0.0215508 |
   | random2_uniform | policy_contrast | 5 | 0.021744 |
   | scalar_low4_key_shift | key_shift | 5 | 0.0214459 |
   | random5_uniform | policy_contrast | 5 | 0.0219712 |
   | scalar_high4_boundary | policy_contrast | 4 | 0.0272573 |
   | signed_high4_key_shift | key_shift | 4 | 0.0287735 |
   | random6_uniform | policy_contrast | 3 | 0.0240758 |
   | scalar_high4_uniform | policy_contrast | 3 | 0.0238298 |
   | signed_low4_value_shift | value_shift | 3 | 0.0212485 |
   | signed_low4_uniform | policy_contrast | 3 | 0.0246616 |
   | (12 more probes) | - | 1-2 each | 0.0200-0.0214 |

   If this were a head-group-specific policy-leverage signal, the offending
   layers would concentrate in one rank group. They do not.

3. **Errors cluster just above the threshold.** 55 of the 70 layers sit between
   0.0200 and 0.0250; only 5 exceed 0.025. The 99th percentile across all layers
   is 0.0093, well below 2%. This is a thin tail of near-misses, consistent
   with a refinement step budget that is one or two iterations short for a
   small fraction of late-denoising layers.

30 of 32 prompt slots contribute at least one offending layer, and both seeds
are represented. There is no prompt-specific outlier.

## 6. Conclusion

- `core64` succeeded: 64 native Self-Forcing trajectories and 132 downstream
  probe records per profile are intact and replay-valid (`max_replay = 0`).
- `analyze` cannot run because `audit` enforces `max_error <= 0.02` and 70
  t020 layers miss it by a thin margin.
- The miss is a generic 8-step refinement convergence limit at the
  late-denoising timestep. It is not evidence for or against the v151
  scalar-low-tail or signed-policy hypotheses.
- Retrying the run cannot help: the generation is seed-frozen and deterministic,
  so the three `analyze` attempts produced byte-identical failures.

## 7. Preserved Artifacts

```text
runs/v151_signed_policy_low_tail/
|-- signed_source/            # signed source screen (PASS)
|-- inputs/                   # prompts, manifest, probe plan, suite metadata
|-- smoke/                    # 4 smoke videos + profiles
`-- core64/
    |-- profiles/             # 64 .pt profiles (each 132 downstream records)
    |-- videos/               # 64 .mp4 native SF trajectories
    `-- logs/                 # 32 shard logs
```

`docs/results/v151_signed_policy_low_tail/` was not created because `package`
never ran. The analysis report (`core64/analysis/report.json`) is absent for
the same reason.

Supervisor script: `/apdcephfs_gy2/share_303214315/cedricnie/v151_supervisor.sh`
(outside the repo). Supervisor and per-stage logs: `/tmp/v151_supervisor/`.

## 8. Options (do not relax the 2% threshold)

Per `docs/152` section 8, the 2% calibration threshold is frozen. The fix
belongs in the calibration path or in a separately preregistered target.

1. **Increase refinement steps for the t020 context only.** Raise
   `CALIBRATION_REFINEMENT_STEPS` from 8 to 16 (or context-conditioned) and
   re-run only the affected probes. The other three contexts already pass at
   8 steps, so this targets the demonstrated shortfall without weakening the
   gate.

2. **Improve calibration initialization at t020.** Investigate why the
   late-denoising state produces a harder projection target; a better initial
   scale or a two-stage refinement may recover 2% within the existing step
   budget.

3. **Separately preregister a t020 target.** Following the `docs/152` escape
   hatch, register a distinct target for the t250 context before re-running,
   keeping the 2% gate for t1000/t750/t500. This must be decided before
   looking at group-level results to avoid post-hoc bias.

4. **Analysis-only tolerance with audit re-gate.** Keep the 2% audit as the
   generation contract, but allow the analysis script to exclude the 70
   flagged layers (0.028%) from group comparisons and report sensitivity.
   This does not relax generation; it weakens the analysis scope and must be
   approved explicitly.

The 64 profiles and videos are sufficient for any of the above without
regeneration, because options 1-2 only re-run the read-only one-step
calibration probes (no new video generation), and options 3-4 are pure
analysis.

## 9. Supervisor Behavior (for reproducibility)

The supervisor implements the user's run contract: run v151, restart on
failure, re-occupy GPUs at the end, and keep utilization high throughout.

- CPU-only stages (`signed_analyze`, `prepare`, `analyze`, `package`) run with
  all 32 GPUs occupied by `occupy_all_gpu.py`.
- GPU stages release the relevant nodes first: `smoke` releases node 0;
  `core64` releases all four nodes. After each GPU stage the supervisor
  re-occupies before continuing.
- Each stage retries up to 3 times. `core64` retries per node independently;
  a failed node does not abort the other three.
- On fatal failure the supervisor re-occupies all 32 GPUs and exits non-zero.

In this run `core64` passed on the first attempt for every node, and the only
retries were the three identical `analyze` attempts.
