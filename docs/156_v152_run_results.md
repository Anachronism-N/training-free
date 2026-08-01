# 156: v152 Online Policy Profiling — Run Results

Date: 2026-08-01
Cluster: 4 nodes x 8x H20 (32 GPUs)
- node0: 29.232.229.115
- node1: 29.119.98.254
- node2: 29.119.98.54
- node3: 29.127.36.158

## 1. Summary

v152 (online state-conditioned policy profiling) ran end-to-end under a
supervisor that releases and re-acquires the GPU occupy layer around GPU
stages, retries failed stages up to three times, and re-occupies all 32 GPUs
at the end. All seven stages passed on the first attempt. The audit gate was
clean: 128 profiles, 6144 policy pairs, 92160 selectors, native replay RMS 0.

The scientific result is negative. The oracle selector (native
policy-error-margin) cannot choose between `uniform8` and `recent8` better
than chance, so no online score — QK or old-mass — can be confirmed. Per
`docs/155` section 6: "Oracle fails -> Local policy preference does not
propagate; stop this policy axis."

## 2. Configuration

Commit `84013bb` on `codex/v98-correctness-fixes`. Same four-node cluster and
checkpoint as v151 (`/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt`,
conda env `longlive`). v152 reuses the v151 signed map and suite metadata.

- 64 Qwen-rewritten MovieBench prompts (disjoint from v150 and v151);
- 2 new seeds = 128 native 30-second profiles (120 latent frames);
- frame 117, timesteps 1000/750/500/250;
- 24 probes (12 groups x 2 policies) + native replay = 100 downstream records
  per profile;
- 4 jobs per GPU x 32 GPUs;
- frozen-native selector protocol: scores computed on native state, head ids
  frozen, both policy replays verified byte-identical scores.

## 3. Run Timeline

Supervisor log: `/tmp/v152_supervisor/supervisor.log`.

| Stage | Node | Start | End | rc | Note |
|---|---|---|---|---|---|
| 1 prepare | 0 (CPU) | 19:30:43 | 19:30:47 | 0 | suite build |
| 2 preflight | 0 (CPU) | 19:30:47 | 19:31:02 | 0 | 9 tests PASS, selector plan PASS |
| 3 smoke | 0 (GPU 0-3) | 19:31:09 | 19:37:35 | 0 | `[v152-audit] PASS profiles=4 replay=0`; `[v152-smoke] frozen selector...PASS` |
| 4 core128 | 0 | 19:38:05 | 19:53:17 | 0 | 32 shards |
| 4 core128 | 1 | 19:38:05 | 19:54:08 | 0 | 32 shards |
| 4 core128 | 2 | 19:38:05 | 19:54:06 | 0 | 32 shards |
| 4 core128 | 3 | 19:38:05 | 19:53:58 | 0 | 32 shards |
| 5 audit | 0 (CPU) | 19:54:09 | 19:55:53 | 0 | `[v152-audit] PASS profiles=128 pairs=6144 selectors=92160 replay=0` |
| 6 analyze | 0 (CPU) | 19:55:53 | 20:00:06 | 0 | report.json + report.md |
| 7 package | 0 (CPU) | 20:00:06 | 20:00:06 | 0 | copied to docs/results/ |

Total wall time ~30 min. `core128` produced 128 profiles and 128 videos in
~16 min across the four nodes at 96-98% GPU utilization. No stage needed a
retry.

## 4. Audit

`[v152-audit] PASS profiles=128 pairs=6144 selectors=92160 replay=0`

All 128 profiles passed the structural, prompt/seed, replay, probe-grid, and
selector-freezing contracts. Native replay maximum relative RMS is 0. This is
a clean artifact set — a clear contrast with v151, where the 2% calibration
gate failed across all four contexts.

## 5. Gate Results

```json
{
  "g0_native_replay_and_contract": true,
  "g1_oracle_policy_choice": false,
  "g2_qk_policy_choice": false,
  "g3_qk_beats_count_matched_random": false,
  "g4_qk_matches_oracle": true,
  "g5_online_qk_candidate_confirmed": false,
  "g6_old_mass_baseline": false
}
```

Qualifying contexts (where a gate's precondition holds):

```text
qk_oracle_alignment:   [noisy_f117_t1000, noisy_f117_t750, noisy_f117_t500, noisy_f117_t250]
oracle_policy_choice:  []
qk_policy_choice:      []
qk_beats_random:       []
old_mass_policy_choice: []
online_qk_candidate:   []
```

### 5.1 What passed

- `g0`: native replay and the frozen-selector contract hold. The dynamic
  classification was computed on the native state and frozen before either
  policy replay; both replays used byte-identical scores and head ids.
- `g4`: the QK score (`qk_policy_margin`) aligns with the oracle score
  (`policy_error_margin`) in all four contexts. The QK score is a faithful
  cheap proxy for the oracle ranking.

### 5.2 What failed

- `g1` oracle policy choice: the oracle selector — which uses the true native
  policy error to pick which heads should keep uniform history — does not
  produce a group whose `uniform8` approximation beats `recent8` on X0 error.
  In other words, even with perfect policy-error knowledge, selecting heads
  for uniform sampling does not improve the downstream approximation.
- `g2` QK policy choice: since the oracle fails, the QK proxy cannot do
  better; its uniform4/recent4 groups do not separate by policy preference.
- `g3` QK beats count-matched random: the QK-selected groups do not
  outperform direction-matched random maps.
- `g5` online QK candidate confirmed: the conjunctive gate (g1-g4 in one
  same context) fails because g1 fails.
- `g6` old-mass baseline: the temporal-mass selector also fails to produce a
  policy-preference group.

## 6. Selection Recurrence

```json
{
  "qk_uniform4": { "adjacent_timestep_median_jaccard": 1.0, "seed_median_jaccard": 0.6 },
  "qk_recent4":  { "adjacent_timestep_median_jaccard": 0.6, "seed_median_jaccard": 0.333 }
}
```

The QK-uniform selection is perfectly stable across adjacent timesteps
(Jaccard 1.0) and moderately stable across seeds (0.6). This means the QK
score picks a consistent head set, but that set does not carry policy-specific
leverage — consistency without utility.

## 7. Conclusion

Per `docs/155` section 6 decision table:

> Oracle fails -> Local policy preference does not propagate; stop this
> policy axis.

The v152 hypothesis was: given the current native state, an online QK score
can choose which heads retain uniformly sampled history versus recent-only
history, and this choice improves the approximation to native full-window
Self-Forcing. The oracle test rejects the premise: even the non-deployable
oracle (true policy error) cannot identify heads where uniform sampling
helps. The QK score faithfully tracks the oracle (g4), but tracking a
non-working oracle is not useful.

Both v151 (static per-layer maps) and v152 (online state-conditioned QK
selection) are now negative. The static maps identify response magnitude, not
policy-specific function (v151 conclusion), and the online QK score tracks
that same non-propagating signal (v152 conclusion).

## 8. Preserved Artifacts

```text
docs/results/v152_online_policy_profile/
|-- v152_probe_plan.json
|-- suite_metadata.json
`-- core/
    |-- report.json
    |-- report.md
    |-- profile_audit.csv
    |-- policy_pair_summary.csv
    |-- policy_pair_effects.csv.gz
    |-- random_control_summary.csv
    |-- random_control_effects.csv.gz
    |-- probe_observations.csv.gz
    |-- selector_alignment_summary.csv
    |-- selector_alignment.csv.gz
    |-- selector_recurrence_summary.csv
    |-- selector_recurrence.csv.gz
    `-- selector_snapshots.csv.gz
```

Raw profiles and videos remain in `runs/v152_online_policy_profile/core128/`
(not committed; `runs/` is gitignored).

## 9. Supervisor Notes

The supervisor (`/apdcephfs_gy2/share_303214315/cedricnie/v152_supervisor.sh`)
coordinated GPU occupy release/re-acquire around `smoke` and `core128`, ran
all CPU-only stages with occupy active, and retried each stage/node up to
three times. No retries were needed.

One operational issue: the supervisor's occupy check used
`pgrep -f 'python3.*occupy_all_gpu'`, which false-matches the SSH shell
command containing that pattern. After `core128`, this left node 0 idle
briefly during the audit stage until the occupy was manually restarted. The
reliable occupancy check is `nvidia-smi` memory usage (>500 MB = occupied),
not `pgrep`. This is recorded for future supervisor revisions.
