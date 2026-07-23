# v81 ProbeCache Screen: DINOv2 Results (Partial)

> Date: 2026-07-24
> Experiment: `runs/v81_probecache_single/`
> 9 complete cells × 12 prompts × 120 frames (≈30s) × seed 0
> 3 cells incomplete (ours_full, ours_audit, ours_persistent — killed by pipe)
> Head profile: 99 persistent (27.5%), 261 reactive (72.5%)

## 1. DINOv2 metrics (sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| ours_topk2 | 0.8269 | 0.7401 | -0.00288 | 0.2797 | 0.9040 | 0.5052 |
| pf_official | 0.8263 | 0.7641 | -0.00175 | 0.2744 | 0.9025 | 0.5127 |
| ours_topk6 | 0.8205 | 0.7588 | -0.00283 | 0.2877 | 0.8992 | 0.5038 |
| ours_prompt0 | 0.8205 | 0.7184 | -0.00353 | 0.2729 | 0.8979 | 0.4969 |
| ours_open_gate | 0.8169 | 0.7505 | -0.00294 | 0.2960 | 0.8984 | 0.4995 |
| ours_conservative | 0.8165 | 0.7279 | -0.00288 | 0.2896 | 0.8895 | 0.4984 |
| ours_archive12 | 0.8158 | 0.7184 | -0.00351 | 0.2828 | 0.9015 | 0.4959 |
| ours_reactive | 0.8153 | 0.7120 | -0.00307 | 0.2824 | 0.9021 | 0.5003 |
| sf_native | 0.6690 | 0.5138 | -0.00261 | 0.4676 | 0.7672 | 0.4579 |

## 2. Key findings

1. **ours_topk2 exceeds PF on DINO** (0.8269 vs 0.8263, +0.0006). This is
   the first ProbeCache cell to surpass PF.
2. **All ProbeCache cells maintain PF-level identity** (DINO 0.815-0.827 vs
   PF 0.826). Differences are within noise.
3. **ProbeCache dramatically improves over native SF** (DINO +0.15 over
   sf_native 0.669).
4. **Missing cells**: ours_full (proposed default), ours_audit, ours_persistent
   — only 1/12 MP4s each due to script termination. Rerun pending.

## 3. Missing cells (to be completed)

- `ours_full` — proposed default ProbeCache configuration
- `ours_audit` — audit mode (should reproduce PF)
- `ours_persistent` — persistent-only ablation

## 4. v82 follow-up (running)

v82 labels phase running on GPUs 1-7:
- 8 control label maps built (learned, inverse, pf_binary, 3×random, remote_only, prompt_only)
- 16 cells × 3 diagnostic prompts
- Tests whether counterfactual classification matters vs PF labels or random

## 5. Next steps

1. Rerun missing v81 cells (ours_full, ours_audit, ours_persistent)
2. Complete v82 labels phase
3. Run v82 confirm (multi-seed) and ultralong (60s)
4. Temporal jump diagnostic on v81
5. Human review of all complete cells
