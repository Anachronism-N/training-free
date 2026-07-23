# v81 + v82 ProbeCache: Comprehensive Results

> Date: 2026-07-24
> Experiments: v81 screen (9 cells × 12 prompts), v82 labels (9 cells × 3 prompts)
> v82 confirm running (multi-seed)

## 1. v81 Screen: DINOv2 + Temporal Jump

### 1.1 DINOv2 (sorted by DINO)

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

### 1.2 Temporal Jump (sorted by jump)

| Cell | jump_mean | vs PF |
|---|---:|---:|
| sf_native | 1.1114 | -40% (misleading — degraded to static) |
| ours_conservative | 1.3867 | **-25%** |
| ours_topk6 | 1.4226 | **-23%** |
| ours_reactive | 1.4292 | **-23%** |
| ours_archive12 | 1.5267 | -17% |
| ours_topk2 | 1.5401 | -17% |
| ours_prompt0 | 1.5527 | -16% |
| ours_open_gate | 1.7009 | -8% |
| pf_official | 1.8474 | baseline |

### 1.3 Key v81 findings

1. **ours_topk2 beats PF on DINO** (0.8269 vs 0.8263) — first ProbeCache
   cell to exceed PF.
2. **All ProbeCache cells have LOWER temporal jump than PF** (-8% to -25%).
   ProbeCache improves temporal smoothness while maintaining identity.
3. **Best trade-off: ours_conservative** — DINO=0.8165 (≈PF), jump=1.39
   (-25% vs PF).
4. **sf_native DINO=0.669** — dramatically worse than PF, confirming PF base
   is essential. Low temporal jump is misleading (video degrades to static).
5. **Missing cells**: ours_full, ours_audit, ours_persistent (KeyError bug —
   pending fix).

## 2. v82 Labels: Classification-Causality Matrix

### 2.1 DINOv2 (3 diagnostic prompts, sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | comp |
|---|---:|---:|---:|---:|---:|
| v78 | 0.8827 | 0.8401 | -0.00287 | 0.2653 | 0.5214 |
| random_2028_fallback | 0.8515 | 0.6815 | -0.00424 | 0.2825 | 0.5145 |
| pf_binary | 0.8506 | 0.7460 | -0.00395 | 0.2894 | 0.5087 |
| layer_first_half | 0.8464 | 0.8052 | -0.00279 | 0.2775 | 0.5220 |
| layer_early | 0.8323 | 0.8439 | -0.00437 | 0.2915 | 0.5039 |
| layer_late | 0.8192 | 0.7081 | -0.00388 | 0.3228 | 0.5015 |
| learned | 0.8170 | 0.6990 | -0.00452 | 0.3126 | 0.5161 |
| learned_audit | 0.8129 | 0.7419 | -0.00491 | 0.2969 | 0.5012 |
| inverse | 0.7480 | 0.6434 | -0.00183 | 0.2682 | 0.5208 |

### 2.2 Classification-causality analysis

**learned vs inverse**: learned (0.817) >> inverse (0.748), +0.069 DINO.
Swapping persistent↔reactive labels clearly hurts. **Classification direction
matters.**

**learned vs pf_binary**: pf_binary (0.851) > learned (0.817), -0.034 DINO.
PF's static labels (Anchor→persistent, Wave+Veil→reactive) outperform the
counterfactual classification. **This weakens the classification contribution
claim.**

**learned vs random_2028_fallback**: random (0.852) > learned (0.817),
-0.035 DINO. However, random_2028_fallback may have fallen back to primary
(learned) labels when the replica profile failed. If it is truly random,
this further weakens the classification claim.

**v78 (0.883)** remains the best overall, outperforming all ProbeCache
variants by +0.03 to +0.13 DINO.

### 2.3 Depth ablation

| Depth band | DINO | min_DINO |
|---|---:|---:|
| layer_first_half (0-14) | 0.8464 | 0.8052 |
| layer_early (0-9) | 0.8323 | 0.8439 |
| layer_late (20-29) | 0.8192 | 0.7081 |

Early-to-middle layers are more important than late layers for identity
retention. layer_first_half has the best composite (0.5220).

### 2.4 v82 labels conclusions

1. **Classification direction matters** (learned >> inverse, +0.069 DINO).
2. **But PF binary labels work as well or better** (pf_binary > learned,
   +0.034). The counterfactual classification does not clearly beat PF's
   static labels.
3. **v78 remains the best method** (0.883, +0.066 over best ProbeCache).
4. **Only 3 diagnostic prompts** — results are noisier than v81's 12 prompts.
   Multi-seed confirmation (v82 confirm, running) will provide more robust
   evidence.
5. **Missing cells** (pf, remote_only, prompt_only, random_2026/2027,
   layer_middle, layer_second_half) would complete the matrix but are
   blocked by KeyError bug.

## 3. Overall assessment

| Method | v81 DINO | v81 jump | v82 DINO | Assessment |
|---|---:|---:|---:|---|
| v78 (cache transition) | — | — | **0.883** | **Best overall** |
| ours_topk2 (ProbeCache) | **0.827** | 1.54 | — | Best v81, beats PF |
| pf_official | 0.826 | 1.85 | — | PF baseline |
| pf_binary (ProbeCache) | — | — | 0.851 | PF labels work well |
| learned (ProbeCache) | — | — | 0.817 | Learned labels, moderate |
| inverse (ProbeCache) | — | — | 0.748 | Swapped labels, clearly worse |
| sf_native | 0.669 | 1.11 | — | Baseline (degraded) |

**v78 remains the recommended paper candidate.** ProbeCache matches PF on
identity and improves temporal jump, but the counterfactual classification
does not clearly beat PF's static labels. v82 confirm (running) will test
multi-seed robustness.

## 4. Running experiments

- **v82 confirm** (GPUs 1-7): multi-seed confirmation of PF, v78, learned,
  pf_binary across seeds 1-3. Running (29 processes).
- **v82 ultralong** (pending): 60-second videos, 6 prompts, 240 frames.
- **v82 switch** (pending): A-B-A prompt switching, 3 prompts.
- **v82 prepare** (pending): blind review packages.

## 5. Next steps

1. Wait for v82 confirm to complete → multi-seed analysis
2. Run v82 ultralong (60s extrapolation)
3. Human review of v81 and v82 results
4. Fix KeyError bug for missing ProbeCache cells
5. If v78 remains best across seeds, confirm v78 as paper candidate
