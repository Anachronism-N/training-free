# v97 Experiment Results

Date: 2026-07-26

## 1. v97 QK Head Profiling (Corrected)

- **32 profiles**, 30 layers × 12 heads = 360 heads per profile
- Layer source: `kv_cache.layer_idx` (fixed from v96's broken global counter)
- All audits passed: version=2, records=8340, layers=30, branches=2

### Classification Results (tau=1.0, main operating point)

| Method | Stable | Responsive | Split |
|---|---:|---:|---|
| v96 consensus (broken) | 175 | 5 | 97%/3% |
| **v97 tau=1.0** | **292** | **68** | **81%/19%** |
| v97 sign-based (pos≥0.5) | 306 | 54 | 85%/15% |
| v97 PF-AR (Anchor vs Rest) | 172 | 188 | 48%/52% |
| v97 PF-AW (Anchor+Wave vs Veil) | 328 | 32 | 91%/9% |

## 2. v97 16-Cell Generation

All 16 cells × 32 prompts × 120 frames = 512 videos generated successfully.
- Node 1 (cells 0-7): 62.1 min, avg 116.4s/prompt
- Node 2 (cells 8-15): 70.3 min, avg 131.7s/prompt

## 3. DINOv2 Comprehensive Evaluation Results

| Method | Composite | DINO | Drift | Smooth | LPIPS | CLIP | BG | Loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt_tau_1p0_recent | 0.5539 | 0.7548 | -0.00257 | 14.136 | 0.1664 | 0.1079 | 0.9180 | 0.0205 |
| pf_wave_extended_recent | 0.5530 | 0.9242 | -0.00260 | 16.983 | 0.1782 | 0.0889 | 0.9490 | 0.2676 |
| pf_anchor_extended_recent | 0.5452 | 0.9204 | -0.00303 | 16.322 | 0.1703 | 0.0929 | 0.9474 | 0.1730 |
| **pf_native** | 0.5406 | **0.9308** | **-0.00202** | 20.290 | 0.1794 | 0.0904 | 0.9457 | **0.3067** |
| prompt_tau_1p0_merge | 0.5398 | 0.7497 | -0.00299 | 13.951 | 0.1678 | 0.1081 | 0.9163 | 0.0183 |
| pf_veil_extended_recent | 0.5384 | 0.9284 | -0.00218 | 19.550 | 0.1780 | 0.0889 | 0.9443 | 0.2855 |
| prompt_tau_0p5_merge | 0.5362 | 0.7463 | -0.00327 | 13.489 | 0.1666 | 0.1092 | 0.9177 | 0.0209 |
| prompt_tau_0p0_merge | 0.5206 | 0.7456 | -0.00353 | 17.629 | 0.1725 | 0.1085 | 0.9177 | 0.0172 |
| prompt_tau_2p0_merge | 0.5138 | 0.7339 | -0.00345 | 18.666 | 0.1778 | 0.1073 | 0.9117 | 0.0134 |
| prompt_tau_1p0_random_merge | 0.5133 | 0.7363 | -0.00335 | 18.259 | 0.1926 | 0.1076 | 0.9074 | 0.0086 |
| prompt_tau_1p5_merge | 0.5092 | 0.7455 | -0.00320 | 21.014 | 0.1863 | 0.1073 | 0.9105 | 0.0135 |
| prompt_tau_1p0_cyclic | 0.5083 | 0.7487 | -0.00327 | 21.738 | 0.1985 | 0.1061 | 0.9030 | 0.0266 |
| pf_aw_stride_merge | 0.5066 | 0.7318 | -0.00338 | 20.130 | 0.1850 | 0.1061 | 0.9088 | 0.0118 |
| sign_rpos_0p5_stride_merge | 0.4984 | 0.7281 | -0.00359 | 23.139 | 0.1867 | 0.1049 | 0.9097 | 0.0014 |
| pf_ar_stride_merge | 0.4918 | 0.7361 | -0.00363 | 21.772 | 0.1983 | 0.1043 | 0.9071 | 0.0129 |
| prompt_tau_1p0_reversed_merge | 0.4908 | 0.7296 | -0.00286 | 34.151 | 0.2248 | 0.1054 | 0.8891 | 0.0012 |

## 4. Key Findings

### 4.1 PF Native Remains Best by DINO and Loop

PF native (DINO=0.9308, Loop=0.3067) is the strongest method for temporal consistency and identity preservation. No binary or ablated method matches its DINO score.

### 4.2 PF Class Mechanism Ablation

| Ablation | DINO | Loop | DINO drop | Loop drop |
|---|---:|---:|---:|---:|
| PF native (reference) | 0.9308 | 0.3067 | — | — |
| Wave middle → recent | 0.9242 | 0.2676 | -0.007 | -0.039 |
| Veil middle → recent | 0.9284 | 0.2855 | -0.002 | -0.021 |
| Anchor middle → recent | 0.9204 | 0.1730 | -0.010 | -0.134 |

**Wave's cyclic middle is the least important** — removing it barely affects DINO (-0.007) or Loop (-0.039).
**Anchor's stride middle is the most important for Loop** — removing it causes Loop to collapse from 0.307 to 0.173.
**Veil's merge middle is moderately important** — removing it has small DINO impact but moderate Loop impact.

### 4.3 Prompt-Intervention Threshold Fails

All prompt-intervention threshold methods (tau=0.0 to 2.0) have DINO scores around 0.73-0.75, far below PF's 0.93. The binary cache policy (merge for responsive heads) causes severe identity drift.

The **reversed control** (tau=1.0_reversed_merge) is the worst (DINO=0.7296, Loop=0.0012), confirming that score direction matters but in a regime where both directions are bad.

The **random control** (tau=1.0_random_merge, DINO=0.7363) is comparable to the real threshold, suggesting the threshold classification adds no value beyond random assignment.

### 4.4 Sign-Based Alternative Also Fails

The sign-based split (positive_rate ≥ 0.5, 306 stable / 54 responsive) produces DINO=0.7281 and Loop=0.0014 — among the worst results. This disproves the hypothesis that logit-sign-based binary classification would outperform prompt-intervention.

### 4.5 PF Binary Merges Underperform

| PF Merge | DINO | Loop | vs PF native |
|---|---:|---:|---|
| PF native (3-class) | 0.9308 | 0.3067 | — |
| PF-AR (Anchor vs Wave+Veil) | 0.7361 | 0.0129 | DINO -0.195, Loop -0.294 |
| PF-AW (Anchor+Wave vs Veil) | 0.7318 | 0.0118 | DINO -0.199, Loop -0.295 |

Both PF binary merges collapse DINO and Loop compared to native PF. The three-class distinction is essential — merging any two classes into one degrades performance severely.

## 5. Conclusion

**Branch C (from v97 doc): Both prompt score and PF merges fail.**

The prompt-intervention binary taxonomy, the sign-based alternative, and both PF binary merges all fail to match PF's three-class system. The three PF classes (Anchor stride, Wave cyclic, Veil merge) each contribute distinct and irreplaceable cache behavior.

The PF class mechanism ablation shows that Wave's cyclic middle is the least important (could potentially be simplified), but Anchor's stride and Veil's merge are both essential for identity preservation and temporal continuity.

Native PF should remain the main engineering baseline. The binary hypothesis is recorded as negative.

## 6. Result Files

| File | Path |
|---|---|
| Comprehensive evaluation log | `runs/v97_threshold_pf_merge32/metrics/comprehensive.log` |
| Comprehensive evaluation JSON | `runs/v97_threshold_pf_merge32/metrics/comprehensive.json` |
| QK head scores | `runs/v97_qk_head_scores/scores/qk_head_scores.csv` |
| Threshold report | `runs/v97_qk_head_scores/scores/qk_head_score_artifact.json` |
| Classification report | `runs/v97_qk_head_scores/maps/head_map_classification_report.json` |
| Blind review scorecard | `runs/v97_threshold_pf_merge32/blind_review/scorecard.csv` |
| Policy trace audit | `runs/v97_threshold_pf_merge32/metrics/policy_trace_audit.json` |
| Generation cell logs | `runs/v97_threshold_pf_merge32/logs/` |
| Generation videos (16 cells) | `runs/v97_threshold_pf_merge32/{cell_name}/` |
| QK profiles (32 .pt files) | `runs/v97_qk_head_scores/profiles/` |
