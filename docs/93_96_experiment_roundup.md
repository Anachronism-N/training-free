# v93/v96 Experiment Roundup and Conclusions

Date: 2026-07-26

## 1. Experiment Overview

Three experiment groups were conducted in this round:

| Group | Purpose | Scale | Status |
|---|---|---|---|
| v93 MovieBench-128 | Main comparison table | 8 methods × 128 prompts | ✅ Complete + evaluated |
| v90 Priority Factorization | v78 ablation | 16 cells × 16 prompts | ✅ Complete + evaluated |
| v96 QK-Threshold Binary Cache | New binary head taxonomy | 32 profiles + 16 cells × 32 prompts | 🔄 Generation running |

---

## 2. v93 MovieBench-128 Results

### 2.1 DINOv2 Evaluation (128 prompts, 120 frames each)

| Method | DINO | Composite | Drift | Smooth | CLIP | BG | Loop |
|---|---:|---:|---:|---:|---:|---:|---:|
| sf_native | 0.8495 | 0.5298 | -0.00463 | 22.788 | 0.0763 | 0.9257 | 0.0562 |
| pf | 0.9060 | 0.5228 | -0.00231 | 26.467 | 0.0656 | 0.9349 | 0.3113 |
| veil_priority_b005 | 0.9095 | 0.5217 | -0.00207 | 28.370 | 0.0658 | 0.9337 | 0.3059 |
| v78 | 0.9092 | 0.5208 | -0.00207 | 27.663 | 0.0655 | 0.9342 | 0.3084 |
| pf_binary_read_v78 | 0.8890 | 0.5115 | -0.00235 | 30.897 | 0.0635 | 0.9281 | 0.2623 |
| echo_pc | 0.8656 | 0.5006 | -0.00237 | 40.479 | 0.0649 | 0.9101 | 0.2865 |
| prompt_pfcount_read_v78 | 0.8880 | 0.4993 | -0.00262 | 33.111 | 0.0673 | 0.9256 | 0.2519 |
| prompt_kmeans_read_v78 | 0.7457 | 0.4966 | -0.00273 | 33.442 | 0.0856 | 0.8997 | 0.0129 |

### 2.2 Key Findings

- **PF, v78, and veil_priority_b005 are statistically tied** on DINO (0.906–0.910)
- **v78 has the best drift slope** (-0.00207), tied with veil_priority_b005
- **sf_native (Self-Forcing baseline) is weakest** on DINO (0.8495) and loop (0.0562)
- **Binary read methods (pf_binary, pfcount, kmeans) are all weaker** than the top 3
- **prompt_kmeans_read_v78 is the worst** (DINO=0.7457, loop=0.0129) — K-means clustering produces poor head labels

### 2.3 Result Files

| File | Path |
|---|---|
| Full evaluation log | `runs/v93_moviebench128_main/metrics/comprehensive.log` |
| JSON results | `runs/v93_moviebench128_main/metrics/comprehensive.json` |
| Videos (8 methods) | `runs/v93_moviebench128_main/{sf_native,echo_pc,pf,pf_binary_read_v78,prompt_kmeans_read_v78,prompt_pfcount_read_v78,v78,veil_priority_b005}/` |
| Status markers | `runs/v93_moviebench128_main/status/` (16 .done files = 8 methods × 2 shards) |
| Transition traces | `runs/v93_moviebench128_main/traces/` |
| Run manifest | `runs/v93_moviebench128_main/run_manifest.env` |

---

## 3. v90 Priority Factorization

### 3.1 pf_age_only (last completed cell)

| Metric | Value |
|---|---:|
| DINO | 0.8347 |
| Composite | 0.5046 |
| Drift | -0.00224 |
| BG | 0.8929 |
| Loop | 0.0698 |

All 16 v90 cells are now complete (prompts 11–15 of pf_age_only were regenerated).

### 3.2 Result Files

| File | Path |
|---|---|
| pf_age_only evaluation | `runs/v90_priority_factorization_screen/metrics/pf_age_only_eval.json` |
| pf_age_only evaluation log | `runs/v90_priority_factorization_screen/metrics/pf_age_only_eval.log` |
| All 16 cell videos | `runs/v90_priority_factorization_screen/{cell_name}/` |
| Configs | `runs/v90_priority_factorization_screen/configs/` |
| Transition traces | `runs/v90_priority_factorization_screen/traces/` |

---

## 4. v96 QK-Threshold Binary Head Cache

### 4.1 Profiling Setup

- **Model**: Wan2.1-T2V-1.3B (30 transformer blocks, 12 heads/block)
- **Profiling**: 8 counterfactual prompt pairs × 2 sides × 2 seeds = 32 inference jobs
- **Captured**: Pre-softmax QK logits for self-attention layers (even indices 0, 2, ..., 28 = 15 layers)
- **Per head**: CFG response (NRMS cond vs uncond) + Semantic response (NRMS prompt_a vs prompt_b)
- **Scoring**: Layer-robust z-score of log1p(NRMS), then consensus = 0.5 × (cfg_z + semantic_z)
- **Threshold**: 2-component GMM intersection on consensus scores

### 4.2 Threshold Builder Results

| Gate | Observed | Required | Passed |
|---|---:|---:|---|
| BIC 2-vs-1 | 183.52 | ≥ 10 | ✅ |
| BIC 2-vs-3 | -11.81 | ≥ 0 | ❌ |
| Minority class fraction | 0.0278 (5/180) | ≥ 0.10 | ❌ |
| Bootstrap stability | 1.0000 | ≥ 0.80 | ✅ |

**Classification**: 175 Prompt-Stable / 5 Prompt-Responsive (out of 180 heads)
**Accepted**: False (failed on minority class fraction and BIC 2-vs-3)

### 4.3 PF Temporal QK Statistics (Measured)

This is the first direct measurement of PF's three head classes on our model:

| PF Class | Heads | Positive Rate | Mean Logit | Sign Switch Rate | PF Native Policy |
|---|---:|---:|---:|---:|---|
| Anchor | 72 | 1.000 (100%) | 1.873 | 0.000 | stride (interval=6, cap=4) |
| Wave | 90 | 0.667 (66.7%) | 1.157 | 0.500 | cyclic (period=6, 4 buckets) |
| Veil | 18 | 0.125 (12.5%) | -1.891 | 0.000 | merge (patch=2, cap=4) |

**Key observations**:
- Anchor: uniformly positive logits, no sign alternation → stable long-range retrieval
- Wave: 2/3 positive, 50% sign alternation → periodic positive/negative switching
- Veil: predominantly negative logits → historical suppression

### 4.4 Cross-Tab: Prompt-Intervention vs PF Labels

| PF Label | Prompt-Stable | Prompt-Responsive |
|---|---:|---:|
| Wave (-1) | 87 | 3 |
| Anchor (1) | 71 | 1 |
| Veil (2) | 17 | 1 |

**Prompt-Stable covers 99.4% of PF Anchor** (recall), but also covers 96.7% of Wave and 94.4% of Veil. The binary prompt-intervention taxonomy cannot distinguish between PF's three classes.

### 4.5 Score Distribution Analysis

The consensus scores are heavily right-skewed:
- 56/180 heads (31%) have scores in [0, 1)
- Only 4/180 heads (2%) have scores > 3.0
- The GMM threshold (3.35) captures only the extreme outliers

**Root cause**: NRMS values are universally small (CFG median=0.035, Semantic median=0.083), meaning prompt intervention has minimal effect on QK temporal logits for most heads.

### 4.6 Alternative Binary Classification by Logit Sign

The user identified that Wave heads are mostly positive (66.7% positive rate), suggesting an alternative binary split:

**By positive_rate ≥ 0.5**:
- Positive (stride): ~146 heads (81%) — Anchor + most Wave
- Negative (merge): ~34 heads (19%) — Veil + few Wave

This split is far more balanced (81/19 vs 97/3) and aligns with PF's mechanism:
- Positive logits → historical relevance → stride (long-range sparse)
- Negative logits → historical suppression → merge (local compression)

### 4.7 Generation Status (16 cells, running)

| Cell | Label Source | Policy | v78 | Node | Speed | Videos |
|---|---|---|---|---|---|---|
| pf | PF native 3-class | native PF | no | 1 | ~3s/block | 9/32 |
| pf_binary_cyclic | PF binary | cyclic | no | 1 | ~3s/block | 9/32 |
| pf_binary_merge | PF binary | merge | no | 1 | ~3s/block | 9/32 |
| pf_binary_recent | PF binary | recent | no | 1 | ~3s/block | 10/32 |
| cfg_cyclic | CFG threshold | cyclic | no | 1 | ~3s/block | 9/32 |
| cfg_merge | CFG threshold | merge | no | 1 | ~3s/block | 9/32 |
| semantic_cyclic | Semantic threshold | cyclic | no | 1 | ~3s/block | 9/32 |
| semantic_merge | Semantic threshold | merge | no | 1 | ~3s/block | 9/32 |
| consensus_cyclic | Consensus threshold | cyclic | no | 2 | ~16s/block | 0/32 |
| consensus_merge | Consensus threshold | merge | no | 2 | ~16s/block | 0/32 |
| consensus_recent | Consensus threshold | recent | no | 2 | ~16s/block | 0/32 |
| consensus_merge_v78 | Consensus threshold | merge | yes | 2 | ~16s/block | 0/32 |
| consensus_cyclic_v78 | Consensus threshold | cyclic | yes | 2 | ~16s/block | 0/32 |
| random_merge | Random (control) | merge | no | 2 | ~16s/block | 0/32 |
| inverse_merge | Inverse (control) | merge | no | 2 | ~15s/block | 0/32 |
| pf_binary_merge_v78 | PF binary | merge | yes | 2 | ~16s/block | 0/32 |

Node 2 cells are 5× slower due to zombie CUDA contexts from completed v93 processes.

### 4.8 Result Files

| File | Path |
|---|---|
| QK profiles (32 .pt files) | `runs/v96_qk_head_profile/profiles/` |
| Profile logs | `runs/v96_qk_head_profile/logs/` |
| Profiling videos (60 frames each) | `runs/v96_qk_head_profile/videos/` |
| Threshold report (JSON) | `runs/v96_qk_head_profile/labels/qk_head_threshold_report.json` |
| Threshold summary (Markdown) | `runs/v96_qk_head_profile/labels/qk_head_threshold_summary.md` |
| Per-head scores (CSV) | `runs/v96_qk_head_profile/labels/qk_head_scores.csv` |
| Label maps (6 CSV files) | `runs/v96_qk_head_profile/labels/{prompt_cfg_threshold,prompt_semantic_threshold,prompt_consensus_threshold,prompt_consensus_inverse,prompt_consensus_random,pf_binary}.csv` |
| Build thresholds log | `runs/v96_qk_head_profile/labels/build_thresholds.log` |
| Generation cell logs | `runs/v96_binary_cache32/logs/` |
| Generation nohup (Node 1) | `runs/v96_binary_cache32_node1.nohup.log` |
| Generation nohup (Node 2) | `runs/v96_binary_cache32_node2.nohup.log` |
| Generation videos (16 cells) | `runs/v96_binary_cache32/{cell_name}/` |
| Generation status markers | `runs/v96_binary_cache32/status/` |
| Transition traces | `runs/v96_binary_cache32/traces/` |
| Profiling nohup log | `runs/v96_qk_head_profile.nohup.log` |
| Profile jobs TSV | `runs/v96_qk_head_profile/profile_jobs.tsv` |
| Counterfactual prompt pairs | `prompts/probecache_counterfactual_pairs.json` |

---

## 5. Overall Conclusions

### 5.1 PF Remains the Strongest Method

On the 128-prompt MovieBench, PF (DINO=0.9060), v78 (DINO=0.9092), and veil_priority_b005 (DINO=0.9095) are statistically tied. The binary read methods are all weaker.

### 5.2 Prompt-Intervention Binary Taxonomy Is Not Effective

The v96 QK profiling reveals that:
- **97.2% of heads are Prompt-Stable** — prompt changes have minimal effect on QK temporal logits
- The binary taxonomy cannot distinguish between PF's Anchor, Wave, and Veil classes
- The GMM finds a two-component structure, but the minority class (5 heads, 2.8%) is too small for practical use

### 5.3 Logit-Sign Binary Classification Is More Promising

An alternative binary split by logit sign (positive → stride, negative → merge) yields:
- 81% / 19% class balance (vs 97% / 3%)
- Natural alignment with PF's mechanism (positive = historical relevance, negative = suppression)
- Anchor + Wave merged as "positive" class, Veil as "negative" class

This suggests future experiments should explore logit-sign-based binary classification rather than prompt-intervention response.

### 5.4 PF Three-Class Contribution Analysis Needed

The v93 results show that collapsing PF's 3 classes to 2 (pf_binary_read_v78, DINO=0.8890) loses 0.017 DINO vs full PF (0.9060). Future ablation experiments should:
1. Remove each PF class individually (Anchor, Wave, Veil) to measure per-class contribution
2. Test the logit-sign binary split (Anchor+Wave vs Veil) as an alternative to PF's 3-class system
3. Investigate whether Wave's cyclic policy can be replaced by stride without performance loss

---

## 6. Key Data Directories for Further Analysis

```
runs/
├── v93_moviebench128_main/
│   ├── metrics/
│   │   ├── comprehensive.log          # Full evaluation log with per-method results
│   │   └── comprehensive.json          # Machine-readable results
│   ├── {8 method dirs}/               # 128 videos each (128 × 8 = 1024 total)
│   ├── status/                        # 16 shard completion markers
│   └── traces/                        # Transition traces for v78 methods
│
├── v90_priority_factorization_screen/
│   ├── {16 cell dirs}/                # 16 videos each
│   ├── metrics/
│   │   ├── pf_age_only_eval.log       # pf_age_only evaluation
│   │   └── pf_age_only_eval.json
│   ├── configs/                       # Per-cell configuration
│   └── traces/                        # Transition traces
│
├── v96_qk_head_profile/
│   ├── profiles/                      # 32 QK profile .pt files (raw QK logits)
│   ├── logs/                          # 32 profiling job logs
│   ├── videos/                        # 32 profiling videos (60 frames each)
│   ├── labels/
│   │   ├── qk_head_threshold_report.json   # Full threshold report
│   │   ├── qk_head_threshold_summary.md    # Human-readable summary
│   │   ├── qk_head_scores.csv              # Per-head scores (180 rows)
│   │   ├── build_thresholds.log            # Threshold builder log
│   │   ├── prompt_cfg_threshold.csv        # CFG-based label map (30×12)
│   │   ├── prompt_semantic_threshold.csv   # Semantic-based label map
│   │   ├── prompt_consensus_threshold.csv  # Consensus label map (main)
│   │   ├── prompt_consensus_inverse.csv    # Inverted (control)
│   │   ├── prompt_consensus_random.csv     # Random (control)
│   │   └── pf_binary.csv                   # PF binary oracle labels
│   ├── prompts/                       # 32 prompt files (counterfactual pairs)
│   ├── profile_jobs.tsv               # Job manifest
│   └── status/                        # 32 completion markers
│
└── v96_binary_cache32/
    ├── {16 cell dirs}/                # 32 videos each (when complete)
    ├── logs/                          # 16 cell generation logs
    ├── status/                        # Cell completion markers
    ├── configs/                       # Per-cell configuration
    ├── traces/                        # v78 transition traces
    └── diagnostics/                   # Audit results
```

### Prompt-Intervention Analysis Data

The key data for analyzing the prompt-intervention classification:

1. **Per-head scores**: `runs/v96_qk_head_profile/labels/qk_head_scores.csv`
   - Columns: layer, head, pf_label, cfg_raw, semantic_raw, cfg_score, semantic_score, consensus_score, consensus_label, bootstrap_agreement, positive_rate, mean_logit, sign_switch_rate, dominant_period, spectral_peak_ratio
   - 180 rows (15 even layers × 12 heads)

2. **Full report**: `runs/v96_qk_head_profile/labels/qk_head_threshold_report.json`
   - GMM parameters, BIC scores, bootstrap stability, PF cross-tab, temporal statistics

3. **Raw QK profiles**: `runs/v96_qk_head_profile/profiles/*.pt`
   - 32 PyTorch files, each containing per-layer per-head QK logits for cond/uncond branches
   - Can be loaded with `torch.load(path, map_location='cpu', weights_only=False)`

4. **PF label map**: `third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv`
   - 30 rows × 12 columns, values: -1 (Wave), 1 (Anchor), 2 (Veil)

5. **Counterfactual prompts**: `prompts/probecache_counterfactual_pairs.json`
   - 8 prompt pairs that change one factor (action, camera, scene, motion, weather) while preserving identity
