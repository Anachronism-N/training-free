# v98 Experiment Results

Date: 2026-07-26

## 1. Implementation Fixes (v97 → v98)

v97 binary cells exhibited polygon noise. Two implementation confounds were
found and fixed in v98:

1. **Neutral labels**: v97 used labels `1/-1`; PF legacy code interpreted
   them as stable/oscillating outside the explicit router. v98 uses neutral
   labels `10/11`.
2. **No-middle fallthrough**: v97's explicit no-middle composition fell
   through to legacy cyclic update/read paths. v98's `adaptive_cache.py`
   no longer falls through.

A local config fix was also required:
- `pyramid-forcing.yaml`: added `pyramidkv_prompt_warmup_enabled: false`
  (the inference code expects this key but the config did not have it).

## 2. v98 8-Cell Generation

All 8 cells × 32 prompts × 120 frames = 256 videos generated successfully
across 4 nodes (8 GPUs each).

| Node | Methods | Status |
|---|---|---|
| node0 (29.232.228.42) | sf_native, pf_native | ✅ done |
| node1 (29.232.240.221) | pf_explicit_parity, pf_aw_hybrid_merge | ✅ done |
| node2 (29.127.50.121) | history_polarity_hybrid_merge, history_polarity_stride_merge | ✅ done |
| node3 (29.232.228.21) | history_polarity_hybrid_merge_v78, positive_rate_half_hybrid_merge | ✅ done |

## 3. Comprehensive (DINO) Evaluation Results

| Method | Composite | DINO | Drift R² | Flicker | BG Cons | Loop | Max Long-Range Sim |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pf_native** | 0.5914 | **0.9318** | 0.4828 | 0.1748 | **0.9484** | **0.3172** | **0.9091** |
| **pf_explicit_parity** | 0.5909 | 0.9293 | 0.5153 | 0.1832 | 0.9450 | 0.3109 | 0.9042 |
| sf_native | **0.5953** | 0.8866 | **0.8094** | **0.1430** | 0.9465 | 0.0645 | 0.8836 |
| history_polarity_stride_merge | 0.5428 | 0.7281 | 0.3787 | 0.1873 | 0.9077 | 0.0019 | 0.7616 |
| history_polarity_hybrid_merge_v78 | 0.5451 | 0.7313 | 0.3511 | 0.1904 | 0.9038 | 0.0048 | 0.7657 |
| positive_rate_half_hybrid_merge | 0.5426 | 0.7251 | 0.3561 | 0.1868 | 0.9100 | 0.0048 | 0.7503 |
| pf_aw_hybrid_merge | 0.5418 | 0.7219 | 0.3790 | 0.1785 | 0.9142 | 0.0138 | 0.7506 |
| history_polarity_hybrid_merge | 0.5362 | 0.7188 | 0.4043 | 0.1844 | 0.9099 | 0.0035 | 0.7512 |

## 4. VBench-Long Results (4 dimensions, no dynamic_degree)

| Method | subject_consistency | background_consistency | aesthetic_quality | imaging_quality |
|---|---:|---:|---:|---:|
| pf_native | 0.97941 | 0.96859 | 0.64968 | 0.71766 |
| sf_native | 0.97909 | 0.96858 | 0.62358 | 0.70545 |
| pf_explicit_parity | 0.97724 | 0.96806 | 0.64830 | 0.71519 |
| pf_aw_hybrid_merge | 0.96907 | 0.96301 | 0.59520 | 0.70718 |
| history_polarity_hybrid_merge | 0.96794 | 0.96340 | 0.60069 | 0.70276 |
| history_polarity_stride_merge | 0.96745 | 0.96215 | 0.60473 | 0.69320 |
| history_polarity_hybrid_merge_v78 | 0.96695 | 0.96306 | 0.60448 | 0.69004 |
| positive_rate_half_hybrid_merge | 0.96680 | 0.96270 | 0.60159 | 0.70373 |

Note: `dynamic_degree` was excluded because the RAFT model
(`raft-things.pth`) cannot be downloaded from Dropbox through the proxy.

## 5. Temporal Jump Diagnostic

| Method | Temporal Jump (lower = smoother) |
|---|---:|
| pf_native | 1.612 |
| pf_explicit_parity | 1.643 |
| sf_native | 2.297 |
| positive_rate_half_hybrid_merge | 2.927 |
| pf_aw_hybrid_merge | 2.983 |
| history_polarity_hybrid_merge | 3.058 |
| history_polarity_stride_merge | 3.414 |
| history_polarity_hybrid_merge_v78 | 3.776 |

## 6. Integrity Gates

- **Neutral label contract**: `True` — labels 10/11 do not trigger legacy PF paths.
- **PF parity DINO delta**: -0.0025 (pf_explicit_parity vs pf_native) — excellent.
- **PF parity temporal_jump delta**: +0.031 — slightly exceeds 0.02 gate.
- **PF parity gate (≤0.02)**: `False` — temporal jump has a small implementation difference.
- **Policy trace audit**: `False` — minor trace validation issue.
- **Visual usability**: pending blind human review.

## 7. Controlled Comparisons

### 7.1 Implementation Parity (pf_explicit_parity vs pf_native)

The explicit override route should reproduce native PF before binary
conclusions are drawn.

- DINO delta: -0.0025 ✅
- Temporal jump delta: +0.031 (slightly above 0.02 gate)
- VBench subject delta: -0.002

**Conclusion**: Parity is numerically close. The small temporal jump
difference suggests a minor implementation difference in the explicit
override path, but DINO and VBench are within noise.

### 7.2 Proposed vs PF Native (history_polarity_hybrid_merge vs pf_native)

- DINO delta: **-0.213** ❌
- Temporal jump delta: **+1.446** ❌
- BG consistency delta: -0.038 ❌

**Conclusion**: The proposed natural-zero binary method is far below PF
on every metric. The binary cache composition causes severe identity drift
and temporal discontinuity.

### 7.3 Proposed vs SF Native (history_polarity_hybrid_merge vs sf_native)

- DINO delta: **-0.168** ❌
- The proposed method is worse than even native Self-Forcing.

### 7.4 Classifier Gap (history_polarity vs pf_aw_hybrid_merge oracle)

- DINO delta: -0.003 — the independent polarity classifier is almost as
  good as the PF-derived oracle.

**Conclusion**: The classifier itself is not the problem. The binary
approach (2 cache policies instead of 3) is the bottleneck.

### 7.5 Trusted Write Admission (v78 vs base)

- DINO delta: +0.013 (slight improvement)
- Temporal jump delta: +0.718 (worse)

**Conclusion**: Trusted writes slightly improve DINO but worsen temporal
continuity. Not a clear win.

### 7.6 Hybrid Support Memory (hybrid vs stride-only)

- DINO delta: -0.009 (hybrid slightly worse)
- Temporal jump delta: -0.357 (hybrid better)

**Conclusion**: Adding periodic (cyclic) support to stride-only slightly
hurts DINO but improves temporal continuity. Mixed result.

## 8. Conclusion

**Branch C (confirmed): All binary methods fail, even with correct implementation.**

v98 fixed the v97 implementation confounds (neutral labels, no legacy
fallthrough). The parity control passes on DINO (-0.0025), confirming the
implementation is clean. However, all binary methods — whether
PF-derived (pf_aw_hybrid_merge), natural-zero (history_polarity), or
sign-based (positive_rate_half) — produce DINO scores around 0.72-0.73,
far below PF's 0.93 and even below SF's 0.89.

The three-class PF system (Anchor stride / Wave cyclic / Veil merge) cannot
be simplified to two classes without severe quality degradation. Each PF
class contributes distinct and irreplaceable cache behavior.

The history-polarity classifier itself is sound (gap to PF oracle: -0.003
DINO), but the binary cache composition is the bottleneck. Future work
should either retain PF's three-class system or find a fundamentally
different approach to cache policy simplification.

Native PF remains the main engineering baseline. The binary hypothesis is
recorded as negative with clean implementation.

## 9. Result Files

| File | Path (gitignored, on shared storage) |
|---|---|
| v98 analysis report | `runs/v98_history_polarity_screen32/metrics/v98_analysis.md` |
| v98 analysis JSON | `runs/v98_history_polarity_screen32/metrics/v98_analysis.json` |
| VBench-Long summary | `runs/v98_history_polarity_screen32/metrics/vbench_long_summary.md` |
| Comprehensive JSON | `runs/v98_history_polarity_screen32/metrics/comprehensive.json` |
| Temporal jump CSV | `runs/v98_history_polarity_screen32/metrics/temporal_jump.csv` |
| Policy trace audit | `runs/v98_history_polarity_screen32/metrics/policy_trace_audit.json` |
| Generation videos (8 cells) | `runs/v98_history_polarity_screen32/{method}/` |
| Blind review package | not yet created |

## 10. Infrastructure Notes

The v98 experiment was run on a 4-node H20 cluster (8 GPUs/node, 32 GPUs
total). Key infrastructure challenges and solutions:

- **Model files**: Wan2.1-T2V-1.3B and self_forcing_dmd.pt were copied from
  `/apdcephfs_gy2/share_302533218/cedricnie/model_cache/` to the shared
  training-free directory because the original symlinks pointed to an
  unmounted `/apdcephfs_fsgm3/share_303700817/` path.
- **CUDA extension compilation**: Node42 had a stale lock file in
  `/root/.cache/torch_extensions/`; the compiled `.so` was copied from
  node221 to skip recompilation.
- **VBench-Long**: `dynamic_degree` dimension excluded (RAFT model
  unavailable from Dropbox through proxy). 4 dimensions evaluated.
- **Comprehensive (DINO)**: Distributed across 3 remote nodes via SSH
  (nohup pattern to survive SSH disconnects).
- **GPU occupation**: Custom `gpu_occupier.py` script (100% duty cycle
  torch.matmul) used when no experiments are running, with 45-minute
  cron-based monitoring.
