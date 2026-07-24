# v90 Human Review (Partial) and Experiment Status

> Date: 2026-07-24
> Reviewer: human
> Status: v90 Node 2 complete (8/16 cells), Node 1 loading (8 OOM cells), v92 not started

## 1. v90 Human Review (prompts 0-0, 1-0)

### veil_priority_b005
- 0-0: Good
- 1-0: Good, but early frames have flashback and artifacts (persistent issue across all methods)

### wave_priority_b005
- Good overall
- 1-0: Mid-late portion shows subject disappearance and reappearance

### pf and v78 (across seeds)
- Similar quality across seeds
- Both have fast acceleration jumps (PF-inherited)
- Jumps cause background hallucinations
- Subject ID retention is good
- Early frame flashback and artifacts persist (universal PF issue)

### Key observations
1. **Background hallucinations from jumps** — the acceleration jumps cause
   background changes/hallucinations. This is a PF-inherited issue, not
   introduced by the transition controller.
2. **Early frame flashback/artifacts** — universal across all methods,
   confirmed as a PF initialization artifact.
3. **Subject ID is consistently good** — v78 and pf maintain identity well.
4. **veil_priority and wave_priority are promising** — specific PF head
   subsets may be more useful than the full binary map.

## 2. v90 DINOv2 Results

Pending — evaluation running on 8 complete cells (pf_s1/s2/s3, v78_s1/s2,
pf_priority_b010, veil_priority_b005, wave_priority_b005).

## 3. v90 OOM Cells (Node 1)

8 cells stuck at CUDA extension compilation on Node 1 (28.7.187.25):
- v78_s3, pf_priority_b005, pf_priority_late, inverse_priority_b005,
  learned_priority_b005, random_priority_b005, pf_age_only, pf_novelty_only

All at 0/16, GPU memory 325 MiB (CUDA context only, models not loaded).

## 4. v92 Experiment (docs/92)

**NOT STARTED.** v92 tests prompt-contrastive binary read topology:
- Actual two-class read topology (not just write control)
- 16 cells × 16 prompts × 120 frames
- Uses prompt-contrastive head maps from `build_prompt_contrastive_head_maps.py`

v92 requires `--profile-report` argument which was missing in earlier attempt.
Need to build prompt-contrastive maps first, then run the screen.

## 5. Current GPU allocation

- Node 2 (8 GPUs): DINOv2 (GPU 0) + VBench-Long (8 evals, GPU 0-7)
- Node 1 (8 GPUs): v90 OOM cells (stuck at CUDA compilation)

## 6. Next steps

1. Fix Node 1 CUDA compilation issue
2. Start v92 on available GPUs
3. Complete DINOv2 and VBench-Long on v90 cells
4. When v90 OOM cells complete, run DINOv2 on them
5. Push all results to GitHub

## 7. v90 DINOv2 Matched-Seed Results (CRITICAL)

### Per-cell results (sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | composite |
|---|---:|---:|---:|---:|---:|
| pf_s1 | 0.8001 | 0.7084 | -0.00247 | 0.2778 | 0.4913 |
| veil_priority_b005 | 0.7993 | 0.7182 | -0.00222 | 0.2828 | 0.4933 |
| wave_priority_b005 | 0.7987 | 0.6853 | -0.00220 | 0.2842 | 0.4927 |
| pf_priority_b010 | 0.7967 | 0.6863 | -0.00218 | 0.2830 | 0.4926 |
| v78_s1 | 0.7871 | 0.6813 | -0.00268 | 0.2906 | 0.4868 |
| v78_s2 | 0.7861 | 0.7095 | -0.00287 | 0.2861 | 0.4857 |
| pf_s2 | 0.7789 | 0.7103 | -0.00337 | 0.2861 | 0.4779 |
| pf_s3 | 0.7754 | 0.6817 | -0.00258 | 0.2875 | 0.4862 |

### Matched-seed comparison

| Seed | PF DINO | v78 DINO | Δ (v78 - PF) | Winner |
|---:|---:|---:|---:|---|
| 0 (v86) | 0.8496 | 0.8536 | +0.0040 | v78 |
| 1 (v90) | 0.8001 | 0.7871 | **-0.0130** | **PF** |
| 2 (v90) | 0.7789 | 0.7861 | +0.0073 | v78 |
| **Average** | **0.8095** | **0.8089** | **-0.0006** | **Tie** |

### Critical finding

**v78 does NOT consistently beat PF across matched seeds.**
- Seed 0: v78 +0.004 (v86, 16 prompts)
- Seed 1: PF +0.013 (v90, 16 prompts) — PF wins!
- Seed 2: v78 +0.007 (v90, 16 prompts)
- Average: PF 0.8095 vs v78 0.8089 — effectively tied

The v86 seed-0 advantage was within noise and was not reproduced on seed 1.
The earlier claim of "+0.021 DINO across seeds" was based on unmatched seeds
(v78 seeds 2/3 vs PF seed 0) and is not valid as a statistical claim.

### Implications

1. **v78 ≈ PF on DINO**, not v78 > PF. The trust-conditioned cache
   transition matches PF but does not clearly exceed it.
2. **veil_priority_b005 (0.7993) and wave_priority_b005 (0.7987) are
   competitive with PF (0.8001)** — PF head subsets may be as useful
   as the full binary map.
3. **pf_priority_b010 (0.7967) ≈ v78 (0.7871)** — weak priority with
   PF binary labels is competitive.
4. **The paper claim must be revised**: v78 matches PF with zero compute
   overhead and improves temporal jump, but does NOT improve DINO.

## 8. v92 Started

Prompt-contrastive binary cache experiment (docs/92) is now running.
10 head maps built. 16 cells × 16 prompts × 120 frames.

## 9. Node 1 v90 OOM Cells

CUDA extension compilation completed (2468s). Models now loading
(14.8 GB GPU memory). Generation should begin shortly.

## 10. v90 Complete DINOv2 (12/16 cells)

### All results (sorted by DINO)

| Cell | DINO | min_DINO | drift | Source |
|---|---:|---:|---:|---|
| random_priority_b005 | 0.8485 | 0.7917 | -0.00149 | Node 1 |
| veil_priority_b005 | 0.7993 | 0.7182 | -0.00222 | Node 2 |
| wave_priority_b005 | 0.7987 | 0.6853 | -0.00220 | Node 2 |
| pf_priority_b010 | 0.7967 | 0.6863 | -0.00218 | Node 2 |
| pf_s1 | 0.8001 | 0.7084 | -0.00247 | Node 2 |
| inverse_priority_b005 | 0.8444 | 0.7611 | -0.00235 | Node 1 |
| learned_priority_b005 | 0.8421 | 0.7486 | -0.00227 | Node 1 |
| v78_s3 | 0.8421 | 0.7517 | -0.00219 | Node 1 |
| v78_s1 | 0.7871 | 0.6813 | -0.00268 | Node 2 |
| v78_s2 | 0.7861 | 0.7095 | -0.00287 | Node 2 |
| pf_s2 | 0.7789 | 0.7103 | -0.00337 | Node 2 |
| pf_s3 | 0.7754 | 0.6817 | -0.00258 | Node 2 |

### Matched-seed comparison (updated with seed 3)

| Seed | PF | v78 | Δ | Winner |
|---:|---:|---:|---:|---|
| 0 (v86) | 0.8496 | 0.8536 | +0.004 | v78 |
| 1 (v90) | 0.8001 | 0.7871 | -0.013 | PF |
| 2 (v90) | 0.7789 | 0.7861 | +0.007 | v78 |
| 3 (v90) | — | 0.8421 | — | v78 only |
| **Avg s0-2** | **0.8095** | **0.8089** | **-0.001** | **Tie** |

### Weak-priority comparison

| Cell | DINO | vs v78_s0 (0.8536) |
|---|---:|---:|
| random_priority_b005 | 0.8485 | -0.005 |
| inverse_priority_b005 | 0.8444 | -0.009 |
| learned_priority_b005 | 0.8421 | -0.012 |
| v78_s3 (uniform) | 0.8421 | -0.012 |

**Weak priority with any label map does NOT improve over uniform v78.**
All priority variants are within noise of v78_s3 (0.8421).

### 4 cells still missing

pf_priority_b005, pf_age_only, pf_novelty_only, pf_priority_late —
failed on Node 1 due to label CSV path issue. Not critical for conclusions.

## 11. v93 MovieBench Status

- **Node 1**: MovieBench-128 main running (8 methods × 128 prompts)
- **Node 2**: MovieBench-32 head32 running (3/16 cells active, 13 OOM)
- Both experiments generating videos
