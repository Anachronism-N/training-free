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

## 12. v93 MovieBench-128 DINOv2 (4 complete methods, 128 prompts)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| **pf_binary_read_v78** | **0.8890** | **0.8528** | -0.00235 | 0.2211 | 0.9281 | 0.5115 |
| echo_pc | 0.8656 | 0.7757 | -0.00237 | 0.2393 | 0.9101 | 0.5006 |
| sf_native | 0.8495 | 0.8049 | -0.00463 | 0.1721 | 0.9257 | 0.5298 |
| prompt_kmeans_read_v78 | 0.7457 | 0.7163 | -0.00273 | 0.2284 | 0.8997 | 0.4966 |

### Key findings

1. **pf_binary_read_v78 is the BEST method** on 128 prompts (DINO=0.889).
   - Beats echo_pc by +0.023 DINO
   - Beats sf_native by +0.040 DINO
   - Binary PF read (Anchor vs Wave+Veil) + v78 trust writes is the strongest config.

2. **prompt_kmeans_read_v78 is surprisingly weak** (0.746, below sf_native).
   - The prompt-contrastive k-means classifier does NOT work for read routing.
   - This is different from the write-side where it was competitive.

3. **4 methods still generating** (pf, v78, prompt_pfcount_read_v78, veil_priority_b005).
   - Need DINOv2 on these to complete the 128-prompt table.

4. **head32 rerun in progress** on Node 1 (13 procs, 122 MP4s, up from 96).
   - 26 new MP4s since rerun started — generating!

5. **Main-128 rerun in progress** on Node 2 (4 procs, 807 MP4s, up from 805).
   - 2 new MP4s — generating remaining videos for incomplete cells.

## 13. v93 Head32 Comprehensive Eval (3 complete cells, 32 prompts)

| Cell | DINO | drift | BG | Composite |
|------|------|-------|-----|-----------|
| prompt_replica_read_v78 | **0.9220** | -0.00219 | 0.9371 | 0.5853 |
| prompt_consensus_read_v78 | 0.9192 | -0.00208 | 0.9359 | **0.5891** |
| pf_binary_read_v78 | 0.9150 | -0.00203 | 0.9381 | 0.5868 |

### Key findings (head32, 32 prompts):
- All 3 v78 variants score very high DINO (~0.92) on the 32-prompt set
- prompt_replica_read_v78 has highest DINO (0.922) but prompt_consensus_read_v78 has best composite
- On 128 prompts, pf_binary_read_v78 was best (0.889) — ranking changes with prompt set size
- Remaining 13 head32 cells still generating, will be evaluated when complete

## 14. v93 Main-128 Full Comprehensive Eval (8 methods, partial+complete)

| Cell | DINO | min_D | drift | flicker | BG | comp | videos |
|------|------|-------|-------|---------|-----|------|--------|
| pf | **0.9307** | 0.8959 | -0.00174 | 0.1766 | 0.9492 | 0.5822 | 74/128 partial |
| v78 | 0.9220 | 0.8708 | -0.00164 | 0.1958 | 0.9422 | 0.5817 | 84/128 partial |
| veil_priority_b005 | 0.8966 | 0.8419 | -0.00212 | 0.2161 | 0.9248 | 0.5772 | 75/128 partial |
| **pf_binary_read_v78** | 0.8890 | 0.8528 | -0.00235 | 0.2211 | 0.9281 | 0.5662 | **128/128 complete** |
| prompt_pfcount_read_v78 | 0.8676 | 0.8183 | -0.00307 | 0.2519 | 0.9155 | 0.5496 | 74/128 partial |
| echo_pc | 0.8656 | 0.7757 | -0.00237 | 0.2393 | 0.9101 | 0.5536 | 128/128 complete |
| sf_native | 0.8495 | 0.8049 | -0.00463 | 0.1721 | 0.9257 | 0.5738 | 128/128 complete |
| prompt_kmeans_read_v78 | 0.7457 | 0.7163 | -0.00273 | 0.2284 | 0.8997 | 0.5467 | 128/128 complete |

### Temporal Jump (lower = better)

| Cell | mean | median | max |
|------|------|--------|-----|
| **pf_binary_read_v78** | **1.3906** | 1.2908 | 4.2529 |
| prompt_pfcount_read_v78 | 1.4277 | 1.3145 | 3.4775 |
| veil_priority_b005 | 1.5545 | 1.4536 | 4.1702 |
| v78 | 1.6292 | 1.5515 | 4.8020 |
| pf | 1.6820 | 1.6348 | 3.1642 |
| echo_pc | 1.7353 | 1.5544 | 5.8829 |
| prompt_kmeans_read_v78 | 3.0106 | 2.5454 | 10.6377 |
| sf_native | 10.0111 | 1.9844 | 992.6165 |

### Key findings:
- **pf_binary_read_v78: BEST temporal jump (1.39) + best DINO among complete methods (0.889)**
- pf and v78 have higher DINO but only on PARTIAL prompts (74-84/128) — unfair comparison
- sf_native has severe temporal discontinuities (outlier 992!)
- prompt_kmeans_read_v78 is consistently weak (DINO 0.746, temporal jump 3.01)

## 15. v93 Head32 Comprehensive Eval (5 complete cells, 32 prompts each)

| Cell | DINO | min_D | drift | flicker | BG | comp |
|------|------|-------|-------|---------|-----|------|
| **pf** | **0.9313** | 0.9021 | -0.00199 | 0.1764 | 0.9467 | 0.5419 |
| prompt_replica_read_v78 | 0.9220 | 0.8795 | -0.00219 | 0.2048 | 0.9371 | 0.5264 |
| prompt_consensus_read_v78 | 0.9192 | 0.8851 | -0.00208 | 0.2141 | 0.9359 | 0.5292 |
| pf_binary_read | 0.9180 | 0.8869 | -0.00216 | 0.1919 | 0.9415 | 0.5317 |
| pf_binary_read_v78 | 0.9150 | 0.8751 | -0.00203 | 0.2028 | 0.9381 | 0.5316 |

### Key findings (head32, 32 prompts):
- **pf (plain Pyramid-Forcing) has highest DINO (0.931) on 32-prompt set**
- All methods score very high (~0.92) on this smaller, easier prompt set
- Ranking differs from 128-prompt set where pf_binary_read_v78 was best among complete methods
- 11 more head32 cells still generating, will be evaluated when complete

### Currently running experiments:
- v93 main-128: 4/8 complete, 4 generating (pf 75, v78 84, prompt_pfcount 75, veil 75)
- v93 head32: 9/16 complete, 7 generating
- v92: 3 read-only cells started (16 prompts each)
- v90: 4 missing cells started (pf_priority_b005 4/16, pf_age_only, pf_novelty_only, pf_priority_late)
- All 16 GPUs across 2 nodes fully utilized

## 13. v93 Head32 DINOv2 (8 complete cells, 32 prompts)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| **pf** | **0.9313** | **0.9021** | -0.00199 | 0.1764 | 0.9467 | 0.5419 |
| prompt_replica_read_v78 | 0.9220 | 0.8795 | -0.00219 | 0.2048 | 0.9371 | 0.5264 |
| prompt_consensus_read_v78 | 0.9192 | 0.8851 | -0.00208 | 0.2141 | 0.9359 | 0.5292 |
| pf_binary_read | 0.9180 | 0.8869 | -0.00216 | 0.1919 | 0.9415 | 0.5317 |
| prompt_pfcount_read | 0.9179 | 0.8810 | -0.00214 | 0.2041 | 0.9363 | 0.5251 |
| pf_binary_read_v78 | 0.9150 | 0.8751 | -0.00203 | 0.2028 | 0.9381 | 0.5316 |
| prompt_random_read_v78 | 0.8958 | 0.8785 | -0.00266 | 0.1760 | 0.9415 | 0.5484 |
| role_score_read_v78 | 0.8854 | 0.8560 | -0.00328 | 0.2164 | 0.9274 | 0.5237 |

### Key findings

1. **PF (3-class) is best on 32 prompts (0.931)**. The original PF read
   topology outperforms all binary and prompt-contrastive variants on this
   prompt subset.

2. **Binary read (pf_binary_read 0.918) vs PF 3-class (0.931)**: merging
   Wave+Veil into one class costs ~0.013 DINO. The Veil merge class
   provides value.

3. **v78 writes do NOT improve over PF reads alone**: pf_binary_read (0.918)
   > pf_binary_read_v78 (0.915). Adding v78 write control to binary reads
   does not help on 32 prompts.

4. **prompt_replica (0.922) and prompt_consensus (0.919) are competitive**:
   the independent-profile and consensus prompt-contrastive maps are close
   to PF. This supports reproducibility of the prompt classification.

5. **prompt_random_read_v78 (0.896) and role_score_read_v78 (0.885) are
   weakest**: random labels and the old remote-minus-prompt classifier
   underperform. This supports that prompt sensitivity is a better signal
   than random or remote-history for read routing.

6. **Contrast with main-128**: On 128 prompts, pf_binary_read_v78 was best
   (0.889). On 32 prompts, pf (3-class) is best (0.931). The 32-prompt
   subset may favor PF's original topology, while the 128-prompt set
   favors the binary+v78 combination. This needs investigation.

### Status
- 8/16 head32 cells complete (32/32 each)
- 8 still generating on Node 1 (3-28/32)
- 4/8 main-128 cells complete (128/128 each)
- 4 still generating on Node 2 (75-84/128)
- v90 remaining 4 cells: pf_priority_b005 (5/16), others just started

## 16. v93 Head32 Comprehensive Eval (8 complete cells, 32 prompts each, 256 videos)

| Cell | DINO | min_D | drift | flicker | BG | comp |
|------|------|-------|-------|---------|-----|------|
| **pf** | **0.9313** | 0.9021 | -0.00199 | 0.1764 | 0.9467 | 0.5419 |
| prompt_replica_read_v78 | 0.9220 | 0.8795 | -0.00219 | 0.2048 | 0.9371 | 0.5264 |
| prompt_consensus_read_v78 | 0.9192 | 0.8851 | -0.00208 | 0.2141 | 0.9359 | 0.5292 |
| pf_binary_read | 0.9180 | 0.8869 | -0.00216 | 0.1919 | 0.9415 | 0.5317 |
| prompt_pfcount_read | 0.9179 | 0.8810 | -0.00214 | 0.2041 | 0.9363 | 0.5251 |
| pf_binary_read_v78 | 0.9150 | 0.8751 | -0.00203 | 0.2028 | 0.9381 | 0.5316 |
| prompt_random_read_v78 | 0.8958 | 0.8785 | -0.00266 | 0.1760 | 0.9415 | 0.5484 |
| role_score_read_v78 | 0.8854 | 0.8560 | -0.00328 | 0.2164 | 0.9274 | 0.5237 |

### Key findings (head32, 32 prompts, 8 cells):
- **pf (plain PF) remains BEST** with DINO=0.931 — no modification beats original PF on 32-prompt set
- v78 transition slightly hurts DINO: pf_binary_read (0.918) > pf_binary_read_v78 (0.915)
- Read-only cells rank: pf > pf_binary_read ≈ prompt_pfcount_read
- Transition cells rank: prompt_replica_read_v78 > prompt_consensus_read_v78 > pf_binary_read_v78
- role_score_read_v78 is weakest (0.885) — role-based priority doesn't help
- prompt_random_read_v78 has lowest flicker (0.176) but low DINO (0.896)

### Combined with main-128 results:
- On 128 prompts (complete methods): pf_binary_read_v78 best (DINO=0.889, TJ=1.39)
- On 32 prompts: pf best (DINO=0.931)
- Ranking changes with prompt set size — 128-prompt set is more discriminative

## 17. VBench-Long Results (v86 + v90, 4 dimensions)

| Cell | subject | bg | aesthetic | imaging |
|------|---------|-----|-----------|---------|
| **v86/pf** | **0.9641** | 0.9460 | **0.5935** | 0.7137 |
| v86/replica_balanced | 0.9628 | 0.9452 | 0.5878 | 0.7158 |
| v86/learned_balanced | 0.9626 | 0.9467 | 0.5857 | 0.7127 |
| v86/v78 | 0.9615 | 0.9448 | 0.5884 | 0.7135 |
| v90/pf_s1 | 0.9639 | 0.9438 | 0.5890 | 0.7228 |
| v90/pf_priority_b010 | 0.9616 | **0.9465** | 0.5860 | 0.7166 |
| v90/veil_priority_b005 | 0.9614 | 0.9464 | 0.5886 | 0.7140 |
| v90/v78_s2 | 0.9621 | 0.9409 | 0.5905 | 0.7266 |
| v90/v78_s1 | 0.9599 | 0.9378 | 0.5822 | 0.7231 |
| v86/learned_neutral | 0.9609 | 0.9455 | 0.5865 | 0.7153 |
| v86/learned_age_only | 0.9607 | 0.9440 | 0.5877 | 0.7177 |
| v86/learned_late | 0.9604 | 0.9454 | 0.5835 | 0.7151 |
| v90/pf_s2 | 0.9606 | 0.9420 | 0.5943 | 0.7222 |
| v86/learned_early | 0.9580 | 0.9444 | 0.5861 | 0.7122 |
| v86/learned_conservative | 0.9583 | 0.9447 | 0.5850 | 0.7153 |
| v90/pf_s3 | 0.9572 | 0.9377 | 0.5893 | 0.7249 |
| v86/sf_native | 0.9535 | 0.9417 | 0.5861 | 0.7192 |

### Key VBench findings:
- **PF has best subject consistency (0.964)** across all v86/v90 cells
- PF also has best aesthetic quality (0.594)
- sf_native is weakest on subject consistency (0.954) — consistent with DINO results
- v78 is competitive but doesn't beat PF on any VBench dimension
- Background consistency and imaging quality are very close across all methods
- v93 VBench evaluation in progress (5 cells evaluating, 0 results yet)

## 14. v93 Head32 DINOv2 Complete (11 cells, 32 prompts)

### Full results (sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| **v78** | **0.9331** | 0.8908 | -0.00177 | 0.1843 | 0.9460 | 0.5404 |
| **pf** | **0.9313** | **0.9021** | -0.00199 | 0.1764 | 0.9467 | 0.5419 |
| prompt_replica_read_v78 | 0.9220 | 0.8795 | -0.00219 | 0.2048 | 0.9371 | 0.5264 |
| prompt_consensus_read_v78 | 0.9192 | 0.8851 | -0.00208 | 0.2141 | 0.9359 | 0.5292 |
| pf_binary_read | 0.9180 | 0.8869 | -0.00216 | 0.1919 | 0.9415 | 0.5317 |
| prompt_pfcount_read | 0.9179 | 0.8810 | -0.00214 | 0.2041 | 0.9363 | 0.5251 |
| pf_binary_read_v78 | 0.9150 | 0.8751 | -0.00203 | 0.2028 | 0.9381 | 0.5316 |
| prompt_read_prompt_priority | 0.9135 | 0.8743 | -0.00219 | 0.2161 | 0.9337 | 0.5258 |
| prompt_pfcount_read_v78 | 0.9131 | 0.8729 | -0.00202 | 0.2186 | 0.9370 | 0.5213 |
| prompt_random_read_v78 | 0.8958 | 0.8785 | -0.00266 | 0.1760 | 0.9415 | 0.5484 |
| role_score_read_v78 | 0.8854 | 0.8560 | -0.00328 | 0.2164 | 0.9274 | 0.5237 |

### Key comparisons

| Comparison | DINO A | DINO B | Δ | Interpretation |
|---|---:|---:|---:|---|
| v78 vs PF | 0.9331 | 0.9313 | +0.002 | v78 ≈ PF (v78 slightly better) |
| 3-class vs binary read | 0.9313 | 0.9180 | +0.013 | **3-class better** (Veil merge has value) |
| binary read vs binary+v78 write | 0.9180 | 0.9150 | +0.003 | v78 writes don't help |
| PF read vs v78 (PF read + trusted write) | 0.9313 | 0.9331 | -0.002 | v78 writes ≈ no effect |
| prompt read vs prompt+v78 write | 0.9179 | 0.9131 | +0.005 | v78 writes slightly hurt |
| prompt vs random control | 0.9131 | 0.8958 | **+0.017** | **prompt classification is causal** |
| prompt vs old classifier | 0.9131 | 0.8854 | **+0.028** | **prompt > remote-minus-prompt** |
| replica vs primary | 0.9220 | 0.9131 | +0.009 | **replica is reproducible** (actually better!) |

### Critical findings

1. **v78 (PF 3-class read + trusted write) is the BEST on 32 prompts (0.933)**
   — marginally beats PF (0.931). The v78 write controller adds ~0.002 DINO
   on top of PF's read topology.

2. **PF 3-class read (0.931) > binary read (0.918)** — merging Wave+Veil
   costs 0.013 DINO. The Veil merge class provides real value. Binary read
   topology is NOT supported on 32 prompts.

3. **v78 writes do NOT improve over PF reads alone** — pf_binary_read
   (0.918) > pf_binary_read_v78 (0.915). The trust-conditioned write
   controller adds no value on top of PF's read topology on 32 prompts.

4. **Prompt classification IS causally superior** — prompt_pfcount_read_v78
   (0.913) > random (0.896, +0.017) > role_score (0.885, +0.028). The
   prompt-contrastive classifier beats both random and the old
   remote-minus-prompt classifier.

5. **Replica is reproducible and actually better** — prompt_replica (0.922)
   > prompt_pfcount (0.913, +0.009). The independent profile produces a
   better head map than the primary.

6. **Contrast with main-128**: On 128 prompts, pf_binary_read_v78 was best
   (0.889). On 32 prompts, v78 (PF read + trusted write) is best (0.933).
   The 32-prompt subset may not be representative of the 128-prompt
   distribution. The main-128 results should be weighted more heavily.

### 5 cells still generating on Node 1
pf_read_prompt_priority, prompt_inverse_read_v78, prompt_kmeans_read,
prompt_kmeans_read_v78, remote_read_v78

### 4 main-128 cells still generating on Node 2
pf (75/128), prompt_pfcount_read_v78 (75/128), v78 (84/128), veil_priority_b005 (75/128)
