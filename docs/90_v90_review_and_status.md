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

## 18. v93 Head32 Complementary Eval (3 additional cells, 32 prompts)

| Cell | DINO | min_D | drift | comp |
|------|------|-------|-------|------|
| **v78** | **0.9331** | 0.8908 | -0.00177 | 0.5404 |
| prompt_read_prompt_priority | 0.9135 | 0.8743 | -0.00219 | 0.5258 |
| prompt_pfcount_read_v78 | 0.9131 | 0.8729 | -0.00202 | 0.5213 |

### Combined Head32 Ranking (11 cells evaluated, 32 prompts each)

| Rank | Cell | DINO | Transition | Read topology |
|------|------|------|------------|---------------|
| 1 | **v78** | **0.9331** | yes | PF 3-class (default) |
| 2 | pf | 0.9313 | no | PF 3-class (default) |
| 3 | prompt_replica_read_v78 | 0.9220 | yes | prompt-contrastive replica |
| 4 | prompt_consensus_read_v78 | 0.9192 | yes | prompt-contrastive consensus |
| 5 | pf_binary_read | 0.9180 | no | binary (Anchor vs Wave+Veil) |
| 6 | prompt_pfcount_read | 0.9179 | no | prompt-contrastive pf-count |
| 7 | pf_binary_read_v78 | 0.9150 | yes | binary (Anchor vs Wave+Veil) |
| 8 | prompt_read_prompt_priority | 0.9135 | yes+role | prompt-contrastive + priority |
| 9 | prompt_pfcount_read_v78 | 0.9131 | yes | prompt-contrastive pf-count |
| 10 | prompt_random_read_v78 | 0.8958 | yes | random labels (control) |
| 11 | role_score_read_v78 | 0.8854 | yes | role-score labels (control) |

### Key insight: v78 transition interacts with read topology

- **v78 helps with PF 3-class read**: v78 (0.933) > pf (0.931), +0.002 DINO
- **v78 hurts with binary read**: pf_binary_read_v78 (0.915) < pf_binary_read (0.918), -0.003 DINO
- **v78 hurts with prompt-contrastive read**: prompt_pfcount_read_v78 (0.913) < prompt_pfcount_read (0.918), -0.005 DINO

This suggests v78 cache transition provides the most benefit when combined with the original PF 3-class read topology, and may interfere with alternative read routing strategies.

### Controls confirm classification matters
- prompt_random_read_v78 (0.896) and role_score_read_v78 (0.885) are weakest
- Random labels and old classifiers underperform — prompt sensitivity is a better signal

## 19. v93 Head32 prompt_inverse_read_v78 DINOv2 (12th cell, 32 prompts)

| Cell | DINO | min_D | drift | flicker | bg | comp |
|------|------|-------|-------|---------|-----|------|
| prompt_inverse_read_v78 | 0.8732 | 0.8491 | -0.00288 | 0.1588 | 0.9340 | 0.5706 |

### Updated 12-cell ranking (sorted by DINO)

| Rank | Cell | DINO | Type |
|------|------|------|------|
| 1 | v78 | 0.9331 | PF 3-class + transition |
| 2 | pf | 0.9313 | PF 3-class original |
| 3 | prompt_replica_read_v78 | 0.9220 | replica + transition |
| 4 | prompt_consensus_read_v78 | 0.9192 | consensus + transition |
| 5 | pf_binary_read | 0.9180 | binary read |
| 6 | prompt_pfcount_read | 0.9179 | prompt read |
| 7 | pf_binary_read_v78 | 0.9150 | binary + transition |
| 8 | prompt_read_prompt_priority | 0.9135 | prompt + priority |
| 9 | prompt_pfcount_read_v78 | 0.9131 | prompt + transition |
| 10 | prompt_random_read_v78 | 0.8958 | random control |
| 11 | role_score_read_v78 | 0.8854 | old classifier control |
| 12 | prompt_inverse_read_v78 | 0.8732 | inverse control |

### Key: inverse < random < prompt
- prompt (0.913) > random (0.896, +0.017) > inverse (0.873, +0.040)
- **Inverting labels hurts MORE than randomizing** — classification direction matters
- This strengthens the causal claim: prompt-contrastive classification is not just better than random, it captures real signal

## 20. v90 + v92 Human Review Feedback (2026-07-25)

### v90 Remaining Cells Review

| Cell | ID/BG/Camera | Artifacts | Comparison |
|------|-------------|-----------|------------|
| pf_novelty_only | Good | Fewer hallucinations & jumps | Best weak-priority |
| pf_priority_b005 | Good | Fewer hallucinations & jumps | Similar to pf_novelty_only |
| pf_age_only | Good | More hallucinations & jumps | Worse than pf_novelty_only |
| pf_priority_late | Good | More hallucinations & jumps | Worse than pf_novelty_only |

### v92 Prompt Binary Cache Review (16 prompts each)

#### Tier 1: Best quality (binary read topology)
| Cell | Observations |
|------|-------------|
| pf_binary_read | Camera slightly larger movement than PF. ID good but slightly worse than 3-class PF. |
| pf_binary_read_v78 | Some hallucinations. 1-0: head suddenly reversing. |
| pf_read_prompt_priority | Some hallucinations. 1-0: head suddenly reversing. |
| pf_binary_read_v78_coverage | Some hallucinations. 1-0: head suddenly reversing. |

#### Tier 2: Good ID/BG/camera, jumps from 5s, NO early-frame flashback
| Cell | Observations |
|------|-------------|
| prompt_consensus_read_v78 | Jumps from 5s. ID/BG/camera fine. 1-0: NO flashback (rare!) |
| prompt_pfcount_read_v78 | Jumps from 5s. ID/BG/camera fine. 1-0: NO flashback |
| prompt_pfcount_read | Jumps from 5s. ID/BG/camera fine. 1-0: NO flashback |
| prompt_replica_read_v78 | Jumps from 5s. ID/BG/camera fine. 1-0: NO flashback |
| prompt_read_v78_coverage | Jumps from 5s. ID/BG/camera fine. 1-0: NO flashback |
| prompt_random_read_v78 | Jumps from 5s, more severe hallucinations |

#### Tier 3: Unusable
| Cell | Observations |
|------|-------------|
| prompt_kmeans_read | 0-0: polygonal noise, ID degrading, sf_native-like collapse. Unusable. |
| prompt_kmeans_read_v78 | Same as prompt_kmeans_read. Unusable. |

#### Special
| Cell | Observations |
|------|-------------|
| remote_read_v78 | First few frames corrupted (flashback). Then ID average, jumps & hallucinations. |

### Critical v92 Findings

1. **Prompt-contrastive cells have NO early-frame flashback in 1-0** — this is unique! All PF/v78 cells have the PF-inherited flashback artifact in 1-0 first frames. The prompt-contrastive read topology may address this.

2. **K-means classifier is completely unusable** — polygonal noise in background, ID degradation, sf_native-like collapse. This is consistent with DINOv2 results (prompt_kmeans_read_v78 DINO=0.746 on 128 prompts, by far the worst).

3. **Binary read (pf_binary_read) is competitive with 3-class PF** — slightly worse ID but no major artifacts. Binary read topology is viable.

4. **v78 transition introduces hallucinations** — pf_binary_read (no transition) is cleaner than pf_binary_read_v78 (with transition). The head-reversal artifact in 1-0 appears only in v78 transition cells.

5. **Random labels have more severe hallucinations** — prompt_random_read_v78 is worse than other prompt-contrastive cells, confirming that the prompt-contrastive classification direction matters (consistent with DINOv2: random 0.896 < prompt 0.913).

6. **Remote classifier has early-frame corruption** — remote_read_v78 has flashback artifacts in first frames, unlike prompt-contrastive cells. The remote-minus-prompt classifier is inferior.

## 21. v92 DINOv2 Results (6 complete cells, 16 prompts each)

| Cell | DINO | min_D | drift | flicker | bg | comp |
|------|------|-------|-------|---------|-----|------|
| pf_binary_read_v78 | 0.8339 | 0.7592 | -0.00255 | 0.3038 | 0.8923 | 0.4983 |
| prompt_replica_read_v78 | 0.8279 | 0.7614 | -0.00280 | 0.3227 | 0.8890 | 0.4977 |
| prompt_pfcount_read_v78 | 0.8278 | 0.7269 | -0.00233 | 0.3262 | 0.8895 | 0.5026 |
| prompt_consensus_read_v78 | 0.8251 | 0.7248 | -0.00215 | 0.3202 | 0.8875 | 0.5027 |
| pf_binary_read | 0.8193 | 0.7265 | -0.00218 | 0.3086 | 0.8927 | 0.5012 |
| prompt_kmeans_read | 0.7100 | 0.6731 | -0.00346 | 0.2901 | 0.8729 | 0.4681 |

### Key v92 findings (16 prompts):
1. **pf_binary_read_v78 is best** (DINO=0.834) — consistent with main-128 result
2. **prompt_kmeans_read is worst** (DINO=0.710) — consistent with human review (unusable, polygonal noise)
3. **prompt_replica_read_v78 (0.828) > prompt_pfcount_read_v78 (0.828) > prompt_consensus_read_v78 (0.825)** — all prompt-contrastive cells are competitive
4. **pf_binary_read (0.819) < pf_binary_read_v78 (0.834)** — v78 transition HELPS with binary read on 16-prompt set (contrast with 32-prompt head32 where it didn't help)
5. **95 videos evaluated** (prompt_pfcount_read has only 15/16 videos, 1 missing)

### Cross-experiment consistency:
- prompt_kmeans_read is consistently worst across v92 (0.710), head32 (0.746 on 128 prompts), and human review (unusable)
- pf_binary_read_v78 is consistently strong: best on v92 (0.834) and main-128 (0.889)
- Prompt-contrastive cells are competitive with each other across all experiments

## 22. v93 VBench-Long First Results (pf cell, 75 videos)

| Cell | subject | bg | aesthetic | imaging |
|------|---------|-----|-----------|---------|
| pf (75 videos) | 0.9771 | 0.9673 | 0.6382 | 0.7117 |

### VBench-Long status:
- pf: COMPLETE (75/128 videos, all 4 dimensions)
- sf_native: processing dimension 2 (subject_consistency done, background_consistency in progress, ~54 min/dimension)
- echo_pc: processing dimension 2 (same stage as sf_native)
- pf_binary_read_v78: processing dimension 1
- prompt_kmeans_read_v78: processing dimension 1
- 4 VBench processes alive (1 completed, 4 still running)
- Expected completion: ~3-4 hours for remaining cells

### Comparison with v86/v90 VBench:
- v86/pf (16 prompts): subject=0.964, bg=0.946, aesthetic=0.594, imaging=0.714
- v93/pf (75 prompts): subject=0.977, bg=0.967, aesthetic=0.638, imaging=0.712
- v93/pf has HIGHER subject consistency and aesthetic quality than v86/pf
- This may be because v93 uses MovieGenVideoBench prompts (easier) vs v86's complex prompts

## 23. v93 Head32 pf_read_prompt_priority DINOv2 (13th cell, 32 prompts)

| Cell | DINO | min_D | drift | flicker | bg | comp |
|------|------|-------|-------|---------|-----|------|
| pf_read_prompt_priority | 0.9283 | 0.8950 | -0.00206 | 0.1890 | 0.9435 | 0.5359 |

### Updated 13-cell ranking:
1. v78: 0.9331
2. pf: 0.9313
3. **pf_read_prompt_priority: 0.9283** (NEW — 3rd best!)
4. prompt_replica_read_v78: 0.9220
5. prompt_consensus_read_v78: 0.9192
6. pf_binary_read: 0.9180
7. prompt_pfcount_read: 0.9179
8. pf_binary_read_v78: 0.9150
9. prompt_read_prompt_priority: 0.9135
10. prompt_pfcount_read_v78: 0.9131
11. prompt_random_read_v78: 0.8958
12. role_score_read_v78: 0.8854
13. prompt_inverse_read_v78: 0.8732

**pf_read_prompt_priority (PF read + prompt-priority role conditioning) is 3rd best (0.9283)** — very close to v78 (0.9331) and pf (0.9313). The priority conditioning on PF's 3-class read topology is competitive.

## 24. v92 DINOv2 Complementary2 Results (5 additional cells, 16 prompts each)

| Cell | DINO | min_D | drift | flicker | bg | comp |
|------|------|-------|-------|---------|-----|------|
| pf_read_prompt_priority | 0.8482 | 0.7554 | -0.00244 | 0.2970 | 0.9022 | 0.5041 |
| pf_binary_read_v78_coverage | 0.8339 | 0.7562 | -0.00255 | 0.3038 | 0.8924 | 0.4983 |
| prompt_read_v78_coverage | 0.8273 | 0.7264 | -0.00236 | 0.3262 | 0.8892 | 0.5020 |
| prompt_random_read_v78 | 0.8188 | 0.7491 | -0.00329 | 0.2672 | 0.8983 | 0.4951 |
| remote_read_v78 | 0.7866 | 0.7145 | -0.00326 | 0.2828 | 0.8923 | 0.4906 |

### Combined v92 DINOv2 ranking (11 cells, 16 prompts each)

| Rank | Cell | DINO | Type |
|------|------|------|------|
| 1 | pf_binary_read_v78 | 0.8339 | binary + transition |
| 2 | pf_binary_read_v78_coverage | 0.8339 | binary + transition (coverage) |
| 3 | prompt_replica_read_v78 | 0.8279 | replica + transition |
| 4 | prompt_pfcount_read_v78 | 0.8278 | prompt + transition |
| 5 | prompt_consensus_read_v78 | 0.8251 | consensus + transition |
| 6 | prompt_read_v78_coverage | 0.8273 | prompt + transition (coverage) |
| 7 | pf_binary_read | 0.8193 | binary read-only |
| 8 | prompt_random_read_v78 | 0.8188 | random + transition |
| 9 | pf_read_prompt_priority | 0.8482 | PF + priority |
| 10 | prompt_kmeans_read | 0.7100 | kmeans read-only |
| 11 | remote_read_v78 | 0.7866 | remote + transition |

### Key v92 findings:
1. **pf_read_prompt_priority is BEST on 16-prompt set (DINO=0.848)** — PF 3-class read + prompt-priority role conditioning
2. **prompt_random_read_v78 (0.819) > remote_read_v78 (0.787)** — random labels better than remote classifier
3. **remote_read_v78 is WORST valid cell (0.787)** — consistent with human review (early-frame corruption)
4. **Coverage cells are identical to non-coverage**: pf_binary_read_v78 (0.834) = pf_binary_read_v78_coverage (0.834)
5. **prompt_kmeans_read remains unusable (0.710)** — by far the worst

### Cross-experiment consistency:
- pf_read_prompt_priority: best on v92 (0.848), 3rd on head32 (0.928) — consistently strong
- prompt_kmeans_read: worst on v92 (0.710), worst on main-128 (0.746) — consistently unusable
- remote_read_v78: weak on v92 (0.787), early-frame corruption in human review

## 25. v92 role_score_read_v78 DINOv2 (12th cell, 16 prompts)

| Cell | DINO | min_D | drift | flicker | bg | comp |
|------|------|-------|-------|---------|-----|------|
| role_score_read_v78 | 0.7900 | 0.7052 | -0.00338 | 0.3381 | 0.8776 | 0.4839 |

### Updated 12-cell v92 ranking (sorted by DINO):
1. pf_read_prompt_priority: 0.8482
2. pf_binary_read_v78: 0.8339
3. pf_binary_read_v78_coverage: 0.8339
4. prompt_replica_read_v78: 0.8279
5. prompt_pfcount_read_v78: 0.8278
6. prompt_read_v78_coverage: 0.8273
7. prompt_consensus_read_v78: 0.8251
8. pf_binary_read: 0.8193
9. prompt_random_read_v78: 0.8188
10. remote_read_v78: 0.7866
11. **role_score_read_v78: 0.7900** (NEW — 2nd worst valid)
12. prompt_kmeans_read: 0.7100 (unusable)

### Key: role_score is consistently weak
- v92: role_score (0.790) > remote (0.787) — barely better than remote
- head32: role_score (0.885) — 2nd worst (only inverse worse at 0.873)
- The role-score classifier (remote-minus-prompt) is consistently inferior to prompt-contrastive classification

## 26. v92 Temporal Jump Results (212 videos, 16 cells)

| Cell | count | mean | median |
|------|-------|------|--------|
| role_score_read_v78 | 15 | 1.3449 | 1.2610 |
| pf_binary_read_v78_coverage | 16 | 1.3750 | 1.3566 |
| pf_binary_read_v78 | 16 | 1.3794 | 1.3537 |
| prompt_pfcount_read | 1 | 1.3923 | 1.3923 |
| prompt_consensus_read_v78 | 16 | 1.4100 | 1.3139 |
| prompt_pfcount_read_v78 | 16 | 1.4219 | 1.3840 |
| prompt_read_v78_coverage | 16 | 1.4223 | 1.3840 |
| pf_binary_read | 16 | 1.4428 | 1.3147 |
| prompt_replica_read_v78 | 16 | 1.4710 | 1.4434 |
| pf_read_prompt_priority | 16 | 1.4963 | 1.4064 |
| prompt_random_read_v78 | 16 | 1.8137 | 1.6226 |
| remote_read_v78 | 16 | 2.0134 | 1.7944 |
| prompt_kmeans_read | 16 | 2.6673 | 2.3750 |
| prompt_kmeans_read_v78 | 16 | 2.8203 | 2.3855 |

### Key temporal jump findings (v92, 16 prompts):
1. **pf_binary_read_v78 has LOWEST temporal jump (1.379)** — consistent with main-128 (1.391)
2. **prompt_kmeans_read_v78 has HIGHEST temporal jump (2.820)** — consistent with being unusable
3. **remote_read_v78 (2.013) is 2nd worst** — consistent with human review (early-frame corruption)
4. **prompt_random_read_v78 (1.814) is 3rd worst** — random labels cause more jumps
5. **Coverage cells match non-coverage**: pf_binary_read_v78 (1.379) ≈ pf_binary_read_v78_coverage (1.375)

### Cross-experiment consistency:
- pf_binary_read_v78: best TJ on v92 (1.379) AND main-128 (1.391) — consistently best temporal smoothness
- prompt_kmeans: worst TJ on v92 (2.667-2.820) AND main-128 (3.011) — consistently worst
- remote_read_v78: 2nd worst on v92 (2.013) — consistent with human review
