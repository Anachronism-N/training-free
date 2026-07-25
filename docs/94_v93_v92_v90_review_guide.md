# Human Review Guide — v93/v92/v90 Experiments

## Video Paths for Review

### v93 Main-128 (128 prompts, 120 frames each)
Path: `runs/v93_moviebench128_main/`

| Method | Videos | Status | DINO | TJ |
|--------|--------|--------|------|-----|
| sf_native | 128/128 | DONE | 0.850 | 10.01 |
| echo_pc | 128/128 | DONE | 0.866 | 1.74 |
| pf_binary_read_v78 | 128/128 | DONE | **0.889** | **1.39** |
| prompt_kmeans_read_v78 | 128/128 | DONE | 0.746 | 3.01 |
| pf | 77/128 | partial | 0.931* | 1.68 |
| v78 | 84/128 | partial | 0.922* | 1.63 |
| prompt_pfcount_read_v78 | 76/128 | partial | 0.868* | 1.43 |
| veil_priority_b005 | 76/128 | partial | 0.897* | 1.55 |

*Partial: DINO computed on available videos only, may be biased toward easier prompts.

### v93 Head32 (32 prompts, 120 frames each)
Path: `runs/v93_moviebench32_head/`

| Method | Videos | Status | DINO |
|--------|--------|--------|------|
| pf | 32/32 | DONE | **0.931** |
| pf_binary_read | 32/32 | DONE | 0.918 |
| pf_binary_read_v78 | 32/32 | DONE | 0.915 |
| prompt_consensus_read_v78 | 32/32 | DONE | 0.919 |
| prompt_pfcount_read | 32/32 | DONE | 0.918 |
| prompt_random_read_v78 | 32/32 | DONE | 0.896 |
| prompt_replica_read_v78 | 32/32 | DONE | 0.922 |
| role_score_read_v78 | 32/32 | DONE | 0.885 |
| prompt_read_prompt_priority | 32/32 | DONE | — |
| prompt_pfcount_read_v78 | 32/32 | DONE | — |
| v78 | 32/32 | DONE | — |
| remote_read_v78 | 11/32 | restarted | — |
| prompt_kmeans_read_v78 | 3/32 | restarted | — |
| pf_read_prompt_priority | 5/32 | restarted | — |
| prompt_inverse_read_v78 | 16/32 | running | — |

### v90 (16 prompts)
Path: `runs/v90_priority_factorization_screen/`

| Method | Videos | Status |
|--------|--------|--------|
| pf_s1, pf_s2, pf_s3 | 16 each | DONE |
| v78_s1, v78_s2, v78_s3 | 16 each | DONE |
| inverse_priority_b005 | 16 | DONE |
| learned_priority_b005 | 16 | DONE |
| pf_priority_b010 | 16 | DONE |
| random_priority_b005 | 16 | DONE |
| veil_priority_b005 | 16 | DONE |
| wave_priority_b005 | 16 | DONE |
| pf_priority_b005 | 16 | DONE |
| pf_age_only | 1/16 | running |
| pf_novelty_only | 1/16 | running |
| pf_priority_late | 0/16 | running |

### v92 (16 prompts)
Path: `runs/v92_prompt_binary_cache_screen/`

| Method | Videos | Status |
|--------|--------|--------|
| pf_binary_read | 14/16 | running |
| pf_binary_read_v78 | 6/16 | running |
| prompt_pfcount_read | 1/16 | running |
| prompt_pfcount_read_v78 | 7/16 | running |
| prompt_kmeans_read | 7/16 | running |
| prompt_replica_read_v78 | 3/16 | running |
| prompt_consensus_read_v78 | 3/16 | running |
| 9 more cells | 0/16 | not started |

## Key Findings

1. **pf_binary_read_v78** is the best complete method on 128 prompts:
   - DINO=0.889 (beats echo_pc 0.866, sf_native 0.850)
   - Temporal jump=1.39 (lowest, best temporal smoothness)
   - Binary PF read (Anchor vs Wave+Veil) + v78 trust-conditioned writes

2. **pf** (original 3-class PF) is best on 32 prompts (DINO=0.931)
   - v78 transition slightly hurts DINO on 32-prompt set
   - Ranking changes with prompt set size

3. **sf_native** has severe temporal discontinuities (TJ=10.01, outlier=992!)
   - PF and all PF variants dramatically improve temporal smoothness

4. **prompt_kmeans_read_v78** is consistently weak (DINO=0.746, TJ=3.01)
   - Prompt-contrastive k-means classifier doesn't work for read routing

5. v90 matched-seed: v78 ≈ PF (not consistently better across seeds 0-2)

## Metrics Available
- Main-128: comprehensive (8 methods) + temporal jump (813 videos) ✅
- Head32: comprehensive (8 cells, 256 videos) + temporal jump (288 videos) ✅
- VBench-Long: 5 cells evaluating (4 dims: subject/bg consistency, aesthetic, imaging)
