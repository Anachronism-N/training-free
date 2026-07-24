# v86 TransitionCache Screen: Partial DINOv2 Results

> Date: 2026-07-24
> Experiment: `runs/v86_role_transition_screen/`
> 9 complete cells (16/16 MP4s) × 16 prompts × 120 frames (≈30s) × seed 0
> 5 cells still generating: pf, v78, learned_balanced, learned_late, replica_balanced
> Prompt suite: `prompts/v86_single_long_complex_16.txt` (16 complex single prompts)

## 1. DINOv2 metrics (9 complete cells, sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| pf_binary_balanced | 0.8529 | 0.7912 | -0.00185 | 0.2903 | 0.9023 | 0.5172 |
| inverse_balanced | 0.8519 | 0.7824 | -0.00142 | 0.2883 | 0.9013 | 0.5225 |
| consensus_balanced | 0.8504 | 0.7791 | -0.00238 | 0.2796 | 0.9053 | 0.5141 |
| learned_age_only | 0.8477 | 0.7890 | -0.00209 | 0.2812 | 0.9026 | 0.5158 |
| random_balanced | 0.8465 | 0.7848 | -0.00251 | 0.2838 | 0.9063 | 0.5129 |
| learned_early | 0.8448 | 0.7635 | -0.00192 | 0.2905 | 0.8946 | 0.5160 |
| learned_neutral | 0.8437 | 0.7515 | -0.00212 | 0.2911 | 0.8992 | 0.5135 |
| learned_conservative | 0.8364 | 0.7513 | -0.00272 | 0.2966 | 0.8961 | 0.5048 |
| sf_native | 0.7848 | 0.6850 | -0.00427 | 0.2471 | 0.8870 | 0.4941 |

## 2. Classification-causality analysis

### Key comparisons (DINO)

| Comparison | Δ DINO | Interpretation |
|---|---:|---|
| learned_neutral (0.844) vs learned_conservative (0.836) | +0.007 | Conservative role contrast slightly hurts |
| learned_conservative vs inverse_balanced (0.852) | **-0.016** | **Inverse labels BEAT learned — classification direction NOT confirmed** |
| learned_conservative vs random_balanced (0.847) | **-0.010** | **Random labels beat learned — classification not causally superior** |
| learned_conservative vs pf_binary_balanced (0.853) | **-0.017** | **PF binary labels beat learned** |
| consensus (0.850) vs learned_conservative (0.836) | +0.014 | Consensus (abstaining on uncertain heads) helps |
| learned_early (0.845) vs learned_age_only (0.848) | -0.003 | Depth restriction slightly worse than age-only |

### Critical finding

**The counterfactual classification does NOT improve DINO on the v86
cache-write lifecycle.** Inverse labels (0.852) and random labels (0.847)
both outperform learned conservative labels (0.836). PF binary labels
(0.853) are the best overall.

This is consistent with the v82 labels result: on the direct-recall path,
learned (0.817) did not beat pf_binary (0.851) or random (0.852). The v86
result extends this to the safer write-lifecycle path.

### What DOES help

1. **consensus_balanced (0.850)** — abstaining on uncertain heads (neutral
   fallback to v78) is better than forcing a conservative role split.
2. **learned_neutral (0.844)** — loading labels but keeping v78's uniform
   policy is better than the conservative role contrast.
3. **learned_age_only (0.848)** — isolating the refresh-age effect without
   role-conditioned novelty/budget is competitive.
4. **pf_binary_balanced (0.853)** — PF's static labels remain the strongest
   label map.

## 3. Pending cells

5 cells still generating (each at 4/16):
- **pf** — PF baseline (critical for comparison)
- **v78** — validated uniform transition (critical)
- **learned_balanced** — primary learned roles with full contrast
- **learned_late** — layers [15,30) only
- **replica_balanced** — independent-profile replication

These cells are essential:
- pf and v78 provide the baseline comparison
- learned_balanced is the primary proposed configuration
- replica_balanced tests reproducibility
- learned_late completes the depth ablation

## 4. Preliminary conclusion

**The role-conditioned cache lifecycle (v86) does not improve over v78
or PF-binary on DINO.** The counterfactual classification contributes
to temporal smoothness (learned < inverse on temporal jump in v82) but
not to identity retention. The recommended paper candidate remains **v78
(uniform trust-conditioned cache transition)**.

The final conclusion depends on:
1. pf and v78 DINO (pending) — if v78 matches/beats pf, v78 is confirmed
2. learned_balanced DINO (pending) — if full-contrast learned beats
   conservative, the role contrast may still have value
3. Temporal jump (pending) — if learned labels reduce jumps vs v78/pf
4. Human review — visual quality cannot be assessed from DINO alone
