# Commit Forcing v76 Screen: Multiscale Bank and Trajectory Re-noising Results

> Date: 2026-07-23
> Experiment: `runs/v76_multiscale_commit_screen/`
> 16 cells × 12 evaluation prompts × 120 latent frames (≈30s) × seed 0
> Run commit: `13e7d4f`
> Depends on: `docs/75_commit_forcing_v74_screen_results.md`

## 1. Summary

**Gate A (trajectory re-noising) FAILED.** Trajectory-coupled re-noising
reduces delta/input from 1.32 to 0.10 (13× weaker). This gentler correction
is too weak to improve identity — DINO drops from 0.7985 (v74 fresh) to 0.7419
(trajectory), which is **below native** (0.7811).

**Gate B (multiscale bank) FAILED.** The multiscale compressed summary bank
does not improve over FIFO under matched re-noise mode.

**Gate C (motion gate)** is moot because both trajectory cells are below
native.

**PF is the clear leader** at DINO 0.8257, far above all Commit Forcing
variants.

## 2. Trace diagnostics

All 13 non-native traces are **nominal** with **zero failures**.

| Cell | delta/input | Max bank | Summary merges |
|---|---:|---:|---:|
| v74_hybrid_fresh | 1.317 | 4 | 0 (FIFO) |
| v74_origin2_fresh | 1.316 | 4 | 0 (FIFO) |
| ms_fresh_nomotion | 1.319 | 6 | 210+ |
| fifo_hybrid_trajectory | 0.100 | 4 | 0 (FIFO) |
| ms_full_motion | 0.099 | 6 | 210+ |
| ms_trajectory_nomotion | 0.098 | 6 | 210+ |

The trajectory mode successfully reduces correction magnitude by 13×, and
the multiscale bank successfully maintains 6-7 slots with active summary
merging. The mechanisms work as designed — the problem is that the weaker
correction is ineffective.

## 3. DINOv2 metrics (sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | bg | comp |
|---|---:|---:|---:|---:|---:|---:|
| pf_official | 0.8257 | 0.7661 | -0.00247 | 0.2776 | 0.9011 | 0.5053 |
| v74_origin2_fresh | 0.8051 | 0.7678 | -0.00304 | 0.2170 | 0.9040 | 0.5214 |
| v74_hybrid_fresh | 0.7985 | 0.7324 | -0.00338 | 0.2408 | 0.8931 | 0.5119 |
| ms_fresh_nomotion | 0.7964 | 0.7445 | -0.00416 | 0.2317 | 0.8925 | 0.5061 |
| sf_native | 0.7811 | 0.6732 | -0.00470 | 0.2409 | 0.8835 | 0.4887 |
| ms_t250_motion | 0.7748 | 0.7306 | -0.00555 | 0.2363 | 0.8904 | 0.4935 |
| fifo_origin2_trajectory | 0.7626 | 0.7182 | -0.00534 | 0.2683 | 0.8664 | 0.4731 |
| ms_representative_motion | 0.7507 | 0.6695 | -0.00602 | 0.2557 | 0.8709 | 0.4671 |
| ms_full_motion | 0.7465 | 0.6718 | -0.00598 | 0.2559 | 0.8660 | 0.4683 |
| ms_summary2_motion | 0.7460 | 0.6693 | -0.00648 | 0.2566 | 0.8698 | 0.4616 |
| ms_origin2_motion | 0.7427 | 0.6736 | -0.00673 | 0.2857 | 0.8573 | 0.4556 |
| ms_no_summary_read | 0.7423 | 0.6726 | -0.00603 | 0.2672 | 0.8604 | 0.4664 |
| fifo_hybrid_trajectory | 0.7419 | 0.6648 | -0.00602 | 0.2672 | 0.8613 | 0.4665 |
| ms_mean_motion | 0.7370 | 0.6762 | -0.00636 | 0.2682 | 0.8620 | 0.4655 |
| ms_trajectory_nomotion | 0.7356 | 0.6407 | -0.00643 | 0.2634 | 0.8648 | 0.4614 |

## 4. Gate analysis

### Gate A: trajectory re-noising — FAILED

| Cell | re-noise | delta/input | DINO | min_DINO | drift |
|---|---|---:|---:|---:|---:|
| v74_hybrid_fresh | fresh | 1.317 | 0.7985 | 0.7324 | -0.00338 |
| fifo_hybrid_trajectory | trajectory | 0.100 | 0.7419 | 0.6648 | -0.00602 |

ΔDINO = **-0.0567**. The trajectory mode is 13× weaker and drops DINO below
native. The correction is too gentle to alter the generation trajectory.

**Conclusion:** The v74 acceleration jumps were NOT primarily caused by fresh
noise. They were caused by the correction strength at specific timesteps.
Reducing correction magnitude eliminates the benefit entirely.

### Gate B: multiscale bank — FAILED

| Cell | bank | DINO | min_DINO |
|---|---|---:|---:|
| fifo_hybrid_trajectory | FIFO (4 slots) | 0.7419 | 0.6648 |
| ms_trajectory_nomotion | multiscale (6 slots) | 0.7356 | 0.6407 |

ΔDINO = **-0.0063**. The multiscale bank does not improve over FIFO under
matched trajectory re-noising.

Under fresh re-noising, ms_fresh_nomotion (0.7964) is slightly below
v74_hybrid_fresh (0.7985), confirming that the multiscale bank does not add
value.

**Conclusion:** The compressed summary bank is not useful. The FIFO trusted
bank in v74 was sufficient.

### Gate C: motion gate — moot

Both trajectory cells are below native, so the motion gate comparison is
not meaningful. ms_full_motion (0.7465) is slightly above
ms_trajectory_nomotion (0.7356), but both are far below v74 fresh.

## 5. Key findings

1. **Trajectory re-noising is too weak.** delta/input 0.10 is 13× weaker
   than fresh (1.32). The correction must be strong enough to actually
   change the generation path. The v74 fresh noise strength was correct.
2. **Multiscale compressed summaries do not help.** The FIFO bank in v74
   was sufficient. Adding dyadic temporal compression does not improve
   identity and slightly increases complexity.
3. **PF is far ahead.** pf_official DINO=0.8257, +0.045 over native and
   +0.020 over the best Commit Forcing cell (v74_origin2_fresh=0.8051).
4. **The best Commit Forcing configuration remains v74_origin2_fresh**
   (gate=0.15, fresh noise, 2 origin frames, DINO=0.8051).
5. **ms_fresh_nomotion** (multiscale + fresh noise, DINO=0.7964) is close
   to v74_hybrid_fresh (0.7985) but does not exceed it.

## 6. Interpretation

The v76 design hypothesized that the v74 failure modes (jumps, motion
freezing, style simplification) were caused by:
1. Fresh noise introducing stochastic discontinuities → **Wrong.** The
   fresh noise was not the problem; the correction strength was the benefit.
2. FIFO trusted history losing useful temporal context → **Wrong.** The
   multiscale bank does not improve identity.
3. Compressed summaries over-constraining motion → **Moot.** The motion
   gate cannot help when the base correction is too weak.

The actual cause of v74's failure modes is likely:
- **Style simplification**: repeated strong correction at the same
  timesteps biases the output distribution, regardless of noise mode.
- **Motion freezing**: the reference context over-constrains the generation
  at every block, regardless of bank structure.
- **Frame jumps**: the correction creates discontinuities at correction
  timesteps; weaker correction eliminates jumps but also eliminates benefit.

## 7. Next steps

1. **Return to v74 fresh noise as the base.** The fresh noise correction
   strength (delta~1.32) is necessary for identity improvement.
2. **Address jumps differently.** Instead of weaker correction, try:
   - Fewer correction timesteps (t250 only, which had DINO=0.7748)
   - Correction only at unreliable blocks (trigger mode)
   - Smoother correction scheduling (interpolate between corrected and
     uncorrected outputs)
3. **Address motion freezing differently.** Instead of motion gating on
   summaries, try:
   - Reducing correction frequency (every other block)
   - Using reference only at t500 (identity) and native at t250 (motion)
4. **Investigate why PF is so much better.** PF DINO=0.8257 vs our best
   0.8051. PF's per-head cache policy may be fundamentally more effective
   than pathwise correction for identity retention.
5. **Do not pursue multiscale bank or trajectory re-noising further.**
   Both are negative ablations.

## 8. Temporal jump diagnostic

| Cell | jump_mean | Δ vs native |
|---|---:|---:|
| pf_official | 1.7136 | -1.6578 |
| echo_pc | 1.7427 | -1.6286 (1/12 videos only) |
| v74_hybrid_fresh | 2.8649 | -0.5064 |
| ms_origin2_motion | 3.0045 | -0.3668 |
| ms_fresh_nomotion | 3.2182 | -0.1532 |
| ms_no_summary_read | 3.2569 | -0.1144 |
| fifo_hybrid_trajectory | 3.2772 | -0.0941 |
| fifo_origin2_trajectory | 3.3176 | -0.0537 |
| ms_t250_motion | 3.3490 | -0.0224 |
| sf_native | 3.3713 | — |
| v74_origin2_fresh | 3.4661 | +0.0948 |
| ms_mean_motion | 3.5339 | +0.1625 |
| ms_full_motion | 3.6449 | +0.2735 |
| ms_trajectory_nomotion | 3.6519 | +0.2805 |
| ms_representative_motion | 3.7252 | +0.3539 |
| ms_summary2_motion | 3.9470 | +0.5756 |

### Key findings

1. **PF is dramatically better on temporal jump**: 1.71 vs native 3.37
   (49% reduction). PF's per-head cache policy produces far fewer
   discontinuities.
2. **v74_hybrid_fresh has the best jump among Commit Forcing**: 2.86 (-15%
   vs native). This confirms the v74 result.
3. **Trajectory re-noising INCREASES jumps** (3.28 vs 2.86 for fresh). The
   hypothesis that fresh noise causes jumps is wrong — trajectory mode is
   actually 14% worse on jumps.
4. **Multiscale cells have worse jumps than FIFO** (ms_full_motion=3.64 vs
   fifo_hybrid_trajectory=3.28). The summary bank adds temporal
   discontinuities.
5. **ms_full_motion (proposed v76 default) is worse than native on both
   DINO (-0.035) AND temporal jump (+0.27).** The v76 design is a complete
   failure.

### Combined DINO + jump comparison

| Cell | DINO | jump | Assessment |
|---|---:|---:|---|
| pf_official | 0.8257 | 1.71 | **Best overall** — far ahead |
| v74_origin2_fresh | 0.8051 | 3.47 | Best CF DINO, but high jump |
| v74_hybrid_fresh | 0.7985 | 2.86 | **Best CF trade-off** |
| ms_fresh_nomotion | 0.7964 | 3.22 | Close to v74 but worse jump |
| sf_native | 0.7811 | 3.37 | Baseline |
| ms_full_motion | 0.7465 | 3.64 | **Worst CF** — below native on both |
| fifo_hybrid_trajectory | 0.7419 | 3.28 | Below native on DINO |

## 9. Human review

### 9.1 Review method

Human review of all 16 cells, prompt 0 (`0-0_ema.mp4`). Textual description,
no scoring.

### 9.2 Category 1: PF — identity maintained throughout

**Cell:** pf_official

- ID is well maintained throughout the entire 30-second video.
- No visible degradation, darkening, or style shift.
- This is the clear quality leader and the target to match.

### 9.3 Category 2: Echo-PC — promising but incomplete

**Cell:** echo_pc (only 1/12 videos generated due to OOM)

- Despite only one video being generated, ID is maintained from start to
  finish.
- This suggests Echo-Forcing's hierarchical scene/cache mechanism is
  effective for identity retention.
- The OOM issue prevents full evaluation, but the single video is
  encouraging.

### 9.4 Category 3: Fresh-noise Commit Forcing — visible improvement but
  limited

**Cells:** ms_fresh_nomotion, v74_hybrid_fresh

- Clear improvement over sf_native — consistent with the v74 review
  (Category 1 in docs/75).
- Same phenomena as v74's hybrid_admit015: delayed but not prevented
  degradation, with style simplification and motion freezing.
- v74_origin2_fresh has more visible jumps than the other fresh cells but
  similar ID retention.

### 9.5 Category 4: Trajectory/multiscale cells — worse than native

**Cells:** fifo_hybrid_trajectory, fifo_origin2_trajectory, ms_full_motion,
ms_mean_motion, ms_no_summary_read, ms_origin2_motion,
ms_representative_motion, ms_summary2_motion, ms_t250_motion,
ms_trajectory_nomotion

- Jumping begins after approximately 5 seconds.
- ID degrades rapidly and becomes unusable quickly.
- Obvious color shifts accompany the jumps.
- These cells are **visibly worse than sf_native**, confirming the metric
  results (all below native DINO).
- The trajectory re-noising and multiscale bank do not just fail to help —
  they actively harm the generation.

### 9.6 sf_native — baseline degradation

- Same as previous reviews: 5s onset, progressive darkening, 15s collapse.

### 9.7 Review conclusions

1. **PF is the only method that maintains ID throughout 30 seconds.** All
   Commit Forcing variants eventually degrade.
2. **Echo-PC is promising** — ID maintained in its single video. Worth
   resolving the OOM to evaluate fully.
3. **Trajectory re-noising is visibly harmful** — not just worse on metrics,
   but produces obvious jumping and color shifts starting at 5s. The gentler
   correction destabilizes rather than stabilizes.
4. **Fresh-noise Commit Forcing (v74 style) remains the best CF option** —
   visible improvement over native, but cannot match PF.
5. **The gap to PF is large and qualitative**, not just quantitative. PF
   maintains full ID; CF delays collapse but cannot prevent it.

## 10. Overall conclusion

The v76 screen is a clear negative result:

1. **Trajectory re-noising**: fails on DINO (-0.057 vs fresh) AND fails on
   temporal jump (+0.41 vs fresh) AND fails on human review (visible jumps
   from 5s, color shifts). All three hypotheses were wrong.
2. **Multiscale bank**: fails on DINO (-0.006 vs FIFO) AND fails on
   temporal jump (+0.36 vs FIFO). No benefit, only added complexity.
3. **Motion gate**: moot, since the base trajectory correction is too weak.
4. **PF dominates** on DINO (+0.045 vs native), temporal jump (-49% vs
   native), AND human review (full ID retention throughout 30s).
5. **Echo-PC shows promise** — ID maintained in its single video, but OOM
   prevents full evaluation.

The best Commit Forcing configuration remains **v74_hybrid_fresh** (fresh
noise, FIFO bank, gate=0.15, DINO=0.7985, jump=2.86). The v74 failure modes
(jumps, freeze, style degradation) must be addressed through a different
mechanism than weaker correction or temporal compression.
mechanism than weaker correction or temporal compression.
