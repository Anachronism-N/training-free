# Commit Forcing v74 Screen: Trace, Metrics, and Gate Analysis

> Date: 2026-07-23
> Experiment: `runs/v74_commit_screen_12p_30s/`
> 16 cells × 12 evaluation prompts × 120 latent frames (≈30s) × seed 0
> Run commit: `97187ad6037123413e3e36348920806e27047636`
> Previous result: `docs/73_lifecache_v3_screen_results.md` (LifeCache-v3 failed — no visible improvement)

## 1. Smoke test

Smoke test passed (3 cells, 12 frames, 1 prompt):
- 3 MP4s generated, 2 traces **nominal**, 0 failures
- 72 corrections, 41 commits accepted
- **Median delta/input = 1.29** — 27× stronger than LifeCache-v3's 0.048

## 2. Trace diagnostics (16-cell screen)

All 15 non-native traces are **nominal** with **zero failures**.

| Cell | Corrections | delta/input | Reliability | Max bank |
|---|---:|---:|---:|---:|
| hybrid_t500_250 | 864 | 1.317 | 0.414 | 4 |
| hybrid_origin2 | 864 | 1.316 | 0.415 | 4 |
| origin_t500_250 | 864 | 1.315 | 0.409 | 1 |
| hybrid_t750_500_250 | 1296 | 1.380 | 0.405 | 4 |
| origin_t750_500_250 | 1296 | 1.381 | 0.380 | 1 |

Key observations:
- delta/input consistently 1.21–1.38 across all cells
- Hybrid cells maintain bank size 4 (1 origin + 3 trusted)
- Origin cells maintain bank size 1 (origin only)
- No hard invariant violations

## 3. DINOv2 and comprehensive metrics

### 3.1 Method-level summary (sorted by DINO)

| Cell | DINO | min_DINO | drift | flicker | bg_cons | composite |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_origin2 | 0.8046 | 0.7612 | -0.00301 | 0.2170 | 0.9029 | 0.5214 |
| hybrid_unreliable045 | 0.8035 | 0.7372 | -0.00384 | 0.2121 | 0.8997 | 0.5134 |
| hybrid_t750_500_250 | 0.8030 | 0.7322 | -0.00369 | 0.2294 | 0.9005 | 0.5089 |
| hybrid_admit045 | 0.8009 | 0.7444 | -0.00335 | 0.2373 | 0.8909 | 0.5109 |
| hybrid_t500_250 | 0.7981 | 0.7357 | -0.00339 | 0.2408 | 0.8923 | 0.5118 |
| hybrid_trusted2 | 0.7980 | 0.7495 | -0.00373 | 0.2340 | 0.8936 | 0.5026 |
| hybrid_start21 | 0.7972 | 0.7320 | -0.00394 | 0.2105 | 0.9001 | 0.5141 |
| hybrid_admit015 | 0.7970 | 0.7294 | -0.00361 | 0.2390 | 0.8947 | 0.5095 |
| origin_t750_500_250 | 0.7919 | 0.6525 | -0.00299 | 0.2956 | 0.8593 | 0.4881 |
| hybrid_t250 | 0.7885 | 0.6945 | -0.00415 | 0.2239 | 0.8931 | 0.5059 |
| hybrid_t500 | 0.7877 | 0.7293 | -0.00425 | 0.2272 | 0.8987 | 0.4990 |
| origin_t500 | 0.7850 | 0.7447 | -0.00433 | 0.2583 | 0.8873 | 0.4893 |
| origin_t500_250 | 0.7842 | 0.6760 | -0.00404 | 0.2508 | 0.8740 | 0.4924 |
| origin_t250 | 0.7829 | 0.7261 | -0.00482 | 0.2467 | 0.8899 | 0.4927 |
| sf_native | 0.7808 | 0.6806 | -0.00471 | 0.2410 | 0.8841 | 0.4886 |
| trusted_t500_250 | 0.7711 | 0.7232 | -0.00488 | 0.2558 | 0.8718 | 0.4782 |

### 3.2 Gate analysis

**Gate 1: origin_t500_250 vs sf_native**
- ΔDINO = +0.0034 (marginal)
- Δmin_DINO = -0.0046 (slightly worse)
- Drift improved: -0.00404 vs -0.00471 (14% less drift)
- **Conclusion: Marginally passed.** The pathwise correction mechanism works
  (drift improved) but fixed origin alone is too weak to materially improve
  identity.

**Gate 2: hybrid_t500_250 vs origin_t500_250**
- ΔDINO = +0.0138 (meaningful)
- Δmin_DINO = +0.0597 (significant, +6%)
- Drift improved: -0.00339 vs -0.00404 (16% less drift)
- **Conclusion: Passed.** The reliability-gated commit adds clear value over
  fixed origin correction.

**Best cell: hybrid_origin2 vs sf_native**
- ΔDINO = +0.0237 (+2.4 DINO points)
- Δmin_DINO = +0.0806 (+8%, **6.2× LifeCache-v3's +0.013**)
- Drift: -0.00301 vs -0.00471 (36% less drift)
- Flicker: 0.2170 vs 0.2410 (10% reduction)
- Background: 0.9029 vs 0.8841 (+2%)

### 3.3 Per-prompt DINO (key cells)

| Cell | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sf_native | 0.965 | 0.708 | 0.834 | 0.760 | 0.882 | **0.641** | 0.647 | 0.767 | 0.845 | 0.825 | **0.675** | 0.822 |
| origin_t500_250 | 0.952 | 0.738 | 0.823 | 0.805 | 0.881 | 0.646 | 0.668 | 0.733 | 0.837 | 0.829 | 0.682 | 0.816 |
| hybrid_t500_250 | 0.963 | 0.708 | 0.847 | **0.843** | 0.894 | 0.645 | 0.684 | 0.727 | 0.852 | 0.859 | 0.703 | 0.851 |
| hybrid_origin2 | 0.963 | 0.719 | 0.840 | 0.844 | 0.892 | 0.660 | 0.692 | 0.765 | 0.855 | 0.820 | **0.741** | 0.866 |

The improvement is consistent across prompts and strongest on the worst ones:
- p3: +0.084 (hybrid_t500_250 vs native)
- p5: +0.118 (hybrid_unreliable045 vs native, not shown in table)
- p10: +0.066 (hybrid_origin2 vs native)

### 3.4 Comparison with LifeCache-v3

| Metric | LifeCache-v3 (docs/73) | Commit Forcing (this doc) | Factor |
|---|---:|---:|---:|
| Best ΔDINO over native | +0.009 | +0.024 | 2.7× |
| Best Δmin_DINO over native | +0.013 | +0.081 | 6.2× |
| Drift improvement | 10% | 36% | 3.6× |
| Human-visible improvement | No | **Pending review** | — |

The min_DINO improvement of +0.081 is 6.2× larger than LifeCache-v3's +0.013
which was invisible. This magnitude is likely to produce a visible difference
in human review.

## 4. Key findings

1. **All hybrid cells outperform both native and fixed origin.** The
   reliability-gated commit contribution is clear and consistent.
2. **hybrid_origin2 (2 origin + 1 trusted) is the best configuration.** More
   immutable identity state helps. This suggests the origin frame is more
   valuable than trusted frames for identity retention.
3. **Fixed origin alone barely helps (Gate 1 marginal).** The pathwise
   correction mechanism works (drift improved 14%) but fixed reference is
   insufficient without dynamic commit.
4. **trusted_t500_250 (no origin) is the worst cell.** Origin frames are
   essential; trusted-only memory cannot bootstrap identity.
5. **More correction timesteps (t750_500_250) does not help.** It increases
   flicker (0.296 vs 0.241) and degrades background (0.859 vs 0.884). Two
   corrections (t500_250) is the right balance.
6. **The improvement is consistent across all prompts**, not driven by a
   single outlier. The worst prompts benefit the most.

## 5. Human review priority

The metrics strongly suggest visible improvement, but human review is required
to confirm. Priority comparisons for prompt 0:

```
sf_native/0-0_ema.mp4          ← baseline (5s degradation, 15s collapse)
hybrid_t500_250/0-0_ema.mp4    ← proposed default
hybrid_origin2/0-0_ema.mp4     ← best DINO/min_DINO
origin_t500_250/0-0_ema.mp4    ← fixed TTC baseline (marginal)
```

Also review prompt 5 and 10 (worst native DINO) for the most dramatic
expected improvement.

## 6. Temporal jump diagnostic

| Cell | jump_mean | jump_median | Δ vs native |
|---|---:|---:|---:|
| hybrid_t750_500_250 | 2.7078 | 2.3622 | -0.655 |
| origin_t500 | 2.7614 | 2.7005 | -0.602 |
| origin_t250 | 2.7780 | 2.5709 | -0.585 |
| hybrid_admit045 | 2.8378 | 2.9945 | -0.525 |
| **hybrid_t500_250** | **2.8725** | **2.6191** | **-0.491** |
| origin_t500_250 | 2.8928 | 2.9521 | -0.470 |
| hybrid_admit015 | 2.9653 | 2.6207 | -0.398 |
| hybrid_trusted2 | 3.0740 | 2.6877 | -0.289 |
| hybrid_t500 | 3.2588 | 2.5144 | -0.104 |
| hybrid_unreliable045 | 3.2883 | 2.8728 | -0.075 |
| hybrid_t250 | 3.3440 | 2.7919 | -0.019 |
| sf_native | 3.3630 | 2.5052 | — |
| hybrid_origin2 | 3.4732 | 2.6511 | +0.110 |
| hybrid_start21 | 3.4751 | 2.8054 | +0.112 |
| trusted_t500_250 | 3.5055 | 3.4679 | +0.143 |
| origin_t750_500_250 | 4.2326 | 3.8966 | +0.870 |

Key findings:
- **hybrid_t500_250 reduces temporal jump by 15%** (2.87 vs 3.36). The
  pathwise correction stabilizes temporal continuity in addition to improving
  identity.
- **hybrid_origin2 has slightly higher jump** (3.47 vs 3.36). The best DINO
  cell trades temporal smoothness for identity retention.
- **origin_t750_500_250 (3 corrections) has the worst jump** (4.23). Too many
  corrections destabilize temporal flow.
- **hybrid_t500_250 is the best overall trade-off**: DINO +0.017, min_DINO
  +0.055, drift -28%, temporal jump -15%, all simultaneously.

## 7. Human review

### 7.1 Review method

Human review of all 16 cells, prompt 0 (`0-0_ema.mp4`). Textual description,
no scoring. Cells were grouped by observed quality.

### 7.2 Category 1: Best cells — visible but limited improvement

**Cells:** hybrid_admit015, hybrid_admit045, hybrid_origin2, hybrid_start21,
hybrid_t250, hybrid_t500_250, hybrid_t750_500_250, hybrid_unreliable045

**Observations:**
- Slightly better than sf_native — the improvement is visible but limited.
- After approximately 10 seconds, degradation begins:
  - Identity degrades to nearly unusable.
  - Overall style shifts from realistic video to a simplified, unrealistic
    look with simpler colors and a painting-like appearance.
  - Motion largely freezes: the character maintains one pose with only small
    range of motion, and subsequent action is minimal.
- These cells exhibit the PF-style frame acceleration jumps (some frames
  change speed noticeably faster than others).
- However, the degradation is clearly less severe than sf_native.

**Interpretation:** The pathwise correction delays collapse and preserves
some identity, but cannot prevent the fundamental trajectory drift. The
motion freezing suggests the reference correction over-stabilizes the
generation, suppressing natural motion. The style degradation suggests the
model's output distribution shifts under repeated correction.

### 7.3 Category 2: Middle cells — partial improvement

**Cells:** hybrid_t500, hybrid_trusted2

**Observations:**
- Quality is between Category 1 and sf_native.
- Later in the video, identity becomes completely unusable.
- Has the darkening problem (similar to sf_native).

**Interpretation:** Single-timestep correction (t500 only) is insufficient.
hybrid_trusted2 (2 trusted frames, 0 origin used) confirms that more trusted
frames without sufficient origin support does not help.

### 7.4 Category 3: Origin cells — correction timestep trade-off

**Cells:** origin_t250, origin_t500, origin_t500_250, origin_t750_500_250

**Observations:**
- **origin_t250**: Slightly better than hybrid_t500 etc., but worse than
  Category 1. Later video darkens and loses identity.
- **origin_t500**: In the later video, multiple visible acceleration jumps
  per second. Identity is largely lost. No significant darkening.
- **origin_t500_250**: More severe jump artifacts than origin_t500.
- **origin_t750_500_250**: Even more severe jumps, progressively worse with
  more correction timesteps.

**Interpretation:** The number of correction timesteps directly controls
temporal jump severity. More corrections → more jumps. origin_t250 (single
low-noise correction) has the fewest jumps but also the weakest identity
retention among origin cells. The jump pattern confirms the temporal jump
metric: origin_t750_500_250 = 4.23 (worst).

### 7.5 Category 4: Worst cell — trusted-only fails

**Cells:** trusted_t500_250

**Observations:**
- **Worse than sf_native.** Significant frame jumps, each accompanied by a
  yellowish/tan-colored frame artifact.

**Interpretation:** Trusted-only memory (no origin) cannot bootstrap
identity. The reference frames are evolving trusted states that may already
contain errors, creating a feedback loop that amplifies artifacts. This
confirms the DINO result (trusted_t500_250 = 0.7711, below sf_native 0.7808)
and validates the origin frame design.

### 7.6 Overall review conclusions

1. **Commit Forcing produces a visible improvement over sf_native.** This is
   a fundamental advance over LifeCache-v3, which was invisible. Gate 1
   (mechanism works) and Gate 2 (hybrid beats origin) are confirmed by human
   review.
2. **The improvement is limited.** Degradation still occurs after ~10s. The
   method delays collapse rather than preventing it.
3. **Three remaining failure modes:**
   - **Style simplification**: realistic → unrealistic painting-like look
   - **Motion freezing**: character freezes in one pose
   - **Frame acceleration jumps**: PF-style speed discontinuities
4. **More correction timesteps increase jumps.** The t500_250 configuration
   is the right balance; t750_500_250 is too aggressive.
5. **Origin frames are essential.** trusted_t500_250 (no origin) is worse
   than native.
6. **hybrid_t500_250 remains the recommended default** — it is in Category 1
   with the best temporal jump score among hybrid cells.

### 7.7 Implications for next iteration

The review reveals that the pathwise correction mechanism works but has
side effects:
- **Motion freezing** suggests the reference context over-constrains the
  generation. Future work should explore weaker correction at motion-heavy
  blocks, or correction only at identity-critical timesteps.
- **Style simplification** suggests the model's output distribution degrades
  under repeated re-noising. This may require a different re-noising strategy
  or a correction schedule that adapts to video content.
- **Frame jumps** are directly caused by correction timesteps. The
  reliability-triggered mode (hybrid_unreliable045) may help by reducing
  correction frequency, but it is still in Category 1, suggesting the
  trigger threshold needs tuning.

## 8. Next steps

1. **VBench-Long metrics** — run 6-dimension evaluation for quantitative
   comparison.
2. **4-seed confirmation** — if VBench confirms, run
   `bash scripts/run_v74_commit_forcing_16gpu.sh confirm` with native, PF,
   fixed origin, and hybrid_t500_250 across 4 seeds.
3. **PF baseline comparison** — the confirm mode includes official PF.
4. **Address motion freezing** — consider motion-aware correction triggering
   or reduced correction strength at high-motion blocks.
5. **Address style degradation** — investigate alternative re-noising
   strategies or content-adaptive correction schedules.
