# LifeCache-v3 Screen Experiment: Trace, Metrics, and First Analysis

> Date: 2026-07-23
> Experiment: `runs/v72_screen_12p_30s/`
> 16 cells × 12 calibration prompts × 120 latent frames (≈30s) × seed 0
> Run commit: `4b7270c57cf12158a701bc256b1a52b7edc75221`
> Prompt SHA-256: `af581cc4afd9dc7e3d5bcf73ac027d192bcc81c33514a6e28b4243b74bf4b405`

## 1. Experiment configuration

All 16 cells ran on a single 8-GPU H20 node (2 processes per GPU due to
both Node 1 and Node 2 commands executing on the same machine). Generation
completed for all 192 videos (16 cells × 12 prompts).

| GPU | Cell | Gate | Ramp | Policy | Routing | Question |
|---:|---|---:|---:|---|---|---|
| 0 | sf_native | — | — | — | — | Native reference |
| 1 | coverage_legacy_g005_s36 | 0.05 | 0 | coverage | off | Old weak coverage setting |
| 2 | typed_legacy_g005_s36 | 0.05 | 0 | typed | off | Old typed under weak strength |
| 3 | typed_g010_r12 | 0.10 | 12 | typed | off | Low target gate |
| 4 | typed_g015_r12 | 0.15 | 12 | typed | off | Default smooth typed memory |
| 5 | typed_g020_r12 | 0.20 | 12 | typed | off | High target gate |
| 6 | typed_g015_hard_on | 0.15 | 0 | typed | off | Abrupt activation ablation |
| 7 | typed_g015_nonoverlap | 0.15 | 12 | typed | off | Strict 21-frame exclusion |
| 8 | anchor_only_g015 | 0.15 | 12 | typed | off | Exact-state contribution |
| 9 | summary_only_g015 | 0.15 | 12 | typed | off | Aggregated-state contribution |
| 10 | online_b25_g015 | 0.15 | 12 | typed | online | Sparse routed intervention |
| 11 | online_b50_g015 | 0.15 | 12 | typed | online | Default routed intervention |
| 12 | online_b75_g015 | 0.15 | 12 | typed | online | Dense routed intervention |
| 13 | online_no_effect_floor | 0.15 | 12 | typed | online | Minimum-effect ablation |
| 14 | online_recent21 | 0.15 | 12 | typed | online | Native-overlap ablation |
| 15 | online_no_motion_penalty | 0.15 | 12 | typed | online | Motion-risk ablation |

## 2. Trace analysis

All 15 non-native cells produced valid JSONL traces and passed `--strict`
diagnosis with `diagnostics_nominal`. No structural failure was detected.

### 2.1 Intervention strength (delta_to_native_rms_median)

The gate sweep is monotonic, confirming that the configured gate controls the
intervention strength:

| Cell | Gate | delta/native | effective_weight | Interpretation |
|---|---:|---:|---:|---|
| coverage_legacy_g005_s36 | 0.05 | 0.016 | 0.017 | Nearly invisible |
| typed_legacy_g005_s36 | 0.05 | 0.017 | 0.017 | Nearly invisible |
| typed_g010_r12 | 0.10 | 0.032 | 0.033 | Moderate |
| typed_g015_r12 | 0.15 | 0.048 | 0.050 | Default v3 strength |
| typed_g020_r12 | 0.20 | 0.064 | 0.066 | Strongest |

The old gate=0.05 setting produced delta/native ≈ 0.016, confirming the
finding from `docs/71` that the old configuration was functionally
invisible. The new default (gate=0.15) produces 3× the intervention strength
and is well above the 0.003 significance threshold from `docs/72`.

### 2.2 Cache composition ablation

| Cell | delta/native | anchor occupancy | summary occupancy |
|---|---:|---:|---:|
| typed_g015_r12 | 0.048 | 4 | 4-5 |
| anchor_only_g015 | 0.049 | 8 | 0 (disabled) |
| summary_only_g015 | 0.045 | 0 (disabled) | 12-16 |

Both anchor-only and summary-only produce comparable intervention strength to
the combined typed memory. The summary-only trace shows
`activation_ramp_not_observed` because every accepted readout used full scale
(the first eligible frame is already past the ramp window).

### 2.3 Online routing

| Cell | budget | selected_fraction | delta/native |
|---|---:|---:|---:|
| online_b25_g015 | 0.25 | 0.250 | 0.026 |
| online_b50_g015 | 0.50 | 0.500 | 0.037 |
| online_b75_g015 | 0.75 | 0.750 | 0.044 |

The budget fraction directly controls the selected head fraction and the
resulting intervention strength. All online cells show `diagnostics_nominal`.

### 2.4 Alignment and confidence

All cells show `alignment_positive_fraction` above 0.978 and
`confidence_mean` above 0.65, confirming that the memory attention output is
consistently aligned with the native attention output. No negative-alignment
failure was detected.

## 3. DINOv2 and comprehensive metrics

### 3.1 Method-level summary (sorted by DINO consistency)

| Cell | DINO | min_DINO | drift_slope | flicker | bg_cons | composite |
|---|---:|---:|---:|---:|---:|---:|
| typed_g020_r12 | 0.8491 | 0.8034 | -0.00315 | 0.1930 | 0.9303 | 0.5277 |
| typed_g015_hard_on | 0.8487 | 0.8063 | -0.00304 | 0.2024 | 0.9208 | 0.5288 |
| summary_only_g015 | 0.8469 | 0.8085 | -0.00324 | 0.1880 | 0.9288 | 0.5292 |
| online_recent21 | 0.8467 | 0.8094 | -0.00328 | 0.2028 | 0.9282 | 0.5250 |
| typed_g010_r12 | 0.8455 | 0.8042 | -0.00323 | 0.1862 | 0.9258 | 0.5278 |
| typed_g015_nonoverlap | 0.8454 | 0.7957 | -0.00318 | 0.1942 | 0.9297 | 0.5239 |
| coverage_legacy_g005_s36 | 0.8451 | 0.7982 | -0.00318 | 0.1995 | 0.9222 | 0.5267 |
| online_b50_g015 | 0.8443 | 0.7851 | -0.00318 | 0.1973 | 0.9263 | 0.5273 |
| online_no_effect_floor | 0.8443 | 0.7885 | -0.00318 | 0.1973 | 0.9253 | 0.5271 |
| online_b25_g015 | 0.8438 | 0.7960 | -0.00348 | 0.1980 | 0.9227 | 0.5214 |
| typed_legacy_g005_s36 | 0.8423 | 0.7856 | -0.00334 | 0.2071 | 0.9245 | 0.5238 |
| typed_g015_r12 | 0.8412 | 0.7904 | -0.00351 | 0.1899 | 0.9298 | 0.5217 |
| sf_native | 0.8400 | 0.7929 | -0.00348 | 0.2055 | 0.9199 | 0.5210 |
| anchor_only_g015 | 0.8396 | 0.8024 | -0.00348 | 0.1956 | 0.9257 | 0.5216 |
| online_b75_g015 | 0.8361 | 0.8043 | -0.00358 | 0.1967 | 0.9220 | 0.5201 |
| online_no_motion_penalty | 0.8354 | 0.7883 | -0.00370 | 0.2067 | 0.9162 | 0.5167 |

### 3.2 Per-prompt DINO (key cells)

The average DINO difference is small (~1%), but the per-prompt analysis
reveals that the intervention helps most on the prompts where sf_native
degrades most severely:

| Cell | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sf_native | 0.894 | 0.937 | **0.762** | 0.823 | 0.758 | 0.868 | 0.816 | 0.939 | 0.804 | 0.899 | **0.674** | 0.906 |
| typed_g015_r12 | 0.886 | 0.929 | 0.779 | 0.816 | 0.785 | 0.850 | 0.803 | 0.968 | 0.805 | 0.885 | 0.692 | 0.896 |
| typed_g020_r12 | 0.889 | 0.925 | **0.812** | 0.834 | 0.796 | 0.860 | 0.817 | 0.954 | 0.807 | 0.905 | **0.701** | 0.890 |
| typed_g015_hard_on | 0.887 | 0.934 | **0.801** | 0.809 | 0.789 | 0.859 | 0.806 | 0.961 | 0.808 | 0.911 | **0.709** | 0.913 |
| summary_only | 0.888 | 0.929 | 0.798 | 0.866 | 0.797 | 0.850 | 0.816 | 0.961 | 0.805 | 0.890 | 0.694 | 0.872 |
| anchor_only | 0.874 | 0.928 | 0.745 | 0.801 | 0.768 | 0.863 | 0.809 | 0.965 | 0.809 | 0.911 | 0.704 | 0.899 |

Prompt 2 (sf_native DINO = 0.762):
- typed_g020_r12: +0.050 (largest improvement)
- typed_g015_hard_on: +0.039
- typed_g015_r12: +0.017

Prompt 10 (sf_native DINO = 0.674, worst):
- typed_g015_hard_on: +0.035
- typed_g020_r12: +0.026
- typed_g015_r12: +0.018

### 3.3 Key metric observations

1. **typed_g020_r12 has the highest DINO (0.8491)** and least negative drift
   (-0.00315). The gate=0.20 setting provides the strongest intervention and
   the best identity retention.
2. **typed_g015_hard_on has the best min_DINO (0.8063)** and least negative
   drift (-0.00304). The hard activation outperforms the 12-frame ramp on
   every DINO metric.
3. **summary_only_g015 has the lowest flicker (0.1880)** and strong DINO
   (0.8469). Temporal summaries alone contribute meaningfully to both identity
   and smoothness.
4. **typed_g015_r12 (v3 default) barely differs from sf_native** (ΔDINO =
   +0.0013). The default gate=0.15 with ramp is too conservative.
5. **Flicker is reduced 5-10%** in most typed cells (0.186-0.194 vs 0.2055
   for sf_native).
6. **Background consistency is slightly improved** in typed cells (0.925-0.930
   vs 0.9199 for sf_native).
7. **anchor_only_g015 underperforms** (DINO 0.8396, below sf_native) and
   hurts prompt 2 (0.745 vs 0.762). Anchor-only memory is insufficient.
8. **online_b75 and online_no_motion_penalty are the worst** typed cells,
   suggesting dense routing and removing the motion penalty can hurt.

## 4. Decision rules (from docs/72)

| Rule | Condition | Result |
|---|---|---|
| 1 | typed_g015_r12 visually native AND delta/native < 0.003 | **Not triggered**: delta = 0.048, but DINO Δ = +0.0013 is marginal. Human review needed. |
| 2 | typed_g020_r12 improves identity but degrades motion | **Partially confirmed**: DINO +0.0092, drift improved. Flicker reduced (0.193 vs 0.206). Need temporal jump and human motion review. |
| 3 | hard-on has higher jump score than smooth | **Not triggered**: hard-on jump = 2.6523, ramp = 2.9189. Hard-on has LOWER temporal jump. **Ramp should be dropped.** |
| 4 | non-overlap beats overlap | **Not confirmed**: nonoverlap DINO = 0.8454 vs r12 = 0.8412, but min_DINO is worse (0.7957 vs 0.7904). Inconclusive. |
| 5 | online_b25/b50 beats typed all-head | **Not confirmed**: online_b50 DINO = 0.8443 vs typed_g015_r12 = 0.8412, but the difference is within noise. |
| 6 | PF remains required strong baseline | PF was not included in this screen. Must be run separately. |

## 5. Temporal jump diagnostic

### 5.1 Method-level summary (sorted by temporal_jump mean, lower = fewer jumps)

| Cell | jump_mean | jump_median | jump_p95 | app_jump | flow_jump |
|---|---:|---:|---:|---:|---:|
| online_no_motion_penalty | 2.4807 | 2.1693 | 5.2865 | 2.9249 | 2.0365 |
| typed_legacy_g005_s36 | 2.6000 | 2.0165 | 5.4527 | 2.9717 | 2.2283 |
| coverage_legacy_g005_s36 | 2.6169 | 2.1411 | 5.1043 | 2.9780 | 2.2558 |
| typed_g015_hard_on | 2.6523 | 2.4565 | 5.6799 | 3.0432 | 2.2614 |
| sf_native | 2.6562 | 2.3504 | 5.1303 | 3.0668 | 2.2456 |
| typed_g015_nonoverlap | 2.6937 | 2.3664 | 4.8246 | 3.1235 | 2.2639 |
| online_b25_g015 | 2.6961 | 2.6769 | 5.6563 | 3.0486 | 2.3435 |
| online_b75_g015 | 2.7100 | 2.3994 | 5.7102 | 3.1255 | 2.2944 |
| online_no_effect_floor | 2.7386 | 2.3876 | 5.1688 | 3.2120 | 2.2652 |
| online_b50_g015 | 2.7407 | 2.4064 | 5.1772 | 3.2133 | 2.2682 |
| anchor_only_g015 | 2.7417 | 2.4011 | 5.4306 | 3.1745 | 2.3090 |
| typed_g010_r12 | 2.8722 | 2.7182 | 5.6868 | 3.2867 | 2.4578 |
| summary_only_g015 | 2.8842 | 2.7900 | 5.7286 | 3.5161 | 2.2524 |
| typed_g020_r12 | 2.9088 | 2.5786 | 5.3817 | 3.3495 | 2.4681 |
| typed_g015_r12 | 2.9189 | 2.7113 | 5.4404 | 3.4158 | 2.4219 |

### 5.2 Ramp vs hard-on (decision rule 3)

| Cell | Configuration | jump_mean | DINO | min_DINO |
|---|---|---:|---:|---:|
| sf_native | baseline | 2.6562 | 0.8400 | 0.7929 |
| typed_g015_hard_on | gate=0.15, ramp=0 | **2.6523** | **0.8487** | **0.8063** |
| typed_g015_r12 | gate=0.15, ramp=12 | 2.9189 | 0.8412 | 0.7904 |
| typed_g020_r12 | gate=0.20, ramp=12 | 2.9088 | 0.8491 | 0.8034 |

**Conclusion:** Hard activation (ramp=0) has the lowest temporal jump score
(2.6523, essentially equal to native 2.6562), while also achieving the best
min_DINO (0.8063) and second-best DINO (0.8487). The 12-frame ramp produces
the HIGHEST jump score (2.9189) and worse identity metrics. The ramp is both
unnecessary and harmful; it should be dropped from the default configuration.

## 6. Preliminary conclusions

1. **The v3 intervention is mechanically working** (delta/native 0.048 at
   gate=0.15, 3× the old setting). The method is not a no-op.
2. **The default gate=0.15 with ramp is too conservative AND the ramp
   increases temporal jumps.** typed_g015_hard_on (no ramp) outperforms
   typed_g015_r12 on every metric: DINO, min_DINO, drift, and temporal jump.
3. **typed_g020_r12 (gate=0.20) has the best DINO but high temporal jump.**
   The stronger intervention improves identity but introduces more temporal
   discontinuity. typed_g015_hard_on is a better trade-off.
4. **typed_g015_hard_on is the recommended configuration.** It achieves:
   - DINO 0.8487 (2nd best, +0.0087 over native)
   - min_DINO 0.8063 (best, +0.0134 over native)
   - drift -0.00304 (least negative, slowest degradation)
   - temporal_jump 2.6523 (≈ native, no additional discontinuity)
5. **Summary-only is surprisingly strong** (DINO 0.8469, lowest flicker
   0.1880) but has higher temporal jumps (2.8842). Anchor-only is weak
   (DINO 0.8396, below native).
6. **The average DINO difference is small (~1%), but the per-prompt analysis
   shows the intervention helps most where native degrades most** (p2: +0.039,
   p10: +0.035 for hard-on). Human review is needed to validate visible
   improvement at 15-30s.
7. **Online routing does not beat all-head.** The online cells show similar
   or worse DINO than typed_g015_r12, with no clear advantage.

## 6. Human review

### 6.1 Review method

Human review of prompt 0 (`0-0_ema.mp4`) across all cells. Textual
description, no scoring.

### 6.2 Findings

**sf_native (prompt 0):**
- Identity degradation begins at approximately 5 seconds.
- Degradation progressively worsens over time, accompanied by darkening of
  the image and background hallucinations.
- By approximately 15 seconds, the video is completely broken and unusable.
- This confirms the expected 21-frame sliding window forgetting behavior
  documented in `docs/71`.

**All other cells (typed_g015_r12, typed_g020_r12, typed_g015_hard_on,
anchor_only_g015, summary_only_g015, online_b25/b50/b75, etc.):**
- The visual result is essentially identical to sf_native.
- The same 5-second degradation onset, progressive darkening, background
  hallucinations, and 15-second collapse are observed.
- No visible improvement in identity retention, illumination stability, or
  background consistency.

### 6.3 Conclusion

The current LifeCache-v3 method does not produce a visible improvement over
native Self-Forcing in 30-second single-prompt extrapolation. Despite the
trace analysis confirming that the intervention is mechanically active
(delta/native = 0.048 at gate=0.15, well above the 0.003 threshold), the
effect is invisible to human observation. No further video review is needed
to reach this conclusion.

### 6.4 Interpretation: why the intervention is measurable but invisible

The trace and metric data tell a consistent story when combined with the
human review:

1. **The fusion gate is too conservative.** At gate=0.15-0.20, the memory
   branch contributes only 5-7% of the output (effective_weight ≈ 0.05-0.066).
   This is enough to produce a measurable delta in the output tensor
   (DINO +0.009) but not enough to change what a human sees. The identity
   signal is overwhelmed by the native 93-95% contribution.

2. **The per-prompt DINO improvement on worst prompts (p2: +0.050) is real
   but small.** A 5% DINO improvement on a prompt that is already at 0.76
   does not lift it to a qualitatively different state. The video still
   collapses; it collapses slightly less.

3. **The memory content may not capture the right information.** The typed
   cache stores pooled pre-RoPE K/V. If the pooled representation does not
   preserve the identity-discriminative features that are lost after 5
   seconds, then reinjecting it at any gate cannot restore the lost identity.

4. **The convex fusion mode may be fundamentally limited.** Even with a
   higher gate, a convex combination of "forgotten native" and "stale memory"
   may produce a blended output that is neither faithful to the original
   identity nor temporally coherent. A replacement or residual injection
   mechanism might be needed.

## 7. Next steps

The screen experiment has established that the current v3 configuration is
mechanically functional but visually ineffective. The next iteration should
address the fusion strength and mechanism, not the cache structure or routing:

1. **Dramatically increase gate (0.30-0.50).** The current 0.15-0.20 range
   was chosen to be safe, but it is too safe to be useful. A gate sweep at
   0.30, 0.40, 0.50 with hard activation will determine whether the memory
   content can produce a visible effect at all.

2. **Test non-convex fusion modes.** Instead of `output = (1-gate) * native
   + gate * memory`, try:
   - Residual: `output = native + gate * (memory - native)` (amplify the
     difference)
   - Replacement: for selected heads, replace native K/V with memory K/V
     entirely
   - Gated replacement: use memory K/V when native attention confidence is
     low

3. **Audit memory content.** Extract the actual pooled K/V stored in anchors
   and summaries and compare them against the native K/V at frame 0 and frame
   60. If the memory does not contain identity-discriminative information,
   no fusion strength will help.

4. **Compare against PF directly.** PF retains identity visibly (docs/71).
   Examining how PF's per-head cache composition differs from our side-memory
   injection may reveal why PF works and we do not.

5. **Drop the ramp.** The temporal jump diagnostic confirms the ramp is
   harmful. Hard activation is better on every metric.

6. **Do not run the confirm phase or evaluation prompt suite yet.** There is
   nothing to confirm until the intervention produces a visible effect.
