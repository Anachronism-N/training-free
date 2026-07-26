# v100 Fast Screen Human Review Results

Date: 2026-07-27

## 1. Review Summary

All 16 cells were generated on 4 nodes (4 cells per node, 1 GPU each, prompt 0,
30 seconds, 120 frames) and reviewed by a human.

### 1.1 Single-prompt cells (10 cells)

| Cell | Polygon noise | Other issues | Verdict |
|---|---|---|---|
| single_pf_native | No | Dual subject in later frames | ⚠️ Reference only |
| legacy_v98_stride_cyclic_sink1 | **Yes** | — | ❌ Fail |
| legacy_v98_stride_cyclic_sink3 | **Yes** | — | ❌ Fail |
| legacy_v98_stride_motion4 | **Yes** | — | ❌ Fail |
| legacy_v98_stride_motion2_cyclic2 | **Yes** | — | ❌ Fail |
| legacy_v98_stride_recent8 | **Yes** | — | ❌ Fail |
| legacy_v98_motion2_cyclic2_v78 | **Yes** | — | ❌ Fail |
| legacy_v98_hybrid_motion2_cyclic2 | **Yes** | — | ❌ Fail |
| legacy_v98_hybrid_motion2_cyclic2_v78 | **Yes** | — | ❌ Fail |
| legacy_v98_motion2_cyclic2_variance_refresh | **Yes** | — | ❌ Fail |

### 1.2 A-B-A cells (6 cells)

| Cell | Polygon noise | Other issues | Verdict |
|---|---|---|---|
| aba_pf_native | No | Scene changes but subject hallucinates: identity unchanged, sudden disappearance, dual subjects | ⚠️ PF ABA reference |
| aba_motion_no_episode | **Yes** | — | ❌ Fail |
| aba_motion_episode_bridge1 | **Yes** | — | ❌ Fail |
| aba_motion_episode_hard | **Yes** | — | ❌ Fail |
| aba_motion_episode_manual_bridge1 | **Yes** | — | ❌ Fail |
| aba_cyclic_sink3_episode_bridge1 | **Yes** | — | ❌ Fail |

## 2. Key Findings

### 2.1 All binary cells still have polygon noise

Every cell using the v98 history-polarity binary map (labels 10/11) produces
polygon noise, regardless of:

- **Responsive cache policy**: cyclic, motion, motion+cyclic, recent8 — all fail
- **Sink size**: sink1 and sink3 both fail
- **Tricks**: v78 writes, hybrid support, variance refresh — all fail
- **ABA mode**: all binary ABA cells fail

This is a stronger negative result than v99: the v99 smoke showed stride/cyclic
was clean, but the v100 fast screen with the **old v98 zero-threshold map**
(304/56 split, not the v99 middle-relative 33/327 split) produces noise in
every configuration.

### 2.2 The map is the problem, not the cache policy

v99 smoke used the v98 middle-relative scores (33/327 split) and stride/cyclic
was clean. v100 uses the old v98 zero-threshold map (304/56 split) and
stride/cyclic has noise. The difference is the head map:

| Map | Split | stride/cyclic result |
|---|---|---|
| v98 middle-relative (v99 smoke) | 33/327 | ✅ Clean |
| v98 zero-threshold (v100) | 304/56 | ❌ Polygon noise |

The 304/56 map assigns 133 of 156 PF Wave heads to the Supportive role
(stride), removing their native cyclic route. This is the same Wave-routing
issue identified in v99: Wave heads need cyclic, and assigning them to stride
causes artifacts.

### 2.3 PF native reference issues

- **single_pf_native**: dual subject in later frames (identity drift)
- **aba_pf_native**: scene changes occur but subject identity hallucinates
  (unchanged identity, sudden disappearance, dual subjects)

These are PF's own baseline issues, not caused by the binary cache.

## 3. Decision

### 3.1 Abandon the old v98 zero-threshold map

The 304/56 split is fundamentally broken for binary caching. 133 Wave heads
lose their cyclic route and produce polygon noise regardless of sink size,
motion events, or write gating.

### 3.2 The v99 middle-relative map (33/327) remains the only viable binary

The v99 smoke test showed that the 33/327 split with stride/cyclic produces
clean videos. This map:
- Assigns only 33 heads (mostly Anchor) to Supportive (stride)
- Assigns 327 heads (all Wave + Veil + 3 Anchor) to Responsive (cyclic)
- Preserves Wave's cyclic route for 156 heads
- Only 3 Anchor heads lose their stride route

### 3.3 Next steps

1. **Re-run v100 with the v99 middle-relative map (33/327)** instead of the
   old v98 zero-threshold map (304/56).
2. Keep stride/cyclic as the cache policy (merge is already abandoned from v99).
3. The motion-event mechanism is untestable under the broken map — re-test
   with the correct map.
4. ABA scene episodes need the correct map before evaluation.

### 3.4 What NOT to do

- Do not use the 304/56 map for any further experiment.
- Do not conclude that motion-event cache is broken — it was never tested
  under a correct map.
- Do not conclude that v78 writes or hybrid support are broken — same reason.
- Do not conclude that ABA episodes are broken — same reason.

## 4. Comparison with v99 smoke

| Aspect | v99 smoke (33/327 map) | v100 fast screen (304/56 map) |
|---|---|---|
| pf_ar stride/cyclic | ✅ Clean | Not tested |
| history_polarity stride/cyclic | ✅ Clean | ❌ Polygon noise |
| history_polarity stride/merge | ❌ Polygon noise | ❌ Polygon noise |
| pf_aw stride/merge | ❌ Polygon noise | Not tested |
| random stride/merge | ❌ Polygon noise | Not tested |
| Motion events | Not tested | ❌ Noise (but untestable) |
| ABA | Not tested | ❌ Noise (but untestable) |

## 5. Root Cause Analysis

The v98 zero-threshold map (`p_h >= 0`) is too aggressive: it assigns 304 of
360 heads to Supportive, including 133/156 Wave heads. Wave heads are
phase-local and require cyclic cache. Assigning them to stride removes their
periodic evidence and causes polygon noise.

The v99 middle-relative map (`middle_relative_logit_margin >= 0`) is more
conservative: only 33 heads are Supportive, and 169/172 Anchor heads are
correctly identified. All 156 Wave heads remain Responsive (cyclic), preserving
their native route.

**The correct binary map must keep Wave heads on the cyclic route.** The
33/327 split achieves this; the 304/56 split does not.

## 6. Infrastructure Notes

- v100 was run on 4 nodes (4 GPUs each, 16 cells total)
- Model loading took ~2 minutes per node
- Each video took ~3-4 minutes to generate (block 40/40 at ~3s/block)
- Total wall time: ~8 minutes from launch to all 16 videos complete
- The `ffprobe`/`ffmpeg` cv2 wrappers were required on all nodes
- A symlink fix was needed: `runs/v98_history_polarity/maps/` →
  `runs/v98_history_polarity_screen32/maps/`
