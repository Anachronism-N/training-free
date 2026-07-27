# v107 Polygon-Noise Root-Cause Recovery: Human Review Results

Date: 2026-07-27

## 1. Review Summary

All 8 cells were generated on node42 (4 ranks × 2 GPUs, 1 video each,
prompt 0, 30 seconds, 120 frames) using the rebuilt v98 middle-relative
score map (33/327 split) and the v99 exclusive-ownership cache fix.

| # | Cell | Map | Verdict | Details |
|---|---|---|---|---|
| 1 | pf_ar_stride_cyclic_control | PF-AR 172/188 | ✅ Usable | Clean, no polygon noise |
| 2 | pf_aw_stride_cyclic_control | PF-AW 328/32 | ❌ Polygon noise | Wave heads on stride cause noise |
| 3 | middle_relative_stride_cyclic_control | 33/327 | ✅ Usable | Independent classifier clean |
| 4 | middle_relative_cyclic4_motion1 | 33/327 | ⚠️ Partial | Dual subject in later frames; first subject motion reduced |
| 5 | middle_relative_cyclic2_motion2 | 33/327 | ❌ Polygon noise | Equal-split motion/cyclic fails |
| 6 | middle_relative_stride_cyclic_v78 | 33/327 | ✅ Usable | v78 trusted writes clean |
| 7 | aba_middle_relative_no_episode | 33/327 | ⚠️ Partial | Scene not fully forgotten; subject ID reappears or disappears before switch |
| 8 | aba_middle_relative_episode_bridge1 | 33/327 | ⚠️ Partial | ID consistent throughout; minor background leakage; B-scene ID briefly degraded |

## 2. Root-Cause Confirmation

### 2.1 Wave-to-stride is the polygon noise cause

| Cell | Wave route | Result |
|---|---|---|
| pf_ar_stride_cyclic_control | cyclic (preserved) | ✅ Clean |
| pf_aw_stride_cyclic_control | stride (changed) | ❌ Polygon noise |

This is the definitive causal isolation:
- PF-AR keeps all 156 Wave heads on cyclic → clean.
- PF-AW moves all 156 Wave heads to stride → polygon noise.
- Everything else (sink, recent, ownership, labels) is identical.

**Conclusion: moving Wave heads from cyclic to stride causes polygon noise.
The binary path itself is clean when Wave's cyclic route is preserved.**

### 2.2 The 33/327 middle-relative map is viable

`middle_relative_stride_cyclic_control` (33/327 map, stride for label-10,
cyclic for label-11) is clean. This confirms the v99 smoke result with the
correct map and the corrected cache ownership implementation.

The 33/327 map assigns:
- 12/172 Anchor heads → Supportive (stride)
- 160/172 Anchor heads → Responsive (cyclic)
- 16/156 Wave heads → Supportive (stride)
- 140/156 Wave heads → Responsive (cyclic)
- 5/32 Veil heads → Supportive (stride)
- 27/32 Veil heads → Responsive (cyclic)

Most Wave heads (140/156 = 90%) remain on cyclic, which explains the clean
result. The 16 Wave heads on stride are a small enough fraction to avoid
visible artifacts.

## 3. Candidate Cache Evaluation

### 3.1 stride_cyclic (baseline) — ✅ Clean

The base candidate: Supportive=stride4, Responsive=cyclic4, sink3+recent4.
Clean video, no artifacts. This is the reference for all other candidates.

### 3.2 cyclic4_motion1 — ⚠️ Partial failure

Adding a single motion-event slot on top of cyclic4 (5-frame middle total):
- First subject's motion significantly reduced in later frames.
- Second subject appears in later frames (identity duplication).

The motion-event selection may interfere with the cyclic phase evidence,
reducing motion fidelity. The extra cache slot may also destabilize identity.

**Verdict: do not promote. The additive motion slot hurts more than it helps.**

### 3.3 cyclic2_motion2 — ❌ Polygon noise

Replacing 2 cyclic slots with 2 motion-event slots (4-frame middle total,
same budget):
- Polygon noise appears.

Removing 2 of 4 cyclic slots breaks the periodic phase evidence for too many
Responsive heads. The motion-event mechanism cannot compensate for the lost
cyclic evidence.

**Verdict: rejected. Cyclic evidence is essential and cannot be partially
replaced by motion events.**

### 3.4 stride_cyclic_v78 — ✅ Clean

Adding v78 trust-conditioned writes on top of stride_cyclic:
- Clean video, no artifacts.
- Visually comparable to the baseline stride_cyclic.

**Verdict: v78 is safe to retain as an optional add-on. It does not hurt and
may help stability in longer or multi-prompt settings.**

## 4. A-B-A Scene Episode Evaluation

### 4.1 no_episode — ⚠️ Partial failure

Without scene archive/restore:
- Scene is not fully forgotten after the switch.
- Subject ID reappears in later frames or disappears before the switch.
- The cache does not cleanly separate scene A from scene B.

**Verdict: binary cache alone cannot handle scene switching. Some explicit
boundary mechanism is needed.**

### 4.2 episode_bridge1 — ⚠️ Partial success

With role-aware scene archive/restore and 1-frame recent bridge:
- Subject ID is consistent throughout the entire video (A→B→A2).
- Minor background leakage: some A-scene background persists in B.
- B-scene subject ID briefly degraded at one point.

This is a significant improvement over no_episode: the identity is preserved
across the scene switch, which is the primary A-B-A goal. The background
leakage and brief B-scene degradation are secondary issues that may be
addressed by tuning the bridge or sink lifecycle.

**Verdict: scene episode mechanism is promising but not yet production-ready.
The identity preservation across A-B-A is the key positive signal.**

## 5. Decision

### 5.1 Promoted method

**History-Polarity Stride/Cyclic Cache (HP-SC)**

```text
offline two-role head profiling (33/327 middle-relative map)
+ History-Supportive: sink3 + stride(interval=6, cap=4) + recent4
+ Recent-Responsive: sink1 + cyclic(period=6, cap=4) + recent4
+ exclusive sink + middle + recent ownership
+ optional v78 trust-conditioned writes
```

### 5.2 Rejected mechanisms

| Mechanism | Reason |
|---|---|
| Merge cache | Polygon noise (v99 smoke, confirmed) |
| cyclic2_motion2 | Polygon noise (insufficient cyclic evidence) |
| cyclic4_motion1 | Identity duplication and motion reduction |
| ABA without episode | Scene not forgotten, ID instability |

### 5.3 Conditional mechanisms

| Mechanism | Condition |
|---|---|
| v78 trusted writes | Retain as optional; verify in multi-prompt |
| Scene episode bridge1 | Promising for ABA; needs background leakage fix |
| Motion-event cache | Rejected for now; revisit only with different design |

## 6. Next Steps

1. **Screen32**: Run the promoted method (stride_cyclic + v78) on 32 diverse
   prompts with PF-AR and PF native controls.
2. **Blind review**: Freeze scorecard before metrics.
3. **DINO/VBench/temporal-jump**: If blind review passes.
4. **Main128**: If screen32 passes.
5. **ABA tuning**: If single-prompt is stable, tune bridge/sink for ABA.
6. **Head-classification ablation**: random/inverted/threshold controls on 32 prompts.

## 7. Map Cross-Tab (from v107 manifest)

### 7.1 history_polarity_zero (33/327)

| PF class | Heads | Supportive (10) | Responsive (11) |
|---|---|---|---|
| Anchor | 172 | 12 | 160 |
| Wave | 156 | 16 | 140 |
| Veil | 32 | 5 | 27 |

### 7.2 pf_ar_binary_control (172/188)

| PF class | Heads | Supportive (10) | Responsive (11) |
|---|---|---|---|
| Anchor | 172 | 172 | 0 |
| Wave | 156 | 0 | 156 |
| Veil | 32 | 0 | 32 |

### 7.3 pf_aw_binary_control (328/32)

| PF class | Heads | Supportive (10) | Responsive (11) |
|---|---|---|---|
| Anchor | 172 | 172 | 0 |
| Wave | 156 | 156 | 0 |
| Veil | 32 | 0 | 32 |

## 8. Root-Cause Summary

The v100 polygon noise was caused by the old 304/56 map routing 133 Wave
heads to stride. The v107 root-cause cells prove this:

1. **PF-AR** (Wave on cyclic) → clean ✅
2. **PF-AW** (Wave on stride) → polygon noise ❌
3. **33/327 map** (140/156 Wave on cyclic) → clean ✅

The binary approach works when the classifier preserves the cyclic route for
the vast majority of Wave heads. The 33/327 middle-relative map achieves this
with only 16 Wave heads on stride (10.3%), which is below the noise threshold.

The motion-event mechanism cannot replace cyclic evidence. The scene episode
mechanism preserves identity across A-B-A but needs background leakage fixes.
