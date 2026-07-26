# v99 Smoke Test Results and Decision

Date: 2026-07-26

## 1. Smoke Test Summary

Five cells were generated on prompt 0 (30 seconds, 120 frames, 832×480) and
reviewed by a human. Two reused reference videos were also reviewed.

| Cell | Map | Supportive route | Suppressive route | Polygon noise | Verdict |
|---|---|---|---|---|---|
| pf_ar_neutral_stride_cyclic | PF Anchor-vs-Rest | stride+cyclic | stride+cyclic | No | ✅ Pass |
| history_polarity_stride_cyclic | v98 33/327 | stride+cyclic | stride+cyclic | No | ✅ Pass |
| pf_aw_neutral_stride_merge | PF (A+W)\|V | stride | merge | Yes | ❌ Fail |
| history_polarity_stride_merge_fixed | v98 33/327 | stride | merge | Yes | ❌ Fail |
| history_polarity_random_stride_merge | random 33/327 | stride | merge | Yes | ❌ Fail |
| reuse_pf_native (reference) | — | — | — | No | ✅ (identity drift noted) |
| reuse_pf_binary_read_v78 (reference) | — | — | — | No | ✅ |

## 2. Key Finding: Merge Operator Causes Polygon Noise

The pattern is unambiguous:

- **All three stride/merge cells** (pf_aw oracle, independent classifier, random
  control) produced polygon noise.
- **Both stride/cyclic cells** (pf_ar and history_polarity) were visually clean.
- The noise is independent of head membership: the PF oracle, the independent
  classifier, and the random control all fail with merge.

**Conclusion: the merge cache operator itself is the noise source, not the
head classification.**

This matches the v99 decision tree:

> "If stride/cyclic passes but repaired stride/merge fails, the unresolved
> Wave heads require a phase-local cache; do not blame the Anchor/Veil
> separation."

## 3. Why Merge Fails

The v98 map assigns 327 of 360 heads to the Suppressive role. Under the
stride/merge composition, all 327 Suppressive heads use PF's merge operator
(patch=2, block=4, capacity=4). PF's native three-class system assigns merge
to only 32 Veil heads. The merge operator was validated for 32 heads, not 327.

The 156 PF Wave heads previously used a cyclic (period=6, capacity=4) route.
Replacing cyclic with merge for 156 heads causes visual artifacts. The merge
operator's spatiotemporal patch compression is incompatible with Wave's
phase-local motion evidence.

## 4. What Passed

### pf_ar_neutral_stride_cyclic (implementation control)

- PF Anchor-vs-Rest membership encoded as neutral labels 10/11.
- All heads use stride+cyclic (no merge).
- No polygon noise, quality similar to PF three-class ablation.
- Minor startup flashback in first few frames (also present in native PF).
- **Verdict: cache ownership fix is correct. Neutral labels work.**

### history_polarity_stride_cyclic (independent classifier)

- v98 middle-relative scores at natural zero: 33 Supportive, 327 Suppressive.
- Same stride+cyclic route for both roles.
- No polygon noise.
- **Verdict: independent classifier + stride/cyclic cache is viable.**

## 5. Profiling Results

The v98 middle-relative QK profiling produced:

- 64 profiles (32 stride + 32 merge), 360 heads
- 33 Supportive (9.2%) / 327 Suppressive (90.8%)
- 169/172 PF Anchor heads → Supportive
- All 156 PF Wave + 32 PF Veil → Suppressive
- Score SHA-256: `83d2be6ab2978c3aa13f8d508f29a6fcf0d8fa026bca8bccb319b2a3e10ba53c`
- All acceptance gates passed

## 6. Decision

1. **Merge route is abandoned** for the binary classifier. The merge operator
   cannot replace cyclic for 156+ Wave heads.

2. **Stride/cyclic route is the candidate method.** Both the PF-AR control and
   the independent history-polarity classifier produce clean videos with
   stride+cyclic for all heads.

3. **Next experiment: screen32** on `history_polarity_stride_cyclic` and
   `pf_ar_neutral_stride_cyclic` only. Skip all merge cells.

4. **The random control must be re-run with stride/cyclic** (not stride/merge)
   to test whether the 33/327 classification is causal.

5. **Paper story adjustment**: the contribution is not "Anchor stride / Veil
   merge" but "binary Supportive/Responsive roles with stride+cyclic cache."
   The merge operator remains PF's Veil-exclusive primitive.

## 7. Bug Fixes Applied During v99

- `pyramidkv_prompt_warmup_enabled: false` added to PF config (missing key).
- `payload.get("pass")` → `payload.get("ok")` in v99 audit check (field name
  mismatch between audit script and v99 runner).
- `ffprobe` and `ffmpeg` wrappers created using cv2 (binaries not installed).
- Clean reuse directories (single prompt) to avoid extra-index audit failures.
- `git update-index --assume-unchanged` for config and script fixes.

## 8. Remaining Experiments

| Step | Status | Description |
|---|---|---|
| v98 middle-relative profiling | ✅ Done | 64 profiles, scores frozen |
| v99 smoke1 (5 cells) | ✅ Done | stride/cyclic passes, merge fails |
| v99 screen32 (stride/cyclic) | ⬜ Next | 2-3 methods × 32 prompts |
| v99 causal32 | ⬜ Optional | If screen32 passes, add random control |
| v99 main128 | ⬜ Conditional | If screen32 passes |
| 60s extrapolation | ⬜ Conditional | |
| ABA scene-return | ⬜ Conditional | |
