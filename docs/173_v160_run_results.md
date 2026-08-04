# 173: v160 Fresh-Motion Recovery — Run Results

Date: 2026-08-04
Commit: `2d2356b` on `codex/v98-correctness-fixes`

## 1. Summary

v160 tests a single isolated change: a freshness-aware recovery rule for the
CoherentMotionPair. When a retained pair is ≥12 frames older than an eligible
candidate, the candidate can bypass the motion-quantile gate. Only 16 new
videos are generated; 64 are reused from v159.

**All stages succeeded**: generation (ok=true), audit, package, and automated
diagnostic screen (safety=PASS). The freshness mechanism gate passed — the
intended stale-refresh bypass is active and dramatically reduces pair age.

The automated screen flagged 2 prompts for human review and prepared a 12-video
adaptive Wave 1. Human review is the next step.

## 2. Fresh-Motion Trace (Mechanism Verification)

| Metric | v159 (stale rule) | v160 (fresh rule) | Change |
|---|---:|---:|---|
| Accepted updates / prompt | 6.125 | 12.562 | **+105%** |
| Pair-age p95 / prompt | 34.49 | 11.006 | **-68%** |
| Maximum pair age | 61 | 13 | **-79%** |
| Below-quantile stale bypasses | 0 | 83 | new |
| 12-23-frame freshness refreshes | 0 | 95 | new |

**Mechanism gate: TRUE.** The freshness-aware recovery rule more than doubled
the motion pair update rate and reduced maximum pair age from 61 to 13 frames.
This directly addresses the v159 failure: old stale pairs that couldn't be
replaced are now refreshed when sufficiently stale.

## 3. Automated Diagnostic Screen

**Safety screen: PASS**

### Flagged prompts
- Prompt 12: subject_consistency_drop
- Prompt 13: late_motion_collapse

### Adaptive review Wave 1 (12 videos)
| Selection criterion | Prompt | Tags |
|---|---|---|
| highest_automatic_risk | 12 | object_identity, tracking_camera, long_motion |
| largest_predicted_gain | 7 | articulated_motion, festival, crowd, colorful |
| largest_metric_disagreement | 1 | multi_subject, animal, snow, camera_depth |
| typical_case | 10 | child_identity, bicycle, season_change |

Wave 2 (another 12 videos) is prepared but only reviewed if Wave 1 is
inconclusive. Normal review load: 12 videos; maximum: 24.

## 4. Generation

- 16 new videos (1 new method: middle10_reservoir2_freshmotionpair1)
- 64 reused from v159 (SF, middle10_reservoir2_motionpair1, middle10_reservoir4, all_recent8)
- 5 methods × 16 prompts = 80 total
- Audit: ok=true, failures=[]
- Package: tarball created

## 5. Isolated Change

v160 tests exactly one change from v159:

```
v159: CoherentMotionPair with max_pair_age=24 (only relaxes replacement gate)
v160: FreshCoherentMotionPair with stale_quantile_bypass (bypasses quantile gate
      when pair is ≥12 frames older than candidate)
```

All other components (reservoir2, sink1, recent4, layer selection, motion
threshold, semantic gate, spacing) are identical to v159.

## 6. Next Steps

1. **Human adaptive review (Wave 1)**: Review 12 anonymous videos at
   `runs/v160_fresh_motion_moviebench16/full8/adaptive_review/wave1/reviewer/`.
   Score motion amount and motion naturalness separately.

2. **Run analyzer**: `bash scripts/run_v160_automated_screen.sh analyze-wave1`

3. **Decision branches** (per doc 172 §7):
   - Exploratory pass → held-out validation with fixed protocol
   - Human rejection → stop freshness route
   - Corruption flag → investigate implementation failure first

## 7. Preserved Artifacts

```text
runs/v160_fresh_motion_moviebench16/full8/
|-- videos/                    # 80 MP4 (16 new + 64 reused v159)
|-- published_manifest.json
|-- contracts/
|-- automated_screen/
|   |-- automated_screen.{json,md}
|   |-- fresh_motion_trace.{json,md}
|   |-- comprehensive.json
|   |-- comprehensive_parts/
|   `-- review_plan.json
`-- v160_diagnostics.tar.gz
```
