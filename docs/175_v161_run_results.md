# 175: v161 State-Matched Motion Retrieval — Run Results

Date: 2026-08-04
Commit: `695b068` on `codex/v98-correctness-fixes`

## 1. Summary

v161 tests state-matched motion retrieval: instead of reading the freshest
motion pair (v160), read only pairs whose endpoint resembles the current
state and whose descriptor transition doesn't oppose the current transition.
Only 16 new videos generated; 80 reused from v160.

**Mechanism gate: TRUE.** The state-matching readout exercised genuine
choice: 445 multi-candidate reads, 273 non-newest selections, 135
negative-direction rejections, 11 abstentions. Zero atomic-pair violations.

**Safety screen: FLAGGED** (3 prompts: background_drift, subject_consistency_drop,
late_motion_collapse). Adaptive Wave 1 prepared (12 videos, 4 prompts × 3
methods) for human review.

## 2. Mechanism Trace

| Metric | Value |
|---|---:|
| Prompts | 16 |
| Multi-candidate reads | 445 |
| Selected non-newest compatible pair | 273 |
| State/direction abstentions | 11 |
| Negative-direction candidate rejections | 135 |
| Atomic-pair violations | 0 |
| Selected age p95 | 22.0 |

**Key insight:** 273 of 445 reads (61%) selected a non-newest pair — state
matching is actively choosing older but more compatible motion pairs over
fresher but incompatible ones. This is the core v161 hypothesis in action.

## 3. Safety Screen

**Result: FLAGGED** (not PASS)

### Flagged prompts
- Prompt 7: background_drift
- Prompt 11: subject_consistency_drop, background_drift
- Prompt 12: late_motion_collapse, temporal_discontinuity

### Wave 1 (12 videos for human review)
| Selection | Prompt | Tags |
|---|---|---|
| highest_automatic_risk | 12 | object_identity, tracking_camera, long_motion |
| largest_predicted_gain | 3 | multi_object, miniature_scale, water_motion |
| largest_metric_disagreement | 6 | vehicle, fast_motion, dust, tracking_camera |
| typical_case | 8 | human_motion, running, cinematic |

### Wave 2 (backup, 12 videos)
Prompts 11, 5, 13, 15 — only reviewed if Wave 1 is inconclusive.

## 4. Generation

- 16 new videos (1 method: middle10_reservoir2_statemotionpair1)
- 80 reused from v160 (5 methods)
- 6 methods × 16 prompts = 96 total
- Audit: ok=true, failures=[]
- Package: partially failed (automated_screen dir not yet created during package)

## 5. State-Matched Readout (v161 isolated change)

v161 tests exactly one change from v160:

```
v160: FreshCoherentMotionPair — bypasses quantile gate when pair is ≥12 frames stale
v161: StateMatchedMotionPair — adds state/direction compatibility check on readout
```

Readout steps:
1. Exclude pairs overlapping sink/recent or older than 24 frames
2. Require endpoint/current-state cosine ≥ -0.25
3. Reject descriptor transition with cosine < 0.0
4. Rank by transition similarity, state similarity, recency
5. Read both frames atomically, or abstain

## 6. Next Steps

1. **Human adaptive review (Wave 1)**: Review 12 anonymous videos at
   `runs/v161_state_matched_motion_moviebench16/full8/adaptive_review/wave1/reviewer/`
2. Run: `bash scripts/run_v161_automated_screen.sh analyze-wave1`
3. Decision branches (per doc 174 §7):
   - Mechanism gate fail → don't review (not applicable — gate passed)
   - Corruption → inspect selected pairs around failure
   - Improve motion naturalness + overall preference over v160 → freeze, held-out test
   - Indistinguishable/worse → reject descriptor-transition retrieval

## 7. Preserved Artifacts

```text
runs/v161_state_matched_motion_moviebench16/full8/
|-- videos/                    # 96 MP4 (16 new + 80 reused v160)
|-- published_manifest.json
|-- automated_screen/
|   |-- automated_screen.{json,md}
|   |-- state_motion_trace.{json,md}
|   |-- comprehensive.json
|   |-- comprehensive_parts/   # per-method results
|   |-- temporal_diagnostics.csv
|   `-- review_plan.json
`-- adaptive_review/wave1/reviewer/  # 12 videos for human review
```
