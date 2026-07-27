# v115 Role Memory Cache Screen: Human Review Results

Date: 2026-07-27

## 1. Generation Summary

16 cells were generated across 4 nodes (4 cells per node, 4 GPUs each, 1
video per cell, prompt 0, 30 seconds, 120 frames). All 16 videos completed.

Node 221 required a CUDA extension lock fix and retry (stale lock file at
`/root/.cache/torch_extensions/py310_cu128/pyramidkv_scatter_v30/lock`).

## 2. Full Review Table

### 2.1 Supportive sweep (Suppressive = recent8)

| Cell | Supportive | Suppressive | Verdict | Details |
|---|---|---|---|---|
| support_prototype4_suppress_recent8 | prototype4 | recent8 | ✅ Good | Minor mid-video ID enlargement |
| support_snapshot4_suppress_recent8 | snapshot4 | recent8 | ⚠️ Limited | Mid/late ID enlargement, possibly too large |
| support_retrieval2_suppress_recent8 | retrieval2+recent6 | recent8 | ⚠️ Weak | ID preserved but significant late enlargement, reduced motion |
| support_retrieval4_suppress_recent8 | retrieval4+recent4 | recent8 | ❌ Fail | ID enlargement + dual subject late |
| support_sparse75_suppress_recent8 | sparse75+recent5 | recent8 | ❌ Fail | Near-frozen mid-video, ID enlargement |

### 2.2 Suppressive sweep (Supportive = landmark4)

| Cell | Supportive | Suppressive | Verdict | Details |
|---|---|---|---|---|
| support_landmark4_suppress_prototype2 | landmark4 | prototype2+recent6 | ✅ Good | Minor mid-video ID enlargement |
| support_landmark4_suppress_snapshot2 | landmark4 | snapshot2+recent6 | ✅ Good | Late face enlargement trend |
| support_landmark4_suppress_retrieval2 | landmark4 | retrieval2+recent6 | ✅ Good | Late face enlargement trend |
| support_landmark4_suppress_motion_pair1 | landmark4 | motion_pair1+recent6 | ✅ Good | Minor mid-video ID enlargement |
| support_landmark4_suppress_sparse75 | landmark4 | sparse75+recent5 | ✅ Good | Minor mid-video ID enlargement |

### 2.3 Joint candidates

| Cell | Supportive | Suppressive | Verdict | Details |
|---|---|---|---|---|
| support_prototype4_suppress_motion_pair1 | prototype4 | motion_pair1 | ✅ Good | Late face enlargement trend |
| support_snapshot4_suppress_motion_pair1 | snapshot4 | motion_pair1 | ⚠️ Limited | Mid/late ID enlargement, possibly too large; ID preserved but effect limited |
| support_retrieval2_suppress_motion_pair1 | retrieval2 | motion_pair1 | ⚠️ Weak | ID preserved but significant late enlargement, reduced motion |
| support_sparse75_suppress_motion_pair1 | sparse75 | motion_pair1 | ❌ Fail | Near-frozen mid-video, ID enlargement |

### 2.4 Same-route controls

| Cell | Supportive | Suppressive | Verdict | Details |
|---|---|---|---|---|
| all_prototype4_control | prototype4 | prototype4 | ✅ Good | Minor mid-video ID enlargement |
| all_snapshot4_control | snapshot4 | snapshot4 | ⚠️ Partial | Late dual subject |

## 3. Key Findings

### 3.1 Prototype4 is the strongest new Supportive cache

`support_prototype4_suppress_recent8` and `support_prototype4_suppress_motion_pair1`
are both good. The temporal prototype mechanism (semantic segment medoid)
provides stable identity with only minor mid-video fluctuation. This is
comparable to `landmark4` from v111.

### 3.2 Landmark4 remains the most stable Supportive cache

All 5 cells with `support=landmark4` are good. The landmark4 mechanism
consistently produces usable videos regardless of the Suppressive cache choice.

### 3.3 Retrieval causes ID anchoring / enlargement

All retrieval cells show significant late ID enlargement:
- `retrieval2_suppress_recent8`: ID preserved but late enlargement, reduced motion
- `retrieval4_suppress_recent8`: ID enlargement + dual subject
- `support_landmark4_suppress_retrieval2`: late face enlargement
- `support_retrieval2_suppress_motion_pair1`: late enlargement, reduced motion

**Interpretation:** Retrieval anchors identity too strongly, causing the model
to over-emphasize the retrieved frame's identity features, leading to face/ID
enlargement in later frames. The strategy is working (ID is preserved) but
needs to be weakened — possibly by reducing retrieval read count, lowering the
MMR weight, or adding a recency bias.

### 3.4 Sparse75 causes motion freezing

Both sparse75 cells show near-frozen motion in the mid-video section:
- `support_sparse75_suppress_recent8`: frozen mid-video, ID enlargement
- `support_sparse75_suppress_motion_pair1`: frozen mid-video, ID enlargement

Token compression to 75% appears to damage temporal dynamics. The sparse
snapshot mechanism is too aggressive for video generation attention.

### 3.5 Snapshot4 is borderline

`support_snapshot4_suppress_recent8` and `support_snapshot4_suppress_motion_pair1`
show ID preservation but with mid/late enlargement that may be too large.
`all_snapshot4_control` shows late dual subject. The snapshot mechanism is
functional but parameter-sensitive — the admission threshold or bank size
may need adjustment.

### 3.6 Motion_pair1 as Suppressive is clean

`support_landmark4_suppress_motion_pair1` is good with only minor mid-video
ID enlargement. This is comparable to `suppress=recent8` and
`suppress=motion_pair2` from v111. The single motion pair + enlarged recent
window (recent6) provides adequate Suppressive cache.

### 3.7 Prototype2 as Suppressive is clean

`support_landmark4_suppress_prototype2` is good with only minor mid-video
ID enlargement. This is the first successful use of prototype memory as a
Suppressive cache.

## 4. Candidate Ranking

### Tier 1: Usable, minor issues

1. **support_landmark4_suppress_recent8** (v111, already reviewed) — best ID
2. **support_landmark4_suppress_motion_pair1** — good, minor mid fluctuation
3. **support_landmark4_suppress_prototype2** — good, minor mid fluctuation
4. **support_prototype4_suppress_recent8** — good, minor mid fluctuation
5. **support_landmark4_suppress_snapshot2** — good, late face trend
6. **support_landmark4_suppress_retrieval2** — good, late face trend
7. **support_prototype4_suppress_motion_pair1** — good, late face trend

### Tier 2: Functional but limited

8. **support_snapshot4_suppress_recent8** — ID preserved but enlargement
9. **support_snapshot4_suppress_motion_pair1** — ID preserved but enlargement
10. **all_prototype4_control** — good but same-route control
11. **all_snapshot4_control** — late dual subject

### Tier 3: Weak or failed

12. **support_retrieval2_suppress_recent8** — significant late enlargement
13. **support_retrieval2_suppress_motion_pair1** — significant late enlargement
14. **support_retrieval4_suppress_recent8** — dual subject
15. **support_sparse75_suppress_recent8** — frozen motion
16. **support_sparse75_suppress_motion_pair1** — frozen motion

## 5. Analysis of ID Enlargement Pattern

A consistent pattern across all cells is **mid-to-late ID enlargement**:
subject faces or bodies appear larger than they should. This occurs in:

- All retrieval cells (strongest)
- All sparse75 cells
- Snapshot cells (moderate)
- Prototype and landmark cells (mild)

This suggests the enlargement is partly a base generation property (also
present in PF native and SF native to varying degrees), amplified by:

1. **Retrieval**: over-anchoring to old identity frames
2. **Sparse compression**: losing spatial detail, causing the model to
   reconstruct identity at wrong scale
3. **Prototype/snapshot**: mild effect, possibly from medoid/snapshot
   selection choosing a high-identity frame

## 6. Bug Record

### 6.1 Node 221 CUDA extension stale lock

- **Issue:** `pyramidkv_scatter_v30/lock` stale file prevented CUDA extension
  loading on node 221.
- **Symptom:** All 4 cells stuck at "Loading CUDA extension" indefinitely.
- **Fix:** Kill processes, remove lock file, relaunch.
- **Status:** Fixed. All 4 cells completed after fix.

### 6.2 Done markers incomplete

- 8/16 cells have done markers. The other 8 (from node 221 retry) generated
  videos but may not have completed the full audit/trace validation due to
  the contract cleanup during retry.
- Videos are usable for review regardless.

## 7. Decision

### 7.1 Top candidates for 16-prompt screen

Based on visual quality and mechanism simplicity:

1. **support_landmark4_suppress_recent8** — simplest, best ID
2. **support_landmark4_suppress_motion_pair1** — adds motion evidence
3. **support_prototype4_suppress_recent8** — new mechanism, comparable quality

### 7.2 Rejected mechanisms

| Mechanism | Reason |
|---|---|
| retrieval (top-2 and top-4) | ID enlargement / dual subject |
| sparse75 | Motion freezing, ID enlargement |
| snapshot4 (standalone) | Borderline, parameter-sensitive |

### 7.3 Needs parameter tuning

- **Retrieval**: Reduce read strength (top-1 instead of top-2/4), add
  recency bias, lower MMR weight. The strategy works (ID preserved) but
  is too strong.
- **Snapshot4**: Adjust admission threshold or bank replacement policy.

### 7.4 Next steps

1. Push review results to GitHub for feedback.
2. Pull latest code (repo may contain next-step experiment scripts).
3. Consider 60s+ extrapolation test for top candidates.
4. Run 16-prompt or 32-prompt screen with top 3 candidates + controls.

## 8. Interpretation clarification

`Good` in the tables means that one reviewed prompt completed without a severe
artifact and is eligible for broader evaluation. It does **not** mean that the
Suppressive cache choices are visually identical or equally effective.

The reviewer observed differences among Suppressive Recent8, Motion-pair1/2,
Prototype2, Snapshot2, Retrieval2, and Sparse75, but those differences were
not stable enough to name or rank from one video. The post-v115 plan therefore
keeps Landmark4 fixed and evaluates all seven Suppressive routes on the same
MovieGenBench-16 prompts with paired metrics. See
`docs/117_post_v115_targeted_candidate_plan.md`.
