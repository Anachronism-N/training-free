# v111 Non-Periodic Role-Event Cache Screen: Review Results

Date: 2026-07-27

## 1. Generation Summary

8 cells were launched on node42 (8 GPUs, 1 video each, prompt 0, 30 seconds).
4 cells completed successfully; 4 cells crashed due to a motion_pair2 bug.

### 1.1 Successful cells (4/8)

| Cell | Supportive | Suppressive | Verdict |
|---|---|---|---|
| all_recent8_control | recent8 | recent8 | ⚠️ Functional but not usable as method |
| all_landmark4_control | landmark4 | landmark4 | ✅ Good |
| support_landmark4_suppress_recent8 | landmark4 | recent8 | ✅ Good |
| support_landmark2_motion1_suppress_recent8 | landmark2+motion1 | recent8 | ✅ Good |

### 1.2 Failed cells (4/8) — motion_pair2 bug

| Cell | Error |
|---|---|
| all_motion_pair2_control | `role_event.py:613: int(victim_end_t)` is NoneType |
| support_landmark4_suppress_motion_pair2 | same |
| support_landmark2_motion1_suppress_motion_pair2 | same |
| support_recent8_suppress_motion_pair2 | same |

**Pre-fix bug location:** `third_party/Pyramid-Forcing/pyramidkv/role_event.py`,
line 613.

**Root cause after code review:** `pair_capacity=2` has a valid intermediate
state in which one pair is stored and the bank is still filling. No eviction
victim should exist in that state, so `victim_end_t=None` is intentional. The
spacing check nevertheless cast `victim_end_t` to `int` for every existing
pair. The crash therefore occurs while filling slot 2, not because a stored
pair is missing its `end_t`.

**Fix:** When filling, check spacing against every stored pair. When replacing
a full bank, check spacing against every pair except the selected victim. The
implementation now also validates that each bank key equals the record's
`end_t`, validates pair adjacency, and records the retained end times and
spacing distances in the policy trace.

## 2. Human Review

### 2.1 all_landmark4_control — ✅ Good

Landmark4 memory for all heads (role-neutral control). Subject identity is
stable with minor degradation in the middle portion, but recovers in later
frames. This is the strongest role-neutral control.

### 2.2 all_recent8_control — ⚠️ Functional but not a method

Recent8 for all heads. The temporal trend resembles SF native but is slightly
better. This is expected: recent8 is essentially SF native with a slightly
longer recent window. It is not a viable method contribution because it adds
no role-conditioned structure — it simply increases the recent budget.

### 2.3 support_landmark4_suppress_recent8 — ✅ Good

Supportive heads use landmark4; Suppressive heads use recent8. Subject
identity shows some enlargement mid-video but recovers. The role-conditioned
split produces a usable result.

### 2.4 support_landmark2_motion1_suppress_recent8 — ✅ Good

Supportive heads use landmark2+motion1 (at most 4 middle frames: two landmarks
plus both endpoints of one motion pair); Suppressive heads use recent8.
Subject identity enlarges mid-video but recovers later. Visually comparable
to landmark4_suppress_recent8.

## 3. Key Observations

1. **Landmark memory is viable on prompt 0**: The role-neutral Landmark4 and
   both Landmark-based candidates completed without polygon noise. One prompt
   establishes implementation viability, not a general quality gain.

2. **A role-conditioned gain is not established yet**: The heterogeneous
   candidates and the all-Landmark control are all visually usable. Without a
   blind ranking across the complete matrix or multi-prompt metrics, this
   result cannot show that the 304/56 routing adds value over a uniform cache.

3. **Mid-video identity fluctuation needs trace correlation**: Both candidates
   show identity enlargement in the middle portion. Landmark replacement is
   one hypothesis, but self-correction or landmark causality cannot be claimed
   until the selected-frame trace is aligned with the failure timestamp.

4. **Motion-pair is only partially tested**: The one-pair component inside
   `landmark2+motion1` completed. Every two-pair route crashed while filling
   its second slot, so isolated Motion-pair quality and Suppressive Motion
   routing remain unknown.

5. **No polygon noise was observed in the four completed prompt-0 videos**:
   This is evidence that Landmark/Recent routes can execute cleanly, but it
   does not yet establish that the strategy generally avoids such artifacts.

## 4. Bug Record

### 4.1 motion_pair2 NoneType crash

- **File:** `third_party/Pyramid-Forcing/pyramidkv/role_event.py`
- **Pre-fix line:** 613
- **Error:** `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`
- **Trigger:** A `CoherentMotionStrategy(pair_capacity=2)` update when the bank
  already contains one pair and attempts to evaluate a second pair.
- **Affected cells:** 4 of 8 v111 cells.
- **Unaffected evidence:** The hybrid `pair_capacity=1` route completed, which
  is consistent with the exact trigger above.
- **Fix:** Separate filling from replacement spacing checks; add bank
  invariants, decision telemetry, and a regression test covering both the
  second fill and a later full-bank replacement.
- **Status:** Code-fixed in the current branch; GPU-server rerun pending.

## 5. Decision

### 5.1 Provisional viable candidates

Two candidates are visually viable:

1. `support_landmark4_suppress_recent8` — landmark4 + recent8
2. `support_landmark2_motion1_suppress_recent8` — landmark2+motion1 + recent8

Both show minor mid-video identity fluctuation but recover. Neither has
polygon noise.

### 5.2 Controls and pending cells

- `all_recent8_control`: retain as a budget-matched control, not a contribution.
- `all_motion_pair2_control` and all Motion-pair2 cells: not rejected on
  quality; they require the corrected targeted rerun.

### 5.3 Next steps

1. **Run the corrected four-cell Motion-pair2 subset** using the dedicated
   `motion_pair2` mode and a fresh output directory.
2. **Blind-review all eight cells together**. Do not compare only the four new
   outputs in isolation, and do not infer a role gain merely from viability.
3. **Inspect policy traces** for fill/replacement counts, selected pairs,
   spacing checks, and the frames around the observed mid-video enlargement.
4. **32-prompt screen**: If a heterogeneous candidate beats or matches the
   strongest role-neutral control, run v112 (32 prompts × 4 methods).
5. **VBench-Long + DINO**: Compute only after the 32-prompt generation and
   audit pass. A 60- or 90-second stress test follows candidate selection.

## 6. Comparison with Previous Rounds

| Round | Map | Cache | Polygon noise | Identity |
|---|---|---|---|---|
| v97 (304/56, stride/merge) | 304/56 | stride+merge | Yes | Severe drift |
| v98 (304/56, hybrid/merge) | 304/56 | hybrid+merge | Yes | Severe drift |
| v99 smoke (33/327, stride/cyclic) | 33/327 | stride+cyclic | No | Good |
| v100 (304/56, stride/cyclic) | 304/56 | stride+cyclic | Yes | Severe drift |
| v107 (33/327, stride/cyclic) | 33/327 | stride+cyclic | No | Good |
| v109 (304/56, all-cyclic) | 304/56 | cyclic+cyclic | No | Decreasing motion |
| **v111 (304/56, landmark+recent)** | **304/56** | **landmark+recent** | **No** | **Good with mid fluctuation** |

v111 is the first experiment using the 304/56 map without polygon noise and
without relying on stride/cyclic. The landmark memory is a new primitive that
avoids the Wave-to-stride routing problem entirely.
