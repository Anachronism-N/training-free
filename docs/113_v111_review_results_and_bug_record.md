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

**Bug location:** `third_party/Pyramid-Forcing/pyramidkv/role_event.py`, line 613.

**Root cause:** The motion-pair eviction logic calls `int(victim_end_t)` where
`victim_end_t` is `None`. This occurs when the eviction candidate search finds
no victim with a valid `end_t` field. The code does not guard against `None`
before calling `int()`.

**Fix needed:** Add a `None` check before the `int()` call, or ensure
`end_t` is always set when a motion-pair entry is created.

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

Supportive heads use landmark2+motion1 (3-frame middle); Suppressive heads
use recent8. Subject identity enlarges mid-video but recovers later. Visually
comparable to landmark4_suppress_recent8.

## 3. Key Observations

1. **Landmark memory works**: The semantic landmark selection produces stable
   identity without stride, cyclic, or merge. This is a new cache primitive
   not borrowed from PF.

2. **Role-conditioned split is visible**: The candidate cells
   (landmark4+recent8, landmark2+motion1+recent8) are usable, while the
   all-recent8 control is explicitly not a method. The role split adds value
   beyond a uniform cache.

3. **Mid-video identity fluctuation**: Both candidate cells show identity
   enlargement in the middle portion. This may be caused by the landmark
   memory selecting a frame that temporarily destabilizes identity. The
   recovery in later frames suggests the landmark mechanism self-corrects.

4. **Motion-pair is untested**: The motion_pair2 strategy crashed before
   generating any video. Its quality is unknown.

5. **No polygon noise**: None of the 4 successful cells show polygon noise.
   The landmark+recent approach avoids the stride/cyclic/merge artifact
   entirely.

## 4. Bug Record

### 4.1 motion_pair2 NoneType crash

- **File:** `third_party/Pyramid-Forcing/pyramidkv/role_event.py`
- **Line:** 613
- **Error:** `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`
- **Trigger:** Any cell using `suppress_policy=motion_pair2` or
  `support_policy` containing motion_pair2.
- **Affected cells:** 4 of 8 v111 cells.
- **Fix:** Guard `victim_end_t` against `None` before `int()` conversion, or
  ensure `end_t` is always populated when creating motion-pair entries.
- **Status:** Unfixed. Needs code change before re-running motion_pair2 cells.

## 5. Decision

### 5.1 Promoted candidates

Two candidates are visually viable:

1. `support_landmark4_suppress_recent8` — landmark4 + recent8
2. `support_landmark2_motion1_suppress_recent8` — landmark2+motion1 + recent8

Both show minor mid-video identity fluctuation but recover. Neither has
polygon noise.

### 5.2 Rejected

- `all_recent8_control`: not a method (just longer recent window)
- `all_motion_pair2_control` and all motion_pair2 cells: crashed, untested

### 5.3 Next steps

1. **Fix motion_pair2 bug** and re-run the 4 failed cells.
2. **Longer extrapolation test**: The current 30-second videos show mid-video
   identity fluctuation that recovers. A 60-second or 90-second test would
   determine whether the recovery is stable or whether identity degrades
   further over longer horizons.
3. **32-prompt screen**: If motion_pair2 is fixed and at least one candidate
   remains clean, run v112 (32 prompts × 4 methods).
4. **VBench-Long + DINO**: After 32-prompt screen passes blind review.

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
