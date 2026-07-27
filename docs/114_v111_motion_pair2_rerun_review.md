# v111 Motion-Pair2 Re-Run Review Results

Date: 2026-07-27

## 1. Review Summary

The 4 motion_pair2 cells were re-generated after the `role_event.py:613`
NoneType bug fix (commit `2aacbc2`). All 4 completed successfully.

| # | Cell | Supportive | Suppressive | Verdict | Details |
|---|---|---|---|---|---|
| 5 | all_motion_pair2_control | motion_pair2 | motion_pair2 | ❌ Fail | Dual subject in later frames; first subject motion decreases; ID retention worse than landmark cells |
| 6 | support_landmark4_suppress_motion_pair2 | landmark4 | motion_pair2 | ✅ Good | Clean; ID retention good |
| 7 | support_landmark2_motion1_suppress_motion_pair2 | landmark2+motion1 | motion_pair2 | ✅ Good | Clean; ID retention slightly worse than landmark4 |
| 8 | support_recent8_suppress_motion_pair2 | recent8 | motion_pair2 | ⚠️ Not a method | Similar trend to SF native but slightly better; not a viable contribution |

## 2. Combined v111 Results (All 8 Cells)

| Cell | Supportive | Suppressive | Noise | ID | Motion | Verdict |
|---|---|---|---|---|---|---|
| all_recent8_control | recent8 | recent8 | No | SF-like | SF-like | ⚠️ Not a method |
| all_landmark4_control | landmark4 | landmark4 | No | Good (mid dip, recovers) | Normal | ✅ |
| all_motion_pair2_control | motion_pair2 | motion_pair2 | No | Dual subject late | Decreasing | ❌ |
| support_landmark4_suppress_recent8 | landmark4 | recent8 | No | Good (mid fluctuation) | Normal | ✅ |
| support_landmark2_motion1_suppress_recent8 | landmark2+motion1 | recent8 | No | Good (mid fluctuation) | Normal | ✅ |
| support_landmark4_suppress_motion_pair2 | landmark4 | motion_pair2 | No | Good | Normal | ✅ |
| support_landmark2_motion1_suppress_motion_pair2 | landmark2+motion1 | motion_pair2 | No | Good (slightly worse than landmark4) | Normal | ✅ |
| support_recent8_suppress_motion_pair2 | recent8 | motion_pair2 | No | SF-like | SF-like | ⚠️ Not a method |

## 3. Key Findings

### 3.1 Landmark memory is the strongest Supportive cache

All cells using `landmark4` as the Supportive cache produce good results
regardless of the Suppressive cache (recent8 or motion_pair2). The landmark
selection mechanism provides stable long-range identity without stride, cyclic,
or merge.

### 3.2 Motion-pair2 alone is insufficient

`all_motion_pair2_control` shows dual subjects and decreasing motion. The
motion-pair mechanism cannot serve as the sole middle memory. It works only
when paired with landmark support.

### 3.3 Recent8 as Supportive is not a method

Both `all_recent8_control` and `support_recent8_suppress_motion_pair2` show
SF-native-like temporal trends. They are functionally longer-recent-window SF,
not role-conditioned methods.

### 3.4 Motion-pair2 as Suppressive is viable

`support_landmark4_suppress_motion_pair2` and
`support_landmark2_motion1_suppress_motion_pair2` are both clean. The
motion-pair mechanism works as a Suppressive cache when the Supportive cache
provides stable identity (landmark).

### 3.5 Landmark4 > landmark2+motion1 for ID retention

Comparing cells 6 and 7: landmark4 provides slightly better ID retention than
landmark2+motion1. The full 4-frame landmark memory is preferable to the
3-frame hybrid.

## 4. Candidate Ranking

1. **support_landmark4_suppress_recent8** — best ID, simple, clean
2. **support_landmark4_suppress_motion_pair2** — equally good, motion-pair adds complexity
3. **support_landmark2_motion1_suppress_recent8** — good, slightly worse ID
4. **support_landmark2_motion1_suppress_motion_pair2** — good, slightly worse ID

## 5. Bug Fix Confirmation

The `role_event.py:613` NoneType crash is fixed (commit `2aacbc2`). All 4
previously-failing cells now generate successfully. The fix adds a `None`
guard before `int(victim_end_t)` conversion in the motion-pair eviction logic.

## 6. Decision

### 6.1 Top candidate

**`support_landmark4_suppress_recent8`** is the leading candidate:
- Best ID retention among all 8 cells
- Simplest design (landmark4 + recent8, no motion-pair complexity)
- No polygon noise
- Mid-video fluctuation recovers

### 6.2 Second candidate

**`support_landmark4_suppress_motion_pair2`** is equally viable if the
motion-pair mechanism is needed for the paper story. However, it adds
implementation complexity without a visible quality gain over recent8.

### 6.3 Next steps

1. **Longer extrapolation (60s+)**: All candidates show mid-video ID
   fluctuation that recovers at 30s. A 60s or 90s test will determine
   whether the recovery is stable or degrades further.
2. **32-prompt screen (v112)**: Run the top candidate on 32 diverse prompts
   with controls (all_recent8, all_landmark4, all_motion_pair2).
3. **VBench-Long + DINO**: After blind review of 32-prompt screen.
