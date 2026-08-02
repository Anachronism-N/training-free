# 168: v158 Interleaved Budget Sweep — Preflight and Human Gate Block

Date: 2026-08-02
Commit: local (v158 code committed)

## 1. Summary

v158 (interleaved budget sweep: 6/8/10/12 layers) is code-complete and
preflight passes, but **generation is hard-blocked** by a pre-registered
human gate: v158 requires the v157 blind review to pass
(`human_promotion_gate = true`) before any GPU generation can start.

The v157 blind review sheet has 128 rows but all 8 human scoring fields
(identity, background, motion, artifacts, late stability, prompt fidelity,
overall preference, severe failure) are empty (0/112 scored). The blind
review is a human task that cannot be automated.

## 2. v158 Preflight

```
[v158-budget-maps] PASS maps=4 selected_heads=[72, 96, 120, 144]
[V158Contract] {"launch_ready":false,"new_videos":48,"reused_videos":80}
[v158-preflight] HOLD node=0/4 tasks=32 gpus=8 blind=missing
```

The preflight passes all map, contract, and reuse checks. The only blocker
is `blind=missing` — the v157 blind review report does not exist.

## 3. v158 Generate Block

```
v158 generation is blocked until the frozen v157 blind review passes:
runs/v157_layer_gated_moviebench16/full8/analysis/v157_blind_review_report.json
```

The generate stage checks for a frozen `v157_blind_review_report.json` with:
- `experiment = v157_layer_gated_moviebench16_blind_review`
- `primary = ours_layer_interleaved10_reservoir4`
- `prompt_count = 16`
- `human_promotion_gate = true`

This file does not exist. The block is by design (doc 167 section 6):
"缺失或失败时 GPU launch 会硬阻断".

## 4. v158 Experiment Design (for reference)

### 4.1 Nested budget maps

| Budget | Reservoir layers | Heads | Role |
|---:|---|---:|---|
| 6 | 1,7,13,16,22,28 | 72 | exploratory lower bound |
| 8 | 1,4,7,13,16,22,25,28 | 96 | **preregistered primary** |
| 10 | 1,4,7,10,13,16,19,22,25,28 | 120 | exact v157 reference (reused) |
| 12 | 0,1,4,7,10,13,16,19,22,25,28,29 | 144 | exploratory upper bound |

All sets are strictly nested. 3 new methods (interleaved6/8/12) + 5 reused
from v157 = 8 methods × 16 prompts = 128 videos (48 new + 80 reused).

### 4.2 Primary hypothesis

interleaved8 uses 20% fewer reservoir layers than v157's interleaved10, and
should still retain the Pareto improvement (high dynamic degree + recovered
temporal stability + non-inferior visual quality).

### 4.3 Frozen gates

interleaved8 must pass the original v157 five gates PLUS non-inferiority vs
interleaved10 reference (dynamic ≥ -0.02, temporal ≥ -0.002, history ≥ -0.002,
visual ≥ -0.005). Blind promotion requires ≤1 severe failure, ≥10/16
non-inferior prompts, and bounded identity/background/motion deltas.

## 5. What is needed to unblock

1. **Complete the v157 blind review**: a human reviewer must watch the 128
   anonymous v157 videos and fill `v157_review_sheet.csv` with scores for
   identity continuity, background continuity, motion quality, artifact-free,
   late stability, prompt fidelity, overall preference, and severe failure.

2. **Run the v157 blind analyzer**: `python scripts/analyze_v157_blind_review.py`
   to produce `v157_blind_review_report.json` with `human_promotion_gate`.

3. If the gate passes, create `contracts/v157_blind_authorization.json` and
   run v158 generate.

## 6. GPU and occupy status

All 32 GPUs remain occupied (813 MiB, 49-100%). The v158 block does not
affect GPU availability — no GPUs were used for the blocked generate attempt.

## 7. Next steps

The v158 experiment cannot proceed without human input. The supervisor
attempted preflight (PASS) and generate (BLOCKED). No retries are possible
because the block is deterministic (missing human review, not a transient
failure). The experiment is ready to launch as soon as the v157 blind review
is completed.
