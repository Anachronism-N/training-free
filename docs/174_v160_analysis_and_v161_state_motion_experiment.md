# 174: v160 Analysis and v161 State-Matched Motion Experiment

Date: 2026-08-04

## 1. Current conclusion

v160 fixed the mechanism failure found in v159. The freshness rule is active:

| Trace metric | v159 | v160 |
|---|---:|---:|
| Accepted motion-pair updates / prompt | 6.125 | 12.562 |
| Pair-age p95 / prompt | 34.490 | 11.006 |
| Maximum pair age | 61 | 13 |
| Below-quantile stale refreshes | 0 | 83 |

This is a mechanism result, not a quality result. In the four-prompt Wave-1
blind review, v160 fresh motion versus the old motion-pair hybrid had:

- overall preference mean delta: `-0.05`;
- motion-naturalness mean delta: `+0.05`;
- identity-continuity mean delta: `+0.125`;
- one severe failure on the dragon/transformation case.

Against reservoir4, v160 was worse in overall preference (`-0.35`), motion
naturalness (`-0.25`), and identity continuity (`-0.375`). Wave 1 is therefore
inconclusive and correctly selected `continue_wave2`. Freshness solved stale
storage, but indiscriminately reading the newest eligible motion pair is not a
reliable video-quality improvement.

The v160 traces were also checked for partial pair reads. Across 7,680
head-level coherent-motion reads, observed read sizes were only zero or two;
no one-frame read occurred. v161 nevertheless makes pair atomicity an explicit
runtime audit so that this invariant cannot silently regress.

## 2. Immediate unfinished v160 action

Complete the frozen second human-review wave before making a final statement
about v160:

```bash
bash scripts/run_v160_automated_screen.sh review-wave2
# Fill the generated v160_wave2_review_sheet.csv without opening private/.
bash scripts/run_v160_automated_screen.sh analyze-wave2
```

This action needs no generation and can run in parallel with v161.

## 3. v161 hypothesis

The v160 failure suggests that a pair can be fresh but incompatible with the
current trajectory. v161 tests one isolated hypothesis:

> A motion memory should be read only when its endpoint resembles the current
> generated state and its descriptor-space transition does not oppose the
> current transition.

"Direction" here is the direction of a normalized clean-K/V descriptor
change, not optical-flow direction. It is an online training-free proxy whose
validity must be established by the experiment.

For each selected Middle10 layer, v161 uses:

```text
sink1 + temporal-reservoir2 + state-matched-motion-pair1 + recent4
```

All other layers use `sink1 + recent8`. The maximum read is nine full-frame
equivalents, equal to the v160 primary. The motion archive can store four
complete adjacent pairs but reads at most one pair.

The selected-layer physical KV storage ceiling is therefore 15 full-frame
equivalents (`1 + 2 + 8 + 4`), versus nine in v160. v161 holds the attention
read budget fixed, not the storage budget. A positive result would still need
an archive-capacity ablation (for example two versus four pairs) and a
storage-matched control before an efficiency claim is valid.

Admission is unchanged from v160: semantic coherence, motion quantile,
adjacency, minimum spacing, 12-frame stale recovery, and stale-quantile
bypass. Readout performs the following steps:

1. Exclude pairs overlapping sink or recent and pairs older than 24 frames.
2. Require endpoint/current-state cosine similarity of at least `-0.25`.
3. Reject an available descriptor transition with cosine below `0.0`.
4. Rank passing pairs lexicographically by transition similarity, state
   similarity, and recency.
5. Read both frames atomically, or abstain when no complete pair passes.

The permissive state floor avoids pretending that a tuned semantic threshold
is already known. Every candidate score is logged, so alternative thresholds
can be analyzed offline before any new generation code is proposed.

## 4. Experiment scope

v161 uses the frozen diverse MovieBench/Qwen 16-prompt suite and generates only
one new method: 16 new 30-second videos. It reuses 80 videos from v160:

| Method | Role | Videos generated in v161 |
|---|---|---:|
| `ours_middle10_reservoir2_statemotionpair1` | new primary | 16 |
| v160 fresh-motion hybrid | direct reference | 0 |
| v159 old motion-pair hybrid | diagnostic reference | 0 |
| reservoir4 | diagnostic reference | 0 |
| all-recent8 | diagnostic reference | 0 |
| SF native | context only | 0 |

The adaptive human comparison contains only the primary, v160 fresh motion,
and reservoir4. Automatic metrics select four diagnostic prompts (12 videos)
for Wave 1. A second 12-video wave is used only when the frozen Wave-1 rule is
inconclusive. This reduces normal manual review from all 96 videos to 12.

## 5. Four-node generation commands

Use the same shared repository revision and shared output filesystem on all
four nodes. Run the matching command on each node with rank `0`, `1`, `2`, or
`3`:

```bash
cd /path/to/training-free
export NODE_RANK=0                 # change on each node
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export V161_REUSE_V160_ROOT=/path/to/training-free/runs/v160_fresh_motion_moviebench16/full8
bash scripts/run_v161_state_matched_motion_moviebench16.sh preflight
bash scripts/run_v161_state_matched_motion_moviebench16.sh generate
```

After every node reports success, run once on node 0:

```bash
export NODE_RANK=0
export NUM_NODES=4
bash scripts/run_v161_state_matched_motion_moviebench16.sh audit
bash scripts/run_v161_automated_screen.sh all
```

`all` runs the mechanism audit, CPU temporal diagnostics, six parallel
comprehensive evaluations, the automatic safety screen, and, only if the
mechanism gate passes, preparation of the 12-video Wave-1 blind review.

After filling the generated review sheet:

```bash
bash scripts/run_v161_automated_screen.sh analyze-wave1
# Only when the result says continue_wave2:
bash scripts/run_v161_automated_screen.sh review-wave2
bash scripts/run_v161_automated_screen.sh analyze-wave2
```

Diagnostics can be packaged with:

```bash
bash scripts/run_v161_state_matched_motion_moviebench16.sh package
```

## 6. Required trace evidence

The mechanism gate requires all of the following:

- archive size reaches at least two candidate pairs;
- at least one read has multiple eligible candidates;
- state matching selects a non-newest pair or abstains on incompatibility;
- every emitted motion read is exactly zero or two adjacent frames;
- no selected pair exceeds the 24-frame read horizon;
- the policy/configuration contract has no failures.

Inspect these fields when debugging:

```text
state.pair_frame_ids
state.last_decision
state.last_retrieval.candidates[*].state_similarity
state.last_retrieval.candidates[*].direction_similarity
state.last_retrieval.candidates[*].state_pass
state.last_retrieval.candidates[*].direction_pass
state.last_retrieval.selected
state.last_retrieval.reason
strategy.frame_ids
```

The aggregate report additionally records multi-candidate reads, non-newest
selections, negative-direction rejections, abstentions, selected ages, and
atomic-read violations.

## 7. Decision rules

1. If the mechanism gate fails, do not review videos. The test did not actually
   exercise state-conditioned choice.
2. If corruption or repeated geometry inversion appears, stop this route and
   inspect the exact selected pairs around the first failure.
3. If v161 improves motion naturalness and overall preference over v160 without
   losing identity, freeze it and run a separate held-out fixed protocol.
4. If it is indistinguishable or worse, reject descriptor-transition retrieval.
   Use the logged score distributions to diagnose why, but do not tune a
   threshold and claim confirmation on these same 16 prompts.

v161 is exploratory failure recovery. Adaptive prompt selection, reused
development references, and post-hoc threshold inspection are not paper
evidence. A positive result must be followed by a preregistered held-out test.
