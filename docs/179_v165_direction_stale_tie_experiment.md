# v165 Stale-Aware Direction-Tie Retrieval

Post-generation VBench collection, frozen decision gates, and the four-video
review protocol are documented in
`docs/180_v165_vbench_decision_and_next_stage.md`.

## 1. Decision from v164

v164 established three useful facts on the 16-prompt development suite:

1. Absolute state cosine is saturated (approximately 0.98-1.00) and is not a
   useful retrieval discriminator.
2. Direction-only motion-pair recall improves DINO consistency, first-last
   drift, background drift, late motion, and temporal jump relative to native
   Self-Forcing. It does not preserve identity merely by freezing motion.
3. The global freshness score
   `direction_similarity - 0.25 * age / 24` changes 95 selections and lowers
   recall age, but loses part of DirectionMatch's identity/background benefit
   and does not improve flicker or repetition.

The next experiment therefore does not change the cache or descriptor. It
tests whether freshness should be a conditional tie breaker instead of a
global additive objective.

## 2. Hypothesis

An old motion pair should remain retrievable when it has clearly better motion
direction compatibility. Recency should matter only when:

1. the direction-optimal pair is stale; and
2. a newer pair is nearly direction-equivalent.

For a set of compatible candidates, let `d*` be the maximum direction cosine
and `a*` the age of that candidate. v165 uses:

```text
if a* <= 12:
    select argmax direction_similarity
else:
    tie_set = {i | d* - direction_similarity_i <= margin}
    select newest(tie_set)
```

The direction floor remains 0.1. If no candidate passes it, the newest
age-eligible adjacent pair is read so that the history-read budget is not
silently reduced.

This rule has an explicit interpretation: preserve strong historical motion
evidence, but do not use an old event when a recent event is directionally
indistinguishable within a bounded cosine loss.

## 3. Frozen-trace calibration

The margins were selected before generating any v165 video. The script
`scripts/analyze_v165_margin_replay.py` replays the rule over all 517 compatible
DirectionMatch retrievals saved by v164.

| Margin | Changed choices | Prompts affected | Mean direction loss | P95 direction loss | Mean age gain |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 13 | 10 | 0.0053 | 0.0088 | 9.15 |
| 0.02 | 24 | 13 | 0.0097 | 0.0186 | 9.33 |
| **0.03** | **38** | **14** | **0.0151** | **0.0271** | **10.32** |
| **0.05** | **57** | **16** | **0.0249** | **0.0465** | **9.42** |
| 0.075 | 81 | 16 | 0.0357 | 0.0699 | 9.36 |
| 0.10 | 98 | 16 | 0.0459 | 0.0906 | 9.43 |

`0.03` is the conservative operating point. `0.05` tests whether a moderately
larger equivalence set better fixes the motion/freshness tradeoff. Larger
margins approach or exceed the intervention count of the failed global
freshness penalty and are not generated in v165.

The frozen replay artifacts are:

```text
docs/results/v165_direction_stale_tie_moviebench16/v164_margin_replay.json
docs/results/v165_direction_stale_tie_moviebench16/v164_margin_replay.md
```

## 4. Frozen cache and experiment grid

The selected Middle10 layers use exactly:

| Component | Capacity/read rule |
|---|---|
| Sink | frame 0, one frame |
| Temporal reservoir | two frames |
| Motion archive | four adjacent pairs |
| Motion recall | one atomic adjacent pair |
| Recent | four frames |
| Maximum read budget | nine full-frame equivalents |
| Admission stale horizon | 12 frames |
| Recall maximum age | 24 frames |

All other layers use `sink1 + recent8`. Prompt, seed, layer placement,
admission, cache capacities, direction descriptor, direction floor, fallback,
RoPE, and read budget are frozen against v164 DirectionMatch.

The six-method published grid is:

1. `sf_native` (reused from v164)
2. `ours_middle10_reservoir2_directionmatch1` (reused)
3. `ours_middle10_reservoir2_dirstaletie003` (new)
4. `ours_middle10_reservoir2_dirstaletie005` (new)
5. `ours_middle10_reservoir2_directionfresh1` (reused)
6. `ours_middle10_reservoir2_statemotionpair1_reference` (reused)

Only 32 videos are newly generated. The other 64 are linked from the audited
v164 result. With four 8-GPU nodes, each node receives 24 tasks, of which eight
are new generations.

## 5. Runtime diagnostics

Each retrieval records:

- all age-eligible candidates and both state/direction similarities;
- direction-pass decisions and the exact direction-optimal pair;
- its age and the complete near-equivalent tie set;
- whether the stale gate and tie rule were applied;
- selected pair, direction loss, and age gain against DirectionMatch;
- fallback reason, selected/read equality, and pair-read budget preservation.

`analyze_v165_direction_stale_tie_trace.py` independently recomputes every
choice. It fails on a changed frozen parameter, malformed pair, candidate or
score mismatch, direction loss above the configured margin, age above 24,
selected/read mismatch, or lost pair-read budget.

## 6. Server commands

All nodes must use the same commit, checkpoint, prompt file, shared output, and
environment.

```bash
git fetch origin
git checkout codex/v165-direction-stale-tie
git pull --ff-only

export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export V165_REUSE_V164_ROOT=${REPO_ROOT}/runs/v164_direction_freshness_moviebench16/full8
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export NODE_RANK=0  # use 0, 1, 2, or 3 on the corresponding node
```

First reproduce the frozen calibration once on node 0:

```bash
bash scripts/run_v165_direction_stale_tie_moviebench16.sh calibrate
```

Then run on all four nodes:

```bash
bash scripts/run_v165_direction_stale_tie_moviebench16.sh preflight
bash scripts/run_v165_direction_stale_tie_moviebench16.sh generate
```

After every node finishes, run on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v165_direction_stale_tie_moviebench16.sh audit
bash scripts/run_v165_automated_screen.sh all
```

Do not review videos before the mechanism and automatic screens pass. If they
pass, run VBench-Long core-9:

```bash
# node 0
bash scripts/run_v165_vbench_long.sh prepare

# all four nodes, with their own NODE_RANK
bash scripts/run_v165_vbench_long.sh split
bash scripts/run_v165_vbench_long.sh preflight
bash scripts/run_v165_vbench_long.sh eval

# node 0 after all workers finish
export NODE_RANK=0
bash scripts/run_v165_vbench_long.sh collect
```

Use `resume-missing` instead of restarting completed VBench jobs.

## 7. Decision rule

Evaluate in this order:

1. **Correctness:** both mechanism gates pass with no contract, atomicity,
   score, age, selected/read, or budget violations.
2. **Intervention:** both margins change real DirectionMatch choices. If a
   margin does not change any choice, it is not a tested method.
3. **No corruption:** no decode, luminance, contrast, polygon-noise, or severe
   temporal-discontinuity flag.
4. **Consistency-motion frontier:** compare DINO/subject/background/overall
   consistency jointly with late-motion ratio, dynamic degree, temporal jump,
   and motion smoothness.
5. **Known v164 weaknesses:** require flicker and loop/repetition behavior not
   to worsen materially relative to DirectionMatch.

Prefer margin 0.03 if it is statistically/visually indistinguishable from
0.05 because it preserves more direction evidence. Promote 0.05 only if it
clearly improves motion or temporal quality without losing identity and
background consistency.

If both are worse than DirectionMatch, reject stale tie-breaking and retain
DirectionMatch. The next change should then enrich the motion descriptor
(magnitude or multi-timescale direction), not scan another age coefficient.

## 8. Minimal human review

The automatic stage evaluates every video. Human review is optional and only
resolves metric disagreement. The first wave is limited to four prompts and
three methods:

```text
DirectionMatch, margin 0.03, margin 0.05
```

This is at most 12 videos. Do not review DirectionFresh, StateMotion, or all 16
prompts unless the first wave exposes a specific unresolved failure mode.

## 9. Files to return

Push these small files before any videos:

```text
runs/v165_direction_stale_tie_moviebench16/full8/published_manifest.json
runs/v165_direction_stale_tie_moviebench16/full8/contracts/experiment.json
runs/v165_direction_stale_tie_moviebench16/full8/calibration/v164_margin_replay.json
runs/v165_direction_stale_tie_moviebench16/full8/automated_screen/direction_stale_tie_trace.json
runs/v165_direction_stale_tie_moviebench16/full8/automated_screen/temporal_diagnostics.csv
runs/v165_direction_stale_tie_moviebench16/full8/automated_screen/comprehensive.json
runs/v165_direction_stale_tie_moviebench16/full8/automated_screen/automated_screen.json
runs/v165_direction_stale_tie_moviebench16/full8/metrics/vbench_core9_summary.json
runs/v165_direction_stale_tie_moviebench16/full8/analysis/v165_vbench_analysis.json
```

v165 is a development experiment. Its 16 prompts and parameter selection must
not be presented as held-out paper evidence.
