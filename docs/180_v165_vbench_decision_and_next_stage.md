# 180: v165 VBench Decision Pipeline and Next Stage

Date: 2026-08-08

## 1. Scope

v165 has completed 16-prompt generation and the automatic non-VBench screen.
The mechanism audit passes for both stale-tie margins, but the current metrics
do not establish a winner:

- margin `0.03` loses too much first-last and background consistency and is
  retained only as a threshold ablation;
- margin `0.05` improves minimum stability and flicker relative to
  DirectionMatch, but has small DINO and first-last regressions;
- no v165 VBench-Long result exists yet.

The next step is therefore a prompt-correct VBench-Long core-9 evaluation and
a frozen decision analysis. This remains a 16-prompt development experiment.
It must not be presented as held-out paper evidence.

## 2. Code correction

The old `run_v165_vbench_long.py` returned no `metric_promotion_gate`, while
the shared collector reads that compatibility field after writing the summary.
Consequently, a fully completed VBench run would fail with a `KeyError` during
the final collect message even though all scores had been written.

The collector now returns:

- `development_candidate_gate`, which has the intended meaning;
- `metric_promotion_gate`, an explicitly documented compatibility alias;
- aggregate Tie05 comparisons and all frozen gate rows.

The alias never means paper promotion.

## 3. Frozen candidate and gates

The primary candidate is fixed to:

```text
ours_middle10_reservoir2_dirstaletie005
```

Tie03, DirectionFresh, and StateMotion remain diagnostic references. The
primary comparisons are DirectionMatch and native Self-Forcing.

All VBench dimensions are oriented so that larger is better. Composite groups
are:

```text
history  = mean(subject, background, overall consistency)
temporal = mean(flicker, motion smoothness, temporal style)
visual   = mean(aesthetic, imaging quality)
dynamic  = dynamic degree
```

The following thresholds were frozen before v165 VBench results were
available:

| Comparison | Metric | Minimum delta |
|---|---|---:|
| Tie05 - DirectionMatch | history | -0.003 |
| Tie05 - DirectionMatch | temporal | +0.001 |
| Tie05 - DirectionMatch | visual | -0.006 |
| Tie05 - DirectionMatch | dynamic | -0.020 |
| Tie05 - SF | history | +0.002 |
| Tie05 - SF | dynamic | +0.020 |
| Tie05 - SF | temporal | -0.004 |
| Tie05 - SF | visual | -0.006 |

The paired prompt check additionally requires a positive majority on:

- temporal quality versus DirectionMatch;
- history consistency versus SF;
- dynamic degree versus SF.

The analyzer reports paired bootstrap intervals but does not use post-hoc
interval tuning to change these gates.

## 4. Server execution

Use the commit containing this document on every node. Existing generated
videos are reused; this stage performs evaluation only.

```bash
export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
cd "${REPO_ROOT}"
```

Prepare once on node 0:

```bash
export NODE_RANK=0
bash scripts/run_v165_vbench_long.sh prepare
```

Prepare the clean split cache on all nodes:

```bash
export NODE_RANK=<0|1|2|3>
bash scripts/run_v165_vbench_long.sh split
bash scripts/run_v165_vbench_long.sh preflight
```

Run evaluation on all nodes:

```bash
bash scripts/run_v165_vbench_long.sh eval
```

Check completion on node 0. Missing jobs should be resumed rather than
restarting completed dimensions:

```bash
export NODE_RANK=0
bash scripts/run_v165_vbench_long.sh status

# Only when status reports missing jobs:
NUM_NODES=1 NODE_RANK=0 bash scripts/run_v165_vbench_long.sh resume-missing
```

Collect on node 0:

```bash
export NUM_NODES=4 NODE_RANK=0
bash scripts/run_v165_vbench_long.sh collect
```

`collect` now writes the VBench summary and automatically runs the final
decision analyzer. The decision can be regenerated without rerunning VBench:

```bash
bash scripts/run_v165_vbench_long.sh decide
```

## 5. Automatic outputs

The main outputs are:

```text
runs/v165_direction_stale_tie_moviebench16/full8/
|-- metrics/
|   |-- vbench_core9_summary.{json,csv,md}
|   `-- vbench_long_parts/
`-- analysis/
    |-- v165_vbench_analysis.{json,md}
    `-- v165_final_decision.{json,md}
```

The final analyzer independently checks:

- all six methods, nine dimensions, 16 prompts, and 15 clips per prompt;
- per-clip detail scores reconstruct every frozen aggregate score;
- the known 0-100 imaging-quality detail scale is normalized correctly;
- boolean per-clip dynamic-degree decisions are normalized to 0/1;
- Tie05 mechanism execution, changed choices, read budget, and contract gate;
- candidate-specific corruption, identity, background, motion-collapse, and
  discontinuity flags;
- paired prompt deltas, positive-prompt counts, and bootstrap intervals.

## 6. Minimal human review

Do not review all 16 prompts. After `collect` or `decide`, create the blind
bundle with:

```bash
bash scripts/run_v165_vbench_long.sh prepare-review
```

The bundle contains exactly two selected prompts and two methods:

```text
Tie05, DirectionMatch
```

It is capped at four videos. Selection first takes at most two prompts with
candidate-specific safety flags, prioritizing severe corruption, identity,
background, or motion failures. If fewer than two are flagged, the remaining
slot is filled by the worst VBench frontier delta or largest metric
disagreement.
Method names are stored only in `private/blind_key.json`; the reviewer sheet
separates motion amount from motion naturalness. Re-running the packager
validates the frozen video identities but preserves reviewer-entered scores.

This is adaptive engineering triage. It cannot be reported as an unbiased
paper human study because prompts are selected after observing diagnostics.

## 7. Decision branches

1. **All aggregate, paired-support, and safety gates pass:** complete the
   four-video review. If it also passes, freeze Tie05 and move to a separately
   frozen held-out 128-prompt comparison.
2. **A small number of gates disagree:** use only the four-video review to
   decide whether VBench misses a visible motion/identity difference.
3. **At least half of the aggregate gates fail:** reject stale tie-breaking.
   Do not scan more margins. The next generator change should enrich the
   motion descriptor with magnitude or multiple temporal scales.
4. **A real corruption flag appears:** inspect the selected-pair trace around
   that prompt before interpreting any quality score.

The current method story remains layer-conditioned dual-timescale memory plus
direction-compatible motion recall. It does not revive the unsupported static
head taxonomy claim.

## 8. Local verification

Completed locally without a GPU runtime:

```text
python -m py_compile: PASS
bash -n scripts/run_v165_vbench_long.sh: PASS
v165 focused tests: 9 passed, 1 skipped
shared VBench runner regressions: 11 passed
synthetic 6-method x 9-dimension x 16-prompt x 15-clip collect/decision: PASS
four-video blind bundle idempotence, score preservation, and label hiding: PASS
```
