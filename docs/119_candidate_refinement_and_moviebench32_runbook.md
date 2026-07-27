# v119 Candidate Refinement and v120 MovieBench-32 Runbook

> Post-result correction: the two sink3 cells produced polygon noise and are
> retired. Current safe execution and split baseline/ours commands are in
> `docs/121_v119_sink3_bugfix_and_v120_safe_launch.md`.

## 1. Goal

This round has two strictly separated stages:

1. **v119, one prompt:** resolve the remaining Retrieval and sink-budget
   questions with five new 30 s videos. This stage is complete.
2. **v120, 32 prompts:** run only SF, native PF, and one manually promoted
   binary-cache candidate, then evaluate all methods with VBench-Long.

Do not add CEMR, v78, or other historical tricks before the v120 base
candidate is selected. Their effects need separate ablations.

## 2. Frame accounting

All cache counts below are **latent/KV frames**, not decoded RGB frames.
Generation uses 120 latent output frames and must decode to exactly 477 frames
at 16 fps (29.8125 s).

- `MotionPair1` is one adjacent pair and therefore occupies **two** cache
  frames.
- `Retrieval1` occupies one exact full frame selected from a bounded archive.
- `age24` allows a retrieved frame to be at most 24 latent frames old, about
  96 decoded frames or 6 s.
- Recent frames are excluded from every middle bank at read time.
- Every middle item stores clean K/V and its original position sidecar.
- Labels 10/11 exclusively own sink, middle, and recent cache. PF's legacy
  dynamic history path cannot run in parallel.

## 3. v119 methods

The frozen head map remains old-v98: 304 History-Supportive heads and 56
History-Suppressive heads.

| Method | Supportive cache | Suppressive cache | Max FFE | Question |
|---|---|---|---:|---|
| `legacy_v98_landmark4_retrieval1` | sink1 + Landmark4 + recent4 | sink1 + Retrieval1 + recent7 | 9 | Does top-1 retrieval avoid late scale enlargement? |
| `legacy_v98_landmark4_retrieval1_age24` | sink1 + Landmark4 + recent4 | sink1 + Retrieval1(age<=24) + recent7 | 9 | Is stale retrieval the source of late enlargement? |
| `legacy_v98_landmark4_retrieval1_motion1_age24` | sink1 + Landmark4 + recent4 | sink1 + Retrieval1(age<=24) + MotionPair1(2 frames) + recent5 | 9 | Can bounded identity recall and coherent motion coexist? |
| `legacy_v98_landmark4_motion1_sink3_extra` | sink3 + Landmark4 + recent4 | sink3 + MotionPair1(2) + recent6 | 11 | Does extra immutable sink context help when budget is allowed to grow? |
| `legacy_v98_landmark2_motion1_sink3_budget9` | sink3 + Landmark2 + recent4 | sink3 + MotionPair1(2) + recent4 | 9 | Does sink3 help after paying for it from middle/recent capacity? |

`FFE` means full-frame equivalent. The 11-FFE method is a diagnostic and
cannot be claimed as a budget-matched main method.

Post-result status: the first three Retrieval cells are clean. Both sink3
cells produced polygon noise because the complete three-frame opening block
was captured as a time-synchronised sink with zero dynamic recent frames.
They are invalid candidates and blocked from new runs.

Existing controls must be reused rather than regenerated:

- v116 `legacy_v98_support_landmark4_suppress_motion_pair1`
- v116 `legacy_v98_support_landmark4_suppress_retrieval2`
- v116 `legacy_v98_support_prototype4_suppress_motion_pair1`

## 4. v119 execution

Launch node 0 first because it freezes the shared experiment contract. Then
launch nodes 1-3. Replace only `REPO_ROOT`, environment activation, and
`NODE_RANK` when needed.

```bash
export REPO_ROOT=/path/to/training-free
cd "$REPO_ROOT"
git pull --ff-only

export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7
export OUT_ROOT="$REPO_ROOT/runs/v119_candidate_refinement_1video"

python scripts/run_v119_candidate_refinement_1video.py retrieval
```

The command is retained only for reproducing the three safe Retrieval cells.
The completed sink3 outputs must not be regenerated or promoted.

Each completed cell must have all of the following:

- one 477-frame video;
- a frozen config and done marker;
- policy trace and audit;
- role-event trace and audit;
- actual sink/middle/recent token counts;
- Retrieval archive capacity, selected frame IDs, similarity/MMR, selected
  age, age-filter count, and `max_age`;
- MotionPair bank contents, spacing decisions, and admission/replacement
  decisions.

Any missing marker or trace is a failed run, even if an MP4 exists.

## 5. Promotion rule

Review the full 30 s video, with extra attention to 20-30 s:

1. reject polygon noise, subject duplication, abrupt scale growth, frozen
   motion, or background collapse;
2. prefer stable identity and background over a small short-term sharpness
   difference;
3. when visually tied, prefer a 9-FFE method with a simple causal mechanism;
4. reject both sink3 cells because they violate the opening-cache contract;
5. record the selected key before starting v120.

Recommended decision order:

1. `landmark_retrieval_motion` if the hybrid is clean and removes Retrieval's
   late enlargement;
2. `landmark_retrieval1_age24` if bounded Retrieval is clean but the hybrid
   hurts motion;
3. retain `landmark_motion1` when neither Retrieval refinement improves the
   existing balanced candidate.

The promotion key is a runner alias, not the full video directory name.

## 6. v120 methods and models

Default v120 contains exactly:

- `sf_native`;
- `pf_native`;
- `ours_landmark_motion1`.

One promoted candidate replaces the default ours key. At most two ours
candidates are allowed when the v119 review is genuinely tied.

Required files:

```text
third_party/Self-Forcing/configs/self_forcing_dmd.yaml
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt
configs/head_maps/legacy_v98_absolute_sign_304_56.csv
```

The prompt file must contain exactly 32 non-empty prompts. Every method uses
the same prompt file, prompt index, seed 0, and per-prompt reseeding.

## 7. v120 generation

### Baselines may run before v119 promotion

When an experimental v119 cache is still under correctness review, SF and PF
may be generated in an isolated method set without enabling any role-memory
candidate:

```bash
export REPO_ROOT=/path/to/training-free
cd "$REPO_ROOT"
git pull --ff-only

export V120_BASELINE_ONLY=1
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

python scripts/run_v120_moviebench32_main.py generate
```

Run the same command on nodes 1-3 with the corresponding `NODE_RANK`. This
creates `runs/v120_moviebench32_main/baselines_seed0` and schedules only 64
videos, 16 per node. It does not require `V119_PROMOTION_APPROVED`.

After all nodes finish:

```bash
python scripts/run_v120_moviebench32_main.py audit
```

Keep `V120_BASELINE_ONLY=1` for the audit command. Do not run the full
SF/PF/ours command until the experimental cache has passed correctness
review. Baseline VBench may run separately with `V120_SCOPE=baselines`.

### Ours-only run reusing completed baselines

After review, generate the established v116 control and the clean v119 hybrid
without regenerating SF/PF:

```bash
unset V120_BASELINE_ONLY
export V120_OURS_ONLY=1
export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_motion1,landmark_retrieval_motion
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

python scripts/run_v120_moviebench32_main.py generate --ours-only
```

Run on all four nodes, then audit once on node 0 with the same environment:

```bash
python scripts/run_v120_moviebench32_main.py audit --ours-only
```

### Full SF/PF/ours run

Example with the bounded Retrieval+Motion candidate:

```bash
export REPO_ROOT=/path/to/training-free
cd "$REPO_ROOT"
git pull --ff-only

export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_retrieval_motion
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

python scripts/run_v120_moviebench32_main.py generate
```

Run on nodes 1-3 with the same variables except `NODE_RANK`. With one ours
candidate, the 96 tasks partition to 24 videos per node and three sequential
videos per GPU. With two ours candidates, there are 128 tasks, 32 per node.

After all nodes finish, run on node 0:

```bash
export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_retrieval_motion
export NODE_RANK=0
export NUM_NODES=4

python scripts/run_v120_moviebench32_main.py audit
```

Do not start VBench until `published_manifest.json` exists and reports
`"ok": true`.

## 8. VBench-Long

The primary metrics in this round are:

- subject consistency;
- background consistency;
- aesthetic quality;
- imaging quality;
- motion smoothness;
- dynamic degree.

Run evaluation on all four nodes:

```bash
export REPO_ROOT=/path/to/training-free
export V119_PROMOTION_APPROVED=1
export V120_CANDIDATES=landmark_retrieval_motion
export NODE_RANK=0
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7

bash scripts/run_v120_vbench_long.sh eval
```

Change `NODE_RANK` on the other nodes. After every evaluation job completes,
collect once on node 0:

```bash
bash scripts/run_v120_vbench_long.sh collect
```

For split generation, set `V120_SCOPE=baselines` or `V120_SCOPE=ours` before
the corresponding evaluation. Merge their summaries with
`scripts/merge_v120_vbench_summaries.py`; the exact command is in docs/121.

Outputs:

```text
runs/v120_moviebench32_main/<method-set>/metrics/vbench_long_summary.json
runs/v120_moviebench32_main/<method-set>/metrics/vbench_long_summary.csv
runs/v120_moviebench32_main/<method-set>/metrics/vbench_long_summary.md
```

This round does not use DINO as the promotion criterion. DINO and loop
diagnostics may be added later as secondary analysis, but must not override a
clear VBench-Long or human-review failure.

## 9. Interpretation

The three v119 Retrieval variants isolate different causes:

- `Retrieval2 -> Retrieval1` tests read strength/top-k count;
- `Retrieval1 -> Retrieval1 age24` tests stale-memory age;
- `Retrieval1 age24 -> Retrieval1 age24 + MotionPair1` tests whether
  retrieval needs a motion-preserving companion.

The sink variants are a negative correctness result. They changed every head
to sink3, collapsed the complete opening block into the static sink, and left
no dynamic recent frame. They cannot be used to infer whether a later
Supportive-only sink lifecycle would help.
