# ProbeCache v82: Deadline-Aware 10-Hour Follow-up Plan

> Date: 2026-07-24
>
> Status: code and schedule prepared locally; GPU execution is pending.
> The v81 profile/main screen that is already running must not be restarted.

## 1. Objective

The available window is approximately 10 hours on 16 H20 GPUs, followed by
human review. The follow-up queue answers questions that the current v81
screen cannot answer:

1. Does the counterfactual head classification matter, or would PF labels,
   inverted labels, or random labels work equally well?
2. Is the learned classification reproducible under independent seeds?
3. Does a candidate improvement survive multiple generation seeds?
4. Does it still work at 60 seconds, where long-horizon identity and state
   errors are easier to expose?
5. Does the method retain its behavior during A-B-A prompt switching?

The first four questions belong to the primary single-prompt paper task.
Prompt switching remains secondary and is the first generation phase skipped
when the deadline is tight.

## 2. Reused v81 results

Do not rerun cells already covered by
`scripts/run_v81_probecache_16gpu.sh single`:

- native SF, official PF, and Echo;
- audit, persistent-only, reactive-only, and full ProbeCache;
- no-trust;
- archive 12/24/36;
- top-k 2/4/6;
- prompt weight 0/0.15/0.30;
- open/default/conservative retrieval admission.

The v82 queue consumes the v81 primary profile:

```text
runs/v81_probecache_profile/labels/probecache_binary_labels.csv
runs/v81_probecache_profile/labels/probecache_profile_report.json
```

If either file is missing, v82 stops before the control-label experiments.

## 3. Time and priority budget

Historical server measurements in `docs/56` give approximately 8 minutes for
one 120-frame video. A cell executes its prompts serially on one GPU while 16
cells run in parallel.

| Priority | Phase | Work | Conservative estimate | Skip policy |
|---:|---|---|---:|---|
| P0 | Existing v81 | Already running profile/main screen | 1.5-2.5 h | Never duplicate |
| P1 | `profile-replica` | 48 profiles, seeds 2/3 | 45 min | Keep if possible |
| P1 | `labels` | 16 cells x 3 prompts x 120f | 50 min | Required for classification claim |
| P1 | `confirm` | 16 cells, 12 prompts, new seeds | 150 min | Required for effect claim |
| P1 | `ultralong` | 16 cells x 6 prompts x 240f | 180 min | Required for primary task |
| P2 | `switch` | Existing v81 switch matrix, 3 prompts | 70 min | First phase to skip |
| P1 | `prepare` | Blind review packages | 10 min | Always run |

The required P1 phases total 7 hours 15 minutes after v81. Adding the optional
switch phase raises the follow-up total to 8 hours 25 minutes. With a
20-minute safety window, v81 plus all P1 phases fits a 10-hour allocation when
v81 finishes in roughly 2 hours; switch runs only when earlier phases finish
ahead of the conservative estimates. Every phase checks the absolute deadline
before it starts; an active phase is not killed halfway.

## 4. Head-profile replication

`profile-replica` repeats the 48 paired profile jobs with seeds 2 and 3 in a
separate output root. It applies the same strict internal profile gates as
v81, then compares the two binary maps using:

- overall head agreement;
- Cohen's kappa;
- persistent and reactive Jaccard;
- per-layer agreement and role counts.

Outputs:

```text
runs/v82_probecache_profile_replica/labels/probecache_binary_labels.csv
runs/v82_probecache_profile_replica/labels/profile_replication.json
runs/v82_probecache_profile_replica/labels/profile_replication.md
```

An overall agreement of 0.60 is the provisional minimum for treating the
classification as reproducible. This threshold is diagnostic, not tuned to
video quality. A failed internal profile gate makes the replica unusable; v82
substitutes a clearly named random control rather than silently using it.

## 5. Classification-causality matrix

The `labels` phase uses three deliberately different 30-second prompts:

1. identity plus evolving object geometry;
2. fast human motion plus camera and layout changes;
3. person-object joint identity plus discrete state/count changes.

The 16 cells are:

| GPU | Cell | Purpose |
|---:|---|---|
| 0 | `pf` | Official PF reference |
| 1 | `v78` | Validated transition reference |
| 2 | `learned` | Primary counterfactual labels |
| 3 | `profile_replica` | Independent labels; random fallback is named |
| 4 | `pf_binary` | PF Anchor -> persistent, Wave+Veil -> reactive |
| 5 | `inverse` | Swap every learned role |
| 6-7 | `random_2026/2027` | Random labels preserving each layer's role count |
| 8 | `remote_only` | Rank only by remote-history response |
| 9 | `prompt_only` | Rank only by prompt response |
| 10-12 | `layer_early/middle/late` | Activate one depth third |
| 13-14 | `layer_first_half/second_half` | Coarse depth control |
| 15 | `learned_audit` | Learned labels but no ProbeCache override |

`scripts/build_probecache_control_labels.py` writes every map and a manifest
containing agreement, role counts, persistent Jaccard, and per-layer counts.
Random controls preserve per-layer role balance, so they do not win or lose
only because one layer receives more persistent heads.

The classification contribution is supported only if:

- `learned` is better than `inverse` and both random controls;
- it is competitive with or better than `pf_binary`;
- `remote_only` and `prompt_only` each lose a complementary capability;
- the active depth result is coherent across prompts;
- traces confirm similar active-cache budgets across label controls.

If all label maps are visually equivalent, head classification must be removed
from the main claim even if ProbeCache itself remains useful.

## 6. Multi-seed confirmation

The `confirm` phase uses all 12 complex single-prompt cases:

- PF, v78, learned, and PF-binary: seeds 1, 2, and 3;
- learned open admission: seeds 1 and 2;
- learned conservative admission: seeds 1 and 2.

Seed 0 already exists in v81. The combined result provides four seeds for the
core methods and three seeds for admission variants without spending GPUs on
already falsified cache budgets.

Do not select a winning seed. Aggregate per-prompt and per-seed results and
report median, interquartile range, and worst-seed behavior.

## 7. Sixty-second single-prompt extrapolation

The `ultralong` phase uses six complex prompts and 240 latent frames. It runs
two seeds for:

- native SF;
- official PF;
- validated v78;
- learned ProbeCache;
- PF-binary ProbeCache;
- independent-profile ProbeCache, or named random fallback;
- learned open admission;
- learned conservative admission.

This is the most important follow-up for the paper's primary task. Human review
must inspect approximately 0, 15, 30, 45, and 60 seconds, recording:

- first irreversible identity change;
- face, clothing, body, object, and count consistency;
- whether the requested state continues to evolve;
- repeated action or frozen motion;
- acceleration jumps, cuts, flashes, darkening, and texture collapse;
- background/layout return or drift.

An improvement visible only before 30 seconds is not a long-video result.

## 8. Prompt-switch phase

The `switch` phase reuses the v81 A-B-A matrix and three complex prompts. It
runs only after the single-prompt phases and only if at least 90 minutes remain.
The persistent archive should make A return possible, while reactive heads
must stop using B events after the segment boundary.

Review:

- A identity before the first switch;
- B prompt compliance and transition latency;
- A identity/layout return;
- stale B leakage after return;
- switch flicker and motion reset.

## 9. Exact server command

Set the deadline from the beginning of the allocation, not from the end of the
currently running command. Pulling the new commit while an existing Python
process is running does not change modules already loaded by that process.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

# Example only: set this to the real Unix timestamp 10 hours after allocation.
export DEADLINE_EPOCH="$(date -d '+10 hours' +%s)"

git pull
mkdir -p runs

WAIT_FOR_IDLE=1 \
DEADLINE_EPOCH="$DEADLINE_EPOCH" \
nohup bash scripts/run_v82_probecache_10h.sh all \
  > runs/v82_probecache_10h.log 2>&1 &
echo $! > runs/v82_probecache_10h.pid
```

`WAIT_FOR_IDLE=1` polls all requested GPUs and starts only when each is below
2048 MiB, preventing overlap with the v81 jobs. Monitor with:

```bash
tail -f runs/v82_probecache_10h.log
ps -fp "$(cat runs/v82_probecache_10h.pid)"
```

The main log prints the remaining deadline budget before every phase.
To execute one phase manually:

```bash
bash scripts/run_v82_probecache_10h.sh labels
bash scripts/run_v82_probecache_10h.sh confirm
bash scripts/run_v82_probecache_10h.sh ultralong
bash scripts/run_v82_probecache_10h.sh switch
bash scripts/run_v82_probecache_10h.sh prepare
```

Resume is safe: complete video/trace cells are skipped unless `FORCE=1`.

## 10. Runtime diagnostics

Every v82 generation phase writes:

```text
<run_root>/configs/<cell>.env
<run_root>/logs/<cell>.log
<run_root>/traces/<cell>.probecache.jsonl
<run_root>/run_audit.json
<run_root>/run_audit.md
```

The strict auditor checks:

- expected MP4 count;
- inference log existence;
- traceback, CUDA OOM, assertion, killed process, and explicit error markers;
- nonempty ProbeCache trace for every ProbeCache cell.

v82 sets `pyramidkv_probecache_trace_selection_stride=4`: archive updates stay
fully logged, while middle selections are sampled every fourth query epoch.
This reduces NFS trace traffic by about 4x without changing cache behavior.
The v81 default remains 1.

ProbeCache traces must additionally satisfy the v81 invariants:

- one archive update per clean block/layer;
- no selected frame inside recent-4 or in the future;
- persistent selected age greater than reactive selected age;
- both roles have nonzero accepted calls;
- abstention has explicit reasons;
- archive size respects the configured budget;
- prompt switches advance segment IDs.

## 11. Blind review packages

The final `prepare` phase creates three packages without computing quality
metrics:

```text
runs/v82_probecache_labels/blind_review/
runs/v82_probecache_ultralong/blind_review/
runs/v82_probecache_switch/blind_review/
```

It also creates `runs/v81_probecache_single/blind_review/` when all nine v81
core methods have 12 videos. An incomplete v81 matrix is reported and skipped
without blocking complete v82 packages.

Review order:

1. v81 `PF / v78 / learned full` on the 12 main prompts.
2. v82 label controls on three diagnostic prompts.
3. v82 60-second shortlist.
4. v82 switch shortlist.
5. Multi-seed spot checks for any apparent winner and failure.

Freeze every scorecard before running DINO, VBench-Long, or composite metrics.
This prevents metric-guided relabeling of subjective results.

After scores are frozen:

```bash
HUMAN_REVIEW_DONE=1 \
bash scripts/postprocess_v82_probecache.sh labels

HUMAN_REVIEW_DONE=1 \
bash scripts/postprocess_v82_probecache.sh confirm

HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 \
bash scripts/postprocess_v82_probecache.sh ultralong

HUMAN_REVIEW_DONE=1 \
bash scripts/postprocess_v82_probecache.sh switch
```

## 12. Decisions after review

### Strong outcome

Use the ProbeCache story if learned labels are reproducible, beat negative
label controls, and learned full is competitive with PF/v78 across seeds while
improving 60-second identity or state progression without motion loss.

### Cache works, classification does not

If learned, PF-binary, and random controls are similar, retain dual-lifecycle
cache as an engineering observation but remove counterfactual classification
from the headline. Prefer the simplest stable mapping.

### v78 remains best

If ProbeCache loses to PF/v78 or freezes motion, retain v78 as the paper
candidate. ProbeCache becomes a documented negative extension, not a result to
repackage.

### Scientific boundary

Do not claim anchor/recent memory, binary heads, Q/K retrieval, novelty update,
or PF's cache layout as individually novel. The only defensible ProbeCache
claim is the measured-role plus role-specific active-memory loop, and only if
the classification controls support it.
