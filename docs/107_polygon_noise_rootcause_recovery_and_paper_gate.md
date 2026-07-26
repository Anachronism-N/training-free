# v107 Polygon-Noise Root-Cause Recovery and Paper Gate

Date: 2026-07-27

Status: code complete; requires one-video server execution and human review.

This document supersedes the immediate execution instructions in docs/104 and
docs/105. It does not erase the v100 results in docs/106. It corrects their
causal interpretation and defines the shortest experiment that can safely
select the next method.

## 1. What is known

The v100 human review establishes the following facts:

1. PF native was free of polygon noise on the selected single and A-B-A
   prompts, although it still duplicated or hallucinated subjects.
2. Every non-native v100 cell used the same tracked old-v98 map:
   `304` label-10 heads and `56` label-11 heads.
3. Every one of those binary cells showed polygon noise.
4. Changing only the label-11 cache among cyclic, motion, cyclic+motion,
   recent-only, scene-cache, and v78 variants did not remove the failure.
5. The old map routes `133/156` PF Wave heads through label 10, whose v100
   route is stride.
6. Earlier v93 and v99 evidence shows that a binary stride/cyclic topology can
   generate usable or clean videos. Therefore binary routing itself is not
   disproved.

The common failed factor is the old `304/56` membership plus its label-10
stride route. The strongest current hypothesis is that moving many Wave heads
from cyclic to stride corrupts generation.

This also explains why changing the label-11 cache did not help: those
experiments changed at most 56 heads while the same 304 heads, including 133
Wave heads, continued to read stride history. A wrong temporal read pattern is
reused at many denoising blocks and autoregressive segments, so an initially
small attention error can become a stable geometric/polygon artifact. This is
a mechanism hypothesis, not proof that the decoder or motion selector itself
is correct.

That hypothesis is not yet a causal result. v100 did not run the two controls
needed to isolate it:

- PF-AR: Anchor -> stride, Wave+Veil -> cyclic;
- PF-AW: Anchor+Wave -> stride, Veil -> cyclic.

## 2. Configuration error

v100 did not use the map family that produced the clean v99
middle-relative smoke result.

Both families used a file named `history_polarity_zero.csv`, but they came
from different directories and different score definitions:

| Map family | Score | Label split | v100 use |
|---|---|---:|---|
| tracked old-v98 | absolute signed history mass | 304/56 | used |
| rebuilt middle-relative | middle-vs-recent intervention margin | 33/327 | not used |

The v100 runner hard-coded the tracked old-v98 path and explicitly required
the `304/56` count. The experiment therefore tested its cache variants under
the wrong classifier family for the intended v99 continuation.

This is an experiment-provenance bug, not evidence that all cache variants are
intrinsically broken.

### 2.1 Correction to docs/106

The statement that a `33/327` map "identifies 169/172 Anchor heads" is
mathematically impossible if label 10 has only 33 heads. The repository also
does not contain the raw server-generated 33/327 CSV and manifest.

The final PF cross-tab must be read from the rebuilt manifest. It must not be
reconstructed from prose. The v107 runner:

1. rebuilds the map from the frozen middle-relative score CSV and artifact;
2. validates all source hashes and acceptance gates;
3. requires exactly `33/327`;
4. records the real PF cross-tab without imposing an overlap claim.

## 3. Code audit

The normal stride and cyclic implementations were compared between the
known-clean v99 revision and the v100 implementation revision.

- `cyclic.py`: normal update/read behavior is unchanged; scene reset support
  was added.
- `stride.py`: normal update/read behavior is unchanged; scene-bank
  archive/restore was added.
- `adaptive_cache.py`: motion-event context and scene-switch traces were
  added; the non-motion, non-scene route remains the exclusive composition
  path introduced by v99.
- `policy_overrides.py`: v100 added new routes, but the plain `cyclic` route
  still resolves to `sink1 + cyclic4 + recent4`.

No static code difference explains why every policy failed while PF native
remained clean. This strengthens the map/routing hypothesis, but only the
current-code PF-AR/PF-AW controls can rule out a shared binary-path regression.

Merge remains excluded from the candidate. It produced polygon artifacts in
multiple corrected runs and is unnecessary for this root-cause test.

## 4. v107 implementation

Entry point:

```text
scripts/run_v107_polygon_rootcause_1video.py
```

The runner is fail-closed:

- it never reads the tracked old-v98 map;
- it rebuilds maps under `OUT_ROOT/maps`;
- it validates the score artifact and map manifest;
- it rejects any primary map other than `33` label-10 and `327` label-11;
- each cell explicitly names its map key;
- every cell generates one 30-second video with seed 0;
- runtime labels, sink/recent sizes, middle strategies, exclusive owner,
  decoded video, motion selections, transitions, and scene events are audited;
- configs, map hashes, implementation hashes, traces, and completion markers
  are immutable.

The historical v100 and v101 runners are now disabled by default. Their
explicit override flags are only for reproducing the known failed matrices.

## 5. Eight one-video cells

### 5.1 Root-cause cells

| Cell | Map | Label 10 | Label 11 | Question |
|---|---|---|---|---|
| `pf_ar_stride_cyclic_control` | PF-AR 172/188 | stride4 | cyclic4 | Is the current binary path clean when all Wave heads stay cyclic? |
| `pf_aw_stride_cyclic_control` | PF-AW 328/32 | stride4 | cyclic4 | Does moving all Wave heads to stride trigger noise? |
| `middle_relative_stride_cyclic_control` | rebuilt 33/327 | stride4 | cyclic4 | Does the intended independent classifier reproduce the clean v99 result? |

All cyclic routes preserve the tested PF Wave layout:

```text
sink1 + cyclic(period 6, capacity 4) + recent4
```

All stride routes use:

```text
sink3 + stride(interval 6, capacity 4) + recent4
```

### 5.2 Candidate cache cells

Membership is fixed to the rebuilt middle-relative map.

| Cell | Label-11 middle | Purpose |
|---|---|---|
| `middle_relative_cyclic4_motion1` | cyclic4 + motion-event1 | Preserve the complete clean cyclic base and add one high-change event slot |
| `middle_relative_cyclic2_motion2` | cyclic2 + motion-event2 | Test the older equal-middle-budget replacement |
| `middle_relative_stride_cyclic_v78` | cyclic4, v78 writes | Test whether trust-conditioned updates add value independently |

The new `cyclic_motion1` policy deliberately does not replace cyclic
evidence:

```text
sink1 + cyclic4 + motion-event1 + recent4
```

Motion-event1 is a small additive screen. If it helps, later experiments must
report its extra cache and runtime cost and include a capacity-matched control.
It is not yet a paper result.

### 5.3 A-B-A cells

| Cell | Scene episode | Purpose |
|---|---|---|
| `aba_middle_relative_no_episode` | off | Binary cyclic A-B-A baseline |
| `aba_middle_relative_episode_bridge1` | on | Archive/restore stride memory and reset local cyclic memory |

A-B-A remains secondary. It cannot promote a method whose single-prompt cell
has polygon noise.

## 6. Server prerequisites

The server must contain:

```text
<SCORE_ROOT>/scores/qk_head_scores.csv
<SCORE_ROOT>/scores/qk_head_score_artifact.json
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
```

`SCORE_ROOT` must be the accepted v98 middle-relative profiling output, not
`runs/v98_history_polarity`.

Verify before launch:

```bash
test -f "$SCORE_ROOT/scores/qk_head_scores.csv"
test -f "$SCORE_ROOT/scores/qk_head_score_artifact.json"
test -f third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt

python -m pytest -q \
  tests/test_v107_polygon_rootcause_contract.py \
  tests/test_v97_policy_contract.py \
  third_party/Pyramid-Forcing/tests/test_factory.py
```

## 7. Launch commands

Pull the same commit on every node. All nodes must see the same repository,
`SCORE_ROOT`, and `OUT_ROOT`. Start rank 0 first because it builds the maps and
writes the immutable contract.

```bash
export SCORE_ROOT="$PWD/runs/v98_middle_relative_scores"
export OUT_ROOT="$PWD/runs/v107_polygon_rootcause_1video"
export PF_CHECKPOINT="$PWD/third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
```

Node 0:

```bash
NODE_RANK=0 nohup python scripts/run_v107_polygon_rootcause_1video.py all \
  > runs/v107_polygon_rootcause.node0.log 2>&1 &
```

Nodes 1-3 use the same command with their rank:

```bash
NODE_RANK=1 nohup python scripts/run_v107_polygon_rootcause_1video.py all \
  > runs/v107_polygon_rootcause.node1.log 2>&1 &
```

Only eight videos are needed, so the four-node run uses two GPUs per node.
The simpler alternative is one eight-GPU node:

```bash
NUM_NODES=1 NODE_RANK=0 \
python scripts/run_v107_polygon_rootcause_1video.py all
```

Do not reuse the v100 `OUT_ROOT`.

## 8. Required review

Before metrics, record for each video:

1. polygon noise: none / mild / severe;
2. subject count and duplication;
3. identity persistence through the final third;
4. motion amount and physical plausibility;
5. background/layout stability;
6. for A-B-A: B formation, A2 identity return, and transition artifacts.

Also inspect:

```text
contracts/all.json
configs/*.json
logs/*.log
traces/*.policy.jsonl
traces/*.motion.jsonl
traces/*.transition.jsonl
traces/*.scene.jsonl
diagnostics/*.json
status/*.json
```

The run is invalid if a completion marker is missing or any diagnostic has
`"ok": false`.

## 9. Causal decision

### Branch A: PF-AR clean, PF-AW noisy

Wave-to-stride routing is the isolated cause. Keep a conservative classifier
that leaves Wave-like heads on cyclic. Promote the middle-relative cell only
if it is also clean.

### Branch B: PF-AR and PF-AW both clean

The old 304/56 membership, rather than Wave-to-stride alone, has a harmful
interaction. Compare its real layer/head topology with the two controls; do
not claim a single PF class caused the failure.

### Branch C: PF-AR is noisy

A current shared binary implementation regression remains. Stop all candidate
and A-B-A promotion. Compare the v107 policy trace against the known-clean v99
trace before any new method experiment.

### Middle-relative gate

- clean: it remains an independent two-role classifier candidate;
- noisy while PF-AR is clean: classification is the problem; use PF-AR only
  as a quality control and redesign the independent score;
- map build fails: recover the original score artifacts; never substitute the
  old same-named CSV.

## 10. Provisional paper method

The paper method is not frozen until v107 review. The highest-feasibility
story is:

1. **Intervention-relative two-role profiling.** Measure each head's
   standardized preference for non-sink, non-recent middle history versus the
   recent tail under balanced uniform probe topologies. Use the natural zero
   boundary without PF labels.
2. **Role-conditioned dual-lifecycle cache.** History-supportive heads retain
   sparse long-horizon stride evidence. Recent-responsive heads retain
   phase-aligned cyclic evidence. One explicit owner composes sink, middle,
   and recent segments.
3. **Phase-preserving event augmentation.** Add a bounded clean-V change event
   slot without deleting the cyclic base. Keep this only if
   `cyclic4+motion1` beats or qualitatively improves `cyclic4`.
4. **Trust-conditioned writes.** Retain v78 only if its one-video and later
   multi-prompt controls improve stability.
5. **Role-aware scene episodes.** For prompt switching, archive only
   long-horizon stride state, reset responsive local state, and recall the
   prior episode on A return. This is a secondary extension.

Difference from PF:

- PF uses three head classes and stride/cyclic/merge routes.
- The proposed method uses an independently measured two-role axis, no merge,
  and a dual lifecycle with optional event augmentation.
- PF operators and the general concept of head-aware cache routing remain
  prior art and must be cited, not claimed as new.

Difference from Echo-Forcing:

- Echo-Forcing's scene pool/recall lifecycle is prior art.
- The proposed scene extension archives only role-specific long-horizon
  state, resets the responsive lifecycle, and is evaluated as a secondary
  A-B-A mechanism.

These claims survive only if the corresponding controls succeed. If the
middle-relative classifier does not beat random/threshold controls later, it
must be presented as analysis rather than a contribution.

## 11. After v107

Do not run 128 prompts before the visual gate.

If one single-prompt candidate is clean:

1. run it on 16 diverse 30-second prompts with PF-AR and PF native controls;
2. review blind and run fast DINO/temporal-jump diagnostics;
3. freeze exactly one method;
4. replace the invalid v101 map matrix with a new MovieGenVideoBench-128
   comparison and ablation runner;
5. reuse compatible existing SF/PF/EF videos by hash;
6. then run VBench-Long and comprehensive metrics.

The old v101 runner remains a historical reproduction only and must not be
used for the paper table.
