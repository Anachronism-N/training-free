# v100 Responsive Event Cache and A-B-A Fast Screen

Date: 2026-07-27

> **Historical experiment only:** docs/106 found polygon noise in every
> non-native v100 cell. The runner used the tracked old `304/56` map rather
> than the intended middle-relative map family. This runner is now disabled
> by default. Use docs/107 and
> `scripts/run_v107_polygon_rootcause_1video.py`.

This is the authoritative implementation and experiment document after the
v99 cache-ownership audit. It supersedes `docs/103` for the immediate
experiment. `docs/103` remains the record of the earlier middle-relative /
stride-cyclic proposal.

## 0. Current decision

The method is not frozen yet. The current **pre-result primary candidate** is:

**History-Polarity Event Cache (HP-Event)**

```text
offline two-role head profiling
+ History-Supportive long-horizon stride memory
+ Recent-Responsive motion-event/phase memory
+ exclusive sink + middle + recent ownership
+ optional role-aware scene episode archive/recall
```

The single-prompt 30-second extrapolation task remains primary. A-B-A
prompt/scene switching is the secondary task.

Important claim boundary:

- The tracked old-v98 map is reproducible and usable for a fast experiment.
- The corrected implementation has not yet established which Responsive
  cache is best.
- `motion2 + cyclic2` is a hypothesis, not a reported improvement.
- Scene episodes and the IDF threshold are also hypotheses until A-B-A videos
  form B correctly and restore A without artifacts.

## 1. Frozen two-role head map

The fast screen uses the tracked old-v98 zero-threshold map:

```text
runs/v98_history_polarity/maps/history_polarity_zero.csv
```

For head `h`, the historical score is:

```text
p_h = median_records(
        sum(history QK logits) / sum(abs(history QK logits))
      )

p_h >= 0  -> label 10, History-Supportive
p_h <  0  -> label 11, Recent-Responsive
```

This classifier does not use PF labels to compute the score, choose the zero
threshold, or assign a class quota. The runner fails unless its exact
statistics are:

| Role | Count |
|---|---:|
| History-Supportive | 304 |
| Recent-Responsive | 56 |

Post-hoc overlap with PF is:

| PF class | Supportive | Responsive |
|---|---:|---:|
| Anchor | 169 | 3 |
| Wave | 133 | 23 |
| Veil | 2 | 30 |

Therefore the map identifies the Anchor-like and Veil-like ends cleanly, while
Wave is mixed and mostly Supportive. It is not a three-way PF relabeling.

Scientific limitation: `p_h` uses the absolute sign of logits and is not
invariant to an additive logit shift. It is acceptable as a reproducible
candidate and causal control, but head discovery becomes a paper contribution
only if it beats count-matched random/inverted/threshold controls in the later
broad experiment.

## 2. Exact single-prompt cache

### 2.1 History-Supportive heads

```text
static sink:      first 3 frames
explicit middle: stride interval 6, capacity 4 frames
dynamic recent:  latest 4 frames
RoPE:             dynamic middle remapping
```

This route preserves sparse subject, layout, and background evidence over the
full generated horizon.

### 2.2 Recent-Responsive heads

The pre-result primary route is:

```text
static sink:      first 3 frames
explicit middle: 2 current-phase cyclic frames
               + 2 high-motion event frames
dynamic recent:  latest 4 frames
RoPE:             dynamic middle remapping
```

The sink is deliberately `sink3`, unlike PF Wave's `sink1`. This removes the
previous sink-size confound when comparing different Responsive middle
policies.

#### Motion-event selection

At each layer and committed clean block, let `R_l` be that layer's Responsive
heads. Sample at most 64 spatial tokens per frame and compute:

```text
d_l,t =
  mean_{h in R_l, x in sampled tokens} ||V_l,t,h,x - V_l,t-1,h,x||^2
  -----------------------------------------------------------------
  mean_{h in R_l, x in sampled tokens}
      (||V_l,t,h,x||^2 + ||V_l,t-1,h,x||^2) + epsilon
```

The highest-scoring frame in each three-frame clean block is admitted to a
two-frame FIFO event bank. Selection is:

- training-free;
- based only on the generated clean V trajectory;
- shared once per layer, not independently synchronized for every head;
- never computed from noisy diffusion iterations;
- reset at a scene boundary;
- written to a separate motion JSONL trace.

The two cyclic slots retain short periodic phase evidence. The two event slots
retain non-periodic motion changes. Their union has the same four-frame middle
read budget as the stride/cyclic controls, with time-id deduplication.

### 2.3 Why not immediately use Merge

The current Merge implementation is the same patch-2 temporal/spatial
primitive used by PF Veil, but the binary map sends 56 heads rather than PF's
32 Veil heads through it. Old polygon-noise cells also changed Wave routing
and previously contained cache-ownership bugs.

Thus the evidence does **not** prove that the Merge operator is intrinsically
broken. It does show that Merge is too risky for the immediate main candidate.
It remains a later diagnostic, not one of the first 16 fast cells.

### 2.4 Why cyclic remains a control

Plain phase cyclic is a strong historical control and may still win. However,
it is inherited from PF Wave and the old `sink1 + cyclic4` route changed both
the middle policy and sink budget. The fast screen separately tests:

```text
cyclic4 + sink1
cyclic4 + sink3
motion4 + sink3
motion2 + cyclic2 + sink3
recent8 + sink3
```

This factorization determines whether any gain comes from sink size, periodic
history, generated-motion events, or simply more recent frames.

## 3. Exclusive cache lifecycle

For labels 10/11, `HeadComposition` exclusively owns:

```text
static sink + explicit middle + dynamic recent
```

The legacy dynamic-history path is disabled. Runtime policy traces record:

- actual sink/middle/recent frame ids and token counts;
- head label, layer, branch, and policy type;
- middle strategy state;
- segment overlap and budget violations;
- whether the explicit composition is the dynamic owner.

No completion marker is written if the decoded-video or trace audit fails.
Native PF remains on its original lifecycle and is not silently converted to
the exclusive route.

## 4. Role-aware A-B-A scene episodes

The new scene mode is enabled only for prompts containing `A || B || A`.

### 4.1 Automatic scene matching

The implementation follows Echo-Forcing's scene-pool lesson but uses a
different cache payload. It builds an IDF-weighted prompt feature after
removing common transition words. For each new segment:

```text
best cosine >= 0.20 -> recall the matched canonical scene id
best cosine <  0.20 -> allocate a new scene id
```

For the three canonical prompts, the frozen lexical similarities are roughly:

| Prompt | A-B | A-A2 |
|---|---:|---:|
| Ceramic studio | 0.063 | 0.365 |
| Observatory | 0.018 | 0.392 |
| Noodle cart | 0.081 | 0.397 |

The automatic result should therefore be `[0, 1, 0]`. A manual `[0,1,0]`
cell is included only to distinguish matching failure from cache failure. It
cannot be used as the final automatic-method result.

### 4.2 Boundary operation

At a switch:

1. Cross-attention is invalidated, as in the existing segmented PF pipeline.
2. Each Supportive stride bank is materialized and archived under the current
   canonical scene id.
3. A matching stride bank is restored on A2; a new empty bank is opened for B.
4. Responsive cyclic/motion/merge/lag middle state is cleared to prevent
   previous-scene texture or motion leakage.
5. The recent cache keeps one frame as a soft bridge, or zero frames in the
   hard-control cell.
6. The static sink is retained as a global identity anchor.
7. Every action, similarity, scene id, restore count, and sampled layer state
   is written to the scene JSONL trace.

Scene banks are bounded to eight scenes. Restored stride anchors keep their
physical times and use dynamic RoPE at readout.

### 4.3 Relation to Echo-Forcing

Borrowed lesson from Echo-Forcing:

- explicit scene boundaries;
- scene-pool lifecycle;
- prompt-based previous-scene selection;
- local-cache reset and controlled old-memory reuse.

Different implementation:

- Echo can store/select/compress all-head scene snapshots and optionally reuse
  old recent memory with decay.
- HP-Event archives only Supportive heads' sparse stride anchors.
- Responsive heads never receive a recalled A snapshot; their local motion
  memory is reset and rebuilt from the current scene.
- The payload is a head-role-specific temporal cache, not a renamed Echo
  compressed snapshot.

The code and paper must cite Echo-Forcing for the scene-pool/recall precedent.

### 4.4 Known ABA risk

The global static sink may still make B too similar to A. The first comparison
is therefore:

```text
no episode vs bridge1 vs bridge0
```

If B fails even with `bridge0` while the scene trace is correct, the next
revision must make the Responsive sink scene-local or recapturable. Do not
misdiagnose that outcome as a head-classification failure.

## 5. Small add-ons

Only one add-on already has broad positive evidence:

- `v78 full_budget075_p1`: trust/novelty/age/budget-conditioned middle writes.

The fast screen also includes:

- Supportive `stride2 + cyclic2` hybrid, testing whether mixed Wave heads need
  some phase evidence.
- The combination of that hybrid with v78.
- A middle-layer variance-only V refresh derived from the earlier CEMR
  ablation.

The last two mechanisms are exploratory. They must be removed if one-video
review shows darkening, reduced motion, duplicates, or geometry artifacts.
Combining many weak tricks is not a valid contribution.

## 6. Frozen 16-cell fast screen

Every cell uses one prompt, seed 0, 120 latent frames, approximately 30
seconds, and one sample.

### 6.1 Responsive cache, six cells

| Cell | Purpose |
|---|---|
| `single_pf_native` | exact PF visual reference |
| `legacy_v98_stride_cyclic_sink1` | old Wave-like route |
| `legacy_v98_stride_cyclic_sink3` | isolate sink1 versus sink3 |
| `legacy_v98_stride_motion4` | event memory without periodic slots |
| `legacy_v98_stride_motion2_cyclic2` | pre-result primary candidate |
| `legacy_v98_stride_recent8` | token-budget-matched recent-only control |

### 6.2 Small tricks, four cells

| Cell | Purpose |
|---|---|
| `legacy_v98_motion2_cyclic2_v78` | validated write control on the new read cache |
| `legacy_v98_hybrid_motion2_cyclic2` | phase support for mixed Supportive/Wave heads |
| `legacy_v98_hybrid_motion2_cyclic2_v78` | combined candidate |
| `legacy_v98_motion2_cyclic2_variance_refresh` | exploratory CEMR-derived value correction |

### 6.3 A-B-A, six cells

| Cell | Purpose |
|---|---|
| `aba_pf_native` | native segmented PF |
| `aba_motion_no_episode` | same cache without scene archive |
| `aba_motion_episode_bridge1` | automatic role-aware episode |
| `aba_motion_episode_hard` | remove recent bridge |
| `aba_motion_episode_manual_bridge1` | matching oracle diagnostic only |
| `aba_cyclic_sink3_episode_bridge1` | isolate motion event value in ABA |

## 7. Server commands

The runner requires:

```text
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
```

Override `PF_CHECKPOINT` if the checkpoint is stored elsewhere.

### 7.1 Run all cells on four nodes

All nodes must share the same repository and `OUT_ROOT`. Start rank 0 first;
the other ranks wait for its immutable contract. Each rank receives four
cells, so four GPUs per node are sufficient.

Node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3 \
OUT_ROOT="$PWD/runs/v100_fast_selection_1video" \
nohup python scripts/run_v100_fast_selection_1video.py all \
  > runs/v100_fast_selection_1video.node0.log 2>&1 &
```

Nodes 1-3 use the same command with matching ranks:

```bash
NODE_RANK=1 NUM_NODES=4 GPU_LIST=0,1,2,3 \
OUT_ROOT="$PWD/runs/v100_fast_selection_1video" \
nohup python scripts/run_v100_fast_selection_1video.py all \
  > runs/v100_fast_selection_1video.node1.log 2>&1 &
```

Repeat for ranks 2 and 3.

### 7.2 Run the three stages on one eight-GPU node

```bash
GPU_LIST=0,1,2,3,4,5 \
OUT_ROOT="$PWD/runs/v100_fast_selection_1video" \
python scripts/run_v100_fast_selection_1video.py responsive

GPU_LIST=0,1,2,3 \
OUT_ROOT="$PWD/runs/v100_fast_selection_1video" \
python scripts/run_v100_fast_selection_1video.py tricks

GPU_LIST=0,1,2,3,4,5 \
OUT_ROOT="$PWD/runs/v100_fast_selection_1video" \
python scripts/run_v100_fast_selection_1video.py aba
```

Use a fresh `OUT_ROOT` after any code, config, prompt, map, or model change.
Partial videos without a matching completion marker are rejected.

## 8. Output and debug contract

```text
runs/v100_fast_selection_1video/
|-- contracts/
|-- configs/
|-- diagnostics/
|-- logs/
|-- status/
|-- traces/
`-- videos/
```

Required log markers:

```text
[PyramidKVRuntimePolicy]
[HistoryPolarityPolicy]        # every non-native binary cell
[PyramidKVMotionEvent]         # every motion cell
[SceneCacheConfig]             # every episode cell
[SceneCacheSwitch]             # three records per episode video
```

Inspect these files first:

```text
diagnostics/<cell>.video.json
diagnostics/<cell>.policy.json
diagnostics/<cell>.motion.json
diagnostics/<cell>.scene.json
traces/<cell>.policy.jsonl
traces/<cell>.motion.jsonl
traces/<cell>.scene.jsonl
status/<cell>.done.json
```

For motion traces verify:

- `responsive_head_count` equals that layer's label-11 count;
- `all_scores` is finite and non-negative;
- `selected_frame_ids = frame_start_t + selected_offsets`;
- selection occurs only once per clean block;
- scene switches reset the previous-frame motion reference.

For scene traces verify:

- canonical ids are `[0,1,0]`;
- B is `new`, A2 is `recall`;
- A2 has `restore_scene > 0`;
- Responsive strategies report `clear_local`;
- bridge1 keeps fewer recent tokens than before the boundary;
- no policy trace reports segment overlap or budget violations.

## 9. Human review

### 9.1 Single-prompt review

Reject a cell immediately for:

- polygon/grid noise;
- duplicated subjects or body parts;
- severe exposure/color shift;
- frozen or near-static motion;
- abrupt geometry collapse.

Then compare:

- face/clothing/object identity in early, middle, and late thirds;
- background layout and unique object persistence;
- natural camera and object motion;
- loops, flashbacks, speed jumps, and first visible failure time;
- overall preference versus PF.

### 9.2 A-B-A review

Score three separate outcomes:

1. **B formation:** B must be visibly different from A.
2. **A return:** A2 must recover A identity, layout, and distinctive objects.
3. **Transition quality:** no flash, polygon noise, duplicate subject, or long
   frozen bridge.

The method fails if A2 similarity is high only because B never formed.
Use `scripts/evaluate_aba_return.py` after the manual review, but retain the
three separate human scores.

## 10. Fast-screen decision tree

1. If any route has polygon noise, first inspect its policy/ownership trace.
   A correct trace plus corruption is a mechanism failure, not a routing bug.
2. If `cyclic_sink3` fixes `cyclic_sink1`, the old result was partly a sink
   confound.
3. If `motion2_cyclic2` is at least visually tied with cyclic while preserving
   more non-periodic motion, promote it for the paper candidate.
4. If `motion4` freezes or drifts but the hybrid does not, retain the periodic
   safety slots.
5. If `recent8` matches all middle policies, the old head routing is not
   causally justified.
6. If v78 helps the selected read cache, retain it as a second component.
7. If automatic ABA fails but manual ABA works, fix scene matching.
8. If both automatic and manual ABA fail with correct traces, fix the cache
   payload or sink lifecycle.
9. If bridge1 prevents B but bridge0 forms B, add a bounded decay/eviction
   bridge rather than restoring all old recent memory.

## 11. Deferred broad experiments and ablations

Do not launch these until the 16 videos are reviewed.

### Main comparison

Run selected methods on MovieBench-128:

```text
SF native
SF + PF
SF + selected two-role cache
SF + selected two-role cache + v78, if promoted
```

All videos are 30 seconds. Compute VBench-Long, comprehensive DINO/CLIP,
temporal jump, dynamic degree, and per-prompt paired deltas.

### Head-classification ablation

Use at least MovieBench-32:

```text
old-v98 zero map
PF (Anchor+Wave)/Veil oracle binary map
PF Anchor/(Wave+Veil) oracle binary map
layer-wise count-matched random
inverted map
nearby fixed thresholds
```

### Cache-mechanism ablation

Under one frozen map and equal read budget:

```text
remove Supportive stride
motion4
cyclic4
motion2 + cyclic2
recent8
remove recent
sink1 versus sink3
disable dynamic RoPE
```

### Scene ablation

```text
no scene operation
cross-attention reset only
archive without Responsive reset
reset without Supportive archive
automatic versus manual matching
bridge0 versus bridge1
scene-local Responsive sink, only if the first screen requires it
```

## 12. Potential paper story

Subject to positive broad results:

> Long autoregressive video attention contains two deployment-level temporal
> roles rather than requiring a fixed three-class taxonomy. History-Supportive
> heads preserve sparse global evidence, while Recent-Responsive heads need a
> bounded mixture of local phase and generated-motion events. We introduce an
> auditable, training-free dual-horizon cache that assigns these memories
> independently per head. The same role decomposition enables scene episodes:
> only long-horizon heads recall prior-scene anchors, while Responsive heads
> reset local motion memory, allowing both scene formation and identity-aware
> return.

Potential contributions:

1. A PF-independent two-role history-polarity head analysis.
2. A fixed-budget Responsive event/phase cache derived online from clean V
   changes.
3. An exclusive, trace-verifiable heterogeneous cache lifecycle.
4. A role-aware scene episode mechanism that recalls global anchors without
   replaying stale local motion.
5. Optional trust-conditioned writes, only if v78 improves the selected base.

Do not claim contribution 1 if random/inversion controls fail. Do not claim
contribution 2 if cyclic or recent-only is clearly better. Do not claim
contribution 4 if B does not form or manual recall is required.

## 13. Provenance and academic-integrity boundary

- **Pyramid Forcing:** source of the PF base, temporal head specialization
  precedent, and stride/cyclic/merge primitives.
- **Echo-Forcing:** source of the explicit scene-pool and prompt-recall
  precedent used to motivate the A-B-A lifecycle.
- **Earlier project v78:** source of trust-conditioned cache writes.
- **Earlier project CEMR:** source of the exploratory variance-only refresh.
- **Flash-VStream/Flash-VAR-style compression notes:** not used by the current
  primary candidate; they remain future alternatives if token compression is
  reintroduced.
- **IAMFlow/entity memory:** related motivation for identity/state semantics,
  not code used by this implementation.

The original repositories and license/provenance ledger are in `README.md`
and `docs/64_related_work_code_provenance_and_claims.md`. Borrowed mechanisms
must be cited by their original names. Our claim, if supported, is the
two-role criterion, generated-motion event memory, role-specific composition,
and role-aware episode lifecycle, not ownership of prior cache primitives.

## 14. Broad follow-up

Do not expand every v100 cell to 128 prompts. After manual review selects the
Supportive route, Responsive route, and v78 setting, freeze those choices and
use `docs/105_v101_paper_ablation_after_fast_screen.md`.

The corresponding entry points are:

```text
scripts/run_v101_paper_ablation_4node.py
scripts/postprocess_v101_paper_ablation.sh
```

They cover MovieGenVideoBench-128 full evaluation, cache/update ablations,
random and PF-AW membership controls, threshold stability, blind review,
VBench-Long, comprehensive metrics, and temporal-jump diagnostics.
