# v99 Binary Cache Recovery, One-Prompt Gate, and Paper Story

Date: 2026-07-26

Status: code complete for server-side smoke testing. GPU execution and human
review are pending.

## 1. What v98 did and did not prove

The v98 videos and metrics remain valid records of those exact runs, but they
do **not** prove that binary head roles are intrinsically ineffective.

The negative result mixed three changes:

1. the head map changed;
2. Wave and Veil were replaced by a new hybrid/merge cache;
3. explicit `HeadComposition` middle memory and PF's legacy dynamic-history
   compaction could both remain active for the same Supportive head.

The third item violates the intended cache definition. A head could read:

```text
static sink
+ explicit stride/hybrid middle
+ legacy stride-compressed dynamic history
+ recent frames
```

instead of exactly:

```text
static sink + one explicit middle policy + recent frames
```

Additional correctness issues were found:

- only physical frame 0 was excluded from middle memory even for `sink3`;
- the first update could retain more than the declared recent window;
- an explicit recent-only composition could fall through to legacy anchors;
- a rejected transition could leave a partial Merge block that was later
  summarized as if it were complete;
- old traces did not report the physical contents of sink and dynamic cache.

The near-parity of `pf_explicit_parity` and `pf_native` does not clear these
binary routes. It exercises PF's three classes, not neutral-label binary
ownership.

Historical evidence also contradicts a universal binary failure: the earlier
`pf_binary_read` / `pf_binary_read_v78` stride-cyclic route produced usable
videos and only a moderate metric loss relative to PF. Those videos should be
reused as references, not regenerated.

### 1.1 What the v97 PF-class ablation actually showed

v97 kept two PF classes unchanged and replaced only the target class's middle
cache with a roughly token-matched longer recent window:

| Target | Native middle | Replacement | DINO | Loop |
|---|---|---|---:|---:|
| PF native | all three native routes | none | 0.9308 | 0.3067 |
| Anchor | stride4 | no middle + recent8 | 0.9204 | 0.1730 |
| Wave | cyclic4 | no middle + recent8 | 0.9242 | 0.2676 |
| Veil | merge4 | no middle + recent5 | 0.9284 | 0.2855 |

Anchor stride is clearly the most important of the three for long-range
continuity. The raw DINO/Loop drops rank Wave second and Veil smallest:
`-0.0066/-0.0391` for Wave versus `-0.0024/-0.0212` for Veil. Therefore the
paper must not claim that Wave has the smallest marginal contribution. It may
claim only that Wave is substantially less critical than Anchor and remains a
plausible class to merge.

These are useful directional results, not final causal evidence. The v97 runs
predate the exclusive-owner audit and assigned methods to fixed machines.
The new `causal32` suite runs every method on every node shard and records the
physical cache contents.

## 2. Corrected cache definition

v99 restores the quality-tested binary topology before testing new compression.

| Role | Function | Exclusive cache |
|---|---|---|
| Supportive / Persistent | preserve identity, layout, and long-range state | `sink3 + stride(interval=6, cap=4) + recent4` |
| Responsive / Phase-local | preserve pose, motion phase, and local change | `sink1 + cyclic(period=6, cap=4) + recent4` |

This PF-AR control is deliberately conservative. PF contains 172 Anchor, 156
Wave, and 32 Veil heads. Under Anchor-vs-Rest plus stride/cyclic, the declared
cache of all 172 Anchor heads remains Anchor-like and all 156 Wave heads
remain Wave-like; only the 32 Veil heads change role. Thus 328/360 declared
routes are preserved, apart from the explicit ownership cleanup being tested.
By contrast, v98 PF-AW hybrid/merge changed the pure stride/cyclic route of
328 heads into a hybrid route. Its collapse is therefore evidence against
that aggressive cache reassignment, not against every binary partition.

The primitives are inherited from PF and must be attributed to PF. The
corrected implementation contribution is the explicit role-to-memory contract:

- `HeadComposition` is the only owner of middle history;
- dynamic cache contains recent frames only;
- sink, middle, and recent physical frame sets cannot overlap;
- each segment has a declared frame/token budget;
- the native PF path keeps its prior behavior for baseline compatibility;
- neutral labels `10/11` cannot trigger hidden PF `-1/1/2` branches.

The switch is `pyramidkv_composition_owns_dynamic`. Its YAML default is
`false`; `--pyramidkv_history_polarity` sets it to `true`.

## 3. One-prompt smoke gate

Do not rerun SF, PF, or the earlier binary reference. Point the runner to
their existing video directories. Each smoke invocation audits and reuses
them, then generates exactly one new 30-second video:

1. `pf_ar_neutral_stride_cyclic`: PF Anchor-vs-Rest membership encoded as
   neutral labels. This isolates cache implementation and label semantics.
2. `pf_aw_neutral_stride_merge`: PF `(Anchor + Wave) | Veil` oracle
   membership with the repaired Anchor/Veil cache. This isolates membership.
3. `history_polarity_stride_cyclic`: the independent v98 binary classifier
   with exactly the same corrected cache.
4. `history_polarity_stride_merge_fixed`: the exact repaired v98 hypothesis:
   Supportive heads use PF's Anchor cache and Suppressive heads use PF's Veil
   cache.
5. `history_polarity_random_stride_merge`: a layer-wise count-matched random
   map under the same repaired stride/merge cache.

The fourth cell is not a new untested idea. The old v98
`history_polarity_stride_merge` cell used the same declared assignment:

```text
Supportive:  sink3 + stride(interval=6, cap=4) + recent4
Suppressive: sink3 + merge(patch=2, block=4, cap=4) + recent4
```

At the natural zero threshold, the frozen v98 map contains 304 Supportive and
56 Suppressive heads. It maps 169/172 PF Anchor heads to Supportive and 30/32
PF Veil heads to Suppressive. PF Wave is unresolved: 133/156 map to
Supportive and 23/156 to Suppressive. Thus the classifier nearly separates
Anchor from Veil, but all 156 Wave heads lose PF's cyclic route.

The old cell produced DINO 0.7281 and temporal jump 3.414 and was visually
unusable. That result was generated before the exclusive-owner, sink
exclusion, and first-update budget fixes, so it records the old implementation
but does not settle the repaired hypothesis.

First run the implementation-control video on one GPU:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

NODE_RANK=0 \
GPU_LIST=0 \
PROMPT_INDEX=0 \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v99_binary_cache_recovery_smoke1" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py smoke1 \
  > runs/v99_binary_cache_recovery_smoke1.nohup.log 2>&1 &
```

Only after that video passes human review, run the independent classifier in
a fresh output directory:

```bash
NODE_RANK=0 \
GPU_LIST=0 \
PROMPT_INDEX=0 \
SMOKE_CELL=history-polarity \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v99_history_polarity_smoke1" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py smoke1 \
  > runs/v99_history_polarity_smoke1.nohup.log 2>&1 &
```

To test the proposed Anchor/Veil routing directly, generate one repaired
stride/merge video in another fresh directory:

```bash
NODE_RANK=0 \
GPU_LIST=0 \
PROMPT_INDEX=0 \
SMOKE_CELL=history-polarity-stride-merge \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v99_history_polarity_stride_merge_smoke1" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py smoke1 \
  > runs/v99_history_polarity_stride_merge_smoke1.nohup.log 2>&1 &
```

The two remaining causal controls use the same command with
`SMOKE_CELL=pf-aw-stride-merge` and
`SMOKE_CELL=history-polarity-random-stride-merge`, each with a distinct
`OUT_ROOT`.

`REUSE_PF_BINARY_DIR` may instead point to a completed non-v78
`pf_binary_read` directory. Record which reference is used. `REUSE_SF_DIR` is
optional because SF is not needed to diagnose this cache bug.

Use a fresh `OUT_ROOT` after any source, map, prompt, or model change. The
runner requires a clean committed checkout and freezes all relevant hashes.

## 4. Required log and trace evidence

Every new run must contain:

```text
[HistoryPolarityPolicy] ... legacy_pf_labels=false exclusive_owner=true
[PyramidKVRuntimePolicy] ... exclusive_dynamic=True
```

Each sampled `middle_selection` event records:

- `composition_present=true`;
- `dynamic_policy_owner=composition_recent`;
- `explicit_composition_owns_dynamic=true`;
- physical `sink_frame_ids`, `recent_frame_ids`, and middle frame ids;
- segment token/frame counts;
- middle-sink and middle-recent overlap;
- `cache_contract_violations`;
- `cache_contract_pass`.

The runner fails before writing a completion marker if dynamic history exceeds
`recent4`, any segment overlaps, a wrong policy is constructed, or owner
metadata is missing.

## 5. Human review decision tree

Review prompt 0 side by side in this order:

1. reused `pf_native`;
2. reused `pf_binary_read` reference;
3. new `pf_ar_neutral_stride_cyclic`;
4. new `pf_aw_neutral_stride_merge`;
5. new `history_polarity_stride_merge_fixed`;
6. new `history_polarity_stride_cyclic`;
7. new `history_polarity_random_stride_merge`.

Check the first, middle, and final thirds for polygon noise, identity drift,
background replacement, frozen/repeated motion, abrupt jumps, and motion
plausibility.

Decision:

- If `pf_ar_neutral_stride_cyclic` has polygon noise, stop. The neutral-label
  cache route still has an implementation mismatch; do not run 32/128 prompts.
- If PF-AR neutral is visually close to the old binary reference, the cache
  repair passes. Compare the independent classifier next.
- If PF-AR passes but history-polarity fails, the remaining problem is the
  classifier, not cache mechanics.
- If stride/cyclic passes but repaired stride/merge fails, the unresolved Wave
  heads require a phase-local cache; do not blame the Anchor/Veil separation.
- If repaired stride/merge is usable, compare it directly with stride/cyclic
  on one or two more hard prompts before any 32-prompt expansion.
- If the PF-AW oracle works but the v98 map fails, classification membership
  is the cause. If both fail similarly, loss of the Wave cyclic route or a
  remaining stride/merge implementation defect is the cause.
- The count-matched random cell must be worse than the v98 map before head
  membership is claimed as a contribution.
- If both pass, run `screen32`; only after human and metric gates pass should
  `main128` be launched.
- Merge remains a candidate, not the recovery default, until this one-video
  gate passes.

## 6. Larger experiments after smoke passes

The same runner supports:

```bash
python scripts/run_v99_binary_cache_recovery_4node_32gpu.py causal32
python scripts/run_v99_binary_cache_recovery_4node_32gpu.py screen32
python scripts/run_v99_binary_cache_recovery_4node_32gpu.py main128
```

For these modes, launch node ranks 0-3. `causal32` needs four GPUs per node
and runs exactly the four generated controls below; PF native is audited and
reused as the fifth comparison. `screen32` and `main128` use eight GPUs per
node. Every node runs every selected method on a disjoint prompt shard, which
removes method/node confounding.

Example `causal32` launch for node 0; replace both rank occurrences on the
other three nodes:

```bash
NODE_RANK=0 \
GPU_LIST=0,1,2,3 \
SCORE_ROOT="$PWD/runs/v98_middle_relative_scores" \
REUSE_PF_DIR="$PWD/runs/v98_history_polarity_screen32/pf_native" \
REUSE_PF_BINARY_DIR="$PWD/runs/v93_moviebench128_main/pf_binary_read_v78" \
OUT_ROOT="$PWD/runs/v99_binary_cache_causal32" \
nohup python scripts/run_v99_binary_cache_recovery_4node_32gpu.py causal32 \
  > runs/v99_binary_cache_causal32.node0.log 2>&1 &
```

| Causal comparison | Map | Supportive route | Suppressive route |
|---|---|---|---|
| `pf_aw_neutral_stride_merge` | PF-AW oracle | Anchor stride | Veil merge |
| `history_polarity_stride_merge_fixed` | v98 304/56 | Anchor stride | Veil merge |
| `history_polarity_stride_cyclic` | v98 304/56 | Anchor stride | Wave cyclic |
| `history_polarity_random_stride_merge` | layer-wise random 304/56 | Anchor stride | Veil merge |

The table's vertical bar in the PF map denotes a binary split, not a third
runtime class.

The larger matrix adds random, inverted, and threshold controls, plus a fixed
Merge ablation and optional v78 write admission. These are causal controls,
not candidates to run before the one-prompt gate.

## 7. Paper story if the corrected route works

The defensible story has three levels.

### Contribution 1: evidence-based binary functional roles

Instead of adopting PF's three sign/temporal-pattern labels, classify heads
using a shift-invariant middle-versus-recent QK margin under counterfactual
prompt probes. A natural threshold partitions heads into Supportive and
Responsive roles. Random, inverted, and threshold sweeps test whether the
membership itself is causal.

This contribution is claimable only if the independent map beats its random
and inverted controls and remains reasonably stable across prompts/seeds.

### Contribution 2: role-conditioned exclusive dual memory

Couple each role to a distinct temporal function:

- Supportive heads retain sparse long-range identity/layout evidence;
- Responsive heads retain bounded phase-local motion/change evidence.

The novelty is not the isolated stride or cyclic operator. It is the
independently discovered binary role criterion, its coupling to two functional
memory horizons, and an exclusive lifecycle that prevents hidden cache paths.

### Contribution 3: lifecycle control for long extrapolation and switching

After the read path is stable, evaluate trust-conditioned middle writes and
prompt-switch invalidation as separate modules. Single-prompt 30-second
generation remains the primary task. ABA/prompt-switch generation is a
secondary stress test: Responsive memory should adapt quickly, while
Supportive memory should preserve reusable identity and layout state.

Do not include v78 or ABA as a headline contribution unless controlled
experiments show a gain without new hallucinations.

## 8. Difference from prior work

- **Pyramid Forcing** supplies the validated stride/cyclic/merge primitives
  and its three-class baseline. Our candidate uses a different two-role
  criterion and an explicit two-memory ownership contract.
- **Echo Forcing and related snapshot caches** motivate controlled historical
  refresh. We do not claim their snapshot/update mechanism as ours.
- **Forcing-KV / Head Forcing** are closely related binary or role-aware memory
  works and must be discussed directly. Distinction must rest on the exact
  role statistic, role membership, cache topology, and controlled evidence,
  not renaming.
- **Flash-VAR/AV token-compression notes** remain possible later compression
  ablations. They should replace the middle representation only after the
  stride-cyclic parity gate, rather than being mixed into the recovery run.

If only PF-derived Anchor-vs-Rest works and the independent classifier does
not, the result is an engineering regrouping of PF rather than a strong new
classification contribution. That version may still be useful, but it is not
enough by itself for a top-conference claim.
