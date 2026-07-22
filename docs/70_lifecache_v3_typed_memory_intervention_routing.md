# LifeCache-v3: Typed Historical Memory and Intervention Routing

> Status: implementation candidate, not a validated result.
> Primary task: training-free 30s+ single-prompt AR video extrapolation.
> Secondary task: prompt/scene switching and scene return.
> Base model: Self-Forcing DMD on Wan2.1-T2V-1.3B.

## 1. Method decision

HREM-v2 used one bounded coverage archive and attempted to route recall with
hand-designed persistence evidence. Existing logs show that the evidence is
nearly uniform across heads, so it does not provide defensible classification.
LifeCache-v3 replaces both weak points:

1. History is explicitly divided by lifecycle into exact anchors and temporal
   summaries. Native recent K/V remains owned by Self-Forcing.
2. Head/layer selection is defined as measured historical-memory intervention
   utility, not as an assumed identity/layout/motion label.
3. Offline counterfactual generation supplies a model-level prior. Online
   evidence admits only safe interventions for the current query and block.
4. The memory branch never mutates the native working cache and must return the
   bitwise native output whenever retrieval or routing abstains.

Working title:

> **LifeCache: Half-Life-Aware Historical Memory for Training-Free Long Video Generation**

Central claim to test:

> Long-video history is not homogeneous. Exact appearance/state anchors,
> temporally aggregated state, and recent dynamics have different useful
> lifetimes. Separating their update rules and routing historical intervention
> by measured utility improves long-range consistency without freezing motion.

## 2. Explicit cache definition

### 2.1 Native recent cache

Owner: upstream Self-Forcing.

- Function: local motion, chunk transition, current appearance and detail.
- Source: clean-context K/V written by the normal SF autoregressive loop.
- Capacity: `local_attn_size=21` latent frames in the current config.
- Update: every generated block; oldest non-sink tokens are evicted by FIFO.
- Sink: current SF-DMD path uses `sink_size=0`.
- Recall: native causal attention; LifeCache does not rewrite this cache.

### 2.2 Exact appearance/state anchors

Owner: `TypedMemoryBank.anchors`.

- Function: retain non-averaged evidence for persistent subject appearance,
  object state and scene layout.
- Source: clean-context pre-RoPE K and V after spatial pooling.
- Default capacity: 4 frames globally for one continuous episode.
- Mandatory admission: the first valid frame of every episode is protected.
- Adaptive admission: a candidate must be at least 6 frames after the last
  anchor and have internal motion score at most 0.35.
- Candidate score:

  `anchor_score = 0.65 * temporal_stability + 0.35 * anchor_novelty`

- Replacement: replace the lowest-scoring unprotected anchor only when the new
  score exceeds it by `0.05`. A new episode can retire the oldest protected
  anchor if the global budget cannot represent both scopes.
- Stored payload: exact pooled K/V, one-frame interval, episode id, prompt
  descriptor, score, motion score and protection state.

The score is an update heuristic, not a claim that the selected frame is an
"identity frame". Its effect must be established by anchor-only ablation.

### 2.3 Temporal state summaries

Owner: `TypedMemoryBank.summaries`.

- Function: preserve medium-term state and suppress frame-specific high
  frequency variation under a fixed budget.
- Source: every clean pooled frame, grouped only within the same episode.
- Default capacity: 12 slots.
- Match: current V descriptor is compared with same-episode summary
  descriptors. Similarity at least 0.90 triggers a merge.
- Merge: same spatial coordinates of pre-RoPE K/V are updated by a running
  mean until the slot reaches 8 frames; the slot then freezes so its end frame
  can age beyond the native recent window.
- New state: a dissimilar frame creates a new slot while capacity remains.
- Full budget: coalesce the closest pair of existing same-episode
  summaries, then open a fresh slot for the current frame. If no legal pair
  exists, replace the globally oldest summary.
- Stored payload: aggregated K/V, `[start_frame,end_frame]`, episode id, prompt
  descriptor, merge count and mean motion score.

This is actual temporal compression, unlike HREM-v2's spatial pooling alone.
Because K/V averaging may blur fast-changing state, exact anchors are retained
in parallel and summary-only is a mandatory ablation.

### 2.4 Scene scope and motion trace

Scene scope is metadata rather than another full K/V bank:

- episode id and normalized prompt descriptor are stored for every slot;
- a scene boundary freezes the old episode scope without deleting its memory;
- intra-episode generation may read only the current episode;
- return recall must pass semantic/visual episode admission and may not fall
  back to the immediately previous scene.

Motion is also metadata rather than long-lived motion K/V:

`motion_score(f) = 1 - cosine(V_descriptor(f), V_descriptor(f-1))`

It is maintained per episode and subtracts a configurable bias from historical
retrieval. The purpose is to suppress stale dynamic evidence, not to claim an
optical-flow-equivalent motion representation.

### 2.5 Readout

The exported memory is `[anchors; summaries]`, with type and motion sidecars.
Current Q first selects top-k slots per head. Retrieval logits receive:

`score_bias = type_bias - motion_penalty * slot_motion`

Default values are anchor `+0.05`, summary `0.0`, and motion penalty `0.10`.
The selected K/V then enters the existing independent memory attention branch.
Recent 12 frames are excluded and default top-k is 4. The first calibration
keeps a deliberate 9-frame overlap with SF's 21-frame native cache so memory
can reinforce state before native eviction; a strict 21-frame exclusion is a
required ablation.

The target fusion gate is 0.15, beginning at frame 12 and linearly ramping for
12 frames. This replaces the old ineffective `gate=0.05,start=36` setting and
avoids a hard activation boundary. See `docs/72_lifecache_v3_post_review_optimization.md`.

## 3. Non-handcrafted head/layer routing

### 3.1 What is classified

We do not classify a head as identity, layout or motion unless an independent
causal analysis validates that semantic role. We estimate a narrower quantity:

> Expected utility and risk of applying historical-memory intervention at
> `(layer, head, denoising call, inference branch)`.

The output is a gate and an abstention reason, not a semantic class name.

### 3.2 Offline counterfactual profiling

The first profiling pass uses 16 cells:

- layer bands: `[0,8)`, `[8,15)`, `[15,22)`, `[22,30)`;
- head groups: `[0,3)`, `[3,6)`, `[6,9)`, `[9,12)`;
- same 12 prompts, seed and typed cache in every cell;
- only the selected layer/head group receives memory.

Every output is paired with native SF by prompt and seed. The profile builder
orients metric deltas, converts every metric to percentile ranks, and averages
the ranks with equal weight. Reliability is based on sample count and sign
consistency. No hand-authored head role is used.

After group screening, refine only positive bands into per-layer and per-head
groups, then optionally separate noisy attention calls. Do not run a 360-head
full factorial before the group screen.

### 3.3 Online intervention routing

For each memory readout, compute per head:

- Q stability against the preceding-block EMA;
- retrieval confidence and top-1/top-2 margin;
- normalized retrieval entropy;
- cosine alignment between native and memory attention outputs;
- candidate `delta_rms / native_rms`, reproducing RMS matching, confidence,
  acceptance and alignment weighting from the configured fusion mode.

Each signal is converted to a tie-aware within-layer percentile mid-rank, so
equal evidence cannot be separated by head index. Their equal-weight
mean forms online utility. Candidate effect receives an increasing rank inside
a bounded safe range; the previous decreasing rank could prefer no-op heads.
This avoids semantic thresholds and remains scale-invariant across layers. A
head is invalid when:

- retrieval already abstained;
- native-memory alignment is below 0;
- candidate intervention delta is below 0.005 of native RMS;
- candidate intervention delta exceeds 0.08 of native RMS.

Among valid heads, retain the top budget fraction. Default candidates are
25%, 50% and 75%. If utility spread is below 0.02, abstain instead of breaking
an uninformative tie. Online utility is smoothed with EMA 0.90.

### 3.4 Offline, online and hybrid modes

- `intervention_online`: current-block signals only.
- `intervention_offline`: counterfactual profile only, with online safety mask.
- `intervention_hybrid`: reliability-weighted offline utility plus online
  utility, followed by the same safety mask and budget.

The hybrid rule uses measured profile reliability. Missing profile entries
fall back to online utility rather than receiving an invented role.

### 3.5 Timestep and CFG scope

`attention_call_index` is recorded and supported by the offline profile, so
different denoising calls can receive different utilities. The first pass uses
all noisy calls; call-specific intervention is a refinement experiment.

The canonical SF-DMD pipeline performs conditional distilled inference and
does not expose a conditional/unconditional CFG pair. Therefore CFG routing is
not a claim of the main method. It can be evaluated only in the repository's
multi-step causal diffusion pipeline, where both branches actually exist.

## 4. Research basis and distinction

- [Forcing-KV](https://arxiv.org/abs/2605.09681) profiles AR video heads from
  frame-wise attention mass and validates static/dynamic roles by masking cache
  context. We adopt the principle that profiling must match the downstream
  intervention, but our target is quality utility of side-memory injection.
- [HALO](https://arxiv.org/abs/2607.11081) validates motion heads through
  cross-frame displacement/flow agreement and structure heads through attention
  entropy. This supports displacement and entropy as diagnostics, not arbitrary
  head indices. HALO is a motion-transfer method, not our implementation.
- [HASTE](https://arxiv.org/abs/2605.14513) uses query-key drift and
  error-guided per-head budget calibration for video-DiT sparse attention. It
  motivates drift and measured output error rather than one global threshold.
- [Sparse-vDiT](https://arxiv.org/abs/2506.03065) uses small-sample offline
  layer/head search and reports stable layer/head pattern dependence. It
  supports coarse-to-fine profiling.
- [MOFT](https://arxiv.org/abs/2405.14864) derives training-free motion features
  by removing content correlation and selecting motion channels. It motivates
  treating motion as a distinct risk signal.
- Pyramid-Forcing provides a strong `sink + middle + recent` cache precedent;
  Echo-Forcing provides scene-indexed snapshot lifecycle; IAMFlow provides
  entity-aware representative memory. They must be cited as precedents.

The typed slots alone are not sufficient novelty. The candidate contribution
is the combination of information-half-life memory with measured intervention
utility and fail-closed online routing, tested primarily on continuous
single-prompt extrapolation.

## 5. Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `STRUCTURED_MEMORY_ARCHIVE_POLICY` | `typed` in v3 scripts | Enable typed lifecycle |
| `STRUCTURED_MEMORY_GATE` | `0.15` | Target memory fusion gate |
| `STRUCTURED_MEMORY_MEMORY_START_FRAME` | `12` | Pre-emptive activation start |
| `STRUCTURED_MEMORY_ACTIVATION_RAMP_FRAMES` | `12` | Linear target-gate ramp |
| `STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES` | `16` | Hard total slot ceiling |
| `STRUCTURED_MEMORY_TYPED_ANCHOR_FRAMES` | `4` | Exact anchor capacity |
| `STRUCTURED_MEMORY_TYPED_SUMMARY_SLOTS` | `12` | Temporal summary capacity |
| `STRUCTURED_MEMORY_TYPED_ANCHOR_MIN_GAP_FRAMES` | `6` | Anchor temporal separation |
| `STRUCTURED_MEMORY_TYPED_ANCHOR_MOTION_CEILING` | `0.35` | Reject rapidly changing anchor candidates |
| `STRUCTURED_MEMORY_TYPED_ANCHOR_REPLACE_MARGIN` | `0.05` | Anchor replacement hysteresis |
| `STRUCTURED_MEMORY_TYPED_SUMMARY_MERGE_SIMILARITY` | `0.90` | Same-episode merge threshold |
| `STRUCTURED_MEMORY_TYPED_SUMMARY_COUNT_CAP` | `8` | Maximum averaging inertia |
| `STRUCTURED_MEMORY_TYPED_ANCHOR_BIAS` | `0.05` | Retrieval prior for exact anchors |
| `STRUCTURED_MEMORY_TYPED_MOTION_PENALTY` | `0.10` | Stale-motion retrieval penalty |
| `STRUCTURED_MEMORY_HEAD_ROUTING` | experiment-specific | `off`, `profile_group`, or intervention mode |
| `STRUCTURED_MEMORY_INTERVENTION_HEAD_BUDGET_FRACTION` | `0.50` | Top valid head fraction |
| `STRUCTURED_MEMORY_INTERVENTION_MIN_DELTA_TO_NATIVE` | `0.005` | Reject functionally ineffective heads |
| `STRUCTURED_MEMORY_INTERVENTION_MAX_DELTA_TO_NATIVE` | `0.08` | Per-head perturbation ceiling |
| `STRUCTURED_MEMORY_INTERVENTION_PROFILE_PATH` | empty | Offline profile JSON |

## 6. 16-GPU protocol

Every command uses 12 complex prompts and 120 latent frames. To prevent method
selection leakage, `screen`, `profile`, and `refine` default to the calibration
suite, while `baselines`, `hybrid`, and `confirm` default to a disjoint
evaluation suite. Outputs are isolated by phase. Run phases in this order.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free

# 4 seeds x SF/PF/Echo/current coverage = 16 jobs. Run once and freeze.
bash scripts/run_v69_typed_cache_16gpu.sh baselines

# One-seed strength, activation, cache composition and router screen.
bash scripts/run_v69_typed_cache_16gpu.sh screen

# 4 layer bands x 4 head groups counterfactual intervention.
bash scripts/run_v69_typed_cache_16gpu.sh profile

# After selecting a positive coarse band, profile every individual site.
# The default [15,22) range is 7 layers x 12 heads = 84 jobs in six waves.
REFINE_LAYER_START=15 REFINE_LAYER_END=22 \
  bash scripts/run_v69_typed_cache_16gpu.sh refine
```

Run comprehensive metrics for the calibration native cell from `screen` and
every `profile` cell. The coarse profile decides which layer bands enter
`refine`; it is not a final classifier. For the final JSON, use individual
`refine` rows in refined bands and coarse rows only in unrefined bands, so the
same observation is not counted twice. Assemble one CSV with these required
columns:

```text
cell,prompt_id,seed,layer_start,layer_end,head_start,head_end,
memory_mode,attention_call_index,dino,min_dino,arcface,motion,
vbench_subject,vbench_dynamic,loop,flicker,temporal_jump
```

Ranges in this CSV are inclusive. The launcher uses half-open ranges, so a
launcher cell with layers `[0,8)` and heads `[0,3)` must be written as
`layer_start=0,layer_end=7,head_start=0,head_end=2` in the CSV. Then build the
profile:

```bash
python scripts/build_intervention_profile.py \
  runs/v72_profile_12p_30s/intervention_metrics.csv \
  configs/lifecache_v3_intervention_profile.json \
  --baseline-cell sf_native

PROFILE_PATH="$PWD/configs/lifecache_v3_intervention_profile.json" \
  bash scripts/run_v69_typed_cache_16gpu.sh hybrid

# Four seeds for native, typed-all, online-25% and online-50%.
bash scripts/run_v69_typed_cache_16gpu.sh confirm
```

The disjoint suites are
`prompts/lifecache_v3_calibration_complex_12.txt` and
`prompts/lifecache_v3_single_long_complex_12.txt`. Existing 3-prompt results
remain valid pilot evidence but cannot be mixed numerically with either suite.
The fixed phases before refinement contain 80 GPU jobs and 960 generated
videos; the default seven-layer individual-site refinement adds 84 jobs and
1,008 calibration videos in six 16-GPU waves.

## 7. Required logs

For every typed commit, inspect:

- run commit, prompt path and prompt SHA-256 before comparing any cells;
- anchor/summary occupancy versus configured capacity;
- `add`, `merge_similar`, `coalesce_*_add`, `replace`, `skip_motion`,
  `skip_gap` and `skip_hysteresis` counts;
- slot intervals, merge counts and motion scores;
- whether every episode retains at least one anchor and summary.

For every readout, inspect:

- selected slot type, interval, age and episode;
- retrieval confidence, margin and entropy;
- online/offline utility per layer/head/call;
- Q stability, alignment, candidate delta/native, valid and selected masks;
- selected fraction, abstention reason and actual fused delta/native.

Failure signatures:

| Signature | Interpretation | Next action |
|---|---|---|
| summaries immediately collapse to one slot | merge threshold too low or descriptors uninformative | raise threshold; compare summary-only |
| anchors never replace | gap/motion/hysteresis too strict | inspect action counts before changing values |
| almost all heads valid and utilities tied | online evidence uninformative | rely on offline profile or stop head claim |
| selected heads improve DINO but motion/loop collapses | identity-motion tradeoff remains | reduce budget/gate; increase motion penalty |
| alignment mostly negative | stored K/V or readout convention is incompatible | stop quality sweep and audit representation |
| hybrid is not better than online/all-head | offline profile does not generalize | remove offline classifier from final method |

## 8. Feasibility and resource estimate

At the default pooled grid, one slot contains roughly `104 x 12 x 128`
elements per K or V tensor. In BF16, 16 slots across 30 layers require about
292 MiB for K+V before small sidecars. With per-head top-k=4, the union can
contain all 16 slots, or about 1,664 memory tokens per layer in the worst case.
The temporary attention logits are roughly 374 MiB in FP32 for a three-frame
query. This is modest relative to an H20, although the seven active layers
still add measurable attention compute.

The implementation is tensor-shape compatible and has CPU unit tests, but the
current workstation has neither PyTorch nor a GPU runtime. Therefore:

- static compilation here establishes syntax only;
- Stage `screen` establishes runtime feasibility;
- no method claim is promoted until videos, metrics and traces agree;
- a CUDA OOM, negative alignment or failed native-fallback invariant is a code
  failure, not a negative scientific result.

## 9. Promotion criteria

Promote typed memory only if it beats both coverage and its anchor-only and
summary-only ablations on identity/subject consistency without a material
motion, loop, flicker or human-preference regression.

Promote online routing only if it beats typed all-head across at least three
seeds and its selected heads are nontrivial but not near 0% or 100%.

Promote offline/hybrid routing only if the calibration profile generalizes to
the disjoint evaluation prompts and held-out seeds, and improves over online
routing. Otherwise retain only the
online safety router and report the profiling result as analysis.

If no routed variant beats typed all-head, the defensible paper direction is
typed historical memory with abstention, not head specialization.
