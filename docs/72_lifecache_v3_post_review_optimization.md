# LifeCache-v3 Post-Review Optimization

> Status: code-complete experiment candidate, not a validated quality claim.
> Evidence source: `docs/71_human_review_and_code_alignment.md`.
> Primary task: training-free 30-second single-prompt extrapolation.
> Trace method version: `3.1`.

## 1. What the first human review actually establishes

The first 3-prompt run provides four useful observations:

1. Native SF starts losing identity or illumination stability after roughly
   five seconds and can become unusable after twenty seconds. Long-term state
   loss is therefore observable in the target setting.
2. Official PF retains identity and background much better, proving that cache
   composition can materially improve this base model without training.
3. PF also shows local speed discontinuities, early flashback artifacts, and a
   high-motion duplication failure. Long-term retention alone is insufficient;
   temporal intervention smoothness is a separate objective.
4. The old ours cells are functionally inconclusive. Their target gate was
   `0.05`, activation started at frame 36, and the old role gate selected almost
   every head. They do not test whether typed memory works at useful strength.

The review contains no evidence that changing `ROLE_THRESHOLD` will rescue the
method. LifeCache-v3 therefore keeps the old role classifier only as historical
ablation code and does not tune or claim it.

## 2. Method correction

### 2.1 Smooth pre-emptive activation

The v3 target memory gate is now `0.15`, activation begins at latent frame 12,
and the gate ramps linearly for 12 latent frames:

```text
activation_scale(t) = clip((t - start + query_frames) / ramp_frames, 0, 1)
effective_gate(t) = target_gate * activation_scale(t) * other_safe_scales
```

With three query frames per block, the first accepted intervention uses scale
`0.25`, then reaches the target over four blocks. This gives the memory branch
time to stabilize identity before the 21-frame native window fully forgets the
beginning, without a hard cache-policy transition like the jump suspected in
PF. `typed_g015_hard_on` is retained as the causal ramp ablation.

The default recent exclusion remains 12 frames during calibration. This creates
a deliberate 9-frame overlap with the 21-frame native window during the
pre-emptive phase. `typed_g015_nonoverlap` and `online_recent21` test whether
the overlap is helpful reinforcement or harmful double counting.

### 2.2 Effect-aware intervention band

The old online utility rewarded smaller candidate perturbations. Combined with
gate `0.05`, this could select heads that are safe precisely because they do
nothing. The router now requires:

```text
0.005 <= candidate_delta_rms / native_rms <= 0.08
native_memory_alignment >= 0
retrieval accepted
```

Within this safe band, a larger candidate effect receives a higher percentile
rank. The remaining rank signals are retrieval confidence, top-1/top-2 margin,
low entropy, Q stability, and native-memory alignment. The maximum remains a
safety ceiling; `0.15` is a target gate, not permission for an unbounded output
change. All ranks use tie-aware mid-ranks: identical head evidence remains
identical and triggers low-spread abstention instead of being ordered by index.
`online_no_effect_floor` directly ablates the new lower bound.

Offline counterfactual quality remains the strongest classifier. Online effect
strength only prevents a degenerate no-op and cannot by itself establish that a
head represents identity, layout, or motion.

### 2.3 Cache definition is unchanged

- Native recent: upstream 21-frame SF FIFO, responsible for local dynamics.
- Exact anchors: 4 non-temporally-averaged pooled clean K/V slots.
- Temporal summaries: 12 same-episode running-mean K/V slots with freeze and
  coalescing under budget.
- Motion trace: retrieval penalty metadata, not long-lived motion K/V.
- Readout: per-head Q-K top-k followed by an independent memory attention
  branch; recalled tensors never enter the native cache.

The new change is when and where that memory is allowed to have a measurable
effect, not a silent redefinition of cache contents.

## 3. Revised 16-GPU screen

`screen` uses the 12 calibration prompts and one fixed seed:

| GPU | Cell | Question |
|---:|---|---|
| 0 | `sf_native` | Native reference |
| 1 | `coverage_legacy_g005_s36` | Reproduce old weak coverage setting |
| 2 | `typed_legacy_g005_s36` | Is typed memory also invisible under old strength? |
| 3 | `typed_g010_r12` | Low target gate |
| 4 | `typed_g015_r12` | Default smooth typed memory |
| 5 | `typed_g020_r12` | High target gate and motion risk |
| 6 | `typed_g015_hard_on` | Does abrupt activation create a jump? |
| 7 | `typed_g015_nonoverlap` | Strictly exclude the full native window |
| 8 | `anchor_only_g015` | Exact-state contribution |
| 9 | `summary_only_g015` | Aggregated-state contribution |
| 10 | `online_b25_g015` | Sparse routed intervention |
| 11 | `online_b50_g015` | Default routed intervention |
| 12 | `online_b75_g015` | Dense routed intervention |
| 13 | `online_no_effect_floor` | Minimum-effect ablation |
| 14 | `online_recent21` | Native-overlap ablation under routing |
| 15 | `online_no_motion_penalty` | Motion-risk metadata ablation |

Run:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
bash scripts/run_v69_typed_cache_16gpu.sh screen
```

Do not proceed to coarse layer/head profiling until at least one typed all-head
cell is visibly different from native and its median fused delta/native is
above `0.003`. Otherwise the correct conclusion is still "intervention did not
reach the model", not "typed memory failed".

## 4. Temporal jump diagnostic

The PF review describes a local change-speed jump not covered well by average
identity metrics. Run the lightweight paired diagnostic on native, PF, and all
screen outputs:

```bash
python scripts/compute_temporal_jump_diagnostic.py \
  runs/v72_screen_12p_30s \
  runs/v72_baselines_12p_30s \
  --output runs/temporal_jump_diagnostic.csv
```

The script reports frame appearance deltas, Farneback flow speed, flow
acceleration, robust outlier fractions, and a `temporal_jump` score. It is a
debug diagnostic only. Paper evaluation must still use blind human review,
VBench-Long, min-DINO/identity metrics, and a stronger flow/perceptual metric.
`temporal_jump` is accepted by the intervention-profile builder as a
lower-is-better optional column.

## 5. Required server return package

For each run, return:

1. `run_manifest.env` and every cell's `run_config.env`;
2. generation logs including the pre-run GPU memory snapshot;
3. every JSONL trace and generated `_diagnosis.json`;
4. all MP4s or the blind-review scorecard;
5. paired min-DINO, DINO, motion, flicker, loop and temporal-jump metrics.

The launcher now fails the phase if any ours cell creates `run_config.env` but
does not create a non-empty trace. This closes the silent trace-loss failure in
the first run. Echo should run only on a GPU with sufficient free memory; its
log now records total, used and free GPU memory before inference.

Inspect these fields before video-level conclusions:

- `base_gate`, `activation_ramp_scale`, `effective_gate`;
- `delta_to_native_rms`, `effective_weight_mean`, alignment and confidence;
- typed-cache actions, occupancy, slot ages, types and motion scores;
- intervention `valid`, `selected`, candidate delta and abstention reason;
- prompt SHA-256 and commit hash.

## 6. Decision rules

1. If `typed_g015_r12` remains visually native and fused delta/native is below
   `0.003`, debug the bridge; do not tune cache semantics.
2. If `typed_g020_r12` improves identity but degrades motion, keep `0.15` and
   increase selectivity rather than increasing gate further.
3. If hard-on has a higher jump score than smooth activation at similar
   identity, retain the ramp as a supported component.
4. If non-overlap beats overlap, change the default recent exclusion to 21;
   otherwise describe overlap as pre-emptive state reinforcement.
5. Promote effect-aware routing only if `online_b25/b50` beats typed all-head
   without collapsing selected fraction to zero.
6. PF remains the required strong baseline. Its identity advantage cannot be
   attributed to our method, and any claimed advantage must include motion and
   temporal-smoothness evidence.

The resulting paper story is narrower and more defensible: typed memory models
different information lifetimes, while effect-aware smooth intervention targets
the two observed failure modes of native long-term forgetting and aggressive
heterogeneous-cache jumps. Whether it actually improves either remains an
experimental question until the revised screen is reviewed.
