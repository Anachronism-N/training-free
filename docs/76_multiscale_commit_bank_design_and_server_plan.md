# Commit Forcing v2: multiscale commit bank and server plan

> Date: 2026-07-23
> Depends on: `docs/74_commit_forcing_research_reset.md` and
> `docs/75_commit_forcing_v74_screen_results.md`
> Status: code complete and statically compiled; GPU tests pending

## 1. Decision from the v74 result

The previous screen establishes four facts:

1. pathwise correction has a visible effect and delays collapse;
2. reliability-gated dynamic commits outperform fixed-origin correction;
3. two origin states improve identity most, but increase temporal jumps;
4. repeated hard correction still causes style simplification, motion freezing,
   and acceleration-like jumps.

The next iteration therefore keeps the validated pathwise mechanism and changes
two weak points:

- replace FIFO trusted history with an explicit `origin + compressed + recent`
  commit lifecycle;
- couple re-noising to the current trajectory instead of drawing unrelated
  fresh noise at every correction.

The provisional method name is:

> **Commit Forcing: Reliability-Gated Multiscale State Commit for
> Training-Free Long Video Extrapolation**

This is a working title, not a novelty or SOTA claim.

## 2. What was borrowed, and what was not

| Work | Idea-level influence used here | What is not copied or claimed |
|---|---|---|
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | AR backbone, native recent KV cache, generation loop | The backbone is explicitly third-party code. |
| [Pathwise TTC](https://github.com/xbxsxp9/Pathwise_TTC) | Fixed-reference pathwise test-time correction is the closest prior | No source was copied. Pathwise correction itself is not our novelty. |
| [Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) | Long-term, compressed, and recent memory should have different lifecycle roles | No Echo cache updater, top-C selector, scene snapshot code, or decay code is ported. |
| [Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | Cyclic/stride/merge policies support the need for multiple temporal scales | No offline head labels or per-head PF policies are used in this iteration. |
| [DeepForcing](https://github.com/cvlab-kaist/DeepForcing) | Stable sinks plus bounded compression are useful long-video baselines | No query-participative compression implementation is ported. |
| [RollingForcing](https://github.com/TencentARC/RollingForcing) | Exact anchors and rolling recent context are strong primitives | No RollingForcing cache code is ported. |
| [MotionCache](https://github.com/MAC-AutoML/MotionCache) | Motion is a necessary cache diagnostic and control signal | MotionCache mainly targets denoising reuse. We independently use adjacent clean-latent change for commit compatibility, not its reuse algorithm. |
| [FlowCache](https://github.com/mikeallen39/FlowCache) | Feature change can control cache decisions | Output/residual reuse is not integrated because it is an efficiency mechanism and would confound quality attribution. |
| [Future Forcing](https://arxiv.org/abs/2605.30083) | Important collision for cache eviction and merge | No future-query prediction or future-aware eviction is used. |

The cache tuple `anchor + compressed + recent` is established prior art and
must not be presented as the contribution. The candidate contribution is the
combination of:

1. denoising-reliability-gated state promotion;
2. reliability-weighted, motion-compatible temporal consolidation;
3. motion-adaptive summary readout;
4. trajectory-coupled path correction.

Every item remains a hypothesis until the planned ablations pass.

### 2.1 Deferred mechanisms

The following mechanisms are deliberately not on the immediate quality path:

- IAMFlow entity/VLM memory: useful for prompt switching and narrative state,
  but adds external-model latency and weakens the single-prompt causal story.
- MIGA long-range frame guidance: potentially strong, but changes the
  generation paradigm and has higher integration and attribution cost.
- FreePCA low-rank compression: may reduce memory, but compression efficiency
  is not the current bottleneck and averaging can worsen style shift.
- LongLive-RAG offload/retrieval: valuable beyond the GPU-resident horizon,
  but unnecessary for this 30-second mechanism screen.
- MemRoPE or FreeLOC position correction: important if diagnostics show
  position OOD; current correction uses a bounded adjacent re-mapping and v74
  did not isolate position error as the limiting factor.
- SWIFT semantic/head-wise injection and PF static head routing: both have
  high claim overlap. They return only if reproducible counterfactual evidence
  shows that all-head state correction is the remaining bottleneck.

This ordering favors changes with a short implementation path, an isolated
ablation, and a plausible effect on the three observed v74 failures.

## 3. Exact cache definition

The native Self-Forcing recent KV cache remains unchanged. Commit Forcing owns
a separate bounded bank used only for extra correction forwards.

### 3.1 Origin anchors

- Function: immutable identity and appearance bootstrap.
- Acquisition: the first `origin_capacity` clean frames in each episode.
- Update: never replaced within an episode.
- Reset: released at an explicit episode/scene boundary.
- Default read: one origin; a two-origin cell tests the identity upper bound.

### 3.2 Exact recent commits

- Function: retain current pose, local motion phase, and recent appearance.
- Admission: frame reliability must exceed `admission_reliability`.
- Spacing: accepted frames must satisfy `trusted_min_gap`.
- Update: FIFO within the exact recent sub-budget.
- Eviction: in multiscale mode, the oldest exact recent state is promoted into
  the summary hierarchy instead of being discarded.

The environment variable remains `COMMIT_FORCING_TRUSTED_USE` for backward
compatibility, while multiscale traces name this role `recent`.

### 3.3 Multiscale compressed summaries

Each summary stores:

```text
pre-RoPE K/V payload
temporal span [start, end]
representative frame id
support count
scale level
reliability
instability
normalized motion
merge method
```

Old recent states enter level 0. Two nodes at the same level merge and carry to
the next level, producing approximately dyadic temporal coverage. If the
summary count still exceeds its budget, the two oldest nodes are consolidated.

For node \(i\), the merge weight is

\[
w_i = n_i \max(r_i, \epsilon)^p,
\]

where \(n_i\) is support, \(r_i\) is reliability, and the default \(p=2\).

Two nodes are motion-compatible when

\[
|\log(1+m_i)-\log(1+m_j)| \leq \tau_m
\]

and neither is in the high-motion regime. The default adaptive merge is:

- compatible: reliability-weighted K/V mean;
- incompatible or high-motion: keep the higher-weight real representative
  payload while merging only its metadata span.

The representative fallback is deliberate. It avoids averaging incompatible
poses into an off-manifold state and gives a direct `mean` versus
`representative` ablation.

### 3.4 Motion estimate

For adjacent clean latent frames \(z_t,z_{t-1}\):

\[
m_t =
\frac{\mathrm{RMS}(z_t-z_{t-1})}
{0.5(\mathrm{RMS}(z_t)+\mathrm{RMS}(z_{t-1}))+\epsilon}
+0.25(1-\cos(z_t,z_{t-1})).
\]

Motion is normalized by an online EMA scale. This is a training-free latent
proxy, not optical flow and not a semantic action label.

When the previous block's normalized motion exceeds `motion_high_ratio`,
compressed summaries are omitted from the active correction view. Origin and
exact recent states remain available. This rule is intended to preserve motion
freedom without abandoning identity constraints.

### 3.5 Readout

The default active view is:

```text
one origin + highest-scale eligible summary + newest exact recent
```

Selected pre-RoPE keys are remapped into adjacent legal positions immediately
before the current block. Values are unchanged. This is a temporary correction
cache; it does not overwrite the native rolling cache.

## 4. Trajectory-coupled re-noising

The v74 implementation re-noised every reference prediction with independent
fresh Gaussian noise. That changes both the clean estimate and stochastic
component, which is a plausible source of acceleration-like jumps.

During native few-step sampling, the current noisy state was generated as

\[
x_t = \mathrm{AddNoise}(\hat{x}^{native}_0, \epsilon_t, t).
\]

The new `trajectory` mode reuses the same \(\epsilon_t\):

\[
x'_t = \mathrm{AddNoise}(\hat{x}^{ref}_0, \epsilon_t, t).
\]

Thus the correction changes the estimated clean path while preserving the
current stochastic realization. `fresh` mode remains available to reproduce
v74 exactly. A missing or shape-mismatched trajectory noise falls back to
fresh noise and emits a `renoise_fallback` trace event.

## 5. Code and controls

Primary files:

- `src/lifecycle_kv/commit_forcing.py`
- `third_party/Self-Forcing/pipeline/causal_inference.py`
- `scripts/summarize_commit_forcing_trace.py`
- `tests/test_commit_forcing.py`
- `scripts/run_v76_multiscale_commit_16gpu.sh`

New controls:

```text
COMMIT_FORCING_BANK_MODE=fifo|multiscale
COMMIT_FORCING_SUMMARY_CAPACITY=0
COMMIT_FORCING_SUMMARY_USE=0
COMMIT_FORCING_SUMMARY_MERGE_MODE=adaptive|mean|representative
COMMIT_FORCING_SUMMARY_RELIABILITY_POWER=2
COMMIT_FORCING_MERGE_MOTION_TOLERANCE=.75
COMMIT_FORCING_MOTION_GATE=0|1
COMMIT_FORCING_MOTION_HIGH_RATIO=1.35
COMMIT_FORCING_MOTION_EMA_DECAY=.90
COMMIT_FORCING_RENOISE_MODE=fresh|trajectory
```

Backward-compatible v74 behavior is:

```text
BANK_MODE=fifo
SUMMARY_CAPACITY=0
SUMMARY_USE=0
MOTION_GATE=0
RENOISE_MODE=fresh
```

## 6. Models and paths

Required:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

For official PF and Echo comparison:

```text
third_party/Pyramid-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Echo-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Echo-Forcing/checkpoints/self_forcing_dmd.pt
```

The default conda environment remains `longlive`. Override paths with
`SF_REPO`, `PF_REPO`, `ECHO_REPO`, and the corresponding checkpoint/config
variables instead of editing the script.

## 7. Server commands

Pull and run syntax/tests first:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pytest tests/test_commit_forcing.py -q
python -m compileall -q src scripts third_party/Self-Forcing/pipeline
```

Four-GPU smoke:

```bash
GPU_LIST=0,1,2,3 \
bash scripts/run_v76_multiscale_commit_16gpu.sh smoke
```

Full 16-GPU screen, 12 complex prompts, 120 latent frames, seed 0:

```bash
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v76_multiscale_commit_16gpu.sh screen
```

Baselines only:

```bash
GPU_LIST=0,1,2 \
bash scripts/run_v76_multiscale_commit_16gpu.sh baselines
```

To rerun an existing output directory:

```bash
FORCE=1 OUT_ROOT="$PWD/runs/v76_multiscale_commit_screen_rerun" \
bash scripts/run_v76_multiscale_commit_16gpu.sh screen
```

Create the blind review after trace checks pass:

```bash
RUN_ROOT="$PWD/runs/v76_multiscale_commit_screen"
python scripts/prepare_blind_review.py \
  --run-root "$RUN_ROOT" \
  --methods sf_native pf_official echo_pc \
    v74_hybrid_fresh v74_origin2_fresh \
    fifo_hybrid_trajectory fifo_origin2_trajectory \
    ms_fresh_nomotion ms_trajectory_nomotion ms_full_motion \
    ms_origin2_motion ms_summary2_motion ms_representative_motion \
    ms_mean_motion ms_no_summary_read ms_t250_motion \
  --prompts prompts/lifecache_v3_single_long_complex_12.txt \
  --output "$RUN_ROOT/blind_review" --prompt-count 12 --seed 7601
```

Freeze `blind_review/scorecard.csv` before metrics, then reuse the generic v74
postprocessor by overriding its run root:

```bash
HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 GPU=0 \
RUN_ROOT="$PWD/runs/v76_multiscale_commit_screen" \
bash scripts/v74_postprocess.sh
```

## 8. Sixteen-cell screen

| GPU | Cell | Question |
|---:|---|---|
| 0 | `sf_native` | Native lower baseline |
| 1 | `pf_official` | Strong cache-policy baseline |
| 2 | `echo_pc` | Hierarchical scene/cache baseline |
| 3 | `v74_hybrid_fresh` | Exact v74 main control |
| 4 | `v74_origin2_fresh` | Exact v74 identity upper bound |
| 5 | `fifo_hybrid_trajectory` | Re-noise effect only |
| 6 | `fifo_origin2_trajectory` | Re-noise effect with two origins |
| 7 | `ms_fresh_nomotion` | Multiscale bank effect under old re-noise |
| 8 | `ms_trajectory_nomotion` | Bank plus trajectory re-noise |
| 9 | `ms_full_motion` | Proposed default |
| 10 | `ms_origin2_motion` | More identity anchors |
| 11 | `ms_summary2_motion` | Read two scales; tests over-constraint |
| 12 | `ms_representative_motion` | Never average K/V |
| 13 | `ms_mean_motion` | Always average K/V |
| 14 | `ms_no_summary_read` | Summaries maintained but not read |
| 15 | `ms_t250_motion` | Single low-noise correction for jump control |

Do not choose a default from DINO alone. Human motion/style review and temporal
jump are co-primary for this screen.

## 9. Required trace checks

The trace summarizer now reports:

- selected reference kind, span, level, support, distance, and motion;
- per-block denoising reliability and normalized motion;
- summary merge counts by `mean` or `representative`;
- motion-gated selection count;
- correction magnitude and reference/native disagreement;
- `fresh` versus `trajectory` re-noise counts;
- bank capacity violations and fallback events.

Inspect:

```bash
cat runs/v76_multiscale_commit_screen/diagnostics/commit_trace_summary.md
grep -R '"event": "renoise_fallback"' \
  runs/v76_multiscale_commit_screen/traces
grep -R '"event": "summary_merge"' \
  runs/v76_multiscale_commit_screen/traces | head
grep -R '\[CommitForcing\]\[motion\]\|\[CommitForcing\]\[merge\]' \
  runs/v76_multiscale_commit_screen/logs | head -n 100
```

Expected:

1. trajectory cells have no re-noise fallback at t500/t250;
2. summary support grows above 1 after early blocks;
3. adaptive mode records meaningful merge decisions;
4. motion gating activates sometimes, but not for every correction;
5. no bank exceeds configured capacity;
6. no summary spans a scene boundary.

If motion gating is never active, lower `motion_high_ratio` only in a new
labeled run. If it is always active, raise it. Do not modify thresholds inside
an existing result directory.

## 10. Evaluation and predeclared gates

For each cell compute:

- DINO mean, minimum, and temporal drift;
- temporal jump mean/median;
- flicker and background consistency;
- VBench-Long, especially subject consistency and dynamic degree;
- a latent or optical-flow motion-amplitude curve;
- first visible identity failure time;
- first style simplification time;
- freeze duration and number of acceleration jumps.

Predeclared decisions:

### Gate A: trajectory re-noising

`fifo_hybrid_trajectory` versus `v74_hybrid_fresh`:

- temporal jump improves by at least 10%, or human acceleration jumps clearly
  decrease;
- DINO does not drop by more than 0.005;
- visible degradation does not begin earlier.

### Gate B: multiscale history

`ms_trajectory_nomotion` versus `fifo_hybrid_trajectory`:

- minimum DINO or late identity improves meaningfully;
- style simplification and freeze do not worsen;
- summary selections occur with support greater than 1.

### Gate C: motion gate

`ms_full_motion` versus `ms_trajectory_nomotion`:

- motion range or dynamic degree improves;
- identity loss is at most 0.01 DINO;
- motion gating is neither always off nor always on.

### Gate D: adaptive consolidation

Compare `adaptive`, `mean`, and `representative`:

- adaptive must beat both fixed choices on the identity-motion trade-off;
- otherwise keep the better fixed strategy and remove the unsupported
  adaptive novelty claim.

Only cells passing human review proceed to four seeds and full VBench-Long.

## 11. Failure interpretation

- Trajectory mode reduces jumps but not freeze: correction strength or
  correction timing, not stochastic phase, is the next target.
- Summary mean worsens style: K/V averaging is off-manifold; use
  representative-only temporal compression.
- Summary read improves identity but freezes motion: reduce summary read
  frequency or use it only at t500, while keeping recent at t250.
- Motion gate improves motion but loses identity: keep one lower-scale exact
  milestone instead of fully abstaining from summary.
- No multiscale cell beats FIFO: retain reliability-gated commit plus
  trajectory re-noise and report multiscale cache as a negative ablation.
- PF or Echo remains clearly better: treat it as the quality target and inspect
  which verified component explains the gap before adding more machinery.

No paper claim should be changed until logs, videos, metrics, and multi-seed
evidence agree.
