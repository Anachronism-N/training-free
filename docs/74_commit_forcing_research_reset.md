# Commit Forcing: research reset, implementation, and server protocol

> Date: 2026-07-23
> Code base audited at: `a8d5104`
> Status: implemented prototype; static compilation passed; GPU behavior and
> quality are not yet validated.

## 1. Decision after the v72 result

`docs/73_lifecache_v3_screen_results.md` changes the research decision:

- all 16 cells and 192 videos completed;
- the side-memory path was active and its effect increased monotonically with
  the fusion gate;
- the best automatic improvement was small, around one DINO point;
- human review found every LifeCache-v3 variant visually equivalent to native
  SF, with degradation near 5 seconds and collapse near 15 seconds;
- online layer/head routing did not outperform all-head intervention;
- ramped activation was worse than hard activation.

This is not a hyperparameter-only failure. The pooled historical K/V branch
contributed roughly 5-7% of a convex attention output, so it could perturb a
metric without changing the generated trajectory enough to prevent collapse.
Increasing the number of slot types, hand-labeling heads, or sweeping another
small gate range would not address that mechanism.

The previous LifeCache/HREM code remains useful as negative evidence and
ablation infrastructure, but it is no longer the recommended paper method.

## 2. New hypothesis

The new working title is:

> **Commit Forcing: Reliability-Gated State Commit for Training-Free Long
> Video Extrapolation**

The central hypothesis is:

> Long autoregressive video generation fails not only because old states are
> evicted, but because every newly generated state is unconditionally promoted
> into future context. Once an unreliable state is committed, subsequent
> blocks condition on its error and can amplify it. A long-video method should
> decide whether a state is trustworthy before using it as a long-term
> reference, then apply trusted references strongly enough to change the
> sampling path.

This reframes the question from:

```text
Which old cache tokens should the current query retrieve?
```

to:

```text
Which generated states are reliable enough to commit?
When current generation needs correction, which committed state should
define a reference-conditioned denoising path?
```

The primary task remains a single complex prompt extrapolated to at least 30
seconds. Prompt/scene switching is secondary and uses a new bank episode at a
declared boundary.

## 3. Method

### 3.1 Diffusion-path reliability

SF already predicts the clean latent multiple times for one autoregressive
block. Let `x_hat[s, f]` be the predicted clean latent for frame `f` at
denoising step `s`. No additional model call is needed to collect these
predictions.

For adjacent predictions, the implementation computes:

```text
relative_rms[s,f] =
    RMS(x_hat[s,f] - x_hat[s-1,f])
    / (0.5 * (RMS(x_hat[s,f]) + RMS(x_hat[s-1,f])) + eps)

cosine_distance[s,f] =
    1 - cosine(flatten(x_hat[s,f]), flatten(x_hat[s-1,f]))

d[s,f] = relative_rms[s,f] + 0.25 * cosine_distance[s,f]
```

Later denoising transitions receive larger linear weights:

```text
D[f] = sum_s w[s] * d[s,f] / sum_s w[s],  w[s] = 1, 2, ...
```

The scale is an online EMA of block-average instability:

```text
scale <- 0.90 * scale + 0.10 * mean_f(D[f])
r[f] = exp(-D[f] / max(scale, 1e-4))
```

`r[f]` is a relative online reliability score, not a calibrated probability
and not a claim of semantic correctness. Its purpose is to reject states whose
denoising trajectory is unusually unstable compared with the current video.

### 3.2 Explicit cache composition

The P0 design uses three components:

| Component | Function | Acquisition | Update | Budget |
|---|---|---|---|---:|
| Origin bank | Identity/layout bootstrap | First clean-context frame(s) of the current episode | Immutable until episode switch | 1 by default |
| Trusted bank | Reliable non-recent state | Clean pre-RoPE K and V of frames with `r >= tau` | Minimum gap, bounded FIFO replacement | 3 by default |
| Native recent cache | Short-term motion and local continuity | Unmodified SF rolling K/V | Unmodified SF update/eviction | Native 21-frame window |

The reference bank stores a complete spatial frame for every transformer layer.
It does **not** pool tokens, merge states, or write recalled tokens into the
native cache. This is deliberate: the first experiment must test state
admission and path correction without repeating the lossy pooled-memory design
that failed in v72.

Default update rules:

```text
reference_capacity = 4 frames
origin_capacity = 1 frame
trusted_capacity = reference_capacity - origin_capacity
trusted admission threshold = 0.30
minimum trusted-frame gap = 3 latent frames
trusted replacement = FIFO
scene/prompt boundary = release the old bank and bootstrap a new origin
```

At four-byte-equivalent K+V accounting, full-resolution references are
expensive. With BF16, 30 layers, 12 heads, head dimension 128, and 1560 tokens
per frame, four references are approximately 1.1 GiB before allocator
overhead. The bank is therefore strictly bounded and old episode payloads are
released.

### 3.3 Reference selection

Three modes are implemented:

- `origin`: only immutable origin frames; this is the fixed-reference
  Pathwise TTC-style baseline.
- `trusted`: only the newest admitted trusted frames.
- `hybrid`: origin frames followed by newest trusted frames; this is the
  proposed default.

The default active reference contains one origin and one trusted frame. Stored
frames are re-RoPE'd into adjacent legal temporal positions immediately before
the current block. Their historical identity is retained in the bank metadata,
while the attention path sees a compact in-distribution temporal layout.

### 3.4 Pathwise reference correction

For selected **nominal** denoising timesteps, default `{500, 250}`, the
algorithm performs:

```text
1. y_ref = G(x_t, reference_cache, text)
2. eps_ref = deterministic local random noise
3. x'_t = scheduler.add_noise(y_ref, eps_ref, actual_scheduler_t)
4. y_current = G(x'_t, native_recent_cache, text)
5. continue the normal SF denoising schedule from y_current
```

The first pass uses the reference bank as the model's main self-attention
context. The second pass returns to the untouched native recent cache. This is
not a small side-output fusion: the reference prediction is re-noised and
becomes the input to the normal denoising path.

With four native denoising steps and corrections at two steps, the method adds
two transformer forwards per generated block, approximately 1.5x denoising
forward count. The clean-context update pass is unchanged.

SF's scheduler warps nominal denoising steps. Configuration and selection use
the nominal values, while the actual mapped scheduler value is used for model
inference and re-noising. Both are logged. This distinction is required;
matching configured `500/250` against warped values can silently produce zero
corrections.

### 3.5 Optional trigger

`always` applies corrections at every configured step after `start_frame`.
This is the P0 mechanism test.

`unreliable` applies them only when the previous block's mean reliability is
below `trigger_reliability`. This may reduce compute, but it must not become the
default until the always-on mechanism has a visible benefit and reliability is
shown to predict future failure.

## 4. Relation to prior work and novelty boundary

### 4.1 Direct base and closest prior

- [Self-Forcing](https://arxiv.org/abs/2506.08009) is the backbone and current
  code base.
- [Pathwise Test-Time Correction](https://arxiv.org/abs/2602.05871) is the
  closest prior for reference-conditioned denoise, re-noise, and normal-context
  denoise. Its public
  [repository](https://github.com/xbxsxp9/Pathwise_TTC) currently provides
  project material but no implementation used here. The correction path in
  this repository is independently implemented from the paper description.

The fixed `origin` cells must be described as a reproduction/reimplementation
of the prior mechanism, not our contribution.

The candidate contribution is the combination of:

```text
free denoising-trajectory reliability evidence
  -> explicit state admission before long-term commit
  -> bounded origin + trusted reference lifecycle
  -> state-dependent reference-conditioned path correction
```

This contribution is supported only if `hybrid` reliably beats both native SF
and fixed `origin` correction.

### 4.2 Distinction from cache-policy work

- [Pyramid Forcing](https://arxiv.org/abs/2605.13111) classifies attention
  heads offline and assigns head-specific sink/middle/recent cache policies.
  Commit Forcing leaves the native recent cache and all heads unchanged; its
  selection object is a generated state, its evidence comes from the diffusion
  trajectory, and its intervention is an additional pathwise denoising pass.
- [Deep Forcing](https://arxiv.org/abs/2512.05081) uses deep sinks and
  participative cache compression. Commit Forcing does not claim sink or
  compression novelty.
- [Echo-Forcing](https://arxiv.org/abs/2605.16003) uses hierarchical anchors,
  compressed history, and scene recall. Commit Forcing currently clears its
  bank at a scene boundary and does not claim scene retrieval.
- [Future Forcing](https://arxiv.org/abs/2605.30083) uses future-query proxies
  for cache eviction/merge. Commit Forcing does not claim query stationarity,
  future-aware eviction, or cache merging.

### 4.3 Head/layer/timestep policy

Manual head or layer classes are removed from P0. The v72 online router did not
beat all-head intervention, while hand-assigned semantic classes would be hard
to defend. Adding head labels now would increase collision with PF and
Forcing-KV without experimental support.

The only current role split is diffusion-native timestep selection, tested by
`750`, `500`, and `250` ablations. Layer/head/CFG specialization can be
reintroduced only after:

1. the full-model correction path has a visible effect;
2. measured per-layer or per-head counterfactual evidence is repeatable across
   prompts and seeds;
3. selective intervention beats the full-model correction at matched compute.

## 5. Implementation map

| File | Purpose |
|---|---|
| `src/lifecycle_kv/commit_forcing.py` | Config validation, reliability, bank lifecycle, reference selection/cache reconstruction, deterministic correction noise, JSONL trace |
| `third_party/Self-Forcing/wan/modules/causal_model.py` | Capture clean pre-RoPE K when Commit Forcing is enabled |
| `third_party/Self-Forcing/pipeline/causal_inference.py` | Nominal/actual timestep handling, extra reference forward, re-noise, reliability observation, clean-state commit |
| `tests/test_commit_forcing.py` | CPU tests for reliability, admission, cache reconstruction, episode release, and deterministic noise |
| `scripts/summarize_commit_forcing_trace.py` | Strict trace checks and mechanism summary |
| `scripts/run_v74_commit_forcing_16gpu.sh` | Smoke, 16-cell screen, and four-seed confirmation |
| `scripts/v74_postprocess.sh` | Post-review trace, comprehensive, jump, and VBench-Long evaluation |

All new paths are disabled by default. Native SF is unchanged unless
`COMMIT_FORCING_ENABLE=1`.

Commit Forcing is mutually exclusive with LifeCache and Structured Memory in
the current P0 implementation. Mixing them before independent validation would
make attribution impossible.

## 6. Model files

Required for native SF, fixed-origin correction, and the proposed hybrid:

```text
third_party/Self-Forcing/
|-- wan_models/
|   `-- Wan2.1-T2V-1.3B/
`-- checkpoints/
    `-- self_forcing_dmd.pt
```

Download from the original model sources:

```bash
cd third_party/Self-Forcing
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt --local-dir .
```

The `confirm` mode also runs official PF and expects:

```text
third_party/Pyramid-Forcing/
|-- wan_models/
|   `-- Wan2.1-T2V-1.3B/
`-- checkpoints/
    `-- self_forcing_dmd.pt
```

Symlinks are acceptable to avoid duplicate weights, provided each repository's
relative paths resolve.

## 7. Server commands

### 7.1 Mandatory smoke test

Run this before allocating all 16 H20 GPUs:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
SMOKE_FRAMES=12 \
GPU_LIST=0,1,2 \
bash scripts/run_v74_commit_forcing_16gpu.sh smoke
```

Smoke mode uses `prompts/smoke_identity_motion.txt`, one prompt, and 12 latent
frames. It is a shape/trigger test, not a quality result.

Then inspect:

```bash
cat runs/v74_commit_smoke/diagnostics/commit_trace_summary.md
grep -E 'CommitForcing|Traceback|CUDA|error' \
  runs/v74_commit_smoke/logs/*.log
```

Do not start the screen if an always-on cell has no correction event, selected
references are empty, the nominal and actual schedule is missing, or the
strict analyzer reports `failed`.

### 7.2 Main 16-GPU screen

```bash
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v74_commit_forcing_16gpu.sh screen
```

Every cell uses the same 12 complex single prompts, seed 0, and 120 latent
frames (about 30 seconds):

| GPU | Cell | Question |
|---:|---|---|
| 0 | `sf_native` | Native failure reference |
| 1 | `origin_t500` | Fixed origin at mid-low noise |
| 2 | `origin_t250` | Fixed origin at low noise |
| 3 | `origin_t500_250` | Closest fixed-reference TTC baseline |
| 4 | `origin_t750_500_250` | More correction versus motion loss |
| 5 | `hybrid_t500` | Dynamic commit at one step |
| 6 | `hybrid_t250` | Dynamic commit at low noise |
| 7 | `hybrid_t500_250` | Proposed default |
| 8 | `hybrid_t750_500_250` | Stronger dynamic correction |
| 9 | `trusted_t500_250` | Is origin necessary? |
| 10 | `hybrid_origin2` | More immutable identity state |
| 11 | `hybrid_trusted2` | More evolving state |
| 12 | `hybrid_start21` | Later activation |
| 13 | `hybrid_admit045` | Strict commit threshold |
| 14 | `hybrid_admit015` | Permissive commit threshold |
| 15 | `hybrid_unreliable045` | Reliability-triggered compute |

### 7.3 Blind review before metrics

First review the mechanism-defining subset:

```bash
python scripts/prepare_blind_review.py \
  --run-root runs/v74_commit_screen_12p_30s \
  --methods sf_native origin_t500_250 hybrid_t500_250 \
            hybrid_t750_500_250 trusted_t500_250 \
  --prompts prompts/lifecache_v3_single_long_complex_12.txt \
  --prompt-count 12 \
  --output runs/v74_commit_screen_12p_30s/blind_review
```

Freeze `blind_review/scorecard.csv` before revealing `key_private.json`.
Record identity, background, motion, camera continuity, artifacts, prompt
alignment, failure time, and overall rank. Prompt 0, 2, and 10 are high-priority
diagnostics because prior SF failure was strong, but all 12 prompts must remain
in the reported aggregate.

### 7.4 Metrics

After the blind scores are frozen:

```bash
HUMAN_REVIEW_DONE=1 RUN_VBENCH=1 GPU=0 \
bash scripts/v74_postprocess.sh
```

The script produces:

- strict Commit Forcing trace summaries;
- DINO consistency and drift;
- RAFT motion smoothness;
- ArcFace identity where a face is measurable;
- LPIPS flicker, CLIP alignment, background consistency, and loop evidence;
- temporal jump diagnostics;
- six VBench-Long dimensions: subject/background consistency, aesthetic and
  imaging quality, motion smoothness, and dynamic degree.

### 7.5 Four-seed confirmation

Do not run this until one dynamic cell visibly beats native and fixed origin:

```bash
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v74_commit_forcing_16gpu.sh confirm
```

This uses four seeds and four methods per seed:

```text
native SF
official PF
fixed origin correction
hybrid reliability-gated commit
```

## 8. Debug trace and review checklist

Each enabled run writes `traces/<cell>.jsonl`.

### 8.1 Expected events

| Event | Required fields | Interpretation |
|---|---|---|
| `video_start` | complete config | Reproducibility and expected trigger mode |
| `episode_start` | old/new episode and evicted frames | Bounded scene lifecycle |
| `block_reliability` | nominal timesteps, per-frame instability/reliability, scale | Admission evidence |
| `commit_accepted` | frame, kind, score, bank frames | Actual reference contents |
| `commit_rejected` | frame, score, reason | Threshold/gap/capacity behavior |
| `commit_evicted` | old frame and reason | FIFO enforcement |
| `correction` | nominal/actual timestep, reference ids/kinds, delta/input | Actual noisy-input replacement magnitude |
| `correction_outcome` | reference/native relative RMS and cosine | How strongly native context changes the reference prediction |

### 8.2 Hard invariants

The server run is invalid if:

- an `always` cell has zero correction events;
- selected references are empty or not older than the current block;
- `origin` mode selects a trusted frame, or `trusted` mode selects an origin;
- bank size exceeds the configured capacity;
- reliability is outside `[0,1]` or any trace value is non-finite;
- median `correction_delta_rms / input_rms < 1e-4`;
- nominal `500/250` is absent because warped timesteps were matched by mistake;
- LifeCache, Structured Memory, legacy Head Role, full-attention overrides, or
  scene-reset interventions are active in the same cell.

### 8.3 Mechanism diagnostics

Report these with quality results:

```text
correction count per video and nominal timestep
median and range of correction delta/input
reference-to-native output disagreement after correction
origin/trusted selection frequency
reference age in latent frames
reliability distribution by block time
accept/reject/evict counts and rejection reasons
correlation between reliability at block k and quality drop at k+1
runtime and peak GPU memory relative to native
```

A large correction delta with no visible improvement means the reference path
is strong but wrong. A tiny delta means re-noising or the reference context is
ineffective. A fixed-origin gain with no hybrid gain means the prior mechanism
works but our state-commit contribution does not.

## 9. Go/no-go decisions

### Gate 0: implementation

Pass only if smoke traces satisfy every hard invariant and videos decode.

### Gate 1: pathwise mechanism

`origin_t500_250` must visibly delay failure or improve identity/background
relative to native SF on multiple prompts without freezing motion. If it does
not, stop the 16-cell sweep and audit the independent TTC implementation,
scheduler mapping, reference cache, and base compatibility.

### Gate 2: proposed contribution

`hybrid_t500_250` must beat fixed origin on paired review and show a meaningful
quality/failure-time improvement. Reliability should predict later failure,
and threshold ablations should change accepted states in the expected order.

If fixed origin works but hybrid does not, the new contribution is unsupported.
Do not rename fixed TTC as Commit Forcing.

### Gate 3: strong baselines

The selected method must be compared with official PF at equal prompt, frame,
seed, and checkpoint conditions. Echo/Deep/Future Forcing comparisons should be
added when their environment is stable, with compute and backbone differences
reported.

### Gate 4: confirmation

Promote the method to a paper claim only after four-seed paired results, frozen
blind review, confidence intervals/effect sizes, and no unacceptable loss in
motion or prompt alignment.

## 10. Paper story if the gates pass

The defensible story is:

1. Long AR diffusion has a **state promotion problem** in addition to a cache
   eviction problem: generated errors can become future conditioning state.
2. The existing denoising trajectory provides a training-free signal for
   deciding which states are stable enough to commit.
3. A bounded origin/trusted/recent lifecycle separates immutable identity,
   evolving reliable state, and short-term motion.
4. Reliability-gated references are used through pathwise correction, strong
   enough to alter generation but temporary enough to preserve native motion.
5. Experiments isolate fixed-reference correction from dynamic commit, then
   compare against cache-policy baselines.

Safe provisional contribution language:

> We study reliability-aware state admission for training-free autoregressive
> video extrapolation. Building on reference-conditioned pathwise correction,
> we use disagreement along the model's existing denoising trajectory to
> maintain a bounded bank of origin and trusted states, and invoke these states
> as temporary correction contexts while preserving the native recent cache.

Do not claim:

- that reference-conditioned pathwise correction is ours;
- that anchors, FIFO memory, pre-RoPE K/V, or re-noising are individually new;
- first training-free long video, first historical memory, or first
  head-aware cache;
- SOTA, significant improvement, or identity preservation before the
  confirmation experiment;
- that a fixed-origin result validates reliability-gated commit.

## 11. Fallback strategy

The current implementation intentionally tests the least entangled base first.

1. If fixed origin and hybrid both work on SF, continue with Commit Forcing and
   use PF only as a strong baseline.
2. If fixed origin works but hybrid fails, redesign the admission signal or
   publish neither as our method; a fixed TTC reproduction is only a baseline.
3. If neither works but official PF remains strong, a PF-based iteration is
   acceptable only with explicit attribution and a fundamental new mechanism,
   such as reliability-gated state commit that demonstrably adds value over
   unchanged PF. PF's head classification, ragged cache, and pyramidal policy
   cannot be repackaged as ours.
4. If SF's absolute-position or backbone collapse dominates every correction,
   test the same state-admission hypothesis on Causal Forcing or a position-safe
   base and report the base change. Do not attribute base improvements to the
   new module.
5. If reliability does not predict quality but the trace dataset is
   informative, retain the result as a diagnostic study rather than forcing a
   SOTA method story.

## 12. Current limitations

- No CUDA run has been performed for this prototype on the current machine.
- Unit tests requiring PyTorch cannot run locally because PyTorch is absent.
- The full model integration is statically compiled but still needs the smoke
  test to validate cache shapes, dtype, scheduler behavior, and peak memory.
- P0 supports inference batch size 1, matching the current generation scripts.
- Reliability may measure denoising difficulty rather than visual correctness.
- Adjacent reference RoPE may preserve identity while suppressing temporal
  progression; motion review is mandatory.
- Extra model forwards increase latency, and an always-on method is not an
  efficiency contribution.
- Scene return is not solved by the current episode reset policy.

These limitations are experiment questions, not details to hide in a paper.
