# ProbeCache: Method, Implementation, and Server Plan

> Status: code complete for CPU/static review; GPU behavior is not yet
> validated. This document supersedes v78 as the current experiment candidate,
> but v78 `full_budget075_p1` remains the validated fallback.

## 1. One-sentence idea

**ProbeCache** uses controlled counterfactual interventions to classify each
attention head by function, then gives persistent and reactive heads different
active-memory lifecycles while inserting selected history directly into Pyramid
Forcing's native middle slots.

The proposed paper title is:

> **ProbeCache: Counterfactual Head Profiling and Dual-Lifecycle Memory for
> Training-Free Long Video Generation**

The primary task is direct 30s+ extrapolation from one prompt. Prompt/scene
switching and A-B-A return are secondary stress tests.

## 2. What is and is not the contribution

Anchor/recent/retrieval caches are common ideas. Using them is legitimate, but
they cannot be claimed as novel by themselves. Our candidate contribution is
the closed loop:

1. **Counterfactual functional profiling:** classify heads from changes in
   native per-head attention outputs under controlled prompt and history
   interventions, not from attention location or a hand-written label rule.
2. **Dual active-memory lifecycles:** one trust-conditioned clean archive
   supports coverage-preserved distant recall for persistent heads and
   current-segment event turnover for reactive heads.
3. **Direct middle-slot realization:** recalled frames replace PF middle
   anchors and use the normal post-prune RoPE/FlashAttention path. There is no
   extra memory-attention output and no residual fusion coefficient.
4. **Fail-closed operation:** low-confidence retrieval falls back to the
   original PF cyclic/stride policy for that head and call.

We must not claim that static/dynamic heads, head-specific caches,
anchor/recent memory, novelty update, or retrieval are new in isolation.

## 3. Counterfactual head profile

### 3.1 Signals

For head `(l,h)`, let `O` be a compact sketch of the native self-attention
output before output projection:

```text
O = concat(mean_token(output), RMS_token(output))
```

The profiler estimates:

```text
S_prompt = ||O(prompt_a) - O(prompt_b)||
           / mean(||O(prompt_a)||, ||O(prompt_b)||)

S_remote = ||O(full_history) - O(recent_only)||
           / mean(||O(full_history)||, ||O(recent_only)||)
```

Every pair fixes seed, model, latent length, identity, style, and most scene
content. Prompt pairs change action, camera, scene, or weather. History pairs
change only whether self-attention can read the full PF cache or recent frames.

Scores are aggregated across:

- 8 prompt pairs;
- 4 single-prompt history cases;
- seeds 0 and 1;
- denoising call indices 0, 2, and 3;
- all matched autoregressive blocks.

Only the conditional CFG cache is profiled. The recorder rejects other
`_cfg_branch` values by default and persists `cfg_branch` in every record, so
unconditional calls cannot shift or contaminate the paired call indices.

Within each layer, median/MAD normalization removes depth-scale differences:

```text
role_score(l,h) = z(S_remote) - z(S_prompt)
```

Deterministic two-means clustering over all 360 role scores produces:

- `1`: persistent, high remote utility relative to prompt sensitivity;
- `-1`: reactive, high prompt sensitivity relative to remote utility.

The builder also reports bootstrap label agreement, cluster margin, raw sample
counts, and cluster balance. PF Wave/Anchor/Veil labels may be compared only
after classification; they must not tune pairs, thresholds, or labels.

### 3.2 Profile acceptance gates

Do not run the full video matrix unless:

- both clusters contain at least 10% of heads;
- every head has prompt and history observations;
- at least 80% of heads have bootstrap agreement >= 0.75;
- persistent heads have higher median `S_remote`;
- reactive heads have higher median `S_prompt`;
- repeated seeds do not collapse all heads into one role.

The builder records cluster balance, bootstrap stability, and the two
signal-direction checks under `acceptance_gates` in the JSON report. The
profiling launcher passes `--strict-gates`, so a failure preserves the report
but returns a nonzero status. Per-seed collapse remains a manual report review
until enough server profiles exist to set a defensible seed-wise threshold.

If these fail, retain the PF binary mapping (`Anchor -> persistent`,
`Wave+Veil -> reactive`) only as an explicitly named oracle/baseline, not as
the learned method.

## 4. Exact cache composition

ProbeCache retains one physical clean archive per layer and exposes different
active views per head.

| Role | Static sink | Direct middle | Recent | Target total |
|---|---:|---:|---:|---:|
| Persistent | first 3 frames | top-4 recalled frames | last 4 frames | 11 |
| Reactive | first 1 frame | top-4 event frames | last 4 frames | 9 |

The current 3-frame generation block is part of the live recent path; it is not
counted as archive capacity. If retrieval abstains, the direct middle is
replaced by the original PF middle policy.

### 4.1 Physical archive

- Maximum: 24 full-spatial BF16 K/V frames per layer.
- Write frequency: at most one frame per clean AR block.
- No noisy-pass writes.
- Payload: K, V, raw position, frame id, prompt descriptor, segment id,
  segment-start bit, per-head reliability, validity, and temporal novelty.
- Candidate within a block: maximum mean
  `novelty * reliability * validity` over its 3 frames.
- Per-head validity comes from v78 trust-conditioned commit decisions.
- Eviction uses farthest-point descriptor coverage while protecting endpoints
  and segment starts.

Approximate archive payload at 24 frames, 1560 tokens/frame, 12 heads, head
dimension 128, BF16 K+V is 230 MB/layer or 6.9 GB for 30 layers. The
unconditional few-step CFG copy explicitly disables ProbeCache. If server
memory is tight, `archive12` is the first fallback.

### 4.2 Persistent recall

Candidates must:

- be older than the recent-4 boundary;
- have valid trust-conditioned writes;
- have reliability >= 0.55;
- not duplicate the static first frame;
- satisfy the optional prompt compatibility floor.

Score:

```text
visual = cosine(mean(current pre-RoPE Q), mean(archive K))
score = reliability * ((1-w_prompt) * visual + w_prompt * prompt_similarity)
```

The selector uses top-k with temporal NMS (minimum spacing 2), then abstains
for low best similarity, low softmax margin, or high normalized entropy.
Accepted full-spatial frames become ordinary `CollectedAnchor` objects with
dynamic post-prune RoPE.

### 4.3 Reactive event memory

Reactive candidates:

- are outside recent-4 but inside the last 12 frames;
- belong to the current prompt segment;
- pass the same validity/reliability check.

Score:

```text
event_score = reliability
              * temporal_KV_novelty
              * clamp((cosine(Q,K) + 1) / 2, 0, 1)
```

Top-4 events use temporal NMS. Flat event scores abstain and fall back to PF's
cyclic middle. Reactive memory turns over immediately because selection is
limited to the current segment and local horizon; it does not use staggered
long-term promotion.

### 4.4 Prompt switches

The pipeline sends every current prompt descriptor to ProbeCache. If cosine
similarity with the previous descriptor falls below 0.55:

- increment `segment_id`;
- mark the first clean archive write as a protected segment anchor;
- reactive heads stop reading prior-segment events;
- persistent history stays physically present;
- an A-B-A return can recall A if Q/K and prompt compatibility support it.

This avoids a VLM/LLM dependency in the core method. Known prompt boundaries
from segmented inference are used directly.

## 5. Why this differs from nearby work

| Work | Idea used as prior/inspiration | ProbeCache boundary |
|---|---|---|
| [Pyramid Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | PF base, sink/middle/recent composition, per-head policies, post-prune RoPE | PF's 3 labels and cyclic/stride/merge policy are not claimed; ProbeCache changes how roles are measured and how middle frames are selected |
| [Forcing-KV](https://github.com/zju-jiyicheng/Forcing-KV) | Static/dynamic binary heads and dynamic temporal compression establish that two roles are plausible | Its classification is attention-location based; ours uses controlled output interventions and trust-conditioned direct retrieval |
| [Head Forcing](https://jiahaotian-sjtu.github.io/headforcing.github.io/) | Local/anchor/memory heads, fast/episodic memory, novelty updates | We use two counterfactually measured roles, a shared physical archive, PF fallback, and direct middle replacement |
| [Echo Forcing](https://github.com/mingqiangWu/Echo-Forcing) | Preserve/recall/forget and scene snapshot lifecycle motivate segment-aware memory | No Echo code is copied; the primary method is single-prompt and retrieval is per-head Q/K inside PF |
| [IAMFlow](https://github.com/Eddie0521/IAMFlow) | Explicit identity/state memory motivates testing identity preservation | No identity encoder, VLM, or trained module is used |
| [MemRoPE](https://github.com/YoungRaeKimm/MemRoPE) | Position-safe memory motivates reusing raw K and applying RoPE at readout | ProbeCache reuses PF's existing post-prune RoPE rather than implementing MemRoPE |
| [Deep Forcing](https://github.com/cvlab-kaist/DeepForcing) | Sink and compressed-history evidence | No participative training or Deep Forcing cache code is copied |
| Internal v78 | Noisy-clean reliability, bounded promotion, trace protocol | v78 is integrated as the persistent-write trust source and remains an ablation |

New `probecache.py` and the profile builder were written against local PF
interfaces. No source code was copied from the repositories above. Any paper
must cite all methods whose concepts are discussed or used.

## 6. Code map

```text
third_party/Pyramid-Forcing/pyramidkv/probecache.py
    shared archive, segment lifecycle, persistent/reactive selectors, traces

third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py
    clean archive commit, query injection, recent-only active tail,
    query-dependent packed-readout invalidation, direct middle override,
    and PF fallback

third_party/Pyramid-Forcing/wan/modules/attention/core.py
    pre-RoPE Q handoff and compact native-output profile capture

third_party/Pyramid-Forcing/pipeline/{pyramidkv_config,causal_inference}.py
    configuration, prompt descriptors, conditional-only lifecycle

third_party/Pyramid-Forcing/inference.py
    CLI overrides, profile metadata, profile persistence

scripts/build_probecache_head_profile.py
    paired scores, robust normalization, two-means, bootstrap, CSV/JSON output

scripts/run_v81_probecache_profile_16gpu.sh
    48 one-time profile jobs in three 16-GPU waves

scripts/run_v81_probecache_16gpu.sh
    16-cell smoke/single/switch matrix

scripts/summarize_probecache_trace.py
    archive, acceptance, reason, selected-age and prompt-switch summary
```

All behavior is disabled by default.

## 7. Server commands

### 7.1 Build the binary head profile

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
bash scripts/run_v81_probecache_profile_16gpu.sh
```

Required output:

```text
runs/v81_probecache_profile/labels/probecache_binary_labels.csv
runs/v81_probecache_profile/labels/probecache_profile_report.json
```

Review the profile gates before continuing.

### 7.2 Smoke

```bash
HEAD_CSV="$PWD/runs/v81_probecache_profile/labels/probecache_binary_labels.csv" \
SMOKE_FRAMES=12 \
bash scripts/run_v81_probecache_16gpu.sh smoke
```

The smoke must produce videos and non-empty archive/selection traces without
NaN, CUDA OOM, shape errors, future-frame reads, or empty profiles.

### 7.3 Main 30s single-prompt screen

```bash
HEAD_CSV="$PWD/runs/v81_probecache_profile/labels/probecache_binary_labels.csv" \
FRAMES=120 \
bash scripts/run_v81_probecache_16gpu.sh single
```

The 16 cells include SF/PF/Echo, audit, persistent-only, reactive-only, full,
no-trust, archive 12/36, top-k 2/6, prompt weight 0/0.30, open admission, and
conservative admission.

### 7.4 Prompt-switch screen

Run only after a full cell is competitive with PF on single-prompt videos:

```bash
HEAD_CSV="$PWD/runs/v81_probecache_profile/labels/probecache_binary_labels.csv" \
FRAMES=120 \
bash scripts/run_v81_probecache_16gpu.sh switch
```

### 7.5 Blind review and metrics

```bash
bash scripts/postprocess_v81_probecache.sh prepare single

# Fill and freeze blind_review/scorecard.csv first.
HUMAN_REVIEW_DONE=1 \
bash scripts/postprocess_v81_probecache.sh metrics single
```

Use `switch` instead of `single` for the A-B-A screen. The metrics phase
produces comprehensive identity/motion metrics, temporal jump, ProbeCache
trace summaries, VBench-Long, and A-B-A return metrics for the switch task.

## 8. Debug trace interpretation

`archive_update` must show:

- exactly one update per clean block/layer;
- `valid_heads` consistent with v78 transition acceptance;
- archive size never above the configured budget;
- nonzero novelty after initialization;
- segment-start writes after hard prompt changes.

`middle_selection` must show:

- no selected time inside recent-4;
- persistent mean selected age greater than reactive mean selected age;
- reactive selected frames all in the current segment;
- `audit` computes diagnostics but has `mode_active=false`;
- low-confidence calls report a reason and fall back to PF;
- full mode has nonzero accepted calls for both roles.

The packed PF readout cache must be invalidated on every ProbeCache query.
Otherwise a new `middle_selection` trace can be emitted while FlashAttention
silently reuses the previous query's packed middle slots. The implementation
does this in `AdaptiveKVCache.set_probecache_query`; a CPU unit test guards the
invalidation.

Failure diagnosis:

| Symptom | First check | Likely action |
|---|---|---|
| No visual change | role acceptance and selected ages | open gates, verify generated CSV is loaded |
| Motion freezes | reactive acceptance, event score spread, top-k | reduce reactive top-k/horizon or increase novelty requirement |
| Identity degrades | persistent wrong-segment recall and prompt similarity | raise prompt weight/floor or use conservative gate |
| PF audit differs | audit video/hash and recent-tail size | audit must not activate managed-head truncation |
| OOM | archive size and CFG copies | use archive12; verify uncond ProbeCache is disabled |
| Flicker at switches | prompt-switch trace and first segment anchor | increase switch threshold or delay reactive activation one block |

## 9. Evaluation and go/no-go

Perform blind human review before metrics. Review:

- face/body/clothing identity at start, 10s, 20s, and end;
- object and scene geometry;
- motion amplitude and naturalness;
- freeze, loops, cuts, flashes, darkening, texture boil;
- prompt obedience and A-B-A return.

Then compute DINO/min-DINO/drift, temporal jump, motion magnitude, LPIPS,
CLIP/text alignment, VBench-Long, and task-specific switch/return metrics.

Advance only if one full configuration:

- is visually better than PF on identity or long-horizon structure;
- is not worse on motion/dynamic degree;
- retains v78's temporal-jump advantage;
- beats both persistent-only and reactive-only on their complementary failure
  cases;
- has trace evidence that the intended mechanism, not an accidental cache
  budget change, caused the result.

If no full cell exceeds PF, keep v78 as the main result and treat ProbeCache as
a falsified extension. Do not reshape the paper story before the mechanism and
visual evidence agree.
