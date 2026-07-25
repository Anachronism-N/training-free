# v96 QK-Threshold Binary Head Cache

> **Superseded correctness notice (v97):** the v96 profiler used a global
> self/cross-attention call counter as the layer index. It observed 15 aliased
> even indices and the builder padded the remaining rows. Learned v96 QK maps
> are invalid and must not be used as evidence. Native PF/PF-binary results are
> unaffected. Use `docs/97_score_artifact_threshold_pf_merge_experiment.md`
> and regenerate version-2 profiles with explicit `kv_cache.layer_idx`.

Status: code complete, GPU validation pending

Primary task: training-free 30-second single-prompt long video extrapolation

Secondary task: prompt or scene transition

This document supersedes the v95 recommendation where it conflicts with the
binary-head direction. It does not claim a result before the v96 server runs.

## 1. Corrections to the PF interpretation

Pyramid Forcing (PF) classifies heads from offline, pre-softmax temporal QK
logits. Its decision is not a generic QK magnitude threshold:

1. Compute the fraction of positive and negative historical QK logits.
2. A strongly positive-dominant head is assigned Anchor.
3. A strongly negative-dominant head is assigned Veil.
4. The remaining mixed-sign heads are analyzed by FFT. A stable short
   oscillation period is assigned Wave.
5. A residual fallback uses the mean sign.

Therefore Wave is not defined as "mostly positive" or "mostly negative."
It is defined by regular positive/negative temporal alternation. The actual
positive rate, sign-switch rate, dominant period, and spectral concentration
of the 156 Wave heads must be measured on our model and prompts. v96 writes
these values to `qk_head_threshold_summary.md`; they must not be guessed.

The vendored PF label map contains:

| PF class | Count | Native PF middle cache |
|---|---:|---|
| Anchor | 172 | stride, interval 6, capacity 4 |
| Wave | 156 | cyclic, period 6, four buckets |
| Veil | 32 | merge, patch size 2, capacity 4 |

Every policy also has sink and recent context. The native compositions are
recorded in `third_party/Pyramid-Forcing/configs/pyramid-forcing.yaml`.

References:

- [Pyramid Forcing paper](https://arxiv.org/abs/2605.13111)
- [Official Pyramid Forcing code](https://github.com/if-lab-pku/Pyramid-Forcing)

## 2. v96 hypothesis

The proposed paper direction is a binary, prompt-intervention taxonomy:

```text
Prompt-Stable:
    QK history response changes little under controlled prompt intervention

Prompt-Responsive:
    QK history response changes strongly under controlled prompt intervention
```

The labels are not copied from PF, and their class count is not forced to
match PF Anchor. PF labels are used only after classification for post-hoc
mechanism analysis.

The central hypothesis has two separable parts:

1. Prompt-response scores have a stable, data-supported two-component
   structure, and the low-response component overlaps strongly with PF
   Anchor without being defined by PF Anchor.
2. Prompt-Responsive heads benefit from compressed recent history rather
   than long-range sparse anchors. A Veil-style merge policy is the first
   low-risk implementation; Wave-style cyclic and recent-only are mandatory
   controls.

The second claim may fail while the first succeeds. The implementation keeps
head membership and cache policy independent for this reason.

## 3. Head discovery

### 3.1 Controlled observations

The profiler runs eight minimal counterfactual prompt pairs with seeds 0 and
1, for 32 inference jobs. Each pair changes one factor such as action, camera,
scene, motion, or weather while preserving identity and most visual details.

All profiling heads use the same temporary cache:

```text
uniform label map -> sink3 + stride6(cap4) + recent4
```

This prevents PF labels or a candidate binary policy from changing the
trajectory differently for different heads during discovery.

The profiler enables fixed-scale few-step CFG only for the discovery run so
that conditional and unconditional interventions are both observable. This
extra branch is profiling cost and is not enabled in the 30-second generation
screen. The profiler stores bounded, frame-level pre-softmax QK logits for the
newest query frame against strictly older key frames that are actually present
in the uniform cache. Missing frame IDs are not padded as zero-logit history.
It records:

- layer and head;
- AR position and denoising pass;
- conditional or unconditional CFG branch;
- historical frame indices;
- one temporal QK-logit vector.

Only logits are captured. Attention probabilities are intentionally disabled
to reduce profiling memory and compute.

### 3.2 Prompt-response scores

For aligned temporal vectors `x` and `y`:

```text
NRMS(x, y) =
    RMS(x - y) /
    (0.5 * (RMS(x) + RMS(y)) + epsilon)
```

Two prompt-response estimates are produced:

```text
CFG response:
    NRMS(QK_cond, QK_uncond)

Semantic-pair response:
    NRMS(QK_prompt_a, QK_prompt_b)
```

CFG is the stronger paired intervention because both branches belong to the
same denoising trajectory. Semantic pairs are matched by seed and observation
location but have separately evolved trajectories, so they are corroborating
evidence rather than a perfect causal intervention.

Each raw score is median-aggregated and normalized independently within every
layer using a robust `log1p` z-score. Three maps are emitted:

- `prompt_cfg_threshold.csv`;
- `prompt_semantic_threshold.csv`;
- `prompt_consensus_threshold.csv`, the average of the two normalized scores.

### 3.3 Data-derived threshold

For each score set, v96 fits one-, two-, and three-component one-dimensional
Gaussian mixture models. The binary threshold is the intersection of the two
components:

```text
score <= threshold -> Prompt-Stable
score >  threshold -> Prompt-Responsive
```

There is no per-layer PF Anchor quota and no global PF Anchor quota. Otsu's
threshold is reported as a sensitivity check but does not select the map.

The following gates test whether the binary story is defensible:

| Gate | Requirement |
|---|---|
| two versus one component | `BIC_1 - BIC_2 >= 10` |
| two versus three components | `BIC_2 <= BIC_3` |
| class balance | minority class at least 10% |
| bootstrap stability | at least 80% of heads have label agreement >= 0.75 |

After the threshold is fixed, the report computes Stable/PF-Anchor precision,
Anchor recall, Jaccard, and the full PF-class cross-tab. High overlap would
support the observation that prompt-stable heads often carry persistent
history. It is evidence, not a construction constraint.

### 3.4 Temporal QK diagnostics

For every head, the report also records:

- positive-logit fraction;
- mean QK logit;
- adjacent sign-switch rate;
- dominant FFT period;
- FFT peak-energy ratio.

These are aggregated by the original PF Wave, Anchor, and Veil labels. This
directly answers whether Wave heads are positive-dominant, negative-dominant,
or approximately balanced with high sign alternation on our workload.

## 4. Binary cache composition

### 4.1 Prompt-Stable heads

```text
sink3 + stride(interval=6, capacity=4) + recent4
```

Function:

- sink preserves the generation origin;
- strided middle states provide sparse long-range identity and layout cues;
- recent states preserve local pose and motion continuity.

This initially reuses the PF Anchor policy. The policy is borrowed; our
candidate contribution is the independently discovered membership and its
coupling to a binary prompt-response taxonomy.

### 4.2 Prompt-Responsive heads: primary policy

```text
sink3 + merge(patch=2, capacity=4) + recent4
```

This is the requested `Wave + Veil -> Veil cache` experiment. It applies
PF's native Veil merge policy to every Responsive head, regardless of its
original PF class. The merge is a bounded local spatiotemporal summary rather
than periodic retrieval of distant frames.

Rationale:

- responsive heads should retain current appearance, action, and motion;
- compressed nearby evidence is less likely to reintroduce obsolete prompt
  details than long-range cyclic slots;
- it restores a compression mechanism lost in the earlier binary baseline.

This rationale is a hypothesis. The merge operator itself is borrowed from
PF and cannot be claimed as our invention.

### 4.3 Mandatory fallback policies

```text
cyclic:
    sink1 + cyclic(period=6, four buckets) + recent4

recent-only:
    sink3 + recent4
```

If merge underperforms cyclic, v96 does not discard the binary taxonomy. It
selects cyclic as the Responsive policy and reports that compressed summaries
were not compatible with this class. If both underperform recent-only, the
middle cache is removed for Responsive heads.

### 4.4 Optional trust-conditioned promotion

Selected cells add the validated v78 cache-transition controller:

- noisy/clean agreement estimates state reliability;
- novelty rejects redundant middle writes;
- a bounded budget and deterministic staggering limit simultaneous commits;
- maximum age prevents indefinite freezing.

This tests whether a new state should become persistent after the binary
policy decides what temporal form the head reads. It does not change the
head threshold.

## 5. Experiment matrix

All generation cells use the same MovieGenBench 32 prompts, 120 frames
(30 seconds), seed 0, and matched per-prompt reseeding.

| GPU | Cell | Membership | Responsive policy | v78 |
|---:|---|---|---|---:|
| 0 | `pf` | native PF 3-class | native PF | no |
| 1 | `pf_binary_cyclic` | PF Anchor vs Wave+Veil oracle | cyclic | no |
| 2 | `pf_binary_merge` | PF Anchor vs Wave+Veil oracle | merge | no |
| 3 | `pf_binary_recent` | PF Anchor vs Wave+Veil oracle | recent-only | no |
| 4 | `cfg_cyclic` | CFG threshold | cyclic | no |
| 5 | `cfg_merge` | CFG threshold | merge | no |
| 6 | `semantic_cyclic` | semantic threshold | cyclic | no |
| 7 | `semantic_merge` | semantic threshold | merge | no |
| 8 | `consensus_cyclic` | consensus threshold | cyclic | no |
| 9 | `consensus_merge` | consensus threshold | merge | no |
| 10 | `consensus_recent` | consensus threshold | recent-only | no |
| 11 | `consensus_merge_v78` | consensus threshold | merge | yes |
| 12 | `consensus_cyclic_v78` | consensus threshold | cyclic | yes |
| 13 | `random_merge` | layer-count-matched random | merge | no |
| 14 | `inverse_merge` | threshold labels reversed | merge | no |
| 15 | `pf_binary_merge_v78` | PF binary oracle | merge | yes |

The PF-binary oracle isolates cache policy from classifier quality. Random and
inverse controls test whether score semantics matter rather than class count.

## 6. Server commands

Recommended staged execution:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

nohup bash scripts/run_v96_qk_head_profile_16gpu.sh \
  > runs/v96_qk_head_profile.nohup.log 2>&1 &
```

After profiling, inspect:

```bash
cat runs/v96_qk_head_profile/labels/qk_head_threshold_summary.md
cat runs/v96_qk_head_profile/labels/build_thresholds.log
```

Then generate all 16 cells:

```bash
nohup bash scripts/run_v96_binary_cache_16gpu.sh \
  > runs/v96_binary_cache32.nohup.log 2>&1 &
```

After all 512 videos are complete:

```bash
nohup bash scripts/postprocess_v96_binary_cache.sh \
  > runs/v96_binary_cache32.postprocess.log 2>&1 &
```

For unattended sequential execution:

```bash
nohup bash scripts/run_v96_10h.sh \
  > runs/v96_10h.nohup.log 2>&1 &
```

The staged form is preferred because it allows classification review before
spending the generation budget. `OUT_ROOT` must be clean when rerunning an
incomplete generation cell; the script never deletes existing videos.

## 7. Outputs and debug contract

Head discovery:

```text
runs/v96_qk_head_profile/
  profiles/*.pt
  logs/*.log
  labels/qk_head_threshold_report.json
  labels/qk_head_threshold_summary.md
  labels/qk_head_scores.csv
  labels/prompt_*_threshold.csv
  labels/prompt_consensus_{random,inverse}.csv
  labels/pf_binary.csv
```

Generation and metrics:

```text
runs/v96_binary_cache32/
  <cell>/*.mp4
  configs/<cell>.env
  logs/<cell>.log
  diagnostics/*.json
  blind_review/
  metrics/vbench_long_summary.*
  metrics/comprehensive.json
  metrics/temporal_jump.csv
  metrics/v96_analysis.{json,md}
```

Required log markers:

- `[HeadQKProfile] enabled`
- `[HeadQKProfile] records=...`
- `[QKHeadThreshold] accepted=...`
- `[BinaryPolicyOverride] stable=stride responsive=...`
- `[PyramidKVRuntimePolicy] ...`

`PyramidKVRuntimePolicy` reports the policy actually constructed at runtime,
including label, strategy, sink, and recent counts. This catches the previous
failure mode where a binary `-1` label silently inherited the Wave cyclic
policy.

The postprocessor audits video indices, scans logs for failure signatures,
prepares a blinded human-review directory, computes VBench-Long subject,
background, aesthetic, imaging, and dynamic metrics, computes comprehensive
identity/background metrics and temporal jump, and writes controlled
merge-versus-cyclic decisions.

## 8. Decision rules

Head taxonomy:

1. Do not call the partition a discovered binary taxonomy unless all
   predeclared mixture and bootstrap gates pass.
2. Report PF overlap after thresholding. Do not retune the threshold to
   improve that overlap.
3. Treat CFG and semantic maps as separate ablations. A consensus-only win is
   insufficient if the component maps are unstable.

Cache policy:

1. Prefer merge only if it beats cyclic on a majority of available quality
   metrics without worsening temporal jump or blind motion review.
2. If merge loses, use cyclic as the implementation fallback and retain the
   failed merge result as an ablation.
3. Promote the learned classifier only if consensus merge/cyclic beats both
   random and inverse controls and approaches the PF-binary oracle.
4. Use v78 only if its isolated comparison improves continuity without
   materially reducing dynamic degree.

Human review must be frozen before opening method identities or metric
rankings. Review identity persistence, late duplication, prompt-detail
retention, scene flashback, motion amplitude, exposure drift, and abrupt
temporal jumps.

## 9. Paper contribution boundary

If v96 passes, the defensible technical points are:

1. **Prompt-intervention QK head discovery.** A data-derived threshold
   separates prompt-stable and prompt-responsive temporal-attention behavior,
   with bootstrap and mixture evidence rather than a hand-set class count.
2. **Post-hoc temporal correspondence.** The prompt-stable group is compared
   with PF Anchor only after discovery, exposing whether semantic stability
   and long-history attention coincide.
3. **Response-conditioned heterogeneous cache.** Independently discovered
   head roles select long-range strided evidence or compressed/recent
   evidence, with membership and policy explicitly decoupled.
4. **Trust-conditioned state persistence.** If the v78 cells pass, trajectory
   reliability controls which generated states are allowed into persistent
   middle memory.
5. **Unified single-prompt and transition interpretation.** The same
   responsive group that controls recent semantic evidence in single-prompt
   extrapolation provides a testable mechanism for prompt/scene transitions.

What is not ours:

- Self-Forcing generation;
- PF's Anchor/Wave/Veil classifier;
- PF stride, cyclic, and merge operators;
- generic head-aware KV caching;
- generic novelty-based memory updates.

Nearby work further narrows the claim. [Forcing-KV](https://arxiv.org/abs/2605.09681)
already divides AR-video heads into static and dynamic groups and applies
different pruning policies. [Head Forcing](https://arxiv.org/abs/2605.14487)
already defines local, anchor, and memory heads, hierarchical memory, dynamic
episodic updates, and head-wise RoPE handling. Consequently, neither binary
head classification, functional head specialization, heterogeneous cache
allocation, nor episodic update is novel by itself.

The contribution must be the intervention criterion, threshold evidence,
binary role membership, policy coupling, lifecycle control if validated, and
the resulting long-video behavior. Borrowed mechanisms and code remain
explicitly cited in `docs/64_related_work_code_provenance_and_claims.md`.

## 10. Follow-up only after v96

If Responsive merge wins but still trails PF, the next controlled change is a
uniqueness-preserving bounded merge inspired by the information-retention
principles documented in `docs/flash_vareason.md`: retain one representative
for redundant neighboring states while protecting a high-novelty state.
This is not implemented or claimed in v96 because changing the classifier
and compression operator simultaneously would make the first result
uninterpretable.
