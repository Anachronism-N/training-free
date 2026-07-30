# v143 Multi-axis Head Taxonomy and Profiling Plan

## 1. Decision summary

v134-v141 are native Self-Forcing profiling experiments. They do not run the
v132 binary cache candidate and they do not validate that retired cache design.
The v132 candidate remains historical evidence only because its 128-prompt
quality was not strong enough for a final method claim.

The current evidence supports three conclusions:

1. The original zero-threshold prompt-sensitivity label is invalid.
2. Several head properties are reproducible as continuous axes, but a stable
   scalar axis alone is not yet a functional taxonomy.
3. A multi-axis taxonomy is worth testing, provided that clustering is fitted
   without PF labels and role names are assigned only after causal cache
   interventions.

v143 therefore combines temporal allocation, prompt modulation, history
specificity, order sensitivity, output-causal policy demand, scene plasticity,
and persistent-episode compatibility. It tests `k=2..6` with held-out prompt
splits. Existing taxonomies are post-hoc references, not supervision.

## 2. What v134 actually did

### 2.1 Backbone and inference

v134 used native Self-Forcing with:

- config: `third_party/Self-Forcing/configs/self_forcing_dmd.yaml`;
- checkpoint: `third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt`;
- effective backbone: Wan2.1-T2V-1.3B, 30 layers and 12 heads per layer;
- 120 latent frames, written at 16 fps, approximately 30 seconds;
- 3 latent frames per autoregressive block;
- four nominal denoising steps: 1000, 750, 500, and 250;
- warped denoising schedule enabled;
- guidance scale 3.0;
- mixed precision and EMA checkpoint;
- CLI seed 0 with per-job seeds from the frozen manifest;
- all LifeCache, structured-memory, commit-forcing, head-role, and scene-reset
  methods disabled.

The YAML `real_name` field still says 14B, but the wrapper paths and model
construction are hard-coded to Wan2.1-T2V-1.3B. The profiling report must use
the effective model, not the stale YAML label.

`SF_FULL_ATTN_MAX_FRAMES` was unset and `local_attn_size=21`. Thus v134 used
native sliding attention, not full attention. With a 3-frame current block, a
saturated profile contains at most 18 completed history frames plus 3 current
frames. "Full history" below means all history visible inside this native
window, not every frame since frame 0.

### 2.2 Prompt suites

The observational suite contains the 128 Qwen-rewritten MovieBench prompts.

The controlled suite contains 16 prompt families times 8 factors:

`identity`, `appearance`, `action`, `scene`, `object`, `camera`,
`atmosphere`, and `style`.

For each family-factor pair:

- base: all eight factor values use variant A and the canonical template;
- semantic edit: exactly the selected factor is replaced by variant B while
  the canonical template remains unchanged;
- equivalent rewrite: all factor values remain identical to base, but the
  sentence template is reordered and rewritten.

The equivalent rewrite is semantically controlled, but it is not
text-embedding-distance matched. It changes the whole sentence surface while
the semantic branch changes one slot. This is the main confound in the
original prompt-sensitivity score.

### 2.3 Captured states

Noisy states:

- AR starts: 3, 21, 42, 63, 84, 117;
- nominal timesteps: 1000, 750, 500, 250;
- total: 24 calls per video.

Clean states:

- AR starts: 21, 63, 117;
- total: 3 calls per video.

Each call records 30 layers, so each branch has 27 states and 810 layer
records per video.

### 2.4 Counterfactual mechanism

The generated latent state, native historical K/V, timestep, and RNG state are
held fixed. Base, semantic, and rewrite text conditions are evaluated as
read-only shadow branches with separate cross-attention caches. A shadow
branch cannot update the generated video or self-attention cache.

For prompt `p`, layer `l`, and head `h`, v134 computes:

```text
O_full(p)   = Attn(Q(p), K_visible_history, V_visible_history)
O_recent(p) = Attn(Q(p), K_last4_history, V_last4_history)
R(p)        = O_full(p) - O_recent(p)
```

It stores a compact signature of `R`, native output, Q, and current K. The
primary per-state score is:

```text
D_sem  = relative_distance(signature(R(semantic)), signature(R(base)))
D_para = relative_distance(signature(R(rewrite)),  signature(R(base)))
CPHI   = log((D_sem + epsilon) / (D_para + epsilon))
```

`CPHI > 0` was intended to mean that a semantic edit changes history use more
than an equivalent rewrite. This physical interpretation depends on a
magnitude-matched control, which v134 did not provide.

### 2.5 Is v134 reasonable?

The shadow execution and cache isolation are reasonable. The experiment is a
valid measurement of relative text-conditioned changes under a fixed
trajectory. The following claims are not supported:

- it does not measure the effect of changing prompts during actual rollout;
- it does not measure full-video historical attention;
- it does not prove a binary head taxonomy;
- it cannot interpret zero as a physical boundary because edit and rewrite
  magnitudes are unmatched;
- layer 0 is expected to be prompt-blind in this design because self-attention
  precedes propagation of text-conditioned information into that layer.

## 3. What v136 actually did

v136 is CPU-only reanalysis of the frozen v134 `.pt` profiles. It generated no
new video, changed no cache, and did not run the v132 method.

### 3.1 Full and recent4

For a selected state:

- `full` is every completed historical frame in the native 21-frame window;
- `recent4` is the last four completed historical frames;
- current 3-frame K/V is excluded from both counterfactual history branches;
- at AR start 3 there are only 3 historical frames, so `full == recent4`.

AR-start-3 states are negative controls and are excluded from long-history
primary statistics. At saturation, `full` contains 18 history frames and
`recent4` contains ages 1 through 4.

### 3.2 Historical age distribution

For each visible historical frame, v134 sampled spatial positions, averaged
matched-position pre-softmax QK logits, and applied softmax over history
frames. v136 treats this vector as an approximate historical age
distribution.

For current frame `t` and historical frame id `i`:

```text
age(i) = t - i
recent = age <= 4
old    = age > 4
```

Derived features include expected age, normalized expected age, recent/old
mass, mass older than 12 frames, entropy, peak age, positive-logit fraction,
sign-switch rate, dominant FFT period, spectral peak ratio, and
age-logit correlation.

This distribution is a matched-spatial proxy, not exact token-level attention
mass. v143 adds a better sampled token-softmax estimator.

### 3.3 v136 axes

- `P`, prompt interaction: CPHI from semantic edit versus rewrite.
- `A`, prompt-induced age routing: semantic/base JS and Wasserstein changes,
  normalized by rewrite/base changes.
- `R`, temporal reach:
  `RMS(O_full - O_recent4) / RMS(O_native)`.
- `T`, middle/recent logit margin:
  `(mean_logit_old - mean_logit_recent) / RMS(centered_logits)`.
- `L`, old-mass excess over a uniform-frame baseline.
- diagnostics: sign, entropy, peak age, FFT, AR, timestep, layer, and factor
  specialization.

### 3.4 v136 result and correct interpretation

Observed zero-threshold labels:

- prompt conditional/invariant: 1 / 359;
- age conditional/invariant: 0 / 360;
- history supportive/recent preferred: 49 / 311;
- long range/local-or-mixed: 46 / 314.

CPHI rank reproducibility was high (`rho=0.8163`), but almost the entire
distribution was below zero. CPHI and the temporal axis were weakly related
(`rho=0.1075`). The temporal middle/recent axis was very reproducible
(`rho=0.9959`).

The precise conclusion is:

- the physically intended zero split is degenerate;
- the score still contains reproducible between-head ordering;
- v136 does not prove that prompt response is intrinsically continuous;
- v136 also does not prove a meaningful binary split.

PF is not evidence that every axis is naturally discrete. PF gives raw QK
polarity a geometric interpretation, then selects empirical thresholds and
validates them by generation quality. Its 80% sign threshold and 6.4 period
threshold are tuned choices, not distribution-free natural constants.

For CPHI, a manual threshold is acceptable only if:

1. the score has a defensible physical meaning after perturbation matching;
2. the threshold transfers to held-out prompt families;
3. both classes have adequate support and are not boundary dominated;
4. class-specific cache interventions establish different causal needs.

## 4. What v138 actually did

v138 also used native 21-frame Self-Forcing with all proposed cache methods
disabled:

- 128 Qwen MovieBench prompts;
- every prompt uses seed 0 for cross-video comparability;
- 120 latent frames;
- AR starts 21, 63, 117;
- noisy timesteps 1000 and 500 plus clean, for 9 states per video;
- recent4 remains unchanged in every local intervention.

### 4.1 Local history interventions

- reverse: reverse middle-history frame content, then reapply RoPE at the
  original chronological destinations;
- phase shift: circularly shift middle-history content by one frame;
- freeze latest: repeat the latest middle frame across all middle positions;
- value mismatch: keep K positioning but shift V content to break K/V
  correspondence.

The corrected implementation stores pre-RoPE K, follows native FIFO indices,
reconstructs the native positioned K, applies the intervention in content
space, and applies RoPE once at destination positions.

The actual audit contract is max relative RoPE error at most `1e-2`, RMS
relative error at most `1e-3`, and recent-V error at most `1e-6`. The observed
max/RMS values were `0.006803` and `3.02e-5`.

### 4.2 Cross-video history specificity

Each Q and historical K frame is represented using 4 spatial samples and a
fixed 16-dimensional Gaussian projection.

- own history: target Q is compared with K descriptors from the same video
  and same layer/state;
- lexically similar wrong history: the donor prompt with maximum bag-of-words
  Jaccard similarity, excluding self;
- deterministic offset donors: prompts at offsets `+1`, `+37`, and `+73`;
- wrong-history score: maximum over the lexical donor and offset donors;
- specificity: own top-1 cosine minus wrong-history top-1 cosine.

The old phrase "random wrong video" is imprecise. These are deterministic
offset controls, not random samples.

### 4.3 v138 result and limits

- history-specificity gate passed;
- 201 heads have positive median specificity and 159 do not;
- split-half Spearman is `0.9711`;
- bootstrap-reliable fraction is `0.9472`;
- order-response rankings are reproducible, but the two-component mixture
  structure is not preferred, so the order-axis class gate failed.

This establishes broad own-trajectory compatibility in the sampled descriptor
space. It does not establish identity memory, scene memory, useful retrieval,
or output-level benefit. Lexical negatives are weak, projection is
approximate, and the analysis only covers the native local window.

## 5. What v140 changed about the threshold conclusion

v140 uses even controlled families for discovery and odd families for
validation, excludes layer 0 from fitting, and tests raw CPHI plus query,
native-output, and current-K adjusted variants.

The zero threshold fails because raw CPHI produces only 3 positive validation
heads among 348 active heads. This invalidates the intended
`semantic > rewrite` interpretation as a usable binary class.

However, the stronger statement "no threshold works" is not justified.
Discovery Otsu on raw CPHI gives:

- threshold `-0.439365`;
- held-out label agreement `0.8621`;
- positive-set Jaccard `0.7904`;
- held-out minority fraction `0.4339`;
- boundary fraction `0.1121`.

Therefore threshold choice matters. The Otsu split is an empirical high/low
prompt-response partition. It is not yet a semantic-versus-rewrite physical
partition: `exp(-0.439) = 0.645`, so a head above this boundary can still
respond less to the semantic edit than to the rewrite.

The next prompt-axis experiment must magnitude-match text perturbations or use
actual prompt switches and then validate cache consequences.

## 6. What v141 did and what AB adds

v141 runs 32 native Self-Forcing A-B-A schedules:

- 16 families times 2 switch types;
- `scene_action` changes action, scene, object, camera, and atmosphere;
- `identity_scene` additionally changes identity and appearance;
- A1 frames 0-38, B frames 39-77, A2 frames 78-119;
- native self-attention cache persists;
- cross-attention cache resets at each prompt boundary;
- capture frames 36, 39, 42, 75, 78, 81, 117;
- each state evaluates base, exact A/B, and paraphrased A/B shadow branches.

The full switch produces a larger residual change than a local paraphrase
(`0.006326` versus `0.003698`), but the held-out head ranking is insufficient:

- split Spearman `0.5558`;
- zero-label agreement `0.6695`;
- 189 of 348 validation heads are positive.

The correct conclusion is not that responses are entirely episode-specific.
Switch-type agreement is relatively strong, while episode/phase stability is
moderate. There is shared structure, but one universal binary membership is
not supported by the current scalar score.

AB and ABA answer different questions:

- AB measures plasticity: suppress stale A, form B, and preserve B continuity.
- ABA measures episodic recall: restore A after an intervening B episode.

A native 21-frame FIFO cache cannot test long-distance A recall once A has
left the window. ABA therefore requires a persistent, read-only A archive.
v142 supplies this archive probe. v143 adds a dedicated AB suite so plasticity
is not confounded with return-to-A recall.

v143 also keeps the six AB axes separately for every
`switch type x frame x timestep` context instead of immediately collapsing
them into one value per head. It then compares head rankings and top-quartile
sets across:

- the two switch types;
- A versus B episodes;
- six boundary/late frames;
- noisy-1000, noisy-500, and clean denoising states;
- switch-type-by-episode combinations.

The resulting Spearman, sign-agreement, and top-25% Jaccard tables directly
test the v141 claim that head response depends on episode, state, or timestep.
Stable rankings with unstable signs support a shared continuous axis with a
context-dependent operating point. Unstable rankings support an online
conditional router. Stable rankings and stable high-head membership are
required before claiming a fixed functional class.

`stale_a_mass` is a special case. It is compared only while B is active and
A frames are still physically present in the native 21-frame window. A zero
after A has been evicted is an availability fact, not evidence that a head
learned to suppress stale context. `ab_context_axes.csv` records
`stale_a_visible` explicitly.

### 6.1 New v142 result and its consequence

The completed v142 bundle at
`docs/results/v142_output_causal_profile/` changes the priority of the next
experiment:

- all native reconstruction and matching-shadow correctness gates pass;
- natural-policy discovery/validation is stable (`rho=0.9661`, label agreement
  `0.9167`), but the static-policy gate narrowly fails because the median
  per-head modal-state fraction is `0.7270`, below the preregistered `0.75`;
- the median advantage of a context-specific oracle over a fixed policy is
  zero, although its p90 normalized regret is `0.1442`; this does not justify
  broad online routing;
- full prompt switches alter cache-policy demand more than paraphrases
  (`0.00192069` versus `0.00125030`), with split `rho=0.8249`, sign agreement
  `0.9417`, and bootstrap reliable fraction `0.9139`;
- persistent-A content selectivity fails (`rho=0.3395`, sign agreement
  `0.6556`, bootstrap reliable fraction `0.6806`).

The positive evidence is therefore prompt-conditioned cache-policy modulation,
not a fixed prompt-sensitive binary map and not persistent episodic recall.
v143 prioritizes a dedicated A-to-B plasticity experiment and state-resolved
output-causal axes. The persistent archive remains a diagnostic feature, but
it cannot motivate an A-B-A method unless v143 or a later causal intervention
passes a stronger stability gate.

## 7. Existing head taxonomies

| Work | Profiling criterion | Classes | Protocol and cache implication |
|---|---|---|---|
| Pyramid Forcing | Last-query-to-history pre-softmax QK; positive/negative sign rate, then FFT period | Anchor, Wave, Veil | Sign threshold 80%, period 6.4, prompt-level labels followed by voting. Anchor uses broad stride, Wave periodic sampling, Veil local merge. |
| Forcing-KV | `(generated chunk mass + transition-frame mass) / (total mass - sink mass)` | Static, Dynamic | Official code uses last 4 key frames, skips first sink frame, threshold 0.8, and one prompt. Static keeps transition/current; Dynamic uses adjacent-frame segment similarity. |
| Head Forcing | Sink, middle, and current probability mass | Anchor, Local, Memory | 20 prompts, 30 seconds, 3 random AR steps after block 3, all 4 denoise steps. Top 25% sink are Anchor; among the rest, top 20% current are Local. |
| Dummy Forcing | Current-frame attention mass and opportunity cost of discarded context | Sink, Neighbor, Dummy | Observation uses top 25% current-focused heads; runtime dynamic programming uses sink/neighbor/current retained-mass objective. |

The internal legacy-v98 map is not a published taxonomy. Its absolute-sign
rule yields 304 Supportive and 56 Suppressive heads. Against the PF map, 169
of 172 Anchor heads are Supportive, 30 of 32 Veil heads are Suppressive, and
Wave splits 133/23 between Supportive/Suppressive. It therefore captures the
Anchor-like and Veil-like polarity extremes but does not isolate periodic
Wave behavior. v143 keeps this map as a post-hoc reference only.

Primary sources:

- Pyramid Forcing: https://arxiv.org/abs/2605.13111
- Forcing-KV: https://arxiv.org/abs/2605.09681
- Forcing-KV official code: https://github.com/zju-jiyicheng/Forcing-KV
- Head Forcing: https://arxiv.org/abs/2605.14487
- Dummy Forcing: https://arxiv.org/abs/2601.20499

PF has a reporting inconsistency that should not be silently copied: Section
5.1 says 32 prompts and 15-second videos, while Appendix Figure 18/A.8
describes majority voting over 256 prompts and frames 0-68. We use the official
published PF label map as the reference, not an asserted exact reproduction.

The PF paper specifies first difference, mean removal, Hanning window, rFFT,
and decaying-weight harmonic folding, but does not publish the exact harmonic
weights. `head_taxonomy.py` therefore labels its PF period routine as a
formula-level diagnostic, not official PF code.

## 8. How PF can access very old K

PF profiling observes long attention sequences to identify temporal patterns.
At inference, PF disables the ordinary unified FIFO ownership and explicitly
retains:

- sink frames, including early frames;
- head-specific representative middle frames;
- forced recent frames.

It then assembles a ragged head-specific K/V sequence and applies Dynamic RoPE
where required. A late query can attend to frame-0 K because that K is still
stored in the explicit sink cache. PF is not retrieving an evicted key from
nothing, and it does not retain every historical frame at runtime.

## 9. v143 hypotheses and features

The experiment starts from hypotheses, not role labels:

- H1: some heads are local structural heads whose output mainly needs current
  chunk and transition context;
- H2: some heads transport historical trajectory information and are
  sensitive to history identity, order, or broad temporal coverage;
- H3: some heads mediate prompt-conditioned plasticity during A-to-B switches
  and should suppress stale episode context;
- H4: some heads can use a persistent episode archive and become relevant for
  A-to-B-to-A recall;
- H5: these functions may be multi-axis and need not reproduce PF classes.

Feature groups:

1. Region allocation: current, oldest/sink, middle, recent4, and last4 mass.
2. Temporal form: positive rate, mean logit, sign switches, spectral strength,
   middle/recent margin, and temporal reach.
3. Prompt modulation: CPHI and prompt-induced age redistribution.
4. History causality: reverse, phase, freeze, and K/V mismatch effects.
5. History specificity: own versus wrong trajectory compatibility.
6. Output causality: error from replacing native history with candidate
   bounded cache policies after output projection.
7. AB plasticity: prompt-history excess, policy modulation, and stale-A mass
   while B is active and A is still physically visible in the native window.
8. Persistent compatibility: content, positioned-K, and output-level response
   to a re-positioned A archive.

The new region mass estimator samples query and key tokens, computes the full
Cartesian sampled QK matrix, applies softmax over sampled key tokens, and then
sums probability by frame. This preserves the meaning of probability mass.
The previous "softmax of frame-averaged logits" estimator did not.

## 10. Clustering protocol

1. Build discovery and validation statistics from independent prompt
   families. Different factors from one controlled family never cross the
   split.
2. Reject any feature with non-finite heads, near-zero discovery IQR, or
   discovery-validation Spearman below 0.30.
3. Robust-scale using discovery median and IQR only.
4. Balance conceptual feature families by multiplying each feature by
   `1/sqrt(number of accepted features in that family)`. This prevents a
   family with many correlated diagnostics from dominating Euclidean distance.
5. Fit deterministic k-means for `k=2..6` with 32 restarts.
6. Assign validation heads using frozen discovery centers, without any
   post-hoc permutation of validation labels.
7. Require:
   - split label agreement at least 0.80;
   - split ARI at least 0.60;
   - bootstrap ARI median at least 0.75;
   - discovery silhouette at least 0.10;
   - every cluster at least 5% of heads;
   - at most 20% of heads with normalized cluster margin below 0.05.
8. Compare accepted clusters with PF, v98, Forcing-KV, Head Forcing, and Dummy
   labels only after clustering.
9. Do not assign functional names until cluster-specific cache interventions
   produce a selective causal effect.

If no `k` passes, the correct result is `no_stable_k`; the script must not
force a taxonomy for storytelling.

The `cluster` action also performs two cheap sensitivity analyses without new
generation:

- repeat the complete fit with minimum feature split-rho thresholds
  `0.30`, `0.50`, and `0.70`;
- leave out each conceptual feature family one at a time.

The threshold sensitivity gate requires all three thresholds to produce a
validated map with the same `k` and minimum pairwise ARI at least `0.80`.
Leave-one-family-out ARI diagnoses whether the partition is almost entirely
defined by one source, especially PF-like temporal allocation. Neither gate
replaces cache-policy intervention.

## 11. v143 execution

Prepare once on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh preflight
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh smoke_natural
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh smoke_ab
```

Run `natural128` concurrently on all four nodes:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh natural128
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh natural128
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh natural128
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh natural128
```

After all natural jobs finish, run `ab32` concurrently on all four nodes:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
```

Analyze on node 0:

```bash
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh cluster
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh package
NODE_RANK=0 bash scripts/run_v143_multiaxis_profile_32gpu.sh status
```

The default natural prompt source is:

```text
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
```

Natural captures AR starts 9, 21, 63, and 117 at noisy timesteps 1000/500 and
clean, for 12 states and 360 layer records per video.

AB contains 16 families times two switch types. Because the current scheduler
assigns two conditions over 40 AR blocks using one-based integer division, the
actual A-to-B boundary is frame 57. Captures are 54, 57, 60, 75, 78, and 117.

## 12. Outputs and debugging

Run root:

```text
runs/v143_multiaxis_profile/
```

Important files:

- `analysis/analysis_report.json`: contract and split metrics;
- `analysis/natural_head_axes.csv`: discovery/validation natural axes;
- `analysis/ab_head_axes.csv`: discovery/validation AB and persistent axes;
- `analysis/ab_context_axes.csv`: six AB axes for every head, switch type,
  frame, and denoising state after prompt-family aggregation;
- `analysis/ab_context_stability.csv`: pairwise rank, sign, and top-quartile
  stability across switch type, episode, frame, and denoising state;
- `analysis/*_profile_audit.csv`: per-video state and parity checks;
- `clustering/feature_audit.csv`: accepted/rejected feature reasons;
- `clustering/head_feature_matrix.csv`: raw and weighted split features for
  every head;
- `clustering/feature_correlations.csv`: discovery/validation pairwise
  correlations for diagnosing redundant axes;
- `clustering/cluster_diagnostics.csv`: every `k=2..6` gate;
- `clustering/head_cluster_assignments.csv`: only if a `k` passes;
- `clustering/reference_comparisons.csv`: post-hoc ARI/NMI to prior labels;
- `clustering/cluster_sensitivity_report.json`: rho-threshold and
  leave-one-feature-family stability;
- `clustering/cluster_sensitivity_pairwise.csv`: ARI/NMI between every
  sensitivity variant that produces a validated map;
- shard logs: causal-policy parity, persistent capture frames, prompt switch
  frame, cache persistence mode, and failures.

The smoke and audit stages enforce:

- profile version and record counts;
- no incomplete calls;
- sampled-token-softmax region estimator;
- exact prompt-schedule frame;
- matching-prompt shadow parity;
- native output reconstruction bounds;
- expected persistent capture count;
- no traceback, CUDA OOM, or assertion failure.

## 13. Next experiment after v143

If a stable `k` passes:

1. Select the smallest stable `k` unless a larger `k` has clearly better
   silhouette and causal interpretability.
2. For each cluster, run one-policy-at-a-time output probes:
   local/transition, sink+recent, broad stride, similarity/motion memory,
   persistent episode recall, and stale-episode suppression.
3. Test the selected cluster-policy pair on one 30-second natural prompt and
   one AB prompt before any 16/32/128-prompt generation.
4. Use all-head policy, random count-matched partition, PF map, and swapped
   cluster-policy assignment as causal controls.
5. Run ABA only for a cluster that shows persistent-A compatibility and AB
   plasticity; otherwise ABA is not mechanistically motivated.

If no stable `k` passes:

- retain the axes as continuous diagnostics;
- test a low-dimensional conditional router using current state, timestep,
  and episode phase;
- do not publish a fixed head taxonomy claim;
- keep PF/v98 labels only as baselines, not as hidden supervision.

## 14. Academic claim boundary

It is valid to combine ideas from prior work when sources are cited and the
new method has a distinct, experimentally supported mechanism. It is not
valid to rename PF classes, fit clusters to PF labels, or describe a
formula-level reimplementation as official code.

A defensible future contribution would be:

1. multi-axis, held-out-stable functional head discovery;
2. output-causal validation that converts descriptive clusters into roles;
3. episode-conditioned cache policies that separate AB plasticity from ABA
   recall;
4. a training-free long-video method whose natural and switching gains are
   validated independently.

These are hypotheses until v143 and subsequent routing interventions pass.
