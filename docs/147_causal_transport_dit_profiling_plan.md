# v147: Downstream-Causal DiT Head and QK-V Transport Profiling

## 1. Why v147 is the next experiment

The current evidence does not justify naming a stable semantic head taxonomy:

- v134 prompt perturbations were not magnitude matched.
- v136 found reproducible temporal-age behavior, but the prompt split was
  degenerate.
- v138 supported history specificity but not a clean order taxonomy.
- v142 found local output effects and prompt modulation, but its static-head
  gate narrowly failed.
- corrected v143/v144 results show that many raw head features are stable,
  while only a small subset remain stable after seed and family controls.
- v145 is therefore testing crossed prompt families and two independent seeds.

Even if v145 finds a reproducible ranking, it is still only a correlation. The
missing link is:

> Does changing the history available to the ranked heads causally change the
> final flow/x0 prediction, and does content-aware retrieval reduce that
> damage?

v147 measures that link on the native Self-Forcing trajectory. It does not
generate a counterfactual video for every probe and does not mutate the native
cache. This keeps the experiment fast enough for 32 GPUs while moving from
local attention proxies to downstream denoiser outputs.

## 2. Preregistered hypotheses

### H0: replay integrity

Repeating the same forward with no head intervention must reproduce the
original flow and x0:

```text
max(relative_RMS(flow), relative_RMS(x0)) <= 1e-4
```

The replay must also leave every native K/V cache index and tensor version
unchanged. Failure invalidates all v147 causal results.

### H1: v145 ranking has downstream causal meaning

For the best eligible cache-related v145 axis, take three heads per layer from
the top and bottom of the layer-residual ranking. Compare these with a
deterministic three-head-per-layer control sampled from the remaining six
middle-ranked heads, disjoint from both extremes.

Under the same `recent4`, `q_retrieval8`, or `value_shift` intervention, a
functional high-score group should produce a larger final x0 change than both
equal-count controls.

A result is considered reproducible only if all hold:

- median paired log effect is positive;
- 95% prompt-bootstrap mean CI is above zero, or the paired win rate is at
  least 0.65;
- seed-replicate Spearman is at least 0.30.

No functional role name may be assigned from the v145 score alone.

### H2: query retrieval rescues useful long history

For the same top-ranked heads:

```text
rescue = (delta_recent4 - delta_q_retrieval8) / delta_recent4
```

`q_retrieval8` keeps recent4 and retrieves four older frames using sampled,
matched-spatial Q-K scores. `uniform8` is the equal-budget control, so retrieval
must outperform both recent4 and uniform8 in the same captured state. It
becomes a cache-design candidate only if:

- median x0 rescue is at least 5%;
- at least 60% of paired runs have positive rescue;
- seed-replicate Spearman is at least 0.30.
- relative to uniform8, median gain is at least 2%, paired win rate is at least
  0.55, and seed-replicate Spearman is at least 0.20.

This separates “the heads are sensitive to removing history” from “our
retrieval rule restores the relevant history.” The same matched-budget gain is
computed for random control heads. A separate selectivity gate tests whether
top-ranked heads gain more from retrieval than random heads; if it fails,
retrieval may still be useful, but it is not evidence for a head-selective
mechanism.

### H3: addressing and content transport are separable

`value_shift` preserves K and the newest four V frames, but circularly shifts
only the older V frames. It therefore keeps the addressing surface while
breaking old K-to-V content alignment.

If this intervention has a reproducible downstream effect, V content transport
matters beyond temporal Q-K preference. If it does not, the proposed method
should not claim a distinct content-transport mechanism.

### H4: the mechanism depends on layer and denoising state

The same top-ranked heads are intervened in three equal-count bands:

- early: layers 0-9;
- middle: layers 10-19;
- late: layers 20-29.

Each band contains 10 layers times 3 heads. Effects are measured at noisy
timestep 1000, noisy timestep 500, and the clean-context refresh. This tests
layer and timestep dependence without hand-labeling a band in advance.

### H5: QK-V transport alignment is descriptive until causally validated

For sampled current-frame spatial tokens, v147 records:

- the top-1 historical position from Q-K similarity;
- the top-1 historical position from current-V/history-V similarity;
- a V-refined choice inside the top-k Q-K candidates;
- coordinate error, top-1 agreement, direction agreement, displacement, and
  normalized Q-K entropy.

This is called **QK-V transport alignment**, not optical flow. V-nearest
positions are not ground-truth motion. The metric supports a mechanism claim
only if its head-group pattern agrees with a passing H1/H3 intervention.

## 3. Source ranking and controls

`scripts/build_v147_causal_transport_suite.py` reads:

- `feature_reproducibility_audit.csv`;
- `head_factor_reproducibility.csv`;

from v145. Candidate axes are identity, scene, or full-semantic changes over
Q shift, K shift, V shift, value-scale shift, and local-policy shift.

The selection rule is frozen:

1. prefer axes passing the v145 reproducibility screen;
2. maximize the lower of family-split and seed-replicate Spearman;
3. use descriptor direction/specificity as a secondary criterion;
4. use the fixed K, policy, Q, V, value-scale order only as a final tie break.

The generated probe plan records the selected axis, scores, exact head maps,
random seed, and overlaps. If no v145 axis passes, v147 may still falsify the
best available ranking, but the source ranking must be reported as exploratory.

## 4. Interventions

All policies retain the current block. History policies are:

| Policy | Historical input |
|---|---|
| `recent4` | newest four historical frames |
| `uniform8` | four uniformly spaced old frames plus recent4 |
| `q_retrieval8` | four per-head Q-K-retrieved old frames plus recent4 |
| `value_shift` | native K; old V shifted by one frame; recent4 V unchanged |

The 15 configured probes are:

- top, bottom, random, and all-head `recent4`;
- top and random `uniform8`;
- top, bottom, and random `q_retrieval8`;
- top, bottom, and random `value_shift`;
- top `recent4` restricted to early, middle, or late layers.

A native replay is added automatically, yielding 16 replays per captured
context and 48 downstream records per profile.

## 5. Experiment grid

- Base: native Self-Forcing sliding-window inference.
- Prompts: the frozen manually stratified 16-prompt subset plus 16 prompts
  chosen by deterministic max-min lexical distance from the 128 Qwen-rewritten
  MovieBench prompts.
- Seeds: two independent seeds per prompt.
- Length: 120 latent frames, approximately 30 seconds.
- Profiles: 64.
- GPUs: 32, exactly two profiles per GPU.
- Captured state: AR start frame 117.
- Captured timesteps: noisy 1000, noisy 500, clean refresh.
- Regular records: 3 states x 30 layers = 90 per profile.
- Downstream records: 3 states x 16 replays = 48 per profile.

The native video is saved once per profile. Probe replays are diagnostic
forwards and do not replace the native trajectory.

## 6. Implementation and correctness guards

### Read-only replay

`CausalInferencePipeline._run_head_profile_downstream_probes`:

1. stores CPU/CUDA RNG states;
2. snapshots every layer's self- and cross-attention K/V storage pointers,
   tensor versions, and self-attention cache indices;
3. enables the existing read-only attention path;
4. runs native replay followed by all probes;
5. checks the cache contract after every replay;
6. restores RNG state and exits read-only mode.

### Head-local replacement

`src/lifecycle_kv/downstream_probe.py` recomputes attention only for the
selected heads. Unselected heads are copied exactly from native output. The
replacement is applied before the self-attention output projection, so later
layers receive the intervention naturally.

### Required debug output

The server logs include:

```text
[HeadProfile] begin ... motion=1 ... downstream=1
[HeadProfile] qk-v-correspondence ...
[HeadProfile] downstream-probe ... name=... flow_rel=... x0_rel=...
```

Strict failures identify replay parity, cache mutation, missing layer coverage,
duplicate probes, incomplete state grids, and invalid profile counts.

## 7. Commands

v147 depends on completed v145 analysis. After all v145 shards finish, run on
node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v147_causal_transport_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v147_causal_transport_profile_32gpu.sh smoke
```

Do not start the 64-profile run unless smoke prints:

```text
[v147-smoke] replay/cache/probe contract: PASS
```

Then launch one command per node:

```bash
# node 0
NODE_RANK=0 bash scripts/run_v147_causal_transport_profile_32gpu.sh causal64

# node 1
NODE_RANK=1 bash scripts/run_v147_causal_transport_profile_32gpu.sh causal64

# node 2
NODE_RANK=2 bash scripts/run_v147_causal_transport_profile_32gpu.sh causal64

# node 3
NODE_RANK=3 bash scripts/run_v147_causal_transport_profile_32gpu.sh causal64
```

After all nodes finish, run on node 0:

```bash
NODE_RANK=0 bash scripts/run_v147_causal_transport_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v147_causal_transport_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v147_causal_transport_profile_32gpu.sh package
```

Progress can be checked from any node:

```bash
bash scripts/run_v147_causal_transport_profile_32gpu.sh status
```

## 8. Analysis artifacts

The analyzer writes:

- `profile_audit.csv`: prompt/seed/runtime integrity;
- `downstream_observations.csv.gz`: every context/probe output delta;
- `downstream_comparisons.csv`: paired head-group and retrieval tests;
- `layer_band_effects.csv`: early/middle/late causal effects;
- `qkv_head_observations.csv.gz`: per-state/layer/head transport metrics;
- `qkv_group_comparisons.csv`: top versus bottom/random transport summaries;
- `report.json` and `report.md`: preregistered gates and interpretation.

The result package is copied to:

```text
docs/results/v147_causal_transport_profile/
```

## 9. Decision after v147

| Result | Decision |
|---|---|
| H0 fails | debug only; discard all mechanism statistics |
| H0 passes, H1 fails | stop static binary head classes; use state-conditioned continuous routing or regressors |
| H1 passes, H2 fails | retain the causal head ranking, but do not use Q retrieval as the method |
| H1 and H2 pass | test trajectory-level selective retrieval on 16 prompts before a 128-prompt method run |
| H1 and H3 pass | describe separate history addressing and value-content transport evidence |
| H2 passes but the selectivity gate fails | test all-head retrieval; do not claim head-selective retrieval |
| only one layer/state is stable | build a layer/timestep-conditioned policy rather than a global head label |

AB/ABA scene switching remains useful, but it should follow rather than
replace the single-prompt long-video mechanism test. If H1/H2 pass, the same
retrieval can later be conditioned on prompt episodes: sensitive heads update
their recent bank at a switch, while long-history retrieval remains available
for an A-B-A return.

## 10. Relationship to prior work

The implementation is original to this repository; no third-party source code
was copied.

- Pyramid Forcing motivates temporal head-dependent cache behavior. v147 does
  not use its three labels as ground truth; its ranking comes from the
  independently measured v145 prompt-factor axes and must pass equal-count
  downstream controls.
- HeadCast classifies Sink/Dummy/Spatial/Global heads using restricted-context
  output similarity at a selected denoising state. v147 borrows the general
  principle that a head label needs an output-effect test, but measures final
  flow/x0 effects over native autoregressive states, paired prompts/seeds, and
  read-only cache interventions:
  https://arxiv.org/html/2607.20125
  and https://github.com/sjlgaga/HeadCast
- HALO uses cross-frame displacement and entropy to identify motion/structure
  heads and validates them by intervention. v147's QK-V diagnostic is designed
  for native long-video K/V history and is explicitly not called optical flow:
  https://arxiv.org/abs/2607.11081
- DiTFastAttnV2 searches head-wise attention policies with local output error.
  v147 instead tests whether a candidate ranking propagates to the complete
  denoiser output and survives independent seeds:
  https://arxiv.org/abs/2503.22796
- Forcing-KV is relevant to static/dynamic K/V distinctions, but v147 tests
  addressing versus V-content alignment directly:
  https://arxiv.org/abs/2605.09681

Any paper must cite these works and describe the above differences. A positive
v147 result is evidence for a distinct measured mechanism, not permission to
rename an existing taxonomy or cache policy.
