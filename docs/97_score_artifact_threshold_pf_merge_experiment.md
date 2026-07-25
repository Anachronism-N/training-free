# v97: immutable head scores, manual thresholds, and PF binary merges

## 1. Status and correctness correction

v97 is the required replacement for the learned QK maps produced by v96.
The old frame-attention capture used one global call counter for both self- and
cross-attention. In the PyramidKV path this aliased the 30 transformer layers
to 15 even indices; the v96 builder then remapped those indices and padded the
remaining rows. Those learned maps are invalid and must not be used as paper
evidence.

v97 fixes the source of the layer index to `kv_cache.layer_idx`, records that
source in every observation, and rejects a profile unless:

- its format version is at least 2;
- its observed layers are exactly `0..29`;
- every layer contains both conditional and unconditional records;
- the only layer-index source is `kv_cache.layer_idx`;
- all 360 heads contain CFG, semantic-pair, and temporal observations.

This correction does not invalidate native PF results or PF-derived binary
maps because those maps do not depend on the broken profiler.

## 2. The score is stored before any classification

The profiling stage and the classification stage are now separate programs.
Profiling is run once. Changing a threshold or classifier must never trigger a
new model run.

For layer `l` and head `h`, aligned frame-level pre-softmax QK histories give:

```text
d_cfg(l,h) = median NRMS(QK_cond, QK_uncond)
d_sem(l,h) = median NRMS(QK_prompt_a, QK_prompt_b)

z_cfg(l,h) = layer-wise robust-z(log1p(d_cfg(l,h)))
z_sem(l,h) = layer-wise robust-z(log1p(d_sem(l,h)))

s(l,h) = 0.5 * (z_cfg(l,h) + z_sem(l,h))
```

The layer-wise center is the median and the scale is MAD. The unthresholded
score artifact also stores:

- positive pre-softmax QK-logit fraction;
- mean and mean-absolute logit;
- signed logit mass;
- sign-switch rate;
- dominant temporal period and spectral peak ratio;
- every per-run CFG and prompt-pair observation.

The immutable outputs are:

```text
runs/v97_qk_head_scores/scores/
|-- qk_head_scores.csv
|-- qk_head_observations.json
|-- qk_head_score_artifact.json
|-- qk_head_score_summary.md
`-- layer_capture_audit.json
```

`qk_head_score_artifact.json` contains the SHA-256 hashes of the score CSV,
observation file, source profiles, and profiling run manifest. Its
`classification` field is `null`. The offline classifier verifies the CSV
hash before producing any map.

## 3. Classification hypotheses

### 3.1 Prompt-intervention threshold

For a manually declared threshold `tau`:

```text
Support head:    s(l,h) <= tau
Responsive head: s(l,h) >  tau
```

Low intervention response means the temporal read pattern is comparatively
stable when conditioning changes. High response means the temporal read
pattern changes under CFG or a counterfactual prompt. The terms are functional
descriptions, not claims about semantic identity.

The primary screen evaluates `tau in {0.0, 0.5, 1.0, 1.5, 2.0}`. `tau=1.0`
is the predeclared main operating point: it marks a head whose combined
response is one robust, MAD-scaled unit above its layer center. The threshold
is not chosen to copy PF's Anchor count. GMM-2 intersection and Otsu
thresholds are saved as diagnostics, not silently substituted for the manual
threshold.

Two layer-wise count-matched controls are generated for `tau=1.0`:

- random membership;
- reversed score direction, assigning the lowest scores to Responsive.

These controls test whether score direction matters beyond the number of
heads assigned to each cache.

### 3.2 Sign-based alternative

The alternative split is:

```text
Support head:    positive_rate >= 0.5
Responsive head: positive_rate <  0.5
```

This directly evaluates the hypothesis raised by the previous Wave-head logs.
It is an alternative classifier, not part of the prompt-sensitivity score.

### 3.3 PF-derived binary fallbacks

Two maps merge PF's published three classes:

```text
PF-AR: Anchor | (Wave + Veil)
PF-AW: (Anchor + Wave) | Veil
```

`PF-AR` tests the earlier proposal that Wave and Veil can share compressed
history. `PF-AW` tests whether Wave's mostly positive QK logits make it more
compatible with Anchor as a temporal-support class.

These maps are explicitly PF-derived controls. If prompt-intervention
classification fails, PF-AR or PF-AW may still be a useful engineering
fallback, but the PF class discovery cannot be claimed as ours.

The classifier writes all maps, map hashes, per-layer class counts, PF
cross-tabs, pairwise map agreement, and the full threshold sweep to:

```text
runs/v97_qk_head_scores/maps/
|-- head_map_manifest.json
|-- head_map_classification_report.json
|-- threshold_sweep.csv
|-- map_agreement.csv
|-- prompt_tau_*.csv
|-- sign_rpos_0p5.csv
|-- pf_anchor_vs_rest.csv
|-- pf_anchor_wave_vs_veil.csv
`-- pf_*_extended_recent.csv
```

## 4. Cache definitions

Membership and cache behavior are separate factors.

### Support cache

The default Support read is:

```text
sink3 + stride(interval=6, capacity=4) + recent4
```

An implemented but non-primary hybrid option is:

```text
sink3 + stride(capacity=2) + cyclic(capacity=2) + recent4
```

The hybrid has the same four-frame middle budget. It remains available for a
follow-up only if the default binary result is promising.

### Responsive cache

The default Responsive read is:

```text
sink3 + spatiotemporal merge(patch=2, capacity=4) + recent4
```

This keeps bounded compressed history rather than dropping all remote context.
Two controls are:

```text
cyclic: sink1 + phase bucket(period=6, capacity=4) + recent4
recent: sink3 + no middle + recent4
```

The merge, cyclic, stride, and sink/recent primitives come from or closely
follow Pyramid Forcing. They are borrowed operators. The tested contribution
is the independently measured binary role, the role-to-cache coupling, and
the controlled simplification, if the evidence supports them.

### PF class mechanism ablation

For each native PF class, one ablation replaces its middle cache with
additional recent frames under an approximately matched token-read budget:

```text
Anchor/Wave native: native sink + 4 full-frame middle slots + recent4
Anchor/Wave replacement: native sink + no middle + recent8

Veil native: sink3 + 4 patch-2 merge blocks + recent4
Veil replacement: sink3 + no middle + recent5
```

At patch size two, four Veil blocks contain roughly one full frame of tokens,
not four full frames. The class-specific replacement avoids giving the Veil
ablation a large token advantage. Running all three identifies which PF
middle mechanism is responsible for quality, instead of inferring contribution
from class counts or logit sign.

## 5. The 16-cell, 32-prompt, 30-second screen

Every cell uses MovieGenVideoBench-32, seed 0, per-prompt reseeding, and 120
output frames.

| Group | Cells |
|---|---|
| Manual threshold | tau 0.0/0.5/1.0/1.5/2.0, Support stride, Responsive merge |
| Cache factorization | tau 1.0 with Responsive cyclic and recent-only |
| Membership controls | tau 1.0 random and reversed, both with merge |
| Alternative classifier | positive-rate 0.5 with stride/merge |
| PF binary fallbacks | PF-AR and PF-AW with stride/merge |
| PF reference | native PF |
| PF mechanism | Anchor, Wave, or Veil middle replaced by extended recent |

Run all stages:

```bash
cd /path/to/training-free
git pull

REPO_ROOT="$PWD" \
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v97_10h.sh
```

Run or resume stages separately:

```bash
bash scripts/run_v97_qk_head_profile_16gpu.sh
bash scripts/run_v97_threshold_pf_merge_16gpu.sh
bash scripts/postprocess_v97_threshold_pf_merge.sh
```

Useful overrides:

```bash
FORCE=1 bash scripts/run_v97_qk_head_profile_16gpu.sh
FORCE_METRICS=1 bash scripts/postprocess_v97_threshold_pf_merge.sh

RUN_PROFILE=0 RUN_GENERATION=1 RUN_METRICS=1 \
  bash scripts/run_v97_10h.sh
```

Do not point v97 at the old v96 profile directory. Version and layer-source
checks will intentionally reject it. To regenerate completed videos, use a
new or empty `OUT_ROOT`; the generation runner does not overwrite existing
video sets because partial overwrites can mix commits.

## 6. Debug and review contract

Before looking at aggregate metrics, freeze:

```text
runs/v97_threshold_pf_merge32/blind_review/scorecard.csv
```

Inspect the following artifacts:

1. `scores/layer_capture_audit.json`: exact 30 layers, both CFG branches, and
   explicit layer source for every profile.
2. `maps/head_map_classification_report.json`: GMM BIC, threshold counts,
   PF overlap, sign statistics, and score hash.
3. `diagnostics/*.map_audit.json`: runtime map shape, hash, and per-layer
   counts.
4. `logs/*.log`: `[PyramidKVRuntimePolicy]`,
   `[BinaryPolicyOverride]`, and PF ablation markers.
5. `traces/*.policy.jsonl`: actual per-head sink/recent values, configured
   strategies, selected frame IDs, and union token counts.
6. `metrics/policy_trace_audit.json`: strict map-to-runtime validation;
   hybrid middle count may never exceed four.
7. `metrics/v97_analysis.md`: threshold curve, count-matched controls, cache
   factorization, PF merge comparison, and PF class mechanism contribution.

Any missing layer, map-hash mismatch, policy mismatch, malformed trace,
incomplete video index, CUDA failure signature, or missing completion marker
is a hard failure. A resumable run also refuses to reuse completed outputs
when the commit, prompt hash, frame count, seed, score hash, or map-manifest
hash differs.

## 7. Result-dependent paper branches

### Branch A: prompt score succeeds

Required evidence:

- GMM-2 is preferred over GMM-1 and is not worse than GMM-3 by the declared
  gate;
- adjacent manual thresholds do not cause an unexplained collapse;
- tau 1.0 beats both count-matched random and reversed controls;
- merge beats cyclic and recent-only for Responsive heads;
- blind review and identity/temporal metrics agree on the direction;
- the selected map replicates on the frozen MovieBench-128 suite.

Possible story:

> Long-video extrapolation requires different temporal evidence for heads
> whose history readout is invariant or responsive to conditioning
> interventions. We measure that response without training, freeze a binary
> functional map, and allocate persistent versus compressed temporal evidence
> under a fixed cache budget.

Potential technical contributions:

1. training-free prompt-intervention head response measurement;
2. calibration/evaluation-separated binary functional classification;
3. bounded role-conditioned temporal cache composition;
4. causal factorization of membership, cache operator, and PF class
   mechanisms.

### Branch B: prompt score fails but one PF merge succeeds

Use PF-AR or PF-AW as a PF-derived binary prior and say so explicitly. The
paper can study whether PF's three roles are over-parameterized for long
extrapolation and whether a two-policy cache preserves quality, but simple
label merging alone is unlikely to be sufficient for a top-tier contribution.
At least one independently validated mechanism, such as lifecycle-controlled
writes or a new compression/readout operator, would still be required.

### Branch C: both prompt score and PF merges fail

Do not continue tuning thresholds on MovieBench-32. Keep native PF/v78 as the
main engineering baseline, record the binary hypothesis as negative, and move
to a different mechanism. Prompt-switch/ABA experiments should only be added
after a single-prompt method survives this screen; they are secondary and
cannot rescue a weak long-extrapolation result.

## 8. Claim and attribution boundary

- PF's three-class labels and native cache operators are prior work.
- PF-AR/PF-AW are our experimental regroupings of PF labels, not independent
  head discoveries.
- GMM/Otsu are standard statistical tools.
- A threshold selected after seeing this screen is exploratory. It must be
  frozen before MovieBench-128 and reported separately from the screen.
- Negative controls and the v96 layer-index failure must remain in the
  experimental record.

This boundary permits broad borrowing with attribution and prevents a
PF-derived engineering variant from being presented as an independently
discovered taxonomy.
