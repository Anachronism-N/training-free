# v185 PF Audit and v189/v190 Factorial Extension

## 1. Latest synchronized result

Remote `main` at `ee0d7bfc` adds sixteen logs under
`runs/v185_pf_baseline/logs/`. The logs jointly cover prompt completion markers
1 through 128 exactly once. Every shard reports:

```text
Number of prompts: 128
Loading PyramidKV config from configs/head_configs/best_labels.csv
```

No shard contains a traceback, OOM, cache-compatibility override, or
history-polarity override. The run used `moviegen_128_full.txt`, 120 latent
frames, and seed 0, matching the v184 generation settings. The current PF map
contains 172 label-1, 156 label--1, and 32 label-2 heads.

This is valid evidence that PF-native generation completed. It is not yet a
paper-ready artifact because:

1. only logs were uploaded; raw PF videos remain on the server;
2. config, prompt, checkpoint, and head-map hashes were not printed into a
   frozen contract before generation;
3. the name `v185` already belongs to the earlier recovered-v181 long60
   evaluation.

The repository therefore calls this result `legacy_v185_pf_native_baseline`.
Do not rerun its 128 videos. Audit the existing server files instead:

```bash
NODE_RANK=0 bash scripts/run_v185_pf_baseline.sh audit-fast
```

Use `audit` instead of `audit-fast` only when a full ffmpeg decode pass is
needed. Expected decision with server videos present:

```text
reuse_for_development_metrics
```

Without raw videos, the expected local decision is:

```text
generation_logs_complete_media_not_uploaded
```

## 2. Why v189 now emits factorized classifiers

The primary hypothesis is an interaction:

```text
the useful history operator depends jointly on head identity and denoising call
```

The previous controls could show that changing heads or shifting calls alters
generation, but they did not answer whether a simpler Head-only or Phase-only
classifier explains the same effect. The updated v189 analyzer freezes three
maps from the same 64 discovery and 32 validation prompts:

### Joint Head x Phase

Each `(call, layer, head)` is gated independently. This remains the primary
candidate.

### Head-only

For each `(layer, head)`, Coverage gain is averaged over all four noisy calls
before applying the same discovery gain, validation, budget, and residual-energy
gates. Its selected membership is copied unchanged to every call.

### Phase/Layer-only

For each `(call, layer)`, Coverage gain is averaged over all twelve heads before
applying the same gates. A selected cell routes all twelve heads to Coverage.

The last 32 prompts remain untouched classifier-holdout data. Factor diagnostics
are written to:

```text
runs/v189_structured_head_phase_profile/analysis/factor_scores.csv
runs/v189_structured_head_phase_profile/analysis/maps/*head_only*.json
runs/v189_structured_head_phase_profile/analysis/maps/*phase_layer_only*.json
```

## 3. Existing v189 jobs do not need to restart

The profiling artifact contract and runtime were not changed. If v189 `.pt`
shards are already running or complete, finish them and rerun only the cheap
audit/analyzer steps after pulling this commit:

```bash
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh status
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh package
```

Do not start v190 unless the new analysis still reports:

```text
advance_head_phase_maps_to_causal_screen
```

## 4. Updated v190 causal screen

For every passing operator, v190 can now contain:

```text
all_recent
all_coverage
joint Head x Phase
Head-only
Phase/Layer-only
count-matched membership shift
cyclic phase shift
active-call/layer dense control
```

Exact duplicate maps are removed before generation and aliased to their
exact-equivalent method during analysis. A factor identical to all-Recent or
another control therefore costs no extra videos; a factor identical to the
joint primary cannot support an interaction claim. All retained methods use the
same holdout prompts, seed, 9-FFE read budget, clean-call Recent policy, and
audited runtime.

The joint method advances only if it:

1. improves motion without violating quality, identity/background, or temporal
   tolerances relative to all-Recent;
2. is supported over Head-only and Phase/Layer-only;
3. is supported over membership-shift and phase-shift controls;
4. uses fewer Coverage cell-calls while remaining non-inferior to all-Coverage.

The v190 collection step also computes lightweight paired Farneback diagnostics
for all generated videos. Repeated differential freezing, temporal jumps, or
luminance/edge failures reject a candidate before broad visual review. These
signals are safety diagnostics, not paper metrics.

VBench Dynamic Degree is audited before it enters the gate. If it varies, the
original positive-motion criterion applies. If every method and prompt is
exactly `1.0`, it can support ceiling-level motion non-regression only. The
candidate must then show a positive quality, identity/background, or temporal
effect and the paper must not claim improved motion from that dimension.

No manual review is needed before this automatic gate. If the gate passes, only
the four automatically selected disagreement or warning prompts need visual
inspection. `collect` creates the temporal CSV automatically; it may also be
started separately on node 0 with:

```bash
NODE_RANK=0 V190_TEMPORAL_WORKERS=8 \
  bash scripts/run_v190_vbench_long.sh temporal
```

## 5. Current scientific status

The uploaded PF logs do not change the method decision. They preserve an
optional external baseline without consuming generation time. The main next
experiment remains v189 followed conditionally by v190.

The intended contribution is now falsifiable at three levels:

1. structured long-history Coverage is useful relative to Recent;
2. selecting where it is used is useful relative to all-Coverage;
3. joint Head x Denoising-Phase selection is useful relative to classifiers
   using either factor alone.

Failure at level 3 means the paper must simplify the classifier claim instead
of presenting the joint interaction as an innovation.
