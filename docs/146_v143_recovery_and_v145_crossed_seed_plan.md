# v143 Recovery and v145 Crossed-seed Head Profiling

Date: 2026-07-31

## 1. Decisions

The next experiments should not reuse the original interpretation that 85
stable v144 features imply semantic head classes.

The corrected evidence is:

- 77/77 raw perturbation features pass split stability;
- only 8/66 `semantic - seed_control` features pass;
- only 3/66 corrected context features pass the median-context gate;
- none passes every captured context at rho >= 0.30;
- dominant semantic labels agree for only 19/64 heads resolved in both splits;
- corrected V, policy, and spatial-topology axes do not pass.

The current working model is therefore:

```text
offline temporal/drift propensity x online prompt/denoising state
```

It is not a fixed identity/scene/action/camera taxonomy.

## 2. v143 persistent-probe fix

### 2.1 Failure mechanism

The old A-B configuration used:

```text
persistent captures: 0, 18, 36, 54
persistent probes:   54, 57, 60, 75, 78, 117
```

At frame 54, noisy calls run before the clean-context call. The frame-54
snapshot is created only during the clean call, so the old probe saw
`[0,18,36]` but required `[0,18,36,54]`.

### 2.2 Correct contract

A persistent probe may use only captures whose frame is strictly less than the
current AR start. This gives:

| Probe frame | Available persistent captures |
|---:|---|
| 54 | 0, 18, 36 |
| 57 and later | 0, 18, 36, 54 |

This is not merely a relaxed assertion. It prevents current-block K/V from
leaking into a diagnostic that is supposed to represent historical memory,
and keeps noisy/clean frame-54 probes on the same archive.

The v143 smoke and full audit now validate this rule for every record.

## 3. Recovering v143

The committed `no_stable_k` result is incomplete:

- it contains no `v143_natural.*` or `v143_ab.*` accepted feature;
- A-B profiles were unavailable;
- it was generated in raw coordinates.

After the fixed A-B profiles finish, `cluster` now creates:

```text
runs/v143_multiaxis_profile/
  clustering/                       # raw coordinate
  clustering_layer_residual/        # within-layer coordinate
  context_conditioned_roles/
```

Both coordinate systems receive rho-threshold and leave-one-group
sensitivity analysis. Only the layer-residual result can support a claim about
head identity beyond layer-wide behavior.

### 3.1 v143 commands

Use the same commit on all nodes. Override the checkpoint path if the cluster
stores it in `/tmp`.

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

export SF_CHECKPOINT=/tmp/self_forcing_dmd.pt

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v143_multiaxis_profile_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v143_multiaxis_profile_32gpu.sh smoke_ab
```

Do not launch `ab32` unless `smoke_ab` prints:

```text
[v143-smoke-ab] contract: PASS
```

Then run on the four nodes:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh ab32
```

On node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh audit
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh analyze
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh cluster
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v143_multiaxis_profile_32gpu.sh package
```

## 4. Why v145 changes the design

v144 used one same-prompt different-seed rollout as a noise estimate. It then
subtracted two non-negative descriptor distances:

```text
distance(base, semantic variant with same seed)
-
distance(base, seed control with different seed)
```

This does not test whether the semantic effect itself repeats under a second
generation trajectory. It also cannot distinguish a factor-specific direction
from a generic head that is sensitive to any rollout change.

v145 crosses every selected prompt factor with two independent seeds.

## 5. v145 suite

```text
16 prompt families
x 2 seed replicates
x 5 prompt variants
= 160 videos
```

Variants:

| Variant | Purpose |
|---|---|
| `base` | paired reference under each seed |
| `paraphrase` | surface-form control |
| `identity` | identity-only change |
| `scene` | scene-only change |
| `full_semantic` | complete regime change |

Identity and scene cover the central long-video consistency targets.
Paraphrase detects token/template sensitivity, while full semantic change is a
proxy for a prompt-switch boundary. Action and camera are deferred because
v144 did not show factor-specific transport or policy evidence, and including
them would expand the first crossed run to 224-256 videos.

Each variant is compared with the base generated using exactly the same seed.
The entire comparison is repeated with a second seed.

The runner does not rely on dataset ordering for this pairing. The manifest
declares the seed for every job, Self-Forcing applies it through
`seed_for_job()`, and both the smoke test and full audit require the recorded
runtime seed to match the declared/reference seed.

## 6. Measurements

At frames 63 and 117, v145 captures noisy timesteps 1000/500 and clean
context. It records:

- Q direction;
- historical K direction;
- historical V direction and V scale;
- fixed-budget causal-policy response.

Spatial topology is disabled because no seed-corrected topology feature passed
v144. This reduces storage and avoids repeating a disproven axis.

For each factor and head, v145 reports:

1. effect magnitude under seed 0 and seed 1;
2. held-out family split Spearman after layer residualization;
3. seed-replicate Spearman after layer residualization;
4. cosine between the projected factor-effect directions under both seeds;
5. same-factor direction minus cross-factor direction;
6. state-resolved versions of these measurements.

A screening candidate must satisfy all four:

```text
family split rho >= 0.30
seed replicate rho >= 0.30
same-factor delta cosine >= 0.05
same-minus-other-factor cosine >= 0.02
```

These are discovery thresholds, not functional labels.

## 7. v145 commands

On node 0:

```bash
export SF_CHECKPOINT=/tmp/self_forcing_dmd.pt

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh prepare

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh smoke
```

After smoke passes, run five videos per GPU:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh crossed160
NODE_RANK=1 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh crossed160
NODE_RANK=2 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh crossed160
NODE_RANK=3 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh crossed160
```

On node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh audit
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh analyze
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v145_crossed_seed_head_profile_32gpu.sh package
```

## 8. Decision table

| Result | Interpretation | Next causal experiment |
|---|---|---|
| Q repeats across seeds, K/V do not | online prompt/state gate | gate cache routing from current Q state |
| K repeats and is factor-specific | stable history-selection propensity | top/bottom K groups with retrieval vs recent |
| V repeats and is factor-specific | stable content-transport propensity | head-selective V memory intervention |
| policy repeats across seeds | direct cache-policy demand | predicted policy vs swapped/random controls |
| only seed susceptibility repeats | drift amplification, not semantics | top/bottom drift-susceptibility routing |
| no crossed axis passes | abandon semantic head classes | retain temporal reach and dynamic per-state scoring |

No 16/32/128-prompt method comparison should start before at least one
head-selective intervention beats a count-matched random map on a single
30-second prompt.
