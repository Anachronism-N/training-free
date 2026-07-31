# v149: Calibrated Susceptibility-Leverage Head Profiling

## 1. Purpose

v148 established that the v145 rankings predict downstream perturbation, but
it did not establish three independent K, V, and policy mechanisms. The main
confound is intervention strength: a top-ranked group can change the final
prediction more simply because the local intervention changed that group more.

v149 separates two measurable quantities:

1. **Local susceptibility**: how strongly a head group's projected attention
   output changes under an unscaled mechanism intervention.
2. **Downstream causal leverage**: how strongly the final flow/x0 changes after
   every compared group receives the same projected local perturbation norm.

This experiment is read-only with respect to the native SF trajectory and KV
cache. It profiles one-step causal propagation at AR frame 117; it does not
claim video-quality improvement.

## 2. Pre-registered hypotheses

### H1: K susceptibility

The v145 `full_semantic / k_shift` ranking is expected to identify heads whose
local output is especially sensitive to old-history key misalignment.

Required evidence:

- susceptibility top > bottom;
- susceptibility top > two middle-rank random maps;
- the effect reproduces in the same denoising context;
- a same-layer, same-PF-label high/low comparison remains positive.

Calibrated leverage is reported separately and is not required for H1.

### H2: Policy leverage

The v145 `full_semantic / policy_shift` ranking may identify heads for which a
small cache-policy change has a disproportionately large downstream effect.

Required evidence:

- calibrated-leverage top > bottom;
- calibrated-leverage top > two middle-rank random maps;
- same-PF-label high > low under the calibrated policy contrast.

This is the most important new hypothesis produced by the v148 post-hoc audit.

### H3: V transport

The V axis remains a falsification control. It is retained because v148 showed
a group-level dose effect, but it is not part of the proposed method unless
both calibrated leverage and PF-matched controls pass.

## 3. Interventions

All interventions preserve the current block.

### 3.1 `key_shift`

- Preserve the newest four historical frames.
- Cyclically shift only the older K frames by one frame.
- Keep V in its original order.

This probes sensitivity to historical addressing/correspondence.

### 3.2 `value_shift`

- Preserve the newest four historical frames.
- Keep K fixed.
- Cyclically shift only the older V frames by one frame.

This probes sensitivity to transported historical content.

### 3.3 `policy_contrast`

This is an equal-budget contrast rather than `recent4` versus native:

- left candidate: `uniform8`, consisting of four uniformly sampled old frames
  and the newest four frames;
- right candidate: `recent8`, consisting of the newest eight frames;
- raw direction: `attention(uniform8) - attention(recent8)`.

The direction is added to the native head output and then calibrated. Both
candidates contain eight complete frames, so token count is not a confound.

## 4. Projected calibration

For selected heads in each layer:

```text
delta_raw  = candidate - native_selected
delta_proj = W_o(selected columns) * delta_raw
y_native   = W_o * native + b

alpha = target * RMS(y_native) / RMS(delta_proj)
output_selected = native_selected + alpha * delta_raw
```

The default target is `0.05`. Calibration is performed independently in every
layer, profile, probe, and denoising context using the real SF output
projection. The implementation records:

- raw and applied pre-projection relative RMS;
- raw and applied projected relative RMS;
- requested and applied scale;
- target-relative error;
- clipping status.

The run is invalid if:

- native replay relative RMS exceeds `1e-4`;
- any calibration scale is clipped;
- projected target-relative error exceeds 2%;
- any scale falls outside `[0.02, 50]`;
- a K/V shift changes at most one old frame;
- `uniform8` and `recent8` resolve to equal or non-eight-frame sets.

## 5. Core experiment

Each v145 axis supplies four disjoint maps per layer:

- top-3;
- bottom-3;
- random-0: three middle-ranked heads;
- random-1: the other three middle-ranked heads.

The core plan contains:

```text
3 axes x 3 interventions x (top, bottom)       = 18 probes
3 matched axis/intervention cells x 2 randoms  =  6 probes
3 same-PF-label controls x (high, low)         =  6 probes
                                                    ---------
                                                     30 probes
```

With native replay and two denoising contexts (`t=1000`, `t=500`), each profile
contains 62 downstream records.

Scale:

- 32 diverse MovieBench Qwen-rewrite prompts;
- two independent seeds per prompt;
- 64 profiles;
- 120 latent frames, approximately 30 seconds;
- four nodes, eight GPUs per node.

## 6. Dose experiment

The optional dose suite uses top/bottom `k=1,2,3,4` for each matched axis and
intervention:

```text
3 axes x 4 doses x (top, bottom) = 24 probes
```

It uses 16 prompts and two seeds, for 32 profiles. Every dose is calibrated to
the same projected RMS, so the test measures whether the direction represented
by a larger group is more reproducible. **Monotonic growth from dose 1 to dose
4 is not a required gate.**

## 7. Gates and interpretation

For one comparison to qualify in a non-pooled context:

- median paired log-ratio is at least `log(1.05)`;
- bootstrap mean CI lower bound is positive, or win rate is at least 0.65;
- prompt-level effect has seed-replicate Spearman at least 0.30.

The analyzer reports these channels independently:

| Channel | Metric | Meaning |
|---|---|---|
| susceptibility | raw projected replacement RMS | local response before calibration |
| leverage | final x0 relative RMS | propagation after equal-strength calibration |

Interpretation:

| Result | Allowed conclusion |
|---|---|
| K susceptibility G1/G2 pass | PF-independent prompt-conditioned historical susceptibility |
| Policy leverage G1/G2 pass | PF-independent downstream cache-policy leverage |
| V leverage G1/G2 pass | candidate content-transport leverage axis |
| Only G1 passes | ranking is useful but may proxy PF temporal classes |
| Specificity fails | do not name the coordinate as an intervention-specific mechanism |
| Calibration G0 fails | discard the entire run |

Even a full pass only justifies a trajectory-level method experiment. It does
not establish improved long-video generation.

## 8. Commands

Run preparation and the four-profile smoke on node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh smoke_core
```

Do not continue unless the log contains:

```text
[v149-smoke] calibrated replay contract: PASS scale_range=[...,...]
```

Run core on all four nodes:

```bash
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh core64
NODE_RANK=1 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh core64
NODE_RANK=2 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh core64
NODE_RANK=3 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh core64
```

Analyze on node 0:

```bash
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh audit_core
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh analyze_core
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh package
```

`package` accepts a completed core run and does not require dose results.

The dose suite may run independently on all four nodes:

```bash
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh dose32
NODE_RANK=1 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh dose32
NODE_RANK=2 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh dose32
NODE_RANK=3 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh dose32
```

Then:

```bash
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh audit_dose
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh analyze_dose
NODE_RANK=0 bash scripts/run_v149_calibrated_causal_profile_32gpu.sh package
```

Progress:

```bash
bash scripts/run_v149_calibrated_causal_profile_32gpu.sh status
```

## 9. Debug information to retain

Keep the complete profile files and logs. The packaged analysis contains:

- `profile_audit.csv`;
- `downstream_observations.csv.gz`;
- `probe_effect_summary.csv`;
- `channel_comparisons.csv`;
- `channel_specificity.csv` for core;
- `report.json`;
- `report.md`;
- frozen probe plans and suite metadata.

When reporting a failure, provide:

1. smoke `scale_range`;
2. maximum calibration error and clipped-layer count;
3. the first traceback or assertion;
4. `report.json`;
5. `channel_comparisons.csv`.

## 10. Next decision

If K susceptibility and policy leverage both pass PF-matched controls, the next
method should use a two-coordinate continuous router:

- susceptibility controls how cautiously historical K/V may be rewritten;
- leverage controls which heads receive conservative versus prompt-switch-aware
  cache policies.

The first trajectory task should then be AB scene switching. ABA recall remains
deferred until the AB forgetting/retention behavior is understood.
