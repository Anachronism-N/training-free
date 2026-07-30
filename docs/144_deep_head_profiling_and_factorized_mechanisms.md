# v144 Deep Head Profiling and Factorized Mechanism Plan

## 1. Current decision

v143 is still running. Do not stop or rerun it. Its raw profiles remain useful,
but the original global clustering is not sufficient to establish a head
taxonomy because several strong axes contain a large layer-level component.

v144 has two independent parts:

1. CPU-only hierarchical reanalysis of v143:
   - compare raw and within-layer-residual coordinates;
   - retain held-out prompt splits for every A/B context;
   - distinguish a static head property from a layer/timestep/episode program.
2. A new 128-video factorized profile:
   - control random trajectory with same-prompt/different-seed pairs;
   - change identity, scene, action, and camera one at a time;
   - keep Q, K, V, output-causal cache-policy, and spatial-topology effects
     separate.

This round is discovery, not a method-quality benchmark. Every video is still
30 seconds because late autoregressive states are part of the question.

## 2. What probably caused the earlier failures

### 2.1 No cache-mutation bug was found in prompt shadows

The v134/v141 prompt shadow path was audited again:

- the base generation runs first;
- each alternate prompt receives its own cross-attention cache;
- self-attention history is read-only;
- current K/V is recomputed from the alternate branch;
- historical K/V is copied from the same fixed generated trajectory;
- native K/V indices are checked before and after every shadow;
- CPU and CUDA RNG states are restored;
- shadow outputs cannot update the generated video.

Therefore, the prompt-sensitivity gate failure is not currently explained by
an obvious cache ownership or RNG bug.

This is narrower than saying the profiler is perfect. Projection, sampling,
and perturbation design can still hide a real effect.

### 2.2 v134 compared perturbations of unequal magnitude

The semantic branch changed one prompt slot. The paraphrase branch rewrote and
reordered the entire prompt. CPHI therefore mixed:

- semantic change;
- surface-form change;
- text-embedding distance;
- sentence-template size.

The score ordering can be reproducible while the zero threshold remains
meaningless. A failed zero-threshold split does not prove that prompt
sensitivity is absent.

### 2.3 Global stability can be layer stability

The corrected v142 policy-modulation score has high global split-half
correlation, but much of its variance is explained by layer. After subtracting
the median of the 12 heads in each layer, head-level stability is much lower.

A global 360-head clustering can therefore rediscover early/middle/late
layers and incorrectly call them head types.

v144 always reports both:

```text
raw head axis
within-layer residual = head score - median score of its 12-head layer
```

Discovery and validation splits are residualized separately. This removes a
nuisance layer offset without using PF labels or validation class labels.

### 2.4 "Long history" was only the native SF window

v134-v143 native profiles use Self-Forcing's 21-frame local-attention window.
At saturation, a 3-frame current block sees at most 18 completed history
frames. These experiments can measure recent/middle allocation and local
periodicity. They cannot alone establish persistent long-range recall.

The v143 persistent archive probe is the separate experiment that can answer
whether an old episode remains compatible after it leaves the native window.

### 2.5 v138 donors mixed all semantic factors

v138 compared own history with:

- a maximum lexical-Jaccard donor;
- fixed offset donors `+1`, `+37`, and `+73`.

This established a useful own-trajectory specificity axis, but a wrong donor
could simultaneously change identity, scene, action, camera, style, and
random seed. It could not tell which content caused the mismatch.

### 2.6 Static classes may be the wrong first model

v141 indicated that responding heads depend on episode, AR state, and
timestep. A defensible model must test three alternatives:

1. static head role;
2. layer-wide denoising-stage role;
3. static head propensity modulated by an online state gate.

The third is currently the most plausible hypothesis.

### 2.7 CFG is not active in the profiled Self-Forcing path

The repository contains a conditional/unconditional CFG implementation in
`pipeline/causal_diffusion_inference.py`. However, v143/v144 run
`pipeline/causal_inference.py`, the distilled Self-Forcing causal generator.
That path performs one conditional generator call per denoising step and does
not combine conditional and negative-prompt predictions at inference time.

Consequently:

- v143/v144 cannot support a claim about conditional versus unconditional
  CFG-head specialization;
- the `noisy`/`clean` axis is a denoising/context-refresh axis, not a CFG axis;
- a CFG axis requires a separate teacher-diffusion profiling experiment with
  matched latent input, timestep, cache state, and explicit `cond`/`uncond`
  branch labels.

This teacher-side experiment is useful later as an external-mechanism check,
but it should not be mixed into the current 128-video factorized run.

## 3. New hypotheses

### H1: trajectory-specific history access is a stable head propensity

Some heads should prefer K/V from the exact generated trajectory even when a
donor has the same text. This is measured by:

```text
same prompt, different seed
```

It is a stronger control than an unrelated wrong video.

### H2: semantic factor selectivity exists after trajectory correction

For a semantic factor `f`, the diagnostic excess is:

```text
factor response(f) - different-seed response
```

Responses are measured separately for:

- Q direction;
- historical K direction;
- historical V direction and scale;
- Q-to-donor-K compatibility;
- preferred fixed-budget cache policy;
- recent cross-frame spatial topology.

This difference-of-differences is still observational. It becomes a
functional role only after a head-selective intervention changes generation
in the predicted way.

### H3: prompt switching is primarily a state gate

The full prompt can alter cache-policy demand broadly, while the identity of
the most affected heads may depend on:

- layer;
- noisy versus clean context;
- denoising timestep;
- distance from the A-to-B boundary;
- whether stale A remains visible.

An axis is called a static head candidate only if it survives both held-out
prompt families and cross-context comparisons after layer residualization.

### H4: selection and transport roles can differ

Q/K compatibility measures history selection. V geometry measures transported
content. A head can have stable K selection but factor-sensitive V content, or
the reverse. Collapsing Q, K, and V into one prompt-sensitivity number can hide
this distinction.

### H5: motion/structure requires a second spatial axis

Temporal age allocation alone is not enough to identify motion heads. v144
adds a recent cross-frame sampled correspondence matrix and records:

- normalized attention entropy;
- same-location diagonal mass;
- expected spatial displacement;
- directional coherence;
- top-1 displacement.

The first current frame in an AR block is compared with the latest completed
history frame. These are attention-topology measurements, not optical flow.
No head may be called a motion head from these values alone.

### H6: CFG specialization is a teacher-side transfer hypothesis

If conditional and unconditional branches use measurably different
self-attention heads in the teacher diffusion model, their within-layer
branch difference can be compared with v144 semantic-factor axes. A stable
overlap would suggest that distilled Self-Forcing retains a related prompt
control propensity even though it no longer executes runtime CFG.

This is deliberately a secondary hypothesis. It must use the teacher path and
must not relabel v144 prompt shadows as CFG branches.

## 4. Relation to prior work

The implementation and paper must preserve the following attribution:

| Work | Relevant idea | What v144 does differently |
|---|---|---|
| [Pyramid Forcing](https://arxiv.org/abs/2605.13111) | QK polarity and periodicity; Anchor/Wave/Veil cache routes | PF labels are post-hoc references only. v144 does not fit clusters to PF and explicitly removes layer effects. |
| [Forcing-KV](https://arxiv.org/abs/2605.09681) and [code](https://github.com/zju-jiyicheng/Forcing-KV) | Static/dynamic heads from recent/transition attention mass | Its published threshold formula remains a reproduced diagnostic. v144 studies semantic-factor, trajectory, Q/K/V, and state axes. |
| [Head Forcing](https://arxiv.org/abs/2605.14487) | Local/anchor/memory heads and episodic memory | Its mass-based labels are references. v144 requires within-layer and context-held-out stability before role naming. |
| [HALO](https://arxiv.org/abs/2607.11081) | Motion-specific heads from displacement maps; structure heads from attention entropy | v144 borrows the motivation to inspect spatial correspondence but uses sampled AR self-attention topology and makes no optical-flow or motion-role claim. |

The safe paper claim is not "first head heterogeneity." The possible claim,
if experiments support it, is a factorized and context-conditioned functional
taxonomy for training-free long autoregressive video memory.

## 5. v143 hierarchical reanalysis

### 5.1 Raw versus layer-residual clustering

The updated clustering script accepts:

```bash
--coordinate-system raw
--coordinate-system layer_residual
```

Each feature audit reports:

- raw discovery/validation Spearman;
- layer-residual discovery/validation Spearman;
- raw and residual IQR;
- discovery and validation layer eta-squared.

Each clustering candidate also reports NMI with:

- exact layer id;
- five six-layer bands.

High cluster/layer NMI is not proof of invalidity, but it blocks a claim that
the result is a head type independent of depth.

### 5.2 Context-conditioned role audit

The updated v143 analyzer writes `all`, `discovery`, and `validation` rows to:

```text
runs/v143_multiaxis_profile/analysis/ab_context_axes.csv
```

`analyze_v144_context_conditioned_head_roles.py` then checks, for each A/B
axis:

- held-out prompt correlation per context;
- held-out top-25% head overlap;
- cross-context correlation;
- cross-context top-25% overlap;
- layer variance;
- per-head rank IQR and recurrence.

Default static-axis gates are:

```text
median layer-residual held-out rho >= 0.50
median layer-residual cross-context rho >= 0.30
```

These thresholds are screening rules, not natural physical constants.
Sensitivity values should be reported if a paper taxonomy is built from them.

## 6. v144 factorized suite

### 6.1 Prompt grid

There are 16 controlled families and 8 jobs per family:

| Variant | Prompt | Seed |
|---|---|---|
| `base` | canonical A fields | family seed |
| `seed_control` | exactly the base text | family seed + 10000 |
| `paraphrase` | identical fields, rewritten template | family seed |
| `identity` | identity only changed | family seed |
| `scene` | scene only changed | family seed |
| `action` | action only changed | family seed |
| `camera` | camera only changed | family seed |
| `full_semantic` | all eight fields changed | family seed |

Every manifest row stores token Jaccard and normalized token edit distance.
These values diagnose perturbation magnitude; they are not used to choose
head labels.

The optional dominant-factor diagnostic does not compare the four raw scores
directly. For each factor, it subtracts the layer median and fits a robust
scale on discovery families only. A head is marked `unresolved` unless the
largest standardized score is at least 0.50 and exceeds the runner-up by at
least 0.25. These are screening thresholds and require sensitivity analysis.

### 6.2 Captured states

For each 30-second video:

- AR starts: 63 and 117;
- noisy timesteps: 1000 and 500;
- clean context at both AR starts;
- total: 6 calls and 180 layer records.

This is deliberately smaller than v143 because every family already has
eight complete rollouts.

### 6.3 New artifacts

Profile version 7 adds:

```text
query_projection
history_key_projection
history_value_projection
history_value_rms
spatial_topology_metrics
causal_policy_metrics
```

The log prints one compact spatial-topology summary for every captured
mode/frame/timestep at layer 0. The final audit checks all 128 profiles,
videos, seeds, variants, calls, records, and required tensor fields.

### 6.4 Main outputs

```text
runs/v144_deep_head_profile/analysis/
  factor_state_head_observations.csv.gz
  family_head_axes.csv
  context_head_axes.csv
  context_feature_audit.csv
  context_feature_stability.csv
  head_factor_axes.csv
  feature_stability_audit.csv
  profile_contract_audit.csv
  analysis_report.json
  analysis_summary.md
```

`factor_state_head_observations.csv.gz` is intentionally retained for later
debugging. It allows analysis by layer, head, prompt family, AR state,
timestep, and factor without rerunning generation.

The three context tables keep frame 63/117 and noisy-1000/noisy-500/clean
separate. They report whether held-out and cross-context stability survives
layer residualization. A factor can therefore be retained as a state-dependent
gate even when it fails as a static head class.

They contain both raw perturbation axes and derived `*_excess_seed` axes. The
derived value subtracts the family/state-matched `seed_control` observation
before any cross-family median. This ordering prevents random trajectory
variation from being converted into an apparent episode or timestep gate.

## 7. Run commands

Use the same checkout and commit on all four nodes.

### 7.1 Prepare and smoke on node 0

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh prepare

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh smoke
```

Do not launch the 128 jobs unless smoke prints:

```text
[v144-smoke] mechanism profile contract: PASS
```

### 7.2 Run four jobs per GPU

Run one command on each node with `NODE_RANK=0,1,2,3`:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh factorized128
```

Each global shard receives exactly four dataset rows.

### 7.3 Audit and analyze on node 0

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh audit

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh analyze
```

### 7.4 Reanalyze v143 without new generation

After v143 has all 128 natural and 32 A/B profiles:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh v143_hierarchical
```

This command reruns the updated v143 analyzer, runs raw and layer-residual
clustering, and runs the context-role audit. It consumes no generation GPU
time.

### 7.5 Package small results

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v144_deep_head_profile_32gpu.sh package
```

Do not commit raw `.pt`, `.mp4`, logs, or the large gzipped observation table.
Commit the JSON/Markdown reports and compact CSV summaries.

## 8. How to interpret outcomes

### Outcome A: residual clusters are stable

Required evidence:

- multiple accepted layer-residual axes;
- stable `k` across discovery/validation and bootstrap;
- low boundary fraction;
- acceptable class balance;
- cluster membership not reducible to layer bands;
- context audit supports at least one static axis.

Then implement top-cluster, bottom-cluster, random-count-matched, and reversed
head interventions. Functional names are assigned from those results.

### Outcome B: raw clusters pass but residual clusters fail

The correct conclusion is layer/stage specialization, not static head
classes. The next method should use layer and denoising-stage policy, with at
most a weak per-head gate.

### Outcome C: factor axes are stable but context axes are not

Use a two-level mechanism:

```text
offline head propensity x online prompt/episode/timestep gate
```

This may be more novel and more accurate than forcing every head into a fixed
binary class.

### Outcome D: all factor axes fail after seed correction

Prompt semantics are not a defensible head classifier for this backbone.
Retain temporal/history-specificity axes, and use prompt changes only as an
online scene-boundary signal.

### Outcome E: spatial topology is stable

It is still only a topology class. Before calling it motion:

1. compare head displacement with optical flow on generated videos;
2. separate foreground and camera motion;
3. run motion-head versus random-head cache interventions;
4. report motion dynamics jointly with identity/consistency metrics.

## 9. Immediate priority

1. Finish v143 as currently running.
2. Pull this code and run `v143_hierarchical`.
3. Run v144 smoke.
4. If smoke passes, run factorized128.
5. Return the compact reports and selected logs.
6. Only then design the first causal head-routing experiment.

This ordering avoids spending another 128-video round on a taxonomy that is
only a layer effect or a seed artifact.
