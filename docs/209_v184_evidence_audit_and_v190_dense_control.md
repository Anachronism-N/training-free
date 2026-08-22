# v184 Evidence Audit and v190 Dense-Control Update

## 1. Latest repository state

This branch now includes remote `main` through `0828484e`:

- `7127175a`: 128 videos from `all_coverage_retrieval`;
- `bcf43516`: nine uploaded VBench-Long dimension artifacts;
- `0828484e`: clarification that previous `all_recent` results were not the
  Pyramid-Forcing default three-class cache.

The uploaded generation itself is useful. Sixteen shard logs consistently show:

```text
recent=20:0 coverage=21:360 episode=22:0
coverage_policy=retrieval
```

Thus, the run is a stable 360-head Retrieval-Coverage execution on 128 prompts,
120 latent frames, seed 0. It is not a Head x Phase method.

## 2. What the uploaded metrics support

The absolute uploaded scores are:

| Dimension | Score | Status |
|---|---:|---|
| subject_consistency | 0.97391404 | descriptive |
| background_consistency | 0.96445077 | descriptive |
| temporal_flickering | 0.97030115 | descriptive |
| motion_smoothness | 0.98475612 | descriptive |
| overall_consistency | 0.24679559 | descriptive |
| dynamic_degree | 1.00000000 | invalid, all 1,920 clip decisions are `True` |
| aesthetic_quality | 0.61013172 | descriptive |
| imaging_quality | 0.68910242 | descriptive |
| temporal_style | 0.24679559 | duplicates overall score in this wrapper |

The result cannot support a treatment-effect claim for three independent
reasons:

1. `comparison_manifest.json` contains only `all_coverage_retrieval`;
2. no same-prompt, same-seed baseline is bound into that manifest;
3. `prompts/moviegen_128_full.txt` is textually different from the earlier
   `moviegen_128_qwen_v129.txt` prompt file.

Therefore, deltas reported against an old PF/all-Recent table are not paired
comparisons. In particular, `dynamic_degree=1.0` must not be used in a paper,
method gate, or configuration decision.

Run the frozen audit locally or on node 0:

```bash
python scripts/audit_v184_retrieval_evidence.py \
  --run-root runs/v184_retrieval_128 \
  --output runs/v184_retrieval_128/analysis/provenance_audit.json
```

Expected decision:

```text
operator_stability_only_reprofile_before_comparative_claim
```

## 3. Why the main experiment remains v189

The valid current evidence is asymmetric:

- v183 shows that all-head Coverage can act on motion but may reduce identity;
- v182 suggests Landmark and Retrieval are promising middle-memory operators,
  but only on an invalidated static five-head map;
- uploaded v184 confirms that all-head Retrieval is executable at scale, but
  does not contain a valid paired baseline;
- previous static head classifiers did not causally beat matched controls.

The next scientific question is therefore not another cache-operator sweep. It
is whether operator compatibility changes jointly with head identity and noisy
denoising call. v189 measures:

```text
R[operator, call, layer, head] in {Recent, Coverage}
```

for Landmark and Retrieval under a representation-complete shadow teacher.
The active trajectory remains all-Recent, so profiling cannot improve or damage
the generated trajectory used to collect the measurements.

## 4. v190 control update

If v189 passes its discovery/validation gates, v190 now compares, on the same
32 classifier-holdout prompts and seed:

1. `all_recent`;
2. `{operator}_all_coverage`, all 1,440 call-layer-head cells exposed;
3. `{operator}_compatible`, the frozen Head x Phase map;
4. `{operator}_head_only`, a call-invariant classifier fitted after averaging
   each layer/head over all noisy calls;
5. `{operator}_phase_layer_only`, a head-invariant classifier fitted after
   averaging each call/layer over all heads;
6. layer/call count-matched head-membership shift;
7. cyclic denoising-phase shift;
8. all-head dense exposure only on the primary map's active call/layer cells.

The dense-phase method is omitted when it equals all-Coverage. Every method uses
the same 9-FFE read budget and clean calls always use Recent.

The full v190 gate now requires all of the following:

- useful effect relative to all-Recent;
- support over both Head-only and Phase/Layer-only factor controls;
- support over the count-matched head-membership control;
- support over the phase-shift control;
- fewer Coverage cell-calls than all-Coverage;
- non-inferiority to all-Coverage on quality, identity/background, motion and
  temporal mechanics.

This directly tests whether classification is useful rather than merely whether
Retrieval Coverage is useful. It also prevents an all-head effect from being
misreported as evidence for a Head x Phase classifier.

The updated collector audits Dynamic Degree before using it. A constant
all-one result is retained only as ceiling non-regression and cannot support a
motion-improvement claim. Lightweight paired optical-flow diagnostics reject
repeated freezing/jump/artifact failures and localize at most four prompts for
review; they are not promoted to paper metrics.

## 5. Execution order

Use branch `codex/v178-v179-causal-validation`.

Node 0:

```bash
git pull
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh preflight
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1 \
  bash scripts/run_v189_structured_head_phase_profile_32gpu.sh smoke
```

Then run on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v189_structured_head_phase_profile_32gpu.sh profile128
```

After all shards complete, node 0:

```bash
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh package
```

Do not run v190 unless `analysis.json` reports:

```text
advance_head_phase_maps_to_causal_screen
```

The complete v190 commands remain in
`docs/208_v189_structured_head_phase_profiling.md`. No manual video review is
required before the final v190 automatic gate.

## 6. Baseline policy

PF default is not required to decide the internal Head x Phase hypothesis.
The mandatory causal controls are all-Recent and all-Coverage under the exact
same runtime. SF native should be included in a later fresh-128 confirmation if
v190 passes. PF default can be added as an external reference, but it does not
block the current experiment and must never be substituted by all-Recent in a
table label.
