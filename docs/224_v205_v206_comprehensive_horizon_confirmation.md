# v205-v206 Comprehensive Horizon Confirmation

## 1. Repository status on 2026-09-08

The latest remote `main` is still `dedbf4f5`. No v189, v200, v201, or v204
server result artifact newer than the existing branch was available in GitHub at
the time of this implementation.

One blocking v201 analysis bug was found locally:

- `bind_temporal_diagnostics.verify_contract()` expects
  `(contract, comparison_manifest, temporal_csv)`.
- v201 passed `(comparison_manifest, temporal_csv, contract)`.
- Generated videos and VBench parts are unaffected, but old v201 `collect` cannot
  produce a trustworthy decision until it is rerun with the fix in this commit.
- The v201 decision now also records SHA256-bound comparison, VBench, and temporal
  sources. v205 refuses an old unbound decision.

## 2. Experimental question

The current method is a frozen **Head x Denoising Phase x AR Horizon** routing map.
At every noisy denoising call, each layer/head reads one of two equal-budget views:

- Recent: `sink1 + recent8`.
- Coverage: `sink1 + structured-middle4 + recent4`.

The structured middle operator is either semantic landmark or semantic retrieval.
The v189-v200 profiler determines which head/phase cells benefit from Coverage and
how that assignment changes with AR horizon. Clean updates always use Recent. The
read budget is fixed at 9 frame equivalents for both routes.

v201 is the causal 32-prompt development screen. It compares the horizon map with
canonical SF and with equal-exposure static/shifted controls. v205 and v206 do not
change the method or tune a threshold.

## 3. v205: primary profiling-disjoint 128 confirmation

### Frozen design

- Gate: only candidates listed in v201 `selected_for_fresh128` can run.
- Methods: `sf_native` plus one or two selected horizon candidates.
- PF is not evaluated as a baseline. The Pyramid-Forcing checkout is only the
  existing runtime host for the cache implementation.
- Preflight verifies that SF and the cache runtime share all sampling-critical
  fields (denoising steps, guidance, timestep shift, negative prompt, shape, and
  AR block size). The deliberate attention difference is frozen: native SF uses
  `local_attn_size=21`, while the cache runtime exposes full history for the
  proposed selector.
- Candidate logs must report neutral labels `10/11`, `exclusive_owner=true`, the
  exact structured operator, and the frozen horizon-map id. Therefore PF's
  legacy `-1/1/2` head semantics and native stride/cyclic/merge routes cannot
  silently contribute to a passing run.
- Prompts: MovieGen source indices 128-255 from the frozen v180 manifest. They are
  disjoint from v189/v201, but they were used by earlier project experiments.
- Leakage check: normalized prompt text must have zero overlap with all 128 v189
  profiling/development prompts.
- Duration: 120 latent frames, about 30 seconds / 477 decoded frames.
- Seed: 20500 with per-prompt reseeding.
- Hardware: exactly 4 nodes x 8 GPUs.
- Schedule traces: one complete cache trace per candidate; all media and all shard
  logs are still audited.
- Tracked SF/PF runtime code, YAML files, and the complete `pyramidkv` package
  must be clean in both the index and worktree before generation starts.

### Primary decision

For each candidate versus `sf_native`, v205 computes prompt-paired results over the
full video and late half. Promotion excludes Dynamic Degree and requires:

1. All five primary axes pass their frozen confidence-bound non-inferiority margin.
2. At least one primary axis has bootstrap CI lower bound above zero after BH
   correction at `q <= 0.10`.
3. Automatic temporal safety passes against SF.

The five axes are quality without Dynamic Degree, identity/background, temporal
mechanics, semantic alignment, and visual quality. A separate late-half flag states
whether the evidence specifically supports long-horizon improvement.

The camera-compensated motion analysis compares local residual motion against SF.
It can support a motion claim only after the paired quality confirmation passes.
The automatic decision requires no human review. At most four targeted pairs are
listed after a pass for perceptual calibration.

## 4. v206: conditional robustness matrix

v206 cannot be prepared unless v205 has at least one confirmed candidate.

| Scope | Prompts | Frames | Seed | Isolated factor |
|---|---:|---:|---:|---|
| `seed20600_30s_128` | 128 | 120 | 20600 | new stochastic seed |
| `long60_seed20500_32` | 32 | 240 | 20500 | 60-second duration |

The 60-second prompts are positions `0,4,...,124` from v205. This systematic rule is
frozen before looking at v205 per-prompt scores.

The new-seed scope must reproduce at least one v205 significant axis with a new
confidence-supported gain. The long scope must preserve full/late non-inferiority,
reproduce a v205 target directionally, retain a positive late-half axis, and pass
temporal safety. The combined decision additionally requires:

- a positive two-seed prompt-paired pooled bootstrap interval;
- camera-compensated motion safety at both 30 and 60 seconds.

## 5. Execution order

All commands below are run from the repository root. Set `NODE_RANK` to 0, 1, 2,
or 3 on the corresponding node. Shared storage is assumed.

### P0: repair and finish v201 decision

Run on node 0 after pulling this commit:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v201_vbench_long.sh collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v201_vbench_long.sh decision
```

If v201 prints no selected candidate, stop. v205 correctly fails closed.

### P1: v205 generation

Node 0 only:

```bash
NODE_RANK=0 bash scripts/run_v205_horizon_confirmation_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v205_horizon_confirmation_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v205_horizon_confirmation_32gpu.sh audit-smoke
```

Then run concurrently on all four nodes:

```bash
NODE_RANK=${RANK} NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v205_horizon_confirmation_32gpu.sh generate128
```

After every node completes, run on node 0:

```bash
NODE_RANK=0 bash scripts/run_v205_horizon_confirmation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v205_horizon_confirmation_32gpu.sh audit-confirm
```

### P1: v205 VBench and automatic diagnostics

Prepare on node 0, then split and evaluate concurrently on all four nodes:

```bash
NODE_RANK=0 bash scripts/run_v205_vbench_long.sh prepare

NODE_RANK=${RANK} NUM_NODES=4 bash scripts/run_v205_vbench_long.sh split
NODE_RANK=${RANK} NUM_NODES=4 bash scripts/run_v205_vbench_long.sh eval
```

Camera motion can run concurrently with VBench evaluation:

```bash
NODE_RANK=${RANK} NUM_NODES=4 bash scripts/run_v205_vbench_long.sh motion-compute
```

Collect on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v205_vbench_long.sh collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v205_vbench_long.sh motion-collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v205_vbench_long.sh motion-analyze
NODE_RANK=0 bash scripts/run_v205_vbench_long.sh decision
```

### P2-P3: v206, only after a v205 pass

Prepare once:

```bash
NODE_RANK=0 bash scripts/run_v206_horizon_robustness_32gpu.sh prepare
```

For each scope, smoke on node 0 and then generate on all nodes:

```bash
SCOPE=seed20600_30s_128 NODE_RANK=0 \
  bash scripts/run_v206_horizon_robustness_32gpu.sh smoke
SCOPE=seed20600_30s_128 NODE_RANK=0 \
  bash scripts/run_v206_horizon_robustness_32gpu.sh audit-smoke
SCOPE=long60_seed20500_32 NODE_RANK=0 \
  bash scripts/run_v206_horizon_robustness_32gpu.sh smoke
SCOPE=long60_seed20500_32 NODE_RANK=0 \
  bash scripts/run_v206_horizon_robustness_32gpu.sh audit-smoke

SCOPE=seed20600_30s_128 NODE_RANK=${RANK} NUM_NODES=4 \
  bash scripts/run_v206_horizon_robustness_32gpu.sh generate
SCOPE=long60_seed20500_32 NODE_RANK=${RANK} NUM_NODES=4 \
  bash scripts/run_v206_horizon_robustness_32gpu.sh generate
```

Audit both on node 0:

```bash
NODE_RANK=0 bash scripts/run_v206_horizon_robustness_32gpu.sh audit-all
```

For each `SCOPE`, run the v206 VBench pipeline using the same four-node pattern:

```bash
SCOPE=${SCOPE} NODE_RANK=0 bash scripts/run_v206_vbench_long.sh prepare
SCOPE=${SCOPE} NODE_RANK=${RANK} NUM_NODES=4 bash scripts/run_v206_vbench_long.sh split
SCOPE=${SCOPE} NODE_RANK=${RANK} NUM_NODES=4 bash scripts/run_v206_vbench_long.sh eval
SCOPE=${SCOPE} NODE_RANK=${RANK} NUM_NODES=4 bash scripts/run_v206_vbench_long.sh motion-compute

SCOPE=${SCOPE} NODE_RANK=0 NUM_NODES=4 bash scripts/run_v206_vbench_long.sh collect
SCOPE=${SCOPE} NODE_RANK=0 NUM_NODES=4 bash scripts/run_v206_vbench_long.sh motion-collect
SCOPE=${SCOPE} NODE_RANK=0 NUM_NODES=4 bash scripts/run_v206_vbench_long.sh motion-analyze
```

After both scopes finish:

```bash
NODE_RANK=0 bash scripts/run_v206_vbench_long.sh decision
```

## 6. Priority under limited time

1. v201 `collect` and `decision`.
2. Complete v205 generation and paired VBench decision.
3. Complete v205 camera-compensated motion evidence.
4. Run v206 new-seed 128x30s.
5. Run v206 32x60s.

Do not start v206 merely to keep GPUs busy if v205 fails. In that case, upload the
v205 small artifacts and return to profiling or route redesign.

## 7. Artifacts to upload

The required small artifacts are manifests, audits, metric JSON/CSV, analysis JSON,
logs, and the one route trace per candidate. Videos do not need to be committed.

```bash
NODE_RANK=0 bash scripts/run_v205_horizon_confirmation_32gpu.sh package
NODE_RANK=0 bash scripts/run_v205_vbench_long.sh package
NODE_RANK=0 bash scripts/run_v206_horizon_robustness_32gpu.sh package
```

The most important decision files are:

- `runs/v205_horizon_confirmation/confirm128/analysis/v205_horizon_confirmation.json`
- `runs/v205_horizon_confirmation/confirm128/analysis/v205_continuous_motion.json`
- `runs/v206_horizon_robustness/analysis/v206_horizon_robustness.json`

## 8. Paper interpretation

- A v205 pass is sufficient evidence for a statistically supported improvement
  over the native SF baseline on this checkpoint and profiling-disjoint suite. It
  does not establish project-history-unseen generalization, beat PF, or claim SOTA.
- A v205 late-half pass strengthens the long-video extrapolation claim.
- A v206 pass supports seed and duration robustness within the checkpoint.
- Camera-motion support distinguishes real local motion improvement from increased
  camera movement; safety-only results support non-collapse but not a motion gain.
- The v201 static/shift controls remain the causal evidence for the AR-horizon
  routing mechanism. v205-v206 are frozen efficacy and robustness confirmation,
  not another opportunity to tune the classifier.
- Before a final paper submission, add one external prompt suite that has not been
  used anywhere in the project history; report it as the true unseen-prompt test.
