# v191 Unseen-128 Head x Denoising-Phase Confirmation

## 1. Current synchronized state

As of 2026-08-23, remote `codex/v178-v179-causal-validation` contains no result
commit after `1d005df7`. No local `runs/v189_*` or `runs/v190_*` result package is
available for numerical analysis. The latest defensible status is therefore:

1. v189/v190 implement the joint Head x Denoising-Phase hypothesis and its
   causal controls;
2. no joint map has yet passed the complete v190 generation gate in the files
   available in this checkout;
3. v191 is a conditional confirmation protocol, not evidence that the method
   already works.

Do not start v191 merely because generation files exist. Its preparer requires
a SHA-bound v190 decision with all of these gates set to true:

```text
baseline effect
Head-only factor control
Phase/Layer-only factor control
head-membership shift control
denoising-phase shift control
selective exposure versus all-Coverage
automatic temporal safety
```

## 2. Question answered by v191

v190 uses 32 classifier-holdout prompts to decide whether the joint routing
mechanism is causally supported. v191 asks a separate question on 128 prompts
that were excluded from v189 fitting and v190 screening:

> Does the frozen joint Head x Phase route retain a useful effect under a new
> prompt distribution, without degrading native Self-Forcing or producing
> repeated freezing, jumps, or polygon artifacts?

The prompt source is frozen to MovieGen source indices 128-255 from the v180
manifest. Preparation checks 128 unique prompts and exact normalized-text zero
overlap with the complete v189 development suite.

## 3. Frozen methods

Only three methods are generated:

| Method | Purpose | Cache read |
|---|---|---|
| `sf_native` | external Self-Forcing reference | native SF runtime |
| `all_recent` | equal-budget local control | `sink1 + recent8` |
| `head_phase_joint` | v190-selected frozen method | per-cell Recent or Coverage |

For the cache methods:

```text
Recent   = sink1 + recent8                         = 9 FFE
Coverage = sink1 + structured middle4 + recent4  = 9 FFE
clean denoising read = Recent
```

`head_phase_joint` copies the exact selected v190 `4 x 30 x 12` map and
operator. No threshold, head membership, denoising call, or operator is tuned
on the 128 confirmation prompts. `all_recent` uses the same cache runtime and
read budget but an all-false route map, isolating the joint route from runtime
differences.

The cache implementation is invoked from the Pyramid-Forcing checkout because
that is where this branch's audited mixed-head runtime lives. v191 does not run
PF-native and does not use PF's three-class head labels. PF and ABA are both
outside this confirmation grid.

## 4. Scale and resource allocation

- prompts: 128 unseen prompts;
- duration: 120 latent frames, approximately 30 seconds;
- seed: 10000;
- methods: 3;
- videos: 384 total;
- hardware: exactly 4 nodes x 8 GPUs;
- load per GPU: 4 prompts per method, 12 videos total;
- VBench-Long: core-9, same prompt mapping for all methods;
- lightweight temporal diagnostics: frame step 8, CPU workers default 8.

Only rank 0 emits the full `4 x 30 x 12` schedule/readout trace for each cache
method. Every generation log is still checked for the frozen runtime markers,
budget, clean-Recent policy, map ID, operator, traceback, OOM, and cache
warnings. This avoids 32 duplicate full traces without weakening route
coverage validation.

## 5. Preregistered decision rule

The primary comparison is `head_phase_joint - all_recent`. The candidate must:

1. pass paired 95% bootstrap-CI non-inferiority:
   quality `>= -0.15`, identity/background `>= -0.001`, temporal mechanics
   `>= -0.002`, and informative Dynamic Degree `>= -0.02`;
2. have a paired CI lower bound above zero on at least one of quality,
   identity/background, temporal mechanics, or informative Dynamic Degree;
3. be on the four-axis primary Pareto front;
4. pass automatic temporal guards against both all-Recent and SF native;
5. pass the looser external-reference CI margins against SF native:
   quality `>= -0.20`, identity/background `>= -0.0015`, temporal mechanics
   `>= -0.003`, and informative Dynamic Degree `>= -0.02`.

These are development tolerances fixed before v191 generation, not universal
equivalence margins. If Dynamic Degree is exactly 1 for every prompt and
method, it supports ceiling non-regression only. It is excluded from the
positive-effect rule and cannot support a motion-improvement claim. A constant
non-ceiling Dynamic Degree fails the gate as an invalid measurement.

The automatic temporal guard permits at most `ceil(N/32)` differential warning
prompts: one for v190/32 and four for v191/128. It is a failure detector and
review localizer, not a paper metric.

Only a full automatic pass creates a manual review queue, capped at four
prompts. Failure creates no broad review task and stops this method.

## 6. Server execution

Use branch `codex/v178-v179-causal-validation`.

### 6.1 Refresh the v190 decision without regenerating videos

The v191 preparer requires v190 report version 6, which binds the comparison
manifest, VBench summary, and temporal CSV by SHA-256. Existing valid v190
videos and VBench parts are reused:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git checkout codex/v178-v179-causal-validation
git pull

NODE_RANK=0 bash scripts/run_v190_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v190_vbench_long.sh decision
```

Stop unless the output is exactly:

```text
advance_head_phase_method_to_fresh128
```

### 6.2 Prepare and smoke v191

Node 0:

```bash
NODE_RANK=0 bash scripts/run_v191_head_phase_confirmation_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v191_head_phase_confirmation_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2 \
  bash scripts/run_v191_head_phase_confirmation_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v191_head_phase_confirmation_32gpu.sh audit-smoke
```

No manual video review is required after smoke. The audit must print
`[v191-audit] PASS` for all three methods.

### 6.3 Generate 384 videos

Run once on each of four nodes with relative ranks 0-3:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v191_head_phase_confirmation_32gpu.sh generate128
```

After every node finishes, node 0:

```bash
NODE_RANK=0 bash scripts/run_v191_head_phase_confirmation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v191_head_phase_confirmation_32gpu.sh audit-confirm
```

Do not start metrics if `audit-confirm` fails.

### 6.4 VBench-Long and automatic diagnostics

Node 0 prepares the prompt-correct hardlink grid:

```bash
NODE_RANK=0 bash scripts/run_v191_vbench_long.sh prepare
```

Each of four nodes then runs:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v191_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v191_vbench_long.sh eval
```

Temporal diagnostics can run on node 0 while GPU metrics are finishing:

```bash
NODE_RANK=0 V191_TEMPORAL_WORKERS=8 \
  bash scripts/run_v191_vbench_long.sh temporal
```

Final collection on node 0:

```bash
NODE_RANK=0 bash scripts/run_v191_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v191_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v191_vbench_long.sh decision
NODE_RANK=0 bash scripts/run_v191_head_phase_confirmation_32gpu.sh package
```

For interrupted VBench jobs, use one node only:

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v191_vbench_long.sh resume-missing
```

## 7. Files to return

Push the small evidence files, not videos:

```text
runs/v191_head_phase_confirmation/inputs/
runs/v191_head_phase_confirmation/smoke/audits/
runs/v191_head_phase_confirmation/confirm128/contracts/
runs/v191_head_phase_confirmation/confirm128/audits/
runs/v191_head_phase_confirmation/confirm128/published_manifest.json
runs/v191_head_phase_confirmation/confirm128/metrics/vbench_core9_summary.json
runs/v191_head_phase_confirmation/confirm128/metrics/temporal_diagnostics.csv
runs/v191_head_phase_confirmation/confirm128/metrics/temporal_diagnostics.contract.json
runs/v191_head_phase_confirmation/confirm128/analysis/
```

Upload the diagnostic archive only on failure. The final decision file is:

```text
runs/v191_head_phase_confirmation/confirm128/analysis/
  v191_head_phase_confirmation.json
```

## 8. What happens next

If v191 passes, freeze the method before running seed replication, 60-second
stress, and a second model. Do not retune the map on v191. If it fails, inspect
the failed automatic gate and return to profiling/operator design; do not rescue
the claim by selecting favorable prompts or changing thresholds after seeing
the confirmation results.
