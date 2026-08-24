# v193 Camera-Compensated Motion Diagnostics

## 1. Latest synchronization and current decision

The branch was synchronized with remote `main` at `7ba5082f`. The new remote
evidence contains the legacy v185 PF-native core-9 VBench-Long result and a
TorchVision-RAFT rerun of Dynamic Degree. The rerun still reports exactly
`dynamic_degree=1.0`.

This does not show that every video has equally good motion. It shows that this
binary VBench dimension is saturated on the current 128-prompt suite. It can be
used only as a motion non-regression check and cannot support a claim that one
method improves motion.

No new v191 or v192 evidence was uploaded. The experiment order therefore does
not change:

1. finish the frozen v191 unseen-128 confirmation;
2. run v192 seed and 60-second robustness only if v191 passes;
3. use v193 on the same videos to diagnose the identity-motion trade-off;
4. do not start cross-model transfer, ABA, or a new cache search before these
   automatic gates are known.

## 2. Why v193 is needed

Raw optical flow and Dynamic Degree both confound two different phenomena:

- global camera motion, such as a pan, zoom, or rotation;
- local residual motion, such as a person walking or an object changing pose.

A method can obtain a high raw-motion score because the camera moves while the
subject is almost frozen. Conversely, a method can retain identity by reducing
all motion. Neither behavior supports the desired paper claim.

v193 fits a robust global affine displacement field to every sampled frame
pair. The fit uses deterministic iteratively reweighted least squares initialized
from median translation. It then subtracts the global field from dense
Farneback flow and reports local residual motion separately.

All speeds are normalized as fractions of the resized frame diagonal per
second (`ndps`). This makes 30-second and 60-second results comparable despite
different sampling intervals. v193 is still a diagnostic, not a validated
perceptual metric.

## 3. Frozen diagnostic signals

The primary local-motion magnitude is:

```text
residual_motion_p90_ndps_median
```

It is the per-video median of the spatial 90th percentile of residual flow.
The 90th percentile focuses on a moving region without requiring it to occupy
most of the frame.

Two complementary coverage signals are frozen:

- `residual_transition_active_fraction`: fraction of sampled transitions whose
  residual p90 speed is at least `0.01` frame diagonals per second;
- `residual_active_area_fraction_mean`: mean spatial fraction above the same
  physical threshold.

Persistence and safety signals are:

- `late_residual_motion_ratio`;
- `longest_low_residual_run_fraction`, with low motion defined as less than
  `0.0025` frame diagonals per second;
- `residual_accel_outlier_fraction`;
- camera-model validity, inlier fraction, and normalized fit error.

Raw flow, global flow, camera-motion fraction, residual energy concentration,
and residual direction entropy are retained for interpretation. An increase in
raw flow without an increase in residual flow is explicitly labeled
`camera_only_motion_increase`.

The thresholds are fixed before v191/v192 results are inspected. They are
diagnostic operating points, not universal perceptual boundaries.

## 4. Metric calibration before method comparison

Every v193 target first checks whether each signal has enough numerical
variation. A signal is not used as evidence when it is constant, nearly
boundary-saturated, or has too few unique values. Absolute Spearman correlation
of at least `0.98` is reported so duplicated signals are not presented as
independent evidence.

The candidate has a directional local-motion signal against one control only
when all of the following hold:

1. the primary residual magnitude and at least one coverage signal are
   informative;
2. mean primary residual magnitude is positive;
3. at least one coverage mean is positive;
4. automatic collapse/discontinuity warnings remain within approximately 3%
   of prompts.

A strong signal additionally requires the paired bootstrap lower bound of the
primary residual magnitude to be positive. Motion alone never promotes the
method. If the paired VBench report is available, quality, identity/background,
and temporal mechanics must also be non-inferior to both `all_recent` and
`sf_native` under the already frozen development margins.

## 5. Four-node execution

v193 reads existing comparison videos. It uses CPU OpenCV and does not consume
the 32 generation GPUs. It can run while VBench jobs occupy the GPUs.

For v191, launch this command on the four nodes with ranks 0 through 3:

```bash
TARGET=v191_confirm128 NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  V193_WORKERS=8 bash scripts/run_v193_camera_motion.sh compute
```

After all nodes finish, run on node 0:

```bash
TARGET=v191_confirm128 NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v193_camera_motion.sh status
TARGET=v191_confirm128 NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v193_camera_motion.sh collect
TARGET=v191_confirm128 NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v193_camera_motion.sh analyze
```

If v191 passes and v192 has been generated, repeat for each target:

```bash
TARGET=v192_seed2026_30s_128 NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  V193_WORKERS=8 bash scripts/run_v193_camera_motion.sh compute

TARGET=v192_long60_seed10000_32 NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  V193_WORKERS=8 bash scripts/run_v193_camera_motion.sh compute
```

Then run `status`, `collect`, and `analyze` on node 0 for the same target. A
single-node fallback is available:

```bash
TARGET=v191_confirm128 NODE_RANK=0 NUM_NODES=1 V193_WORKERS=16 \
  bash scripts/run_v193_camera_motion.sh all-local
```

For a custom audited comparison, set `TARGET=custom`,
`V193_SOURCE_RUN_ROOT`, `CANDIDATE`, and `CONTROLS`. PF and ABA are deliberately
absent from the required v193 matrix.

## 6. Output and review policy

Each target writes:

```text
runs/v193_camera_motion/<target>/
  parts/part_XX_of_04.{csv,contract.json}
  metrics/camera_compensated_motion.{csv,contract.json}
  analysis/v193_camera_motion.json
```

The merge rejects mixed OpenCV/NumPy versions, implementation hashes,
frame-step settings, comparison manifests, shard layouts, or video paths.

No broad manual review is required. If the automatic motion signal fails, the
review queue is empty. If it passes, v193 emits at most four high-disagreement
prompts. Those videos should be reviewed only together with the paired VBench
identity/quality result.

## 7. How this affects the method story

v193 is evaluation infrastructure, not a new cache component. The current
method hypothesis remains:

> Long-history exposure should be selected jointly by head identity and
> denoising phase, while all methods share the same structured Coverage bank,
> update rule, clean-call policy, and 9-FFE read budget.

The intended causal chain is still tested at three levels:

1. structured long-history Coverage versus all-Recent;
2. selective exposure versus all-Coverage;
3. joint Head x Phase selection versus Head-only and Phase/Layer-only factors.

v193 closes one evidence gap: it tests whether any apparent motion advantage is
local generated motion rather than camera motion, and whether identity is being
preserved simply by freezing the video. It must not be described as a method
contribution unless later human calibration establishes perceptual validity.
