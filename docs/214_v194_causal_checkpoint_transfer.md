# v194: Frozen Head x Phase Route Transfer to Causal-Forcing

> Date: 2026-08-25  
> Status: code ready; execution is conditional on a passing v192 decision  
> Compute: 4 nodes x 8 GPUs, 64 prompts x 3 methods x 30 seconds

## 1. Repository and evidence status

At the start of this round, the local experiment branch and its remote branch were
both at `3e195912e94d60e6afeec03f80446a15e8d2165a`; `main` was at
`7ba5082f4cc199b38ebdd780410910943f3e622f`. No new v191, v192, or v193 result
bundle was present. Therefore this round does not make a new empirical claim.

v194 is deliberately conditional. `prepare` refuses to create an experiment unless
the frozen v192 decision says all of the following:

- `within_model_seed_length_robustness_confirmed == true`;
- all three v192 combined gates are true;
- `recommendation == freeze_within_model_head_phase_method_for_cross_model_transfer`;
- all v192 input, map, report, and runtime hashes still match.

This prevents an unconfirmed within-model effect from being promoted to a transfer
experiment.

## 2. Question answered by v194

v191/v192 evaluate one frozen Head x Denoising-Phase route on the Self-Forcing
checkpoint. v194 changes only the generator checkpoint to the official chunkwise
Causal-Forcing checkpoint and asks:

> Without refitting heads, phases, thresholds, or cache operators, does the same
> 9-FFE route retain a prompt-paired advantage on a generator trained with a
> different causal objective?

This is a **cross-checkpoint / cross-training-objective** test within the shared Wan
architecture. It is not a cross-architecture result. All methods run through the
same audited PyramidKV code host so that the checkpoint is the controlled variable.

## 3. Frozen methods

| Method | Role | Historical read |
|---|---|---|
| `cf_native_21` | Causal checkpoint native baseline | official 21-frame rolling window |
| `cf_all_recent_9ffe` | primary equal-budget control | sink1 + recent8, 9 FFE |
| `cf_head_phase_transfer` | no-refit transferred candidate | noisy calls use the frozen v192 Head x Phase map; clean uses Recent; 9 FFE |

The native method is a practical backbone baseline, not an equal-budget control.
Only the latter two methods can support a claim about heterogeneous historical
selection at fixed read budget.

All three methods are fixed to:

- checkpoint: `../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt`;
- checkpoint branch: top-level `generator`, not EMA and not automatic guessing;
- strict state-dict loading after the known Causal-Forcing internal FSDP prefix is
  normalized;
- `model_kwargs.local_attn_size=21`;
- seed `10000`, with reseeding by global prompt index;
- 120 latent frames, decoded as 477 frames at 16 FPS, 832 x 480.

The candidate and all-Recent maps, bank map, selected operator, positive metrics,
and 9-FFE contract are byte-copied from v192. No v194 code path can classify or tune
heads.

## 4. Prompt selection and pairing

v194 uses odd positions `1, 3, ..., 127` from the frozen 128-prompt v192 scope,
giving 64 prompts. The rule is fixed without consulting per-prompt v192 scores.

These prompts use seed 10000, matching their v191 Self-Forcing runs. This enables a
same-prompt, same-noise comparison of the candidate-minus-all-Recent effect across
the two checkpoints. It is not a fresh-prompt claim; prompt generalization was
already handled by v191.

The generation load is 192 videos total. With 32 GPUs, each GPU generates two
prompts per method, or six 30-second videos.

## 5. Code added

- `scripts/prepare_v194_cf_checkpoint_transfer.py`: strict prerequisite validation,
  prompt/map/checkpoint/runtime freezing, and re-verification;
- `scripts/run_v194_cf_checkpoint_transfer_32gpu.sh`: smoke, 4 x 8 sharded
  generation, status, audit, and packaging;
- `scripts/audit_v194_cf_checkpoint_transfer.py`: decoded media checks, exact
  checkpoint/window log checks, and full route-trace validation;
- `scripts/prepare_v194_vbench_comparison.py`: prompt-correct audited VBench grid;
- `scripts/run_v194_vbench_long.py` and `scripts/run_v194_vbench_long.sh`: core-9,
  lightweight temporal safety, v193 camera-compensated motion, and final decision;
- `scripts/analyze_v194_cf_checkpoint_transfer.py`: paired confidence intervals,
  noninferiority, same-prompt cross-checkpoint effect, and bounded review queue;
- `tests/test_v194_cf_checkpoint_transfer.py`: prerequisite, freezing, drift, effect,
  and inference-control tests.

`third_party/Pyramid-Forcing/inference.py` now exposes two opt-in controls:

- `--checkpoint_state_key generator|generator_ema`;
- `--model_local_attn_size -1|N`.

Defaults preserve all earlier commands. v194 uses both explicitly and writes
`[ModelAttentionContract]` and `[CheckpointLoad]` debug markers.

## 6. Run order

Run `prepare`, smoke, and audits on node 0. Run `generate`, `split`, `eval`, and
`camera-compute` concurrently on all four nodes with `NODE_RANK=0..3`.

### 6.1 Freeze inputs and smoke

```bash
cd /path/to/training-free

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh prepare

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh smoke

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh audit-smoke
```

`prepare` failing because v192 is absent or did not pass is the intended behavior.
Do not bypass this check.

### 6.2 Full 64-prompt generation

Launch once per node, in parallel:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh generate
```

Then on node 0:

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh status

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v194_cf_checkpoint_transfer_32gpu.sh audit
```

The audit checks 192 decoded videos, all logs, the exact checkpoint SHA/state branch,
the common 21-frame model setting, and one complete 4 x 30 x 12 schedule/readout
trace for each cache method. It does not spend time hashing every MP4 again.

### 6.3 VBench-Long and temporal safety

On node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh prepare
```

On all four nodes, in parallel:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v194_vbench_long.sh split
NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v194_vbench_long.sh eval
```

On node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh status
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh collect
```

`collect` creates the core-9 summary, computes the existing lightweight temporal
diagnostics, and writes an initial paired v194 decision without making a motion
claim.

### 6.4 Camera-compensated motion

On all four nodes, in parallel:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
  bash scripts/run_v194_vbench_long.sh camera-compute
```

On node 0:

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh camera-status
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh camera-collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v194_vbench_long.sh decision
```

This step separates estimated global affine camera flow from local residual motion.
VBench Dynamic Degree remains a nonregression diagnostic when it is saturated; it
is not accepted as evidence of increased subject motion.

## 7. Automatic decision gates

The transfer claim passes only if all six gates pass:

1. candidate is noninferior to 9-FFE all-Recent;
2. candidate is noninferior to the Causal native 21-frame baseline;
3. at least one positive metric frozen by v192 has a Causal-checkpoint paired CI
   strictly above zero versus all-Recent;
4. for at least one such metric, both SF and CF checkpoint means are positive and
   the same-prompt two-checkpoint pooled CI is above zero;
5. automatic temporal safety passes versus all-Recent;
6. automatic temporal safety passes versus native21.

The camera-motion report is orthogonal. It can enable a motion-improvement claim
only when its measurement calibration, local-motion direction, and paired quality
noninferiority all pass. Failure to show motion gain does not turn a valid quality
transfer result into a motion result.

Human review is requested only after the six core gates pass. The queue is capped at
four prompt triplets and prioritizes automatic safety flags and metric/temporal
disagreement.

## 8. Interpretation branches

- **All gates pass:** the current method has evidence of zero-refit transfer across
  two generator checkpoints/training objectives. Freeze the route before any new
  cache optimization. The paper can claim shared-architecture checkpoint transfer,
  not cross-architecture universality.
- **Candidate beats all-Recent on SF but not CF:** the map is checkpoint-specific.
  Return to profiling on a small Causal prompt set and test whether the same feature
  construction, rather than the same head identities, transfers.
- **Candidate is noninferior but no frozen positive target replicates:** retain the
  mechanism as a compression-compatible cache, but do not claim a quality benefit.
- **Native comparison fails while equal-budget comparison passes:** the 9-FFE budget
  itself is too aggressive for this checkpoint; do not attribute the gap to the
  Head x Phase classifier.
- **Strict checkpoint load fails:** inspect the checkpoint provenance/container.
  Do not switch to `strict=False` or automatic state-key guessing.
- **Camera diagnostic fails:** make no increased-motion claim and inspect only its
  automatic failure rows; do not review a broad random video set.

ABA/prompt-switch generation and further cache tricks remain deferred. They should
not consume GPUs until the single-prompt cross-checkpoint result is known.
