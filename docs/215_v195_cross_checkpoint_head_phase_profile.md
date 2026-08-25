# v195: Cross-Checkpoint Head x Denoising-Phase Mechanism Profile

> Date: 2026-08-25
> Status: code ready; execution starts only after a complete v194 automatic decision
> Compute: 4 nodes x 8 GPUs, 128 prompts, one frozen operator, no video decode

## 1. Synchronized evidence status

The remote experiment branch was still at `14c7580709e6ff18d0e758ac950e4acfbe4ba6fe`
when v195 was prepared. No v191, v192, v193, or v194 result bundle had been added.
Consequently, this document makes no new empirical claim.

v194 tests whether one SF-selected Head x Denoising-Phase route improves generated
videos after a zero-refit transfer to the Causal-Forcing chunkwise checkpoint. A
pass or failure alone does not identify what transferred:

1. the exact `(call, layer, head)` membership;
2. only a coarser denoising-phase/layer allocation;
3. the Coverage operator irrespective of classification; or
4. nothing measurable under the shadow compatibility objective.

v195 distinguishes these cases without generating or manually reviewing another
video grid.

## 2. Experiment question

v195 runs the v189 representation-complete shadow teacher on the Causal checkpoint:

```text
Recent   = sink1 + recent8                        = 9 FFE
Coverage = sink1 + structured middle4 + recent4 = 9 FFE
Union    = all candidate K representations      <= 13 FFE
```

For every prompt, noisy denoising call, layer, and head:

```text
gain = log(error Recent -> Union) - log(error Coverage -> Union)
```

Positive gain means Coverage is closer than Recent to the same representation-
complete teacher. The active trajectory remains all-Recent. Candidate and Union
readouts are shadow measurements and cannot affect latent generation.

The selected operator is copied from v194. No v195 metric can select Landmark versus
Retrieval, alter the SF map, or create a new generation method.

## 3. Frozen scope

- checkpoint: the exact Causal-Forcing checkpoint and top-level `generator` branch
  bound by v194;
- model attention window: `local_attn_size=21`;
- prompts: the exact 128 Qwen-rewritten MovieGen prompts used by v189 profiling;
- split: the original v189 64 discovery / 32 validation / 32 generation-holdout;
- duration parameter: 120 latent frames, approximately 30 seconds;
- seed: 0 with per-prompt reseeding;
- calls: noisy calls 0, 1, 2, and 3;
- cells: `4 x 30 x 12 = 1,440`;
- records: `128 x 4 x 30 x 12 = 184,320`;
- execution: exactly 32 shards, prompt ID modulo 32;
- decode: disabled.

The primary transfer readout uses the final 32 prompt IDs. They were not used to fit
the frozen SF map. Discovery and validation rows are retained for descriptive
cross-checkpoint correlation and a diagnostic Causal refit.

## 4. Attribution controls

The frozen SF route is evaluated on Causal holdout gains with two preregistered
10,000-draw controls.

### 4.1 Call-count matched

For each denoising call, preserve the number of selected cells and randomize their
layer/head membership. Beating this control supports useful phase/layer allocation.

### 4.2 Call/layer-count matched

For every `(call, layer)`, preserve the number of selected heads and randomize only
head identity. Beating this stronger control supports exact head membership beyond
phase/layer allocation.

Both controls use fixed seed `1950000` plus a documented offset. They are computed
from the untouched 32-prompt Causal holdout mean and cannot be selected after seeing
videos.

## 5. Automatic interpretation

The frozen selected route first needs:

1. selected-cell holdout gain with prompt-bootstrap 95% lower bound above zero;
2. positive gain on at least 60% of holdout prompts;
3. selected-minus-complement gain with bootstrap lower bound above zero.

Then:

| Level | Additional evidence |
|---|---|
| `exact_head_identity` | both random controls have one-sided empirical `p <= 0.05` |
| `phase_layer_structure` | call-count control passes but head-identity control does not |
| `operator_only` | selected route is positive, but membership enrichment is unsupported |
| `unsupported` | frozen selected route is not positive on the Causal holdout |

The analyzer also reports Pearson, Spearman, and nonzero-sign agreement at five
resolutions: exact call/layer/head, phase/layer, layer/head, layer, and phase.
These correlations are mechanism diagnostics, not extra gates.

A Causal-compatible map is refit with the unchanged v189 thresholds and compared to
the frozen SF map. It is diagnostic only. It must not replace the frozen map in v194
or be promoted directly to generation.

## 6. Why v194 must finish first

`prepare` accepts either a passing or failing v194 result because both outcomes need
mechanistic diagnosis. It rejects an absent, partial, hash-drifted, or internally
inconsistent v194 decision.

This ordering gives four useful branches:

- generation pass + exact profile transfer: freeze the route with shared-architecture
  checkpoint mechanism evidence;
- generation pass + only coarse/operator support: retain the empirical generation
  result but narrow the classifier claim;
- generation failure + profile transfer: the shadow objective does not predict the
  generated-video effect, so stop and diagnose the measurement/runtime interaction;
- generation failure + no profile transfer: stop cross-checkpoint route transfer.

## 7. Server commands

### 7.1 Prepare and smoke on node 0

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh prepare

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh smoke

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh audit-smoke
```

Do not bypass a failed `prepare`. A missing v194 decision means v194 must finish
first. A v194 decision with failed gates is valid as long as it is complete.

### 7.2 Profile on all four nodes

Launch once per node, concurrently:

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh profile128
```

Each GPU processes four prompts. Existing complete shards are skipped. Use `FORCE=1`
only when an audited shard is known to be invalid.

### 7.3 Audit, analyze, and package on node 0

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh status

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh audit

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh analyze

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v195_cross_checkpoint_profile_32gpu.sh package
```

## 8. Debug and correctness checks

The audit requires all of the following for every shard:

- strict checkpoint load from `generator`, with `use_ema=False`;
- CLI-overridden `local_attn_size=21`;
- exact checkpoint, prompt, all-head map, operator, seed, and profile contract;
- exactly four assigned prompts and 1,440 records per prompt;
- Recent/Coverage/Union policy contract and representation-superset checks;
- no duplicate prompt ownership, traceback, OOM, or profile-save failure;
- all 184,320 records before analysis.

Profile metadata now records `config_path`, `checkpoint_path`,
`checkpoint_state_key`, `use_ema`, and `model_local_attn_size`. These fields make a
wrong-checkpoint or wrong-window run detectable from the returned small artifacts.

## 9. Files to return

Push the small archive or the following files:

```text
runs/v195_cross_checkpoint_head_phase_profile/inputs/
runs/v195_cross_checkpoint_head_phase_profile/profile_audit.json
runs/v195_cross_checkpoint_head_phase_profile/analysis/analysis.json
runs/v195_cross_checkpoint_head_phase_profile/analysis/analysis.md
runs/v195_cross_checkpoint_head_phase_profile/analysis/cell_transfer.csv
runs/v195_cross_checkpoint_head_phase_profile/analysis/holdout_prompt_effects.csv
runs/v195_cross_checkpoint_head_phase_profile/logs/
```

The `.pt` profiles remain on the server unless the audit or analyzer fails. No MP4 is
created, and no manual review is required.

ABA/prompt-switch generation and additional cache tricks remain deferred until v194
and v195 jointly establish what, if anything, transfers across checkpoints.
