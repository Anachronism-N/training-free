# v93 MovieBench 10-Hour Main Table and Head-Classification Plan

> Date: 2026-07-25
> Status: implementation complete; GPU execution pending
> Hardware target: one node with 16 H20 GPUs
> Primary protocol: 120 latent frames (about 30 seconds), seed 0

## 1. Why this experiment is needed now

v90 changed the evidence boundary:

- matched-seed PF and v78 have almost identical mean DINO;
- v78 is not a reproducible DINO improvement over PF;
- `veil_priority_b005` and `wave_priority_b005` remain competitive partial
  results, but do not establish a new classifier;
- v92 started the first real prompt-contrastive binary read-policy test, but
  16 prompts are too small for the main comparison.

v93 therefore uses the available compute for two separate questions:

1. **Main quality:** which existing or current method is strongest over the
   full 128-prompt MovieGenVideoBench subset?
2. **Head-classification causality:** does prompt sensitivity identify a useful
   two-timescale head partition, or is any result explained by class count,
   random membership, profile noise, or the v78 write controller?

This is a 30-second experiment from the start. No 6-second screening stage is
inserted.

## 2. Current method candidate

The current result-conditional method is:

**Prompt-Contrastive Dual-Timescale TransitionCache**

It combines two independently testable components.

### 2.1 Prompt-contrastive binary read routing

An offline, training-free intervention profiles every attention head by the
change in its output when prompt evidence is perturbed under a matched latent
and history state:

```text
prompt_response(h) =
  median ||output_cond(h) - output_perturbed(h)|| /
  (mean output magnitude + epsilon)
```

The primary classifier preserves PF's per-layer Anchor count only as a budget
control:

```text
lowest prompt-response heads -> prompt-stable
remaining heads              -> prompt-responsive
```

The executed cache has two read timescales:

| Role | Read composition | Intended evidence |
|---|---|---|
| prompt-stable | sink3 + bounded strided middle + recent4 | identity, appearance, persistent layout |
| prompt-responsive | sink1 + bounded cyclic middle + recent4 | pose, motion, camera and prompt change |

This differs from PF in the measurement criterion, number of classes,
membership and lack of a Veil merge class. PF's implementation and
sink/middle/recent design remain the base and must be cited.

### 2.2 Trust-conditioned cache-state promotion

The v78 controller operates on writes to the bounded middle cache. It uses
same-block noisy/clean disagreement, shock from the last promoted state,
novelty, age, asynchronous phase and a per-layer update budget. Sink and recent
regions retain PF's update path.

This asks a different question from read routing:

```text
read routing: which history should this head consume?
state promotion: which generated state is reliable enough to become history?
```

The weak `0.05` prompt-responsive or Veil priority is a secondary factor. It is
not part of the core method unless the large experiment improves motion or
human quality without sacrificing identity, background or physics.

### 2.3 Explicit cache composition

The current core does not inject a separate archive:

| Region | Acquisition | Update | Function |
|---|---|---|---|
| sink/anchor | earliest complete K/V frames | fixed after initialization | origin and persistent appearance |
| structured middle | bounded complete historical K/V | PF write, optionally filtered by v78 | medium/long history |
| recent | latest complete K/V frames | rolling every AR block | local motion and continuity |

Coverage archives and coherent Echo snapshots remain optional follow-ups. The
earlier direct archive experiments caused flashbacks or duplicated subjects,
so they are not mixed into the main 128-prompt result.

## 3. MovieBench-128 main table

Prompt file:

```text
third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num128.txt
```

Each method is divided into global prompt ranges `[0,64)` and `[64,128)`.
The two shards write into one method directory. Sixteen GPUs therefore execute
eight methods concurrently.

| GPUs | Method | Purpose |
|---:|---|---|
| 0-1 | `sf_native` | native Self-Forcing baseline |
| 2-3 | `pf` | strongest external cache baseline |
| 4-5 | `echo_pc` | Echo-Forcing baseline |
| 6-7 | `v78` | historical best write-controller candidate |
| 8-9 | `pf_binary_read_v78` | Anchor versus Wave+Veil with v78 writes |
| 10-11 | `prompt_pfcount_read_v78` | primary prompt-sensitive candidate |
| 12-13 | `prompt_kmeans_read_v78` | count-free prompt classifier |
| 14-15 | `veil_priority_b005` | best currently completed v90 priority cell |

Every engine reseeds with `seed + global_prompt_index`. This gives matched
prompt/seed pairs across methods and prevents the second shard from repeating
the first shard's random sequence.

The 128-prompt table is the main comparison. It is not the complete
classification ablation.

## 4. MovieBench-32 head-classification table

Prompt file:

```text
third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt
```

One method runs on each GPU:

| GPU | Method | Question |
|---:|---|---|
| 0 | `pf` | original three-class reference |
| 1 | `pf_binary_read` | does Anchor versus Wave+Veil work without new writes? |
| 2 | `prompt_pfcount_read` | prompt membership alone |
| 3 | `prompt_kmeans_read` | natural two-cluster membership alone |
| 4 | `v78` | PF read plus trusted writes |
| 5 | `pf_binary_read_v78` | binary topology plus trusted writes |
| 6 | `prompt_pfcount_read_v78` | primary combined candidate |
| 7 | `prompt_kmeans_read_v78` | count-free combined candidate |
| 8 | `prompt_replica_read_v78` | independent-profile reproducibility |
| 9 | `prompt_consensus_read_v78` | profile averaging |
| 10 | `prompt_inverse_read_v78` | causal direction control |
| 11 | `prompt_random_read_v78` | matched-budget membership control |
| 12 | `remote_read_v78` | remote-history signal control |
| 13 | `role_score_read_v78` | previous remote-minus-prompt classifier |
| 14 | `pf_read_prompt_priority` | priority-only effect on PF reads |
| 15 | `prompt_read_prompt_priority` | weak priority on prompt reads |

This matrix separates:

- three classes versus two classes;
- PF membership versus prompt membership;
- read-policy effects versus transition-write effects;
- primary versus replica/consensus maps;
- intended direction versus inverse/random controls;
- prompt response versus older remote-history signals;
- read routing versus weak update priority.

## 5. Metrics and evidence

The postprocessor computes several families because no single metric is
sufficient:

1. `evaluate_comprehensive.py`: DINO mean/min stability, drift, CLIP alignment,
   background consistency, flicker, loop/repetition diagnostics and composite.
   RAFT M3 is skipped because it has repeatedly failed to download reliably.
2. VBench-Long: `subject_consistency`, `background_consistency`,
   `aesthetic_quality`, `imaging_quality`, and `dynamic_degree`.
3. Temporal jump: appearance and optical-flow discontinuity, subsampling every
   four decoded frames to keep the 128-prompt pass tractable.
4. Transition traces: accepted/rejected updates, reliability, novelty, age,
   branch, role and coherence.
5. Blind human review: identity, background, motion, camera, artifacts, prompt
   alignment, overall rank and first failure time.

`dynamic_degree` is behavior, not a monotonic quality score. A frozen method or
a violently unstable method can both distort its interpretation.

The head-classification automated screen requires the primary prompt map to:

- win a majority of available quality metrics over inverse;
- win a majority over matched random;
- remain within `0.005` DINO of PF-binary;
- reproduce within `0.01` DINO using the independent profile.

This screen is necessary, not sufficient. Human review can reject duplicated
subjects, flashbacks, scene leakage, frozen motion or physics errors even when
the screen passes.

## 6. Server execution

Required assets at the default locations:

```text
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Echo-Forcing/checkpoints/self_forcing_dmd.pt
third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
runs/v81_probecache_profile/labels/probecache_profile_report.json
runs/v82_probecache_profile_replica/labels/probecache_profile_report.json
../research_sprint/bench_baselines/VBench/vbench2_beta_long/
```

The default Conda environment is `longlive`. VBench-Long and the comprehensive
evaluator must already have their normal model assets available in the
server-side caches; the launcher intentionally does not download or modify
models.

From the repository root:

```bash
git pull --ff-only
nohup bash scripts/run_v93_moviebench_10h.sh \
  > runs/v93_moviebench_10h.log 2>&1 &
```

Monitor:

```bash
tail -f runs/v93_moviebench_10h.log
find runs/v93_moviebench128_main/status -name '*.done' | wc -l
find runs/v93_moviebench32_head/status -name '*.done' | wc -l
```

The queue executes:

```text
MovieBench-128 generation
MovieBench-32 head generation
MovieBench-128 VBench/comprehensive/temporal metrics
MovieBench-32 VBench/comprehensive/temporal metrics
```

Run stages independently when needed:

```bash
bash scripts/run_v93_moviebench_main_16gpu.sh
bash scripts/run_v93_moviebench_head32_16gpu.sh
bash scripts/postprocess_v93_moviebench.sh main
bash scripts/postprocess_v93_moviebench.sh head32
```

Resume the same command after interruption. A shard/cell is skipped only when
its completion marker exists and the global indexed-video audit passes. Do not
set `FORCE=1` unless intentionally replacing a complete run.

Useful overrides:

```bash
REPO_ROOT=/path/to/training-free \
PRIMARY_REPORT=/path/to/primary/probecache_profile_report.json \
REPLICA_REPORT=/path/to/replica/probecache_profile_report.json \
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
bash scripts/run_v93_moviebench_10h.sh
```

Skip a queue stage:

```bash
RUN_MAIN=0 RUN_HEAD32=1 RUN_METRICS=0 \
  bash scripts/run_v93_moviebench_10h.sh
```

If the PyramidKV extension is already compiled and preload itself is
undesirable:

```bash
PRELOAD_PYRAMIDKV=0 bash scripts/run_v93_moviebench_main_16gpu.sh
```

## 7. Outputs to inspect

Main:

```text
runs/v93_moviebench128_main/
  <method>/*.mp4
  logs/<method>.shard{0,1}.log
  configs/<method>.shard{0,1}.env
  traces/*.transition.jsonl
  diagnostics/*.audit.json
  blind_review/manifest_public.json
  blind_review/scorecard.csv
  blind_review/key_private.json
  metrics/vbench_long_summary.{json,csv,md}
  metrics/comprehensive.json
  metrics/temporal_jump.csv
  metrics/v93_analysis.{json,md}
```

Head32 uses the same layout under `runs/v93_moviebench32_head/`, with one log
and config per method.

Before opening `key_private.json` or any metric table, score and freeze the
blind-review CSV. The postprocessor creates the blind package before metrics,
but cannot prevent a reviewer from reading metric files.

## 8. Required debug checks

Generation is invalid if any of these conditions occur:

- an indexed video is missing, empty or duplicated;
- a PF cell does not print `[PyramidKVHeadMap]`;
- a transition cell produces no JSONL trace;
- a trace misses layers or has malformed per-head fields;
- logs contain traceback, OOM or key-error signatures;
- a metric silently evaluates fewer than 128 or 32 videos;
- prompt hash, map hash, commit or seed differs across a claimed comparison.

The launchers record prompt SHA-256, commit, map hashes, ranges and reseeding
mode. The analyzers retain metric sources and transition-trace paths.

## 9. Decision after the run

Use the following result branches:

1. **Prompt map passes controls and is competitive with PF.** Promote the
   binary prompt-sensitive read topology plus trust-conditioned state promotion
   as the main method. Keep weak priority only if independently beneficial.
2. **PF-binary works but prompt membership fails inverse/random.** The
   two-timescale topology may be useful, but prompt sensitivity is not a
   contribution. Continue with another measurable binary criterion.
3. **v78/PF remain best.** Do not claim a classifier contribution. Treat v78
   as a matched-quality lifecycle analysis and use the 128-prompt result to
   decide whether a stronger cache update or coherent memory component is
   necessary.
4. **Echo wins only on scene behavior.** Keep it as an attributed external
   baseline or a separately factorized snapshot-selection follow-up, not as an
   unexplained part of the method.

No paper claim should be finalized from a different formula alone. The
classifier must survive causal controls, broad prompts, multiple metrics and
artifact-aware review.
