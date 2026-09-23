# v224 findings and v225 fixed-input replay

Date: 2026-09-24. Branch: `worktree-v210-lphc`.
Imported `8fa8bee3` as `6bf16f28`. The new material is v224 input-audit and
table-export results, not a new method-quality experiment.

## Writing decision

**Start a focused method/results draft, but do not describe the current evidence
as robust overall improvement or submission/acceptance assurance.** A short
paper can make a narrow contribution; a small mixed result remains small and
mixed regardless of page count.

The frozen LPHC method, main SF comparison, random/pooled controls and dose/phase
ablation are unchanged. In the 64-prompt main cohorts, Imaging is +0.379 points
at 30s and +0.229 at 60s, on a 0-100 scale; the individual paired intervals cross
zero. Other metrics are mixed. The earlier matched two-seed Imaging result is
positive within its stated scope, not general seed robustness. See docs/242,
245 and 246 for complete results and limits.

The new audit neither strengthens those gains nor proves that the generation
implementation is defective. The immediate next question is how much
measurement variation affects a claim this small. v225 answers that using
existing videos. It does not search for a better seed, method or evaluator.

## What v224 actually established

Recomputed from all eight rank receipts and checked against the collected report:

| Quantity | Result |
|---|---:|
| Reused source-video pairs | 64 (SF and LPHC, 32 prompts each) |
| Different source-video hashes | 0 |
| Clip pairs | 960 |
| Different encoded clip bytes | 827 |
| Different decoded RGB frame fingerprints | 825 |
| Different decoded frame timing | 0 |

Every source pair has at least one changed clip. This does **not** mean that all
960 clips changed, that the changes are visually large, or that the method has
polygon artifacts. Hash equality is binary and does not measure distortion
magnitude. The reported decoder was identical across ranks. Both versions have
32 decoded frames per two-second clip.

The publication commit describes score drift as attributable to regenerated
splits. That is too strong as a causal conclusion: the audit establishes changed
inputs, but has no exact-input repeat control. Evaluation/runtime variation may
also contribute. v225 is designed to distinguish these factors.

### Relevant preprocessing behavior

The pinned [VBench split implementation](https://github.com/Vchitect/VBench/blob/45e79ec14e69a2187202c675d2dbce1a71843d53/vbench2_beta_long/utils.py#L117)
decodes the source and writes each segment with `torchvision.io.write_video`.
It is a re-encoding path, not copying source MP4 byte ranges. It overrides the
`fps=8` function argument using the detected source FPS. Our 16-fps sources
therefore yield 32 frames per two-second clip. The last segment borrows preceding
frames to reach a full segment. Preserve this established protocol; do not
silently change it to 8 fps.

The local `prepare_v129_vbench_splits.py` validates clip names/counts and source
sizes, not immutable decoded input identity across campaigns. Rebuilding a
subset can therefore produce another valid but non-identical set of clips.
Neither this observation nor upstream source inspection identifies whether
decoder behavior, encoder settings, library versions, threading, or something
else caused the observed differences. We do not patch inference on that basis.

## v225 experiment

Use the same predetermined 32-source subset and original SF/LPHC videos as
v223. Keep `21600 + source_index`; no new videos are generated. For each method:

| Evaluation condition | Clip input | Purpose |
|---|---|---|
| `reference` | Existing v219 clips, reindexed by source ID | Evaluate the earlier input |
| `resplit` | Existing v223 clips | Evaluate the later input |
| `repeat` | A byte-identical copy of `reference` | Measure exact-input repeat variation |

Preparation decodes each old/new clip only to verify the recorded v224 pixel
and timing fingerprint. It then copies encoded bytes into separate output views,
checks hashes, and freezes every source/clip file. It never regenerates a video,
re-encodes a clip, overwrites an old split cache, or replaces an old score.
Independent view directories prevent previously generated evaluator artifacts
from being reused as if they were fresh repeat measurements.

Use all nine existing dimensions and the derived Quality aggregate. Overall
consistency and temporal style are duplicated alignment outputs in this mode;
Quality is an aggregate, not a tenth independently measured dimension. Do not
introduce semantic dimensions using unrelated auxiliary prompt labels.

The evaluator, wrapper, RAFT/AMT weights and full-info hashes must match the
original evaluation dependencies. The legacy evaluator RNG behavior is retained:
the repeat estimates the realized variability of that same evaluator, not a
newly seeded variant. There is no generation-seed change or evaluation-seed
search. Existing numerical/non-deterministic behavior is not silently disabled.

## Eight-node placement

There are 18 independent groups: two methods x nine dimensions. Each group runs
its three conditions sequentially on the same physical GPU. Condition order
cycles through all six permutations, rather than always placing one condition
first. Host/GPU placement and package versions are recorded; restart a group on
the same host/device.

| Nodes | GPUs used per node | Group count |
|---|---|---:|
| Rank 0 and rank 1 | Slots 0,1,2 | 3 each |
| Rank 2 through rank 7 | Slots 0,1 | 2 each |

Total: **18 of 64 GPUs**, 54 metric jobs, 32 prompts per job. All eight nodes are
used; 46 GPUs remain available. This preserves paired same-device measurements
without using spare capacity to launch unneeded generation or extra seeds.
These jobs are metric evaluations, not 54 video-generation configurations.

Preparation runs once on rank 0 with eight CPU workers by default. It reads
1920 old/new clips for fingerprint checks and writes six independent metric-input
views. It can take some time on shared storage; `[v225-copy]` identifies each
completed source pair. Do not rerun old `split` actions while this is running.

## Analysis and stop rule

All scores are paired at the **32 source-prompt level**, never at the 480-clip
level. Repeated evaluation is not another generation seed or independent prompt
set. The report includes full, early-half and late-half windows and retains all
metrics, whether favorable or not.

Let `D_c` be the vector of per-prompt LPHC-minus-SF differences in condition c:

- Input contrast: `D_resplit - (D_reference + D_repeat) / 2`.
- Exact-input repeat contrast: `D_repeat - D_reference`.
- Also report the corresponding input/repeat changes for SF and LPHC separately.
- Report all three condition means and paired bootstrap intervals. Bootstrap RNG
  only makes uncertainty estimates reproducible; it is not a video-generation seed.

One repeat cannot establish a full noise distribution. The input contrast still
contains residual evaluator noise, so do not interpret it as an exact causal
decomposition of all old score changes. There is no invented pass/fail threshold
or automatic choice of the best repeat.

After this batch:

1. If the method-effect direction and scale are stable across inputs/repeats,
   retain the existing narrow Imaging claim and write the sensitivity limitation.
2. If fluctuations are comparable to the claimed effect or reverse it, weaken
   the efficacy claim. Do not select the most favorable split or replay as the
   headline number.
3. If a concrete evaluation implementation error is identified, repair only that
   error and reassess affected metrics symmetrically on existing videos. Mere
   hash inequality is not enough to order a whole-study rerun.

No follow-on seed sweep, new generation, ABA/CF expansion, or mass human review
is scheduled. Do not prolong a sequence of diagnostics automatically. Any
future method-improvement generation needs a separate decision, since the last
authorized generation ablation is already complete.

## Commands

Use a new analysis checkout. Keep original generation/evaluation outputs intact.
The wrapper defaults to the recorded server paths; set variables explicitly if
your actual paths differ. It activates the same `longlive` environment as before.

```bash
git fetch origin
git worktree add --detach ../training-free-v225 origin/worktree-v210-lphc
cd ../training-free-v225

export V225_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v225_runs/replay32
export V224_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v224_runs/closure
export VBENCH_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v218_runtime/VBench_cf9b3b45
export VBENCH_CACHE_DIR=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/vbench

# These cache paths must exist on EACH node; reuse the paths used for v223.
export TORCH_HUB_DIR=/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/torch_hub
export VBENCH_RUNTIME_HOME=/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/dreamsim_home

# Once, rank 0. Requires ffmpeg on PATH. Does not run VBench or inference.
NODE_RANK=0 bash scripts/run_v225_replay.sh prepare
NODE_RANK=0 bash scripts/run_v225_replay.sh schedule

# After preparation succeeds, on each of eight DISTINCT nodes:
# Set NODE_RANK to 0..7, not zero on every node. Keep the same checkout/env/roots.
NODE_RANK=0 bash scripts/run_v225_replay.sh eval

# Completed jobs are verified and skipped by the same eval command.
# Once, after all groups finish:
NODE_RANK=0 bash scripts/run_v225_replay.sh status
NODE_RANK=0 bash scripts/run_v225_replay.sh collect
NODE_RANK=0 bash scripts/run_v225_replay.sh package
```

Use `CONDA_SH`/`CONDA_ENV` overrides if needed. Do not change evaluation source,
models or the prepared checkout mid-run. If a cache path no longer exists,
restore the established local model cache rather than allowing a new download.
The wrapper does not invoke model download/bootstrap actions.

Logs are under `metrics/vbench_long_parts/<method>__<condition>/<dimension>/`.
`groups/` records condition order, physical device and completion receipts.
On an abnormal process termination, `running.lock` can remain: only after
confirming its recorded process is dead should that specific lock be removed.
Never start two copies of a rank concurrently.

Return `v225_small_artifacts.tar.gz`, especially
`analysis/v225_analysis.{json,md}`, `v225_means.csv`, `v225_contrasts.csv`,
`comparison_manifest.json`, `groups/` and the metric receipts/logs. The archive
excludes `published/` media. Do not overwrite v219/v223 results with these outputs.

## Four-page story

Working title: **Local-Preserving History Correction for Training-Free Long
Video Generation**.

The single core contribution is an auxiliary historical attention readout
converted into a norm-bounded correction of the original local output. Native
FIFO21 remains intact; the 12-frame archive, four-frame selection and first-phase
alpha 0.02 operating point are explicit implementation choices, not separately
validated discoveries of head function or optimal schedule.

The draft needs one computation diagram/equation, the 30s/60s main comparison,
a compact random/pooled and dose/phase discussion, and an honest tradeoff paragraph.
Keep Subject, DD and Quality visible alongside Imaging. The new replay study is
measurement sensitivity, not another method innovation or independent success.
Credit prior cache/retrieval methods for borrowed ideas. A focused narrative can
improve clarity, but cannot create an empirical advantage that the controls lack.

## Local verification

The uploaded v224 rank reports were reaggregated locally. The exact counts are
saved under `artifacts/experiment_results/v225_planning_6bf16f28/`.
CPU tests cover byte-preserving copies, complete split views, source membership,
interruption/resume, mutation rejection, paired triplet placement, and the
difference-of-effects analysis. Actual media copying/decoding and VBench model
execution still require the server; no local GPU run was performed.

Verification: 27 focused CPU tests passed (v224 closure and v225 replay),
including compact analysis using the actual uploaded metric results. Python
compilation and Bash syntax checks passed. Inference code was not modified.
