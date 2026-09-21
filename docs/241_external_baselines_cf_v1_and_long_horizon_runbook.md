# External baselines, original CF, and long-horizon closure

Date: 2026-09-21. Branch: `worktree-v210-lphc`.

## 1. What can run now

- **v219 is ready**: use `docs/240_v216_results_v219_mechanism_and_writing_closure.md`.
  It replaces the unstarted v217; do not launch both. It uses eight nodes, eight GPUs
  each, 64 fixed prompts, four methods, 256 new 30-second videos.
- **Original CF checkpoint preparation is ready**, using the commands below.
  Download verification is not inference parity verification.
- **v220 SF 60-second extension is ready** in the follow-up commit. See section 6.
- CF/Ours inference and matched Deep Forcing generation are **not yet launch-ready**
  in this batch. Do not run an old CF campaign with current LPHC labels.

Do not pull updates into an active generation checkout. Existing campaigns bind
their source commit and runtime hashes. Use a separate clean checkout/worktree for
new experiments, retain old output roots, and keep GPUs exclusive between campaigns.

## 2. Frozen experiment priorities

| Order | Experiment | Scale | Purpose |
|---|---|---|---|
| Running/ready | v219 SF/Ours/random/pooled | 64 x 30s x 4 | New-seed method and selector confirmation |
| Next | v220 SF/Ours | 64 x 60s x 2 | Longer-horizon behavior, without retuning LPHC |
| Next | Original CF local baseline / CF + LPHC | 64 x 30s x 2 | Transfer between distilled models |
| Next | SF + Deep Forcing DS+PC | 128 x 30s | Same-checkpoint external method comparison |
| Conditional | SF + Echo-Forcing | provenance first | Structured-memory comparison |

Each matched pair uses the same prompt, noise seed, model weights, sampling schedule,
resolution, and duration. Freeze method/endpoint before new results. Report all fixed
prompts, not only favorable subsets. Current SF gains are small; adding baselines
does not itself establish an advantage. No new PF generation or ABA in this batch.

Use the eight-node pool in **waves**, not eight overlapping full-pool campaigns.
Keep v219's current placement intact. New generation campaigns should distribute
paired methods over the same GPU and rotate their order. Metric jobs use released
GPUs only. Do not promise a completion time from old H20 timings.

## 3. Original Causal Forcing checkpoint

The first chunk-wise upload is pinned, not the moving Hugging Face `main` revision:

```text
repo_id: zhuhz22/Causal-Forcing
revision: 373037a987c3e06eaab3ec7b2fc2f5c9c296b649
filename: chunkwise/causal_forcing.pt
bytes: 5676282643
sha256: cf75ee5cc6f4e2e336c59c973f5544655d8f0aa481761efe6de1b9cb2eb0cd9d
variant: original chunk-wise four-step CF
```

Prepare on the shared filesystem once, not separately on all 64 GPUs:

```bash
python scripts/prepare_cf_v1_checkpoint.py download \
  --local-dir /apdcephfs_gy2/share_302533218/cedricnie/model_cache/causal_forcing_v1 \
  --receipt /apdcephfs_gy2/share_302533218/cedricnie/model_cache/causal_forcing_v1/verified.json
```

If the original weight already exists, do not redownload it:

```bash
python scripts/prepare_cf_v1_checkpoint.py verify \
  --checkpoint /absolute/path/to/causal_forcing.pt
```

The script fails on an incorrect size/hash, including a Git LFS pointer. It never
loads a pickle or allocates GPU memory. Do not substitute `causal_cd.pt`,
`longvideo.pt`, frame-wise CF, or CF++. Model-base path and model cache key must be
explicit in future job receipts; SF's `generator_ema` must not be assumed for CF.

Before scaling CF: pin the upstream source/config, verify strict weight loading,
compare native reference with the disabled-LPHC adapter, then run the alpha-zero
pair. Preserve the official original sampling schedule. Do not choose CF-specific
alpha/phase based on evaluation prompts and call it zero-tuning transfer.
The vendored `third_party/Causal-Forcing/pipeline/causal_inference.py` contains
historical HREM hooks; it must not be described as an untouched official reference.
Use a clean, pinned upstream checkout for parity, and explicitly disable legacy
experiment environment switches in the future transfer worker.

Original CF was not trained for >81 output frames. Label our comparison as
**controlled length extrapolation of the original CF checkpoint**, not a comparison
against the authors' complete long-video system. SF and CF share a Wan architecture;
success would support cross-distillation transfer, not cross-architecture generality.

## 4. Reuse external baselines before generating

`docs/131_v129_progress_summary.md` reports 128 completed videos each for Deep
Forcing, Rolling Forcing, and LongLive. This is a historical record, not confirmation
that the server files are still present. Locate their publication manifests first.

Reuse requires checking prompt text/hash, output duration, checkpoint hash and the
loaded state key (EMA vs non-EMA), inference config, code revision, and media validity.
The v129 runner uses seed 0; current campaigns use a source-index-dependent seed.
Do not label old-vs-new comparisons as same-seed paired tests. Use a separate clearly
labeled historical table, or regenerate the missing matched external arm. Do not
regenerate already valid matched outputs.

Reevaluate using the repaired VBench-Long implementation and frozen fingerprints.
Old Dynamic Degree and derived Quality must not be mixed with corrected results.
Do not invent official Total/Semantic scores from a core-only metric profile.
Deep Forcing means the full **Deep Sink + Participative Compression** path, not DS
alone. Retain the official method configuration, record its cache/compute cost, and
do not silently change it to our cache policy for an apparently equal-budget table.

Echo-Forcing's public repository currently says its code was temporarily withdrawn.
Verify the provenance of our retained snapshot before calling a run an official
reproduction. Rolling Forcing and LongLive use additional training and belong in a
separate system-comparison group rather than the same-checkpoint method group.

## 5. Primary sources

- PF, experiments: https://arxiv.org/html/2605.13111v1#S5
  SF and CF, 128 MovieGen prompts, 30/60s, includes Deep Forcing.
- EF, experiments: https://arxiv.org/html/2605.16003v1#S4
  SF and LongLive bases, long-video 128 x 60s / 64 x 120s; includes Deep Forcing.
- CF code and short-horizon warning: https://github.com/thu-ml/Causal-Forcing
- CF checkpoint history: https://huggingface.co/zhuhz22/Causal-Forcing/commits/main
- CF original file metadata: https://huggingface.co/api/models/zhuhz22/Causal-Forcing/tree/373037a987c3e06eaab3ec7b2fc2f5c9c296b649?recursive=true
- Deep Forcing official DS+PC: https://github.com/cvlab-kaist/DeepForcing
- EF availability: https://github.com/mingqiangWu/Echo-Forcing

PF/EF paper scores are not directly comparable with our rewritten prompts and
evaluation revision. Cite them as related work; do not paste their scores into a
table presented as our controlled reproduction.

## 6. v220: frozen SF/Ours at 60 seconds

This is a complete generation/evaluation campaign, not a new cache search:

- 64 uniform ordinal selections from v216's 80 non-selection prompts, identical to
  v219's source list. Seed is `21600 + source_index`, also identical to v219.
- SF FIFO21 versus the v216-selected `headwise_correct` configuration. Alpha .02,
  phase e1, archive12/history4 and the inference operator are unchanged.
- 240 latent frames, 80 AR blocks, 957 decoded frames at 16 fps (59.8125s),
  832x480. The VBench split has 30 clips, not the old 15.
- 128 full videos over eight nodes x eight GPUs. One matched pair per physical GPU,
  four SF-first and four Ours-first pairs on each node. No inter-node DDP required.
- Primary metric/window inherited from v216 (the actual run froze Imaging/full).
  Report all core-9 raw metrics, Quality, early/late halves, paired intervals and cost.
- Same-prompt results at 30s and 60s are dependent, and separately sampled rollouts
  need not have bitwise-identical prefixes. Do not count them as 128 independent prompts.

### 6.1 Isolated source and common environment

Do not update the running v219 worktree. On every node create a new clean checkout
at the **same pinned follow-up commit**, for example:

```bash
git fetch origin
git worktree add --detach /tmp/training-free-v220-long60 origin/worktree-v210-lphc
cd /tmp/training-free-v220-long60
git rev-parse HEAD
```

Compare that SHA across all nodes before `prepare`. Never pull it afterwards. The
run binds the source hash and rejects drift, including helper-script changes.

Set on all eight nodes, retaining the same values except the node rank:

```bash
export V220_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v220_runs/v220_frozen_long60
export V220_V216_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v216_runs/v216_64f5c72a_eightnode_imaging80
export V220_SOURCE_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export WAN_MODEL=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3
export VBENCH_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v218_runtime/VBench_cf9b3b45
export VBENCH_CACHE_DIR=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/vbench
export TORCH_HUB_DIR=/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/torch_hub
export VBENCH_RUNTIME_HOME=/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/dreamsim_home
export GPU_LIST=0,1,2,3,4,5,6,7
export NODE_RANK=0  # Set 0..7 according to the frozen node order on each node.
NODES=(28.216.19.213 28.216.19.143 28.216.19.137 28.216.19.225 28.216.18.144 28.216.18.136 28.216.19.69 28.216.17.70)
export V220_NODE_ADDRESS="${NODES[$NODE_RANK]}"
```

Node IPs are inherited from v216 and must match a real local interface. Do not
guess replacement IPs or run on occupied GPUs. If allocation changes, freeze a new
explicit topology rather than editing an existing manifest. Parent evidence comes
from v216; no v219 result or manual review is required to begin v220.

### 6.2 Rank 0: freeze, prepare, numerical checks, first full pair

```bash
bash scripts/run_v220_experiment.sh freeze
bash scripts/run_v220_experiment.sh prepare
bash scripts/run_v220_experiment.sh schedule
bash scripts/run_v220_experiment.sh baseline
bash scripts/run_v220_experiment.sh gate0
bash scripts/run_v220_experiment.sh smoke
```

`baseline` and `gate0` are short numerical implementation checks after local-cache
eviction, not new short-video quality experiments. They bind this new run and cannot
be replaced by copying a receipt from another run root. `smoke` generates the first
full 60s pair; it is reused by `generate64`, not generated twice. Batch VAE decoding
is retained to match the frozen method; peak memory at 60s needs server verification.
If OOM occurs, retain stderr and stop; do not silently change resolution or decode mode.

### 6.3 All eight nodes: generate and evaluate in barriers

After rank 0 completes `smoke`, launch once on each node:

```bash
bash scripts/run_v220_experiment.sh generate64
```

After **all eight** generators finish, rank 0:

```bash
bash scripts/run_v220_experiment.sh status
bash scripts/run_v220_experiment.sh publish
```

On every node, run `split`. Wait for all split processes before `eval`:

```bash
bash scripts/run_v220_experiment.sh split
# Global barrier: all eight split commands must finish.
bash scripts/run_v220_experiment.sh preflight
bash scripts/run_v220_experiment.sh eval
```

After all eight evaluations finish, rank 0:

```bash
bash scripts/run_v220_experiment.sh collect
bash scripts/run_v220_experiment.sh analyze
bash scripts/run_v220_experiment.sh package
```

Use `status`, `eval-status`, and `eval-missing` for resuming. Never delete a valid
completion to force a rerun. Corrupt/incomplete jobs are quarantined by the existing
runner; paired jobs cannot silently migrate to a different physical GPU.

### 6.4 Return artifacts and interpretation

Upload `v220_small_artifacts.tar.gz`, especially:

- `inputs/selection.json`, `inputs/manifest.json`, `inputs/placement.json`;
- `decisions/sf_upstream_gate.json`, `decisions/gate0.json`;
- per-job `invocation.json`, `done.json`, `trace.jsonl`, stdout/stderr on errors;
- `evaluation/metrics/vbench_core9_summary.json` and raw metric parts/receipts;
- `evaluation/analysis/v220_long60.{json,md,csv}`.

The analyzer checks full 80-block LPHC trace coverage and all 30 clips, reports
full/early/late values, and queues at most two automatically flagged pairs for
diagnosis. All flags remain in JSON even if only two are reviewed. An automatic flag
is not proof of visual failure. Keep the shared six-pair review budget across runs.
Use the standalone v220 report; the existing 30s paper exporter is intentionally
not relabeled to merge horizons into one score. Do not claim a 60s benefit before
the paired results arrive, and do not replace a negative 30s row with a positive 60s row.

## 7. Implementation status and remaining work

- Ready: original-CF immutable download/hash verification; v220 freeze, scheduling,
  generation, resume, numerical checks, publication, 60s VBench, analysis, packaging.
- Unchanged: v219 method, its default 30s duration, frozen endpoint, and running outputs.
- Pending: CF-native parity adapter and paired transfer launcher; external Deep
  Forcing provenance/reuse decision and a matched current-protocol runner.
- Local verification is CPU-only contract/unit/syntax testing. No local GPU inference,
  long-video decoding, or VBench scoring has been performed.
- Follow-up checks: 195 tests passed, 3 skipped across the affected v210-v220 protocol,
  baseline, evaluator, paper-evidence and checkpoint suites; Python compilation and
  Bash syntax passed. This does not certify GPU runtime or visual quality.
