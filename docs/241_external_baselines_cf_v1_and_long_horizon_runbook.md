# External baselines, original CF, and long-horizon closure

Date: 2026-09-21. Branch: `worktree-v210-lphc`.

## 1. What can run now

- **v219 is ready**: use `docs/240_v216_results_v219_mechanism_and_writing_closure.md`.
  It replaces the unstarted v217; do not launch both. It uses eight nodes, eight GPUs
  each, 64 fixed prompts, four methods, 256 new 30-second videos.
- **Original CF checkpoint preparation is ready**, using the commands below.
  Download verification is not inference parity verification.
- v220 SF 60-second extension is being added as a separate campaign in this batch.
- CF/Ours inference and matched Deep Forcing generation are **not yet launch-ready**
  in this initial commit. Do not run an old CF campaign with current LPHC labels.

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
