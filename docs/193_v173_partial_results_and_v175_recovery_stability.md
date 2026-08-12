# v173 Partial Results and v175 Recovery/Stability Plan

## 1. Current status

The server pushed commit `9eb6d424` with 10 of 16 v173 shards. The artifacts
contain 80 of 128 prompts and 57,600 profile records. Each observed
prompt/layer has the expected 24 records, and the recorded call/update/branch
contract is `call={0,2}`, `update=noisy`, `branch=cond`.

This is a structurally valid **partial diagnostic**, not a completed profiling
result. It must not authorize generation-side claims.

## 2. What the partial result says

On the observed-only 39/41 calibration/validation split:

- 341/360 heads prefer Recent on calibration;
- 18 prefer Coverage, of which 6 pass every validation gate;
- 1 prefers Episode but fails the full-budget and validation gates;
- the six tentative Coverage heads are L0H10, L1H5, L5H3, L6H6, L8H6, L23H2;
- no tentative Episode head is supported.

The six heads are sparse and distributed across depth. This does not support a
fixed middle-layer rule. It is compatible with the narrower RCCP hypothesis
that a small subset can use a coverage-oriented cache, but that hypothesis is
not accepted until complete-data stability and generation interventions pass.

The Episode result is currently negative. Motion-pair retention should not be
promoted as a head class unless the full result changes this conclusion.

## 3. Why six shards failed

Ranks 1, 3, 6, 7, 8, and 12 reached the final AR blocks, then failed during VAE
decoding or full-video normalization. The logs show concurrent processes using
approximately 58 GB and 36 GB on the same H20, leaving too little memory for
the 294 MB to 2.13 GB decode allocations.

Two implementation properties amplified this failure:

1. profiling unnecessarily decoded and wrote videos even though only attention
   aggregates were needed;
2. the profile shard was saved only after the complete process exited, so a
   decode failure discarded all records from that process.

The missing prompts follow failed shard identities, not a random missingness
mechanism. The old partial analysis incorrectly reported
`generation_ready=true`; that flag is invalidated.

## 4. Recovery implementation

The updated profiling path now:

- skips all VAE initialization, streaming decode, concatenation, normalization,
  and video writing under `--skip_video_decode`;
- atomically checkpoints the shard after every completed prompt;
- restores complete prompts from an existing shard and skips them;
- drops an interrupted prompt before resuming it, preventing duplicate partial
  records;
- verifies that an existing shard belongs to the requested sharding topology;
- requires exactly 128 prompts and 24 records per prompt/layer before setting
  `complete_profile=true` or `generation_ready=true`.

The uploaded run used 16 logical shards. Do not resume it as 32 logical shards.
Six available GPUs can recover only the failed logical shards:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
V173_PROFILE_WORLD_SHARDS=16 \
V173_SHARD_IDS=1,3,6,7,8,12 \
GPU_LIST=0,1,2,3,4,5 NUM_NODES=1 NODE_RANK=0 \
bash scripts/run_v173_cache_compat_profile_32gpu.sh profile128
```

If those GPUs are spread across nodes, invoke the same command per node with
the local GPU list and matching `V173_SHARD_IDS`; the logical shard IDs, rather
than `NODE_RANK`, determine prompt ownership.

After recovery:

```bash
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v173_cache_compat_profile_32gpu.sh stability
```

## 5. v175 stability experiment

The complete 128 prompts are deterministically divided into 64 discovery and
64 transfer prompts. Transfer prompts never participate in head selection.

Within discovery64, RCCP is repeated over 12 pre-registered 32/32 splits. A
nonlocal head is retained only when:

- the same nonlocal policy is selected in at least 9/12 splits;
- no split assigns the competing nonlocal policy;
- the original per-split calibration, bootstrap, BH, call, AR, and budget gates
  pass whenever that policy is selected.

The stability report records selection frequency, outcome counts, non-empty
pairwise Jaccard, and a stable map. Zero stable nonlocal heads is a valid
negative result and blocks generation.

## 6. Generation-side causal test

If stability passes, v175 creates these maps:

- `stable_matched`: stable RCCP membership;
- `stable_all_recent`: no nonlocal operator;
- `hard_negative_0..3`: the same per-layer policy counts assigned to the most
  compatible rejected heads, without overlapping stable heads.

The first 32 transfer prompts form an automatic screen. If matched quality and
identity/background beat the hard-negative ensemble, expand to transfer64;
the first 32 videos are linked and reused rather than regenerated.

```bash
NODE_RANK=0 bash scripts/run_v175_rccp_generation_32gpu.sh prepare
NODE_RANK=<rank> NUM_NODES=4 bash scripts/run_v175_rccp_generation_32gpu.sh screen32
NODE_RANK=0 bash scripts/run_v175_rccp_generation_32gpu.sh audit_screen

V175_SCOPE=screen32 NODE_RANK=0 bash scripts/run_v175_vbench_long.sh prepare
V175_SCOPE=screen32 NODE_RANK=<rank> NUM_NODES=4 bash scripts/run_v175_vbench_long.sh split
V175_SCOPE=screen32 NODE_RANK=<rank> NUM_NODES=4 bash scripts/run_v175_vbench_long.sh eval
V175_SCOPE=screen32 NODE_RANK=0 bash scripts/run_v175_vbench_long.sh collect
```

Only after the screen gate passes:

```bash
NODE_RANK=<rank> NUM_NODES=4 bash scripts/run_v175_rccp_generation_32gpu.sh confirm64
NODE_RANK=0 bash scripts/run_v175_rccp_generation_32gpu.sh audit_confirm

V175_SCOPE=confirm64 NODE_RANK=0 bash scripts/run_v175_vbench_long.sh prepare
V175_SCOPE=confirm64 NODE_RANK=<rank> NUM_NODES=4 bash scripts/run_v175_vbench_long.sh split
V175_SCOPE=confirm64 NODE_RANK=<rank> NUM_NODES=4 bash scripts/run_v175_vbench_long.sh eval
V175_SCOPE=confirm64 NODE_RANK=0 bash scripts/run_v175_vbench_long.sh collect
```

No PF baseline or ABA experiment is scheduled in this round. The causal
question is whether RCCP membership beats strong layer/count-matched rejected
heads on held-out prompts. Manual review is optional and should be limited to
metric disagreements after automatic auditing.

## 7. Claim boundary

Current defensible statement:

> Equal-budget cache operators have strongly head-dependent residual effects;
> the partial sample tentatively identifies a sparse Coverage-compatible subset,
> while Episode compatibility is unsupported.

Not yet defensible:

> RCCP is a transferable head classifier or improves long-video generation.

That statement requires complete 128-prompt profiling, discovery-split
stability, and held-out transfer generation superiority over hard negatives.
