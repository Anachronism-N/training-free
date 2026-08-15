# v180 Uploaded Audit and v183 Recovery Evaluation

## 1. What was uploaded

Commit `0bf7f0d0` adds the frozen v180 inputs, 64 shard logs, and 64 completion
markers. It does not add raw videos, media audits, VBench outputs, or paired
metrics.

The offline log audit now passes:

| Method | Shards | Prompt evidence | Route 20/21/22 | Runtime failures |
|---|---:|---:|---:|---:|
| `sf_native` | 16 | terminal run + status markers; media audit still required | native, no custom route | 0 |
| `rccp_matched` | 16 | 128/128 | 355/5/0 | 0 |
| `all_recent` | 16 | 128/128 | 360/0/0 | 0 |
| `all_coverage` | 16 | 128/128 | 0/360/0 | 0 |

Each shard owns eight indices with stride 16. The shorter `all_coverage` logs
for shards 1, 8, and 10 are valid resume logs: they explicitly skip existing
outputs and generate the remaining assigned indices. The union of generated
and skipped indices is exact for every shard.

The input prompt, map, v177, and v178 hashes in the uploaded manifest match
the committed Linux blobs. Different hashes seen in a Windows checkout are
only CRLF conversion. The recorded generation implementation corresponds to
the pre-v182 operator code; in particular the recorded `inference.py` and
`policy_overrides.py` hashes match commit `e085d36a` and its descendants before
`ad509959`.

## 2. Critical evidence boundary

The recorded v178 result is not a real metric result:

```json
{
  "decision": "pass",
  "metric_runtime_fingerprint": "2node-skip"
}
```

Its source hashes are empty, the published method list is empty, the contract
contains no complete media grid, and there are no paired comparisons. The new
independent checker rejects it on 31 required fields and artifacts, including the experiment,
32-prompt cardinality, membership gate, methods, metric fingerprint, comparison
hash, complete publication, and no-leakage contract.

This creates two separate conclusions:

1. The uploaded logs are strong evidence that the 512-video v180 generation
   finished with the intended four routes. A server-side decode/hash audit is
   still required because the raw videos were not uploaded.
2. v180 cannot be presented as confirmation that RCCP selected better heads
   than layer/count-matched alternatives. It can still provide a valid
   exploratory comparison of generated videos and cache operators.

The v183 recovery path preserves this boundary in every contract and result.
It never upgrades an exploratory result into a formal RCCP membership claim.

## 3. Questions answered by v183

The existing 512 videos are sufficient to answer the following without any
new generation:

| Paired comparison | Question |
|---|---|
| `rccp_matched - sf_native` | Does the complete strict-five method improve end-to-end generation over native SF? |
| `rccp_matched - all_recent` | Does adding Coverage to the five frozen heads help over the equal-budget local operator? |
| `all_coverage - all_recent` | Is Coverage itself a useful operator when membership is ignored? |
| `all_recent - sf_native` | How much change comes from the equal-budget PF-host cache path rather than selected nonlocal heads? |
| `rccp_matched - all_coverage` | Is sparse Coverage preferable to assigning Coverage to every head? |

The primary readout is VBench-Long core-9 at prompt level. The analysis reports
official quality, identity/background, and dynamic degree jointly. A method is
not considered directionally non-regressive when identity improves only by
reducing dynamic degree by more than 0.02. Confidence intervals and q-values
are included, but remain exploratory because the upstream gate was not valid.

Only six automatically selected identity-motion conflict cases enter the
manual-review queue. Broad blind review is not required for this decision.

## 4. Server commands

No video generation should be rerun. On node 0:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

bash scripts/run_v183_v180_recovery.sh logs
bash scripts/run_v183_v180_recovery.sh audit
bash scripts/run_v183_v180_recovery.sh prepare
```

The full audit decodes all 512 videos, verifies 477 frames at 16 fps and
832x480, hashes every video, rejects incomplete indices, and rejects a custom
route that produced videos identical to another method for all 128 prompts.

Split and evaluate on both 8-GPU nodes:

```bash
NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v183_v180_recovery.sh split

NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v183_v180_recovery.sh preflight

NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v183_v180_recovery.sh eval
```

Collect on node 0:

```bash
NODE_RANK=0 NUM_NODES=2 bash scripts/run_v183_v180_recovery.sh status
NODE_RANK=0 NUM_NODES=2 bash scripts/run_v183_v180_recovery.sh collect
NODE_RANK=0 bash scripts/run_v183_v180_recovery.sh decision
```

For interrupted jobs, inspect `status`, then use one node to run only missing
jobs:

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v183_v180_recovery.sh resume-missing
```

## 5. Automatic next decision

The analyzer emits one of four recommendations:

| Recommendation | Next action |
|---|---|
| `rerun_formal_membership_controls` | strict-five is positive versus both SF and all-Recent; regenerate only the missing 32-prompt v178 hard-negative grid and run the real paired gate |
| `strict5_end_to_end_promising_membership_unresolved` | preserve the method candidate, but first isolate host/cache effects before a classifier claim |
| `reprofile_coverage_operator_before_new_membership_test` | Coverage is useful, but the static five-head map is not; profile the selected deterministic v182 operator and learn membership again |
| `stop_static_strict5_and_revisit_operator` | neither sparse strict-five nor dense Coverage has a useful identity-motion-quality tradeoff; stop this static route |

The first recommendation does not mean RCCP is confirmed. It means the already
generated 128-prompt evidence is strong enough to justify completing the
missing matched-versus-hard-negative causal test.

## 6. Artifacts to push back

Push these small artifacts after completion:

```text
runs/v180_rccp_fresh128/recovery_v183/audits/
runs/v180_rccp_fresh128/recovery_v183/contracts/
runs/v180_rccp_fresh128/recovery_v183/published_manifest.json
runs/v180_rccp_fresh128/recovery_v183/metrics/vbench_core9_summary.{json,md,csv}
runs/v180_rccp_fresh128/recovery_v183/analysis/v183_v180_recovery_metrics.{json,md}
```

Do not push raw videos, canonical hard links, split clips, or per-job model
caches. If convenient, create a diagnostics archive with:

```bash
bash scripts/run_v183_v180_recovery.sh package
```
