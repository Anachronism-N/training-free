# v177-v178 Strict RCCP Execution and Decision

## 1. Current scientific status

The latest uploaded v176 run completed 128 prompts, but its four selected
Coverage heads are invalid. The Union teacher omitted candidate cache
representations in 4,668 runtime events, and the uploaded implementation
downgraded the required assertion to a warning. Therefore:

- do not generate from the v176 `matched.csv`;
- do not report `L0H10`, `L5H3`, `L6H6`, or `L8H6` as discovered heads;
- do not claim that residual cache compatibility is already effective.

v177 is the corrected profiling experiment. v178 is prepared now so that a
valid v177 result can immediately enter untouched-prompt causal validation.
The method hypothesis is:

> A head can be assigned to an equal-budget Recent, Coverage, or Episode
> operator according to which operator best preserves its residual
> contribution under a representation-complete teacher.

This is not PF's temporal-QK Anchor/Wave/Veil taxonomy. It profiles local
residual compatibility with three explicitly budget-matched memory operators.
The hypothesis becomes a generation method only if v178 passes.

## 2. v177 correction

The candidate operators are unchanged and all have at most 9 frame
equivalents (FFE):

| Operator | Cache composition | Role |
|---|---|---|
| Recent | sink1 + recent8 | local continuity |
| Coverage | sink1 + reservoir4 + recent4 | temporal coverage |
| Episode | sink1 + reservoir2 + coherent motion pair2 + recent4 | event/motion history |

The Union teacher has at most 17 FFE. v177 differs from the invalid v176 run
in three mandatory ways:

1. Coverage and Episode middle banks use their own recent4 eligibility
   boundary inside Union; the boundary includes the actual three-frame AR
   update extent.
2. Superset identity is `(physical frame, representation family)`, separating
   saved, time-mapped, and dynamic-RoPE K. A Recent K and a dynamic-RoPE
   anchor K from the same frame are not interchangeable. If two anchor banks
   select the same frame, their K/V/position tensors and RoPE mode must match
   exactly before Union may deduplicate them.
3. Every record must contain 36 successful checks, zero failures, and the
   `v177` verification contract. Any missing identity raises immediately, and
   the offline audit repeats the test on trace layers 0/10/20/29.

The discovery split remains frozen at seed `1762026`: 64 discovery prompts,
32 validation prompts, and 32 untouched generation prompts. The thresholds
were not changed after viewing v176. The v177 analysis also binds the exact
128-prompt file and profiling input manifest by SHA-256; v178 refuses a
different 128-line suite even when the prompt indices are valid.

## 3. Run v177 first

Pull the pushed commit on the server. On node 0, prepare and run only the
one-prompt smoke:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh smoke
```

The smoke automatically requires 1,440 records and all representation-level
checks. Do not launch 128 prompts if it fails. After it passes, run one command
on each of the four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v177_strict_superset_rccp_32gpu.sh profile128
```

After every node completes, run on node 0:

```bash
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v177_strict_superset_rccp_32gpu.sh package
```

The result is `runs/v177_strict_superset_rccp/analysis/analysis.json`.

## 4. Automatic v177 decision

Inspect only these top-level fields first:

```text
profile_audit.complete_profile
teacher_contract.candidate_representation_superset_required
supported_nonlocal_head_count
assigned_policy_counts
generation_ready
```

- If `generation_ready=false`, stop this static classifier line. v178
  `prepare` will also refuse to run.
- If `generation_ready=true`, proceed to v178 without manually choosing or
  editing heads.
- Do not lower the frozen margin, validation, FDR, call/AR consistency, budget,
  or stability gates after seeing the result.

## 5. v178 untouched 32-prompt generation

v178 freezes exactly the 32 prompt IDs declared by v177 as the generation
holdout. It copies all maps and records hashes into a self-contained input
manifest. Its six methods are:

| Method | Purpose |
|---|---|
| `matched` | v177 membership hypothesis |
| `all_recent` | operator-utility control; all heads use Recent |
| `hard_negative_0..3` | four unique, per-layer policy-count-matched rejected memberships |

No PF baseline, repeated SF baseline, or ABA experiment is included here.
Those do not answer whether v177 selected the right heads.

On node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh preflight
```

Then launch on all four nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_rccp_holdout_generation_32gpu.sh generate32
```

This produces 6 methods x 32 prompts = 192 videos. Each GPU handles one
prompt per method. On node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v178_rccp_holdout_generation_32gpu.sh package
```

The audit fully decodes all videos and checks 477 frames, 16 fps, 832x480,
one video per prompt, no runtime failure pattern, and the exact runtime count
of labels 20/21/22 for every method and shard. A marker or nonempty file alone
does not pass.

## 6. VBench-Long core-9

After the media audit passes, prepare on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh prepare
```

Run split preparation and evaluation on all nodes:

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v178_vbench_long.sh eval
```

If interrupted, replace `eval` with `resume-missing`. After all jobs finish,
collect on node 0:

```bash
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v178_vbench_long.sh collect
```

Primary output:

```text
runs/v178_rccp_holdout_generation/analysis/v178_paired_metrics.json
```

## 7. Frozen v178 decision rule

Aggregate VBench scores describe behavior but do not validate membership. The
classifier passes only when all of the following hold on the untouched set:

1. `matched` has positive paired mean deltas over the four-hard-negative
   ensemble for official quality and identity/background.
2. Both primary 95% bootstrap confidence intervals are above zero.
3. Both primary BH-corrected q-values are at most 0.10.
4. Both primary win fractions are at least 0.55.
5. `matched` also has positive primary mean deltas over `all_recent`.
6. Dynamic-degree mean deltas versus the ensemble and all-Recent are no worse
   than -0.02.

The analyzer emits exactly one of:

```text
advance_rccp_membership_to_broader_generation
reject_static_rccp_membership_for_generation
```

No broad video review is needed before this result. If the gate passes, review
only the largest paired wins/losses and then expand to 128-prompt generation
and scene-switching tests. If it fails, the cache operators may still be
useful, but this head-membership rule is not supported and should not be
packaged as the method.

## 8. Debug artifacts to return

For a v177 failure, return:

```text
runs/v177_strict_superset_rccp/v177_strict_superset_rccp_diagnostics.tar.gz
```

For a v178 generation or audit failure, return:

```text
runs/v178_rccp_holdout_generation/v178_rccp_holdout_diagnostics.tar.gz
```

For a metric result, push the comparison manifest, core-9 summary,
`v178_paired_metrics.json`, and its Markdown rendering. Raw videos do not need
to be pushed to GitHub.
