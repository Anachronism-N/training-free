# v169 Server Runbook

## 1. 资源与代码

```bash
export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "${REPO_ROOT}"

git fetch origin
git switch codex/v169-soft-cross-scale
git pull --ff-only origin codex/v169-soft-cross-scale

export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
```

模型仍使用：

```text
/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
```

默认 prompt：

```text
prompts/moviegen_128_qwen_v154_diverse16.txt
```

每条视频 30 秒。本轮只新生成 Query-weighted 和 Bottleneck，共 32 个视频。

## 2. 生成前离线检查

只在 node 0：

```bash
export NODE_RANK=0
bash scripts/run_v169_soft_cross_scale_moviebench16.sh offline
bash scripts/run_v169_soft_cross_scale_moviebench16.sh source-audit
```

必须看到：

```text
controlled-change gate: true
query-weighted changed_from_v166: 23
bottleneck changed_from_v166: 57
corrected v168 mechanism gate: true
```

输出：

```text
runs/v169_soft_cross_scale_moviebench16/full8/offline_counterfactual.json
runs/v169_soft_cross_scale_moviebench16/full8/source_v168_trace_corrected.json
```

## 3. 两视频 smoke

只在 node 0：

```bash
export NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1
bash scripts/run_v169_soft_cross_scale_moviebench16.sh smoke
```

默认使用 prompt 14，每个新方法一个视频。这里只检查：

- MP4 可解码且长度正确；
- 没有 polygon noise、重复主体或立即崩坏；
- trace 中 CoherentMotion read 为空或连续两帧；
- config 中 selector mode 分别为 query-weighted 和 bottleneck。

不要用两个 smoke 视频选择方法。若出现结构噪声，停止 full16 并上传 smoke 的
config、trace 和 stdout/stderr。

## 4. 四节点 preflight

四个节点分别设置自己的 rank：

```bash
export NODE_RANK=<0|1|2|3>
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7

bash scripts/run_v169_soft_cross_scale_moviebench16.sh preflight
```

每个节点预期：

```text
methods=6 total_tasks=96 node_tasks=24 new=8 reused=16 gpus=8
```

preflight 会重新核对：

- v168 published manifest、prompt hash 和 experiment contract hash；
- checkpoint size 与 PF config hash；
- 四个复用方法各 16 个非空视频及 total bytes；
- v168 Pareto 的 16 条 trace 经修正分析器重新通过；
- Middle10 layer map 和全部实现文件 hash。

## 5. 四节点生成

四节点同时运行：

```bash
export NODE_RANK=<0|1|2|3>
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7

bash scripts/run_v169_soft_cross_scale_moviebench16.sh generate
```

每节点约生成 8 个新视频、链接复用 16 个视频。中断后可以用相同命令恢复，已经
满足 frozen marker 的任务不会重复生成。

查看节点状态：

```bash
bash scripts/run_v169_soft_cross_scale_moviebench16.sh status
```

## 6. 汇总与机制审计

全部节点完成后，只在 node 0：

```bash
export NODE_RANK=0 NUM_NODES=4
bash scripts/run_v169_soft_cross_scale_moviebench16.sh audit
bash scripts/run_v169_soft_cross_scale_moviebench16.sh mechanism
```

必须满足：

```text
published manifest ok=true
both methods mechanism_gate=true
contract_failure_count=0
read_budget_violation_count=0
changed_from_v166_count>0
old_recall_count>0
```

若 mechanism 失败，不运行后续指标。优先上传：

```text
runs/v169_soft_cross_scale_moviebench16/full8/automated_screen/soft_cross_scale_trace.json
runs/v169_soft_cross_scale_moviebench16/full8/traces/
runs/v169_soft_cross_scale_moviebench16/full8/configs/
runs/v169_soft_cross_scale_moviebench16/full8/logs/
runs/v169_soft_cross_scale_moviebench16/full8/contracts/experiment.json
```

## 7. 自动安全与代理指标

机制通过后，在一个节点使用 6 张评测卡：

```bash
export EVAL_GPUS=0,1,2,3,4,5
bash scripts/run_v169_automated_screen.sh all
```

该命令计算 temporal diagnostics、comprehensive metrics 和自动 failure
localization。此时不要盲审全部视频。

## 8. VBench-Long core-9

node 0 准备 comparison：

```bash
export NODE_RANK=0 NUM_NODES=4
bash scripts/run_v169_vbench_long.sh prepare
```

四节点分别 split：

```bash
export NODE_RANK=<0|1|2|3> NUM_NODES=4
bash scripts/run_v169_vbench_long.sh split
```

四节点 preflight 和 eval：

```bash
export NODE_RANK=<0|1|2|3> NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7

bash scripts/run_v169_vbench_long.sh preflight
bash scripts/run_v169_vbench_long.sh eval
```

查看完成情况：

```bash
export NODE_RANK=0 NUM_NODES=4
bash scripts/run_v169_vbench_long.sh status
```

只在确有 missing jobs 时，单节点补缺：

```bash
export NODE_RANK=0 NUM_NODES=1
export GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v169_vbench_long.sh resume-missing
```

全部完成后 node 0：

```bash
export NODE_RANK=0 NUM_NODES=4
bash scripts/run_v169_vbench_long.sh collect
bash scripts/run_v169_vbench_long.sh prepare-review
```

`collect` 自动生成 corrected paired decision。`prepare-review` 根据冻结规则生成：

- 0 个视频：两个方法明显失败，无需人工 review；
- 最多 4 个视频：一个 near-frontier candidate 与 v166 的两条 prompt 盲审；
- promotion 通过仍会准备诊断 bundle，但 128-prompt confirmation 不以该自适应
  review 作为论文证据。

## 9. 需要回传的文件

优先推送小文件：

```text
runs/v169_soft_cross_scale_moviebench16/full8/published_manifest.json
runs/v169_soft_cross_scale_moviebench16/full8/offline_counterfactual.json
runs/v169_soft_cross_scale_moviebench16/full8/source_v168_trace_corrected.json
runs/v169_soft_cross_scale_moviebench16/full8/automated_screen/soft_cross_scale_trace.json
runs/v169_soft_cross_scale_moviebench16/full8/automated_screen/automated_screen.json
runs/v169_soft_cross_scale_moviebench16/full8/automated_screen/temporal_diagnostics.csv
runs/v169_soft_cross_scale_moviebench16/full8/automated_screen/comprehensive.json
runs/v169_soft_cross_scale_moviebench16/full8/metrics/vbench_core9_summary.json
runs/v169_soft_cross_scale_moviebench16/full8/analysis/v169_corrected_metrics.json
runs/v169_soft_cross_scale_moviebench16/full8/minimal_review/review_manifest.json
```

若失败，再补充对应 prompt 的 policy trace、config 和日志。不要首先推送全部视频。

## 10. 决策分支

1. **机制失败**：修代码或配置，不解释视频质量。
2. **两个方法明显低于 v166**：不人工 review，冻结为负结果。
3. **只有一个方法 near-frontier**：完成最多四视频盲审，判断指标是否遗漏可见收益。
4. **严格四轴 gate 通过**：冻结该方法，下一轮运行独立 MovieBench-Qwen 128
   confirmation。
5. **128 prompts 确认后仍成立**：再设计 ABA/AB scene-switch extension；不要在
   当前候选未确认前混入 prompt-switch 机制。
