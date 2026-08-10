# v171 服务器运行手册

## 1. 实验规模

- 4 个节点，每节点 8 张 GPU，共 32 卡；
- 固定 16 条 MovieBench-Qwen prompt；
- 每条生成 30 秒；
- 复用 v170-v166 lane A 的 16 个视频；
- 新生成两个候选，各 16 个，共 32 个新视频；
- 每节点 12 个逻辑任务：8 个生成、4 个验证后链接复用；
- 默认不做人工 review。

## 2. 拉取分支

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git fetch origin
git switch codex/v171-demand-gated-recall
git pull --ff-only origin codex/v171-demand-gated-recall
```

默认依赖：

```text
checkpoint:
/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt

prompts:
prompts/moviegen_128_qwen_v154_diverse16.txt

v170 reuse root:
runs/v170_matched_attribution_moviebench16/full8
```

如 v170 结果在其他位置：

```bash
export V171_REUSE_V170_ROOT=/absolute/path/to/v170/full8
```

## 3. 离线反事实复核

只在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v171_demand_gated_motion_moviebench16.sh offline
```

预期：

```text
offline_gate=true
full_query_weighted_changes=258
DeficitQuery changed=86, healthy_changed=0
DeficitBaseline changed=180, healthy_changed=0
```

若 source trace hash 或计数变化，不要直接生成；先上传：

```text
runs/v171_demand_gated_motion_moviebench16/offline/v171_counterfactual.json
runs/v171_demand_gated_motion_moviebench16/offline/v171_counterfactual.md
```

## 4. 可选 smoke

smoke 只验证运行、解码和 trace，不用于选方法。默认 prompt 6，共两个视频：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1 \
  bash scripts/run_v171_demand_gated_motion_moviebench16.sh smoke
```

不需要逐帧盲审；只检查视频可解码、无多边形结构噪声、日志无异常终止。若失败，
停止 full16 并运行 smoke trace 定位。

## 5. 四节点生成

四个节点分别设置 `NODE_RANK=0,1,2,3`。

### 5.1 Preflight

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v171_demand_gated_motion_moviebench16.sh preflight
```

每个节点应报告：

```text
total_tasks=48 node_tasks=12 new=8 reused=4 gpus=8
```

### 5.2 Generate

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v171_demand_gated_motion_moviebench16.sh generate
```

### 5.3 Audit 与 mechanism

全部节点完成后，只在 node 0：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v171_demand_gated_motion_moviebench16.sh audit

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v171_demand_gated_motion_moviebench16.sh mechanism
```

机制报告：

```text
runs/v171_demand_gated_motion_moviebench16/full8/automated_screen/full_layer_trace.json
runs/v171_demand_gated_motion_moviebench16/full8/automated_screen/full_layer_trace.md
```

必须满足：

1. 两个候选各有 16 prompts x 10 layers 的 head-0 trace；
2. healthy 和 deficit 分支均执行；
3. healthy 状态相对 v166 的选择改变为 0；
4. 两个候选都至少一次改变 v166 选择；
5. baseline target、候选分数和最终 pair 可独立重算；
6. 读取为 0 或一个连续两帧 pair；
7. cache contract、read budget 和 selected/read mismatch 均为 0。

mechanism gate 失败时不要运行 VBench；上传 `traces/`、`logs/`、`configs/`、
`contracts/experiment.json` 和 mechanism 报告。

## 6. VBench-Long core-9

### 6.1 Node 0 准备 comparison

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v171_vbench_long.sh prepare
```

### 6.2 四节点切分

```bash
NODE_RANK=<0..3> NUM_NODES=4 bash scripts/run_v171_vbench_long.sh split
```

### 6.3 四节点评测

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v171_vbench_long.sh preflight

NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v171_vbench_long.sh eval
```

### 6.4 中断补缺

补缺只能单节点：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v171_vbench_long.sh resume-missing
```

### 6.5 汇总与自动决策

只在 node 0：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v171_vbench_long.sh collect
```

关键输出：

```text
runs/v171_demand_gated_motion_moviebench16/full8/analysis/v171_corrected_metrics.json
runs/v171_demand_gated_motion_moviebench16/full8/analysis/v171_corrected_metrics.md
```

决策只会给出：

```text
run_order_balanced_matched_confirmation
```

或：

```text
reject_both_without_manual_review
```

本轮没有 `prepare-review` 默认动作。自动指标若互相矛盾，先分析 per-prompt、
late segment 和 trace；只有进入 matched confirmation 后才设计极小规模盲审。

## 7. Debug 字段

每次 retrieval 的关键字段：

```text
motion_deficit.ready / triggered
motion_deficit.local_ratio / context_ratio
baseline_local_magnitude_target
baseline_context_magnitude_target_per_step
motion_signature_selected
query_weighted_selected
deficit_baseline_selected
selected
selection_reason
demand_gate_enabled / triggered
selection_changed_from_motion_signature
read_budget_preserved
```

每个 candidate 的关键字段：

```text
local/context_direction_similarity
query/candidate_local_magnitude
query/candidate_context_magnitude_per_step
motion_signature_score
query_weighted_motion_score
baseline_local/context_magnitude_similarity
baseline_magnitude_similarity
deficit_baseline_motion_score
state_pass / direction_pass
```

这些字段足以区分：gate 没触发、baseline 计算错误、候选集错误、排序错误、
fallback、实际读取不一致和 cache ownership 问题。
