# v168 Server Runbook

## 1. 拉取代码

```bash
git fetch origin
git checkout codex/v168-cross-scale-consensus
git pull --ff-only origin codex/v168-cross-scale-consensus
```

不要把旧输出写入 v168 默认目录，也不要修改冻结的 16-prompt 文件。

## 2. 离线反事实

该步骤不使用 GPU，但要求服务器保留 v166 trace：

```bash
bash scripts/run_v168_cross_scale_consensus_moviebench16.sh offline
```

预期输出：

```text
runs/v168_cross_scale_consensus_moviebench16/full8/offline_counterfactual.json
runs/v168_cross_scale_consensus_moviebench16/full8/offline_counterfactual.md
```

`branch_coverage_gate` 必须为 `true`。

## 3. Smoke：只生成两个视频

在一个节点执行：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1 \
  bash scripts/run_v168_cross_scale_consensus_moviebench16.sh smoke
```

默认 prompt index 为 14，只生成 Pareto 和 Strict Consensus 各一个视频。

检查：

- 视频可播放；
- 无多边形、纯色、解码失败或严重结构噪声；
- 两个任务 status 均成功；
- trace 中存在 `pareto_pass` 和 `scale_argmax_agreement`。

若失败，停止 full16，并提交 smoke 目录中的 `logs/`、`traces/`、`configs/`、`diagnostics/` 和 `status/`。

## 4. 四节点 preflight

每个节点分别运行，`NODE_RANK` 取 0、1、2、3：

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_cross_scale_consensus_moviebench16.sh preflight
```

每个节点预期 24 个 task。全局 96 个 task 中，64 个从 v167 复用，32 个为新生成。

preflight 会严格检查 v167 manifest、prompt hash、experiment contract、checkpoint/config、StateRank trace gate、corrected metrics 和视频字节。

## 5. 四节点生成

每个节点分别运行：

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_cross_scale_consensus_moviebench16.sh generate
```

等待四个节点全部结束后，在 rank 0 执行：

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_cross_scale_consensus_moviebench16.sh audit

bash scripts/run_v168_cross_scale_consensus_moviebench16.sh mechanism
```

必须得到：

```text
published_manifest.json: ok=true
cross_scale_consensus_trace.json: mechanism_gate=true
```

查看状态与打包 debug：

```bash
bash scripts/run_v168_cross_scale_consensus_moviebench16.sh status
bash scripts/run_v168_cross_scale_consensus_moviebench16.sh package
```

## 6. 自动诊断

机制审计、时序诊断和 comprehensive 可在 VBench 运行期间并行推进：

```bash
bash scripts/run_v168_automated_screen.sh mechanism
bash scripts/run_v168_automated_screen.sh temporal

EVAL_GPUS=0,1,2,3,4,5 \
  bash scripts/run_v168_automated_screen.sh comprehensive

bash scripts/run_v168_automated_screen.sh screen
```

也可串行运行：

```bash
EVAL_GPUS=0,1,2,3,4,5 \
  bash scripts/run_v168_automated_screen.sh all
```

在 `screen` 完成前不需要人工 review 视频。

## 7. VBench-Long

rank 0 准备输入：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v168_vbench_long.sh prepare
```

四节点分别预切分：

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_vbench_long.sh split
```

四节点先运行 preflight，再运行 eval：

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_vbench_long.sh preflight

NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_vbench_long.sh eval
```

缺失任务只能在单节点恢复：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v168_vbench_long.sh resume-missing
```

rank 0 汇总与决定：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v168_vbench_long.sh collect
```

核心输出：

```text
runs/v168_cross_scale_consensus_moviebench16/full8/analysis/v168_corrected_metrics.json
runs/v168_cross_scale_consensus_moviebench16/full8/analysis/v168_corrected_metrics.md
```

## 8. 最小盲审

只有 corrected analysis 完成后执行：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v168_vbench_long.sh prepare-review
```

脚本自动选择：

- 相对 v166 MultiScaleMotion 的最大 Quality downside prompt；
- motion 不下降条件下的最大 Quality upside prompt。

每个 prompt 只比较自动选中的 v168 candidate 与 MultiScaleMotion，最多两个 prompt、四个视频。reviewer 不应查看 `private/blind_key.json`。

## 9. 需要回传的文件

优先推送小文件：

```text
published_manifest.json
contracts/experiment.json
automated_screen/cross_scale_consensus_trace.json
automated_screen/cross_scale_consensus_trace.md
automated_screen/automated_screen.json
automated_screen/automated_screen.md
analysis/v168_corrected_metrics.json
analysis/v168_corrected_metrics.md
metrics/vbench_core9_summary.json
minimal_review/review_manifest.json
```

若失败，再补 `status/`、`logs/`、`diagnostics/` 和对应失败 prompt 的 policy trace。不需要先上传全部视频。

## 10. 决策

### A. 某一候选通过开发门

`development_decision.recommendation == run_128_prompt_confirmation`：锁定获胜候选，不再在 16 prompt 上调规则，新建独立 128-prompt confirmation，再决定是否开展 ABA。

### B. 身份提高但 motion 降低

不直接扩大。查看自动筛出的低运动 prompt 和 old-recall 分布，判断过度回退到 newest 还是旧召回本身抑制运动。

### C. motion 提高但身份下降

查看冲突分支、selected age 和 background drift，不以 Dynamic 单轴结果作为方法成功。

### D. 两个候选都不优于 MultiScaleMotion

停止扩大 v168，保留跨尺度 conflict profiling 作为诊断结论；回到 archive admission 或跨层/timestep 作用机制，而不是继续调比较阈值。
