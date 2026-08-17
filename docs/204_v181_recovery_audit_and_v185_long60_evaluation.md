# v181 恢复审计与 v185 60 秒探索性评测

## 1. 本轮同步内容

远端 `origin/main` 新增了 v181 `long60_seed0` 的完整生成文件记录。当前可确认：

| 方法 | 视频数 | 解码帧数 | FPS | 分辨率 | 媒体审计 |
|---|---:|---:|---:|---:|---|
| `sf_native` | 128 | 957 | 16 | 832x480 | PASS |
| `rccp_matched` | 128 | 957 | 16 | 832x480 | PASS |
| `all_recent` | 128 | 957 | 16 | 832x480 | PASS |

三组方法在相同 prompt index 上没有任何字节级重复视频，
`map_route_appears_globally_ignored=false`。因此这些视频可以用于探索性量化分析。

## 2. 为什么 v181 不能作为正式论文证据

视频完整不等于实验 provenance 完整。本轮发现三个独立问题。

### 2.1 上游 v178 decision 是占位文件

v181 manifest 记录的上游 decision 为 `pass`。对应
`runs/v178_rccp_holdout_generation/analysis/v178_paired_metrics.json` 只包含手工占位字段，
缺少正式 paired comparison、classifier gate、input provenance 和 experiment contract。
它不能证明五个 RCCP heads 已通过因果 membership 验证。

### 2.2 恢复运行使用了混合 shard 布局

最终生成由两个节点上的 8-way/16-way stride 恢复完成，而旧 audit 固定期待 32 个
shard 日志。当前日志仍可确认所有观察到的 PF 进程使用正确 route：

```text
rccp_matched: Recent=355, Coverage=5, Episode=0
all_recent:   Recent=360, Coverage=0, Episode=0
```

但日志不能为每个最终视频提供唯一的 successful-process attribution；四个旧 RCCP
日志还保留先前 OOM traceback。因此 `runtime_logs.json` 正确状态应为 FAIL，而不是
formal PASS。

### 2.3 远端补丁错误地把失败降级为 warning

远端曾把 runtime、media 和 duplicate audit 的异常从 `raise` 改为 warning，并仍然
写出 `ok=true` 的 published manifest；同时跳过 runtime 和 frozen provenance 校验。
本轮已经恢复 formal v181 路径的 fail-fast 行为。正式入口不会接受当前 recovered
artifact，也不会把它误写成 classifier confirmation。

## 3. v185 的定位

v185 不重新生成视频，而是为已有 384 个视频建立独立的
`exploratory_recovered` 评测路径。它回答：

> 静态五个 Coverage heads 在 60 秒外推下，相对 SF 和 all-Recent 是否表现出值得
> 继续研究的长期收益或明确的 identity-motion trade-off？

它不回答 head classifier 是否成立。无论指标多好，输出中都固定：

```text
formal_classifier_claim_eligible = false
```

## 4. v185 准备阶段的强制检查

`prepare_v185_recovered_long60_comparison.py` 会重新检查：

- 128 prompts 和 source indices `256..383`；
- 两个 30x12 head map 的 SHA 与 `355/5/0`、`360/0/0` route 数量；
- 384 个 raw video 的文件 SHA 与上传 media audit 一致；
- 每个视频为 957 帧、16 FPS、832x480，并已完整解码；
- 所有已观察 PF 日志中的 route 配置一致；
- 三组方法之间 exact duplicate 数量为零；
- 上游 v178 placeholder、mixed shard 和 stale OOM 被写入 limitations。

任一媒体、route 或 duplicate 条件失败都会停止，不会创建 VBench comparison。

## 5. 指标与自动分析

v185 运行三方法乘 VBench-Long core-9，共 27 个 metric jobs。每个 60 秒视频拆为
30 个两秒 clips，分别统计：

- `full`: clips 0-29；
- `early_half`: clips 0-14；
- `late_half`: clips 15-29。

主要 paired comparisons：

```text
rccp_matched - sf_native
rccp_matched - all_recent
all_recent   - sf_native
```

分析输出 Quality、identity/background、dynamic、temporal、semantic 和 visual 的
逐 prompt delta、bootstrap CI、sign test 和 BH q-value。由于 generation provenance
不完整，所有统计都标为 exploratory；q-value 仅用于观察稳定性，不用于论文确认。

自动 verdict 包括：

- `static_five_long60_promising_exploratory`；
- `static_five_identity_motion_tradeoff`；
- `static_five_motion_identity_tradeoff`；
- `static_five_long60_not_supported`。

决策不要求人工 review。脚本只输出最多四个 late-half identity-motion 冲突样本，供
需要时定位，不再盲审 384 个视频。

## 6. 服务器运行命令

同步实验分支后，在 node 0 锁定 recovered comparison：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git fetch origin
git checkout codex/v178-v179-causal-validation
git pull --ff-only

NODE_RANK=0 bash scripts/run_v185_recovered_long60_vbench.sh prepare
```

四节点分别拆分并评测：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v185_recovered_long60_vbench.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v185_recovered_long60_vbench.sh eval
```

完成后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v185_recovered_long60_vbench.sh status
NODE_RANK=0 bash scripts/run_v185_recovered_long60_vbench.sh collect
NODE_RANK=0 bash scripts/run_v185_recovered_long60_vbench.sh decision
```

若只有零散 metric job 缺失，可用单节点恢复：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v185_recovered_long60_vbench.sh resume-missing
```

## 7. 与 v184 主线的关系

v185 只回收已有 60 秒数据，不是新方法主线。当前主线仍是 v184 的 denoising-phase
Coverage exposure：所有方法共享相同 bank update 和 9-FFE read budget，只比较
Coverage 在 early/late noisy calls 的作用。

资源允许时，v185 的 27 个 metric jobs 与 v184 生成可以并行。优先级为：

1. v184 single-prompt smoke 与 cache trace audit；
2. v184 systematic-32 generation 和 core-9；
3. v185 recovered long60 core-9；
4. 暂不运行 ABA、第二 seed 或新的静态 head 阈值扫描。

## 8. 下一轮分支

- v184 early schedule 晋级：固定阶段，比较 deterministic landmark、prototype 和
  retrieval Coverage operator。
- v184 只有 all-Coverage 增加运动：设计基于 clean-latent motion deficit 的在线
  gate，不再固定全程启用。
- v185 静态五头较优：只能说明 60 秒视频值得重新做一次 clean provenance
  confirmation，不能恢复旧 classifier claim。
- v185 静态五头无收益：归档 strict-five 路线，论文方法完全转向 phase/state
  conditioned memory exposure。

## 9. 需要回传的文件

```text
runs/v185_recovered_v181_long60/vbench_comparison/comparison_manifest.json
runs/v185_recovered_v181_long60/metrics/vbench_core9_summary.{json,md,csv}
runs/v185_recovered_v181_long60/analysis/v185_recovered_long60_metrics.{json,md}
```

无需推送 raw video 和 split clips。
