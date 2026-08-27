# v198：已生成 60 秒视频的审计与自动化方法决策

> 日期：2026-08-28
> 状态：代码已完成，等待服务器评测；不生成新视频
> 规模：4 methods × 128 prompts × 60 seconds，复用现有 512 个视频

## 1. 拉取后的事实

`main` 新增了 v186 的两组完整日志：

- `pf_native`：16 shards，每个 shard 8 条，共 128/128；
- `all_coverage_retrieval`：16 shards，每个 shard 8 条，共 128/128；
- 两组都使用 v181 `long60_seed0` 的 128 prompts、seed 0 和 240 latent frames；
- 日志中没有 traceback、OOM、assertion 或未完成 prompt；
- Retrieval 每个 shard 都打印：

```text
[CacheCompatibilityPolicy] recent=20:0 coverage=21:360 episode=22:0
coverage_policy=retrieval budget=9FFE read_budget=9FFE owner=HeadComposition
```

因此这一方法确实是 **360 个 heads 全部使用 Retrieval Coverage，读取预算为
9 FFE**，不是旧的五头 RCCP map。

PF 脚本中的旧注释曾写“no head config”，但真实日志明确加载
`configs/head_configs/best_labels.csv`。这是 PF 原生默认行为，不是 all-Recent；本轮已
修正文档注释，审计器会强制检查该 marker，并确认 PF 日志中没有
`CacheCompatibilityPolicy` 泄漏。

目前仓库只有日志，没有上传 MP4 或新指标，所以现在不能判断 Retrieval 与 PF 谁更
好。v198 的目的正是补齐这个缺口。

## 2. 为什么可以复用 v181，以及证据边界

比较清单包含：

| 方法 | 来源 | 角色 | 晋级必需 |
|---|---|---|---:|
| `all_coverage_retrieval` | v186 | 当前候选，all-head Retrieval，9 FFE | 是 |
| `all_recent` | v181 | 复用的等读取预算局部缓存对照，9 FFE | 是 |
| `sf_native` | v181 | 复用的 Self-Forcing 原生参考 | 是 |
| `pf_native` | v186 | 同轮次 PF 上下文 | 否 |

初看当前实验分支时，v181 manifest 绑定的 runtime SHA 与当前文件不同；但当前
工作树不是 v186 的生成快照。进一步按 Git 历史检查后得到：

- v186 PF 日志提交为 `448867aa`，Retrieval 日志提交为 `237198df`；
- 两个 artifact commit 中的 9 个 PF/SF runtime Git blob 与 v181 manifest **逐个
  SHA-256 完全一致**；
- 两个 artifact commit 中的 `pyramid-forcing.yaml` 也与 v181 完全一致；
- `448867aa..237198df` 之间只新增 Retrieval 日志，没有 tracked runtime 文件变化。

因此 v181 `all_recent` 可以作为匹配 tracked runtime、prompt、seed、checkpoint、
长度与 9-FFE read budget 的复用对照，不需要因为当前分支后来加入 profiling 代码而
重跑。审计器会直接对两个 artifact commit 执行 `git show` 并重新计算 blob SHA，
而不是比较当前工作树。

因此分析器会明确写入：

```text
matched_tracked_runtime_control_available = true
execution_clean_worktree_recorded = false
paper_claim_ready = false
```

第二项限制表示旧日志没有记录服务器当时是否存在未提交修改；它是 provenance 边界，
不是已经发现的 runtime mismatch。`paper_claim_ready=false` 的原因也不是缺少
all-Recent，而是 all-head Retrieval 只验证了 operator，尚未证明 selective
head/phase routing、跨 seed 或跨模型结论。另一个明确差异是 Retrieval 读取预算为
9 FFE，但内部 archive 存 12 FFE；它与 all-Recent 是等读取预算而非等存储预算。

## 3. 自动评测内容

### 3.1 媒体与来源审计

四个 CPU shard 并行完成：

1. 精确检查 0–127 的视频文件，禁止缺失、重复和额外 index；
2. `ffprobe` 检查 957 decoded frames、16 FPS、832×480；
3. `ffmpeg -xerror` 完整解码每个视频；
4. 计算每个 MP4 的 SHA-256，禁止同 prompt 跨方法误复用完全相同文件；
5. 检查 v186 的 32 个日志、每个 shard 的八个真实 prompt index 和 80/80 block；
6. 绑定 prompt、head map、PF labels、config、脚本及 v181 audit provenance。

审计成功后只创建 hardlink/symlink，不复制或重新编码视频。

### 3.2 VBench-Long

在一个统一 evaluator runtime 中重算 core-9。分析同时输出：

- full 60-second 指标；
- late-half 指标，用于观察后 30 秒的身份、背景、画质和时序退化；
- 逐 prompt paired delta、bootstrap 95% CI、win fraction 和 BH 校正；
- official Quality Score，以及互斥的 identity/background、temporal、semantic、
  visual、dynamic 分组。

`Dynamic Degree` 如果全为 1 或近似常数，只能说明 ceiling non-regression，不能作为
运动提升证据。

### 3.3 自动失败检测与相机补偿运动

`compute_temporal_jump_diagnostic.py` 检测：

- 长时间低运动或后期运动崩溃；
- temporal jump / appearance outlier；
- 黑帧、白帧、低对比和 edge-density 异常。

随后复用 v193 的相机补偿光流，把全局相机运动从残余局部运动中分离。运动主判断只
比较 Retrieval 与 `all_recent`、`sf_native`；PF 不参与晋级门槛。

## 4. 自动决策

本轮是 exploratory method selection，不设置“必须击败 PF”的条件。

1. Retrieval 相对匹配 tracked runtime 的 `all_recent` 和 `sf_native` 的 full 与
   late-half quality、identity、temporal、visual 必须位于预先写入的开发容忍区；
2. 自动 temporal safety 必须通过；
3. 至少满足以下一项：
   - 相对 all-Recent 的 quality 或 identity 有经 paired CI/BH 支持的正增益；
   - 相机补偿后的局部运动相对 all-Recent 和 SF 都呈正向信号。

输出可能为：

| Recommendation | 含义 |
|---|---|
| `promote_retrieval_operator_to_selective_routing_validation` | 候选通过，直接进入 selective head/phase routing 验证，不重跑 all-Recent |
| `promising_requires_same_runtime_all_recent_confirmation` | 只有 Git artifact runtime 校验意外失败时才补 all-Recent |
| `noninferior_but_no_clear_long_history_gain` | 没有明显退化，但尚无选择 Retrieval 的证据 |
| `do_not_promote_all_head_retrieval` | 质量、身份或时序超出容忍区 |
| `reject_retrieval_due_to_automatic_temporal_failure` | 自动检测到重复的长程异常 |

自动结论不要求人工 review。分析器最多列出四个 metric conflict / failure prompt，供
定位原因；不要盲审全部 512 个视频。

## 5. 服务器运行命令

先更新代码。四个节点的 `NODE_RANK` 分别为 0、1、2、3。

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
```

### 5.1 四节点并行媒体审计

每个节点执行一条；这是 CPU/共享存储任务，不占 GPU：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> V198_AUDIT_WORKERS=8 \
  bash scripts/run_v198_long60_evaluation.sh audit
```

节点 0 检查并构建统一 comparison manifest：

```bash
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh audit-status
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh prepare
```

若只有一个 CPU 节点，可改用：

```bash
NUM_NODES=1 NODE_RANK=0 V198_AUDIT_WORKERS=4 \
  bash scripts/run_v198_long60_evaluation.sh audit-all
NUM_NODES=1 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh prepare
```

### 5.2 四节点、32 卡 VBench-Long

四个节点分别执行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v198_long60_evaluation.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v198_long60_evaluation.sh eval
```

节点 0 查看状态并汇总；`collect` 会同时计算 temporal diagnostics：

```bash
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh status
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh collect
```

若仅有少量 VBench job 缺失，可在一个节点恢复，而不是重算完整网格：

```bash
NUM_NODES=1 NODE_RANK=0 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v198_long60_evaluation.sh resume-missing
```

### 5.3 相机补偿运动

四个节点分别执行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> V198_CAMERA_WORKERS=8 \
  bash scripts/run_v198_long60_evaluation.sh camera-compute
```

节点 0 合并并给出最终决策：

```bash
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh camera-status
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh camera-collect
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh decision
NUM_NODES=4 NODE_RANK=0 bash scripts/run_v198_long60_evaluation.sh package
```

## 6. 需要推送的小文件

运行结束后推送：

```text
runs/v198_audited_long60/contracts/source_manifest.json
runs/v198_audited_long60/audits/*.json
runs/v198_audited_long60/vbench_comparison/comparison_manifest.json
runs/v198_audited_long60/metrics/vbench_core9_summary.{json,csv,md}
runs/v198_audited_long60/metrics/temporal_diagnostics.{csv,contract.json}
runs/v198_audited_long60/analysis/v198_long60_operator.{json,md}
runs/v198_audited_long60/camera_motion/metrics/*.csv
runs/v198_audited_long60/camera_motion/metrics/*.json
runs/v198_audited_long60/camera_motion/analysis/v193_camera_motion.json
runs/v198_audited_long60/evidence_manifest.json
```

不需要上传 MP4、VBench clip、模型或 `vbench_long_parts` 中的大量中间文件。

## 7. 与当前主 idea 的关系

v198 只回答“**all-head Retrieval 作为长期历史算子是否值得继续**”，不是最终论文
方法，也不能替代 v189–v195 的 Head × Denoising Phase profiling/causal ladder。

- 若 v198 正向：把 Retrieval 作为可用 operator，直接继续验证哪些 head/phase 真正
  从它受益；现有 all-Recent 已通过 artifact blob 校验，不重复生成。
- 若 v198 非劣但无增益：保留它作为 profiling candidate，不把 all-head Retrieval
  写成贡献。
- 若 v198 明显退化：停止围绕 Retrieval 优化分类器，回到 Landmark 或 Recent 的
  operator 选择，避免继续消耗生成算力。

这一步把已生成视频的价值用完，同时将下一次 GPU 生成限制为真正必要的一组对照。
