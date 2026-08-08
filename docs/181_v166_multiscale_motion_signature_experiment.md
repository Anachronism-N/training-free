# v166: Multi-scale Motion-Signature Recall

## 1. 本阶段目标

v165 已经证明：在固定 `sink1 + reservoir2 + recalled pair2 + recent4`
预算下，基于历史运动方向检索一个原子相邻帧对是可运行的；但 Tie05 在个别
prompt 上出现明显的后期运动衰减，且原始综合指标把两个完全重复的 ViCLIP
分数分别计入 history 和 temporal，解释口径不可靠。

v166 只回答两个问题：

1. 当前块的长尺度方向与历史相邻帧对的短尺度方向直接比较，是否存在尺度
   不匹配？
2. 在方向一致之外，检索与当前运动幅度相近的历史帧对，能否减少后期减速，
   同时保留身份、背景和视觉质量？

本轮不运行 PF，不运行 ABA，不扫阈值，也不改变 cache 预算。四个已有方法
全部复用 v165 视频；只新生成两个方法，共 `2 * 16 = 32` 个 30 秒视频。

## 2. v165 指标口径修正

新增：

- `scripts/vbench_quality_contract.py`
- `scripts/analyze_v165_corrected_metrics.py`

审计了 6 个方法、16 个 prompt、15 个 split clip，即 1440 对结果：
`overall_consistency` 与 `temporal_style` 逐 clip 在 `1e-12` 内完全相同。
因此旧口径对同一个 custom-prompt ViCLIP 分数重复计权。

修正后诊断组互不重叠：

| 组 | 原始维度 |
|---|---|
| Identity/background | subject consistency, background consistency |
| Temporal mechanics | temporal flickering, motion smoothness |
| Semantic alignment | overall consistency，仅一次 |
| Visual quality | aesthetic quality, imaging quality |
| Motion amount | dynamic degree |

同时按 VBench 官方 `scripts/constant.py` 的 min/max 和权重计算 Quality Score；
dynamic degree 权重为 0.5，其余六个质量维度权重为 1。

修正后的 v165 开发集结果：

| 方法 | Quality Score | Identity/background | Temporal mechanics | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| Tie05 | 84.3704 | 0.967547 | 0.971237 | 0.236061 | 0.672114 | 0.766667 |
| StateMotion | 84.2036 | 0.967969 | 0.971306 | 0.238391 | 0.667076 | 0.762500 |
| DirectionMatch | 84.1985 | 0.966953 | 0.970755 | 0.236783 | 0.667644 | 0.770833 |
| DirectionFresh | 84.1850 | 0.967479 | 0.970573 | 0.238664 | 0.669190 | 0.762500 |
| Tie03 | 84.0450 | 0.967008 | 0.970841 | 0.237122 | 0.667469 | 0.750000 |
| SF native | 83.0371 | 0.964820 | 0.975106 | 0.233174 | 0.652452 | 0.641667 |

Tie05 相对 DirectionMatch 的 paired Quality Score 为 `+0.1719`，10/16 prompt
为正，但 95% bootstrap CI 为 `[-0.0338, 0.3974]`；它不是稳定胜出。相对
SF 为 `+1.3333`，11/16 为正，CI 为 `[0.3450, 2.3681]`。v165 冻结的旧
decision 不重写；本文件是解释口径 addendum。

## 3. 描述子与现有问题

每层为启用该路由的 heads 共享一次 clean-KV 描述子计算：

1. 对每帧按固定 token step 采样 clean K/V；
2. 对采样 token 和路由 heads 求 `K mean`、`V mean`、`V std`；
3. 拼接后 L2 normalize，得到帧描述子 `z_t`。

v164/v165 DirectionMatch 使用：

```text
query direction     = normalize(z_block_last - z_block_first)
candidate direction = normalize(z_pair_end - z_pair_start)
score               = cosine(query direction, candidate direction)
```

query 跨整个生成块，candidate 只跨一个相邻帧间隔。这不是同一时间尺度，
而且 cosine 完全丢弃位移幅度；一个方向正确但速度很慢的旧片段可以成为最佳
历史。v166 针对这两个具体问题，不引入新模型或额外视觉 encoder。

## 4. v166 方法

### 4.1 两尺度方向

当前 query：

```text
q_local   = normalize(z_last - z_last-1)
q_context = normalize(z_last - z_first)
```

历史 pair 在写入 memory bank 时同时保存：

```text
m_local   = normalize(z_pair_end - z_pair_start)
m_context = normalize(z_source_block_last - z_source_block_first)
```

方向分数：

```text
s_local   = cosine(q_local, m_local)
s_context = cosine(q_context, m_context)
s_dir     = mean(s_local, s_context)
```

若某个 delta 的 norm 接近 0，则只使用可用尺度。方向 gate 保持为
`s_dir >= 0.1`，没有新增阈值。

### 4.2 无尺度参数的运动幅度匹配

局部幅度为相邻描述子 delta norm。块级幅度除以块内间隔数，转换成每步
幅度后再比较：

```text
r_local   = min(|q_local|, |m_local|) / max(|q_local|, |m_local|)
r_context = min(|q_context|/steps_q, |m_context|/steps_m)
            / max(|q_context|/steps_q, |m_context|/steps_m)
r_mag     = sqrt(r_local * r_context)
```

`r_mag` 在 `[0, 1]`，不需要温度或带宽。主方法分数：

```text
s_motion = s_dir * r_mag
```

候选按 score 降序、再按时间更新者优先。若没有候选通过方向 gate，仍读取
age-eligible 的最新原子帧对，保证读取预算与旧方法相同。

### 4.3 两个隔离变量

| 方法 | 排序分数 | 目的 |
|---|---|---|
| MultiScaleDir | `s_dir` | 只验证时间尺度对齐 |
| MultiScaleMotion | `s_dir * r_mag` | 再验证幅度匹配是否修复减速 |

没有 stale tie、age penalty、学习权重或 metric-tuned 参数。

## 5. Cache 契约

仅 Middle10 层使用实验 cache；其余层为 `sink1 + recent8`。

Middle10 每个 head：

| 组件 | 写入容量 | 读取容量 | 更新 |
|---|---:|---:|---|
| Sink | 1 帧 | 1 帧 | 固定首帧 |
| Temporal reservoir | 2 帧 | 2 帧 | 原有 online reservoir |
| Motion-pair archive | 4 对 | 1 对，即 2 帧 | coherent-motion admission，stale refresh horizon 12 |
| Recent | 4 帧 | 4 帧 | FIFO |

最大读取为 `1 + 2 + 2 + 4 = 9` 个 full-frame equivalents。motion pair 必须
连续、成对写入和成对读取；composition 是唯一 dynamic-history owner。

## 6. 冻结方法网格

| 方法 key | 来源 | 视频数 |
|---|---|---:|
| `sf_native` | 复用 v165 | 16 |
| `ours_middle10_reservoir2_directionmatch1` | 复用 v165 | 16 |
| `ours_middle10_reservoir2_dirstaletie005` | 复用 v165 | 16 |
| `ours_middle10_reservoir2_multiscaledir1` | v166 新生成 | 16 |
| `ours_middle10_reservoir2_multiscalemotion1` | v166 新生成 | 16 |
| `ours_middle10_reservoir2_statemotionpair1_reference` | 复用 v165 | 16 |

prompt 为 `moviegen_128_qwen_v154_diverse16.txt` 中冻结的 16 条多样化
MovieBench prompt；每条 30 秒、沿用相同 seed 和生成配置。

四节点、每节点 8 卡时，总任务为 96，但只有 32 个生成任务；每节点约 8 个
新生成，其他任务只做经过 hash/contract 校验的链接复用。

## 7. 服务器命令

### 7.1 每个节点先 preflight

四个节点分别设置 `NODE_RANK=0,1,2,3`：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git fetch origin
git switch codex/v166-multiscale-motion
git pull --ff-only origin codex/v166-multiscale-motion

NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v166_multiscale_motion_moviebench16.sh preflight
```

preflight 必须确认：

```text
methods=6, total_tasks=96；每个节点 node_tasks=24、new=8、reused=16
```

### 7.2 四节点并行生成

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v166_multiscale_motion_moviebench16.sh generate
```

全部节点结束后，只在 node 0：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v166_multiscale_motion_moviebench16.sh audit

bash scripts/run_v166_multiscale_motion_moviebench16.sh mechanism
```

若 mechanism gate 为 false，不运行视频指标；先上传：

```text
runs/v166_multiscale_motion_moviebench16/full8/automated_screen/multiscale_motion_trace.json
runs/v166_multiscale_motion_moviebench16/full8/traces/*.policy.jsonl
runs/v166_multiscale_motion_moviebench16/full8/logs/
```

### 7.3 自动安全与代理指标

只在一个节点执行，六个方法各占一张评测卡：

```bash
EVAL_GPUS=0,1,2,3,4,5 \
  bash scripts/run_v166_automated_screen.sh all
```

这一阶段不需要先看视频。

### 7.4 VBench-Long core-9

node 0 准备 comparison：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v166_vbench_long.sh prepare
```

四个节点分别 split：

```bash
NODE_RANK=<0..3> NUM_NODES=4 bash scripts/run_v166_vbench_long.sh split
```

四节点 preflight 和评测：

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v166_vbench_long.sh preflight

NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v166_vbench_long.sh eval
```

中断后在任意一个节点只补缺失 job。底层会先汇总全部 54 个 job，因此该动作
必须使用单节点调度：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v166_vbench_long.sh resume-missing
```

全部完成后 node 0：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v166_vbench_long.sh collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v166_vbench_long.sh prepare-review
```

`collect` 自动运行逐 prompt paired 分析；`prepare-review` 自动选择最多两个
prompt，并只比较 MultiScaleMotion 与 DirectionMatch，最多四个盲审视频。

## 8. Debug 与机制 gate

每个 candidate trace 包含：

- `local_direction_similarity`
- `context_direction_similarity`
- `multiscale_direction_similarity`
- query/candidate local magnitude
- query/candidate context magnitude per step
- local/context/multi-scale magnitude similarity
- `motion_signature_score`
- `direction_pass`, `state_pass`, `selection_score`
- selected pair、DirectionMatch counterfactual 的最终 pair 与 fallback、selected age
- archive 中真实 pair frame ids 和实际读取 frame ids

`analyze_v166_multiscale_motion_trace.py` 以 trace 中记录的 cosine 与原始 norm
为基础，独立重算 magnitude match、聚合 score、gate、DirectionMatch
counterfactual、最终 argmax 和实际读取。它不声称从未保存的高维描述子重新
验证 cosine 本身。mechanism gate 要求：

1. 16/16 prompt trace 完整；
2. archive 至少出现两个候选，multi-candidate 排序真实执行；
3. 两个新方法均至少一次改变 legacy direction choice；
4. 所有分数和选择可在 tolerance 内重算；
5. 每次读取为 0 或一个连续原子 pair；
6. 没有 read-budget violation 或 contract failure。

## 9. 评测与最小人工复核

主要结果：

1. 官方 VBench Quality Score；
2. dynamic degree；
3. identity/background；
4. temporal mechanics；
5. semantic alignment；
6. visual quality；
7. 每 prompt paired delta、win/tie/loss 和 bootstrap CI；
8. late-motion ratio、background drift、flicker、loop 和综合代理指标。

自动盲审选择：

1. MultiScaleMotion 相对 DirectionMatch 的最大 Quality Score 下跌 prompt；
2. 在 dynamic 不下降的 prompt 中，Quality Score 提升最大的 prompt。

这最多产生两对、四个视频。该 review 是 metric-adaptive engineering triage，
不能作为无偏论文人评。

## 10. 结果决策

### A. MultiScaleMotion 优于 MultiScaleDir，且 p10 类减速消失

保留完整 motion signature，进入更大 prompt 集。论文技术点可表述为：多尺度
方向与尺度无关的幅度匹配共同定义 motion-compatible episodic recall。

### B. MultiScaleDir 优于 MultiScaleMotion

保留尺度对齐，删除幅度乘法。说明幅度 norm 对层或 timestep 的尺度仍不够
可比；不能为了故事保留负贡献模块。

### C. 两者均不优于 DirectionMatch

拒绝本假设，不继续扫混合权重或幅度温度。转向改变 descriptor 本身或只在
检测到运动衰减时触发 recall，而不是重复调排名超参数。

### D. 指标提升但自动安全失败

先定位具体 prompt、late segment 与 trace selection。只有自动选择出的最多四
个视频需要人工 review；不能用平均分掩盖 polygon noise、身份替换、背景漂移
或明显静止。

## 11. Claim boundary

- v166 是 16-prompt adaptive development experiment，不是 held-out 论文结果。
- MultiScaleDir 与 MultiScaleMotion 只使用现有 clean-KV descriptor，不训练、
  不引入外部模型。
- cache、读预算、head/layer gate、prompt、seed 与生成配置均冻结。
- 本轮不声称优于 PF，也不运行 PF；PF 仅是历史工程基座的一部分。
- ABA 场景切换延后，直到单 prompt 长视频主路径有稳定收益。
