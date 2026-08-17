# v184 同步状态与 v186 Phase-Conditioned Operator Screen

## 1. 本轮同步结论

2026-08-17 同步后，本分支和远端均停在 `02aea92f`，`origin/main` 仍为
`8c269c4d`。仓库没有新增的 v184 或 v185 结果；本地也只有 v181 的历史视频。
因此当前不能声称 early/late schedule 或某个确定性 Coverage operator 已经有效。

当前可靠证据仍是：

- v183 的 static strict-five RCCP 相对 all-Recent 没有生成收益，应停止该 membership；
- all-Coverage 相对 all-Recent 提高 Quality `+0.6576` 和 Dynamic `+0.1099`，但
  Identity 下降 `-0.00169`；
- Coverage 对低运动 prompt 的作用更明显，对高运动 prompt 基本无增益；
- v184 正在检验长期历史应注入哪几个 noisy denoising calls；
- v185 只评估恢复出的 v181 60 秒视频，因 provenance 不完整，只能作为探索证据。

## 2. 为什么下一步是 v186

v182 已经实现 `landmark/prototype/retrieval`，但只给 v177 的五个静态 heads 使用。
v183 已否定这组静态 membership，所以不能把 v182 直接当作后续方法。

v186 将 operator 比较迁移到 v184 的全头 phase-conditioned 框架，只回答：

> 在 v184 自动选出的同一个 noisy-call schedule 下，哪一种确定性的四帧长期记忆
> 构造能够保留运动增益，同时减少 Reservoir 的身份或时序代价？

它不再比较 head membership，不运行 PF 或 ABA，也不重复 all-Recent 和 Reservoir
视频。v184 的这两组已审计视频会被直接复用。

## 3. 自动选择 v184 schedule

v184 分析器新增了预先固定的最小干预规则：

1. 候选必须在四个主指标的 Pareto front，且通过原有 directional gate；
2. 优先选择 Coverage noisy calls 数量最少的候选；
3. 剂量相同时，依次按 Identity、Temporal、Quality、Dynamic 相对 all-Recent 的
   delta 排序。

分析文件新增：

```text
selected_for_operator_screen
operator_screen_selection_rule
```

v186 `prepare` 会校验 v184 decision、generation manifest、experiment contract、audit、
prompt 和视频集合的哈希。只有 recommendation 为
`advance_phase_schedule_to_operator_screen` 且自动选择结果属于 promoted 集合时才会
继续。v184 没通过时，v186 会直接失败，不允许手工指定一个看起来较好的 schedule。

## 4. v186 方法与缓存

所有方法使用相同 32 条 systematic MovieGen-Qwen prompts、seed 0、约 30 秒视频。

| 方法 | 来源 | Noisy Coverage operator | Middle 读取/存储 | 作用 |
|---|---|---|---:|---|
| `all_recent` | 复用 v184 | 无 | 0/0 | 局部控制组 |
| `phase_reservoir` | 复用 v184 获胜 schedule | 固定 seed Reservoir | 4/4 | 随机长期覆盖参考 |
| `phase_landmark` | 新生成 | 语义一致性和新颖性在线地标 | 4/4 | 确定性 coreset |
| `phase_prototype` | 新生成 | 连续片段的 temporal medoid | 4/4 | 确定性片段原型 |
| `phase_retrieval` | 新生成 | query 相关性和多样性检索 | 4/12 | 额外存储的检索上界 |

统一读取预算为：

```text
Recent   = sink1 + recent8                         = 9 FFE
Coverage = sink1 + exactly one middle4 + recent4  = 9 FFE
Clean    = Recent                                  = 9 FFE
```

每次 clean commit 更新 Recent 与所选 structured middle bank。Noisy calls 不写长期
memory，只按 v184 schedule 在 Recent/Coverage readout 间切换。Retrieval 读取仍为四
帧，但 archive 为 12 帧，分析和选优会显式报告该存储差异。

## 5. 新增防错与 debug 信息

新增参数：

```text
--pyramidkv_cache_compatibility_denoise_coverage_policy \
  {reservoir,landmark,prototype,retrieval}
```

运行时和审计会检查：

- operator 与 history policy 必须严格对应；
- structured Coverage 每个 head 只能有一个 capacity=4 的 middle strategy；
- clean pass 必须使用 Recent；
- 每次读取不得超过 9 FFE，Coverage middle 不得超过 4 FFE；
- trace 中的 `coverage_operator` 必须与命令一致；
- 实际 `source_kind` 必须分别为 `semantic_landmark`、`temporal_prototype` 或
  `semantic_retrieval`；
- trace 记录真实 physical frame id 和 frame age，middle age 必须至少为 4；
- 至少观察到一次真实 structured middle readout，防止命令正确但算子没有生效；
- 任意 traceback、OOM、预算漂移或 trace warning 都会阻止发布视频。

## 6. 服务器执行

### 6.1 先重新生成 v184 自动 decision

v184 core-9 已完成时，仅需在 node 0 重跑分析，不重复评测：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v184_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v184_vbench_long.sh decision
```

若输出不是 `advance_phase_schedule_to_operator_screen`，停止 v186，并上传 v184 的
`analysis/v184_denoise_phase_screen.json`。若通过，继续：

```bash
NODE_RANK=0 bash scripts/run_v186_phase_operator_screen_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v186_phase_operator_screen_32gpu.sh preflight
```

### 6.2 三方法 smoke

只需三张 GPU：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2 \
  bash scripts/run_v186_phase_operator_screen_32gpu.sh smoke
NODE_RANK=0 NUM_NODES=1 \
  bash scripts/run_v186_phase_operator_screen_32gpu.sh audit-smoke
```

`audit-smoke` 通过即可进入 32 prompts，不需要先人工排序三个视频。若失败，上传
对应 log 和 schedule trace，不要绕过审计。

### 6.3 32 prompts 生成

四个 8-GPU 节点分别使用相对 `NODE_RANK=0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v186_phase_operator_screen_32gpu.sh generate32
```

每个方法依次使用 32 张卡，只新生成 `3 x 32 = 96` 个视频。完成后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v186_phase_operator_screen_32gpu.sh status
NODE_RANK=0 bash scripts/run_v186_phase_operator_screen_32gpu.sh audit-screen
NODE_RANK=0 bash scripts/run_v186_vbench_long.sh prepare
```

### 6.4 VBench-Long core-9

四节点分别执行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v186_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v186_vbench_long.sh eval
```

最后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v186_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v186_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v186_vbench_long.sh decision
NODE_RANK=0 bash scripts/run_v186_phase_operator_screen_32gpu.sh package
```

五方法共 `5 x 9 = 45` 个评测任务，其中两个方法直接使用 v184 视频。

## 7. 自动晋级标准

确定性候选首先必须相对 all-Recent 保留 v184 的基本效应：

- Quality delta `>= 0`；
- Identity delta `>= -0.001`；
- Dynamic delta `>= +0.02`；
- Temporal delta `>= -0.002`。

随后相对 Reservoir 必须满足：

- Quality、Identity、Dynamic、Temporal 分别不低于 `-0.10/-0.0005/-0.02/-0.001`；
- Identity `+0.0005`、Temporal `+0.001` 或 Quality `+0.10` 至少满足一项；
- 位于四指标 Pareto front。

这些是 32-prompt development gate，不是论文显著性或 non-inferiority margin。多个
候选通过时，优先等存储的 Landmark/Prototype，再按 Identity、Temporal、Quality、
Dynamic 排序；Retrieval 不会凭借隐藏的 12 帧 archive 获得优先级。

自动 recommendation 不依赖人工 review。只有候选晋级后才输出最多四组三视频
`all_recent / reservoir / candidate` 的定向冲突样本。

## 8. 结果后的下一步

- **确定性候选晋级**：冻结该 operator 和 schedule，在未参与 v184/v186 开发的 fresh
  128 prompts 上与 SF、all-Recent、Reservoir 做确认，再考虑 60 秒和跨模型。
- **Reservoir 有效但确定性候选均失败**：结论是 phase actuator 可复现，但当前
  structured memory 不成立；不要把随机 Reservoir 包装成最终创新，应进入在线
  motion-deficit gate 或重新设计 operator。
- **所有方法都未保留 v184 效应**：优先审计 v184 复用 provenance、runtime 和评测，
  不做阈值扫描。

只有 fresh128 确认后，论文主线才可以写成“phase-conditioned long-history exposure +
deterministic structured Coverage”。v186 本身仍是方法开发实验。
