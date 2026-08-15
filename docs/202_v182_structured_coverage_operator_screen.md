# v182：确定性 Structured Coverage 缓存筛选

## 1. 拉取后的当前结论

仓库远端与本地分支均停在 `e085d36a`。没有新增的正式 v178/v179
VBench-Long 结果：v177 profiling 已完成，v178/v179 仍只有部分视频，因而当前
5-head RCCP map 仍是 profiling hypothesis，而不是已经证明有效的生成方法。

v177 的 `Coverage = sink1 + reservoir4 + recent4` 使用固定随机种子的
Reservoir Sampling。它适合作为以下问题的无偏对照：

> 在不知道哪些历史帧更重要时，均匀覆盖长期历史是否有帮助？

但它不适合作为最终论文的主要缓存创新：选择规则与视频语义、状态和运动都无关，
难以解释为什么被保留的帧对身份或背景有用。本轮因此不调整 head membership，
只替换 Coverage 的 middle operator，先找出可解释且不退化的确定性候选。

## 2. 冻结的实验变量

### 2.1 Head membership

沿用 v177 的五个 Coverage heads：

```text
L0H10, L5H3, L6H6, L8H6, L23H2
```

其余 355 heads 全部使用 Recent。固定 map 已写入：

```text
configs/head_maps/v177_strict5_coverage.csv
```

### 2.2 Cache read budget

```text
Recent   = sink1 + recent8                       = 9 FFE
Coverage = sink1 + middle4 + recent4             = 9 FFE
```

所有路由由 `HeadComposition` 独占，开启 dynamic RoPE；禁止 legacy dynamic
history、LifeCache、Structured Memory、Commit Forcing 和 scene reset。这样方法间
唯一变化是 label 21 的四个 middle frames 如何选择。

## 3. 五个方法

| 方法 | Coverage middle | 随机 | 读取/存储 FFE | 作用 |
|---|---|---:|---:|---|
| `all_recent` | 无，改为 recent8 | 否 | 0/0 | 局部缓存控制组 |
| `strict5_reservoir` | 历史均匀 Reservoir | 是（固定 seed） | 4/4 | v177 原始随机对照 |
| `strict5_landmark` | 语义一致性 + 新颖性在线 coreset | 否 | 4/4 | 保留身份一致且互补的历史地标 |
| `strict5_prototype` | 连续稳定片段压缩为 medoid | 否 | 4/4 | 用代表帧覆盖不同时间片段 |
| `strict5_retrieval` | 当前 query 相关性 + MMR 多样性检索 | 否 | 4/12 | 按当前生成状态读取相关历史 |

Retrieval 的读取预算仍为四帧，但内部 archive 默认保存 12 帧；因此它不是严格的
等存储对照。代码和结果会单独报告这一点。若它获胜，下一轮必须补做 archive4 或
显式 memory/latency 对比，不能把额外存储隐藏掉。

`landmark`、`prototype` 和 `retrieval` 使用仓库中已经实现并测试过的策略。本轮的
贡献不是把既有策略改名，而是确定哪一种能与 RCCP head compatibility 形成有效
组合。论文写作必须追溯这些策略的来源；只有后续针对 RCCP 做出的新选择准则、
profiling 和因果验证才能作为我们的贡献。

## 4. 代码改动与防错

新增 CLI：

```text
--pyramidkv_cache_compatibility_coverage_policy \
  {reservoir,landmark,prototype,retrieval}
```

默认值仍为 `reservoir`，所以 v173-v181 的旧命令和行为不变。运行时强制检查：

1. label 21 只能激活一个 middle operator；
2. middle capacity 必须为 4；
3. Coverage 必须为 `sink1 + middle4 + recent4`；
4. structured operator 不允许混入 reservoir-based profiling；
5. trace 中总读取不得超过 9 FFE；
6. middle 不得与 sink/recent 重叠；
7. 实际策略类型必须与命令一致；
8. route count 必须是 `355/5/0` 或 `360/0/0`。

每个 shard 保存 policy trace 和 role-event trace。自动分析输出真实 middle frame、
frame age、更新/检索原因、每个 head 的读取次数及存储开销。

## 5. 第一阶段：一个 30 秒视频

节点 0：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v182_structured_coverage_32gpu.sh prepare
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v182_structured_coverage_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v182_structured_coverage_32gpu.sh status
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v182_structured_coverage_32gpu.sh audit-smoke
NODE_RANK=0 bash scripts/run_v182_structured_coverage_32gpu.sh trace-smoke
```

五个方法并行占用五张卡，默认生成 diverse16 中的 prompt 3。只有当 audit 全部
PASS、没有多边形噪声且 trace 的实际路由正确时，才进入 16-prompt 筛选。此阶段
人工只需查看五个同 prompt 视频是否存在明显崩溃，不做细粒度排序。

## 6. 第二阶段：16 prompts × 30 秒

推荐使用两个 8-GPU 节点；所有方法依次运行，避免 method/node 混杂。两个节点
分别设置相对 `NODE_RANK=0` 和 `1`：

```bash
NUM_NODES=2 NODE_RANK=<0|1> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v182_structured_coverage_32gpu.sh generate16
```

完成后在节点 0：

```bash
NODE_RANK=0 bash scripts/run_v182_structured_coverage_32gpu.sh status
NODE_RANK=0 bash scripts/run_v182_structured_coverage_32gpu.sh audit-screen
NODE_RANK=0 bash scripts/run_v182_structured_coverage_32gpu.sh trace-screen
NODE_RANK=0 bash scripts/run_v182_vbench_long.sh prepare
```

VBench-Long core-9 可使用四个节点：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v182_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v182_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v182_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v182_vbench_long.sh decision
```

主要观察 `official_quality_score`、`identity_background` 和
`dynamic_degree`，同时保留全部 core-9 指标。16 prompts 只用于开发筛选，不做
显著性或论文 SOTA 声明。分析器最多输出四个 identity/motion/quality 冲突样本供
人工 review，不再盲审 80 个视频。

## 7. 自动晋级规则

确定性候选只有同时满足以下条件才进入下一轮：

1. 位于 quality、identity/background、dynamic degree 的 Pareto front；
2. 相对 Reservoir 和 all-Recent，quality 与 identity 平均下降均不超过 0.005；
3. 相对两者，dynamic degree 平均下降不超过 0.02；
4. 相对 Reservoir，quality 或 identity 至少提升 0.002。

这些阈值只是节省算力的开发容忍区，不是论文统计阈值。若多个候选晋级，优先顺序
是等存储的 `landmark/prototype`，其次才是 archive 更大的 `retrieval`。

## 8. 晋级后的必要工作

v177 的五个 heads 是在 Reservoir operator 下得到的，不能直接把 structured
operator 的 16-prompt 好结果包装成最终方法。晋级后必须：

1. 为获胜 operator 重建 representation-complete Union teacher；
2. 在新的 profiling prompts 上重新计算每个 head 的 operator compatibility；
3. 冻结新 map 和阈值；
4. 在未用于 profiling/筛选的新 prompts 上，与 all-Recent、Reservoir 及
   layer/count-matched hard negatives 做生成侧因果验证；
5. 最后才扩大到 128 prompts 和跨模型实验。

若三种确定性候选都未晋级，正确结论是“当前 RCCP proxy 只支持随机 Coverage
control，尚未形成可投稿方法”，而不是在这 16 prompts 上调阈值或挑视频。

## 9. 当前可讲的论文方向

如果某个确定性 operator 通过重新 profiling 和独立生成验证，故事可以变为：

1. **Operator-aware head profiling**：不用 PF 的时间 QK 正负类别，而是在完整
   teacher 下直接测量每个 head 对不同有限预算记忆算子的 residual compatibility。
2. **Selective structured history**：绝大多数 heads 使用 local Recent，仅把统计稳定
   的少数 heads 路由到可解释的 semantic/segment/query Coverage memory。
3. **Causal budget-matched validation**：用相同 head 数、相同 layer、相同读取预算的
   hard negatives 证明收益来自 compatibility，而不是多看历史或降低运动。

目前只有第 1 点完成了 profiling 内部验证；第 2、3 点仍需 v182 及后续独立实验。
