# v187 结果复盘与 v201 AR-Horizon 因果实验

## 1. 本轮同步结论

本地已同步到 `origin/main@40d4e942`。v187 的 128-prompt、30 秒生成和
VBench-Long 评测已经完成。该实验中的 `pf_hybrid_retrieval` 保留 PF 的
Oscillatory/Wave cyclic 路由，但同时做了三项改动：

1. PF Stable-positive/Anchor 的 stride middle 改为 Retrieval；
2. PF Stable-negative/Veil 的 merge middle 也改为 Retrieval；
3. 两类 Stable head 的 `sink3` 改为 `sink1`。

与同 prompt、同 seed 的 PF native 相比，结果如下。`temporal_style` 与
`overall_consistency` 是重复读数，因此不作为第八个独立证据；
`dynamic_degree=1.0` 在两边均饱和，也不能用于区分方法。

| 指标 | PF native | Hybrid retrieval | 差值 |
|---|---:|---:|---:|
| Aesthetic quality | 0.615354 | 0.609384 | -0.005969 |
| Background consistency | 0.968033 | 0.964524 | -0.003508 |
| Imaging quality | 0.693689 | 0.691034 | -0.002655 |
| Motion smoothness | 0.986678 | 0.985437 | -0.001241 |
| Overall consistency | 0.249485 | 0.246192 | -0.003293 |
| Subject consistency | 0.978939 | 0.974839 | -0.004100 |
| Temporal flickering | 0.974417 | 0.971928 | -0.002489 |

因此，**v187 当前候选不能作为主方法继续扩大实验**。但它也不能证明
“PF 三类 middle operator 都不可替换”，因为 head 集合、两个 operator 和 sink
预算被同时修改，缺少单因素归因。该负结果真正支持的是：缓存算子不能只按一个
静态 head 标签整体替换，需要继续验证算子效用是否随 denoising call 和 AR 历史长度变化。

## 2. v201 要回答的问题

v201 是 v189/v200 profiling 的第一个严格因果生成实验，假设为：

> 在相同 Coverage 读预算下，根据 `(AR position, denoising call, layer, head)`
> 动态选择读取历史的单元，应优于固定的 `(call, layer, head)` 选择；如果把同一个
> 动态选择表沿 AR 轴错位，收益也应消失。

这里的 Coverage operator 分别由 v200 自动晋级的 Landmark 或 Retrieval 构成。
没有通过 v200 统计 gate 的 operator 不会进入生成。v201 不依赖 v187 的视频，也不再
尝试整体替换 PF 路由。

## 3. v200 新增的运行时产物

对每个通过 gate 的 operator，v200 现在冻结三个版本 2 路由图：

- `static_top10`：所有 AR 位置复用同一组 top-10% Head×Phase 单元；
- `horizon_top10`：每个 profiling AR 位置独立选择 top-10%；
- `horizon_shift_top10`：将动态路由沿 AR 轴循环移动半个周期。

三者在每一个 AR 位置都有完全相同的 Coverage 单元数。排序使用稳定
`mergesort`，因此分数相同时选择结果也可复现。每份 map 都带 SHA256、选择数量和
12 个 profiling 位置。生成时，当前 AR frame 映射到最近的 profiling 位置；距离相同
时选择更早位置。实际 map id、位置索引、参考 frame、head mask 和 cache source 都写入
trace，审计失败时不发布视频。

## 4. v201 方法与预算

每个通过 v200 gate 的 operator 生成 5 个方法，每个方法使用冻结的 32 条分类
holdout prompt、seed `20100`、30 秒视频：

| 方法 | 作用 |
|---|---|
| `all_recent` | operator-matched 的纯局部端点 |
| `all_coverage` | 所有 Head×Phase×Horizon 单元读取 Coverage 的上界端点 |
| `static_top10` | 等预算静态选择对照 |
| `horizon_top10` | 主候选 |
| `horizon_shift_top10` | 等预算、错误 AR 对齐的因果对照 |

关键比较是 `horizon_top10` 对 `static_top10` 和 `horizon_shift_top10`。前者排除
“只是选到一组好 head”的解释，后者排除“只要使用任意随时间变化的 mask 就有效”的
解释。`all_recent` 与 `all_coverage` 只给出局部基线和全覆盖上界。

## 5. 硬门控

只有 v200 的 `recommendation=advance_head_phase_horizon_to_runtime_design` 且至少一个
operator 通过 horizon gate 时，v201 `prepare` 才能成功。否则脚本会直接退出，不消耗
GPU。v201 的 32 条 holdout 从未参与 v200 selector 构造。

生成后的自动决策要求主候选：

1. 对 `all_recent`、`static_top10`、`horizon_shift_top10` 和 `all_coverage` 均满足
   预冻结的 non-inferiority 条件；
2. 对 static 和 shifted control 有正向 paired 支持；
3. 通过自动 temporal jump/flicker safety guard；
4. Coverage exposure 严格少于 `all_coverage`。

完整通过才晋级 fresh-128；仅方向一致则补一个 seed；否则停止该分类路线。决策无需
批量人工 review，脚本最多输出 4 个可选的定点排障视频。

## 6. 服务器执行顺序

先完成 v189。随后 v200 是零 GPU 分析：

```bash
bash scripts/run_v200_head_phase_horizon_audit.sh analyze
bash scripts/run_v200_head_phase_horizon_audit.sh show
bash scripts/run_v200_head_phase_horizon_audit.sh package
```

若 `show` 输出允许进入 runtime design，node 0 准备并做单卡 smoke：

```bash
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh preflight
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0 \
  bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh audit-smoke
```

smoke 审计通过后，在 4 个节点分别执行，`NODE_RANK` 取 `0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0..3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh generate32
```

生成完成后 node 0 审计并准备评测：

```bash
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh status
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh audit-screen
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh prepare
```

4 个节点分别完成 split 和 eval：

```bash
NUM_NODES=4 NODE_RANK=<0..3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v201_vbench_long.sh split
NUM_NODES=4 NODE_RANK=<0..3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v201_vbench_long.sh eval
```

最后由 node 0 自动汇总：

```bash
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh temporal
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh decision
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh package
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh package
```

## 7. 当前边界

当前仓库尚未包含 v189/v200 的服务器结果，所以 v201 代码完成不等于假设成立。
只有 v201 在未参与分类的 holdout 上同时优于 static 与 time-shift control，才有资格把
“Head×Denoising Phase×AR Horizon 条件化缓存路由”写成方法贡献。否则应保留 profiling
结论，停止生成侧包装，转向 operator 或 representation 设计。
