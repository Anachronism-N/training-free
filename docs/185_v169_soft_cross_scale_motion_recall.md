# v169: Soft Cross-scale Motion Recall

## 1. 当前结论与本轮问题

v168 已完成 16-prompt、30 秒生成与 VBench-Long core-9：

| Method | Quality | ID/background | Temporal | Dynamic |
|---|---:|---:|---:|---:|
| v166 MultiScaleMotion | 84.4361 | 0.968516 | 0.971133 | 0.7833 |
| v168 Pareto | 83.9481 | 0.967246 | 0.971060 | 0.7333 |
| v168 Consensus | 83.8487 | 0.967382 | 0.970587 | 0.7208 |
| SF native | 83.0370 | 0.964832 | 0.975111 | 0.6417 |

Pareto 和 Consensus 都在跨尺度证据冲突时偏向最新历史。约束越严格，旧历史
召回越少，Quality 和 Dynamic 越低。因此当前证据否定的是：

> 跨尺度冲突应触发一个二值拒绝，并强制回退到最新候选。

它没有否定局部/长尺度证据本身。v169 回答更窄的问题：

> 能否把跨尺度差异作为连续排序信号，同时保留有价值的旧历史召回？

本轮不修改 cache 组成、写入策略、候选集合、层位置、方向 gate 或读取预算。

## 2. v168 审计修正

v168 原报告中的每个新方法有 86 条 `selection reason mismatch`。根因是：

- 运行时在没有 passing candidate 时记录 `no_passing_candidate`；
- 离线分析器期待 `no_compatible_candidate`。

修正该字符串契约后，两个 v168 方法均满足机制 gate：

- contract failure: 0；
- read-budget violation: 0；
- archive capacity、atomic pair、age 和 selected/read 全部通过。

该修正只改变机制报告，不改变 v168 的视频和 VBench 负结论。v169 在复用 Pareto
视频前会重新运行修正后的 v168 trace 审计，不信任旧的 `mechanism_gate=False`
字段。

## 3. 冻结 cache 契约

Middle10 层的所有 heads：

```text
sink1 + temporal reservoir2 + recalled atomic pair2 + recent4
```

其他层：

```text
sink1 + recent8
```

| Component | Store | Read | Update |
|---|---:|---:|---|
| Sink | 1 frame | 1 | fixed frame 0 |
| Temporal reservoir | 2 frames | 2 | frozen online reservoir |
| Motion archive | 4 adjacent pairs | 1 pair | frozen coherent admission |
| Recent | 4 frames | 4 | FIFO |

最大读取仍为 9 个 full-frame equivalents。motion pair 必须相邻、成对写入和成对
读取。composition 是唯一 dynamic-history owner。以下全部冻结：

- MovieBench-Qwen diverse16 prompt、seed 和 30 秒生成参数；
- Middle10 layer map；
- archive admission、stale refresh horizon 12 和 max read age 24；
- mean-direction floor `0.1`；
- RoPE、CFG 和 attention budget；
- 无候选时读取最新 age-eligible 原子帧对。

本轮没有 PF head taxonomy、stride、cyclic 或 merge。

## 4. 两尺度基础量

使用现有 layer-shared clean-KV descriptor，不增加 encoder 或训练：

```text
q_local   = z_last - z_last-1
q_context = (z_last - z_first) / context_steps
```

历史原子帧对保存对应的 local/context displacement。定义：

```text
L_i = cosine(q_local, m_local_i) * magnitude_match(q_local, m_local_i)
C_i = cosine(q_context, m_context_i) * magnitude_match(q_context, m_context_i)

magnitude_match(a,b) = min(||a||,||b||) / max(||a||,||b||)
```

v166 使用平均方向乘几何平均幅度。v169 保留 v166 的 candidate gate，只改变通过
gate 后的排序。

## 5. Candidate A: Query-weighted Recall

当前 query 在两个尺度上的位移大小决定其连续权重：

```text
w_local   = ||q_local|| / (||q_local|| + ||q_context||)
w_context = ||q_context|| / (||q_local|| + ||q_context||)

S_query(i) = w_local * L_i + w_context * C_i
```

若只有一个可用分量，其权重为 1；若所有可用 query norm 都接近 0，则在可用分量
间均匀分配。排序 tie 依次使用 v166 score 和更新的 pair。

假设：短时运动活跃时，local 证据应获得更高权重；持续趋势更清晰时，context
证据自然增加，而不是用固定权重或冲突阈值。

Method key：

```text
ours_middle10_reservoir2_multiscalequeryweighted1
```

## 6. Candidate B: Bottleneck-balanced Recall

```text
S_bottleneck(i) = min(L_i, C_i)
```

如果只有一个分量可用，就使用该分量。排序 tie 同样使用 v166 score 和更新的 pair。

这不是 v168 的 Pareto guard。它不会把某个分量低于最新候选作为二值拒绝，也不
强制回退最新历史；它在全部候选中选择 weakest-scale evidence 最大的 pair。

Method key：

```text
ours_middle10_reservoir2_multiscalebottleneck1
```

## 7. 生成前离线反事实

在冻结的 v166 16-prompt representative traces 上，共有 640 次 retrieval、518 次
passing decision：

| Method | Changed vs v166 | Change rate | Old recalls | Conflict changes | Median age |
|---|---:|---:|---:|---:|---:|
| Query-weighted | 23 | 4.44% | 177 | 23 | 10 |
| Bottleneck | 57 | 11.00% | 188 | 57 | 10 |

两者的变化都发生在 local/context argmax 冲突处，但没有像 v168 那样把旧召回
压缩到 124 或 85 次。该结果只证明选择器是 bounded non-no-op，不预测视频质量。

离线 gate 要求：

- 每个方法至少改变一次 v166 选择；
- 至少一次变化发生于跨尺度冲突；
- 保留旧历史召回；
- changed/passing 不超过 20%；
- 不扫描阈值、权重或 cache 容量。

## 8. 实验网格

| Method | Source | Purpose |
|---|---|---|
| SF native | reuse v168 | base generator |
| DirectionMatch | reuse v168 | single-scale retrieval reference |
| v166 MultiScaleMotion | reuse v168 | primary reference |
| v168 Pareto | reuse v168 | hard-conflict negative boundary |
| Query-weighted | new 16 | primary v169 candidate |
| Bottleneck | new 16 | balanced-evidence candidate |

逻辑网格为 96 个视频，只新生成 32 个。PF 与 ABA 不占用本轮资源。

## 9. Debug 和机制 gate

每个 candidate trace 新增并保留：

- primitive local/context cosine 和 magnitude match；
- `local_motion_component`、`context_motion_component`；
- `query_weighted_component_weights`；
- `query_weighted_motion_score`；
- `bottleneck_motion_score`；
- v166、Query-weighted、Bottleneck 三个反事实 selected pair；
- actual selected、selection reason、fallback、selected age；
- archive 中真实 pair、actual read ids 和 read-budget flag。

`analyze_v169_soft_cross_scale_trace.py` 独立重算全部数值和选择。机制 gate 要求：

1. 两个方法各有 16 条完整 trace；
2. archive 最大四对，实际读取只能为空或一个连续 pair；
3. selected pair 与实际读取完全一致，age 不超过 24；
4. 每个 logged score、weight、counterfactual pair 均可重算；
5. 两个方法都改变过 v166 选择且保留旧召回；
6. contract failure 和 budget violation 均为 0。

机制 gate 失败时，不运行视频指标。

## 10. 自动评测与最少人工 review

顺序固定为：

1. trace mechanism audit；
2. temporal jump 和 comprehensive automatic screen；
3. prompt-correct VBench-Long core-9；
4. 去除重复 ViCLIP 计权的 corrected paired analysis；
5. 条件式 0 或最多 4 个盲审视频。

相对 v166 MultiScaleMotion，进入 128 prompts 必须同时满足：

- mechanism gate；
- aggregate Official Quality delta `>= 0`；
- identity/background delta `>= 0`；
- temporal mechanics delta `>= 0`；
- dynamic degree delta `>= 0`。

人工 review 不是默认步骤。只有候选同时满足以下冻结的 near-frontier 才准备两条
prompt、最多四个视频：

```text
Quality >= -0.15
ID/background >= -0.001
temporal mechanics >= -0.001
dynamic >= -0.02
```

这些容差只决定是否值得诊断性盲审，不能让方法通过 promotion gate。若两个候选
都明显失败，review manifest 的视频数为 0。

## 11. 论文边界

若 Query-weighted 成立，可以形成的技术叙事是：

1. training-free long-video memory 不应把历史召回视为单一相似度问题；
2. local motion 与 long-context trend 是尺度不同的在线证据；
3. 冲突不是遗忘历史的充分条件，硬回退会降低运动和质量；
4. query-conditioned continuous scale allocation 在固定预算中选择历史状态；
5. 原子 motion pair、reservoir 和 recent context 分别保留事件、覆盖和即时连续性。

若 Bottleneck 成立，贡献应描述为 robust weakest-scale retrieval，而不是
query-conditioned weighting。若两者都失败，v169 只作为负消融，继续保留 v166
为工程最优，不包装为论文主贡献。

本轮 16 prompts 是反复使用的自适应开发集，不能作为 held-out 论文证据。只有
方法冻结后，才运行新的 128-prompt confirmation 和之后的场景切换实验。
