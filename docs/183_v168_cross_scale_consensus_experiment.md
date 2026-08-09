# v168: Threshold-free Cross-scale Consensus Recall

## 1. 当前结论

v168 不再修改已被 v166 验证的 cache 组成、写入规则或层位置，只回答一个更窄且可审计的问题：

> 当历史候选在短时运动和长时趋势上给出冲突证据时，是否应该让它覆盖最新的原子帧对？

当前冻结两个训练自由候选：

1. **Pareto Motion Recall**：先按 v166 的多尺度运动分数找历史候选；只有该候选在 local 和 context 两个分量上都不差于最新兼容帧对时，才允许召回。
2. **Strict Scale Consensus**：local 和 context 分量分别独立排序；只有两个 argmax 指向同一帧对时才允许召回，否则使用最新兼容帧对。

Pareto 是主候选，Strict Consensus 是更保守的机制边界。两者均不引入学习参数、聚类模型或为视频结果调节的阈值。

## 2. 为什么推进 v168

### 2.1 已确认的有效基线

v167 corrected 16-prompt 开发集结果中：

| 方法 | Official VBench Quality | Dynamic | Identity/background | Temporal mechanics |
|---|---:|---:|---:|---:|
| SF native | 83.1087 | 0.6500 | 0.964873 | 0.975109 |
| DirectionMatch | 84.2283 | 0.7750 | 0.966929 | 0.970755 |
| **v166 MultiScaleMotion** | **84.4697** | **0.7875** | **0.968511** | 0.971128 |
| v167 StateRank | 84.3436 | 0.7792 | 0.968273 | 0.970784 |

因此 v168 以 v166 MultiScaleMotion 为当前方法基线，而不是继续扩展未带来收益的 StateRank/Deficit 逻辑。

### 2.2 v166 的剩余问题

v166 把两种证据压成一个标量：

```text
direction = mean(local_cosine, context_cosine)
magnitude = sqrt(local_magnitude_match * context_magnitude_match)
score = direction * magnitude
```

标量较高不代表两个尺度都支持召回。例如旧候选可能只在 local 上占优，却在 context 上明显弱于最新候选。该候选仍可能凭平均分成为 winner，并把局部运动方向正确但长时趋势错误的历史注入当前生成。

这提供了一个不依赖手工年龄阈值的新假设：

> 长历史只有在跨尺度证据自洽时才应覆盖最新状态；证据冲突时，最新状态是更安全的默认值。

## 3. 冻结 cache 契约

v168 不重新搜索 cache 预算。所有候选与 v166 完全一致。

Middle10 层：

```text
sink1 + reservoir2 + recalled atomic pair2 + recent4
```

- sink：固定 frame 0，一帧。
- reservoir：两帧容量，沿用既有 temporal reservoir 更新。
- motion archive：最多四个相邻原子帧对，只读取一个帧对。
- recent：最近四帧。
- 最大读取量：九个完整帧等价。
- 最大历史读取年龄：24 个 frame blocks。
- 平均方向相似度下限：0.1，继承 v166，不在 v168 调参。

其他层：

```text
sink1 + recent8
```

以下因素保持不变：prompt、seed、30 秒长度、分辨率、Middle10 层集合、archive admission、容量、读取年龄、fallback、RoPE、CFG、attention budget 和 exclusive dynamic owner。

本轮没有 merge、stride、cyclic 或 PF 三类 head 路由。因此视频差异只能来自“哪一个 archive 帧对被读出”。

## 4. 两个选择器

设候选帧对为 `m`，当前 query 的局部和上下文运动描述分别为 `q_l`、`q_c`。

### 4.1 分量

```text
L(m) = cosine(q_l, m_l) * magnitude_match(q_l, m_l)
C(m) = cosine(q_c, m_c) * magnitude_match(q_c, m_c)
```

其中：

```text
magnitude_match(a, b) = min(norm(a), norm(b)) / max(norm(a), norm(b))
```

`context` displacement 按时间步数归一化，避免长区间天然具有更大幅值。

### 4.2 Pareto Motion Recall

1. 在通过 v166 方向门的候选中，按原 v166 `score` 找到 `m*`。
2. 找到最新兼容帧对 `m_new`。
3. 若 `m* == m_new`，直接使用最新帧对。
4. 若 `L(m*) >= L(m_new)` 且 `C(m*) >= C(m_new)`，召回 `m*`。
5. 否则使用 `m_new`。

实现中只有 `1e-12` 浮点比较容差，不是可调方法阈值。

该规则保留 v166 的联合 motion winner，同时禁止“牺牲一个尺度换取另一个尺度高分”的旧历史覆盖最新状态。

### 4.3 Strict Scale Consensus

1. 分别计算 `argmax L(m)` 与 `argmax C(m)`，分数相同时选择更新的帧对。
2. 若两个 argmax 相同，读取该帧对。
3. 若两个 argmax 冲突或任一尺度不可用，读取最新兼容帧对。

该版本不使用 v166 的联合标量决定旧历史是否进入 attention，是更严格的机制对照。

## 5. 冻结 trace 的离线反事实

在 v166 的 16-prompt、640 次代表性读取上，代码已做确定性反事实回放：

| 规则 | 旧历史读取 | 相对 v166 改变 | Pareto 接受旧历史 | Pareto 拒绝 | 双尺度一致 | 双尺度冲突 |
|---|---:|---:|---:|---:|---:|---:|
| Pareto | 101 | 73 | 101 | 73 | 329 | 189 |
| Strict Consensus | 70 | 104 | 101 | 73 | 329 | 189 |

另有 37 次读取没有通过兼容门，按冻结规则 fallback 到最新 age-eligible 原子帧对。

该结果只说明所有关键分支都能在真实 trace 中被触发，不能预测视频质量。

## 6. 实验网格

固定六个方法：

1. `sf_native`
2. `ours_middle10_reservoir2_directionmatch1`
3. `ours_middle10_reservoir2_multiscalemotion1`
4. `ours_middle10_reservoir2_staterankmotion1`
5. `ours_middle10_reservoir2_multiscalepareto1`
6. `ours_middle10_reservoir2_multiscaleconsensus1`

前四个方法严格验证 v167 manifest、prompt hash、runtime、trace gate 和视频字节后复用。只生成后两个方法，共 32 个新视频。

PF 不属于本轮必要对比，ABA 暂不运行。它们不会占用本轮生成资源。

自动评测顺序：

1. 新 trace 逐候选重算；
2. temporal jump、low-motion 和 outlier 自动筛查；
3. comprehensive diagnostics；
4. prompt-correct VBench-Long core-9；
5. 去除重复 ViCLIP 权重的 corrected paired analysis；
6. 最多两个自动选中 prompt 的盲审。

Official Quality、identity/background、temporal mechanics 和 dynamic degree 必须分开报告。

## 7. 扩大到 128 prompt 的条件

一个候选只有同时满足以下开发门，才进入 128-prompt confirmation：

- 对应 mechanism gate 通过；
- 相对 v166 MultiScaleMotion，aggregate Official Quality 不下降；
- identity/background 不下降；
- temporal mechanics 不下降；
- dynamic degree 不下降。

该规则没有人为容忍区间。16 prompt 是反复使用的自适应开发集，不能作为论文 held-out 结果。

## 8. Debug 与问题定位

每个候选 trace 新增：

- `local_motion_component`
- `context_motion_component`
- `local_component_rank`
- `context_component_rank`

每次读取新增：

- `motion_signature_selected`
- `pareto_candidate`
- `pareto_pass`
- `pareto_component_delta.local/context`
- `local_component_best`
- `context_component_best`
- `scale_argmax_agreement`
- `cross_scale_conflict`
- `selected_component_scores`
- `newest_component_scores`
- `selection_changed_from_motion_signature`
- `selection_reason`

`analyze_v168_cross_scale_consensus_trace.py` 会独立重算所有分量、排名、反事实和最终读取，并检查：archive 不超过四个原子帧对；候选确实来自 archive；读取始终为连续两帧；age 不超过 24；fallback 不丢失预算；StateRank/Deficit 没有泄漏；四种新分支都实际执行。

## 9. 多边形噪声停止条件

本轮 smoke 只生成 prompt 14 的两个新方法。若任一视频出现明显多边形噪声：

1. 不启动 full16；
2. 先运行 mechanism audit；
3. 对照 `selection_reason`、读取帧对和 `read_budget_preserved`；
4. 检查同一任务的 config、stdout/stderr、policy trace 和视频 audit；
5. 确认 source checkpoint/config hash 与 v167 一致。

v168 没有 merge 和异构 PF cache 分支。如果出现结构噪声，优先怀疑读取帧对或 runtime/config 契约，而不是再调整分类阈值。

## 10. 可写论文的故事边界

只有 128-prompt confirmation 与独立 held-out 实验成立后，才能考虑如下故事：

> 长视频生成中的历史 recall 不是单一相似度检索问题。局部运动连续性和长时运动趋势可能对同一历史给出矛盾证据。我们提出训练自由的跨尺度一致性 recall：旧历史只有在多尺度证据共同支持时才能覆盖最新状态，从而在保留身份/背景记忆的同时减少错误运动注入。

潜在技术点：

1. 从 clean-KV 在线构造 local/context motion signature；
2. 显式识别跨尺度 retrieval conflict；
3. 无训练、无调阈值的 Pareto 或 consensus recall；
4. 在固定预算的原子帧对 memory 中执行，并提供逐决策可验证 trace。

与 PF 的区别必须明确：

- 不使用 PF 的 Wave/Anchor/Veil 分类作为方法定义；
- 不使用其 stride/cyclic/merge 三路 cache；
- 当前 Middle10 只是冻结的开发位置，不是声称发现了新 head taxonomy；
- 方法贡献是跨尺度历史检索条件，而不是重新命名 PF 路由。

与 v166 的区别也必须明确：v166 用单一平均分直接召回，v168 对“旧历史覆盖最新状态”增加跨尺度证据约束。

如果 16/128 prompt 结果不能同时改善身份、运动和总体质量，则 v168 只能作为负结果或诊断工具，不能包装成已验证论文贡献。
