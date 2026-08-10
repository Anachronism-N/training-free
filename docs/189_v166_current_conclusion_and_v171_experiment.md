# v166 当前结论、方法定义与 v171 下一步实验

## 1. 截至 v170 的结论

### 1.1 当前可复现的最好工程基线仍是 v166

在冻结的 16 条 MovieBench-Qwen 开发 prompt、30 秒生成和修正后的
VBench-Long core-9 口径下：

| 方法 | Quality | Identity/background | Temporal mechanics | Dynamic |
|---|---:|---:|---:|---:|
| SF native | 83.1087 | 0.964873 | 0.975109 | 0.6500 |
| DirectionMatch | 84.2283 | 0.966929 | 0.970755 | 0.7750 |
| **v166 MultiScaleMotion** | **84.4697** | **0.968511** | 0.971128 | **0.7875** |
| v167 StateRank | 84.3436 | 0.968273 | 0.970784 | 0.7792 |
| v167 DeficitStateRank | 84.2618 | 0.967772 | 0.971181 | 0.7667 |

因此，v166 是当前方法搜索中的最好起点，但不能称为已经确认的论文方法：

1. 这 16 条 prompt 已被多轮使用，是 adaptive development set；
2. v166 的早期自动代理指标并非全面改善，例如 first-last 和 DINO 曾给出负面
   信号；后续修正 VBench 更偏向 v166；
3. v166 相对 SF 的 temporal mechanics 仍较低；
4. 尚无独立 held-out 128-prompt 结果和重复运行归因。

### 1.2 v170 排除了“Query-weighted 只是代码故障”的解释

v170 在相同 GPU 内成对运行 v166 与 Query-weighted，并用第二条 lane 反转顺序。
全活动层 trace 通过，零 cache ownership、预算、原子帧对或选择器重算错误。

Query-weighted 相对 v166 的 matched effect：

| 指标 | Delta | 同方法 replica noise | 结论 |
|---|---:|---:|---|
| Quality | +0.193533 | 0.061557 | 提升，但 prompt 间不稳定 |
| Dynamic | +0.027083 | 0.006250 | 运动增加 |
| Visual | +0.002083 | 0.000436 | 略升 |
| Identity/background | -0.000754 | 0.000268 | 可靠下降 |
| Temporal mechanics | -0.000544 | 0.000014 | 可靠下降 |
| Semantic | -0.000914 | 0.000299 | 下降 |

两条 lane 的符号一致，顺序效应很小；因此这是实际方法 trade-off，不是旧版本
的 cache bug。Query-weighted 不应作为全程选择器继续推广。

## 2. v166 具体怎么做

### 2.1 Clean-KV 帧描述子

对启用方法的每一层，使用生成时已经存在的 clean K/V：

1. 以固定 token step 对一帧内的 token 采样；
2. 对参与路由的 heads 和采样 token 分别计算 `K mean`、`V mean`、`V std`；
3. 拼接并 L2 normalize，得到帧描述子 `z_t`。

它不需要额外视觉编码器、训练或反向传播。

### 2.2 Local/context 双尺度运动签名

当前生成块：

```text
q_local   = normalize(z_last - z_last-1)
q_context = normalize(z_last - z_first)
```

历史相邻帧对写入时保存：

```text
m_local   = normalize(z_pair_end - z_pair_start)
m_context = normalize(z_source_block_last - z_source_block_first)
```

方向兼容度：

```text
s_local = cosine(q_local, m_local)
s_context = cosine(q_context, m_context)
s_dir = mean(available scales)
```

候选必须满足冻结的 `s_dir >= 0.1`。

### 2.3 幅度兼容度与最终选择

定义无量纲幅度匹配：

```text
rho(a, b) = min(norm(a), norm(b)) / max(norm(a), norm(b))
```

context delta 除以各自时间步数后再比较。v166 使用：

```text
r_mag = sqrt(rho_local * rho_context)
s_motion = s_dir * r_mag
```

候选先按 `s_motion`，再按更新的时间位置排序。没有候选通过方向门时，读取最新的
age-eligible 原子帧对，从而保持固定读取预算。

### 2.4 Cache 组成和读取

只在 Transformer 第 10--19 层的全部 heads 上启用，称为 Middle10 layer
policy；它不是已经验证的语义 head taxonomy。

Middle10 每个 head：

```text
sink1 + temporal reservoir2 + recalled atomic motion pair2 + recent4
```

其余层：

```text
sink1 + recent8
```

Middle10 的最大读取预算为 `1 + 2 + 2 + 4 = 9` 个完整帧等价：

| 组件 | 存储/读取 | 更新规则 |
|---|---|---|
| Sink | 固定 1 帧 | 保留 frame 0，不更新 |
| Reservoir | 存 2、读 2 | 已冻结的在线 temporal reservoir |
| Motion archive | 存 4 对、读 1 对 | 每个 clean block 选一个高 utility 相邻帧对候选 |
| Recent | 读最近 4 帧 | FIFO |

Motion archive 的写入规则：

1. 候选必须是连续相邻帧，且通过 semantic coherence 与正运动检查；
2. 候选 utility 为运动量和语义一致性的组合；
3. 相邻 archive pair 的 end time 至少间隔 4；
4. warmup 后通常要求运动量达到在线 0.7 quantile；
5. bank 未满则填充；bank 满时替换最旧 pair；候选需提升 utility 5%，或最旧 pair
   已达到 12-block stale horizon；
6. stale pair 可以绕过 motion quantile，防止 archive 永久冻结；
7. 读取允许最大 age 为 24 blocks，写入刷新 horizon 与读取 age 不相同；
8. 两帧必须原子写入、原子读取，composition 是唯一 dynamic-history owner。

## 3. v166 中可写的技术点与边界

### 3.1 可以作为技术贡献候选的内容

1. **Training-free clean-KV motion signature**：直接由生成模型内部 K/V 构造局部
   和上下文运动描述，不引入外部 encoder。
2. **Cross-scale motion-compatible episodic recall**：历史不是按固定 stride 或
   单帧语义相似度读取，而是同时匹配短时方向、长时趋势和尺度归一化幅度。
3. **Atomic motion episode memory**：将相邻两帧作为不可拆分的运动证据写入和
   读取，避免单帧 snapshot 丢失方向信息。
4. **Fixed-budget heterogeneous temporal memory**：sink、reservoir、episodic pair
   和 recent 各自承担起始身份、时间覆盖、运动召回和局部连续性。
5. **Auditable online execution**：每次 admission、候选评分、fallback、真实读取、
   RoPE、cache ownership 和预算都可由 trace 独立重算。

### 3.2 目前不能声称的内容

1. `sink + middle + recent` 的一般形式不是新颖点，多个已有工作有相似结构；
2. Middle10 是开发集上冻结的 layer policy，不是新的 head 分类发现；
3. 当前方法没有使用 PF 的 Anchor/Wave/Veil 作为定义，也没有使用其
   stride/cyclic/merge 三路路由；PF 代码只是工程底座；
4. 16-prompt 开发结果不能作为论文最终 benchmark；
5. v166 尚未证明同时优于所有长视频方法，不能写“全面 SOTA”。

## 4. v171 的新假设

v166 有一个结构性问题：它始终把**当前运动幅度**作为检索目标。当长生成已经
出现运动衰减时，低幅度 query 会倾向于召回低幅度历史，可能形成：

```text
motion collapse -> low-motion query -> low-motion recall -> stronger collapse
```

v170 表明，改变历史 pair 排序能够增加运动，但全程改变会伤害身份和时序。因此
v171 不再全程替换选择器，而只在检测到运动不足时干预。

## 5. v171 两个隔离候选

### 5.1 无调参的双尺度 demand gate

每层维护此前 clean block 的：

```text
a_local(t)   = norm(z_last - z_previous)
a_context(t) = norm(z_last - z_first) / block_steps
```

至少 4 次历史更新后，分别计算此前值的在线中位数 `b_local`、`b_context`。仅当：

```text
a_local < b_local and a_context < b_context
```

才触发 motion deficit。当前 block 不参与自身 baseline，未扫描开发集阈值。

### 5.2 Deficit-gated Query control

```text
healthy: v166 MultiScaleMotion
deficit: v170 Query-weighted selector
```

它不是主创新，而是回答：v170 的身份损失是否只是因为干预过于频繁。

### 5.3 Baseline-calibrated Motion Recall，主候选

方向仍匹配当前 query，但 deficit 时的幅度目标改为运动衰减前的在线 baseline：

```text
s_dir(i) = mean(cos(q_local, m_local_i), cos(q_context, m_context_i))
r_base(i) = sqrt(rho(b_local, |m_local_i|)
                 * rho(b_context, |m_context_i|))
s_restore(i) = s_dir(i) * r_base(i)
```

```text
healthy: select argmax v166 s_motion
deficit: select argmax s_restore
```

因此它不是盲目选最大运动历史，而是在保持当前方向兼容的条件下，恢复该层自身的
正常运动尺度。没有学习参数、数据集阈值或额外 cache 读取。

## 6. 离线反事实结果

在 v170 lane-A 的 16 prompts、10 个活动层、head 0 全 trace 上：

| 统计 | 数值 |
|---|---:|
| 总读取决策 | 6400 |
| gate ready | 5600 |
| deficit trigger | 1890 |
| 全程 Query 相对 v166 改变 | 258 |
| Deficit-gated Query 改变 | 86 |
| Baseline-calibrated 改变 | 180 |
| 两种方法在 healthy 状态的改变 | 0 |

Baseline-calibrated 改变选择时，相对 v166 所选 pair：

- local candidate magnitude 平均增加 `0.007984`；
- context per-step magnitude 平均增加 `0.005531`。

这证明分支会真实执行，并且主候选确实选择更接近正常运动尺度的历史。它不能预测
AR 视频结果，所以仍需重新生成 32 个视频。

## 7. 下一步决策

v171 固定三种逻辑方法：

1. v166 reference：复用 v170 lane A 的 16 个视频；
2. Deficit-gated Query：新生成 16 个；
3. Baseline-calibrated Recall：新生成 16 个。

先做完整 mechanism audit，再计算 prompt-correct VBench-Long core-9。默认不要求
人工 review。只有候选满足以下自动条件，才进入 order-balanced matched run：

1. 全层机制 gate 通过；
2. Quality 不低于 v166；
3. Dynamic Degree 高于 v166；
4. identity/background 和 temporal 的下降不超过 v170 实测同方法 replica noise。

该 noise-aware gate 只用于选择下一轮归因实验，不是论文 non-inferiority margin。
若主候选通过 matched confirmation，再冻结方法并运行未参与开发的 128-prompt
held-out confirmation；否则拒绝当前 homeostatic selector，继续修改 deficit signal
或 archive admission，而不是扫描阈值挽救结果。

## 8. 论文故事的当前工作版本

暂定核心叙事：

> 长 AR 视频中的历史检索不仅有“记住什么”的问题，还有由当前退化状态驱动的
> recall feedback：运动已经衰减时，当前状态匹配会继续召回低运动历史。我们提出
> training-free motion-homeostatic episodic memory，从 DiT clean-KV 构建跨尺度运动
> 签名，以原子运动片段维护固定预算历史，并仅在在线检测到双尺度运动不足时，将
> 检索幅度目标从退化的当前状态切换到该层自身的历史正常尺度。

如果 v171 不改善身份、运动和总体质量，这个故事只保留为负假设，不能先写成既定
贡献。当前最稳妥的创新中心是“demand-gated baseline-calibrated episodic recall”，
而不是笼统声称提出了新的多尺度 cache 或 head 分类。
