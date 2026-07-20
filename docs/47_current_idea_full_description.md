# 当前方法详细描述与差异化分析

> 日期：2026-07-20
> 目的：明确当前idea的完整技术描述、与PF的本质区别、以及科研故事可行性分析

---

## 1. Pyramid-Forcing (PF) 的核心创新

PF的论文标题是"Head-Aware Pyramid KV Cache Policy"。其核心贡献是：

### 1.1 离线三模式头分类 (Offline Tri-Pattern Head Classification)

PF将30层×12头=360个attention head分为三类：

- **Anchor heads (stable, label=1)**: 时间上稳定的head，负责身份、布局等不变信息。使用**Adaptive Strided Sliding Window**策略——stride采样远期历史帧，保留关键锚点。
- **Wave heads (oscillating, label=-1)**: 时间上周期性变化的head，负责运动、动态信息。使用**Periodic Sampling**策略——周期性采样中间帧。
- **Veil heads (stable-sparse, label=2)**: 稀疏稳定head。使用**Cache Merging**策略——合并相邻帧。

分类方法：对pre-softmax logits计算sign-rate统计量 + FFT频域周期性检测。在32个prompt×15秒的calibration set上运行一次，得到`best_labels.csv`（30×12矩阵），之后所有推理复用该标签。

### 1.2 PF的核心限制

1. **标签是静态的**：同一个`(layer, head)`在所有prompt、所有时间步、所有视频上都使用相同的cache策略。
2. **标签是离线计算的**：需要额外的calibration set和分类流程。
3. **分类基于时间模式**：sign-rate/FFT描述的是head的时间访问模式，不是head的语义功能（identity/motion/layout）。
4. **无远期memory**：PF只管理近程cache的保留策略（sink+middle+recent），不构建独立的远期memory archive。
5. **固定CFG**：所有head、所有时间步使用相同的guidance_scale。

### 1.3 PF不做什么

- 不做retrieval：PF不根据当前query检索历史帧，只按固定策略保留/采样。
- 不做memory archive：PF不构建独立的远期frame bank。
- 不做dynamic routing：PF的head路由在推理前就固定了。
- 不做CFG modulation：PF不根据memory质量调整guidance。
- 不做parallel attention：PF的memory和recent在同一个softmax中竞争。

---

## 2. 我们当前实现的实际内容

### 2.1 已实现的组件

当前代码在PF baseline之上添加了以下模块：

#### A. Clean-frame Archive (远期视觉记忆)

```
每个block的clean pass完成
  → clean K/V (完整空间网格, 不压缩) 写入archive
  → archive上限64帧，超限时保留frame 0 (identity anchor) + 末帧 + 中间均匀采样
```

这是PF没有的——PF只有近程cache的保留策略，没有独立的远期frame bank。

#### B. Query-Conditioned Frame Retrieval (查询条件检索)

```
每个attention forward (layers 15-20):
  1. 用当前query与archive各帧的mean K计算相似度
  2. 排除最近4帧，选top-3帧，softmax加权 (temperature=0.3)
  3. token-level attention → x_memory
  4. confidence = max相似度 > 0.25才贡献
```

这是PF没有的——PF不根据query检索历史，只按固定策略保留。

#### C. Parallel Attention with Convex Fusion (并行注意力融合)

```
x_recent = native_PF_attention(q, recent_k, recent_v)  # PF原生attention
x_memory = memory_attention(q, archive_k, archive_v)   # 独立softmax
x = x_recent * (1-w) + x_memory * w
  w = gate × confidence × alignment × head_mask
```

这是PF没有的——PF的memory和recent在同一个softmax中竞争概率。

#### D. Confidence-Adaptive Head Routing (置信度自适应头路由)

```python
# 当前实现：两种模式
if routing_mode == "static":
    # 使用PF的离线标签 {1,2} 做head mask
    head_mask = [label in {1,2} for label in pf_labels]
elif routing_mode == "confidence_adaptive":
    # 用retrieval confidence做per-head sigmoid mask
    soft_mask = sigmoid(sharpness * (conf - threshold))
```

**问题**：`static`模式直接复用PF标签，没有差异化。`confidence_adaptive`模式是创新，但需要验证有效性。

#### E. Dynamic CFG (动态分类器自由引导)

```python
# 全局动态CFG
effective_scale = max_scale + conf * (min_scale - max_scale)
flow_pred = uncond + effective_scale * (cond - uncond)

# Per-head动态CFG
cfg_per_head = max_scale + conf_per_head * (min_scale - max_scale)
flow_pred = uncond + cfg_per_head * (cond - uncond)
```

这是PF没有的——PF使用固定guidance_scale=3.0。

### 2.2 当前的核心问题

**如果去掉组件D的`confidence_adaptive`模式和组件E，当前方法就是"PF + 远期frame archive + parallel attention"。这个组合本身有工程价值，但head routing部分完全依赖PF的标签，科研故事不够独立。**

---

## 3. 与PF的逐项对比

| 维度 | PF | 我们当前 | 是否差异化 |
|------|-----|---------|-----------|
| **Head分类方法** | 离线FFT/sign-rate → Anchor/Wave/Veil | 复用PF标签(static) 或 retrieval confidence(adaptive) | static模式: **无差异**; adaptive模式: **有差异** |
| **Head路由时机** | 离线一次性固定 | static: 离线; adaptive: 在线per-query | adaptive模式: **有差异** |
| **Cache策略** | Per-head [sink+middle+recent] | PF原生cache + 远期archive | **有差异**（新增archive） |
| **历史访问方式** | 固定保留/采样，不检索 | Query-conditioned retrieval | **有差异** |
| **Attention结构** | 单一softmax | Parallel attention (独立softmax) | **有差异** |
| **CFG** | 固定3.0 | 动态/per-head (基于confidence) | **有差异** |
| **远期memory** | 无 | Clean-frame archive (64帧) | **有差异** |
| **训练需求** | 无 | 无 | 相同 |

---

## 4. "动态分类"的含义解释

你问的"动态分类是什么意思"——当前代码中有两种理解：

### 4.1 当前`confidence_adaptive`模式的实际含义

```python
# 每个head的retrieval confidence ∈ [0, 1]
# confidence来自query与archive帧的相似度
conf = max_similarity_per_head  # [H] tensor

# Sigmoid soft mask
head_mask = sigmoid(sharpness * (conf - threshold))
# conf > threshold → mask ≈ 1 (允许读memory)
# conf < threshold → mask ≈ 0 (禁止读memory)
```

**这不是"分类"**，而是**连续路由**。它不把head分成离散类别，而是根据当前query与历史的匹配程度，连续地控制每个head的memory访问强度。

### 4.2 与PF"分类"的本质区别

| | PF | 我们 |
|---|---|---|
| **输入** | 预softmax logits的时间序列 | 当前query与archive的相似度 |
| **方法** | FFT + sign-rate | Cosine similarity + sigmoid |
| **输出** | 离散标签 {-1, 1, 2} | 连续值 [0, 1] |
| **时机** | 离线一次性 | 在线每个forward |
| **是否随query变化** | 否 | 是 |
| **是否随时间步变化** | 否 | 是 |
| **语义** | 时间访问模式 | 内容相关度 |

### 4.3 问题：这够不够作为论文创新？

**不够**。原因：

1. "用confidence做sigmoid mask"本身太简单，不足以作为核心贡献。
2. PF的标签描述的是**head的时间行为模式**（这个head是周期性访问历史还是稳定访问），我们的confidence描述的是**当前query与历史的匹配度**——这是两个不同的维度，不是简单的"动态版PF"。
3. 如果只是把PF的静态标签换成动态confidence，reviewer会认为这是incremental改进。

---

## 5. 真正的差异化方向

要让这个工作成为独立于PF的论文贡献，需要一个**PF完全不做、且无法简单扩展去做**的核心机制。以下是三个候选方向：

### 方向A：Memory-Guided Dynamic Guidance (MGDG) — 最推荐

**核心洞察**：PF（以及所有现有AR视频生成方法）的CFG是固定的。但不同时间步、不同空间区域、不同head对历史memory的需求是不同的。我们用memory retrieval的置信度来**动态调节guidance**，这改变了生成过程本身，而不仅仅是cache管理。

**具体机制**：

```
1. Per-head CFG: 每个head根据其memory confidence获得不同的guidance scale
   - 高confidence head: memory找到了相关历史 → 降低CFG，让memory引导
   - 低confidence head: 无相关历史 → 保持高CFG，prompt引导

2. Per-timestep CFG: 早期denoising step（高噪声）用高CFG确定结构，
   后期step（低噪声）根据memory confidence降低CFG让memory精修细节

3. Per-region CFG: 不同空间区域（前景vs背景）的memory confidence不同，
   前景人物区域可能高confidence（保持identity），背景可能低confidence（允许演化）
```

**为什么PF无法简单扩展到这个方向**：
- PF的标签是离线固定的，无法根据当前query动态调整guidance
- PF不做retrieval，没有confidence信号
- PF的cache管理和guidance是解耦的，我们让它们耦合——memory质量直接影响guidance

**论文故事**："不是所有历史记忆都同样可靠，也不是所有head都需要同样强度的prompt引导。我们用memory retrieval confidence作为在线信号，动态调节per-head guidance，让可靠的历史记忆替代部分prompt引导， unreliable的memory区域保持prompt驱动。"

### 方向B：Decoupled Memory Readout with Confidence Gating

**核心洞察**：PF把历史和近期放在同一个softmax中竞争。我们用独立softmax的parallel attention，并用confidence控制memory的注入强度。

**具体机制**：

```
x = x_recent * (1-w) + x_memory * w
w = gate × confidence × alignment × head_mask

其中：
- confidence: retrieval的置信度（这个历史帧是否相关）
- alignment: memory output与recent output的对齐度（memory方向是否一致）
- head_mask: per-head的memory访问权限
```

**问题**：这个方向已经实现了，但单独不够作为核心贡献——它更像是工程优化。

### 方向C：Causal Functional Head Profiling (因果功能头分析)

**核心洞察**：PF用FFT/sign-rate分类head，但这描述的是时间模式，不是语义功能。我们通过**causal intervention**（drop/swap/shuffle）在线测量每个head的真实功能角色。

**具体机制**：

```
对每个head做在线因果干预：
1. Drop test: 移除该head的memory → 测量identity变化 → identity reliance
2. Shuffle test: 打乱该head的memory时间顺序 → 测量motion变化 → motion reliance  
3. Swap test: 替换为错误场景memory → 测量scene leakage → scene specificity

输出: 每个(layer, head)的连续角色向量 [identity, motion, scene, stability]
```

**问题**：在线因果干预的计算开销大，可能不实用。

---

## 6. 推荐的论文定位

### 6.1 推荐核心贡献

**Memory-Guided Dynamic Guidance for Long-Horizon Video Generation**

1. **Training-free clean-frame archive**: 从clean pass构建远期visual memory（PF不做）
2. **Query-conditioned retrieval**: 按当前query检索相关历史帧（PF不做retrieval）
3. **Per-head dynamic CFG**: 用retrieval confidence动态调节每个head的guidance scale（PF用固定CFG）
4. **Parallel attention fusion**: 独立softmax的memory branch，避免与recent竞争（PF用单一softmax）

### 6.2 与PF的关系

- **PF是baseline**，不是竞争对手。我们在PF之上构建，PF负责近程cache管理，我们负责远期memory + dynamic guidance。
- **PF的head标签是可选的**，不是我们方法的必要组件。我们的confidence-adaptive routing可以完全替代PF标签。
- **核心差异在guidance层面**，不在cache管理层面。PF改的是"保留什么历史"，我们改的是"如何使用历史来引导生成"。

### 6.3 消融矩阵

| 变体 | Archive | Retrieval | Parallel Attn | Per-head CFG | Head路由 | 验证 |
|------|---------|-----------|---------------|-------------|---------|------|
| PF baseline | ✗ | ✗ | ✗ | ✗ | PF static | 基线 |
| +Archive+Retrieval | ✓ | ✓ | ✗ (union) | ✗ | PF static | Archive+Retrieval的贡献 |
| +Parallel Attn | ✓ | ✓ | ✓ | ✗ | PF static | Parallel vs union |
| +Confidence Routing | ✓ | ✓ | ✓ | ✗ | Confidence | Adaptive vs static routing |
| +Per-head CFG | ✓ | ✓ | ✓ | ✓ | Confidence | Per-head CFG的贡献 |
| Full method | ✓ | ✓ | ✓ | ✓ | Confidence | 完整方法 |

### 6.4 关键实验

1. **PF baseline vs Full method**: 整体效果对比
2. **Static routing vs Confidence routing**: 验证动态路由 > PF静态标签
3. **Fixed CFG vs Dynamic CFG vs Per-head CFG**: 验证CFG调制的效果
4. **Union attention vs Parallel attention**: 验证独立softmax的价值
5. **Wrong memory vs Correct memory**: 内容特异性验证
6. **60s/120s/240s**: 长时收益验证

---

## 7. 当前代码状态与差距

### 已实现
- ✅ Clean-frame archive (64帧, preservation-aware)
- ✅ Query-conditioned retrieval (top-3, temp=0.3)
- ✅ Parallel attention with convex fusion
- ✅ Confidence-adaptive head routing (sigmoid mask)
- ✅ Global dynamic CFG
- ✅ Per-head dynamic CFG
- ✅ Warmup ramp (6 blocks)
- ✅ CFG preservation (neg cache禁用memory)

### 待验证
- ❓ Confidence routing是否优于PF static routing
- ❓ Per-head CFG是否优于fixed CFG
- ❓ 整体方法是否优于PF baseline
- ❓ 各组件的独立贡献

### 待完成
- [ ] 运行完整消融矩阵 (A0-A4)
- [ ] 运行wrong memory control
- [ ] 移植到SF native和Causal-Forcing
- [ ] VBench-Long定量评估
- [ ] 人工review

---

## 8. 总结

当前方法与PF的核心区别不在于head分类（那是PF的贡献），而在于：

1. **PF管理"保留什么历史"，我们管理"如何使用历史来引导生成"**
2. **PF的head标签是离线静态的，我们的routing是在线动态的（基于retrieval confidence）**
3. **PF用固定CFG，我们用per-head dynamic CFG（memory质量影响guidance）**
4. **PF不做retrieval，我们做query-conditioned frame retrieval**

论文的核心故事应该是：**"不是所有历史记忆都同样可靠，也不是所有head都需要同样强度的prompt引导。我们用memory retrieval confidence作为在线信号，动态调节per-head guidance，让可靠的历史记忆替代部分prompt引导。"**

这个story是PF无法覆盖的——PF不retrieval、不modulate guidance、不做远期memory。
