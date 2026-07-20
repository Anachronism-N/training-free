# QACP方案可行性分析：真的能分出类吗？真的更好吗？

> 日期：2026-07-20
> 目的：严肃质疑QACP方案的可行性，分析其失败风险

---

## 1. 核心质疑

### 质疑1：在线分类真的能分出有意义的类别吗？

QACP提出用两个信号（temporal sensitivity + content specificity）将head分为4类。但：

**问题A: temporal sensitivity可能对所有head都很高**

AR视频生成的滑动窗口机制下，所有head都会受到"远期历史是否可用"的影响。如果所有head的temporal sensitivity都很高，这个信号就没有区分力。

**问题B: content specificity可能对所有head都很低**

如果query与任何历史帧的相似度都不高（高运动视频、场景切换频繁），那么conf_correct ≈ conf_random，content specificity趋近于0，无法区分Identity head和Motion head。

**问题C: 分类可能不稳定**

如果分类结果在每个时间步都剧烈变化（head 1在block 5是Identity，block 6变成Motion），cache策略的频繁切换可能导致不连续。

### 质疑2：动态分类真的比PF静态分类好吗？

PF的标签虽然离线固定，但它基于大量calibration数据（32 prompts×15s）的统计，可能比单次推理的在线估计更鲁棒。动态分类基于当前query的瞬时信号，可能：
- 对噪声敏感
- 对当前prompt的特异性过拟合
- 缺乏统计稳定性

### 质疑3：4类head的cache策略真的有区别吗？

Identity head做retrieval，Motion head做周期采样——但如果retrieval和周期采样的结果类似（都选了相似的历史帧），区分它们就没有意义。

---

## 2. 数据分析

### 2.1 PF标签的实际分布

```
Total heads: 360
  oscillating(-1): 156 (43.3%)
  stable(1):       172 (47.8%)
  sparse(2):        32 (8.9%)

相邻层标签持久性: 46.3%
```

**观察**: PF标签的层间一致性只有46.3%，意味着同一个head编号在不同层频繁换类。这说明PF的标签本身就是不稳定的——但这不一定意味着动态分类更好，反而可能说明head的"功能角色"本来就难以稳定定义。

### 2.2 Layers 15-20的标签分布

```
L15: 1 osc, 10 stable, 1 sparse  → 几乎全是stable
L16: 4 osc, 7 stable, 1 sparse
L17: 8 osc, 3 stable, 1 sparse  → 大部分oscillating
L18: 2 osc, 10 stable           → 几乎全是stable
L19: 4 osc, 6 stable, 2 sparse
L20: 8 osc, 4 stable            → 大部分oscillating
```

**观察**: 在memory-enabled layers (15-20)中，标签分布波动很大。L15几乎全是stable，L17/L20大部分是oscillating。这意味着PF在这些层的cache策略也很不同——但PF的标签是固定的，不会随query变化。

### 2.3 AMA之前的发现

docs/37记录了AMA的重要发现：
- SF/CF的`|QK|` proxy把360/360个head全判成identity → 失败
- PF的`-1/1/2`标签描述的是**时间访问模式**，不是**语义功能**
- LifeCache v3.2错误地把PF oscillating heads当成motion heads → 分类语义错误

**关键教训**: head分类的proxy选择至关重要，错误的proxy会导致系统性错误。

---

## 3. QACP方案的风险评估

### 3.1 高风险点

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| temporal sensitivity无区分力 | 中 | 4类退化为2类 | 需要实验验证 |
| content specificity全为0 | 中-高 | 无法区分Identity和Motion | 需要实验验证 |
| 分类不稳定导致cache跳变 | 中 | 视频不连续 | 加入EMA平滑 |
| 动态分类不如PF静态 | 中-高 | 方法无优势 | 需要消融证明 |
| 4类cache策略效果无差异 | 中 | 分类无意义 | 需要消融证明 |

### 3.2 最大的风险

**最大的风险不是"分不出类"，而是"分出了类但没有用"。**

即使temporal sensitivity和content specificity能区分出4类head，如果Identity head做retrieval vs Layout head做stride的结果类似（因为历史帧本身就很相似），那么分类的价值就无法体现。

### 3.3 PF可能已经"足够好"

PF的3类分类虽然粗糙，但它覆盖了主要的cache策略空间：
- stride（保留远期）
- 周期（保留中间）
- 合并（压缩近期）

QACP的4类分类可能在实践中与PF的3类高度重叠：
- Identity head ≈ Anchor head (stride + retrieval)
- Motion head ≈ Wave head (周期)
- Layout head ≈ Anchor head (stride)
- Recent head ≈ Veil head (合并/仅recent)

如果重叠太大，QACP相对于PF的唯一优势就是"Identity head做retrieval而不是stride"——但这只是archive retrieval的一个子功能，不是分类方法的创新。

---

## 4. 更诚实的方案评估

### 4.1 QACP的真正价值可能不在"分类"

如果动态分类本身不可靠或没有显著优势，QACP的真正价值可能在于：

**不是"动态分类比静态分类好"，而是"retrieval比固定采样好"。**

即：不管理论上head应该怎么分类，只要我们让部分head从"固定pattern采样"变成"query-conditioned retrieval"，就是相对于PF的改进。

### 4.2 可能更务实的方向

与其追求"动态head分类"（可能分不出有意义的类），不如：

**方向1: 全head retrieval + confidence gating**

不做head分类，所有head都获得archive retrieval，但通过confidence gate控制：
- 高confidence → memory贡献大
- 低confidence → memory贡献小
- 这就是当前的`confidence_adaptive`模式

**优势**: 不需要分类，不会分错，confidence本身是连续的。

**方向2: 分层retrieval（按layer而非按head）**

不同层有不同的功能（浅层=纹理，中层=结构，深层=identity），对每层用不同的retrieval策略：
- 深层(25-29): identity retrieval（精确匹配历史identity帧）
- 中层(15-20): scene retrieval（匹配历史场景帧）
- 浅层(0-9): 不做retrieval（纹理不需要远期历史）

**优势**: 层的功能比head功能更稳定、更容易定义。

**方向3: Retrieval + Dynamic CFG（不做head分类）**

保持PF的cache策略不变，只在PF之上添加：
1. Archive retrieval（所有head共享，confidence gating）
2. Per-head dynamic CFG（基于retrieval confidence）
3. 不改PF的head标签和cache策略

**优势**: 不与PF竞争"head分类"，而是在PF之上添加"retrieval + guidance modulation"层。

---

## 5. 建议的实验验证

在实现完整QACP之前，先做以下**快速验证实验**来判断方向：

### 实验1: Temporal Sensitivity是否有区分力？

在单个prompt的推理过程中，测量每个head的temporal sensitivity：
```python
# 对layers 15-20的每个head
# 比较 attention(q, all_cache) vs attention(q, recent_only)
# 计算 ||x_full - x_recent|| / ||x_recent||
```

如果所有head的temporal sensitivity都接近→分类无意义→放弃QACP。
如果有明显双峰分布→有区分力→可以继续。

### 实验2: Content Specificity是否有区分力？

```python
# 对layers 15-20的每个head
# 比较 conf(q, correct_archive) vs conf(q, random_archive)
# 计算 conf_correct - conf_random
```

如果content specificity全为0→无法区分Identity和Motion→简化为2类。
如果有明显差异→有区分力→可以继续。

### 实验3: Retrieval vs Stride是否真的不同？

```python
# 对同一个head，比较：
# A: 用query retrieval选1帧
# B: 用stride采样选1帧
# 计算两次attention output的差异
```

如果差异很小→retrieval vs stride无区别→分类无意义。
如果差异大→retrieval有价值→值得继续。

---

## 6. 修正后的推荐方案

基于以上分析，**修正推荐方案**：

### 6.1 不做"动态head分类"作为核心贡献

原因：
- head分类是PF的核心贡献，我们在同一维度竞争很难赢
- 在线分类的可靠性存疑
- 4类vs3类可能没有实际差异

### 6.2 核心贡献改为"Retrieval-Augmented Cache with Dynamic Guidance"

1. **Archive retrieval**: 在PF cache之上添加query-conditioned frame retrieval（PF不retrieval）
2. **Confidence gating**: 用retrieval confidence控制memory注入强度（不分类head，所有head共享memory，但confidence不同）
3. **Per-head dynamic CFG**: 用retrieval confidence调节per-head guidance（PF用固定CFG）
4. **Parallel attention**: 独立softmax避免竞争（PF用单一softmax）

### 6.3 PF的关系

- PF的head分类和cache策略**保持不变**，作为baseline
- 我们在PF之上添加retrieval层和guidance调制层
- 不与PF竞争"谁分类分得好"，而是在PF不做的问题上做贡献

### 6.4 论文故事

"PF解决了'不同head保留什么历史'的问题（静态cache策略）。我们解决'如何使用这些历史来引导生成'的问题（动态retrieval + guidance modulation）。两个问题是正交的，可以叠加。"

---

## 7. 结论

### QACP方案的可行性判断

| 问题 | 判断 | 理由 |
|------|------|------|
| 能分出类吗？ | **不确定** | 需要实验验证temporal sensitivity和content specificity的区分力 |
| 分类后cache规划更好吗？ | **不确定** | 需要验证retrieval vs stride的实际差异 |
| 比PF静态分类好吗？ | **可能不** | PF的离线统计可能更鲁棒 |
| 值得作为核心贡献吗？ | **风险高** | 在PF的核心维度竞争，需要压倒性证据 |

### 修正建议

1. **不把"动态head分类"作为核心贡献**——风险太高
2. **把"retrieval + dynamic CFG"作为核心贡献**——PF不做retrieval，不做CFG modulation
3. **先做快速验证实验**（实验1-3）再决定是否实现QACP
4. **如果验证通过**，QACP可以作为额外贡献，但不是主story

### 最终推荐

**核心方法 = PF baseline + Archive Retrieval + Per-Head Dynamic CFG**

- 不改PF的head分类和cache策略
- 在PF之上添加archive retrieval（所有head共享，confidence gating）
- 用retrieval confidence做per-head dynamic CFG
- 这个方案与PF正交，不竞争，有明确的差异化
