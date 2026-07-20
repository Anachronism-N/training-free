# 动态Head功能分析与Cache规划方案

> 日期：2026-07-20
> 目的：设计真正差异化的动态head分类 + 动态cache规划机制

---

## 1. PF的Head分类与Cache策略的精确关系

PF不是"分类完就完了"——分类直接决定了每个head的cache组成：

```
PF标签 → cache策略映射：
  -1 (oscillating/Wave) → [sink=1] + [周期采样中间帧] + [recent=4]
   1 (stable/Anchor)    → [sink=1] + [stride采样远期帧] + [recent=4]
   2 (stable-sparse/Veil) → [sink=1] + [合并相邻帧] + [recent=4]
```

每个head的cache由三部分组成：`[sink | middle | recent]`
- **sink**: 固定保留的最旧帧（所有head都有）
- **middle**: 根据标签不同的策略——stride采样/周期采样/合并
- **recent**: 最近的N帧（所有head都有，但N可以不同）

**PF的关键限制**：
1. 标签离线固定 → 同一个head在所有prompt/时间步/视频上使用相同的[sink|middle|recent]组成
2. middle策略固定 → Anchor head永远做stride，Wave head永远做周期采样，不管当前内容是什么
3. 无retrieval → 不根据当前query决定保留哪些历史帧，只按固定pattern采样

---

## 2. 我们的核心问题

**如何动态分类？** 和 **分类后如何规划cache？**

这两个问题必须一起回答，因为分类的价值体现在cache规划上。

### 2.1 为什么不能用简单的confidence做"动态分类"

当前`confidence_adaptive`模式的问题：
- 它只做"是否允许读memory"的binary mask，不改变cache本身的组成
- 它没有回答"这个head应该保留什么历史、保留多少"
- 它本质上是memory branch的gate，不是cache policy的动态规划

### 2.2 真正的动态cache规划需要回答的问题

对每个head、每个时间步：

1. **这个head当前需要什么类型的历史？**
   - 身份信息（人物外观不变的部分）
   - 运动信息（动作的时序变化）
   - 布局信息（背景的空间结构）
   - 或者不需要远期历史（只看recent）

2. **这个head应该保留多少历史？**
   - 有些head需要长程历史（21帧全部）
   - 有些head只需要近期（4帧）
   - 有些head需要特定时间段（如5帧前的动作相位）

3. **这个head的历史应该怎么组织？**
   - 完整保留（stride）
   - 压缩合并（merge）
   - 周期采样（cyclic）
   - 按query检索（retrieval）← **PF不做的**

---

## 3. 推荐方案：Query-Adaptive Cache Planning (QACP)

### 3.1 核心思想

**不在离线给head贴标签，而是在线测量每个head当前需要什么类型的历史，然后动态规划cache组成。**

PF: `离线标签 → 固定cache策略`
我们: `在线query分析 → 动态cache规划`

### 3.2 在线Head功能分析（如何动态分类）

不用FFT/sign-rate，而是用**两个在线信号**分析head的当前功能：

#### 信号1: Temporal Sensitivity (时间敏感度)

```python
# 对每个head，比较"看全部历史"vs"只看recent"的attention output差异
x_full = attention(q, all_history_k, all_history_v)     # 完整历史
x_recent = attention(q, recent_k, recent_v)              # 仅近期

# temporal_sensitivity = ||x_full - x_recent|| / ||x_recent||
# 高值: 这个head强烈依赖远期历史（需要保留远期）
# 低值: 这个head只关注近期（不需要远期历史）
```

#### 信号2: Content Specificity (内容特异性)

```python
# 对每个head，比较"正确历史"vs"错误历史"的retrieval confidence
conf_correct = retrieval_confidence(q, correct_archive)  # 正确场景历史
conf_random = retrieval_confidence(q, random_archive)    # 随机历史

# content_specificity = conf_correct - conf_random
# 高值: 这个head能区分正确vs错误历史（需要精确retrieval）
# 低值: 这个head对历史内容不敏感（可以粗略保留或不需要）
```

### 3.3 动态Cache规划（分类后如何使用cache）

根据两个信号，将head动态分为四类，每类有不同的cache策略：

```
                    Content Specificity
                    高              低
Temporal    高  │ Identity Head   │ Motion Head
Sensitivity     │ (精确retrieval) │ (周期采样)
            低  │ Layout Head     │ Recent Head
                │ (stride保留)    │ (仅recent)
```

#### 四类head的cache策略：

**Identity Head** (高时间敏感 + 高内容特异):
- 需要远期历史，且需要正确的历史
- Cache策略: `recent + query-retrieved archive frame`
- 这是PF完全没有的——PF不retrieval，只按固定pattern保留

**Motion Head** (高时间敏感 + 低内容特异):
- 需要远期历史，但对内容不敏感（关注运动模式而非内容）
- Cache策略: `recent + 周期采样中间帧`（类似PF的Wave）
- 但周期参数可以根据当前运动强度动态调整

**Layout Head** (低时间敏感 + 高内容特异):
- 不需要远期时序，但需要正确的空间结构
- Cache策略: `recent + stride采样远期帧`（类似PF的Anchor）
- 但stride间隔可以根据场景变化频率动态调整

**Recent Head** (低时间敏感 + 低内容特异):
- 只关注近期，不需要远期历史
- Cache策略: `仅recent`（类似PF的Veil/stable-sparse）
- recent帧数可以根据当前运动速度动态调整

### 3.4 与PF的本质区别

| 维度 | PF | QACP |
|------|-----|------|
| **分类方法** | FFT/sign-rate (离线) | Temporal sensitivity + content specificity (在线) |
| **分类维度** | 时间模式 (1维: 振荡/稳定/稀疏) | 时间×内容 (2维: 4类) |
| **分类时机** | 离线一次性 | 每个时间步动态更新 |
| **Cache策略** | 每类固定1种 | 每类1种，但参数动态 |
| **Retrieval** | 无 | Identity head做query retrieval |
| **是否随query变化** | 否 | 是 |
| **是否随prompt变化** | 否 | 是 |

### 3.5 为什么PF无法简单扩展到QACP

1. PF的标签是离线计算的，无法获取"当前query与历史的匹配度"
2. PF不做retrieval，无法测量content specificity
3. PF的分类维度只有时间模式（1维），无法区分"需要正确历史"和"需要任何历史"
4. PF的cache策略是互斥的（每个head只用一种middle strategy），QACP允许Identity head同时用retrieval + recent

---

## 4. 实现方案

### 4.1 Phase 1: 在线Head功能分析 (轻量级)

```python
def analyze_head_functions(q, kv_cache, archive, current_frame):
    """在线分析每个head的功能角色。
    
    返回: (temporal_sensitivity[H], content_specificity[H])
    """
    H = q.shape[2]  # head数
    
    # Signal 1: Temporal Sensitivity
    # 比较"有远期历史"vs"无远期历史"的attention output差异
    x_with_history = attention(q, all_cache_k, all_cache_v)
    x_recent_only = attention(q, recent_k, recent_v)
    temporal_sens = (x_with_history - x_recent_only).norm(dim=-1) / x_recent_only.norm(dim=-1).clamp(min=1e-6)
    # [H] tensor
    
    # Signal 2: Content Specificity
    # 比较正确archive vs 随机archive的retrieval confidence
    conf_correct = retrieval_confidence(q, archive)  # [H]
    conf_random = retrieval_confidence(q, random_frames)  # [H]
    content_spec = conf_correct - conf_random  # [H]
    
    return temporal_sens, content_spec
```

**优化**: 不需要每个时间步都做完整分析。可以：
- 每3-5个block更新一次（降低开销）
- 只在memory-enabled layers (15-20)做分析
- 用轻量级proxy（如mean K相似度）代替完整attention

### 4.2 Phase 2: 动态Cache规划

```python
def plan_cache_per_head(temporal_sens, content_spec, thresholds):
    """根据功能分析结果，为每个head规划cache策略。
    
    返回: head_plans[H] — 每个head的cache策略
    """
    plans = []
    for h in range(H):
        ts = temporal_sens[h]
        cs = content_spec[h]
        
        if ts > thresholds['ts'] and cs > thresholds['cs']:
            plan = CachePlan(
                type='identity',
                recent_frames=4,
                archive_retrieval=True,  # query-conditioned retrieval
                archive_top_k=3,
            )
        elif ts > thresholds['ts'] and cs <= thresholds['cs']:
            plan = CachePlan(
                type='motion',
                recent_frames=4,
                cyclic_sampling=True,
                cyclic_period=adaptive_period,  # 根据运动速度调整
            )
        elif ts <= thresholds['ts'] and cs > thresholds['cs']:
            plan = CachePlan(
                type='layout',
                recent_frames=4,
                stride_sampling=True,
                stride_interval=adaptive_interval,  # 根据场景变化调整
            )
        else:
            plan = CachePlan(
                type='recent',
                recent_frames=adaptive_recent,  # 可以比4更多或更少
            )
        plans.append(plan)
    return plans
```

### 4.3 Phase 3: 与现有memory branch的关系

QACP不替换PF的cache管理，而是在PF之上添加**动态retrieval层**：

```
PF原生cache (sink + middle + recent) ← 保留，作为baseline
  +
QACP动态层:
  - Identity heads: 额外获得query-retrieved archive frame
  - Motion heads: 保持PF的周期采样（但参数可能调整）
  - Layout heads: 保持PF的stride采样（但参数可能调整）
  - Recent heads: 保持PF的recent-only
```

**关键区别**: PF的head分类决定了middle strategy；QACP的分类决定了**是否额外获得archive retrieval**。这是PF完全不做的。

---

## 5. 消融实验设计

### 5.1 验证动态分类的意义

| 变体 | 分类方法 | Cache规划 | 验证 |
|------|---------|-----------|------|
| PF baseline | PF静态标签 | PF固定策略 | 基线 |
| QACP-static | PF静态标签 + QACP规划 | Identity head做retrieval | PF标签 + 我们的cache规划 |
| QACP-dynamic | QACP在线分析 | 4类动态规划 | 完整QACP |
| QACP-random | 随机分类 | 4类动态规划 | 控制组 |

### 5.2 验证每个信号的贡献

| 变体 | Temporal Sens | Content Spec | 验证 |
|------|--------------|--------------|------|
| QACP-full | ✓ | ✓ | 完整方法 |
| QACP-ts-only | ✓ | ✗ | 只用时间敏感度 |
| QACP-cs-only | ✗ | ✓ | 只用内容特异性 |
| QACP-no-analysis | 固定4类均分 | 固定4类均分 | 不做分析，随机分配 |

### 5.3 验证retrieval的价值

| 变体 | Identity Head策略 | 验证 |
|------|------------------|------|
| QACP + retrieval | Query-conditioned retrieval | 完整方法 |
| QACP + stride | 用stride代替retrieval | retrieval vs 固定采样 |
| QACP + random frame | 随机选archive帧 | retrieval的特异性 |

---

## 6. 与其他工作的融合

### 6.1 LongLive-RAG的frame-level descriptor

LongLive-RAG训练了一个autoencoder来做frame retrieval。我们保持training-free，用K summary的cosine similarity代替。但如果QACP的Identity head需要更精准的retrieval，可以考虑：
- 用clean latent的mean/std作为frame descriptor（不训练）
- 用cross-attention的prompt embedding作为semantic descriptor

### 6.2 Echo-Forcing的场景切换

Echo-Forcing的场景切换/recall机制可以自然融入QACP：
- 当检测到场景切换时（content specificity突然下降），暂时关闭Identity head的retrieval
- 当场景回访时（content specificity恢复），重新启用retrieval

### 6.3 Flash-VAReason的固定预算压缩

Flash-VAReason的"固定预算 + uniqueness retention"可以用于archive管理：
- 当archive超限时，不只做uniform subsampling
- 而是用uniqueness score保留不可替代的帧
- 这与QACP的Identity head retrieval互补——更好的archive质量 → 更好的retrieval

### 6.4 我们之前提及的压缩和历史信息使用

之前讨论过的压缩方案可以融入QACP：
- Identity head: 不压缩，精确retrieval完整帧
- Motion head: 轻度压缩，保留时序模式
- Layout head: 中度压缩，保留空间结构
- Recent head: 不需要远期，无压缩

---

## 7. 论文故事

**标题**: Query-Adaptive Cache Planning for Long-Horizon Video Generation

**核心贡献**:

1. **在线Head功能分析**: 不用离线FFT分类，而是在线测量每个head的temporal sensitivity和content specificity，得到2维功能角色。

2. **动态Cache规划**: 根据功能角色，为每个head动态规划cache策略——Identity head做query retrieval，Motion head做自适应周期采样，Layout head做自适应stride，Recent head仅保留近期。

3. **Query-Conditioned Retrieval**: Identity head从clean-frame archive中按当前query检索相关历史帧，这是PF完全不做的。

4. **Per-Head Dynamic CFG**: 根据retrieval confidence动态调节每个head的guidance scale。

**与PF的关系**: PF是baseline。PF的离线head分类是固定的1维时间模式；QACP是在线2维功能分析。PF的cache策略是固定pattern；QACP是query-adaptive的动态规划。PF不retrieval；QACP对Identity head做retrieval。

**可发表性**: QACP回答了一个PF没有回答的问题——"同一个head在不同prompt/不同时间步是否需要不同的历史？" 答案是是的，而且可以通过在线分析来动态规划。
