# HREM-v2 与 SWIFT 碰撞分析

> 更新日期：2026-07-22
>
> 触发：docs/64 要求完成 SWIFT/Echo/LongLive-RAG/PF/Forcing-KV 逐项碰撞审计
>
> 状态：首轮机制级分析，尚未完成逐行代码对比

## 1. SWIFT 概述

**论文**: [SWIFT: Prompt-Adaptive Memory for Efficient Interactive Long Video Generation](https://arxiv.org/abs/2605.09442)  
**代码**: [ShanwenTan/SWIFT](https://github.com/ShanwenTan/SWIFT)  
**许可证**: Apache-2.0  
**任务**: multi-prompt long-video generation（与 HREM-v2 相同）

### SWIFT 的核心机制

| 机制 | 描述 |
|---|---|
| **Semantic Injection Cache** | 在 prompt 边界增强 cached memory，不重建 cache |
| **Head-wise Semantic Injection** | 每个 attention head 按其对当前 video state 的对齐度接收不同的 prompt update |
| **Adaptive Dynamic Window** | 按 prompt phase 分配 temporal memory（切换附近大窗口，稳定段小窗口） |
| **Segment-level Semantic Anchors** | 压缩 prompt-conditioned 历史 summary 为 compact memory tokens |

## 2. 与 HREM-v2 的重叠分析

### 2.1 机制级对比

| 维度 | SWIFT | HREM-v2 | 重叠程度 |
|---|---|---|---|
| **任务设定** | multi-prompt long video | A-B-A episodic return + multi-prompt long video | 🔴 高度重叠 |
| **记忆类型** | working cache + semantic injection | 独立 episodic archive sidecar | 🟡 不同架构 |
| **Head 处理** | **head-wise semantic injection**（按对齐度分配 prompt 更新） | **per-head K/V persistence + query drift gate**（按稳定性分配 readout 权） | 🔴 **高度重叠** |
| **时序策略** | Adaptive Dynamic Window（切换边界大窗口） | memory_start_episode + warmup_blocks | 🟡 概念重叠 |
| **长期一致性** | Segment-level anchors（压缩历史） | Episode-balanced coverage archive | 🟡 目的重叠 |
| **Abstention** | 首轮机制审计尚未发现显式 abstention | fail-closed 精确回退 native | 当前实现差异；不是“世界首次”结论 |

### 2.2 最危险的碰撞：Head-wise 选择性

SWIFT 的 head-wise semantic injection：
```
每个 attention head 接收 prompt update ∝ alignment(current_video_state, head)
→ 对齐度高的 head 接收更多 injection
→ 对齐度低的 head 保持 native behavior
```

HREM-v2 的 per-head admission：
```
每个 attention head 的 gate = sigmoid(sharpness × (K_persistence + query_stability - motion_risk - threshold))
→ 稳定的 head 可读取 episode memory
→ 不稳定的 head 回退到 native
```

两者共享“按 head 选择性注入信息”的高层模式。SWIFT 注入的是 prompt semantic update，HREM 注入的是 episode K/V；因此不能把 **head-wise selectivity** 本身作为新颖性，剩余差异必须由 payload、证据、episode 约束和实验共同支撑。

### 2.3 真正的差异（可辩护的）

| 差异点 | SWIFT | HREM-v2 |
|---|---|---|
| **注入对象** | prompt-conditioned semantic signal | historical episode K/V payload |
| **注入位置** | 修改 working cache | 独立 side branch（不污染 native attention） |
| **决策证据** | head-to-video alignment（单次匹配） | K/V persistence + query drift（在线动态证据） |
| **Episode 约束** | 首轮机制审计未发现与 HREM 相同的 non-recent episode admission | 显式 non-recent episode exclude + dual-evidence admission |
| **安全机制** | 首轮机制审计尚未发现同构显式 abstention | 失败回退 native（confidence × head × alignment = 0 时 output unchanged） |
| **因果审计** | 尚未完成同口径代码级审计 | 完整 trace + 逐层 causal controls |

### 2.4 论文措辞影响

| 禁止/风险措辞 | 原因 | 替代措辞 |
|---|---|---|
| "per-head memory access" | SWIFT 已有 head-wise injection | "online head eligibility estimation via K/V persistence" |
| "head-aware episodic recall" | SWIFT 已有 head-aware semantic recall | "factorized admission (episode × head) with explicit historical payload" |
| "head specialization for memory" | PF + Forcing-KV + SWIFT 均已覆盖 | "motivated by head specialization findings, we estimate per-head readout eligibility" |

## 3. 对论文故事的影响

### Story A (factorized admission) 可行性下降

SWIFT 的 head-wise injection 削弱了 head admission 的新颖性。**Story A 仍然可行**，但需要在论文中：

1. 明确引用 SWIFT 作为 head-wise memory access 的先例
2. 在 experiment section 直接对比 SWIFT（或最接近的 semantic-injection ablation）
3. 将贡献从 "head-wise" 重新聚焦到 **"episode-constrained head eligibility + causal controls"**

### 推荐调整后的 Story A

```
贡献 1: non-recent episode admission（Echo/LongLive-RAG 有 scene recall，但没有 explicit non-recent exclusion + dual evidence）
贡献 2: K/V persistence-based head eligibility（SWIFT 有 head-wise alignment，但我们用在线动态证据而非单次匹配，且控制的是 historical payload 而非 prompt injection）
贡献 3: Factorized decision + causal trace + fail-closed（SWIFT/PF 都有 per-head 操作，但没有 episode × head 两级 factorized decision with complete audit trail）
```

### 如果 head gate 无稳定收益 → Story B/C

- Story B（selective episodic recall with abstention）：删除 head gate，聚焦 episode admission + safety
  - 与 SWIFT 的区分点：explicit episode concept + non-recent exclusion + fail-closed
- Story C（diagnostic paper）：贡献因果分解和失败模式分析
  - 与 SWIFT 无直接冲突（诊断 vs 方法）

## 4. 必须做的实验

| 实验 | 目的 | 优先级 |
|---|---|---|
| 同 prompt/frame/backbone 下与 SWIFT 的 head-wise injection 比较 | 证明在线 K/V persistence 优于 alignment-based | P0 |
| SWIFT-style semantic injection ablation（用我们的 archive 实现 SWIFT 风格的 head-alignment gate） | 隔离"injection target" vs "head evidence type" 的贡献 | P1 |
| Head gate 消融：K-persistence only, V-persistence only, query-stability only, motion-risk only | 证明每个 evidence component 的必要性 | P1 |

## 5. SWIFT 碰撞审计结论

SWIFT 是当前与 HREM-v2 概念重叠最深的工作：
- **head-wise selectivity** 已被覆盖
- **multi-prompt memory** 已被覆盖
- **segment-level history** 已被覆盖

HREM-v2 可以区分的剩余空间：
1. 显式 non-recent episode admission（首轮 SWIFT 审计未发现同构机制）
2. Explicit historical K/V payload（SWIFT 注入 semantic signal，非完整 K/V）
3. Online K/V persistence evidence（SWIFT 使用 single-shot alignment）
4. Factorized episode × head decision
5. Complete causal audit + fail-closed

**论文必须直接引用 SWIFT 并做 head-level ablation 对比，否则 reviewer 会质疑 novelty。**
