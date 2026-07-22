# AMA/RollingForcing 历史实验启发与 HREM-v2 关联

> 日期: 2026-07-22
> 来源: `develop/research_sprint/AMA/` 和 `develop/research_sprint/RollingForcing/`
> 目的: 从 RollingForcing 上的 82 方法 × 20 prompt 大规模实验中提取对当前 HREM-v2 的直接启发

## 1. AMA/RollingForcing 核心成果

### 1.1 方法栈

| 组件 | 全称 | 机制 | 贡献 |
|---|---|---|---|
| **AAI** | Anchor Attention Injection | query-dependent anchor attention 作为 additive residual | **+6.1% DINO**（主导） |
| **HRMR** | Head-Role Memory Routing | per-head anchor K-scaling based on profiling | +2.2% DINO |
| **DARV** | Drift-Aware Reference Verification | step-wise anchor refresh | +0.5%（仅在 AAI+HRMR 之上） |

### 1.2 关键数值（20-prompt × 2min）

| 配置 | DINO | min_DINO | Motion | Loop | 说明 |
|---|---|---|---|---|---|
| RF baseline | 0.8413 | 0.7878 | 33.23 | 0.044 | 无增强 |
| AAI only | 0.9042 | 0.8376 | 18.09 | 0.349 | 主导组件 |
| V1 (AAI+HRMR+DARV) | 0.9216 | 0.8614 | 14.86 | 0.312 | 原 paper 主配置 |
| S5 (HRMR thresh=0.15) | 0.9628 | 0.9005 | 18.21 | 0.736 | identity 分布式 |
| V2 (AAI 0.07-0.18 + S5) | **0.9698** | 0.9136 | **11.17** | 0.857 | DINO 最优但运动冻结 |
| Z3 (B1+DARV+profile) | 0.8975 | **0.8117** | — | **0.2145** | 最均衡 |

### 1.3 跨 backbone VBench-Long 对比（60s × 128 MovieGen）

| 方法 | VBench-Long 总分 |
|---|---|
| Self-Forcing | 77.87 |
| Rolling Forcing | 79.80 |
| Causal-Forcing | 79.14 |
| LongLive | 80.47 |
| SF+Deep-Forcing | 80.08 |
| **SF+Pyramid-Forcing** | **81.21** (SOTA) |
| SF+ours (RF) | dynamic 43.12, aesthetic 60.11, imaging 68.20 |
| SF+PF | dynamic 43.75, aesthetic 55.62, imaging 64.28 |

**关键**: SF+ours 在 aesthetic (+4.5) 和 imaging (+3.9) 上优于 SF+PF，但 dynamic 持平。

## 2. 对 HREM-v2 的直接启发

### 2.1 AAI ≈ HREM-v2 的独立 memory attention 分支

| 维度 | AAI (RF) | HREM-v2 (SF) |
|---|---|---|
| 注入方式 | additive residual: O += α × anchor_attention | convex fusion: O = (1-w)×native + w×memory |
| 独立性 | 独立 attention 计算 | 独立 attention 分支 |
| query-dependent | ✅ 是 | ✅ 是（Q-K retrieval） |
| 效果 | +6.1% DINO | 待验证 |

**启发**: HREM-v2 的独立分支方向正确，但需要验证在纯长视频中的效果。AAI 在 RF 上有效是因为 RF 有 anchor segment，HREM-v2 需要证明 archive 能替代 anchor。

### 2.2 HRMR ≈ HREM-v2 的 head gate

| 维度 | HRMR (RF) | HREM-v2 (SF) |
|---|---|---|
| 分类方式 | offline profiling → identity/global/mixed | online K/V persistence + query drift |
| id_thresh | 0.15（38.9% heads）最优 | 0.45（几乎所有 heads 通过）→ 太松 |
| scale | identity ×1.5, motion ×0.7 | sigmoid gate |
| 效果 | +3.78% DINO（thresh 0.25→0.15） | +0.014 margin（不显著） |

**关键启发**: 
1. **id_thresh=0.15 >> id_thresh=0.25** → identity 是**分布式**的，不是集中在少数 head
2. HREM-v2 的 threshold=0.45 对应 HRMR 的 thresh=0.25（太严格），应该降低到 ~0.15
3. **但 HRMR 用的是 offline profiling**，HREM-v2 用的是 online evidence → 需要**验证 online evidence 与 offline profiling 的一致性**

### 2.3 失败方法的教训

| 失败方法 | 原因 | HREM-v2 对应风险 |
|---|---|---|
| K-scaling (C1-C6) | anchor K boost → info leakage → ghosting | HREM-v2 不做 K-scaling ✅ 安全 |
| Content-Aware RoPE | 修改位置编码 → 破坏学习到的 routing | position_mode=none ✅ 安全 |
| Q/K/V LoRA | representation 修改 → 破坏 routing | training-free ✅ 安全 |
| Attention Temperature | catastrophic | 不修改 temperature ✅ 安全 |
| EMA-Sink fusion | sink representation 污染 | 独立 branch ✅ 安全 |
| Plan E (RelRoPE) | 5min 时 DINO -3.7% | 不使用 relative position ✅ 安全 |

**总结**: HREM-v2 的设计避免了所有已知的失败模式。

### 2.4 Identity-Motion Tradeoff

```
+1% DINO ≈ -7% Motion
```

| AAI alpha | DINO | Motion | 问题 |
|---|---|---|---|
| 0.04-0.12 | 0.91 | ~18 | 均衡 |
| 0.07-0.18 | 0.97 | 11.17 | **运动冻结** |
| 0.03-0.10 | 0.89 | ~22 | 运动充分 |

**启发**: HREM-v2 的 fusion gate=0.10 对应 AAI alpha ~0.05，在安全区间。但如果 head gate 不 selective，effective weight 可能过高。

### 2.5 评估指标启发

| 指标 | 说明 | HREM-v2 当前使用? |
|---|---|---|
| **DINO avg** | 帧间主体一致性 | ✅ 但不够 |
| **min_DINO** | 最低帧一致性（捕捉局部崩溃） | ❌ 应该加 |
| **Motion (RAFT)** | 光流加速度 | ❌ 应该加 |
| **ArcFace ID** | 面部 identity | ❌ 应该加 |
| **LPIPS Flicker** | 相邻帧感知变化 | ❌ 应该加 |
| **CLIP-Text** | prompt 对齐 | ❌ 应该加 |
| **Loop Score** | 帧重复（高=冻结/循环） | ❌ 应该加 |
| **Subject/Camera Motion** | homography 分解 | ❌ 应该加 |
| **VBench-Long** | 6 维标准 benchmark | ❌ **必须加** |

**关键**: **avg DINO 会掩盖局部崩溃**。V1 在 4:15 有 ghosting 但 avg DINO=0.89 看不出来。必须用 min_DINO + max_flicker。

## 3. 最有启发性的方向

### 3.1 "WHICH frames" → "WHAT information in those frames"

**所有现有方法**（包括 AAI/HREM-v2）都是选择"哪些帧"来注入，但注入的是整帧的 K/V（identity+background+motion+lighting 混合）。

**新方向**: 分离 K/V 中的 identity 信息和 motion 信息：
- **R-A**: KV subspace projection — 用 PCA 将 anchor K/V 投影到 identity subspace
- **R-B**: Temporal frequency decomposition — 低频（identity/layout）注入，高频（motion/lighting）丢弃
- **R-C**: Channel-level gating — 比 head-level 更细粒度

**这直接解决 identity↑motion↓ 的根本矛盾**。

### 3.2 Identity 是分布式的

HRMR 的 S5 实验：
- id_thresh=0.25 → 7.8% heads 被标记为 identity → DINO 0.89
- id_thresh=0.15 → 38.9% heads → DINO 0.96

**启发**: HREM-v2 的 head gate 应该让 ~40% 的 heads 通过，而不是当前的 ~99%。threshold 应该大幅降低。

### 3.3 SF DMD 的 15s 崩溃

SF DMD 是 5s 训练的模型，15s+ 面部 identity 完全崩溃。这意味着：
- **30s 纯长视频实验中 SF native 会崩溃** → 我们的 method 如果能延缓崩溃就是贡献
- **需要对比 CF longvideo.pt**（RF-based，更长时训练）

## 4. 对当前实验的即时建议

1. **head gate threshold**: 从 0.45 降到 0.15-0.25（参考 HRMR 的最优值）
2. **评估指标**: 必须加 min_DINO + Motion + Loop + VBench-Long
3. **AAI 对比**: HREM-v2 应该与 AAI 做直接对比（两者都是独立 attention 分支）
4. **alpha/gate 控制**: 注意 identity-motion tradeoff，gate 不要太大
5. **纯长视频**: SF native 30s 会崩溃 → HREM-v2 的 intra_episode scope 如果能延缓崩溃就是核心贡献

## 5. 人工 Review 维度（文字描述版）

用户建议用文字描述而非打分。以下是建议的 review 维度：

### 每个视频记录以下文字描述：

1. **主体 identity**: "30秒内人物面部/服装/体型是否保持一致？何时开始变化？"
2. **背景/布局**: "背景是否漂移？有无突然变化？场景是否保持？"
3. **运动自然度**: "人物动作是否自然？有无冻结/循环/重复？镜头运动是否流畅？"
4. **伪影**: "有无重影/纹理粘连/闪烁/错误物体复制？出现在什么时间？"
5. **首次失败**: "第一个明显问题的出现时间（如 '15s 时面部开始模糊'）"
6. **总体印象**: "一句话总结视频质量"

### 对比维度：

- "A 和 B 哪个 identity 更好？"
- "A 的运动是否比 B 更自然？"
- "A 有无 B 没有的伪影？"
