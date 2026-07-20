# Prompt-Aware Functional Routing：创新收敛与论文执行计划

> 日期：2026-07-20
> 状态：真实 head 诊断、32-prompt 消融和 VBench 正在执行

## 1. 必须纠正的两个技术判断

### 1.1 不能在最终 flow prediction 上直接做 per-head CFG

DiT 最终输出的维度是 latent channel，不再包含 attention-head 轴。把最终 `flow_pred`
reshape 成 `[B,T,H,D]` 并按 head 施加 CFG 没有可靠语义；此前该路径实际上因没有保存
per-head confidence 而未生效。当前代码已明确禁用这一做法。

如果要做真正的 head-specific guidance，必须发生在 attention 内部，或者将 cond/uncond
attention 输出按 layer/head 成对处理。这比最终 flow 层的缩放复杂得多，不应在没有正确性
证明时进入主实验。

### 1.2 “随机打乱 archive 帧后取最大相似度”不能测 content specificity

如果候选集合没有改变，只是打乱顺序，`max(similarity)` 完全不变。因此旧的
`conf_correct - conf_random_permutation` 定义恒为零，不是有效信号。

新的真实诊断改为三个在线信号：

1. `prompt_reliance_h = ||A_cond,h - A_uncond,h|| / ||A_uncond,h||`
2. `history_confidence_h`：该 head 的 query/archive 检索置信度
3. `retrieval_margin_h = top1_weight - top2_weight`
4. 辅助信号 `memory_alignment_h = cos(A_native,h, A_memory,h)`

其中 prompt reliance 直接测量 head 对文本条件的响应，最接近“Veil/prompt-sensitive”假设；
retrieval margin 测量历史选择是否明确，避免把不确定检索当成可靠记忆。

## 2. 动态 head 分析的正确目标

不再试图在线复刻 PF 的 Anchor/Wave/Veil 三分类。PF 标签表示离线统计得到的时间访问模式，
而我们要估计的是推理时刻的功能状态：

| 在线状态 | Prompt reliance | History confidence/margin | 建议行为 |
|---|---:|---:|---|
| Semantic memory | 高 | 高 | 允许内容特异历史读取；保持 prompt 约束 |
| Prompt driven | 高 | 低 | 禁止不确定历史；保持较强 CFG |
| Layout/history | 低 | 高 | 允许结构历史；使用较低 memory gate |
| Local/motion | 低 | 低 | recent-only；不读 archive |

这不是永久类别，而是 `(layer, head, query block)` 的连续角色。实际路由应使用 EMA 和
hysteresis，避免每个 denoising step 频繁跳类。

## 3. 动态 Cache 规划

### 3.1 连续预算，而不是离散硬切换

对每个 head 计算：

```text
semantic_score = prompt_reliance * history_confidence * retrieval_margin
layout_score   = (1 - prompt_reliance_norm) * history_confidence
local_score    = 1 - history_confidence
```

然后分配三类预算：

```text
recent_budget_h  = base_recent + local_score * extra_recent
archive_budget_h = semantic_score/layout_score 对应的历史预算
periodic_budget_h = motion proxy 对应的周期历史预算
```

第一版不应直接改 PF ragged cache 结构，而应只动态控制独立 memory branch 的 per-head gate。
验证有效后，再把连续角色映射为 `[sink | middle | recent | retrieved]` 的实际 token 预算。

### 3.2 检索准入与 abstention

从 Echo-Forcing 审计得到的重要空缺：Echo 自动 recall 使用无条件 top-1，没有绝对阈值、
margin 或 entropy abstention。我们的准入规则应是：

```text
accept historical frame iff:
  top1_similarity >= tau_abs
  and top1 - top2 >= tau_margin
  and retrieval_entropy <= tau_entropy
```

不满足时必须 abstain，退回 native/PF recent cache。该机制直接针对 false memory 和高运动
pose mismatch，且与 Echo 的 post-hoc decay 不同。

### 3.3 Preview-before-commit

Echo 在第一块生成前直接注入历史，冲突度要到第一块 clean pass 后才能测量。更安全的设计：

```text
当前 noisy query 先做无 memory preview
→ 测量候选 memory 与 preview output 的 alignment/conflict
→ 决定 accept / attenuate / reject
→ 仅对后续 denoising step 或下一 block 提交 memory
```

这可作为独立贡献：memory admission timing，而不是照搬 Echo 的 difference-aware decay。

## 4. 历史压缩与其他工作融合

### 4.1 LongLive-RAG：只吸收训练无关部分

可吸收：
- clean latent/frame descriptor；
- recent exclusion；
- frame-level full K/V retrieval；
- fixed active budget。

不复制：
- 训练的 retrieval AE；
- 将 retrieved frames 直接拼入 native softmax；
- 所有 recalled frame 共享 temporal slot 0。

我们的差异应保持为：training-free descriptor、bounded archive、独立 memory attention、
confidence/alignment-controlled fusion。

### 4.2 Flash-VAReason：archive 维护

当前 full-frame archive 超限时主要均匀采样。后续改为三预算：

```text
archive budget: 长期保留多少帧
scan budget: 检索时看多少 descriptor
readout budget: 实际读取多少完整帧
```

archive 选择使用 relevance-independent uniqueness/coverage，retrieval 时再使用 query relevance。
这样不会因当前 query 偏好而永久删除未来可能需要的独特场景。

### 4.3 MemRoPE：位置正确性

只吸收通用原则：pre-RoPE K、真实时空 sidecar、Q/K 同一局部坐标系。不能声称 dual-rate EMA
或 Online RoPE Indexing 本身为我们的创新。当前 PF memory branch 仍偏内容 attention，正式
方法需要为 recalled full frames 加独立、bounded、order-preserving memory RoPE。

### 4.4 IAMFlow：身份与场景分离

IAMFlow 已覆盖 entity registry + identity frame retrieval，不能把“entity-organized memory”
作为主创新。可采用更轻量的 training-free 双生命周期：

- identity archive：稳定身份 exemplar；
- scene/state archive：可变布局、光照、物体状态；
- state version 变化时只失效 scene/state memory，不删除 identity core。

第一版不引入 LLM/VLM，使用已有 prompt/T5、K/V 和 coarse spatial descriptors。

## 5. 推荐论文主线

### 方法名候选

**Prompt-Aware Historical Memory Routing for Long-Horizon Video Diffusion**

### 核心论点

PF 解决离线的 head-aware history retention；我们解决在线的 historical-memory admission and
readout：当前 head 是否对 prompt 敏感、历史检索是否明确、memory 是否与当前生成方向一致，
共同决定是否读取、读取多少、何时提交。

### 四个贡献

1. 在线 prompt-reliance 与 history-certainty functional signals；
2. confidence-abstaining、preview-before-commit 历史准入；
3. bounded full-frame archive + independent memory attention；
4. prompt/memory confidence 驱动的全局 dynamic CFG（合法发生在 model output），而 per-head
   控制发生在 attention memory gate，不错误地施加于 latent channel。

## 6. 必须通过的判定门

### Gate 1：信号可分

真实推理中：prompt reliance 与 history confidence/margin 跨 head 的 CV 均需有明显方差，且
两者不能高度相关。否则放弃“动态分类”故事，保留 retrieval abstention。

### Gate 2：路由有因果价值

32 prompts 比较：
- PF static routing；
- no routing；
- confidence routing；
- prompt-aware routing；
- random/shuffled routing。

动态路由必须稳定优于 static 和 random，且不是靠冻结获得 VBench 一致性。

### Gate 3：历史内容特异

correct memory 必须优于 wrong-scene、shuffled-V 和 abstain。否则 archive 不是有效内容记忆。

### Gate 4：视觉质量

人工检查必须不增加肢体液化、拖影、错误回放、曝光漂移和背景硬切。VBench 只作辅助。

## 7. 当前实验解释限制

当前 32-prompt `full` 变体是在 per-head CFG 被正确实现前启动的；该变体不能作为 per-head CFG
有效性证据。它最多用于检验 adaptive memory routing。新方向 30s 中 `retrieval_cfg` 同样不能
证明 per-head CFG；有效差异主要来自 retrieval 和合法的全局 dynamic CFG。

后续报告必须显式标注，不能把 no-op/无效路径写成正向消融。
