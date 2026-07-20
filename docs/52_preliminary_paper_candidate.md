# 初步论文候选：Coverage-Aware Episodic Memory Readout

> 日期：2026-07-20
> 状态：三-prompt 30秒通过初筛；匹配的32-prompt PF对照正在运行

## 1. 研究问题

PF 通过离线 head temporal-pattern 分类，决定每个 head 的 `[sink | middle | recent]` 保留策略。
它解决的是 **retention**：有限原生 cache 中保留哪种历史。

本工作研究一个正交问题：

> 当原生 cache 已经丢弃远期内容后，如何用固定长期预算维护视觉事件，并在不扰乱原生
> self-attention 的条件下，按当前 query 选择性读出？

## 2. 方法

候选方法名称：**Coverage-Aware Episodic Memory Readout (CEMR)**。

### 2.1 Clean full-frame episodic archive

每个生成 block 的 clean consolidation pass 将完整空间 K/V 帧写入 archive。长期 archive
只保存 clean memory，不使用 noisy eviction。

### 2.2 Query-independent coverage maintenance

Archive 超过固定帧预算时，不按时间均匀采样，也不按当前 query 永久删除内容。对 training-free
frame descriptor 使用 greedy k-center：

```text
始终保留首尾帧
→ 逐次选择与已选集合距离最大的完整帧
→ 保持全历史内容覆盖
```

Archive maintenance 优化全局 coverage，query retrieval 只在读取时优化 relevance，二者解耦。

### 2.3 Query-conditioned episodic retrieval

读取时排除最近4帧，从真正远期 archive 中选择 top-3 内容相关帧。当前 descriptor 来自 raw K
的 per-head summary；无需训练 retrieval encoder。

### 2.4 Independent memory attention

Retrieved full-frame K/V 不拼入 PF 原生 self-attention softmax，而通过独立 memory attention
读取，再以 confidence/alignment 控制的 convex replacement 融合：

```text
x_native = PF attention(...)
x_memory = episodic attention(q, retrieved full-frame K,V)
w = gate × retrieval confidence × native-memory alignment
x = (1-w) × x_native + w × x_memory
```

这避免了远期 token 改变原生 recent attention 的概率分母。

### 2.5 Uncertainty-aware abstention

支持绝对 confidence、top1/top2 margin 与 retrieval entropy 三个准入信号；不可靠历��可以完全
abstain。当前自然单场景 prompt 中硬阈值容易过严，因此最终候选暂采用低 gate 的 soft
confidence；hard abstention 主要保留给 scene-switch/false-recall 任务。

## 3. 与 PF 的实质区别

| 维度 | PF | CEMR |
|---|---|---|
| 历史范围 | 原生有限 cache | 独立 bounded long-term archive |
| 目标 | retention | episodic retrieval/readout |
| 决策 | 离线 head temporal labels | 在线 query-content relevance |
| 历史选择 | stride/cyclic/merge pattern | content-addressed top-k frames |
| Archive压缩 | 无独立archive | query-independent coverage selection |
| Attention | 原生 self-attention | 独立 memory branch |
| 不确定性 | 无 recall abstention | confidence/margin/entropy admission |

CEMR 不重新声称 PF 的 head 分类，不依赖 PF labels 作为方法贡献。PF可以作为 recent-cache
backend；CEMR负责长期episodic memory。

## 4. 与相关工作的边界

- LongLive-RAG：已有 full-frame K/V retrieval 与 recent exclusion；CEMR 的区别是 bounded
  query-independent archive coverage、training-free descriptor、独立 memory attention 与
  controlled fusion，不使用训练AE，不将memory拼入native softmax。
- Echo-Forcing：已有scene recall和post-hoc conflict decay；CEMR不声称scene recall/decay，
  而使用在线 retrieval uncertainty admission。未来 scene-switch 版本应强调 abstention 和
  preview-before-commit，而不是复制Echo控制器。
- MemRoPE：已有pre-RoPE存储与online re-indexing；CEMR当前候选使用position-decoupled
  episodic content readout。local-grid RoPE消融没有形成优势，因此不作为主方法。
- IAMFlow：已有entity-organized full-frame memory；CEMR不使用LLM/VLM entity registry，主张
  bounded visual coverage和decoupled episodic readout，而非entity-memory首创。

## 5. 当前证据

### 5.1 三-prompt 30秒候选（coverage archive，gate=0.075）

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.93283 | 0.92584 | **0.60863** | 0.62708 | 0.96791 | 0.95556 |
| CEMR候选 | **0.93693** | **0.93032** | 0.60581 | **0.64647** | **0.97218** | **0.97778** |

候选在5/6指标上高于PF，仅aesthetic低0.00282；不像此前高gate变体那样通过降低dynamic换取
一致性。这是目前最接近Pareto改进的配置。

人工review入口：

```text
runs/REVIEW_v53_candidate_30s/
```

左PF，右CEMR候选。

### 5.2 K/V内容特异性

正常aligned K/V相对shuffled-V：subject +0.00135、background +0.00355、aesthetic +0.00978、
motion +0.00215；imaging下降0.00870。说明K/V空间对齐携带有用内容，但还没有跨场景wrong-memory
的严格因果证据。

### 5.3 负结果

- Dynamic head routing没有稳定优于no-routing/PF-static，因此不作为主贡献。
- Global dynamic CFG只改善imaging，降低其余多数维度并增加约67%开销，已停止。
- Hard margin=0.03+entropy=0.85导致0%接收，说明自然单场景生成不适合过强abstention。
- Local-grid memory RoPE没有明显优于raw position-decoupled readout，不进入候选。

## 6. 当前可写论文的初步故事

> 长视频AR扩散中的远期历史管理包含两个正交问题：PF类方法解决有限cache retention；我们
> 研究bounded episodic archive的构建与解者读取。CEMR以clean完整帧为单位，用query-independent
> coverage维持长期内容多样性，再按当前query检索远期事件，通过独立memory attention避免扰动
> 原生局部动力学。

## 7. 仍需完成的论文门槛

1. 匹配seed与prompt分片的32-prompt PF vs CEMR评估（正在运行）；
2. 至少2–3 seeds的最终候选复验；
3. 人工确认无额外肢体、闪回、曝光崩溃和动作回放；
4. scene-switch correct/wrong/abstain任务证明真正的episodic content specificity；
5. 移植到原生SF/CF或证明archive/readout不依赖PF专属机制；
6. 报告latency、archive显存/内存和readout开销。

## 8. 匹配32-prompt结果

使用同一32条MovieGen prompts、相同seed与120 latent frames，按四个8-prompt shard在对应GPU
运行PF与CEMR，并重新聚合评估：

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.97761 | 0.96548 | 0.64681 | 0.72195 | 0.98718 | **0.58750** |
| CEMR | **0.97882** | **0.96671** | **0.64739** | **0.72667** | **0.98747** | 0.56250 |
| Δ | +0.00121 | +0.00123 | +0.00058 | +0.00472 | +0.00029 | -0.02500 |

CEMR在5个质量/一致性维度上保持小幅正增益，说明3-prompt趋势能扩展到32 prompts；但Dynamic
Degree下降0.025，方法仍有保守生成倾向，绝对提升也偏小。因此现在可以称为**初步可行候选**，
但还不能称为显著SOTA或完成投稿门槛。

下一步必须依赖：

1. seed1/2结果是否保持正方向；
2. 人工review是否无新增冻结/动作回放；
3. 真实scene-switch wrong-memory对照；
4. 至少一个非PF backend。
