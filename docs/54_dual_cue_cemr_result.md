# Dual-Cue CEMR：初步可行结果与最终方法收敛

> 日期：2026-07-20
> 状态：seed0 机制闭环完成；seed1/2复验运行中

## 1. 为什么需要 Dual-Cue

纯视觉 query/K retrieval 在显式 A-B-A 场景回访中没有稳定选回A1：`correct visual` 的full
return margin为 `-0.1057`，甚至低于PF。原因是A2刚从B过渡，当前视觉query仍可能被B状态
污染；仅靠生成状态相似度无法表达“用户现在要求返回A”。

因此CEMR引入两个互补query：

```text
visual continuity cue: current raw query vs historical frame K
semantic intent cue: current prompt embedding vs archived prompt embedding

score = (1 - lambda) * visual_similarity + lambda * prompt_similarity
```

Archive仍使用query-independent coverage维护；semantic prior只影响读取，不改变长期保留内容。
该设计无需训练额外retrieval encoder，也不需要LLM/VLM entity registry。

## 2. Controlled A-B-A结果

三个显式block-aligned prompt分别测试：人物/公园返回、跑酷/屋顶返回、咖啡店返回。每条视频
由A1、B、A2三个等长片段组成，切换时只刷新cross-attention，episodic archive持续存在。

DINO segment centroid指标：

| 方法 | Full A1-A2 | Full B-A2 | Full return margin | BG return margin |
|---|---:|---:|---:|---:|
| PF | 0.8500 | 0.8270 | 0.0230 | 0.0571 |
| Pure visual retrieval | 0.7706 | 0.8763 | -0.1057 | 0.0681 |
| Wrong-B control | 0.8645 | 0.7960 | 0.0685 | 0.0667 |
| Shuffled-V | 0.9046 | 0.7964 | 0.1082 | 0.0592 |
| Dual-cue λ=0.25 | 0.8808 | 0.7871 | 0.0937 | **0.0820** |
| **Dual-cue λ=0.50** | **0.9165** | **0.7759** | **0.1406** | 0.0579 |
| Dual-cue λ=0.75 | 0.7815 | 0.8909 | -0.1094 | 0.0549 |

λ=0.50将full return margin从PF的0.023提升到0.141，同时降低B→A2相似度；过强semantic prior
(0.75)反而退化，说明视觉状态与prompt意图必须平衡，而非简单用prompt取代visual retrieval。

## 3. A-B-A整体质量

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.96722 | 0.94320 | **0.62896** | 0.70079 | **0.98364** | 0.88889 |
| Dual-cue λ=0.25 | **0.96982** | **0.94392** | **0.63241** | 0.69405 | 0.98317 | 0.86667 |
| **Dual-cue λ=0.50** | 0.96798 | 0.94335 | 0.62848 | **0.70275** | 0.98349 | 0.88889 |
| Dual-cue λ=0.75 | 0.96664 | 0.94204 | 0.63183 | 0.69344 | 0.98304 | **0.91111** |

λ=0.50在保持PF dynamic degree的同时提高subject/background/imaging，aesthetic/motion仅轻微下降，
并获得最强scene-return margin。它是当前最平衡的scene-recall候选。

## 4. 与PF的核心区别

PF：

```text
offline head temporal classification
→ fixed sink/middle/recent retention policy
```

Dual-Cue CEMR：

```text
bounded clean episodic archive
→ query-independent coverage maintenance
→ online visual-continuity + semantic-intent retrieval
→ independent historical attention
→ confidence/alignment-controlled fusion
```

PF没有独立archive、content retrieval、prompt-guided scene selection或独立memory softmax。因此方法
不再依赖“改进PF分类”作为创新。

## 5. 与相关工作的区别

- LongLive-RAG：主要用latent descriptor检索并拼入native attention；Dual-Cue CEMR使用visual+
  prompt双query、bounded coverage archive和decoupled readout。
- Echo-Forcing：使用prompt scene routing与scene recall frame；Dual-Cue CEMR不复制其场景控制器、
  difference-aware decay或RoPE jump，而在通用连续生成中组合visual state和prompt intent进行
  frame-level retrieval。
- IAMFlow：使用LLM/VLM entity registry；本方法不使用额外模型，不做entity-ID memory。
- MemRoPE：常驻EMA memory slots；本方法显式维护可检索完整帧事件。

## 6. 当前可写的论文主张

> In long autoregressive video generation, the current visual state alone can retrieve the wrong episode
> after a scene transition, while prompt-only retrieval ignores generation continuity. We introduce a
> training-free dual-cue episodic memory that balances visual continuity and semantic intent over a bounded
> coverage archive, and reads selected full-frame history through a decoupled attention branch.

保守标题：

> **Dual-Cue Episodic Memory Readout for Training-Free Long-Horizon Video Diffusion**

## 7. 剩余门槛

1. seed1/2的A-B-A return margin复验；
2. 人工确认dual-cue没有新增flashback、肢体错误或背景硬切；
3. 32-prompt普通长视频中保留CEMR的温和正趋势；
4. 非PF backend验证；
5. 报告计算/存储成本。

当前已满足“初步可行、与PF有明确方法区别、可以形成论文故事”的最低条件，但尚未满足正式
投稿的完整证据标准。
