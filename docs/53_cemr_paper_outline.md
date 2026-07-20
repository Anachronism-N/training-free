# CEMR 论文草案结构

> 工作标题：**Coverage-Aware Episodic Memory Readout for Training-Free Long-Horizon Video Diffusion**

## Abstract 核心逻辑

现有training-free长视频方法主要优化有限KV cache中的history retention；但被淘汰的远期视觉
事件无法按内容重新访问。直接把历史KV拼回self-attention又会改变局部上下文的softmax竞争，
造成冻结、重影和错误回放。本文提出CEMR：以clean完整视觉帧为episodic memory单位，用
query-independent coverage维护固定archive预算，再按当前query检索远期事件，通过独立memory
attention读出并以置信度控制融合。方法无需训练，也不依赖额外VLM/LLM。初步30秒结果相对PF
在subject、background、imaging、motion和dynamic五维提升，同时保持dynamic degree；匹配的
32-prompt评估正在执行。

## 1. Introduction

### 观察

- AR视频扩散的recent cache适合局部运动；远期一致性需要更长历史。
- PF等方法解决“有限cache中保留什么”，但不是content-addressable memory。
- LongLive-RAG类retrieval将历史拼入原softmax，存在竞争和预算问题。
- 历史memory不是越多越好：错误历史会导致flashback、双轮廓和动作相位污染。

### 核心问题

如何在不训练检索器、不改变原生局部attention、不无限增长memory的条件下，维护并读取远期
视觉事件？

### 贡献

1. Query-independent coverage-aware bounded archive；
2. Clean full-frame episodic memory，保持K/V空间结构；
3. Query-conditioned remote-event retrieval与recent exclusion；
4. Decoupled memory attention和confidence/alignment fusion；
5. Correct/shuffled/abstain、archive policy、position和gate系列机制消融。

## 2. Related Work

- AR video diffusion and Self/Causal Forcing；
- KV retention：PF、RollingForcing、Forcing-KV；
- Retrieval memory：LongLive-RAG；
- Scene recall/forgetting：Echo-Forcing；
- Position reassignment：MemRoPE；
- Entity memory：IAMFlow。

必须明确：CEMR不声称首次full-frame retrieval、scene recall或entity memory。创新位于bounded
coverage archive + decoupled episodic readout的组合和实证分析。

## 3. Method

### 3.1 Native recent/cache backend

CEMR可叠加在PF或其他recent-cache backend上。Native branch完全保持原样。

### 3.2 Clean episodic archive

每个完成block的clean K/V按完整frame写入。Archive三种预算：

- archive budget：长期保留帧数；
- scan budget：descriptor检索范围；
- readout budget：实际读取帧数。

### 3.3 Coverage-aware archive maintenance

Descriptor为training-free activation statistics。Archive超限时使用endpoint-preserving greedy
k-center，优化全历史覆盖，而非当前query相关性。

### 3.4 Query-conditioned retrieval

排除recent gap后，对archive frame descriptor进行query匹配，选择top-k完整帧。候选帧保持
完整空间K/V和时间interval。

### 3.5 Decoupled readout

```text
x_native = Attention_native(Q, K_native, V_native)
x_mem    = Attention_memory(Q_content, K_archive, V_archive)
w        = gate * confidence * alignment
x        = (1-w)x_native + w x_mem
```

Independent memory attention避免改变native local softmax分母。

### 3.6 Uncertainty and controls

- absolute confidence；
- top1/top2 margin；
- retrieval entropy；
- wrong/least-similar history；
- shuffled V；
- abstain。

自然单场景中使用soft confidence；hard abstention主要用于scene-switch和false-recall。

## 4. Experiments

### 4.1 Baselines

- SF native；
- PF；
- PF + uniform archive；
- CEMR coverage archive；
- LongLive-RAG（若能公平复现）；
- Echo scene-switch任务。

### 4.2 Metrics

VBench-Long：subject/background/aesthetic/imaging/motion/dynamic；另加：

- block boundary；
- luminance drift；
- latency/VRAM/CPU memory；
- human artifact review。

### 4.3 Main 30s preliminary result

| Method | Subject | BG | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.93283 | 0.92584 | **0.60863** | 0.62708 | 0.96791 | 0.95556 |
| CEMR | **0.93693** | **0.93032** | 0.60581 | **0.64647** | **0.97218** | **0.97778** |

### 4.4 Ablations

1. uniform vs coverage archive；
2. gate 0.03/0.05/0.075/0.10；
3. raw content readout vs local-grid RoPE；
4. correct vs shuffled-V vs abstain；
5. no-routing/PF-static/confidence/functional；
6. no-CFG/fixed-CFG/dynamic-CFG。

### 4.5 Negative findings

- 高gate造成保守生成与dynamic下降；
- dynamic head routing信号可分，但没有稳定全面优势；
- global CFG成本高且整体不如no-CFG；
- 过强hard abstention退化为no-memory；
- local-grid memory RoPE没有优于position-decoupled content readout。

## 5. Limitations

- 目前主结果仍依赖PF作为recent backend；需移植SF/CF；
- 单prompt中的wrong history不是真正wrong-scene；需A-B-A；
- 当前descriptor是activation statistics，语义能力有限；
- full-frame archive有CPU/GPU存储开销；
- 需要更多seed和人工评审。

### 4.6 Matched 32-prompt result

| Method | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.97761 | 0.96548 | 0.64681 | 0.72195 | 0.98718 | **0.58750** |
| CEMR | **0.97882** | **0.96671** | **0.64739** | **0.72667** | **0.98747** | 0.56250 |

CEMR在5个非dynamic维度上小幅提升，但dynamic下降0.025。论文必须把这描述为受限的质量-动态
权衡，并通过多seed与人工review排除冻结因素。

## 6. Submission Gate

只有满足以下条件才进入投稿：

1. 匹配32-prompt PF vs CEMR保持主要维度提升；
2. seed1/2复验不反转；
3. 人工评审无新增严重伪影；
4. scene-switch正确历史优于wrong/shuffled/abstain；
5. 至少一个非PF backend上成立。
