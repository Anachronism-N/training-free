# Historical KV Cache 的可行性与 LifeCache 后续研究方向

> 2026-07-16  
> 本文用于统一当前项目判断：历史 KV cache 记忆在理论与已有工作中均具有充分可行性；LifeCache 当前尚未取得一般可用结果，不能被解释为“历史 KV 无效”，而应被解释为当前实现、记忆单位、位置编码、注入方式和评估协议尚未完全闭环。

---

## 1. 核心立场

LifeCache 不应因为前几轮负结果而放弃历史 KV cache 路线。

大量长上下文、长视频生成和世界模型工作都在历史 KV 上进行保留、压缩、检索、重编码、重组或重新注入。它们共同说明：

1. 历史 K/V 确实携带已经生成内容的可复用状态；
2. 长期历史不必全部保留，可以压缩为有限预算；
3. 历史 K/V 可以在后续 attention 中重新参与计算；
4. 位置编码、空间结构、head 角色和注入预算是成败关键；
5. “历史 KV 可行”不等于“任意 sparse KV 拼接都有效”。

因此，当前项目需要区分两个命题：

```text
命题 A：历史 KV cache 能否作为长期记忆载体？
结论：理论上可行，已有大量工作和代码实现支持。

命题 B：当前 LifeCache v3 的 sparse evict-compress-recall-append 是否已经有效？
结论：尚未得到稳定、一般化的正结果。
```

不能用命题 B 的阶段性失败否定命题 A。

---

## 2. 仓库中已有工作的共同证据

当前仓库 `third_party/` 下已经包含多类历史 KV 方法。它们虽然实现形式不同，但都证明“对历史 KV 做操作”是一条成立的技术路线。

### 2.1 Smart Retention：在淘汰前选择性保留

代表实现：

- `third_party/Pyramid-Forcing/`
- `third_party/Forcing-KV/`
- `third_party/DeepForcing/`
- `third_party/RollingForcing/`

这些方法的核心不是保存全部历史，而是根据 layer/head 的功能差异，选择需要长期保留的 token、frame 或 cache region。

其共同思想是：

```text
recent window
+ stable / anchor / layout history
+ head-specific cache allocation
```

这说明历史 KV 的价值并不是均匀分布的，正确做法通常是按 head、layer、时间尺度和内容结构分配预算。

### 2.2 Structured Recall：召回完整 frame、chunk 或 scene block

代表实现：

- `third_party/LongLive-RAG/`
- `third_party/Echo-Forcing/`
- 仓库审查中涉及的 WorldKV 类设计

这些方法不会简单地把所有历史 token 放回当前窗口，而是：

```text
历史 frame / chunk 写入长期存储
→ 构建 descriptor 或索引
→ 检索相关历史单元
→ 在固定或受控预算下重新注入 attention
```

这直接支持 LifeCache 的长期召回设想。真正需要检验的是：

- memory unit 应该是完整 frame、patch block 还是 sparse token；
- 检索时应该保留多少结构；
- 注入时如何重新编码位置；
- 哪些 heads 应该访问 memory。

### 2.3 Dynamic RoPE：历史 K 可以重新赋予合法位置

代表实现：

- `third_party/MemRoPE/`
- `third_party/Echo-Forcing/`
- `third_party/Pyramid-Forcing/` 中的位置重映射机制

这些工作说明历史 K 不必永久绑定原始 RoPE 相位。更合理的方式是保存 raw/pre-RoPE K，并在使用时根据当前上下文重新编码。

其核心原则是：

```text
存储 raw K
+ 保存真实 temporal / spatial coordinates
+ recall 时重新施加合法的 3D RoPE
```

因此，LifeCache 的 pre-RoPE bank 方向是正确的。当前需要继续保证：

- K、V、frame index、spatial index 始终严格对齐；
- sparse token 的位置编码与原始 full-grid RoPE 数值一致；
- query、recent K 和 historical K 位于兼容的坐标系；
- 不发生二次 clamp、伪造 grid 或位置 metadata 丢失。

### 2.4 Historical Access：历史 KV 可以非连续访问

代表实现：

- OmniMem 类 per-head scattered access
- LongLive-RAG 的历史 frame 检索
- Forcing-KV 的 head-specific cache views

这些方法说明，attention 并不要求历史 K/V 必须是连续的本地窗口。非连续历史访问是可行的，但需要合理的数据布局、head 路由和预算约束。

这进一步说明 LifeCache 的问题不在于“历史 token 被取回”本身，而在于当前取回内容是否：

- 正确；
- 有结构；
- 有位置；
- 有因果相关性；
- 被合适的 heads 使用。

---

## 3. 为什么当前负结果不能否定历史 KV 路线

截至当前版本，项目已经发现过多个会直接使实验失效的问题：

1. timestep filter 曾过滤掉全部 eviction payload；
2. layer-majority routing 曾让 layer 29 使用 `GENERIC` role，导致 recall budget 为 0；
3. `RecallResult` 曾丢失 frame/spatial metadata；
4. sparse K 曾使用错误或伪造的位置；
5. 配置中 `recall_top_tokens=32`，实际却由 region budget 召回 512；
6. `region_bias_beta` 曾配置为非零，但没有进入真实 attention；
7. 早期 QK diagnostic 比较过不一致的 RoPE 空间；
8. 早期实验主要使用 MP4 文件大小判断效果；
9. A-B-A 场景切换曾依赖单条自然语言 prompt，而非显式 prompt schedule；
10. 当前 full-frame oracle 仍主要停留在配置层，还没有形成完整独立代码路径。

最新版本第一次从 trace 上证明：

```text
BANK 实际写入
RECALL 实际选中
COMPOSE 实际追加 recalled tokens
attention 的 K/V 长度实际发生变化
```

因此，之前许多“无提升”结果并不是在一个完全正确执行的 LifeCache 上得到的。

当前合理结论应为：

> 现有实验尚未证明历史 KV recall 无效，只能说明早期实现和当前 sparse recall 版本尚未形成稳定、一般可用的收益。

---

## 4. LifeCache 的研究假设应如何重新表述

建议将项目核心假设从宽泛的：

```text
召回历史 KV 可以提升长视频生成。
```

收敛为更严格、可验证的分层假设。

### H1：历史 KV 中存在可复用信息

给定同一场景 A 的 clean historical K/V，在返回 A 时，正确历史 memory 应比错误场景 memory、随机 memory 和 shuffled-V memory 更有帮助。

### H2：结构化 memory 优于无结构 sparse token

完整 frame 或规则 patch block 保留空间关系，因此应比来自任意位置的 global sparse top-k 更稳定。

### H3：位置重编码是必要条件

历史 K 必须与当前 query/recent K 位于兼容的 RoPE 坐标系；错误位置会抵消 memory 内容价值。

### H4：历史 memory 只对部分 heads 有益

layout、anchor、identity 或 memory heads 更可能使用远程历史；motion/local heads 可能被旧信息污染。

### H5：固定预算优于无约束追加

简单 append 会改变 softmax 竞争和 active token 总量；更合理的系统应在 recent 与 historical token 之间重新分配固定预算。

### H6：检索应发生在正确 memory unit 上

检索完整 frame/chunk，再在内部压缩，通常比直接从所有历史 token 中全局 top-k 更合理。

---

## 5. 后续路线：不是证明“能不能做”，而是找到“怎样做才有效”

当前不再将主要问题定义为：

```text
历史 KV 是否可行？
```

而应定义为：

```text
在 Self-Forcing 上，哪一种历史 KV 表示、位置编码、访问方式和预算分配能够稳定产生收益？
```

后续推进分为四个阶段。

---

## 6. Stage 1：完成可信 sparse-v3 基线

当前 sparse-v3 已经第一次实际运行，但仍需要完成以下闭环。

### 6.1 配置真实性

必须保证：

```text
configured recall tokens
= effective recall tokens
= actual recalled tokens
```

`recall_top_tokens` 应成为真实上限，不再被默认 `RegionBudget(recall=512)` 静默覆盖。

### 6.2 RoPE 单一坐标规则

需要明确选择一种方案。

方案 A：

```text
native query/recent K 保持原始绝对坐标
historical K 映射到 current-TR+1 之后的合法绝对位置
```

方案 B：

```text
query、recent raw K、historical raw K
全部重新映射到统一 local coordinate
```

不能先算相对位置，再在 sparse RoPE 内二次截断到 `[0, TR-1]`。

### 6.3 Strict correctness

当 recalled token 缺少 frame/spatial metadata 时，应：

- strict 模式直接报错；或
- 删除 invalid recalled tokens 并退回 native recent。

不能继续把未旋转 historical K 输入 post-RoPE attention。

### 6.4 真实 attention 诊断

必须在完成 RoPE 后计算真实 scaled dot-product attention mass：

```text
memory attention mass
recent attention mass
per-head memory mass
WAVE vs non-WAVE memory mass
```

当前 pre-RoPE cosine proxy 只能作为检索分数，不能证明真实 attention 使用程度。

---

## 7. Stage 2：实现 Clean Full-Frame Oracle

这是下一阶段最重要的实验，因为它能排除：

- compression 错误；
- retrieval 错误；
- sparse selection 错误；
- 空间结构损失；
- noisy eviction memory。

### 7.1 Oracle memory

从 clean-context forward 中直接捕获 layer 29 的一整个历史 frame：

```text
1560 raw K
1560 aligned V
完整 H×W 空间顺序
明确 source frame
明确 prompt segment
```

### 7.2 Deterministic injection

在 A2 阶段直接注入 A1 memory，不经过普通 `recall_tokens()`：

```text
A1 clean full-frame memory
→ 原生 full-grid 3D RoPE
→ [historical frame | recent window]
→ attention
```

### 7.3 必要对照

必须同时运行：

| 实验 | 目的 |
|---|---|
| Native | 基线 |
| Correct A1 memory | 检验正确历史内容 |
| Wrong B memory | 检验场景特异性 |
| Random memory | 排除增加 token 数的影响 |
| Correct K + shuffled V | 检验 K/V 对齐 |
| Zero V | 检验历史 value 的真实贡献 |
| Fixed-budget full frame | 排除 append 总预算变化 |

只有正确 A1 memory 稳定优于错误和随机对照，才能证明历史 KV 的内容特异收益。

---

## 8. Stage 3：比较 memory unit，而不是继续盲调 top-k

Oracle 成功后，依次比较：

```text
M0：完整历史 frame
M1：规则空间 patch grid
M2：连续 patch blocks
M3：frame 内 top-k，但保留真实坐标
M4：当前 global sparse token top-k
```

重点回答：

> 性能下降从哪一种压缩粒度开始出现？

这比继续做 `32/64/128/512` 的 token budget sweep 更有信息量。

推荐先检索 frame/chunk，再在选中的 memory 内压缩，而不是从全部历史 token 中直接全局 top-k。

---

## 9. Stage 4：形成 Hybrid LifeCache

如果 full-frame oracle 和 structured memory 有效，最终方法建议为：

```text
Native Recent Window
+ Head-aware Smart Retention
+ Archival Structured KV Memory
```

### 9.1 Recent Window

负责短期运动、局部细节和连续性。

### 9.2 Smart Retention

借鉴 Pyramid-Forcing / Forcing-KV：

- 对稳定、layout、anchor heads 保留更长历史；
- motion/local heads 保持短窗口；
- 按 layer/head 分配预算。

### 9.3 Archival Memory

借鉴 LongLive-RAG / Echo-Forcing / WorldKV 类设计：

- 只保存被 recent/retention 淘汰的更久历史；
- memory unit 为 frame、patch block 或 chunk；
- 在显式 scene revisit 或高相似度时召回；
- 采用固定预算替换而非无限 append。

最终目标不是让所有历史 memory 始终参与 attention，而是：

```text
当前需要什么历史
→ 只召回对应历史
→ 只开放给需要它的 heads
→ 保持总 active budget 可控
```

---

## 10. Go / No-Go 应如何解释

即使最终 full-frame oracle 在 Self-Forcing 上没有显著收益，也不能得出：

```text
历史 KV cache 路线理论上不可行。
```

更准确的结论只能是：

```text
在当前 Self-Forcing backbone、当前任务和当前注入层上，
历史 KV 不是主要性能瓶颈，或需要不同层级/表示形式。
```

### Go

继续 historical KV recall 的条件：

- clean full-frame correct memory 在多个 seed 上提升场景恢复；
- wrong/random/shuffled controls 不产生同等提升；
- attention mass 显示目标 heads 实际使用 memory；
- 提升不以明显运动冻结、画质下降为代价。

### Partial Go

若 full-frame 有效、sparse 无效：

- 继续 structured frame/patch memory；
- 不再以 arbitrary sparse token 作为主要 memory unit。

若 smart retention 有效、recall 较弱：

- 采用 retention 为主、archival recall 为辅的 hybrid。

### Backbone-specific No-Go

若 clean full-frame correct memory 在多层、多 seed 和受控 schedule 下仍与错误 memory 无法区分：

- 暂停在 Self-Forcing 上继续堆叠 recall scorer；
- 转向 Causal-Forcing、latent memory 或显式 scene/world-state memory；
- 保留 historical KV 的工程与研究结论，而不是否定整个方向。

---

## 11. 评估原则

从本阶段开始，不再使用 MP4 文件大小作为方法优劣的主要依据。

至少记录：

```text
latent max/mean difference
A1-A2 DINO/CLIP similarity
主体身份一致性
背景布局恢复
关键物体重新出现率
错误场景泄漏
temporal flicker
dynamic degree
真实 memory attention mass
paired human preference
```

必须使用显式 A-B-A prompt schedule，而不是依赖一条长 prompt 自发完成场景切换。

每个关键实验至少运行 3 个 seeds。

---

## 12. 近期具体提交计划

### Commit A：Sparse v3 truthfulness

```text
fix: enforce sparse recall budget and RoPE invariants
```

- `recall_top_tokens` 真正生效；
- trace configured/effective/actual budget；
- 删除 temporal double clamp；
- invalid metadata fail-fast；
- `max_frame_distance` 贯穿调用链。

### Commit B：Attention diagnostics

```text
feat: add post-RoPE per-head memory attention diagnostics
```

- 采样真实 attention logits；
- 输出 recall/recent mass；
- 输出 WAVE/non-WAVE 分组；
- 记录 memory 对 attention output 的增量。

### Commit C：Full-frame oracle

```text
feat: implement clean deterministic full-frame KV oracle
```

- 解析 oracle 配置；
- clean-context capture；
- 完整 frame raw-K/V；
- deterministic A1→A2 injection；
- append 与 fixed-budget 两种模式。

### Commit D：Controlled A-B-A benchmark

```text
feat: add segmented A-B-A schedule and oracle controls
```

- 显式 prompt segment；
- correct/wrong/random/shuffled/zero-V controls；
- 自动导出指标与实验 manifest。

---

## 13. 最终结论

历史 KV cache 作为长期记忆载体是理论上成立、工程上已有大量成功先例的方向。

LifeCache 当前的问题不是“这个方向是否可行”，而是：

```text
应该保存什么单位？
应该如何保持时空结构？
应该如何重新编码位置？
应该何时召回？
应该让哪些 heads 访问？
应该如何在固定预算中注入？
```

因此，项目后续不应因早期负结果退回到“是否值得继续”的争论，而应转向严格的机制识别：

```text
可信 sparse baseline
→ clean full-frame oracle
→ structured memory unit
→ head-aware fixed-budget injection
→ hybrid retention + archival recall
```

当前优先级最高的任务仍然是：

> 实现一个真正独立、干净、确定性注入的 full-frame KV oracle，并使用正确场景、错误场景和 shuffled-V 等对照，证明历史 KV 的内容特异因果作用。
