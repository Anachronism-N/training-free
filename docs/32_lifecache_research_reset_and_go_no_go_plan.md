# LifeCache 研究重置与 Go / No-Go 推进方案

> 更新时间：2026-07-15  
> 当前基线：Self-Forcing + LifeCache v3 Phase 0  
> 相关记录：`docs/27`–`docs/31`  
> 目标：在多轮迭代仍未获得一般可用结果的情况下，停止无边界调参，用可证伪、可复现、阶段化的实验决定继续、转向或终止当前技术路线。

---

## 1. 当前阶段的诚实判断

LifeCache 已经历多轮迭代，先后尝试了：

- eviction 后的 KV bank；
- QK proxy compression；
- pre-RoPE K 存储；
- sparse token recall；
- RoPE remap；
- timestep filtering；
- layer / budget / random recall ablation；
- head-role routing；
- A-B-A scene revisit prompt；
- absolute frame 与 spatial position sidecar；
- sparse 3D RoPE。

截至 `docs/31_lifecache_v3_phase0_results.md`，最新 v3 Phase 0 输出仍未显示稳定、可重复、跨 prompt 的明显收益。部分结果甚至与 native Self-Forcing 完全一致。

因此，当前不能继续把工作模式维持为：

```text
发现一个问题
→ 修改一处代码
→ 跑少量视频
→ 根据文件大小或主观观感继续猜测
→ 再增加一个机制
```

这会不断增加实现复杂度，却无法判断：

1. 历史 KV 本身是否对当前 backbone 有用；
2. 有用的是 retention、recall，还是 prompt conditioning；
3. 无收益来自实现错误、评价不敏感，还是机制不适配；
4. 哪些配置真正生效；
5. 当前路线是否值得继续投入。

从现在开始，LifeCache 应进入 **研究重置阶段**。

---

## 2. 重置后的核心原则

### 2.1 一次只回答一个问题

后续实验不再同时改变：

```text
memory content
retrieval
compression
RoPE
attention bias
head routing
budget
prompt
```

每次实验只允许一个主要变量发生变化。

### 2.2 Oracle 优先于完整方法

在实现复杂 retrieval、compression 和 semantic memory 前，先验证：

> 当模型获得一个完全正确、完整、结构化、位置合法的历史 KV frame 时，生成是否会得到可测量收益？

若 oracle 都无效，则不应继续优化 sparse recall。

### 2.3 评价优先于调参

没有稳定 benchmark 和自动指标时，不允许得出“有提升”或“无提升”的结论。

### 2.4 所有实验必须可复现

每个结果必须能追溯到：

```text
commit SHA
config 文件
prompt schedule
seed
输出目录
trace 文件
评价结果
运行时间与峰值显存
```

### 2.5 设置停止条件

当前路线必须存在明确的 No-Go 条件，避免因为“也许下一个修复会有效”而无限延长。

---

## 3. 立即冻结的内容

在完成本计划的 Stage 0–2 前，暂停以下工作：

- 新增更多 memory region；
- 新增 agent / RAG / world-state 模块；
- 扩展到更多 layers；
- region bias sweep；
- recall budget 大范围 sweep；
- random recall；
- 更多 compression heuristic；
- 在 Self-Forcing 与 Causal-Forcing 间反复迁移；
- 仅凭 MP4 文件大小判断效果；
- 仅用单 seed 得出结论；
- 在 correctness 未通过时运行长视频大规模实验。

当前只保留两个开发分支：

```text
A. correctness / instrumentation
B. clean full-frame oracle
```

---

## 4. Stage 0：冻结可信基线

### 4.1 建立正式 baseline

创建独立配置，不再沿用名称与行为不一致的 `lifecache_v2_optimized.yaml`：

```text
configs/lifecache/v3_native_control.yaml
configs/lifecache/v3_sparse_recall.yaml
configs/lifecache/v3_full_frame_oracle.yaml
```

基线固定为：

```text
Backbone: Self-Forcing
Frames: 120
Resolution: 固定
Layers: native / layer 29
Seeds: 0, 1, 2
Prompts: 固定 benchmark suite
```

### 4.2 两个等价性检查

必须首先通过：

```text
E0：LIFECACHE_ENABLE=0 与 native 输出完全一致
E1：LIFECACHE_ENABLE=1，但 recall_budget=0，与 native 输出完全一致
```

要求：

- latent / decoded frame 在允许的数值误差内一致；
- 运行路径和配置被 trace 明确记录；
- 不以视频文件大小作为等价性证明。

### Stage 0 通过条件

```text
[ ] native baseline 可稳定复现
[ ] 三个 seeds 输出与指标记录完整
[ ] disabled control 通过
[ ] zero-budget control 通过
[ ] 所有配置有 effective-config trace
```

未通过时，不进入后续阶段。

---

## 5. Stage 1：Correctness Gate

Stage 1 的目标不是改善视频，而是证明一个 recalled token 在全链路中始终是同一个 token。

### 5.1 Metadata 不变量

所有 memory token 必须满足：

```python
len(k) == len(v)
len(k) == len(token_indices)
len(k) == len(frame_positions)
len(k) == len(spatial_positions)
len(k) == len(source_set_ids)
```

以及：

```python
frame_positions.min() >= 0
spatial_positions.min() >= 0
spatial_positions.max() < grid_h * grid_w
```

需要覆盖：

- compression top-k；
- min-score filtering；
- recall top-k；
- clone / crop；
- CPU / GPU transfer；
- random control；
- active-cache composition。

### 5.2 Recall metadata 全链路

必须完成：

```text
TokenSet
→ RecallResult
→ recall:view
→ ActiveCacheView
→ attention remap
```

`CacheRegion.RECALL` 中禁止出现：

```text
frame_position = -1
spatial_position = -1
synthetic fallback position
```

发现缺失时直接报错，不再静默继续推理。

### 5.3 Sparse 3D RoPE parity

增加单元测试：

> 对完整 frame 先执行原生 RoPE，再选取 sparse tokens；结果必须与对同一组 raw sparse tokens 使用 token-wise 3D RoPE 的结果一致。

覆盖：

- 单帧；
- 多帧；
- 随机 sparse tokens；
- 连续 patch block；
- 不同 H/W；
- fp32；
- bf16。

### 5.4 配置真实性

以下配置必须在 trace 中同时记录：

```text
configured value
effective value
actual runtime value
```

至少包括：

- recall_top_tokens；
- recall_top_sets；
- max_frame_distance；
- enabled_layers；
- head-role count；
- region_bias_beta；
- capture policy；
- actual recalled token count。

未接入 attention 的配置不得保持“看似可用”：

```text
region_bias_beta > 0 but bias path disabled
```

应直接 warning 或 fail-fast。

### Stage 1 通过条件

```text
[ ] 所有 metadata 单元测试通过
[ ] recall:view 无非法位置
[ ] sparse/full RoPE parity 通过
[ ] relative distance 映射合法
[ ] effective recall budget 等于配置
[ ] distance filter 确实生效
[ ] head roles 正确加载
[ ] CI / smoke test 可自动运行
```

---

## 6. Stage 2：Clean Full-Frame Oracle

这是下一阶段最重要的实验。

### 6.1 Oracle 要回答的问题

> 对第二次出现的场景 A，注入第一次场景 A 的完整、干净、对齐的历史 frame KV，是否能显著改善场景、主体或物体一致性？

### 6.2 Oracle 必须绕过的模块

Oracle 第一版不使用：

- eviction capture；
- QK retrieval；
- sparse top-k；
- compression；
- semantic descriptor；
- bank pruning；
- region bias；
- rho decay。

它只测试：

```text
clean full-frame raw K/V
+ deterministic injection
+ native full-grid RoPE
```

### 6.3 捕获方式

在 clean-context refresh 中直接捕获 layer 29 的：

```text
raw / pre-RoPE K
aligned V
完整 frame grid
capture frame index
segment id
```

优先从当前 forward 的 raw K/V hook 捕获，而不是从发生过 roll 的 cache 中推断。

### 6.4 注入方式

第一轮：

```text
O1：append one historical frame
```

确认 memory 是否有任何因果作用。

第二轮：

```text
O2：replace one recent frame
20 recent + 1 historical = native 21-frame budget
```

确认在固定预算下是否仍有收益。

### 6.5 对照组

必须同时运行：

| ID | 设置 | 目的 |
|---|---|---|
| O0 | Native | 基线 |
| O1 | 正确 A1 full-frame，append | 检查历史 memory 是否产生作用 |
| O2 | 正确 A1 full-frame，fixed budget | 检查公平预算下的收益 |
| O3 | 正确 A1，仅 stable heads | 检查 head-specific access |
| O4 | 错误场景 full-frame | 排除单纯增加 token 的作用 |
| O5 | A1 K + shuffled V | 检查 K/V 内容对齐是否必要 |
| O6 | 相同 token 数的 random memory | 排除容量效应 |

每项至少 3 seeds。

### Stage 2 Go 条件

满足以下至少两项，并且跨 seeds 一致：

- A1–A2 scene similarity 明显高于 native；
- 主体 identity similarity 提升；
- 关键物体回归率提升；
- garden leakage / stale scene 降低；
- 正确 memory 优于 wrong-scene 和 random controls；
- attention 中存在稳定、可解释的 memory mass。

### Stage 2 No-Go 条件

以下情况应停止继续优化 sparse retrieval：

```text
正确 full-frame oracle 在 3 个 seeds、多个 prompts 上
均不优于 native，或与错误 / random memory 无差别。
```

此时应转向：

```text
smart retention
latent memory
prompt / condition memory
world-state memory
或新的 backbone
```

---

## 7. Stage 3：确定 memory unit

只有 Stage 2 Oracle 成立后，才比较 memory 表示形式：

```text
M0：完整 frame
M1：规则 spatial grid
M2：连续 patch block
M3：frame-level compressed block
M4：当前 arbitrary sparse top-k
M5：random sparse control
```

目标是回答：

> 历史记忆需要保留多少空间结构？

若完整 frame 有效、sparse top-k 无效，则后续方法应以 frame / patch block 为基本 memory unit，而不是继续优化散点 token 检索。

---

## 8. Stage 4：确定 retention 与 recall 的关系

对比：

```text
R0：native local window
R1：Pyramid-style smart retention
R2：archival recall only
R3：smart retention + archival recall
```

### 决策规则

- R1 有效、R2 无效：主线转为 smart retention；
- R2 有效：继续 structured recall；
- R3 最优：形成 Hybrid LifeCache；
- 全部无效：停止 Self-Forcing KV-memory 路线。

推荐最终方向只有在证据支持时才成立：

```text
Native Recent Window
+ Head-aware Smart Retention
+ Archival Structured Memory
```

---

## 9. Benchmark 与评价重建

### 9.1 Prompt schedule

不再仅依赖一条自然语言长 prompt 描述“后来返回”。

实现显式 segment schedule：

```text
A1：frames 0–29
B：frames 30–69
A2：frames 70–119
```

每个 segment 使用独立 condition，边界与 generation block 对齐。

### 9.2 Prompt suite

至少包括：

1. 蓝色厨房 / 红杯 → 花园 → 蓝色厨房 / 红杯；
2. 同一只狗离开画面 → 新场景 → 同一只狗返回；
3. 实验室 → 走廊 → 原实验室；
4. 红色房间 → 蓝色房间 → 红色房间；
5. 主体换动作但身份不变；
6. 物体状态改变后回访，检查 stale memory。

### 9.3 指标

主要指标：

- A1–A2 DINO / CLIP scene similarity；
- subject identity similarity；
- background similarity；
- object presence / color consistency；
- scene leakage；
- temporal flicker；
- dynamic degree；
- paired human preference。

诊断指标：

- actual memory attention mass；
- recent attention mass；
- per-head memory mass；
- recall source-frame distribution；
- mapped temporal distance；
- invalid metadata count；
- actual active token count。

MP4 文件大小只作为编码信息，不作为质量指标。

---

## 10. 实验记录规范

每个实验必须创建：

```text
runs/<experiment_id>/
├── config.yaml
├── git_commit.txt
├── environment.txt
├── prompts.yaml
├── seeds.txt
├── trace.jsonl
├── metrics.json
├── summary.md
└── videos/
```

`summary.md` 统一包含：

```text
研究问题
唯一变量
基线
预期结果
实际结果
自动指标
主观观察
失败原因
下一步决定
```

禁止只留下视频文件而没有配置和结论。

---

## 11. 分支与提交管理

建议分支：

```text
main
├── lifecache/correctness-gates
├── lifecache/full-frame-oracle
├── lifecache/prompt-schedule
└── lifecache/evaluation
```

每个提交只解决一个问题：

```text
test: add metadata invariants
fix: propagate recall positions end to end
test: add sparse/full RoPE parity
fix: enforce effective recall configuration
feat: add clean full-frame oracle
feat: add segmented A-B-A schedule
feat: add memory evaluation metrics
```

不要再把 runnable、metadata、RoPE、oracle 和实验脚本合并到一个大提交中。

---

## 12. 两周内的建议执行顺序

### 第 1–2 天：冻结与清理

- 新建 v3 独立配置；
- 固定 native baseline；
- 补全 effective-config trace；
- 整理已有 runs 与 commit 对应关系。

### 第 3–5 天：Correctness Gate

- TokenSet / RecallResult metadata 测试；
- sparse/full RoPE parity；
- disabled 与 zero-budget control；
- 删除所有 silent fallback。

### 第 6–8 天：Full-Frame Oracle

- clean raw K/V capture；
- deterministic injection；
- append / replace modes；
- wrong-scene / shuffled-V controls。

### 第 9–10 天：Prompt Schedule 与指标

- 显式 A-B-A condition 切换；
- A1–A2 similarity；
- identity / object / leakage 评价；
- sampled real attention mass。

### 第 11–12 天：三 seed 正式实验

只运行 O0–O6，不增加新机制。

### 第 13–14 天：Go / No-Go 评审

输出一份决策文档：

```text
GO structured recall
GO smart retention
GO hybrid
NO-GO Self-Forcing KV memory
```

---

## 13. 最终决策表

| 观察结果 | 结论 | 下一步 |
|---|---|---|
| Full-frame oracle 稳定提升 | 历史 KV 有效 | 推进 structured compression / retrieval |
| Full-frame 有效，sparse 无效 | 空间结构必要 | 放弃 arbitrary sparse token，使用 frame / patch block |
| Smart retention 有效，oracle recall 无效 | 连续保留优于回收 | 转向 Pyramid / Forcing-KV 路线 |
| Hybrid 最优 | 两类机制互补 | 构建 Hybrid LifeCache |
| 正确 memory 与 wrong/random 无差别 | memory 内容未被有效使用 | 检查 attention gating；若确认后仍无效则停止 |
| Oracle、retention 均无效 | Self-Forcing KV memory 不适配 | 转向 latent / world-state memory 或更换 backbone |

---

## 14. 本阶段唯一优先研究问题

接下来不要同时回答十个问题。

本阶段只回答：

> **在受控 A-B-A 生成中，给 Self-Forcing 一个完整、干净、位置合法、内容正确的历史 frame K/V，是否能比 native 更好地恢复第二次场景 A？**

在该问题获得明确答案之前，不再增加新的 memory 机制。

这将决定 LifeCache 是继续做 structured recall、转向 smart retention，还是停止当前技术路线。