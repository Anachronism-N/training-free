# v207 等上下文预算的 Phase-Gated Retrieval 与论文收束计划

> 状态：代码已进入 32-prompt development screen；在 v207 自动门禁通过前，不能写成论文结论。
>
> 当前分支：`codex/v178-v179-causal-validation`

## 1. 为什么需要 v207

v189-v204 已经排除了一个容易误判的故事：目前没有任何静态
Head、Head x Denoising Phase 或 Head x Phase x AR Horizon 分类器，同时通过稳定性、
因果干预、matched control 和 SF-facing generation gain。

已有证据可压缩为四点：

1. v189/v200 的 shadow readout 确实能得到结构化 Head x Phase/Horizon 图，但这只是相关性证据。
2. v190 的 head-phase classifier 没有优于 phase-shift/count-matched controls。
3. v201 的 horizon/static/shift 方法均没有超过 SF，因而 v205/v206 必须保持 fail-closed。
4. v198 60 秒实验表明 retrieval 相对 9-FFE Recent 可显著改善 quality、semantic 和 visual，
   但相对 SF 会损失 temporal mechanics，late-half identity 也不够稳定。

此前所有 cache-schedule 方法还存在一个关键混杂：SF 原生窗口为 21 帧，而候选只有
`sink1 + middle4 + recent4 = 9 FFE`。因此无法区分结果来自长期记忆，还是来自把局部上下文
从 21 帧压缩到 9 FFE。v207 首先消除这个混杂。

## 2. 当前可检验方法

论文暂用名：**Budget-Preserving Phase-Gated Retrieval (BPPR)**。

### 2.1 等预算上下文重分配

给定总 read budget `B`：

- Local/Recent：`sink1 + recent(B-1)`；
- Retrieval/Coverage：`sink1 + retrieved4 + recent(B-5)`。

因此在主配置 `B=21` 中：

- Local：`sink1 + recent20 = 21 FFE`；
- Retrieval：`sink1 + retrieved4 + recent16 = 21 FFE`。

候选没有增加 attention read length，只把 4 个局部帧替换为 4 个非近期历史帧。`B=13`
提供预算曲线，并有独立的 `recent13` matched control。

### 2.2 Training-free semantic retrieval

每个 clean AR update 将可用历史的 exact full-frame K/V 写入有界 archive。当前 retrieval：

1. 从模型 clean update 本身提取 descriptor，不训练外部 encoder；
2. 排除 sink 与 active recent window 重叠的帧；
3. 依据 query relevance 与 novelty admission 更新 archive；
4. 以 MMR 选择最多 4 个兼顾相关性和多样性的历史帧；
5. readout 使用保存的 exact K/V，并按现有 dynamic-RoPE 合约映射位置；
6. archive 上限为 12 FFE，read budget 始终只有 4 FFE middle memory。

### 2.3 Denoising-phase exposure

四步 few-step denoising 中，clean K/V commit 永远读取 Recent，只有 noisy readout 可以读取
retrieval：

- `early1`：仅 call 0；
- `early2`：call 0、1；
- `late2`：call 2、3，作为 equal-dose phase control；
- `full`：四个 noisy calls，作为 exposure 上界。

假设是：高噪声阶段负责全局结构和长程语义，适合少量历史 recall；低噪声阶段负责局部细节和
运动连续性，应保留更密集的近期历史。该假设必须由 `early2 > late2/full` 支持，不能仅凭直觉写入论文。

### 2.4 当前不使用 head classifier

v207 的 360 个 head 使用同一套 shadow banks 和 phase schedule。这样能先判断 retrieval 与 phase
本身是否有效，不把失败的 head membership 混入结果。历史 profiling 可放入 supplementary，作为
为什么放弃静态 head taxonomy 的负结果；除非新实验给出独立因果证据，否则不能把 head classification
列为主贡献。

## 3. v207 32-prompt 矩阵

| Method | Read | Noisy retrieval calls | 作用 |
|---|---|---:|---|
| `sf_native` | native recent21 | - | 论文主基线 |
| `recent21` | sink1 + recent20 | 0 | 等预算 local control |
| `retrieval21_early1` | sink1 + retrieval4 + recent16 | 0 | 最保守 phase 候选 |
| `retrieval21_early2` | 同上 | 0,1 | 主 phase 候选 |
| `retrieval21_late2` | 同上 | 2,3 | equal-dose phase control |
| `retrieval21_full` | 同上 | 0,1,2,3 | full-exposure control |
| `landmark21_early2` | sink1 + landmark4 + recent16 | 0,1 | operator control |
| `recent13` | sink1 + recent12 | 0 | 13-FFE local control |
| `retrieval13_early2` | sink1 + retrieval4 + recent8 | 0,1 | budget-curve candidate |

Prompt 固定为 Qwen-rewritten MovieGen-128 的系统抽样 `1+4k`，共 32 条；30 秒、同 prompt、
同 seed、逐 prompt reseed。四节点的方法执行顺序循环平移，减少 method/node/order 混杂。

## 4. 自动决策，不依赖大规模人工 review

主指标来自 VBench-Long core-9，但 `overall_consistency` 与 `temporal_style` 的重复 ViCLIP 只计一次。
由于历史 Dynamic Degree 实现曾退化为常数 1，promotion 使用固定 DD 后的
`quality_without_dynamic_degree`，并同时报告：

- identity/background；
- temporal mechanics；
- semantic alignment；
- visual quality；
- full、early-half、late-half；
- paired bootstrap CI、sign test 和 BH-FDR；
- frame jump、flow speed、motion coverage、late-motion ratio；
- 候选确定后的 camera-compensated local motion。

Development non-inferiority margin 沿用 v201 冻结合约：

| 指标 | CI 下界阈值 |
|---|---:|
| quality without DD | -0.15 |
| identity/background | -0.0015 |
| temporal mechanics | -0.0030 |
| semantic alignment | -0.0030 |
| visual quality | -0.0040 |

候选必须同时满足：

1. 对 `sf_native` 和 `recent21` 的 full/late-half 非劣；
2. 对两者至少存在预声明阈值以上的正向指标；
3. 两组 temporal automatic guards 均通过；
4. 最多只 promotion 一个候选到 fresh128。

自动通过时不要求人工 review；失败时仅输出最多 4 个异常最大的 prompt 供定位问题，不能用人工选择
改变 promotion 结论。

## 5. 服务器运行顺序

第一批生成代码已经在 commit `a9d027c7`。先更新当前分支：

```bash
git pull origin codex/v178-v179-causal-validation
```

节点 0：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v207_context_budget_phase_screen_32gpu.sh prepare

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v207_context_budget_phase_screen_32gpu.sh smoke

NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v207_context_budget_phase_screen_32gpu.sh audit-smoke
```

smoke audit 通过后，四个节点分别令 `NODE_RANK=0,1,2,3`，同时执行：

```bash
NODE_RANK=<0..3> NUM_NODES=4 \
  bash scripts/run_v207_context_budget_phase_screen_32gpu.sh generate32
```

生成结束后节点 0：

```bash
bash scripts/run_v207_context_budget_phase_screen_32gpu.sh status
NODE_RANK=0 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh audit-screen
NODE_RANK=0 bash scripts/run_v207_vbench_long.sh prepare
```

VBench split/eval 在四节点分别执行：

```bash
NODE_RANK=<0..3> NUM_NODES=4 bash scripts/run_v207_vbench_long.sh split
NODE_RANK=<0..3> NUM_NODES=4 bash scripts/run_v207_vbench_long.sh eval
```

节点 0 收集并打印自动决策：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v207_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v207_vbench_long.sh decision
```

若出现中断，只运行缺失项：

```bash
NODE_RANK=<0..3> NUM_NODES=4 bash scripts/run_v207_vbench_long.sh resume-missing
```

候选确定后，再执行 camera-compensated motion；四节点跑 `motion-compute`，节点 0 跑
`motion-collect`。该结果是论文 motion 分析，不反向改变已冻结的 v207 promotion rule。

## 6. 与相近工作的实验对齐

### Pyramid Forcing

PF 使用 MovieGen-128、30/60 秒、VBench-Long，并报告 latency 与 peak GPU memory；head profiling
使用 32 prompts x 15 秒。我们应对齐 30/60 秒、128 prompts、VBench-Long 和效率表，但 PF
不是当前每轮开发必须重跑的基线。若引用其论文数字，必须明确是 reported number，不与本地同机结果
做显著性检验。

- 论文：<https://arxiv.org/abs/2605.13111>
- 原始代码：<https://github.com/if-lab-pku/Pyramid-Forcing>

### Echo-Forcing

Echo 报告 60/120 秒单 prompt 长视频，以及 smooth transition、hard cut、scene recall 三类交互任务，
每类 64 prompts。当前论文先完成最关键的单 prompt 外推；AB/ABA 是 secondary extension，不应在
主方法尚未通过 SF gate 时消耗 GPU。

- 论文：<https://arxiv.org/abs/2605.16003>
- 原始代码：<https://github.com/mingqiangWu/Echo-Forcing>

### LongLive-RAG

LongLive-RAG 强调同 GPU、同 prompt、同 seed 的 back-to-back A/B，并使用训练得到的 retrieval AE
访问生成历史。v207 对齐其同机配对要求，但 descriptor 来自 frozen generator，本方法不训练 retrieval
encoder，也不应声称首次提出历史检索。

- 论文：<https://arxiv.org/abs/2606.02553>
- 原始代码：<https://github.com/qixinhu11/LongLive-RAG>

### Forcing-KV 与 Head Forcing

二者说明 head-specific cache/compression 和 local/anchor/memory 分工已有明确先例。当前本地
`third_party/Forcing-KV/` 为空，Head Forcing 也未 vendored，因此只能依据论文描述，不能声称完成
代码复现。v207 不以 head 分类为贡献，从而避免与这些工作的核心 claim 重叠。

- Forcing-KV：<https://arxiv.org/abs/2605.09681>
- Head Forcing：<https://arxiv.org/abs/2605.14487>

## 7. 论文最小实验表

### Main table A：30 秒 MovieGen-128

至少包括 `sf_native`、`recent21`、v207 选中的 ours；报告 VBench-Long 官方维度、去重后的分组、
paired CI。外部方法只在协议真正一致时加入同一统计比较。

### Main table B：60 秒 MovieGen-128

同样的三方法，重点报告 full 与 late-half。v198 可作为开发证据，但不能替代新方法的正式 60 秒结果。

### Table C：效率

报告 median/IQR seconds per video、attention read FFE、archive storage FFE、coverage call fraction，
并额外在单卡同机测 peak allocated/reserved GPU memory。只报告 host RSS 不足以替代 GPU memory。

### Table D：组件消融

从 v207 直接得到：

- retrieval vs equal-budget Recent；
- 21 vs 13 FFE；
- early1/early2 vs late2/full；
- retrieval vs landmark；
- archive capacity、retrieved frame count、sink 数量留到主配置确认后的窄消融。

## 8. 不同结果下如何收束

1. **Early1/Early2 同时优于 SF 与 Recent，且 temporal safe**：进入 128 x 30 秒确认，之后做
   128 x 60 秒与效率；这是可写论文的主路线。
2. **只优于 Recent、不优于 SF**：说明 retrieval 能补偿压缩，但没有改善原生生成；不能作为主论文方法。
3. **Recent21 优于 SF、retrieval 不优于 Recent21**：收益来自 cache runtime/attention composition，
   retrieval claim 失败；应先审计 SF/PF runtime parity。
4. **视觉/语义提升但 temporal 失败**：优先选择 early1，之后只做一个更窄 exposure/gate 实验；
   不再增加静态 head classifier。
5. **所有配置失败**：停止方法包装，保留 profiling 与 evaluation protocol 作为内部负结果，不继续扩展 ABA。

## 9. 当前可写与不可写的贡献

若 v207 和 fresh128 均通过，可以写：

1. equal-context-budget 的 local/history redistribution，消除 cache compression 混杂；
2. 无训练、使用 frozen DiT 内部 descriptor 的 exact-K/V bounded retrieval；
3. clean-write/noisy-read 分离与 denoising-phase exposure；
4. full/late-half、matched-budget、phase-shift、motion-safety 的因果评估协议。

当前不能写：

- 我们发现了可靠的静态 head taxonomy；
- 首次提出 long-video retrieval、anchor/recent memory 或 head heterogeneity；
- v189/v200 shadow map 已证明生成因果作用；
- Dynamic Degree 常数结果代表运动提升；
- v207 32-prompt development 结果等价于论文最终 benchmark。
