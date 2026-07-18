# PF/Echo 融合方向与 v3.5 初步结果

> 日期: 2026-07-18
> 状态: 三提示筛选完成，尚未达到论文主结果规模

## 1. 当前结论

1. 原 LifeCache v1-v3 的 full-frame KV recall 路线终止继续调 gate。它能减少部分幻觉，但没有解决 SF/CF 原生的曝光、清晰度、身份和背景漂移，也没有稳定超过 native。
2. 本地 3-prompt/120-latent-frame 结果中，Pyramid-Forcing (PF) 明显强于 SF native，可作为当前默认强基线。
3. 无条件历史 V 统计刷新是失败方案：一致性提高，但运动与场景推进明显下降。
4. 加入 Echo 风格 discrepancy gate 后，候选在严格同噪声 A/B 中初步超过 PF，但增益仍小，不能称为显著领先。
5. AMA 已证明旧 SF/CF head proxy 失效，因此近期不再把新 head 分类作为前置条件。分类只作为后续解释和路由工具。

## 2. 为什么先选 PF

现有同 prompt、seed=0、120 帧旧结果：

| 方法 | Composite | DINO | Drift | Motion smooth | CLIP | BG |
|---|---:|---:|---:|---:|---:|---:|
| SF native | 0.4448 | 0.6964 | -0.00730 | 73.65 | 0.2666 | 0.7929 |
| PF | 0.5294 | 0.8356 | -0.00271 | 78.22 | 0.2797 | 0.8465 |

人工观察也一致：PF 的人物曝光、身份和跑酷连贯性更好，但夜景招牌、背景结构仍持续重绘。因此 PF 是更强起点，不是问题已经解决。

相较其他本地工作：

- Echo-Forcing: 最适合多场景、交互和 recall，包含 hierarchical memory 与 difference-aware decay。应作为替代 base 和模块来源直接复现。
- DeepForcing: deep sink 与 participative compression 简单，适合做配置级融合，但更大 sink 有冻结风险。
- IAMFlow: 用 VLM/LLM 维护显式 entity memory，适合 narrative 分支；系统依赖和任务设定与当前单 prompt SF/CF 不同。
- LongLive-RAG: retrieval encoder 需要训练，适合作为强 retrieval baseline，不应包装为完全 training-free 的核心组件。
- MemRoPE: dual-rate EMA memory 和 online RoPE 有价值，但 EMA 历史 V 可能重复引入外观平均化问题。

## 3. AMA head 分类复盘

关键证据来自 AMA docs 30/31/93/94/96/97/98：

- RF 的真实 attention profile 有区分度；SF/CF 因 FlashAttention 不返回权重，旧实现改用 `|QK|` proxy。
- 该 proxy 将 SF/CF 的 360/360 heads 全判为 identity，随后全头 anchor K scaling 导致背景锁死和运动下降。
- CF 曾复用 SF profile；即使单独 profile，坏 proxy 仍产生同样错误。
- AAI 对身份有帮助，但高强度 anchor 路径和 DARV/anti-drift 组合曾显著降低 dynamic degree。
- 128-prompt 指标曾出现正向结果，但 seed/noise、推理入口和人工视觉问题说明不能直接继承为当前结论。

因此不能简单换阈值重做 identity/motion 分类。若后续分类，必须使用真实 softmax attention 或 counterfactual intervention，并单独验证 SF/CF 跨 prompt、跨 seed 稳定性。

## 4. v3.5: PF + stale-V refresh + Echo gate

### 4.1 方法

PF 保持原来的 per-head cache selection、K 和 RoPE。只在 attention readout 时，将 stale history V 的逐通道均值/方差部分匹配到 recent live V：

```text
V_hist' = V_hist + alpha * gate * (match(V_hist, stats(V_live)) - V_hist)
gate = exp(-lambda * (1 - cosine(mean(V_hist), mean(V_live))))
```

`alpha=0` 时完全关闭。Echo gate 在 stale/live 冲突时把 refresh 压到接近 0，避免强行稳定动作或场景转换。cache 本体不被改写。

### 4.2 无 gate 失败结果

`alpha=0.5, lambda=0` 对旧 PF：

| 方法 | Composite | DINO | Drift | Motion smooth | CLIP | BG |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.5294 | 0.8356 | -0.00271 | 78.22 | 0.2797 | 0.8465 |
| PF + V-refresh | 0.5292 | 0.8576 | -0.00327 | 62.88 | 0.2794 | 0.8710 |

结论：DINO/BG 提升来自过度稳定，motion 大幅下降，人工可见夜景锁在咖啡店附近。淘汰。

### 4.3 Echo gate 公平 A/B

协议：同一 PF inference、同 checkpoint/config、同 prompt 顺序；每条 prompt 在采样前使用 `seed + prompt_index` 独立 reseed。

| 方法 | Composite | DINO | Drift | Motion smooth | CLIP | BG |
|---|---:|---:|---:|---:|---:|---:|
| PF fair base | 0.5247 | 0.7978 | -0.00275 | 82.73 | **0.2847** | 0.8367 |
| PF + V-refresh + Echo gate | **0.5422** | **0.8190** | **-0.00133** | **96.22** | 0.2821 | **0.8533** |

人工观察：公园基本持平；跑酷保持运动但与栏杆交互略少；夜景主体/皮夹克更稳定且场景继续进入街道。该版本通过第一轮筛选，但 CLIP 略降、loop score 上升，且仅 3 prompts，不能宣称显著提升。

三方公平综合指标：

| 方法 | Composite | DINO | Drift | Motion smooth | CLIP | BG |
|---|---:|---:|---:|---:|---:|---:|
| SF native | 0.4540 | 0.6831 | -0.00611 | 116.29 | 0.2514 | 0.7984 |
| PF | 0.5247 | 0.7978 | -0.00275 | 82.73 | **0.2847** | 0.8367 |
| PF + V-refresh + Echo gate | **0.5422** | **0.8190** | **-0.00133** | 96.22 | 0.2821 | **0.8533** |

轻量 motion/luma 诊断（Farneback，只用于筛选）：

| 方法 | Mean flow | Dynamic pixels | Luma Q4/Q1 | Sharpness |
|---|---:|---:|---:|---:|
| SF native | 5.49 | 0.802 | -32.9% | 1749 |
| PF | **6.69** | **0.880** | **+1.2%** | 1436 |
| PF + V-refresh + Echo gate | 6.56 | 0.873 | -3.7% | **1705** |

候选相对 native SF 已经是明显提升，并保留更多真实运动。相对 PF，它提高一致性和清晰度，但亮度稳定性退化，夜景单例 Q4/Q1 为 -11.7%。因此当前候选不是默认最终版本；下一步目标是保留 gated read 的清晰度，同时恢复 PF 的曝光稳定。

运行开销：PF base 约 113.5 秒/提示，当前 Python readout 实现约 201.9 秒/提示。若方法保留，必须融合统计计算以降低开销。

## 5. 顶会论文路线

短期最稳的主线不是“多个论文组件堆叠”，而是一个统一问题：

> PF 决定历史中存什么，但没有区分历史内容是否仍兼容当前生成，也没有处理 stale V 的外观污染。我们研究何时读取历史以及如何读取历史。

候选贡献：

1. Head-aware storage: 继承 PF，作为强 baseline，不 claim 新颖。
2. Compatibility-gated read: 将 Echo 的 scene discrepancy 从全局 memory decay 扩展到 PF 的 per-head ragged memory readout。
3. Asymmetric memory transport: 历史 K 提供匹配/路由，V 根据 live context 校正或替换，减少颜色、曝光和姿态污染。
4. Optional causal head analysis: 用 counterfactual K/V intervention 解释哪些 PF heads 受益；只有稳定时才升级为方法组件。

这个叙事必须通过直接 PF、Echo、PF+gate、PF+gate+asymmetric V 消融来证明。简单融合先用于获得强结果，最终论文需要归纳为一个统一机制，而不是组件清单。

## 6. LifeCache 去留

- 终止：v1-v3 full-frame recall、继续扫固定 gate、旧 proxy head profile、全头 K scaling。
- 可保留：memory lifecycle 的问题定义、trace/evaluation 基础设施、SF/CF 双骨干适配。
- v4 仅指重构后的 `storage + compatibility + asymmetric read`，不继承旧 full-frame 算法。
- 停止条件：在 PF 和至少一个 CF 路径上，无法同时提升人工质量、identity/background、prompt adherence 和 dynamic，则不再使用 LifeCache 名称。

## 7. 下一步门槛

1. 完成同协议 SF native 对照，确认候选显著强于原生 SF。
2. 增加 dynamic degree/光流幅度与分段亮度，不以 motion smoothness 代替运动量。
3. 直接复现 Echo 原版 3 prompts/120 frames，与 PF 公平比较 base 质量。
4. 扩到 16 prompts x 2 seeds；候选需在大多数 prompt 上赢，并无人工可见冻结/幻觉。
5. 再扩到 240/480 latent frames 和 SF/CF 双骨干。
6. 只有通过上述门槛后，才优化 kernel 开销并进入 128-prompt/VBench-Long 主表。

## 8. 结果路径

- 旧 PF/native 接触表与指标: `runs/base_selection/pf_existing_120f/`
- 无 gate 结果: `runs/v35_pf_value_refresh/20260718_133117/`
- gated 筛选（随机流不可与旧结果直接比较）: `runs/v35_pf_value_refresh/20260718_141914/`
- 严格公平 PF base: `runs/v35_pf_value_refresh/20260718_fair_base/`
- 严格公平 gated candidate: `runs/v35_pf_value_refresh/20260718_fair_gate/`
- 人工 review 与公平指标: `runs/v35_pf_value_refresh/FAIR_REVIEW_v35_gate/`
- 严格公平 SF native: `runs/v35_pf_value_refresh/20260718_fair_native/`
