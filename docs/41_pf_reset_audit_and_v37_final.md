# PF reset 审计与 v3.7 最终筛选

> 日期: 2026-07-18
> 状态: reset-fixed 3-prompt/120-latent-frame 筛选完成，尚未达到论文主结果规模

## 1. 本轮最重要的定位结果

同一个 variance-only cafe，独立运行（seed 2）和三提示中的第三条（reseed 后也应为
seed 2）出现 `+3.8%` 与 `-14.0%` 的亮度变化。检查发现
`AdaptiveKVCache.reset()` 只清了 K/V 和部分 workspace，没有清理：

- `_steady_state_reached` / `_prev_cu_seqlens`
- `_last_readout_shape_key` / `_last_readout_anchor_shape_key`
- `_cuda_refresh_desc_key` / `_cuda_refresh_disabled`
- `prompt_v`, cached grid/frame state 和 shadow state

第二、第三个提示可能从第一帧起沿用上一提示的稳态/readout 优化路径。因此旧三提示
PF 即使 `STRENGTH=0` 也不是干净的逐提示 baseline；这不是 LifeCache 或 stale-V
独有问题，而是 PF 多提示评测入口的状态重置漏洞。

修复后 PF cafe 的亮度从旧结果 `+1.0%` 变为 `+9.9%`，证明影响足以改变方法排序。
旧三提示 PF 及基于它的 v3.5/v3.6 数字全部降级为探索记录。

## 2. 修复

`AdaptiveKVCache.reset()` 现在恢复完整的 per-generation 状态，包括稳态检测、上一轮
readout keys、CUDA descriptor key、prompt V、tail/grid/frame state 和 shadow。

新增回归测试 `test_reset_clears_cross_prompt_optimization_state`。修复后的公平重跑脚本：

```text
scripts/run_v37_reset_fair.sh
```

它在三张 GPU 上并行生成 PF、middle-full、variance-only，三者使用相同 prompt、
per-prompt seed、checkpoint 和 120 latent frames。

## 3. 最终五方统一评分

所有方法由工作目录内的离线 evaluator、同一 CLIP checkpoint 统一重评：

| 方法 | Composite | DINO | Drift | Smooth | LPIPS | CLIP | BG | Loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SF native | 0.4558 | 0.6831 | -0.00611 | 116.29 | 0.4022 | 0.2592 | 0.7984 | **0.0000** |
| PF reset-fixed | 0.5236 | 0.7948 | -0.00300 | 80.56 | 0.4145 | 0.2961 | 0.8438 | 0.0133 |
| middle full | **0.5420** | **0.8195** | **-0.00184** | **97.53** | 0.4135 | 0.2973 | 0.8433 | 0.0145 |
| variance-only | 0.5279 | 0.8146 | -0.00284 | 91.16 | 0.4074 | **0.2988** | **0.8574** | 0.0490 |
| mean-only | 0.5356 | 0.8052 | -0.00189 | 80.64 | **0.3993** | 0.2975 | 0.8368 | 0.0320 |

轻量 motion/luma 诊断：

| 方法 | Mean flow | Dynamic pixels | Luma Q4/Q1 | Sharpness | Cafe luma |
|---|---:|---:|---:|---:|---:|
| SF native | 5.485 | 0.802 | -32.9% | 1749 | -38.3% |
| PF reset-fixed | **6.940** | **0.891** | +2.9% | 1396 | +9.9% |
| middle full | 6.585 | 0.888 | -0.7% | 1527 | -9.6% |
| variance-only | 6.909 | 0.889 | +2.3% | 1462 | -4.6% |
| mean-only | 6.212 | 0.869 | -2.8% | **1599** | -15.4% |

## 4. 方法选择

当前 `ours` 选择 variance-only，而不是 Composite 更高的 middle/mean：

- 相对 reset-fixed PF，DINO `+0.0198`，BG `+0.0136`，smoothness `+13.2%`，
  CLIP 略升，sharpness `+4.7%`。
- flow 只下降 `0.45%`，dynamic pixels 只下降 `0.26%`，平均亮度仍为正变化。
- middle 的 flow 下降约 `5.1%` 且 cafe 压暗；mean 的 flow 下降约 `10.5%` 且
  cafe 压暗 `-15.4%`，都不满足长外推动态门槛。
- variance 的 loop score 从 PF `0.0133` 增至 `0.0490`，仍是风险；扩大实验前必须
  人工检查是否为真实重复，而不能只报告一致性收益。

相对 SF native，variance-only 已是显著提升：Composite `+15.8%`，DINO `+19.2%`，
BG `+7.4%`，mean flow `+25.9%`，并消除了 SF 的整体严重变暗。相对 PF 的增益仍小，
所以当前结果满足“显著强于 SF native”的阶段目标，但不满足顶会论文最终贡献门槛。

## 5. 当前创新点的统一表述

目前不是简单堆叠多个工作，而是一个统一的 memory read 问题：

1. PF temporal-pattern storage：PF 决定每个 `(layer, head)` 保存哪些时间模式；
   label 是层条件时间访问模式，不是固定 identity/motion 语义。
2. Echo-style compatibility gate：stale/live 冲突时抑制历史校正，避免强行冻结场景切换。
3. Asymmetric moment transport：K/position 保持 PF 路由，只对 stale V 的 variance 与
   live context 对齐，明确不搬运 mean，减少曝光/颜色污染。
4. Depth/class intervention：支持按 PF label、深度和二维 route 做因果消融；当前结果
   说明静态 head index 分类无效，简单类别与深度规则叠加也没有线性增益。

这吸收了 PF storage 与 Echo discrepancy 的思想，但提出的是 layer-conditioned,
compatibility-gated asymmetric read，而不是把其他工作作为组件清单拼接。

## 6. Head/depth 结论（修正后）

PF 30 layers x 12 heads 的相邻层标签保持率只有 `0.463`。固定 head index 沿深度
最多发生 20 次标签转换，因此不能稳定解释为身份或运动 head。

修正 baseline 后的 cafe 单提示仍支持以下定性结论：

- oscillating/stable 提高 DINO/drift，但分别产生 `-12.5%/-17.2%` 亮度变化并降低 flow。
- stable-sparse 保持 flow，但质量收益较小。
- early 提高清晰度但损失 flow/CLIP；middle 提高 DINO/BG，仍有动态代价；late 最接近 PF。
- `osc/stable -> middle, sparse -> all layers` 的二维规则没有超过单变量最优。

单提示只用于解释，不能用于最终选方法。统一结果在：

- `runs/head_ablation/v37_cafe_all_valid_comprehensive.json`
- `runs/head_ablation/v37_cafe_all_valid_motion_luma.json`

## 7. 人工 review

最终三方同步视频：

```text
runs/REVIEW_v37_threeway/
  native/       SF native
  pf/           reset-fixed SF + PF
  ours/         SF + PF + compatibility-gated variance transport
  comparisons/  三列同步视频与 contact sheets
```

三条视频为秋日公园、跑酷、夜间咖啡街景。抽帧显示：variance 保持 PF 的主体曝光与
场景推进，显著避免 SF native 后段的变暗、身份和背景崩坏。人工 review 仍需重点看
完整运动中的障碍交互、路人生成/消失以及是否出现重复循环。

## 8. 评估基础设施修复

AMA evaluator 原先每次从 Hugging Face 请求 OpenCLIP，代理不可用时评分失败。
`scripts/evaluate_comprehensive.py` 已纳入工作目录，并优先加载
`~/.cache/clip/ViT-L-14.pt`（可用 `CLIP_CHECKPOINT` 覆盖）。结果和 evaluator 均可
离线复现，不能再混用旧/新 CLIP evaluator 的 Composite。

验证结果：

- training-free: `13 passed`
- PF history/cache/config focused tests: `49 passed`
- shell syntax、Python compile、`git diff --check`: passed

## 9. 下一阶段（论文门槛）

1. 扩到 16 prompts x 2 seeds，每个提示同时生成 SF native / PF / ours；按身份、背景、
   曝光、动态、交互、loop 分层，不再根据 3 prompts 宣称稳定提升。
2. 直接复现 Echo 原版作为 baseline，区分 Echo 自身、PF 自身与 asymmetric read 的收益。
3. 将 compatibility-gated variance transport 适配回原生 SF 和 CF。PF 只用于快速筛选，
   SF/CF 双骨干验证是必须项，不能用 PF 结果替代。
4. 通过 16x2 后再扩到 240/480 latent frames；否则停止长视频算力扩张并修改方法。
5. 将当前约 1.7x 的 Python readout 统计开销融合进 attention/kernel；效率未解决前不做
   speed claim。
6. 若相对 PF 在 16x2 上仍只有当前量级的小增益，则需要引入新的 scene-change/loop
   counterfactual signal，或转向 Echo base，而不是继续微调固定 gate。

## 10. 最终结果路径

- 人工 review: `runs/REVIEW_v37_threeway/`
- 五方综合指标: `runs/head_ablation/v37_final_fiveway_comprehensive.json`
- 五方 motion/luma: `runs/head_ablation/v37_final_fiveway_motion_luma.json`
- PF: `runs/v35_pf_value_refresh/20260718_v37_reset_pf/`
- middle: `runs/v35_pf_value_refresh/20260718_v37_reset_middle/`
- variance (ours): `runs/v35_pf_value_refresh/20260718_v37_reset_variance/`
- mean: `runs/v35_pf_value_refresh/20260718_v37_reset_mean/`
