# PF head-depth 与 moment 解耦实验

> 日期: 2026-07-18
> 状态: **已被 reset audit 部分推翻，请以 docs/41 为准**

> 重要：本文件记录了发现 reset 漏洞之前的实验过程。旧三提示 PF 在提示之间没有
> 完整清理优化状态，单提示消融所引用的 PF baseline 又来自旧三提示的第三条，因而
> 第 4-7 节的定量比较不能作为最终结论。保留本文是为了记录 hypothesis 和失败过程；
> 修复、重跑与最终数字见 `docs/41_pf_reset_audit_and_v37_final.md`。

## 1. 本轮目标

v3.5 的全 head stale-V refresh 相对 SF native 明显更好，但相对 PF 的主要收益是
一致性和清晰度，夜景仍有 `-11.7%` 的亮度衰减。本轮不再继续盲扫 strength，
而是定位收益和副作用来自哪些 head、哪些深度、以及 V 的 mean/variance 哪一部分。

所有消融使用相同 PF checkpoint、prompt、seed、120 latent frames 和独立 prompt
reseed。单提示实验只用于因果筛选，最终结论以三提示复核为准。

## 2. 三方人工 review

当前可直接 review 的公平三方视频：

```text
runs/REVIEW_v35_threeway/
  native/       SF native 原视频
  pf/           SF + PF 原视频
  ours/         SF + PF + gated stale-V refresh 原视频
  comparisons/  三列同步对比视频
```

三个同步视频分别为公园、跑酷、夜间咖啡街景。列顺序固定为 `SF native | SF+PF |
SF+PF+ours`。所有文件均在 `training-free/runs`，没有结果写入 `/tmp`。

夜景抽帧确认：SF native 后段主体衣服和背景大幅变化；PF 明显改善；v3.5 ours
进一步稳定主体轮廓，但有压暗倾向。这说明变暗/身份/背景幻觉是原生长外推问题，
不是 LifeCache 独有问题；我们当前方法只解决了其中一部分。

## 3. PF 分类应如何重新定义

PF 的原始 30-layer x 12-head 标签统计：

| 深度 | oscillating | stable | stable-sparse |
|---|---:|---:|---:|
| early 0-9 | 74 | 37 | 9 |
| middle 10-19 | 35 | 71 | 14 |
| late 20-29 | 47 | 64 | 9 |

相邻层标签保持率只有 `0.463`。固定 head index 沿深度频繁换类，例如 head 3 有
20 次深度标签转换，head 9 有 18 次。因此不能把固定编号 head 稳定解释为身份、
运动或背景。

更可靠的新分类标准是：

> `(layer, head)` 的标签描述该 head 在当前深度的时间访问模式，而不是跨层不变的
> 语义身份。

这与 AMA 的失败结论一致：SF/CF 的 `|QK|` proxy 曾把 360/360 heads 判为 identity，
全头 anchor K scaling 随后导致背景锁死和 dynamic degree 下降。新代码因此按每层
真实 `head_labels` 路由，不复用固定 head index 语义。

完整统计在：

- `runs/head_analysis/pf_head_depth.md`
- `runs/head_analysis/pf_head_depth.json`
- 分析脚本: `scripts/analyze_pf_head_depth.py`

## 4. Head 类别消融

夜景单提示、相同噪声：

| 路由 | Composite | DINO | Drift | CLIP | Mean flow | Luma Q4/Q1 | Sharpness |
|---|---:|---:|---:|---:|---:|---:|---:|
| PF | 0.4728 | 0.6826 | -0.00426 | 0.2485 | 7.21 | +1.0% | 1469 |
| all heads | 0.5048 | 0.7025 | -0.00167 | 0.2433 | 7.04 | -11.7% | 2040 |
| oscillating only | 0.5115 | 0.7141 | -0.00139 | 0.2388 | 6.66 | -12.5% | 1699 |
| stable only | 0.5106 | 0.6971 | -0.00125 | 0.2437 | 6.58 | -17.2% | 1581 |
| stable-sparse only | 0.4925 | 0.6968 | -0.00280 | 0.2526 | 7.03 | +1.6% | 1498 |

结论：

1. oscillating/stable 是一致性和压暗的共同主要来源，不能只看 Composite 选择。
2. stable-sparse 基本保留 PF 的曝光和运动，但增益很小。
3. “identity head / motion head” 解释没有得到证据；时间模式标签可以作为路由变量，
   但不是论文贡献本身。

## 5. 深度消融

夜景单提示、全类别：

| 深度 | Composite | DINO | CLIP | Mean flow | Luma Q4/Q1 | Sharpness |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.4728 | 0.6826 | 0.2485 | 7.21 | +1.0% | 1469 |
| early 0-9 | 0.4920 | 0.7052 | 0.2353 | 6.15 | +4.9% | 1848 |
| middle 10-19 | 0.4930 | 0.7238 | 0.2494 | 6.85 | +2.7% | 1623 |
| late 20-29 | 0.4780 | 0.6949 | 0.2498 | 7.33 | +9.9% | 1488 |

中层是最好的质量/运动折中。early 虽清晰，但 flow 和 CLIP 损失明显；late 接近 PF。

进一步测试 `osc/stable -> middle, sparse -> all layers` 的二维规则，Composite 只有
`0.4914`，flow `6.57`。简单叠加单变量最优规则没有线性增益，不能作为主方法。

## 6. 三提示复核

本表中的四个方法由同一版离线 CLIP evaluator 统一重评，不能把 Composite/CLIP
直接与旧 evaluator 的数字混用；DINO、drift、motion 和 BG 不受该修复影响。

| 方法 | Composite | DINO | Drift | Smooth | CLIP | BG |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.5278 | 0.7978 | -0.00275 | 82.73 | **0.2987** | 0.8367 |
| all-head full | **0.5450** | **0.8190** | **-0.00133** | **96.22** | 0.2948 | **0.8533** |
| middle full | 0.5353 | 0.8032 | -0.00216 | 91.61 | 0.2956 | 0.8460 |
| stable-sparse full | 0.5277 | 0.8011 | -0.00265 | 85.62 | 0.2967 | 0.8352 |

| 方法 | Mean flow | Dynamic pixels | Luma Q4/Q1 | Sharpness |
|---|---:|---:|---:|---:|
| PF | 6.694 | 0.880 | +1.2% | 1436 |
| all-head full | 6.564 | 0.873 | -3.7% | **1705** |
| middle full | 6.680 | 0.881 | -1.7% | 1531 |
| stable-sparse full | 6.693 | **0.885** | +3.1% | 1429 |

当前默认候选是 `middle full`，不是最高 Composite 的 `all-head full`。原因是它相对
PF 基本不损失 flow/dynamic，亮度漂移显著小于 all-head，同时仍提高 DINO、drift、
smoothness、BG 和 sharpness。它相对 PF 的增益仍不够大，尚不能作为论文最终主结果。

## 7. Moment 解耦

全 moment transport 为：

```text
V' = normalize(V_stale; mean_stale, std_stale) * std_live + mean_live
```

现在拆成三种可控路径：

- `full`: 搬运 live mean 和 live variance。
- `variance_only`: 保留 stale mean，只匹配 live variance。
- `mean_only`: 只搬运 live mean。

夜景 `variance_only` 的初步结果：Composite `0.4976` vs PF `0.4798`，DINO
`0.6977` vs `0.6826`，BG `0.8311` vs `0.8185`；亮度从全 moment 的 `-11.7%`
恢复到 `+3.8%`，flow 保持 `7.15`。但 sharpness 下降到 `1244`，所以必须等三提示
结果后再决定是否保留。

这一结果定位了一个重要原因：live mean transport 是压暗的主要嫌疑，mean 与
variance 不应无条件绑定。若三提示成立，论文方法可统一表述为
`compatibility-gated, depth-conditioned asymmetric moment transport`。

## 8. 实现与可复现性

新增配置/CLI：

- `pyramidkv_history_value_labels`
- `pyramidkv_history_value_layer_start/end`
- `pyramidkv_history_value_label_layer_routes`
- `pyramidkv_history_value_moment_mode`

运行脚本支持 `HEAD_LABELS`, `LAYER_START`, `LAYER_END`, `HEAD_ROUTES`,
`MOMENT_MODE`。路由 mask 根据每层实际 PF label 构造。

AMA evaluator 原先每次通过 Hugging Face 请求 OpenCLIP，在代理不可用时无法评分。
`scripts/evaluate_comprehensive.py` 现在优先读取本机
`~/.cache/clip/ViT-L-14.pt`，可离线复现；也可通过 `CLIP_CHECKPOINT` 覆盖。

验证：

- training-free tests: `13 passed`
- PF history/config focused tests: `21 passed`
- shell syntax、Python compile、`git diff --check`: passed

## 9. 后续门槛

1. 完成 variance-only 三提示并人工 review；只在跨 prompt 保住质量时升级为候选。
2. 将当前最佳候选重新打包成 `SF native | SF+PF | SF+PF+ours` 三方同步视频。
3. 扩到至少 16 prompts x 2 seeds；按场景转换、人物身份、背景、曝光、动态分层报告。
4. 复现 Echo 原版作为直接 baseline，证明收益不是 PF 或 Echo 单独已有。
5. 在 SF 与 CF 路径验证相同机制。PF 只负责快速筛选，不能代替 SF/CF 主实验。
6. 通过短集门槛后再做 240/480 latent frames；否则停止扩大算力。

## 10. 结果路径

- 三方人工 review: `runs/REVIEW_v35_threeway/`
- head-depth 统计: `runs/head_analysis/`
- 单提示类别/深度/二维/moment 指标: `runs/head_ablation/cafe_*`
- 三提示 v3.6 指标: `runs/head_ablation/v36_3prompt_*`
- middle 视频: `runs/v35_pf_value_refresh/20260718_v36_middle_3prompt/`
- sparse 视频: `runs/v35_pf_value_refresh/20260718_v36_sparse_3prompt/`
- variance-only 视频: `runs/v35_pf_value_refresh/20260718_v36_variance_3prompt/`
