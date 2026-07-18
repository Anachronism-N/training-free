# SF/CF 原生长时变暗的 Latent 诊断

> 日期：2026-07-18  
> 状态：只读 trace 完成；已定位 latent 统计漂移，尚未实施校正。

## 1. 动机

v3.3 人工 review 表明 LifeCache 主要减少部分幻觉，并没有显著改善原生 SF/CF 的清晰度、曝光和纹理。继续增强 historical full-frame memory 只会增加 stale recall 和冻结风险。

因此先回答：长时变暗首先发生在 denoised latent，还是只发生在 VAE decode？

## 2. Trace 方法

新增环境变量 `AR_LATENT_TRACE_PATH`，在不修改任何生成张量的情况下记录：

- 每个 3-frame denoised block 的 overall/channel mean、std、RMS；
- 完整 decode 后逐帧 RGB mean、luma 和 std；
- 多 prompt 的独立 `video_index`。

运行脚本：

```bash
bash scripts/run_v34_native_trace.sh
```

分析脚本：

```bash
python scripts/analyze_latent_trace.py \
  runs/v34_latent_trace/20260718_125319/sf/latent_trace.jsonl \
  runs/v34_latent_trace/20260718_125319/cf/latent_trace.jsonl
```

结果目录：`runs/v34_latent_trace/20260718_125319/`。

## 3. 结果

单 prompt、120 latent frames、477 decoded frames：

| Backbone | Luma first quarter | Luma last quarter | Relative change | Latent mean slope | Latent std slope | Latent RMS slope |
|---|---:|---:|---:|---:|---:|---:|
| SF | 0.4383 | 0.2989 | **-31.8%** | -0.0104 | +0.0539 | +0.0524 |
| CF | 0.5120 | 0.2025 | **-60.5%** | -0.0152 | +0.3941 | +0.3869 |

SF latent mean 从首四分之一 0.1310 降到末四分之一 0.1252，std 从 0.9551 升到 0.9860。

CF latent mean 从 0.2203 降到 0.1993，std 从 1.2661 升到 1.5655。CF 的统计漂移明显强于 SF，与其更严重的后段曝光下降一致。

结论：变暗在 denoised latent 中已经形成，VAE 不是唯一根因。

## 4. Channel 与亮度相关性

将 477 帧 luma 按 40 个 latent blocks 聚合，并与每个 latent channel 的 block mean/std 比较。

SF channel mean 与 luma 的最高相关：

| Channel | Mean slope | Luma correlation |
|---:|---:|---:|
| 15 | -0.4868 | +0.9691 |
| 2 | -0.9471 | +0.9294 |
| 3 | -0.6415 | +0.9194 |
| 10 | -0.1396 | +0.9074 |
| 11 | -0.4393 | +0.8589 |

CF channel mean 与 luma 的最高相关：

| Channel | Mean slope | Luma correlation |
|---:|---:|---:|
| 14 | -1.5618 | +0.9784 |
| 10 | -0.8469 | +0.9723 |
| 8 | +1.6451 | -0.9709 |
| 9 | +1.6305 | -0.9603 |
| 3 | -1.0051 | +0.9489 |

两个 checkpoint 的曝光相关通道与漂移方向不同，不能使用一套固定 channel mask 或全通道 mean/std 校正。

## 5. 对旧 Anti-Drift 的判断

AMA/RollingForcing 的历史 anti-drift：

- 在 streaming single segment 中曾长期是 no-op；
- 修复后收益很弱；
- anti-drift-only 某些实验明显退化；
- 检测 drift 后向整个 latent 注入随机噪声，并不针对曝光方向。

因此不直接移植该实现。

## 6. 下一步方案

### E1：VAE luminance direction

使用 VAE 对 latent channel 的有限差分或局部 Jacobian，估计“改变亮度但尽量不改变结构”的低秩方向。只校正 latent mean 在该方向上的漂移，并满足：

- 从原生窗口淘汰后才启动；
- correction 有硬上限；
- 保留与亮度方向正交的内容/运动分量；
- SF/CF 分别自动校准，不写死 channel。

先做离线 latent decode counterfactual；只有能提升亮度且不产生色偏/闪烁，才进入自回归循环。

### E2：Historical-K + live-V

当前 memory branch 同时读取历史 K/V。历史 V 可能降低当前清晰度、颜色和运动自由度。参考 RollingForcing 后期修复，比较：

- historical K + historical V；
- historical K + historical V matched to live V statistics；
- historical K + latest live V；
- no memory。

该消融用于判断 LifeCache 是否可以只提供“检索/布局约束”，而由当前窗口提供高质量 appearance values。

### E3：2x2 最终组合

在 E1/E2 分别通过人工 review 后比较：

| | No memory | Low-pollution memory |
|---|---|---|
| Native latent | baseline | memory only |
| Exposure-stabilized latent | exposure only | combined |

Go 条件仍然是 SF/CF 双骨干上人工可见质量显著提升，而不是只提高 DINO 或减少幻觉。
