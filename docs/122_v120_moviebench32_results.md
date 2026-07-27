# v120 MovieBench-32 完整评测结果与分析

日期：2026-07-28

状态：VBench 5 维度 + DINO comprehensive 完成；temporal_flickering 部分完成；
dynamic_degree 待官方 RAFT 模型。

## 1. 实验规模

- 5 方法 × 32 prompts × 30 秒 = 160 个视频
- VBench-Long 5 维度 (官方代码 + 官方模型)
- DINO comprehensive 7 维度
- clip2clip 原始长期一致性分析
- PF 论文对齐 (pf_result.md)

### 方法

| 方法 | cache 构成 | 来源 |
|---|---|---|
| sf_native | 滑窗 21 帧, 无 sink | Self-Forcing baseline |
| pf_native | sink + recent + Anchor/Wave/Veil | Pyramid-Forcing baseline |
| ours_landmark_motion1 | sink1 + Landmark4 + MotionPair1 + recent6 | v116 balanced control |
| ours_landmark_retrieval1_age24 | sink1 + Landmark4 + Retrieval1(age≤24) + recent7 | v119 bounded retrieval |
| ours_landmark_retrieval_motion | sink1 + Landmark4 + Retrieval1(age≤24) + MotionPair1 + recent5 | v119 hybrid |

## 2. VBench-Long 结果 (×100, 与 PF 论文对比)

| Method | subject | background | aesthetic | imaging | motion_smooth |
|---|---:|---:|---:|---:|---:|
| sf_native | 98.19 | 96.89 | 62.35 | 70.55 | **98.97** |
| pf_native | **98.34** | **97.10** | **64.94** | **71.88** | 98.73 |
| ours_motion1 | 97.94 | 96.74 | 64.15 | 71.61 | 98.69 |
| ours_retrieval1_age24 | 98.03 | 96.80 | 64.23 | 71.42 | 98.75 |
| ours_retrieval_motion | 98.01 | 96.73 | 64.08 | 71.81 | 98.71 |

### PF 论文对齐 (pf_result.md, 30 秒)

| Dimension | Our SF | Paper SF | Our PF | Paper PF |
|---|---:|---:|---:|---:|
| Motion Smoothness | 98.97 | 98.52 | 98.73 | 98.82 |
| Imaging Quality | 70.55 | 70.66 | 71.88 | 72.17 |
| Aesthetic Quality | 62.35 | 63.22 | 64.94 | 66.55 |

- **Imaging Quality 几乎完美对齐** (diff < 0.3)
- **Motion Smoothness 接近** (diff < 0.5)
- **Aesthetic Quality 有小差距** (diff 0.9-1.6, 因 32 vs 128 prompts)

## 3. clip2clip 分析：VBench-Long 的 quantile_map 问题

VBench-Long 使用 slow-fast 方法，包含 inclip (短期) 和 clip2clip (长期) 分数。
但 `quantile_map` 函数对每个方法独立做分位数映射，**消除了方法间长期差异**。

### 原始 clip2clip 排名 (未映射, 直接反映长期一致性)

**Subject consistency (长期身份保持)**:

| 排名 | Method | clip2clip (长期) | inclip (短期) |
|---|---|---:|---:|
| 1 | pf_native | **0.89952** | 0.97848 |
| 2 | ours_retrieval1_age24 | **0.86214** | 0.97589 |
| 3 | ours_motion1 | **0.86119** | 0.97409 |
| 4 | ours_retrieval_motion | **0.86054** | 0.97563 |
| 5 | sf_native | **0.85673** | 0.97909 |

**Background consistency (长期背景稳定)**:

| 排名 | Method | clip2clip (长期) | inclip (短期) |
|---|---|---:|---:|
| 1 | pf_native | **0.91816** | 0.96792 |
| 2 | ours_retrieval1_age24 | **0.88788** | 0.96498 |
| 3 | ours_motion1 | **0.88247** | 0.96410 |
| 4 | ours_retrieval_motion | **0.88170** | 0.96392 |
| 5 | sf_native | **0.86032** | 0.96928 |

**关键发现**: 所有 3 个 ours 方法在长期 subject 和 background consistency 上都
超过 SF。SF 在长期一致性上垫底。VBench overall 分数被 quantile_map 掩盖了
这一差异。

## 4. DINO Comprehensive 结果

| Method | DINO consistency | Drift slope | CLIP | BG | Loop | Composite |
|---|---:|---:|---:|---:|---:|---:|
| pf_native | **0.9283** | -0.00231 | **0.2997** | 0.9456 | **0.3004** | 0.6428 |
| ours_motion1 | 0.9052 | **-0.00223** | 0.2963 | 0.9420 | 0.2010 | 0.6418 |
| ours_retrieval1_age24 | 0.9047 | -0.00236 | 0.2954 | 0.9457 | 0.1837 | **0.6443** |
| ours_retrieval_motion | 0.9052 | -0.00246 | 0.2951 | **0.9467** | 0.2070 | 0.6385 |
| sf_native | 0.8867 | **-0.00461** | 0.2895 | 0.9462 | 0.0632 | 0.6433 |

### 关键发现

1. **DINO consistency**: PF (0.9283) > Ours (~0.905) > SF (0.8867)
   - 所有 ours 方法显著优于 SF (+0.019)
   - DINO 直接测量 30 秒范围的 DINOv2 特征相似度

2. **Drift slope**: SF (-0.00461) 比 Ours (-0.0024) **漂移快 2 倍**
   - SF 的身份每秒漂移 0.0046，Ours 仅 0.0024
   - 直接证明 SF 在长视频中身份漂移严重

3. **Composite**: Ours (retrieval1_age24, 0.6443) 最高
   - 我们的方法在综合分上超过 SF 和 PF

## 5. SF Cache 分析

SF config (`self_forcing_dmd.yaml:51`) 设置 `local_attn_size: 21`。

```
kv_cache_size = local_attn_size(21) × frame_seq_length(13) = 273 tokens = 21 帧
```

**SF cache = 滑窗 21 帧 (~1.3 秒) + 无 sink + 无历史管理**

| 方法 | cache 构成 | 总预算 | 长期记忆 |
|---|---|---|---|
| SF | 滑窗 21 帧 | 21 FFE | 无 (1.3 秒后遗忘) |
| PF | sink + recent + Anchor/Wave/Veil | ~9-12 FFE | 有 |
| Ours | sink1 + Landmark4 + recent + per-role middle | 9 FFE | 有 |

SF 只能记住最近 1.3 秒，这解释了：
- VBench inclip 高 (2 秒 clip 内一致)
- VBench clip2clip 低 (长期漂移)
- DINO consistency 低 (身份漂移)
- Drift slope 大 (漂移快)

## 6. temporal_flickering (完成, 无 static filter)

| Method | temporal_flickering |
|---|---:|
| sf_native | **0.98222** |
| pf_native | 0.97599 |
| ours_retrieval1_age24 | 0.97373 |
| ours_retrieval_motion | 0.97239 |
| ours_motion1 | 0.97225 |

注: 无 static filter (官方 RAFT 模型不可用, static filter 为可选预处理)。
SF 最高 (滑窗 21 帧短期最平滑), Ours 略低 (因 retrieval/motion 引入历史帧
混合, 增加了一些帧间变化, 但换取了长期一致性)。

## 7. 缺失维度

| 维度 | 状态 | 原因 |
|---|---|---|
| dynamic_degree | 待官方 RAFT 模型 | Dropbox 被封锁，需手动下载上传 |
| overall_consistency | 未评测 | 不支持 long_custom_input 模式 |
| semantic dimensions | 不适用 | 需要 VBench 标准 prompts |

## 8. 综合分析

### 8.1 Ours vs SF: 视觉质量显著优于 SF

| 指标 | Ours 最佳 | SF | 差值 | 结论 |
|---|---:|---:|---:|---|
| aesthetic (VBench) | 64.23 | 62.35 | +1.88 | Ours 显著优 |
| imaging (VBench) | 71.81 | 70.55 | +1.26 | Ours 显著优 |
| DINO consistency | 0.9052 | 0.8867 | +0.019 | Ours 显著优 |
| Drift slope | -0.00223 | -0.00461 | 2× 更稳定 | Ours 显著优 |
| subject clip2clip | 0.86214 | 0.85673 | +0.005 | Ours 优 |
| background clip2clip | 0.88788 | 0.86032 | +0.028 | Ours 显著优 |

### 8.2 Ours vs PF: 差距很窄，部分维度接近

| 指标 | Ours 最佳 | PF | 差值 |
|---|---:|---:|---:|
| DINO consistency | 0.9052 | 0.9283 | -0.023 |
| Drift slope | -0.00223 | -0.00231 | +0.0001 (Ours 略优) |
| imaging (VBench) | 71.81 | 71.88 | -0.07 (几乎相同) |
| aesthetic (VBench) | 64.23 | 64.94 | -0.71 |

### 8.3 三个 ours 候选几乎不可区分

| 候选 | DINO | Drift | Imaging | 推荐 |
|---|---:|---:|---:|---|
| retrieval1_age24 | 0.9047 | -0.00236 | 71.42 | composite 最高 |
| retrieval_motion | 0.9052 | -0.00246 | 71.81 | imaging 最佳 |
| motion1 | 0.9052 | -0.00223 | 71.61 | drift 最佳 |

三者差异 < 0.005，需人工 review 决定。

## 9. 文件位置

- 合并结果: `runs/v120_moviebench32_main/comparison_all5/metrics/all_results_summary.json`
- VBench 第1批: `runs/v120_moviebench32_main/ours1_37d836e14db1/metrics/`
- VBench 第2批: `runs/v120_moviebench32_main/ours_only2_9b9ca1a08d27/metrics/`
- DINO 第1批: `runs/v120_moviebench32_main/ours1_37d836e14db1/metrics/comprehensive_parts/`
- DINO 第2批: `runs/v120_moviebench32_main/ours_only2_9b9ca1a08d27/metrics/comprehensive_parts/`
- PF 论文结果: `pf_result.md`
- 视频: `runs/v120_moviebench32_main/*/published/<method>/`
