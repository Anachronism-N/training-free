# v120 VBench-Long 分析、clip2clip 发现、DINO 评测与 PF 论文对齐

日期：2026-07-28

> 审计更正：官方评测使用预先计算的固定 mapping table，并非对每个待评方法独立
> 做分位数重排。高分区间压缩现象存在，但本文第 3.2 节的原因表述不准确。
> `docs/122` 与本文的 DINO 辅助指标表也存在口径冲突。论文级解释和待补文件见
> `docs/124_v120_metric_human_alignment_audit.md`。

状态：VBench-Long 5 维度 + DINO comprehensive 完成；temporal_flickering 重试中。
待 temporal_flickering 完成后统一推送。

## 1. 实验概览

- 5 方法 × 32 prompts × 30 秒 = 160 个视频
- VBench-Long 5 维度 (subject, background, aesthetic, imaging, motion_smoothness)
- DINO comprehensive (DINO consistency, drift slope, CLIP, BG, loop, composite)
- temporal_flickering 重试中 (无 static filter，RAFT 不可用)
- dynamic_degree 缺失 (RAFT 模型无法通过 proxy 下载)

## 2. VBench-Long 结果 (overall)

| Method | subject | background | aesthetic | imaging | motion_smooth |
|---|---:|---:|---:|---:|---:|
| pf_native | **0.98342** | **0.97098** | **0.64937** | **0.71880** | 0.98729 |
| sf_native | 0.98187 | 0.96886 | 0.62345 | 0.70546 | **0.98971** |
| ours_landmark_retrieval1_age24 | 0.98033 | 0.96803 | 0.64230 | 0.71419 | 0.98747 |
| ours_landmark_motion1 | 0.97944 | 0.96738 | 0.64151 | 0.71609 | 0.98686 |
| ours_landmark_retrieval_motion | 0.98011 | 0.96727 | 0.64076 | 0.71813 | 0.98706 |

## 3. clip2clip 发现：VBench-Long 的 quantile_map 问题

### 3.1 VBench-Long slow-fast 评测机制

VBench-Long 将 30 秒视频切分为 2 秒 clip，对 subject/background consistency 使用
slow-fast 方法：

- **inclip (短期)**: clip 内帧间一致性 (~2 秒范围)
- **clip2clip (长期)**: 跨 clip 第一帧之间的相似度 (30 秒范围)

最终分数 = 0.5 × inclip + 0.5 × mapped_clip2clip

### 3.2 固定 mapping table 压缩高分区间的长期差异

VBench-Long 使用预先计算的固定 mapping table，将 clip2clip 分数映射到
inclip 的量纲。该表在本实验所在的高分区间斜率很小，导致方法间长期一致性差异
被大幅压缩：

| Method | raw clip2clip (长期) | mapped clip2clip | 差异压缩 |
|---|---:|---:|---|
| sf_native | 0.85673 | 0.98466 | +0.128 |
| ours_retrieval1_age24 | 0.86214 | 0.98477 | +0.123 |
| 差异 | **0.00541** | **0.00011** | **压缩 98%** |

### 3.3 原始 clip2clip 排名 (未映射)

**Subject consistency (长期身份保持)**:

| 排名 | Method | clip2clip | inclip | overall (报告) |
|---|---|---:|---:|---:|
| 1 | pf_native | **0.89952** | 0.97848 | 0.98342 |
| 2 | ours_retrieval1_age24 | **0.86214** | 0.97589 | 0.98033 |
| 3 | ours_motion1 | **0.86119** | 0.97409 | 0.97944 |
| 4 | ours_retrieval_motion | **0.86054** | 0.97563 | 0.98011 |
| 5 | sf_native | **0.85673** | 0.97909 | 0.98187 |

**Background consistency (长期背景稳定)**:

| 排名 | Method | clip2clip | inclip |
|---|---|---:|---:|
| 1 | pf_native | **0.91816** | 0.96792 |
| 2 | ours_retrieval1_age24 | **0.88788** | 0.96498 |
| 3 | ours_motion1 | **0.88247** | 0.96410 |
| 4 | ours_retrieval_motion | **0.88170** | 0.96392 |
| 5 | sf_native | **0.86032** | 0.96928 |

**关键结论**: 所有 3 个 ours 方法在长期 subject 和 background consistency 上
都超过 SF。SF 在长期一致性上垫底。VBench overall 分数被 quantile_map 掩盖了
这一差异。

## 4. DINO Comprehensive 结果 (长期全局表观诊断)

| Method | DINO consistency | Drift slope | CLIP | BG | Loop | Composite |
|---|---:|---:|---:|---:|---:|---:|
| pf_native | **0.9283** | -0.00231 | 0.2961 | 0.9135 | 0.1118 | 0.6428 |
| ours_motion1 | 0.9052 | **-0.00223** | 0.2958 | 0.9114 | 0.1150 | 0.6418 |
| ours_retrieval1_age24 | 0.9047 | -0.00236 | 0.2961 | 0.9117 | 0.0933 | **0.6443** |
| ours_retrieval_motion | 0.9052 | -0.00246 | 0.2960 | 0.9103 | 0.1093 | 0.6385 |
| sf_native | 0.8867 | **-0.00461** | 0.2952 | 0.9092 | 0.0898 | 0.6433 |

### 4.1 关键发现

1. **DINO consistency**: PF (0.9283) > Ours (~0.905) > SF (0.8867)
   - 所有 ours 方法的聚合均值高于 SF (+0.019)
   - DINO 测量 30 秒范围内的全局表观特征相似度，不受 quantile mapping 影响

2. **Drift slope**: SF (-0.00461) 比 Ours (-0.0024) **漂移快 2 倍**
   - 当前斜率单位是每个均匀采样点，不是每秒
   - 该聚合结果支持 SF 漂移更快，仍需逐 prompt 配对统计

3. **Composite**: Ours (retrieval1_age24, 0.6443) > SF (0.6433) > PF (0.6428)
   - 这是仓库内部手工加权诊断分，差值很小，不作为论文主结果

4. **CLIP text alignment**: 所有方法非常接近 (0.295-0.296)

## 5. SF Cache 分析

### 5.1 SF 使用滑窗 21 帧，无 sink

SF config (`self_forcing_dmd.yaml:51`) 设置 `local_attn_size: 21`。
SF 代码 (`causal_inference.py:1567`) 据此计算 KV cache 大小：

```
kv_cache_size = local_attn_size(21) × frame_seq_length(13) = 273 tokens = 21 帧
```

120 帧视频需要 1560 tokens，但 cache 只有 273 tokens → **必须是滑窗**。

**SF cache = 滑窗 21 帧 (~1.3 秒) + 无 sink + 无历史管理**

### 5.2 三种方法 cache 对比

| 方法 | cache 构成 | 总预算 | 长期记忆 |
|---|---|---|---|
| SF | 滑窗 21 帧 | 21 FFE | **无** (1.3 秒后遗忘) |
| PF | sink + recent + Anchor/Wave/Veil | ~9-12 FFE | 有 |
| Ours | sink1 + Landmark4 + recent4-7 + per-role middle | 9 FFE | 有 |

SF 只能记住最近 1.3 秒的视频内容，这解释了：
- VBench inclip 高 (2 秒 clip 内一致)
- VBench clip2clip 低 (长期漂移)
- DINO consistency 低 (全局表观漂移)
- Drift slope 大 (漂移快)

## 6. 与 PF 论文对齐分析

### 6.1 PF 论文报告的维度 (pf_result.md)

PF 论文在 30 秒视频上报告了 8 个维度 (0-100 scale)：

| Method | Dynamic Degree | Motion Smoothness | Overall Consistency | Imaging Quality | Aesthetic Quality | Quality Score | Semantic Score | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Self Forcing | 44.34 | 98.52 | 24.66 | 70.66 | 63.22 | 87.14 | 54.31 | 80.57 |
| + Pyramid Forcing | 55.07 | 98.82 | 25.26 | 72.17 | 66.55 | 88.93 | 55.52 | 82.25 |

### 6.2 我们的评测结果 (×100 与 PF 论文对比)

| Dimension | Our SF | Paper SF | Diff | Our PF | Paper PF | Diff |
|---|---:|---:|---:|---:|---:|---:|
| Motion Smoothness | 98.97 | 98.52 | +0.45 | 98.73 | 98.82 | -0.09 |
| Imaging Quality | 70.55 | 70.66 | -0.11 | 71.88 | 72.17 | -0.29 |
| Aesthetic Quality | 62.35 | 63.22 | -0.87 | 64.94 | 66.55 | -1.61 |

**对齐分析**:
- **Imaging Quality 几乎完美对齐** (diff < 0.3)
- **Motion Smoothness 接近** (diff < 0.5)
- **Aesthetic Quality 有小差距** (diff 0.9-1.6)，可能因为：
  1. 我们用 32 prompts，PF 论文用 128 prompts
  2. 不同 prompt 子集的 aesthetic 分布不同
  3. VBench 版本可能略有差异

### 6.3 缺失维度

| 维度 | 可评测？ | 原因 |
|---|---|---|
| Dynamic Degree | ✗ | RAFT 模型 (Dropbox 被 proxy 阻断) |
| Overall Consistency | 未评测 | 需要 CLIP + 10s clip (可补充) |
| Quality Score | ✗ | 需要全部 quality 维度 |
| Semantic Score | ✗ | 需要 VBench 标准 prompts (非 MovieGen) |
| Total Score | ✗ | 需要全部 16 维度 |

### 6.4 对齐建议

1. **补充 overall_consistency** (CLIP 可用，10s clip)
2. **在 128 prompts 上评测** (与 PF 论文对齐)
3. **尝试其他方式获取 RAFT 模型** (如手动下载后通过共享文件系统分发)
4. **报告原始 clip2clip 分数** 作为 VBench-Long 的补充

## 7. 评测有效性评估

### 7.1 VBench-Long overall 分数

**问题**: quantile_map 压缩了方法间长期一致性差异，导致 overall 分数不能
反映实际视觉质量差异。

**但仍有效**: 
- 提供与 PF 论文的标准对比 (imaging 对齐良好)
- aesthetic 和 imaging 分数显示 Ours 优于 SF
- 是论文中必需的标准评测

### 7.2 原始 clip2clip 分数

**有效**: 直接展示长期一致性排名 (Ours > SF)，但不是标准报告指标。
建议在论文中作为补充分析。

### 7.3 DINO Comprehensive

**有用的补充诊断**: 测量 30 秒范围的 DINO 全局表观一致性和 drift slope，
不受 quantile mapping 影响。聚合均值支持 Ours 优于 SF，但 DINO 也会奖励
静止或重复，且逐 prompt 置信区间仍待补齐。

### 7.4 评测策略建议

论文中应报告：
1. VBench-Long overall 分数 (标准对比，与 PF 论文对齐)
2. 原始 clip2clip 分数 (展示长期优势，分析 quantile_map 问题)
3. DINO consistency 和 drift slope (直接证据)
4. 人工 review (最终判据)

## 8. 文件位置

- VBench-Long 第1批: `runs/v120_moviebench32_main/ours1_37d836e14db1/metrics/vbench_long_summary.{md,json}`
- VBench-Long 第2批: `runs/v120_moviebench32_main/ours_only2_9b9ca1a08d27/metrics/vbench_long_summary.{md,json}`
- 合并 5 方法 VBench: `runs/v120_moviebench32_main/comparison_all5/metrics/vbench_long_summary.{md,json}`
- DINO 第1批: `runs/v120_moviebench32_main/ours1_37d836e14db1/metrics/comprehensive_parts/`
- DINO 第2批: `runs/v120_moviebench32_main/ours_only2_9b9ca1a08d27/metrics/comprehensive_parts/`
- PF 论文结果: `pf_result.md`
