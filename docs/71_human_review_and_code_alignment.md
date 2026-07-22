# 首轮 30s 纯长视频实验：人工 Review 反馈与代码对齐确认

> 日期: 2026-07-23
> 实验: `runs/paper_single_30s_s0/`，3 prompts × 5 cells × 120 latent frames (≈30s)
> Review 方式: 人工观看全部已生成视频，文字描述

## 1. 实验完成状态

| Cell | MP4s | 状态 | 说明 |
|---|---|---|---|
| sf_native | 3/3 | ✅ | 纯 Self-Forcing baseline |
| sf_pyramid_forcing | 3/3 | ✅ | Pyramid-Forcing official config |
| sf_echo_forcing | 0/3 | ❌ OOM | Echo 比 SF/PF 多用内存，50GB+43GB>95GB |
| ours_all_heads | 3/3 | ✅ | 但视觉上与 sf_native 无区别 |
| ours_role | 3/3 | ✅ | 但视觉上与 sf_native 无区别 |

总计: 12/15 MP4s 生成成功。

## 2. 人工 Review 逐 cell 反馈

### 2.1 sf_native

**0-0**:
- 5s 后 ID 开始退化，画面逐渐变暗
- 10s 后 ID 退化明显
- 20s 后几乎不可用且很暗，出现幻觉

**1-0**:
- ID 保持尚可
- 后续背景变暗退化严重

**结论**: SF native 的 21 帧滑动窗口在 30s 纯长视频上严重退化，符合预期（21帧 ÷ 4 ≈ 5.25s 后窗口完全遗忘早期帧）。

### 2.2 sf_pyramid_forcing

**整体**:
- ID 和背景保持很好，30s 几乎都保持得不错
- 没有过多的镜头大移动，但有轻微的镜头晃动
- **存在不连贯的跳变**: 不是突兀的 ID 或背景跳变，而是可以明确看出部分几帧的变化速度相比其他帧快很多，形成的跳变

**1-0 特殊问题**:
- 前几帧存在闪回和伪影
- 后期出现两个主体人物和背景幻觉（高运动场景下存在问题）

**跳变原因分析**: PF 使用 per-head 异构 cache 策略（osc/sta+/sta-），不同 head 的 cache 组成不同（sink1/3 + cyclic/stride/merge + recent4），可能导致帧间不一致。需对比官方 PF repo 输出确认是否为固有行为。

### 2.3 ours_all_heads

**与 sf_native 无视觉区别。**

原因:
- `STRUCTURED_MEMORY_GATE=0.05` — memory 分支最多贡献 5% 输出，太低
- `STRUCTURED_MEMORY_MEMORY_START_FRAME=36` — 9 秒后才激活，前 9 秒完全等同 native
- `STRUCTURED_MEMORY_HEAD_ROUTING=off` — all heads 通过，无选择性

### 2.4 ours_role

**与 sf_native 无视觉区别。**

原因与 ours_all_heads 相同。此外:
- `STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45` — gate mean=0.93，几乎所有 head 通过
- role evidence 的 K/V persistence 和 query drift 未能产生有选择性的 gate

### 2.5 sf_echo_forcing

**OOM 崩溃**，无视频输出。

```
GPU: 50.56 GiB (其他用户) + 42.68 GiB (Echo) = 93.24 GiB > 95 GiB
```

Echo 的 scene pool + decay + compress 基础设施比 SF/PF 多用约 5GB 显存。

## 3. 代码对齐确认

### 3.1 SF native 配置

```yaml
# configs/self_forcing_dmd.yaml
model_kwargs:
  local_attn_size: 21    # 21 帧滑动窗口
  timestep_shift: 5.0
  # sink_size 未设置
```

```python
# utils/wan_wrapper.py line 123
sink_size=0    # 确认: SF native 不使用 sink token
```

**确认: SF native 就是简单的 21 帧滑动窗口，无 sink token。** 这与 PF/EF 论文中描述的 SF baseline 配置一致。5s 后 ID 退化是该配置的预期行为。

### 3.2 PF 配置

```yaml
# configs/pyramid-forcing.yaml
# Per-head adaptive KV-cache:
#   osc  (label -1): sink1 + cyclic(period=6, bucket_cap=4) + recent4
#   sta+ (label  1): sink3 + stride(interval=6, cap=4) + recent4
#   sta- (label  2): sink3 + merge(patch_size=2, cap=4) + recent4
```

PF 使用 `best_labels.csv` 进行 offline head 分类，每个 head 角色有不同的 cache 策略。**这是 PF 的官方配置，不是我们的复现代码。**

帧跳变可能是 PF 固有行为：不同 head 的 cache 在同一帧看到的上下文不同，导致帧间生成速度不一致。

### 3.3 ours 配置问题

```yaml
# 当前配置（导致无效果）
STRUCTURED_MEMORY_GATE=0.05              # 太低! 最多 5% 贡献
STRUCTURED_MEMORY_MEMORY_START_FRAME=36  # 9s 后才激活
STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45    # 太松! gate mean=0.93
```

**需要调整**:
- `GATE`: 0.05 → 0.15-0.20（参考 AMA 的 AAI alpha 0.05-0.15）
- `MEMORY_START_FRAME`: 36 → 12（3s 后就开始 recall）
- `ROLE_THRESHOLD`: 0.45 → 0.15-0.25（参考 AMA 的 HRMR id_thresh=0.15）

## 4. AMA/RollingForcing 经验对照

| AMA 发现 | 当前实验验证 |
|---|---|
| SF DMD 15s+ 面部崩溃 | ✅ 确认：sf_native 5s 开始退化，20s 不可用 |
| AAI gate 0.05-0.15 最优 | 当前 gate=0.05 在下限，效果不可见 |
| HRMR thresh=0.15 >> 0.25 | 当前 thresh=0.45 太松，应降到 0.15-0.25 |
| +1% DINO ≈ -7% motion | 需要注意 gate 不能太大 |
| min_DINO 比 avg DINO 更重要 | 应增加 min_DINO 指标 |
| PF 帧跳变可能来自 ragged cache | 需对比官方 PF 输出 |

## 5. 待修复问题

| 问题 | 优先级 | 修复方案 |
|---|---|---|
| ours gate 太低无效果 | P0 | gate 0.05→0.15, start_frame 36→12, thresh 0.45→0.20 |
| Echo OOM | P0 | 换更空闲 GPU 或减少 scene pool 内存 |
| PF 帧跳变 | P1 | 对比官方 PF repo 输出 |
| 无 trace 文件 | P1 | ours cells 的 trace 未生成（tokenizer 错误导致中断） |
| 无 VBench 指标 | P1 | 需要运行 `run_paper_metrics.sh` |

## 6. 下一步计划

1. **拉取最新代码** — GitHub 有新版本（`1b5db95 Implement LifeCache v3 typed memory experiments`）
2. **重跑 Echo-Forcing** — 换更空闲的 GPU
3. **调整 ours 参数后重跑** — gate=0.15, start_frame=12, threshold=0.20
4. **运行 VBench 指标** — 在所有 15 个视频上运行 6 维度评估
5. **对比官方 PF** — 确认帧跳变是否为固有行为
