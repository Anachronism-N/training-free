# HREM-v2 Gate Sweep 结果与人工 Review 反馈

> 日期: 2026-07-22  
> 实验: head gate threshold sweep (0.45/0.55/0.65/0.75/0.85) × 3 prompts × seed 0  
> 人工 Review: 用户已完成视频观看

## 1. Gate Sweep 定量结果

### 1.1 DINOv2 ViT-S/14 Return Margin

| Threshold | P0 (陶艺) | P1 (天文台) | P2 (餐车) | **Mean** | Δ vs native_reset |
|---|---|---|---|---|---|
| native_reset (baseline) | +0.4372 | +0.3358 | +0.5008 | +0.4246 | — |
| t=0.45 (原配置) | **+0.4961** | +0.2654 | +0.5120 | +0.4245 | -0.0001 |
| t=0.55 | +0.4622 | +0.2887 | +0.5054 | +0.4188 | -0.0058 |
| t=0.65 | +0.4464 | +0.3251 | **+0.5279** | +0.4331 | +0.0085 |
| **t=0.75** | +0.4693 | **+0.3495** | +0.4969 | **+0.4386** | **+0.0140** |
| t=0.85 | +0.4488 | +0.3486 | +0.4963 | +0.4312 | +0.0066 |

### 1.2 DINOv2 A1-A2 Similarity

| Threshold | P0 | P1 | P2 | **Mean** |
|---|---|---|---|---|
| native_reset | 0.6233 | 0.5196 | 0.6918 | 0.6116 |
| t=0.45 | **0.6848** | 0.4665 | 0.6891 | 0.6135 |
| t=0.65 | 0.6311 | 0.5260 | **0.7074** | **0.6215** |
| t=0.75 | 0.6515 | 0.5218 | 0.6836 | 0.6190 |
| t=0.85 | 0.6204 | **0.5269** | 0.6925 | 0.6133 |

## 2. 人工 Review 发现的问题

### 2.1 场景切换伪影

**现象**: `||` 边界处出现短暂重影、纹理粘连、闪烁。

**原因分析**:
- `SCENE_TRANSITION_RESET=1` → working cache 硬清零，没有 smooth transition
- Memory branch 在 B→A2 时突然激活（`memory_start_episode=2`）
- Fusion gate=0.10 可能导致 memory contribution 不足以平滑过渡

**影响**: 所有 threshold 配置都有此问题，t=0.75 肉眼与其他 threshold 无明显区别。

### 2.2 跨场景 Identity 不保持

**现象**: A2 虽然回到了 A1 的场景布局（陶艺工作室/天文台/餐车），但人物着装、姿态、外观细节与 A1 不同。

**原因分析**:
- `readout_mode=noisy_only` → 只读取 noisy 层 K/V，可能丢失 identity 信息
- `position_mode=none` → pre-RoPE K/V 没有位置对应，无法精确恢复人物位置
- Archive budget=36 帧分布在 3 个 episode 中，identity 关键帧可能被 evict
- DINOv2 指标无法检测 identity 问题（DINO 更关注场景整体相似度）

### 2.3 DINOv2 指标的局限性

**问题**: DINOv2 场景级相似度不能准确衡量：
- Identity 保持（人物外观一致性）
- Motion 自然度（是否冻结/重复）
- 切换伪影严重度

**用户建议**: **应使用 VBench-Long** 作为主指标，DINO/motion 作为辅助诊断。

### 2.4 实验范围局限

**当前实验**: 仅做了 A-B-A 场景切换实验（3段×10s拼接），**未做单纯长视频实验**。

**问题**:
- A-B-A 相当于 3 次独立 10s 推理的拼接，不是真正的长视频生成
- 无法验证 HREM 在连续长视频（如 30s/60s 单场景）中是否有效
- 论文需要同时覆盖 scene transition 和 long video consistency 两类任务

## 3. 实验配置记录

### 3.1 native_raw 配置（runs/hrem_v2_evidence_s0/native_raw）

```
STRUCTURED_MEMORY_ENABLE=0    # 完全关闭 memory
SCENE_TRANSITION_RESET=0      # 不在 || 边界重置 KV cache
```

纯 Self-Forcing 原生推理。B 场景不形成（场景惯性），不是合法 baseline。

### 3.2 native_reset 配置（正确 baseline）

```
STRUCTURED_MEMORY_ENABLE=0    # 关闭 memory
SCENE_TRANSITION_RESET=1      # 在 || 边界重置 working cache
```

强制 B 形成但无历史 memory。**这是正确的 episodic return baseline**。

### 3.3 HREM-v2 gate sweep 配置

```
STRUCTURED_MEMORY_ENABLE=1
STRUCTURED_MEMORY_GATE=0.10
STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES=36
STRUCTURED_MEMORY_LAYER_START=15, END=21
STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence
STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2
STRUCTURED_MEMORY_HEAD_ROUTING=role_evidence
STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0
STRUCTURED_MEMORY_ROLE_THRESHOLD={0.45, 0.55, 0.65, 0.75, 0.85}
SCENE_TRANSITION_RESET=1
```

## 4. 当前实验的 4 个主要局限

| 局限 | 严重程度 | 修复方向 |
|---|---|---|
| **仅 DINOv2 指标，无 VBench-Long** | 🔴 高 | 必须 32-prompt VBench-Long 评估 |
| **仅 A-B-A 拼接，无纯长视频** | 🔴 高 | 需要 30s/60s 单场景连续生成实验 |
| **场景切换伪影** | 🟡 中 | smooth transition / temporal blending |
| **Identity 不保持** | 🟡 中 | readout_mode=all / position_mode=local_grid / 增大 archive |
| **t=0.75 肉眼无区别** | 🟡 中 | threshold sweep 在 DINO 上有差异但视觉不明显 |
| **仅 1 seed** | 🟡 中 | 需要 3 seeds 统计验证 |

## 5. 代码变更记录

### 本轮拉取（vs 上次）新增 commit

| Commit | 内容 |
|---|---|
| `768c704` | feat: calibrate HREM head-role routing |
| `d2262b6` | docs: audit related work provenance and claims (docs/64) |
| `4dff83f` | docs: SWIFT collision audit + head gate sweep script (docs/65) |

### 代码修改

- `scripts/run_hrem_v2_evidence.sh`: 修复 conda 激活路径
- `scripts/wrap_hrem_v2_seq.sh`: 顺序执行 5 cell 的 wrapper
- `scripts/run_headgate_sweep.sh`: threshold sweep 脚本

## 6. 下一步优先级

### P0: 必须完成

1. **VBench-Long 评估** — 在 32 prompts 上跑 native_reset vs HREM t=0.75，用 VBench-Long 6 维度评估
2. **纯长视频实验** — 30s/60s 单场景连续生成，验证 HREM 不降质
3. **Multi-seed** — t=0.75 × seed 0,1,2 × 3 prompts 统计验证

### P1: 应该完成

4. **场景切换伪影修复** — smooth transition 或 temporal blending
5. **Identity 保持改进** — readout_mode sweep / position_mode=local_grid
6. **SWIFT ablation 对比** — 同 archive，head evidence 用 alignment vs persistence

### P2: 可以延后

7. **Echo-Forcing 集成** — 跨 backend 验证
8. **A-B-C-A 多候选测试** — dual evidence selector 压力测试
9. **Memory/latency 开销报告**

## 7. 论文故事调整建议

基于人工 review 反馈，论文故事需要调整：

### 当前 Story A 的问题

- DINO 改善 +0.014 但肉眼不明显 → reviewer 会质疑实际效果
- 场景伪影 + Identity 变化 → 质量问题可能被 VBench 暴露
- 仅 A-B-A 拼接 → 不够全面

### 建议调整

1. **主指标改为 VBench-Long**（subject consistency, background consistency, aesthetic, imaging, motion smoothness, dynamic degree）
2. **增加纯长视频实验**（30s 单场景，证明 HREM 不降质）
3. **A-B-A 作为辅助分析**（diagnostic，不是主 claim）
4. **如果 VBench-Long 无显著改善 → 降级到 Story C（diagnostic paper）**
