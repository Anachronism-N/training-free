# HREM-v2 实验结果与迭代分析

> 日期: 2026-07-22  
> Base: Self-Forcing (Wan2.1-T2V-1.3B, DMD蒸馏)  
> 实验: 3 prompts × 5 cells × 120 frames, seed 0
> P0 head-role 校准实现与服务器命令见 `docs/63_hrem_v2_p0_role_calibration.md`。

## 1. 代码依赖说明

### 核心 Backend

| 组件 | 来源 | 作者/机构 | 角色 |
|---|---|---|---|
| Wan2.1-T2V-1.3B | Alibaba Wan-AI | 预训练视频扩散 transformer | 模型 backbone |
| Self-Forcing DMD | `third_party/Self-Forcing/` | 第三方 repo (distilled AR inference) | 推理管线 |
| **EpisodicArchive** | `src/lifecycle_kv/episodic_archive.py` | **本项目实现** | bounded episodic memory |
| **RoleEpisodic** | `src/lifecycle_kv/role_episodic.py` | **本项目实现** | dual-evidence episode selector + online head gate |
| **AttentionFusion** | `src/lifecycle_kv/attention_fusion.py` | **本项目实现** | memory attention + bounded convex fusion |
| **SF Pipeline Bridge** | `third_party/Self-Forcing/pipeline/causal_inference.py` (修改) | 本项目对 SF 的集成修改 | episode生命周期、clean commit、scene boundary |
| **SF Model Bridge** | `third_party/Self-Forcing/wan/modules/causal_model.py` (修改) | 本项目对 SF 的集成修改 | pre-RoPE capture、memory readout、head gate |

### 可借鉴的基础设施

本项目新增的 archive/gate/fusion 模块构成了 training-free episodic-memory 插件。Archive、Q-K retrieval 和 memory attention 等单项均有相关工作，最终原创性声明应聚焦 factorized episode/head admission，并等待完整文献检索和消融。该实现理论上可移植到其他 AR video DiT：
- **Self-Forcing** (当前已集成，验证通过)
- **Echo-Forcing** (scene pool 概念可对比)
- **Pyramid-Forcing** (head specialization 概念可对比)
- **Causal-Forcing** (代码已集成，缺少 checkpoint)

## 2. 实验结果

### 2.1 B formation 验证

| Prompt | native_raw (A1-B) | native_reset (A1-B) | 结论 |
|---|---|---|---|
| 0 (陶艺→地铁→陶艺) | 0.89 (B未形成) | **0.16** (B形成!) | ✅ |
| 1 (天文台→温室→天文台) | 0.95 (B未形成) | **0.25** (B形成!) | ✅ |
| 2 (餐车→剧场→餐车) | 0.95 (B未形成) | **0.28** (B形成!) | ✅ |

`native_raw` 的 B 段始终留在 A 场景（场景惯性）。`native_reset` 通过在 `||` 边界 reset working cache 强制形成 B → **正确 baseline**。

### 2.2 A2 return 验证

#### Prompt 0 (陶艺→地铁→陶艺) — HREM 最优

| Cell | A1-A2 | margin |
|---|---|---|
| native_reset | 0.62 | +0.44 |
| oracle_episode0 | 0.61 | +0.43 |
| dual_episode_only | 0.61 | +0.43 |
| **hrem_v2** | **0.68** | **+0.50** |

**在该 prompt、该 seed 上 HREM 优于 native_reset**，A1-A2 +0.06，margin +0.06。A2 成功回归到陶艺工作室；单样本差值不能称为统计显著。

#### Prompt 1 (天文台→温室→天文台) — native_reset 最优

| Cell | A1-A2 | margin |
|---|---|---|
| **native_reset** | **0.52** | **+0.34** |
| oracle_episode0 | 0.46 | +0.24 |
| dual_episode_only | 0.46 | +0.24 |
| hrem_v2 | 0.47 | +0.27 |

HREM 退化。可能原因：head gate 干扰了 native_reset 已有效的场景切换。

#### Prompt 2 (餐车→剧场→餐车) — oracle/dual 最优

| Cell | A1-A2 | margin |
|---|---|---|
| native_reset | 0.69 | +0.50 |
| oracle_episode0 | **0.71** | **+0.52** |
| dual_episode_only | **0.71** | **+0.52** |
| hrem_v2 | 0.69 | +0.51 |

所有方法都接近。Episode selection 本身已接近 optimal，head gate 可有可无。

### 2.3 人工 Review 发现的问题

#### 问题 1: A2 主体 identity 变化

**现象**：A2 虽然回到了陶艺工作室/天文台/餐车的布局和背景，但人物（着装、姿态、外观细节）与 A1 不同。

**可能原因**：
- `readout_mode=noisy_only` → 只读取 noisy-only 层的 K/V，这些层可能更关注纹理而非 identity
- `position_mode=none` → 没有位置信息，pre-RoPE K/V 可能丢失空间对应
- Archive 覆盖率不足：只有 36 帧分布在 3 个 episode 中，每个 episode 的 identity 帧可能被 evict

**迭代方向**：
- 尝试 `readout_mode=all` 替代 `noisy_only`
- 加入 MemRoPE 的位置安全 readout
- 增大 archive budget 到 48 帧

#### 问题 2: 场景切换伪影

**现象**：|| 边界出现短暂的重影、纹理粘连、闪烁。

**可能原因**：
- `memory_start_episode=2` → memory branch 在 A1→B 时不激活，但在 B→A2 时突然激活，导致 abrupt change
- `SCENE_TRANSITION_RESET=1` → working cache 清零是硬切换，没有 smooth transition
- fusion gate=0.10 可能太小 → memory contribution too weak to smooth

**迭代方向**：
- 调大 `MEMORY_GATE` 到 0.15-0.20
- 尝试 warm start：B→A2 transition 的 memory branch 逐渐激活而不是突然激活
- 在 scene boundary 应用 temporal blending

#### 问题 3: Head gate 不够 selective

**现象**：diagnostic 显示 `head_gate_mean=0.93`, `accepted_head_fraction=0.99`

**字段校正**：`accepted_head_fraction` 来自 retrieval admission，不是 role gate active fraction。后续 trace 已拆分为 `retrieval_accepted_head_fraction` 与 `role_active_head_fraction`；`head_gate_mean=0.93` 仍足以说明旧 role gate 过强。

**原因**：threshold=0.45/sharpness=8.0 组合下几乎所有 head 都通过。head gate 几乎没有过滤作用。

**迭代方向**：
- 提高 `ROLE_THRESHOLD` 到 0.60-0.75
- 降低 `ROLE_SHARPNESS` 到 4.0
- 或者改用 per-head 统计阈值（auto-calibrated from clean K/V persistence）

## 3. 当前方法的主要问题

### 3.1 架构问题

| 问题 | 严重程度 | 修复方向 |
|---|---|---|
| **Pre-RoPE K/V 丢失空间对应** | 高 | position_mode=local_grid 或 MemRoPE 校正 |
| **Head gate 未发挥作用** | 高 | 提高阈值、统计校准 |
| **Readout mode 影响 identity** | 中 | sweep all/clean_only/noisy_only |
| **Archive budget 32-36 偏小** | 中 | 增大到 48 |

### 3.2 实验问题

| 问题 | 严重程度 | 修复方向 |
|---|---|---|
| **仅 1 seed** | 高 | seed 0,1,2 三 seed |
| **仅 3 prompts** | 中 | 扩展到 6-8 prompts |
| **仅 SF backend** | 中 | Echo-Forcing 对比 |
| **无 VBench 质量评估** | 中 | 需要 32-prompt VBench |

### 3.3 方法层面问题

| 问题 | 严重程度 | 修复方向 |
|---|---|---|
| **Boundary 依赖 hard-coded || schedule** | 中 | 论文必须明确这是任务设定 |
| **Dual evidence selector 在 A-B-A 中只有 1 个候选** | 低 | A-B-C-A 多候选验证 |
| **Contrastive gate 依赖 prompt descriptor** | 中 | 需要视觉-only 的 fallback |

## 4. 迭代优先级

### P0 (本周必须完成)

1. **Head gate 选择性修复** — 提高 threshold，运行 ablation sweep
2. **Multi-seed 验证** — seed 1, 2 重跑 5 cells
3. **Position mode 实验准备** — 当前 stride-4 archive 与 local_grid 不兼容；先完成 head gate P0，再实现 pooled-grid position 或运行 stride-1 上界

### P1 (下周)

4. **VBench 32-prompt 质量评估** — 确保不降质
5. **Multi-prompt 扩展** — 6-8 A-B-A prompts
6. **Ablation 实验** — shuffled V, wrong episode, all heads

### P2 (后续)

7. **Echo-Forcing 集成** — 独立 backend 验证
8. **Multi-candidate A-B-C-A 测试**
9. **Memory/latency 开销报告**

## 5. 当前可以声明的结论

- ✅ `SCENE_TRANSITION_RESET` 机制有效 — 正确 baseline 建立
- ✅ Episode admission (dual evidence) 在 3 prompts 上 route 100% 正确 (2→0)
- ✅ Trace audit 0 violations
- ✅ HREM 在 Prompt 0、seed 0 上优于 native_reset (+0.06 A1-A2)，尚不构成统计显著性
- ⚠️ Head gate 尚未发挥区分作用 (mean 0.93)
- ⚠️ A2 identity 回归不完全（人工 review 确认）
- ⚠️ 场景切换存在伪影
