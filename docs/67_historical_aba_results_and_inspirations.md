# 历史 A-B-A 实验结果汇总与启发

> 日期: 2026-07-22  
> 目的: 汇总所有 A-B-A 场景切换实验的关键数值和发现，为当前纯长视频方向提供启发  
> 数据来源: docs/55-66, runs/hrem_v2_evidence_s0, runs/hrem_v2_gate_sweep, runs/hrem_smoke

## 1. CEMR 时代（PF backend, docs/55）

### 1.1 Oracle 诊断实验（v62-v64）

| 方法 | P1 full return margin | P1 BG return margin | Validity |
|---|---|---|---|
| PF no-CFG (v62) | +0.0073 | -0.1132 | pass |
| PF CFG3 (v62) | +0.2042 | +0.0103 | pass（画质退化） |
| A1 oracle prior-on (v64) | +0.1428 | -0.1133 | pass |
| A1 oracle prior-off (v64) | +0.2060 | +0.0967 | pass |
| CEG strict (v63b) | +0.0469 | -0.0979 | pass |
| CEG relative (v63b) | -0.3967 | -0.3998 | pass |
| PF uniform head (v66, ablation) | -0.2210 | -0.0726 | **fail** |

**关键启发**:
- Oracle episode 0 将 margin 从 -0.35 提升到 +0.14 → **episode selection 是主要瓶颈**
- CEG relative winner=A1 但 margin=-0.40 → **correct episode 是 necessary but not sufficient**
- PF uniform head ablation fail → **head specialization 是 PF 效果的主要来源**

### 1.2 CEMR 32-prompt VBench 结果

| 维度 | native | CEMR | Δ |
|---|---|---|---|
| Subject consistency | baseline | +小幅 | ✓ |
| Background consistency | baseline | +小幅 | ✓ |
| Aesthetic | baseline | +小幅 | ✓ |
| Imaging | baseline | +小幅 | ✓ |
| Motion smoothness | baseline | +小幅 | ✓ |
| **Dynamic degree** | baseline | **-0.025** | ✗ |

**关键启发**: all-heads 历史注入导致 Dynamic 下降 → **需要 per-head 控制**

## 2. HREM-v1 时代（SF backend, docs/57-58）

### 2.1 SF native A-B-A 硬切换

| 配置 | A1-A2 | A1-B | B-A2 | margin | B形成? |
|---|---|---|---|---|---|
| SF native (rooftop\|\|gym\|\|rooftop) | 0.7790 | 0.9048 | 0.8835 | -0.1045 | ❌ |

**关键启发**: SF 的场景惯性太强，hard-cut prompt 不足以切换场景。

### 2.2 HREM-v1 固定 head 清除（3 seeds）

| Seed | native A1-A2 | hrem_v1 A1-A2 | Δ |
|---|---|---|---|
| 0 | 0.8453 | 0.8736 | +0.0283 |
| 1 | 0.7987 | 0.7907 | -0.0080 |
| 2 | 0.8432 | 0.7830 | -0.0602 |
| **Mean** | 0.8291 | 0.8157 | **-0.0133** |

**关键启发**: 固定 head split（0:4=layout, 4:8=texture, 8:12=motion）**不可靠**，seed 0 的正面效果是偶然。

### 2.3 K-stability 校准实验

| Layer | #Structure | #Detail | min_sim | max_sim | mean_sim |
|---|---|---|---|---|---|
| 0-29 (all) | 12/12 | 0/12 | 0.91-0.97 | 0.98-1.00 | 0.95-0.99 |

**关键启发**: clean context 下的 K embedding **跨 seed 完全确定** → **K-stability 无法区分 head 角色**。

### 2.4 Echo-Forcing 2-scene 验证

| Cell | frame 0 mean/std | frame 110 mean/std | MD5 |
|---|---|---|---|
| EF native | 126.4/63.1 | 120.0/61.0 | 22a633c4 |
| EF + HREM | 126.4/63.1 | **103.2/82.0** | 1de508af |

**关键启发**: HREM 在 EF 上改变了 B 场景统计（std 从 61→82），说明 per-head 清除在 EF 的 scene pool 基础上有效果。

## 3. HREM-v2 时代（SF backend, docs/61-66）

### 3.1 五 cell 主实验（3 prompts × seed 0）

#### Prompt 0 (陶艺→地铁→陶艺)

| Cell | A1-A2 | A1-B | B-A2 | margin | MD5 |
|---|---|---|---|---|---|
| native_raw | 0.6775 | 0.8924 | 0.8839 | -0.2064 | 4fbf40ec |
| native_reset | 0.6233 | 0.1631 | 0.1860 | +0.4372 | b4f86e79 |
| oracle_episode0 | 0.6069 | 0.1650 | 0.1787 | +0.4281 | 321fde0f |
| dual_episode_only | 0.6082 | 0.1656 | 0.1783 | +0.4300 | 25ff0bcd |
| **hrem_v2** | **0.6848** | 0.1665 | 0.1887 | **+0.4961** | de44fe93 |

#### Prompt 1 (天文台→温室→天文台)

| Cell | A1-A2 | A1-B | B-A2 | margin | MD5 |
|---|---|---|---|---|---|
| native_raw | 0.8393 | 0.9540 | 0.9420 | -0.1027 | 462b1ab2 |
| **native_reset** | **0.5196** | 0.2502 | 0.1838 | **+0.3358** | cd2f9a96 |
| oracle_episode0 | 0.4578 | 0.2478 | 0.2164 | +0.2413 | 4b5af3ec |
| dual_episode_only | 0.4611 | 0.2486 | 0.2164 | +0.2447 | 877b1c43 |
| hrem_v2 | 0.4665 | 0.2499 | 0.2010 | +0.2654 | 495d67f4 |

#### Prompt 2 (餐车→剧场→餐车)

| Cell | A1-A2 | A1-B | B-A2 | margin | MD5 |
|---|---|---|---|---|---|
| native_raw | 0.8642 | 0.9518 | 0.9437 | -0.0795 | 8340a1d0 |
| native_reset | 0.6918 | 0.2821 | 0.1909 | +0.5008 | 70fe5c6a |
| **oracle_episode0** | **0.7053** | 0.2847 | 0.1880 | **+0.5173** | bc1e8294 |
| dual_episode_only | 0.7069 | 0.2856 | 0.1905 | +0.5164 | a52db170 |
| hrem_v2 | 0.6891 | 0.2862 | 0.1771 | +0.5120 | c26d95e4 |

### 3.2 关键发现

1. **native_raw 不是合法 baseline**（B 不形成，A1-B=0.89-0.95）
2. **native_reset 是正确 baseline**（B 形成，A1-B=0.16-0.28）
3. **HREM-v2 在 P0 上最优**（margin +0.4961 vs native_reset +0.4372, Δ=+0.059）
4. **HREM-v2 在 P1 上退化**（margin +0.2654 vs native_reset +0.3358, Δ=-0.070）
5. **Trace audit 0 violations**，episode route 全部正确 `2→0`
6. **Head gate mean=0.93**，几乎不 selective

### 3.3 Gate threshold sweep（P0 only, seed 0）

| Threshold | P0 margin | P1 margin | P2 margin | Mean margin |
|---|---|---|---|---|
| native_reset | +0.4372 | +0.3358 | +0.5008 | +0.4246 |
| t=0.45 | +0.4961 | +0.2654 | +0.5120 | +0.4245 |
| t=0.55 | +0.4622 | +0.2887 | +0.5054 | +0.4188 |
| t=0.65 | +0.4464 | +0.3251 | +0.5279 | +0.4331 |
| **t=0.75** | +0.4693 | **+0.3495** | +0.4969 | **+0.4386** |
| t=0.85 | +0.4488 | +0.3486 | +0.4963 | +0.4312 |

**关键启发**: t=0.75 是最优 threshold（+0.0140 vs native_reset），但肉眼不明显。

### 3.4 人工 Review 发现

| 问题 | 严重程度 | 原因 |
|---|---|---|
| 场景切换伪影 | 中 | cache 硬清零 + memory 突然激活 |
| A2 identity 变化 | 中 | noisy_only readout + position_mode=none |
| DINO 指标局限性 | 高 | 无法检测 identity/motion/artifact |
| 仅 A-B-A 拼接 | 高 | 不是真正的长视频生成 |

## 4. 对当前纯长视频方向的启发

### 4.1 从 CEMR 学到的

1. **Episode selection > fusion strength**: oracle 把 margin 从 -0.35 提到 +0.14，说明"选对历史"比"融合多少"更重要
2. **All-heads 注入伤 Dynamic**: 需要 per-head 控制，但固定 split 不可靠
3. **CEG relative 失败**: correct episode 是 necessary but not sufficient → within-episode retrieval 和 fusion 也要正确

### 4.2 从 HREM-v1 学到的

1. **固定 head split 不可靠**: seed 0 正面但 seed 1/2 负面 → 需要在线 evidence
2. **K-stability 无法区分 head**: clean K 跨 seed 确定性太高 → 需要其他 evidence（如 query drift, V persistence）
3. **SF 场景惯性极强**: hard-cut 不工作 → 需要 cache reset

### 4.3 从 HREM-v2 学到的

1. **native_reset 是必需的**: cache reset 强制 B 形成
2. **Episode admission 100% 正确**: dual evidence route 全部 `2→0` → episode selection 已解决
3. **Head gate 不够 selective**: mean=0.93 → 需要更高 threshold 或更好的 evidence
4. **DINO 不够**: 需要 VBench-Long 衡量 identity/motion/artifact
5. **A-B-A 拼接 ≠ 长视频**: 当前方法在纯长视频中完全不激活

### 4.4 对 intra_episode scope 的设计启发

| 启发 | 设计建议 |
|---|---|
| Episode selection 已解决 | intra_episode 不需要 episode admission，用 age-based exclusion |
| Head gate 需要 selectivity | 用 K/V persistence（跨帧变化率）而非 K-stability |
| All-heads 伤 Dynamic | head gate 在 intra_episode 中更重要（持续注入 vs 一次性注入） |
| position_mode=none 丢 identity | 尝试 local_grid 或 MemRoPE |
| noisy_only 可能不够 | 尝试 all 或 clean_only |
| DINO 不够 | 用 VBench-Long + ArcFace identity |

## 5. 当前实验状态

### 正在运行

| Cell | 状态 | MP4s |
|---|---|---|
| sf_native | 生成中（prompt 1/3） | 1 |
| sf_pyramid_forcing | 等待 | 0 |
| sf_echo_forcing | 等待 | 0 |
| ours_all_heads | 等待 | 0 |
| ours_role | 等待 | 0 |

### 人工 Review 流程（docs/69 §6）

实验完成后自动生成 `blind_review/` 目录：

```
runs/paper_single_30s_s0/blind_review/
├── manifest_public.json    ← reviewer 可见的随机标签
├── scorecard.csv           ← 评分表（冻结后才能解盲）
├── key_private.json        ← 方法映射（评分冻结后才看）
└── prompt_00/A.mp4         ← 视频硬链接
```

**Review 步骤**:
1. 打开 `blind_review/` 目录
2. 对每个 prompt 的 5 个视频（A-E 随机标签）逐个观看
3. 在 `scorecard.csv` 中记录：
   - 主体 identity（面部、服装、几何、特征物体）
   - 背景/布局漂移和场景污染
   - 运动幅度、自然度、冻结、循环、重复主体
   - 镜头方向和连续性
   - 首次可见失败时间
   - prompt 对齐度和总体排名
4. **冻结 scorecard.csv**
5. 运行 `HUMAN_REVIEW_DONE=1 bash scripts/run_paper_metrics.sh single 0` 解盲并计算指标
