# HREM-v2 P0: Head-Role Calibration and Server Feedback Protocol

> 基于 `docs/62_hrem_v2_results_and_iteration.md` 的首轮服务器结果。
> 目标：先证明 head gate 具有可重复的选择性，再讨论 identity、position mode 或更大 archive。
> 当前机器不运行 PyTorch/CUDA；所有机制判断依赖服务器 trace 与视频反馈。

## 1. 首轮结果带来的决策

首轮 Stage-1 已经建立三个重要事实：

1. `native_reset` 能让三个 prompt 的 B 场景形成，是当前公平 baseline；
2. dual-evidence route 在 A2 均为 `2 -> 0`，trace 没有因果违规；
3. 旧 HREM 的 `head_gate_mean=0.93`，head gate 基本退化为 all-head。

因此下一步不应同时改变 archive、position、readout mode 和 global gate。P0 只改变 head gate，保持以下部分不变：

```text
archive=36, spatial_stride=4, top_k=3
readout_mode=noisy_only, position_mode=none
episode_gate=dual_evidence, memory_start_episode=2
fusion=convex, global_gate=0.10, layers=15..20
```

## 2. 旧日志解释的校正

旧 diagnosis 中：

```text
accepted_head_fraction=0.99
```

这个字段来自 frame retrieval 的 `memory.accepted`，表示多少 head 通过 retrieval confidence/margin/entropy，不表示多少 head 通过 role gate。新的 analyzer 同时报告：

```text
retrieval_accepted_head_fraction
role_active_head_fraction
```

只有后者才能用于判断 head-role selectivity。

## 3. 新的校准模式

### 3.1 Absolute

保留原实现作为可复现基线：

```text
gate_abs = sigmoid(sharpness * (evidence - absolute_threshold))
```

P0 测试 threshold `0.60` 和 `0.75`。它能回答单纯提高阈值是否足够，但不同 layer 的 evidence scale 可能不同。

### 3.2 Relative

每层、每次 attention 调用内将 head evidence 归一到 `[0,1]`，再按目标 keep fraction 计算 gate：

```text
relative_evidence = (evidence - min_head) / (max_head - min_head)
relative_cutoff = quantile(relative_evidence, 1 - keep_fraction)
relative_gate = sigmoid(sharpness * (relative_evidence - relative_cutoff))
```

P0 使用 `keep_fraction=0.50`。该模式测试“相对最稳定的一半 head”是否比 all-head 更好。

### 3.3 Hybrid

```text
gate_hybrid = gate_abs * relative_gate
```

Hybrid 要求 head 同时具有足够的绝对 persistence evidence 和相对排名。若 max-min evidence spread 小于阈值，全部 role gate 置零，精确回退 native branch。

P0 使用：

```text
absolute_threshold=0.45
keep_fraction=0.50
min_evidence_spread=0.01
```

## 4. 新增诊断信息

每个 trace 首先写入 `config` 事件，包含：

- active layers；
- archive budget/policy/spatial stride 与 episode-gate 配置；
- 全部 readout、episode、role 和 fusion 配置；
- scene reset 及 LifeCache/head-role 共存方法开关；
- debug layers 与频率。

每个 `readout` 新增：

```text
trajectory_id
attention_call_index
head_routing
head_gate_mean/std/min/max/p10/p50/p90
head_gate_active_count/fraction
role_calibration
role_keep_fraction
role_evidence_mean/std/min/max/spread
role_calibration_threshold/valid
role_relative_threshold/role_relative_rank_threshold
head_role.persistent_evidence
head_role.relative_evidence
```

`trajectory_id` 防止三个 prompt 的同 layer/block 被混合统计；`attention_call_index` 用来检查同一 trajectory、同一生成 block 在不同 denoising calls 中是否反复更换 active head。

Analyzer 新增：

- role gate contrast；
- raw evidence spread；
- calibration validity；
- 同 block gate-mean range；
- active-head Jaccard；
- retrieval acceptance 与 role active fraction 分离统计。

## 5. P0 矩阵

| Cell | Calibration | 参数 | 回答问题 |
|---|---|---|---|
| `native_reset` | none | memory off | 公平 baseline |
| `dual_all_heads` | none | role off | episode-only/all-head recall |
| `role_abs_060` | absolute | threshold 0.60 | 温和提高阈值 |
| `role_abs_075` | absolute | threshold 0.75 | 强 absolute 筛选 |
| `role_relative_050` | relative | top 50%, no spread rejection | 相对排名是否有用 |
| `role_hybrid_050` | hybrid | top 50%, min spread 0.01 | 保守、可拒绝的相对筛选 |

该矩阵仍使用相同三个复杂 A-B-A prompts、120 frames 和 seed 0。

## 6. 服务器命令

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

PYTHONPATH=src pytest -q \
  tests/test_role_episodic.py \
  tests/test_hrem_debug_analyzer.py \
  tests/test_hrem_role_comparison.py \
  tests/test_structured_memory_readout.py

GPU=1 SEED=0 FORCE=1 \
  bash scripts/run_hrem_v2_role_ablation.sh
```

若 DINOv2 暂时不可用，可先生成和分析 trace：

```bash
GPU=1 SEED=0 FORCE=1 RUN_EVAL=0 \
  bash scripts/run_hrem_v2_role_ablation.sh
```

之后单独评估：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_hrem_v2.py \
  --run-root runs/hrem_v2_role_s0 \
  --methods native_reset dual_all_heads role_abs_060 role_abs_075 \
            role_relative_050 role_hybrid_050 \
  --baseline native_reset \
  --output runs/hrem_v2_role_s0/metrics_role_ablation.json

python scripts/compare_hrem_role_ablation.py \
  --run-root runs/hrem_v2_role_s0
```

## 7. 需要回传的最小信息

优先回传以下小文件，不需要先上传全部视频：

```text
runs/hrem_v2_role_s0/metrics_role_ablation.json
runs/hrem_v2_role_s0/role_ablation_comparison.json
runs/hrem_v2_role_s0/traces/*_diagnosis.json
runs/hrem_v2_role_s0/logs/*.log 中 [HREMv2] 行
```

提取日志：

```bash
for log in runs/hrem_v2_role_s0/logs/role_*.log; do
  echo "===== $log ====="
  grep -E '\[HREMv2\]' "$log" | tail -n 160
done
```

人工 review 还需每个 cell 的三段视频，至少记录：A2 identity、背景/布局、动态是否冻结、边界重影/闪烁。

## 8. 机制晋级标准

候选 role gate 必须同时满足：

1. trace route 仍为 `2 -> 0`，0 causal violation；
2. `role_calibration_valid_fraction >= 0.90`；
3. `0.10 <= role_active_head_fraction <= 0.90`；
4. `role_gate_std >= 0.02`，不能只是把所有 gate 从 0.93 平移到 0.50；
5. `role_active_head_jaccard >= 0.50`，避免 denoising call 间 head 身份随机变化；
6. fusion `delta_to_native_rms` 不应进入 `>0.25` 的过强区间；
7. paired return 不低于 all-head，且人工 motion 不差于 all-head。

`compare_hrem_role_ablation.py` 只给出 structural eligibility，不自动宣布 winner。最终选择必须结合视频 motion 和 boundary artifact。
该脚本还会比较各 cell 的结构化 `config`，除 head-routing/calibration 参数外出现任何配置差异都会标为 `config_mismatch`，防止消融矩阵混入 archive、retrieval 或 fusion 改动。

## 9. Position 实验暂缓原因

当前 archive 使用 `spatial_stride=4`，而现有 `local_grid` readout 要求 archive spatial token 数与 native `H*W` 完全相等。直接设置：

```text
STRUCTURED_MEMORY_POSITION_MODE=local_grid
```

会触发 `local_grid_incompatible_with_spatial_pooling` 并 abstain，不构成有效对比。可选方案是：

1. `spatial_stride=1` 做高显存 local-grid 上界；
2. 为 pooled archive 实现独立 pooled-grid coordinates；
3. 采用 MemRoPE 风格的 memory position mapping。

在 head gate P0 通过前不同时修改 position，以免无法归因。

## 10. 下一步决策树

```text
raw evidence spread < 0.01 in most layers
  -> 当前 K/V/query evidence 不支持 head specialization；analyzer 将全量 fail-closed
     记为机制警告而非运行错误，保留 native fallback 视频用于评估；之后重做 evidence，不调阈值

evidence spread 足够，但 active-head Jaccard < 0.50
  -> head identity 随 denoising call 不稳定；需要跨 call 聚合/冻结 gate

relative/hybrid 结构健康，但 return 和 motion 不优于 all-head
  -> head-role 假设未获支持；论文主线退回 selective episode recall

hybrid 保持 return 且改善 motion/identity
  -> 用 winner 配置运行 seeds 1,2，再进入 position/readout ablation
```

这轮实验的目的不是立即找到最高分，而是判断 head admission 是否能成为论文中的独立贡献。
