# HREM-v2 Post-Sweep Optimization and Server Protocol

> 日期：2026-07-22
>
> 状态：代码与静态检查阶段；新增实验尚未在 GPU 服务器运行。
>
> 输入证据：`docs/65_swift_collision_audit.md`、`docs/66_gate_sweep_results_and_review.md` 和当前 HREM-v2 实现。

## 1. 本轮结论

当前不能把 absolute threshold sweep 中的 `t=0.75` 设为最终配置：

- 只有 3 prompts、seed 0；
- mean return margin 相对 native_reset 仅 `+0.0140`；
- prompt 间方向不一致；
- 人工 review 未观察到稳定优势；
- 部分 cell 来自不同 run root，旧脚本没有保存完整 config trace；
- absolute threshold 只能改变 gate 强弱，尚未证明 active head 的相对身份稳定。

因此本轮不继续围绕 `0.75` 做更细阈值搜索。下一步先回答两个可证伪问题：

1. relative/hybrid calibration 是否能形成稳定、非全 head 的 online eligibility；
2. B→A2 首块突然启用 memory 是否贡献了 HREM-specific 边界伪影。

## 2. 对当前实验解释的校正

### 2.1 A-B-A 不是三次独立推理

当前实现对每条 prompt 只调用一次 `pipeline.inference`。在同一 AR trajectory 中：

1. `||` 被拆成 A1、B、A2 三个 conditioning；
2. block 进入新 segment 时切换 cross-attention conditioning；
3. `SCENE_TRANSITION_RESET=1` 清空 native working K/V；
4. `EpisodicArchive` 不清空，继续保存完整历史 sidecar。

它是带受控 reset 的 episodic-return protocol，不是三个视频生成后拼接；但也不能代表完全无边界干预的自然长视频。

### 2.2 单 prompt 长视频的作用有限

当前主配置：

```text
STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2
```

单 prompt 30s/60s 运行始终处于 episode 0，memory readout 会 fail-closed。因此它可以测：

- sidecar capture 是否保持 output 等价；
- archive 显存、延迟和稳定性开销；
- 原生 Self-Forcing 的连续长视频质量。

它不能证明 episodic recall 有效。方法有效性仍需 A-B-A、A-B-C-A 或明确的 multi-prompt return 任务。

### 2.3 当前“原因”都只是待验证假设

Identity 变化不能直接归因给 `noisy_only`、`position_mode=none` 或 archive budget；边界伪影也不能直接归因给 reset 或 memory gate。必须用单变量对照和 trace 才能区分。

## 3. 本轮代码修改

### 3.1 修复边界审计 helper 覆盖

`src/lifecycle_kv/attention_fusion.py` 之前重复定义：

```text
summarize_tensor_state
summarize_episode_boundary_state
```

Python 实际只保留第二个版本。第二个版本依赖旧 cache flags，面对当前 `EpisodicArchive` 可能跳过全部 layer。现在只保留一个实现，并直接读取：

- `_sm_active`；
- `structured_memory_{k,v,intervals,episode_ids}`；
- `archive.config.episode_gate_*`。

新增测试会确认当前 archive 形状能产生非空 layer summary。

当前 pipeline 已把该 summary 接入每个 `boundary` trace，记录 active layer、interval/episode sidecar、K/V shape、checksum、mean 和 RMS。旧 trace 缺少快照时 analyzer 给 warning；新快照存在但 layer 为空时给 error。

### 3.2 Episode-local fusion ramp

新增环境变量：

```text
STRUCTURED_MEMORY_EPISODE_WARMUP_BLOCKS=0
```

默认 `0`，旧路径不变。设置为 `2` 时，B→A2 后的 memory global gate 依次乘：

```text
episode block 0: 1/3
episode block 1: 2/3
episode block 2+: 1
```

它只平滑 memory branch，不修改 native working-cache reset，也不做视频后处理。若 episode 起始帧 metadata 缺失，ramp 启用路径会 fail-closed，而不是意外全强度注入。

该机制目前只是边界归因消融，不是论文贡献。只有人工 review 和 transition metric 都支持时才保留。

### 3.3 新增 trace 字段

每条 accepted readout 新增：

```text
base_gate
global_warmup_scale
episode_warmup_blocks
episode_warmup_scale
episode_block_index
effective_gate
```

Analyzer 新增：

```text
episode_warmup_scale_mean/min/max
episode_first_block_effective_gate_mean
episode_warmup_trace_missing
episode_warmup_not_observed
```

服务器 review 必须确认 ramp cell 的 `episode_warmup_scale_min < 1`。否则不能用该视频判断 ramp 是否有效。

### 3.4 Sweep 脚本收敛

`scripts/run_headgate_sweep.sh` 现在仅作为 supplemental absolute sweep：

- 不再无条件删除整个输出目录；
- 支持 `REPO_ROOT/GPU/SEED/FRAMES/THRESHOLDS/FORCE`；
- 每个 cell 保存 trace、diagnosis 和独立 log；
- 任一 cell 或 analyzer 失败会反映到最终退出码；
- 已有视频但缺少 trace 时自动重跑。

两个正式 runner 都会在 config trace 中记录 `run_commit`、`run_cell`、`run_seed`、`run_frames` 和 `prompt_sha256`，用于阻止跨 commit/跨 prompt 误比较。

正式 P0 使用 `scripts/run_hrem_v2_role_ablation.sh`。

## 4. 下一轮受控矩阵

| Cell | Episode gate | Head gate | Episode ramp | 目的 |
|---|---|---|---|---|
| `native_reset` | off | off | off | 公平 reset baseline |
| `dual_all_heads` | dual | off | off | episode-only 上界/对照 |
| `role_abs_060` | dual | absolute 0.60 | off | absolute 温和筛选复现 |
| `role_abs_075` | dual | absolute 0.75 | off | 当前弱候选受控复现 |
| `role_relative_050` | dual | relative top 50% | off | head 相对排序是否有效 |
| `role_hybrid_050` | dual | hybrid top 50% | off | 绝对证据与相对排序联合 |
| `role_hybrid_050_ramp2` | dual | 与上一 cell 相同 | 2 blocks | 仅隔离 memory activation ramp |

所有 cell 必须使用同一 commit、prompt、seed、frame count、archive、retrieval、layers 和 global gate。

## 5. 服务器运行指令

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

PYTHONPATH=src pytest -q \
  tests/test_role_episodic.py \
  tests/test_episodic_archive.py \
  tests/test_structured_memory_readout.py \
  tests/test_hrem_debug_analyzer.py \
  tests/test_hrem_role_comparison.py

GPU=1 SEED=0 FORCE=0 RUN_EVAL=1 \
  bash scripts/run_hrem_v2_role_ablation.sh
```

若 DINO 环境暂时不可用：

```bash
GPU=1 SEED=0 FORCE=0 RUN_EVAL=0 \
  bash scripts/run_hrem_v2_role_ablation.sh
```

之后单独运行：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_hrem_v2.py \
  --run-root runs/hrem_v2_role_s0 \
  --methods native_reset dual_all_heads role_abs_060 role_abs_075 \
            role_relative_050 role_hybrid_050 role_hybrid_050_ramp2 \
  --baseline native_reset \
  --output runs/hrem_v2_role_s0/metrics_role_ablation.json

python scripts/compare_hrem_role_ablation.py \
  --run-root runs/hrem_v2_role_s0 \
  --json-output runs/hrem_v2_role_s0/role_ablation_comparison.json
```

## 6. 必须回传的信息

最小回传集合：

```text
runs/hrem_v2_role_s0/metrics_role_ablation.json
runs/hrem_v2_role_s0/role_ablation_comparison.json
runs/hrem_v2_role_s0/traces/*_diagnosis.json
runs/hrem_v2_role_s0/logs/role_relative_050.log
runs/hrem_v2_role_s0/logs/role_hybrid_050.log
runs/hrem_v2_role_s0/logs/role_hybrid_050_ramp2.log
```

日志提取：

```bash
grep -E '\[HREMv2\]' \
  runs/hrem_v2_role_s0/logs/role_hybrid_050_ramp2.log | tail -n 200
```

人工 review 对 `native_reset`、`role_hybrid_050`、`role_hybrid_050_ramp2` 逐 prompt 记录：

| 字段 | 记录要求 |
|---|---|
| A→B boundary | 重影、纹理粘连、闪烁持续帧数 |
| B→A2 boundary | 同上；特别比较 no-ramp 与 ramp2 |
| identity | 脸、服装、标志物、独特物体几何 |
| layout | A1 与 A2 背景和对象相对位置 |
| motion | 主体、相机、背景运动是否冻结或循环 |
| preference | blind pair preference，不能先显示方法名 |

## 7. Go/No-Go 规则

### Head admission 保留条件

只有同时满足以下条件，head gate 才继续作为论文贡献：

1. `role_active_head_fraction` 位于 `[0.10, 0.90]`；
2. `role_gate_std >= 0.02`；
3. `role_calibration_valid_fraction >= 0.90`；
4. `role_active_head_jaccard >= 0.50`；
5. paired return 不低于 `dual_all_heads`；
6. 人工 motion/identity 至少一项稳定优于 all-head，另一项不明显退化。

若 relative/hybrid 仍无稳定收益，停止调 head threshold，论文故事退回 selective episode recall 或 diagnostic analysis。

### Episode ramp 保留条件

Ramp 只在以下条件同时成立时保留为工程组件：

1. trace 证明 scale 实际从 `<1` 上升到 `1`；
2. B→A2 边界伪影优于 no-ramp；
3. A2 return/identity 没有明显下降；
4. A→B artifact 不被错误归因给 ramp，因为该处 memory 尚未激活。

若 ramp 只让切换变慢或降低 return，则删除，不扩大搜索。

## 8. 后续顺序

1. 先跑 seed 0 完整 role/ramp 矩阵；
2. 只将结构健康且视频有优势的候选扩展到 seeds 1/2；
3. 候选成立后再做 VBench-Long 和 identity-aware metric；
4. 再做 A-B-C-A，多候选 episode 才能真正检验 selector；
5. 最后决定是否实现 pooled-grid position。不要同时改变 position、readout mode、budget 和 gate。

当前最重要的论文决策仍是：head eligibility 是否有可重复价值。如果答案是否定的，应主动删去该贡献，而不是继续通过阈值搜索包装弱结果。
