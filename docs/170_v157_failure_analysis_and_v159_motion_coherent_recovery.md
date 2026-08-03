# 170: v157 失败分析与 v159 Motion-Coherent Reservoir

日期：2026-08-04

状态：代码完成；本机完成语法、静态合同与分析测试；等待 GPU 生成。

## 1. 当前结论

最新仓库版本首先给出了两个必须同时保留的事实：

1. v155 表明 long-history cache 有用，但 QK top/bottom/random membership 没有表现出
   可复现的选择性，因此不能继续把当前方法描述成已验证的 head classifier。
2. v157 的 layer gate 在 VBench core-9 上通过全部五项预注册门槛，但 64-video 人评
   没有通过确认门控，所以 v158 budget sweep 仍应保持 `HOLD`，不能绕过门控运行。

v157 人评中，Interleaved10 相对 Recent8 的身份、背景和总体评分分别提高
`+0.5000 / +0.4375 / +0.40625`，说明它不是整体失效；真正的问题集中在相对
All-Reservoir 的运动质量 `-0.3125`，以及 `4/16` severe failures。

新增的可复现诊断见：

```text
scripts/analyze_v157_motion_failure_diagnosis.py
docs/results/v157_layer_gated_moviebench16/v157_motion_failure_diagnosis.json
docs/results/v157_layer_gated_moviebench16/v157_motion_failure_diagnosis.md
```

诊断结果为：

- 运动劣于 All-Reservoir 的 prompts：`[0, 1, 5, 6, 7, 15]`；
- 其余 10 条全部打平，没有 prompt 的运动评分优于 All-Reservoir；
- Interleaved10 severe prompts：`[0, 6, 8, 15]`；
- 多方法共同困难 prompts：`[0, 4, 6, 8, 10, 15]`；
- 仅 Interleaved10、而 All-Reservoir 不 severe 的 prompts：`[0, 8, 15]`。

因此，VBench Dynamic Degree 的高分只能说明运动量较大，不能说明运动连贯、方向正确
或不存在姿态反折。下一步必须直接修复人眼运动质量，不能继续只调 layer 数量，也不能
把 v157 的 VBench gate 当成人评通过。

## 2. v159 假设

`TemporalReservoir4` 提供分散的长期覆盖，但四张相互独立的历史帧没有显式保留运动
方向。对于行走、镜头深度变化、旋转和变形等 prompt，孤立的旧状态可能同时增加运动量
和姿态冲突。相邻两帧则提供一个最小的方向性运动单元。

v159 的假设是：

> 在相同读取预算下，将一半随机历史帧替换为一个语义连贯的相邻运动对，可以保留
> dispersed history 对身份/背景的帮助，同时改善运动连贯性。

这不是根据 VBench 后验调阈值，而是由 v157 人评暴露的 motion amount / motion quality
矛盾驱动的机制实验。

## 3. Dual-Timescale Cache

### 3.1 Primary cache

选中 layer 的每个 head 使用：

```text
Sink1 + TemporalReservoir2 + CoherentMotionPair1(2 frames) + Recent4
```

- `Sink1`：固定保留第一个物理帧，提供起始外观参考；
- `TemporalReservoir2`：帧离开 Recent4 后才有资格进入 deterministic Algorithm-R
  reservoir，保存两个精确 K/V 帧，不做 merge 或 feature averaging；
- `CoherentMotionPair1`：从同一生成 block 中选择相邻两帧；运动分数来自 layer-shared
  clean-value change，并用 pair similarity 和初始 identity reference 做语义门控；bank
  满时只有更强事件或超过 24 帧的 stale pair 才能替换；
- `Recent4`：保留局部生成连续性。

两个 middle strategy 按物理时间 `t` 去重。最大读取为
`1 + 2 + 2 + 4 = 9` 个 full-frame equivalents；发生重叠时只会更少，不会超预算。

未选中 layer 使用 `Sink1 + Recent8`，同样为 9 FFE。所有方法启用
`pyramidkv_composition_owns_dynamic=True`，不会并行读取 legacy dynamic history。

### 3.2 Layer routes

- Interleaved10：`[1, 4, 7, 10, 13, 16, 19, 22, 25, 28]`；
- Middle10：`[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`。

这里的 10/11 标签表示 layer policy gate，不表示已验证的 semantic head class。每个选中
layer 的 12 个 head 使用同一个被测 cache，选中总数固定为 120 heads。

## 4. 冻结实验矩阵

全部方法使用同一 16 prompts、seed 0、120 latent frames、477 decoded frames、16 FPS。
不运行 PF baseline。

| Method | Selected layers | Middle4 | 作用 | 来源 |
|---|---|---|---|---|
| `sf_native` | - | - | 原生基线 | 复用 v157 |
| `ours_interleaved10_reservoir2_motionpair1` | Interleaved10 | Reservoir2 + MotionPair1 | 预注册 primary | 新生成 |
| `ours_interleaved10_motionpair2` | Interleaved10 | MotionPair2 | motion-only 机制对照 | 新生成 |
| `ours_middle10_reservoir2_motionpair1` | Middle10 | Reservoir2 + MotionPair1 | layer placement 对照 | 新生成 |
| `ours_interleaved10_reservoir4_reference` | Interleaved10 | Reservoir4 | v157 直接对照 | 复用 v157 |
| `ours_middle10_reservoir4_reference` | Middle10 | Reservoir4 | v157 placement 对照 | 复用 v157 |
| `ours_all_reservoir4_reference` | all | Reservoir4 | 高运动端点 | 复用 v157 |
| `ours_all_recent8_reference` | none | none | recent-only 端点 | 复用 v157 |

共 128 个 published tasks，但只有 3 methods x 16 prompts = **48 个新视频**。四节点分片后
每节点 32 tasks，其中 12 个需要推理、20 个只建立带合同校验的链接；8 卡节点每卡最多
顺序运行 2 个新视频。

## 5. 生成命令

四个节点都先拉取相同 commit：

```bash
export REPO_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
cd "${REPO_ROOT}"
git pull --ff-only

export V159_REUSE_V157_ROOT="${REPO_ROOT}/runs/v157_layer_gated_moviebench16/full8"
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
```

节点 0 先做 preflight：

```bash
export NODE_RANK=0
bash scripts/run_v159_motion_coherent_reservoir_moviebench16.sh preflight
```

确认输出包含 `new=12 reused=20` 后，四个节点分别运行：

```bash
export NODE_RANK=<0|1|2|3>
bash scripts/run_v159_motion_coherent_reservoir_moviebench16.sh generate
```

全部完成后仅在节点 0 审计：

```bash
export NODE_RANK=0
bash scripts/run_v159_motion_coherent_reservoir_moviebench16.sh audit
```

必须看到 `published_manifest.json` 中 `ok=true` 后才能评测。若个别任务失败，直接用相同
命令重跑；完成 marker、config hash 和视频 fingerprint 一致时会 resume。

## 6. Debug 与正确性检查

每个新视频都有：

```text
logs/<task>.log
traces/<task>.policy.jsonl
traces/<task>.role_event.jsonl
diagnostics/<task>.policy.json
diagnostics/<task>.role_event.json
diagnostics/<task>.video.json
configs/<task>.json
```

runner 会自动拒绝以下情况：

- 当前 PF config 的 `num_frame_per_block` 不是 3，或其哈希、checkpoint 大小与冻结的 v157 对照不一致；
- 运行日志缺少 `HistoryPolarityPolicy`、`legacy_pf_labels=false` 或
  `exclusive_owner=true`；
- 实际 strategy 顺序不是
  `CoherentMotionStrategy, TemporalReservoirStrategy`；
- sink/recent/middle 的物理帧或 token 重叠；
- middle union 超过 4 帧，或总 read 超过 9 FFE；
- reservoir anchor/pending 超容量、重复或乱序；
- motion pair 不是相邻帧、pair bank 超容量或 spacing invariant 失败；
- role-event context 的 head ids 与 layer map 不一致；
- 视频不是 477 帧、16 FPS、832x480，或出现黑帧/解码失败。

人工 review 前重点查看 `role_event.jsonl` 中：

- `motion_scores` 与 `adjacent_semantic_similarity`；
- `candidate_pair`、`motion_threshold`、`semantic`、`utility`；
- `reason` 是 `fill_motion_pair`、`stronger_motion_event`、
  `stale_motion_refresh` 还是 gate rejection；
- `pairs_after` 是否长期不更新或在失败时间点发生替换；
- reservoir 的 `anchor_frame_ids`、`sample_span` 和 `max_sample_gap`；
- `union_frame_ids` 是否因 pair/reservoir 重叠而长期少于 4。

## 7. VBench-Long

VBench 只做 safety diagnostic，不再单独决定 promotion。

节点 0：

```bash
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v159_vbench_long.sh prepare
```

四个节点分别预切分并运行 core-9：

```bash
export V159_VBENCH_DIMENSIONS=subject_consistency,background_consistency,temporal_flickering,motion_smoothness,overall_consistency,dynamic_degree,aesthetic_quality,imaging_quality,temporal_style
export NODE_RANK=<0|1|2|3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
bash scripts/run_v159_vbench_long.sh split
bash scripts/run_v159_vbench_long.sh preflight
bash scripts/run_v159_vbench_long.sh eval
```

完成后节点 0 收集：

```bash
export NODE_RANK=0
bash scripts/run_v159_vbench_long.sh collect-core
```

`metric_safety_gate` 要求 primary 相对 Interleaved Reservoir4 的 Dynamic、Temporal、
History、Visual 分别不低于 `-0.020 / -0.004 / -0.003 / -0.006`。即使全部通过，
`metric_promotion_gate` 仍固定为 false，必须等待人评运动质量。

## 8. 盲审

盲审只包含 4 个最直接方法，共 64 videos：primary、Interleaved MotionPair2、
Middle10 hybrid 和 Interleaved Reservoir4。

节点 0 生成匿名包：

```bash
bash scripts/run_v159_blind_review.sh prepare
```

填写：

```text
runs/v159_motion_coherent_reservoir_moviebench16/full8/
  blind_review64/reviewer/v159_review_sheet.csv
```

然后分析：

```bash
bash scripts/run_v159_blind_review.sh analyze
```

探索性 recovery gate 要求：

- primary severe failures `<=2/16`，且不多于 Reservoir4；
- primary 相对 Reservoir4 的平均 motion quality 至少 `+0.125`；
- 相对三个 controls 的 overall noninferior prompts 均至少 `10/16`；
- 相对三个 controls 的 identity/background 平均差均至少 `-0.125`。

该 gate 只用于选择下一轮方法，不是论文 promotion gate，因为 v159 是看过 v157 人评后
设计的。通过后还需要在 held-out prompts 上确认。

## 9. 结果分支

1. **Hybrid 通过**：冻结 cache 与 layer route，在未参与 v154/v157 选择的 32 或 128
   prompts 上确认，并补充 pair/reservoir ablation。
2. **MotionPair2 优于 Hybrid**：说明方向性历史比随机覆盖更重要；先用 held-out 集验证
   motion-only，再决定是否保留 reservoir 作为身份分支。
3. **Middle10 Hybrid 优于 Interleaved10 Hybrid**：采用 middle placement；这说明
   v157 背景优势和运动机制需要不同深度分布，不能继续预设 interleaved 最优。
4. **三种新方法均不改善人评运动**：拒绝 dual-timescale 假设，不扩大到 128 prompts；
   回到生成状态/attention profiling，分析姿态反折发生前后的 query 与 cached frame。
5. **出现 polygon noise 或 trace contract failure**：视为实现失败，不作为方法负结果；
   先提交对应 log、policy trace、role-event trace 和 diagnostics package。

## 10. 本地验证

本机没有 PyTorch/CUDA，因此未运行模型，也未执行需要真实 v157 `runs/` 目录的 preflight。
已完成：

- Python `py_compile`：全部新增和修改 Python 文件通过；
- Bash `bash -n`：三个新增 shell entrypoint 通过；
- v157 layer map frozen check：4 maps、每个 120 selected heads，通过；
- v159 单元测试：7 passed；
- 相关历史回归（排除已知环境项后）：34 passed；另有 5 个本机环境项，分别来自 Windows symlink privilege
  和未同步的远端 v157 run artifacts，与本次策略代码无关；
- PyTorch-dependent composition test 在本机 skip，服务器 preflight/generation trace audit
  是最终运行时验证。
