# v116 MovieBench-16 评测结果

日期：2026-07-27

状态：生成 + VBench-Long + DINO comprehensive 全部完成；待人工 review。

## 1. 实验规模

- 9 方法 × 16 prompts × 1 sample = 144 个视频
- 每视频 477 帧 / 16 FPS / 832×480
- VBench-Long 4 维度（subject_consistency, background_consistency, aesthetic_quality, imaging_quality；dynamic_degree 因 RAFT 不可用已排除）
- DINO comprehensive 7 维度（DINO consistency, drift slope, temporal flickering, CLIP-text alignment, background consistency, loop score, composite；M3 motion smoothness 与 M4 ArcFace ID 已 skip）
- 全部评测在 3 个远程节点（node221 / node121 / node21）并行完成，每节点 3 方法 × GPU 0/1/2，约 11 分钟跑完

## 2. 方法清单

| Method key | Supportive | Suppressive | 主要观察 |
|---|---|---|---|
| `landmark_recent8` | Landmark4 + Recent4 | Recent8 | 无 middle 的局部参考 |
| `landmark_motion2` | Landmark4 + Recent4 | Motion-pair2 + Recent4 | 较强运动事件记忆 |
| `landmark_motion1` | Landmark4 + Recent4 | Motion-pair1 + Recent6 | 较弱运动事件与更多近期上下文 |
| `landmark_prototype2` | Landmark4 + Recent4 | Prototype2 + Recent6 | 中期语义段压缩 |
| `landmark_snapshot2` | Landmark4 + Recent4 | Snapshot2 + Recent6 | relevance/uniqueness 快照 |
| `landmark_retrieval2` | Landmark4 + Recent4 | Retrieval2 + Recent6 | 当前状态相关的非近期读取 |
| `landmark_sparse75` | Landmark4 + Recent4 | 4×75% sparse + Recent5 | 小比例 heads 的 token 压缩 |
| `support_prototype_recent` | Prototype4 + Recent4 | Recent8 | Landmark vs Prototype 直接比较 |
| `prototype_motion1` | Prototype4 + Recent4 | Motion-pair1 + Recent6 | Prototype + 轻量运动 cache |

前 7 项只改变 Suppressive cache，是本轮的核心直接比较。后 2 项用于判断 Prototype4 是否值得替代 Landmark4。

## 3. 综合排行榜

按 DINO composite 降序：

| Method | Comp | VB-Sub | VB-BG | VB-Aes | VB-Img | DINO | Drift | Flick | CLIP | BG-D | Loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| support_prototype_recent | 0.6310 | 0.9548 | 0.9494 | 0.6180 | 0.6854 | 0.8420 | -0.00150 | 0.3065 | 0.2880 | 0.9016 | 0.0866 |
| prototype_motion1 | 0.6286 | 0.9555 | 0.9494 | 0.6210 | 0.6892 | 0.8435 | -0.00168 | 0.2996 | 0.2887 | 0.9050 | 0.0933 |
| landmark_retrieval2 | 0.6286 | 0.9566 | 0.9490 | 0.6212 | 0.6818 | 0.8415 | -0.00151 | 0.2967 | 0.2932 | 0.9094 | 0.1256 |
| landmark_motion1 | 0.6256 | 0.9590 | 0.9514 | 0.6258 | 0.6849 | 0.8431 | -0.00171 | 0.2950 | 0.2914 | 0.9090 | 0.1210 |
| landmark_motion2 | 0.6237 | 0.9559 | 0.9488 | 0.6254 | 0.6865 | 0.8393 | -0.00173 | 0.3037 | 0.2932 | 0.9098 | 0.1201 |
| landmark_prototype2 | 0.6205 | 0.9595 | 0.9516 | 0.6249 | 0.6857 | 0.8404 | -0.00186 | 0.2949 | 0.2913 | 0.9120 | 0.1476 |
| landmark_recent8 | 0.6204 | 0.9563 | 0.9484 | 0.6220 | 0.6827 | 0.8338 | -0.00232 | 0.2961 | 0.2890 | 0.9083 | 0.0841 |
| landmark_snapshot2 | 0.6159 | 0.9602 | 0.9491 | 0.6215 | 0.6851 | 0.8374 | -0.00237 | 0.2911 | 0.2923 | 0.9114 | 0.1236 |
| landmark_sparse75 | 0.6134 | 0.9566 | 0.9476 | 0.6216 | 0.6829 | 0.8323 | -0.00252 | 0.3025 | 0.2919 | 0.9065 | 0.1218 |

## 4. 关键观察

### 4.1 总分区间很窄

Composite 最高 0.6310，最低 0.6134，跨度仅 0.0175。所有方法在 16 prompt 上都
"可用"，没有出现崩溃或严重退化。这与 v115 单 prompt 筛选结论一致：在 Landmark4
支撑下，Suppressive cache 的选择不会带来崩塌式差异。

### 4.2 Prototype-Supportive 略优于 Landmark-Supportive

| Group | 平均 Composite | 平均 DINO | 平均 Drift |
|---|---:|---:|---:|
| Prototype-Supportive (n=2) | 0.6298 | 0.8428 | -0.00159 |
| Landmark-Supportive (n=7) | 0.6212 | 0.8383 | -0.00200 |

Prototype4 在 DINO consistency 和 drift 上都略优于 Landmark4，且在 VBench
imaging_quality 上 `prototype_motion1` 拿到第一 (0.6892)。这与 v115 单 prompt
review 中 "Prototype4 也可用" 的判断一致。

### 4.3 Suppressive cache 在 Landmark4 下的内部排序

按 composite 排：

1. `landmark_retrieval2` 0.6286 — 最好的 drift (-0.00151) 和 CLIP (0.2932)
2. `landmark_motion1` 0.6256 — 最好的 DINO (0.8431)
3. `landmark_motion2` 0.6237 — VBench aesthetic 第一 (0.6254)
4. `landmark_prototype2` 0.6205 — VBench subject 第一档 (0.9595)，但 loop 偏高 (0.1476)
5. `landmark_recent8` 0.6204 — 最好的 (最低) loop (0.0841)，但 drift 最差 (-0.00232)
6. `landmark_snapshot2` 0.6159 — VBench subject 第一 (0.9602) 但 drift 倒数第一 (-0.00237)
7. `landmark_sparse75` 0.6134 — 全部维度最差或倒数第二

`landmark_retrieval2` 和 `landmark_motion1` 是 Landmark4 下表现最稳的两个
Suppressive 选项：前者 drift 最小，后者 DINO 最高。两者 loop 都在 0.12 左右，
处于中等水平。

### 4.4 loop score 与其他指标的矛盾

`landmark_recent8` loop 最低 (0.0841)，但 DINO 和 drift 都最差；这说明
"不重复" 不等于 "一致"——主体可能在持续漂移而不是循环。反之
`landmark_prototype2` loop 最高 (0.1476) 但 drift 相对较好 (-0.00186)，
提示它更倾向于让主体回到原型位置，因此被 loop 检测器识别为重复。

### 4.5 VBench vs DINO 的分歧

- VBench subject_consistency 最高：`landmark_snapshot2` (0.9602)
- DINO consistency 最高：`prototype_motion1` (0.8435)

VBench 基于 RAFT 光流和 DINO 后台特征，对短期表观一致更敏感；DINO 则对长期
语义一致更敏感。`landmark_snapshot2` 在 VBench 上领先但在 DINO 上倒数第二，
说明它的短期表观一致是以长期语义漂移为代价换来的。

## 5. 候选方法建议

基于本轮指标证据，建议如下候选顺序（仍需人工 review 确认）：

| 优先级 | 候选 | 理由 |
|---|---|---|
| P1 | `landmark_retrieval2` | Landmark4 下 composite 第一，drift 最小，CLIP 第一 |
| P2 | `landmark_motion1` | Landmark4 下 DINO 第一，drift 第二好 |
| P3 | `support_prototype_recent` | 全场 composite 第一，drift 最好，但 VBench subject 略低 |
| P4 | `prototype_motion1` | DINO 第一，imaging 第一，但 Supportive 切换风险 |
| 观望 | `landmark_recent8` | loop 最低但 drift 最差，不推荐作为主方法 |
| 观望 | `landmark_snapshot2` | VBench subject 最高但 drift 最差，矛盾信号 |
| 排除 | `landmark_sparse75` | 全维度最差 |

P1/P2 直接对决 Suppressive cache 的 retrieval vs motion 路线；P3/P4 验证
Prototype4 是否值得替换 Landmark4。本轮指标不足以单独定夺，需要人工 review
在身份保持、运动幅度、背景稳定三方面的判断。

## 6. 已知限制

1. **16 prompt 不足以支撑论文主表**。仅用于候选收缩。最终主方法确定后仍需在
   MovieGenVideoBench-128 上与 SF / PF / Echo-Forcing 做同 seed 对比。
2. **dynamic_degree 缺失**。RAFT 模型在沙箱内不可下载，VBench-Long 只跑了 4
   维度。如需 dynamic_degree，需要在能访问外网的节点上预下载 RAFT 权重。
3. **M3 motion smoothness 和 M4 ArcFace ID 已 skip**。`--skip_m3` 用于节省
   时间；M4 在非人脸 prompt 上信号弱，本轮未启用。
4. **Composite 分数区间很窄**。0.0175 的跨度可能落在评测噪声内，因此排名顺序
   不能直接当作强弱证据，只能作为辅助。
5. **指标与人眼可能不一致**。4.4 / 4.5 已指出 loop、VBench subject 和 DINO
   之间存在矛盾，必须以人工 review 为最终判据。

## 7. 下一步

1. **人工 review**：在 9 个方法 × 16 prompt 上做盲评或公开评，重点比较
   P1-P4 候选的身份保持、运动幅度衰减、背景稳定性和场景演化。
2. **主方法确定后**：在 MovieGenVideoBench-128 上做同 seed 对比，包括与
   SF / PF / Echo-Forcing 的 baseline 对比和 all-head control 消融。
3. **容量消融 (P1)**：在确定主方法后，按 docs/117 §5 的 P1 计划做容量
   ablation（token 预算 2×/4×/8×）。
4. **生命周期更新 (P2)**：在主方法上启用 lifecycle admission/replacement，
   验证是否带来额外增益。
5. **sink 实验 (P3)**：测试 sink size 0/4/8/16 对稳定性的影响。
6. **长视频 ABA (P4)**：在 60s / 90s / 120s 视频上验证 cache 是否持续有效。

## 8. 文件位置

- 视频目录：`runs/v116_role_memory_diverse16/m9_7a14c511d500/<method>_flat/`
- VBench-Long 汇总：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/vbench_long_summary.{md,json,csv}`
- DINO comprehensive 汇总：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/comprehensive_summary.json`
- 合并指标：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/combined_summary.json`
- 各方法原始结果：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/comprehensive_parts/<method>.json`
- DINO 日志：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/logs/comp.<method>.log`
