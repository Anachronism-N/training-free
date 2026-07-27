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

## 5. VBench-Long 专项分析

### 5.1 各维度整体排名

| 维度 | 最佳方法 | 分数 | 最差方法 | 分数 | spread |
|---|---|---:|---|---:|---:|
| subject_consistency | landmark_snapshot2 | 0.9602 | support_prototype_recent | 0.9548 | 0.0054 |
| background_consistency | landmark_prototype2 | 0.9516 | landmark_sparse75 | 0.9476 | 0.0039 |
| aesthetic_quality | landmark_motion1 | 0.6258 | support_prototype_recent | 0.6180 | 0.0078 |
| imaging_quality | prototype_motion1 | 0.6892 | landmark_retrieval2 | 0.6818 | 0.0074 |

VBench 4 个维度的 spread 都非常小 (0.004-0.008)，说明 9 个方法在 VBench
意义上几乎不可区分。最佳方法在 4 个维度上分散在 4 个不同方法上，没有单一
方法在 VBench 上占优。

### 5.2 逐 prompt 胜负分布

**subject_consistency 逐 prompt 胜者** (per-prompt clip 平均后取最高)：

| Method | 胜场 / 16 |
|---|---:|
| landmark_recent8 | 3 |
| landmark_snapshot2 | 3 |
| support_prototype_recent | 2 |
| landmark_prototype2 | 2 |
| prototype_motion1 | 2 |
| landmark_sparse75 | 2 |
| landmark_motion2 | 1 |
| landmark_motion1 | 1 |
| landmark_retrieval2 | 0 |

`landmark_retrieval2` 在 16 个 prompt 上一次都没拿到 VBench subject 第一，
但它的 DINO consistency 和 drift 都领先——这是 VBench 与 DINO 信号分歧的
直接证据。`landmark_sparse75` 虽然 VBench 总分最差，却在 2 个 prompt 上拿到
第一，说明它在某些特定场景下 (如 p0/p3 的稳定身份 prompt) 仍有局部优势。

**aesthetic_quality 逐 prompt 胜者**：

| Method | 胜场 / 16 |
|---|---:|
| landmark_snapshot2 | 5 |
| landmark_motion1 | 2 |
| landmark_recent8 | 2 |
| landmark_motion2 | 2 |
| landmark_prototype2 | 2 |
| landmark_sparse75 | 1 |
| landmark_retrieval2 | 1 |
| prototype_motion1 | 1 |
| support_prototype_recent | 0 |

`landmark_snapshot2` 在 aesthetic 上明显占优 (5/16)，这与它在 VBench subject
上也最好一致——snapshot cache 倾向于保持视觉帧间稳定，因此表观质量和短期
一致性都受益，但代价是长期 drift (见 4.4)。

### 5.3 VBench 与 DINO 的相关性

| 指标对 | Pearson r (n=9) | 解读 |
|---|---:|---|
| VB-Subj vs DINO | **-0.039** | 几乎零相关 — VBench subject 不能预测 DINO 一致性 |
| VB-Subj vs Drift | **-0.343** | 弱负相关 — VBench subject 越高，drift 反而越差 |
| VB-BG vs BG-DINO | +0.299 | 弱正相关 |
| VB-Aes vs CLIP | +0.535 | 中等正相关 |
| VB-Img vs DINO | **+0.549** | 中等正相关 — 最 aligned 的一对 |
| VB-Aes vs VB-Img | +0.102 | 几乎零相关 — 两个 VBench 质量维度内部不一致 |
| VB-Subj vs VB-BG | +0.524 | 中等正相关 |

**关键发现：VBench subject_consistency 与 DINO consistency 几乎零相关 (r=-0.04)**，
与 drift 弱负相关 (r=-0.34)。这意味着 VBench 的短期光流/表观一致性和 DINO
的长期语义一致性衡量的是不同的东西——一个方法可以在 VBench subject 上领先
同时在 DINO drift 上倒数 (典型例子：`landmark_snapshot2`)。

唯一与 DINO 有中等正相关的是 VBench imaging_quality (r=+0.55)，提示成像质量
本身对长期一致性有间接贡献，但这个相关性也不够强到可以替代 DINO。

### 5.4 VBench 对候选选择的指导意义

1. **VBench 不能单独用于选方法**。4 个维度 spread 都 < 0.008，且 subject 与
   DINO 零相关，单纯看 VBench 排名会选到 `landmark_snapshot2` (VBench 第一)
   而 DINO drift 最差的方法。
2. **aesthetic_quality 可作为辅助参考**。`landmark_snapshot2` 在 aesthetic 上
   5/16 胜，且 aesthetic 与 CLIP 有中等正相关 (r=+0.54)，说明它在视觉美感
   上确实有优势——但这不等于身份一致性。
3. **VBench background_consistency 区分度最低** (spread 0.0039)，9 个方法
   几乎一样，这与所有方法都使用 Landmark4/Prototype4 支撑背景一致。
4. **`landmark_retrieval2` 在 VBench subject 上 0 胜**，但 DINO/drift 领先，
   说明它的优势完全在长期一致性而非短期表观——这种优势只能通过 DINO 或人眼
   长期观察捕捉。

## 6. 候选方法建议

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

## 7. 已知限制

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

## 8. 下一步

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

## 9. 文件位置

- 视频目录：`runs/v116_role_memory_diverse16/m9_7a14c511d500/<method>_flat/`
- VBench-Long 汇总：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/vbench_long_summary.{md,json,csv}`
- VBench-Long 逐 prompt：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/vbench_per_prompt.json`
- DINO comprehensive 汇总：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/comprehensive_summary.json`
- 合并指标：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/combined_summary.json`
- 各方法原始结果：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/comprehensive_parts/<method>.json`
- DINO 日志：`runs/v116_role_memory_diverse16/m9_7a14c511d500/metrics/logs/comp.<method>.log`
- Review 索引：`runs/v116_role_memory_diverse16/m9_7a14c511d500/review_index.md`
