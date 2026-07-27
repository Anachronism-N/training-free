# v119 Review Results and sink3 Polygon Noise Bug

日期：2026-07-27

状态：v119 5-cell 生成完成；2 个 sink3 cell 出现多边形噪声，根因已定位。

## 1. 人工 Review 结果

| Cell | 视觉评价 | 问题 |
|---|---|---|
| `legacy_v98_landmark4_retrieval1` | 后期有一些放大，运动略微减小，但 ID 和背景不错 | 轻微后期放大 |
| `legacy_v98_landmark4_retrieval1_age24` | 没有什么大问题 | 无 |
| `legacy_v98_landmark4_retrieval1_motion1_age24` | 还行 | 无 |
| `legacy_v98_landmark4_motion1_sink3_extra` | **多边形噪声** | sink3 + 11 FFE |
| `legacy_v98_landmark2_motion1_sink3_budget9` | **多边形噪声** | sink3 + 9 FFE |

结论：retrieval1_age24 和 retrieval1_motion1_age24 可以考虑保留；两个 sink3 variant
均有 polygon noise，不再作为 v120 候选。retrieval1 虽有轻微后期放大但仍可保留。

## 2. sink3 多边形噪声根因分析

### 2.1 现象

两个 sink3 cell (`sink3_extra` 和 `sink3_budget9`) 均出现多边形噪声，而使用相同
head map (legacy_v98 304/56) 和相同 support/suppress policy (landmark + motion_pair1)
的 v116 `landmark_motion1` 控制组在同一个 prompt 上是干净的。唯一差异是
`history_budget_profile` 从 `"default"` 变为 `"sink3_extra"` 或 `"sink3_budget9"`。

### 2.2 根因

在 `third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py` 第 406-414 行：

```python
if budget == "sink3_extra":
    support_sink = 3
    suppress_sink = 3          # <-- BUG
elif budget == "sink3_budget9":
    support_sink = 3
    suppress_sink = 3          # <-- BUG
    support_landmark_capacity = 2
    support_recent = 4
    suppress_recent = 4
```

sink3 profile 同时把 **Supportive (label 10)** 和 **Suppressive (label 11)** 的
sink 从 1 增加到 3。这导致在生成启动阶段 (sync_t=0)，前 3 帧 [0, 1, 2] 全部被
sink 占用，recent 窗口为空：

**Policy trace 对比 (sync_t=0, label 11 Suppressive)：**

| Cell | sink_frame_ids | recent_frame_count | recent_frame_ids | union_count |
|---|---|---:|---|---:|
| `retrieval1_age24` (clean, sink=1) | [0] | 2 | [1, 2] | 0 |
| `motion1_sink3_extra` (polygon, sink=3) | [0, 1, 2] | **0** | **[]** | 0 |

**Policy trace 对比 (sync_t=0, label 10 Supportive)：**

| Cell | sink_frame_ids | recent_frame_count | recent_frame_ids | union_count |
|---|---|---:|---|---:|
| `retrieval1_age24` (clean, sink=1) | [0] | 2 | [1, 2] | 0 |
| `motion1_sink3_extra` (polygon, sink=3) | [0, 1, 2] | **0** | **[]** | 0 |

在 sync_t=0 时，sink=3 消耗了全部可用帧 [0, 1, 2]，导致 recent 窗口为空。
两类 heads 都只读到 3 个不可变 sink 帧，没有任何近期上下文。

**sync_t=15 后恢复正常：**

| Cell | sink | recent_count | union_count |
|---|---|---:|---:|
| `retrieval1_age24` (clean) | [0] | 7 | 0-1 |
| `motion1_sink3_extra` (polygon) | [0, 1, 2] | 6 | 1-2 |

稳态下 sink3 的 recent 窗口 (6 帧) 与 clean 的 (7 帧) 接近，union 也相近。
**问题仅出现在启动阶段**，但启动阶段生成的第一帧块带有 polygon noise，这个噪声
会通过 AR 生成传播到后续帧。

### 2.3 机制解释

Suppressive heads (label 11, 56 个) 的设计意图是抑制历史、聚焦近期上下文。
在 sync_t=0 时给它们 3 个不可变 sink 帧而没有任何 recent，等价于强制它们只看
最前 3 帧的静态特征。这与它们的设计意图完全相反，导致 attention 分布退化，
产生多边形噪声——与 v107 中发现的 "PF Wave heads 移到 stride 导致 polygon noise"
机制类似：当 heads 被迫读取不合适的上下文时，attention 退化产生几何 artifact。

### 2.4 修复建议

**方案 A (最小改动)：** sink3 profile 只增加 Supportive 的 sink，不改变
Suppressive 的 sink：

```python
if budget == "sink3_extra":
    support_sink = 3
    # suppress_sink 保持默认值 1
elif budget == "sink3_budget9":
    support_sink = 3
    # suppress_sink 保持默认值 1
    support_landmark_capacity = 2
    support_recent = 4
    suppress_recent = 4
```

**方案 B (更鲁棒)：** 添加动态 sink cap，确保启动阶段 `sink + recent <= available_frames`：

```python
# 在 HeadComposition 中，动态限制 sink 大小
effective_sink = min(configured_sink, max(1, available_frames - min_recent))
```

**方案 C (本次实验)：** 跳过 sink3 候选，不在 v120 中使用。retrieval1_age24
和 retrieval1_motion1_age24 已经是更优的候选，且无需修改代码。

本轮推荐方案 C，因为 v119 的 retrieval variants 已通过 review，sink3 不是
v120 的必要候选。方案 A/B 留作后续消融实验。

## 3. v119 候选评价

| Cell | FFE | 后期放大 | 运动衰减 | ID 保持 | 背景稳定 | Polygon | 结论 |
|---|---:|---|---|---|---|---|---|
| `retrieval1` | 9 | 轻微 | 轻微 | 好 | 好 | 无 | 可保留 |
| `retrieval1_age24` | 9 | 无 | 无 | 好 | 好 | 无 | **推荐** |
| `retrieval1_motion1_age24` | 9 | 无 | 无 | 好 | 好 | 无 | **推荐** |
| `motion1_sink3_extra` | 11 | - | - | - | - | **有** | 排除 (bug) |
| `landmark2_motion1_sink3_budget9` | 9 | - | - | - | - | **有** | 排除 (bug) |

对比 v116 控制组 (同 prompt)：
- `landmark_motion1` (v116 P2)：v119 未重新生成，v116 评价为干净
- `landmark_retrieval2` (v116 P1)：v119 未重新生成，v116 评价为干净
- `prototype_motion1` (v116 P4)：v119 未重新生成，v116 评价为干净

## 4. v120 候选建议

根据 v119 review 和 docs/119 §5 推荐决策顺序：

1. **`landmark_retrieval_motion`** (即 `retrieval1_motion1_age24`) — 如果 hybrid
   干净且消除了 retrieval 的后期放大 → **本轮 review 通过**
2. **`landmark_retrieval1_age24`** — 如果有界 retrieval 干净 → **本轮 review 通过**
3. `landmark_motion1_sink3_budget9` — sink3 有 polygon noise → **排除**
4. 保留 `landmark_motion1` — v116 已验证的平衡候选

建议在 v120 中使用 `landmark_retrieval_motion` 或 `landmark_retrieval1_age24`
作为 ours 候选，与 SF 和 PF baseline 对比。如果难以决定，可以同时跑两个 ours
候选 (v120 允许最多 2 个)。

## 5. 文件位置

- v119 视频：`runs/v119_candidate_refinement_1video/videos/<cell>/0-0_ema.mp4`
- v119 配置：`runs/v119_candidate_refinement_1video/configs/<cell>.json`
- v119 policy trace：`runs/v119_candidate_refinement_1video/traces/<cell>.policy.jsonl`
- v119 审计：`runs/v119_candidate_refinement_1video/diagnostics/<cell>.{policy,role_event,video}.json`
- v116 控制组视频：`runs/v115_role_memory_cache_1video/videos/<cell>/0-0_ema.mp4`
