# v4.3-v4.4 完整帧检索与稳定头路由

> 日期：2026-07-20
> 状态：两 prompt、30 秒筛选完成；stable labels `{1,2}` 为待人工 review 候选

## 1. 为什么停止 v4.2 参数扫描

v4.2 的 PF、all-step memory 和 clean-only memory 在 30 秒 contact sheet 中差异很小，
没有显著提升。继续扫描 additive gate 缺少依据，因此按照预先设定的停止条件，转向
不做跨帧 K/V 均值的完整帧 archive。

代码审计同时发现一个具体错误：memory output 已乘 retrieval confidence，但随后的 RMS
matching 会重新把其幅值放大到 native output 的尺度，基本抵消 confidence。v4.3 将
confidence 显式放入最终 fusion weight，使低置信度历史无法被 RMS matching 恢复强度。

## 2. v4.3 实现

当前实现保持 training-free：

1. 仅归档 clean pass 的完整空间 K/V，不把不同历史帧融合成伪帧；
2. archive 上限 64 帧，超过上限时均匀保留，不训练 retrieval encoder；
3. 排除最近 4 帧，避免重复 native recent context；
4. 用当前 raw query 与历史完整帧 descriptor 选择 top-1；
5. 在 token attention 前物理裁剪未选历史帧，30 秒速度只比 PF 慢约 2%；
6. 使用 confidence + alignment 控制的 convex replacement，不再额外加 memory residual。

限制：convex replacement 控制的是 output mixture budget，还不是 LongLive-RAG 的 native
attention token 总预算替换。当前分支也没有对 recalled token 应用单独的 temporal RoPE；
完整空间网格避免了 arbitrary sparse token 的伪空间坐标，但 position-safe native cache
composition 仍是后续工作。

## 3. v4.3 30 秒结果

prompt 为高速 parkour 和 cafe-to-rainy-street，全部 120 latent frames、29.8125 秒。

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.91484 | 0.90919 | 0.57474 | 0.52458 | **0.97136** | 1.0 |
| Confidence-fixed compressed | 0.91107 | 0.90931 | **0.59509** | **0.55555** | 0.96648 | 1.0 |
| Full-frame query | **0.91523** | **0.91015** | 0.56771 | 0.54096 | 0.97135 | 1.0 |
| Full-frame lag-5 | 0.90499 | 0.90825 | 0.56936 | 0.54524 | 0.97020 | 1.0 |

结论：固定 lag-5 是负结果；query retrieval 只有极小一致性收益，仍不足以晋级。parkour
抽帧中 all-head query 约 18 秒存在疑似多肢体，说明历史动作相位会干扰 motion heads。

## 4. v4.4 PF head routing

v4.4 使用同一个 full-frame query archive，只改变允许读取 memory 的 PF 标签：

- `stable1`：只允许 label `1`；
- `stable12`：允许 label `{1,2}`，排除 oscillating label `-1`；
- `stable1g30`：label `1`，更强 gate 0.30。

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.91484 | 0.90919 | 0.57474 | 0.52458 | 0.97136 | 1.0 |
| All-head query | 0.91523 | 0.91015 | 0.56771 | **0.54096** | 0.97135 | 1.0 |
| Stable `1`, gate 0.20 | 0.91310 | 0.91292 | 0.56745 | 0.53436 | 0.97286 | 1.0 |
| Stable `{1,2}`, gate 0.20 | **0.91884** | **0.91672** | **0.57619** | 0.53023 | 0.97397 | 1.0 |
| Stable `1`, gate 0.30 | 0.91481 | 0.90610 | 0.56436 | 0.53045 | **0.97512** | 1.0 |

`stable12` 是当前唯一在五个非饱和维度均高于 PF 的候选，但绝对提升仍小，样本只有两条；
不能称为显著优于 PF。`stable1g30` 的 cafe 后段主体明显偏暗，作为负消融保留。

这组实验支持的机制结论不是“label 1 是 identity head”，而是：oscillating `-1` heads
不适合读取远期视觉 archive，label `2` 仍包含有用稳定信息。该结论将用于后续 causal
functional head routing，不能替代真正的 identity/motion 干预分类。

## 5. 人工 review

“列”指同一个同步视频画面被分成五个竖直面板，并不是五个时间段。由左到右：

1. PF baseline：只使用原生 Pyramid Forcing，不启用我们的 archive；
2. All-head query：所有 PF head 都可读取完整帧 archive；
3. Stable-1 0.20：只有 PF label `1` 可读取 archive；
4. Stable-1+2 0.20：labels `{1,2}` 可读取、排除 `-1`，是当前候选；
5. Stable-1 0.30：更强 gate 的负消融。

```text
runs/REVIEW_v44_headroute_30s/comparisons/
  0_fiveway.mp4     # parkour
  1_fiveway.mp4     # cafe -> rainy street
  0_opening.png
  1_opening.png
  0_timeline.png
  1_timeline.png
```

五列为 PF、all-head query、stable-1 0.20、stable-1+2 0.20、stable-1 0.30。必须完整观看，
重点检查多肢体、重影、背景块边界、主体变暗、face ID 和动作回放。任何严重人工伪影都
覆盖 VBench 的晋级判断。

更易 review 的双列和盲评入口见 `docs/46_manual_review_protocol_and_method_summary.md`。

### 伪影归因限制

目前不能笼统声称“PF 原生有伪影”。parkour 的 PF 与 stable12 前 25 个解码帧完全一致，
cafe 首帧两者 PSNR 为 54.3 dB、视觉近似一致，但共同出现的模糊可能是合理运动模糊，
不一定是生成错误。有效归因必须带方法、时间戳和现象：

- PF 独有且重复出现：该 PF sample 的问题；
- 两者共同、且发生在分化前：base/sample 共有现象；
- Ours 独有且发生在分化后：我们的方法引入；
- 正常 motion blur 不计为伪影，只有错误双轮廓、额外肢体、闪回或不连续才计。

## 6. 下一决策

1. 人工确认 `stable12` 是否在两条视频中均不劣于 PF；若否，停止 PF archive 路线。
2. 若通过，先扩至 60 秒和更多 prompt，确认收益随长度增长，而不是两 prompt 方差。
3. 之后移植到 native SF 和 Causal Forcing；PF 结果本身不能构成论文有效性结论。
4. 同时建立 Echo A-B/A-B-A 场景切换任务，测试 retrieval 是否召回正确场景。
5. 最终将静态 PF label mask 替换为 drop/swap/shuffle 干预得到的连续功能路由。

## 7. 验证

- 仓库测试：25 passed
- PF cache/memory 聚焦测试：39 passed
- Python compile、shell `bash -n` 和 `git diff --check` 通过
- v4.3/v4.4 raw VBench JSON：`runs/vbench_long/v43_archive_30s/`、
  `runs/vbench_long/v44_headroute_30s/`
