# 当前方法摘要与人工 Review 协议

> 日期：2026-07-20
> 目的：用最少歧义说明当前 idea、视频面板和人工判定流程

## 1. 当前 idea

一句话版本：**在 PF 原生近程/分头 cache 之外维护 training-free clean-frame archive，
按当前 query 检索远期完整视觉帧，但只让非 oscillating heads 读取，并用置信度控制的
替换式融合更新下一块 clean representation。**

具体组件：

1. PF 仍是 baseline，继续负责原生 per-head recent/history cache；
2. 每个完成块的 clean K/V 以完整空间网格写入远期 archive，不跨帧平均；
3. 检索排除最近 4 帧，从远期历史选 query 最匹配的 top-1 帧；
4. 未选帧在 token attention 前裁剪，archive 上限 64 帧；
5. memory 只在 clean consolidation pass 读取，不直接反复注入 noisy denoising steps；
6. confidence 与 native-memory alignment 共同控制 convex replacement；
7. 当前只允许 PF labels `{1,2}` 读取，排除 oscillating label `-1`。

与原始 PF 的区别：PF 的标签决定不同的缓存保留策略；我们新增的是一个远期完整帧
archive、query retrieval、独立读取时机和 memory-specific head routing。当前 PF 标签只是
第一版 router，后续目标是用 drop/swap/shuffle 干预得到 continuous functional roles。

已吸收的外部思路：LongLive-RAG 的 retrieval/recent exclusion、Flash-VAReason 启发的
clean history 组织、PF 的 head heterogeneity。未完成：Echo 场景切换/recall、MemRoPE
native token position composition、IAMFlow entity registry、SF/CF 正式验证。

## 2. 最简单的双列 Review

```text
runs/REVIEW_v44_headroute_30s/pairwise/
  0_pf_vs_ours.mp4   # parkour；左 PF，右 Ours
  1_pf_vs_ours.mp4   # cafe；左 PF，右 Ours
```

这里的 Ours 是 stable labels `{1,2}`、gate 0.20。两边严格同步，均为 29.8125 秒。

原始单视频：

```text
runs/REVIEW_v44_headroute_30s/pf/        # PF baseline
runs/REVIEW_v44_headroute_30s/stable12/  # 当前候选
```

## 3. 推荐盲评

先不要查看 `BLIND_MAPPING.txt`，依次播放：

```text
runs/REVIEW_v44_headroute_30s/pairwise/0_blind_AB.mp4
runs/REVIEW_v44_headroute_30s/pairwise/1_blind_AB.mp4
```

第一遍全屏、1x、不中断观看，记录每条整体更好的 A/B 或无差异。第二遍暂停并记录问题的
`prompt / A-B / timestamp / artifact`，最后再打开 mapping。

建议逐项给 1-5 分：

| 项目 | 主要检查内容 |
|---|---|
| Identity | 脸、发型、体型、服装是否持续为同一主体 |
| Anatomy | 手脚数量、关节、落地/翻越时是否液化 |
| Ghosting | 双轮廓、残影、旧姿态闪回 |
| Background | 建筑/道路边界是否周期性突变 |
| Motion | 是否连续，是否冻结、回放或重复轨迹 |
| Exposure | 是否逐段变暗、过曝或颜色突然改变 |

硬否决条件：Ours 独有的额外肢体、明显闪回、背景硬切、持续变暗或动作回放。初步晋级
要求：两条视频 Ours 均不触发硬否决，且至少一条被明确偏好；“几乎看不出差异”不算
论文级提升，只允许进入更多 prompt 的筛选。

## 4. 关于 PF 原生伪影

目前没有足够证据概括为“PF 原生有伪影”。parkour 两版本前 25 个解码帧完全一致；
cafe 首帧近似一致但非逐像素相同。共同运动模糊可能是合理摄影效果。只有错误多轮廓、
额外肢体、闪回、背景不连续等才能记为伪影，并必须记录发生在哪一侧及时间戳。

## 5. 五列视频说明

`comparisons/*_fiveway.mp4` 是五个方法同步横向排列。左到右依次为：PF、all-head
archive、label-1 archive、labels-{1,2} archive、label-1 强 gate。五列用于机制消融；
判断当前方案是否优于 PF 时，应优先使用上面的双列盲评视频。
