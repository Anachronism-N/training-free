# v120 人工观感与指标分歧审计

日期：2026-07-28

状态：已完成聚合结果审计；逐 prompt 统计等待原始 JSON 推送。本文是
`docs/122` 和 `docs/123` 的解释层更正，不改写原始实验记录。

## 1. 当前结论

1. **Ours 相对 SF 的优势已经被部分指标捕捉到**。三个 Ours 在 DINO
   consistency、drift、原始 long-range clip2clip、aesthetic 和 imaging 上均
   优于 SF。标准 VBench subject/background overall 没有呈现这一优势。
2. **Ours 相对 PF 不是聚合指标赢家，而是可能处于 consistency-motion
   trade-off 的另一位置**。PF 的 DINO 和 long-range consistency 更高；Ours
   的成像和 drift 接近 PF。人工观感提示 Ours 保留了更多有效演化，但两份文档的
   loop 数值互相冲突，必须用原始 JSON 和 motion 指标确认，不能先当作结论。
3. **现在不能写“显著优于”**。32 个 prompt 的原始逐 prompt 结果和配对置信区间
   没有进入 Git，当前只有均值。
4. **不需要重新生成视频**。下一步全部是对现有 160 个视频补充逐 prompt 统计、
   motion/repetition 评测和盲评。

## 2. 聚合结果的可靠解读

### 2.1 Ours vs SF

以下采用每个指标上最好的 Ours 候选：

| Signal | Ours - SF | 解读 |
|---|---:|---|
| VBench aesthetic | +0.01885 | Ours 更好 |
| VBench imaging | +0.01267 | Ours 更好 |
| DINO consistency | +0.01850 | Ours 更稳定 |
| DINO drift slope | +0.00238 | 更接近 0，但单位是每个采样点 |
| raw subject clip2clip | +0.00541 | 长期主体表观略好 |
| raw background clip2clip | +0.02756 | 长期背景明显更好 |
| VBench subject overall | -0.00154 | 标准融合分略低 |
| VBench background overall | -0.00083 | 标准融合分略低 |

因此，“指标完全没有体现视觉优势”并不准确。更准确的说法是：

> 长期一致性和帧质量指标体现了 Ours 相对 SF 的收益，但 VBench-Long 的标准
> subject/background 融合分没有体现。

### 2.2 Ours vs PF

| Signal | 最接近 PF 的 Ours - PF | 解读 |
|---|---:|---|
| VBench imaging | -0.00067 | 基本持平 |
| VBench motion smoothness | +0.00018 | 基本持平 |
| DINO drift slope | +0.00008 | 基本持平 |
| VBench aesthetic | -0.00707 | PF 更高 |
| DINO consistency | -0.02310 | PF 更高 |
| raw subject clip2clip | -0.03738 | PF 更高 |
| raw background clip2clip | -0.03028 | PF 更高 |
| loop score (`docs/122`) | -0.0934 至 -0.1167 | Ours 更低 |
| loop score (`docs/123`) | -0.0185 至 +0.0032 | 只有部分 Ours 更低 |

相对 SF，Ours 已取得 PF-SF DINO 增益的约 44%，取得 PF-SF imaging 增益的约
95%。这支持继续检验“在保持与演化之间取得平衡”的故事，不支持现在就宣称
“Ours 比 PF 更少重复”，也不支持“所有自动指标超过 PF”的故事。

## 3. 为什么 VBench overall 与人工观感不一致

### 3.1 slow-fast 融合压缩了高分区间的长期差异

官方 VBench-Long 对 subject/background 使用：

```text
overall = 0.5 * inclip + 0.5 * mapped_clip2clip
```

`clip2clip` 通过官方预先计算的固定 mapping table 映射到 `inclip` 的量纲。该映射
在 0.85 以上的高分区间很平，因而会压缩绝对差值。它不是对每个待评方法重新做
一次分位数排名。

以 subject 为例：

```text
SF:                 inclip 0.97909, raw clip2clip 0.85673
Ours retrieval:     inclip 0.97589, raw clip2clip 0.86214
```

Ours 的长期分数更高，但 SF 的短期分数高 0.00320。长期差值映射后被压缩，最终
overall 仍由 SF 略胜。背景维度也是同一机制。

### 3.2 consistency 和 smoothness 会奖励低运动

- DINO consistency 同时比较当前帧与首帧、相邻帧；静止、回到旧构图或重复轨迹
  都可能提高分数。
- VBench temporal flickering 和 motion smoothness 更偏好局部变化小且平滑的
  视频，不判断动作是否丰富、合理或推进故事。
- v120 缺失 `dynamic_degree`，恰好缺少了最直接的运动侧信号；官方
  `dynamic_degree` 本身还是二值视频级判定，后续仍应补连续光流统计。
- `docs/122` 的 loop score 为“PF 的一致性收益部分来自重复/保守运动”提供了
  线索，但 `docs/123` 不支持所有 Ours 都更低。该信号必须回到原始 JSON，并与
  人工逐 prompt 记录共同使用。

### 3.3 全视频均值会稀释稀疏但严重的失败

身份突变、肢体复制、背景几何崩坏或后 10 秒漂移可能只发生在少量时间点。对 30 秒
全部采样帧求均值会将这些错误稀释。人工 review 往往会对一次严重失败直接降级，
均值指标不会。

## 4. 当前结果记录中的完整性问题

在原始 JSON 到位前，下面的问题必须保留为 blocker：

1. `docs/122` 与 `docs/123` 的 DINO 表不一致。DINO consistency 和 drift 相同，
   但 CLIP、background 和 loop 不同。例如 PF 的 loop 分别为 `0.3004` 和
   `0.1118`。这可能是不同 `sample_frames` 运行或表格混用，不能任选一组写论文。
2. Git 中没有 v120 的 VBench `results.json`、merged comprehensive JSON、
   frozen contract 或逐 prompt 结果，无法核验方法覆盖、参数和配对差值。
3. `drift_slope` 以均匀采样点的 index 做回归，不是“每秒漂移”；评测器默认
   `sample_frames=64`，但本轮原始 config 尚未推送。方法间同协议比较有效，
   论文中必须写 per sampled step，或重新按秒归一化。
4. repository `composite` 是手工加权诊断分，不是公开 benchmark。其
   `0.6443 vs 0.6433 vs 0.6428` 差值很小，不能作为主结果或 SOTA 依据。
5. “显著优于”尚无配对 bootstrap CI 或随机化检验支持。
6. temporal flickering 是无 static filter 版本，主表和复现实验中必须明确标注。

## 5. 当前候选的选择方式

三个 Ours 的聚合结果非常接近，不能只按某一个均值选择：

- `retrieval1_age24`：aesthetic、raw background clip2clip 和诊断 composite
  较好，结构也最简单。
- `landmark_motion1`：drift 最好，适合作为去掉 retrieval 的直接消融。
- `retrieval_motion`：imaging 最接近 PF；如果盲评确认其动作与整体观感最好，
  它最适合作为包含完整 retrieval + motion 技术点的主方法。

建议的决策规则是：

```text
先看 32-prompt 盲评 overall preference
  -> 再要求 DINO > SF、raw long-range background > SF
  -> 再确认 loop/repetition 与 continuous motion 相对 PF 的真实方向
  -> 最后在通过者中选择机制更简洁的版本
```

不要根据自定义 composite 先选方法，再解释人工结果。

## 6. 无需重新生成的补评测

### 6.1 推送小型原始文件

至少需要：

```text
runs/v120_moviebench32_main/*/published_manifest.json
runs/v120_moviebench32_main/*/contracts/experiment.json
runs/v120_moviebench32_main/*/metrics/vbench_long/**/results.json
runs/v120_moviebench32_main/*/metrics/vbench_long/**/*_eval_results.json
runs/v120_moviebench32_main/*/metrics/comprehensive_parts/*.json
runs/v120_moviebench32_main/comparison_all5/metrics/all_results_summary.json
```

`runs/` 被 `.gitignore` 忽略，需要对这些小 JSON 使用 `git add -f`。不要上传视频、
模型或 split clips。

### 6.2 运行逐 prompt 配对分析

仓库新增 `scripts/analyze_v120_paired_metrics.py`。示例：

```bash
python scripts/analyze_v120_paired_metrics.py \
  --vbench sf_native=/path/sf_native/results.json \
  --vbench pf_native=/path/pf_native/results.json \
  --vbench ours_landmark_motion1=/path/ours_motion1/results.json \
  --vbench ours_landmark_retrieval1_age24=/path/ours_retrieval/results.json \
  --vbench ours_landmark_retrieval_motion=/path/ours_hybrid/results.json \
  --comprehensive /path/all_results_summary.json \
  --references sf_native pf_native \
  --candidates \
    ours_landmark_motion1 \
    ours_landmark_retrieval1_age24 \
    ours_landmark_retrieval_motion \
  --output-json /path/v120_paired_analysis.json \
  --output-md /path/v120_paired_analysis.md
```

输出包含每个指标的 candidate/reference 均值、配对均值差、bootstrap 95% CI、
逐 prompt W/T/L 和 paired randomization p-value，并单列
`inclip/clip2clip/mapped_clip2clip`。

### 6.3 补 motion 与后程指标

直接复用现有视频补：

1. RAFT continuous flow magnitude、flow acceleration 和静止帧比例；
2. DINO sim-to-first 的 0-10s、10-20s、20-30s 分段均值；
3. last-third DINO、first-last similarity 和 time-to-failure；
4. loop/repetition 的逐 prompt 分布，而不是只报均值；
5. 32-prompt 盲评，随机左右顺序，至少比较 Ours vs SF 和 Ours vs PF。

## 7. 论文叙事边界

当前最合适的主线不是“比 PF 的一致性分更高”，而是：

> Long autoregressive extrapolation must preserve persistent identity without
> collapsing temporal evolution. A binary head-role cache separates
> persistence-oriented landmark memory from change-oriented bounded
> retrieval/motion memory, targeting lower SF drift without over-constraining
> temporal evolution.

在补齐逐 prompt 统计和盲评后，可以写：

- Ours 相对 SF 提升长期身份和背景保持；
- Ours 在成像与 drift 上接近 PF；
- 若补充结果成立，Ours 相对 PF 减少重复并获得更高人工偏好；
- binary role memory 用不同于 PF 三分类的 304/56 头划分和两类 cache 生命周期
  实现上述平衡。

当前不能写：

- Ours 在 VBench-Long 全面超过 PF；
- quantile mapping 对每个方法独立重排；
- custom composite 是标准综合分；
- drift slope 是每秒身份漂移；
- 未经配对统计的“显著提升”。
