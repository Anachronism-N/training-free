# 157: v152 单侧重分析与 v153 生成迁移实验

日期：2026-08-01
代码基线：`5a1b677` (`codex/v98-correctness-fixes`)

## 1. 本轮结论

`docs/156` 中“oracle 不能优于随机，因此整个 policy axis 应停止”的表述过强。
原始 v152 门控要求同时找到：

1. 适合 `uniform8` 的高分组；
2. 适合 `recent8` 的低分组；
3. 两组都跨 seed 稳定并优于随机组。

这个**对称二分类假设仍然失败**，但失败原因主要是低分组并没有稳定偏向
`recent8`。它不能否定高分组的单侧结果。

v152 原始数据支持更窄、可检验的新假设：

> 对每层而言，`uniform8 QK compatibility - recent8 QK compatibility`
> 较高的一小组 head 是 History-Critical 候选；其余 head 只是 Default，
> 不再声称它们构成“Recent-preferring”类别。

这已经足以进入**最小生成迁移实验**，但还不足以作为论文生成结论。

## 2. 单侧证据

### 2.1 QK 高分组确实需要分散历史

`qk_uniform4` 使用真正可在线计算的 QK policy margin。`uniform8` 相对
`recent8` 的 X0 log-error 优势如下：

| timestep | median effect | positive fraction | seed Spearman | 原门槛 |
|---|---:|---:|---:|---|
| 1000 | 0.3723 | 0.9688 | 0.3250 | PASS |
| 750 | 0.2240 | 0.8672 | 0.3859 | PASS |
| 500 | 0.1994 | 0.8594 | 0.6573 | PASS |
| 250 | 0.2567 | 0.9844 | 0.2883 | 仅差 seed 相关阈值 0.30 |

QK 高分组相对 count-matched random 在 `t750/t500/t250` 通过原始随机
对照门槛。它与不可部署 oracle 的 score Spearman 在四个 timestep 均为
`0.531-0.538`。

### 2.2 失败的是对称低分组

`qk_recent4` 只在 `t1000` 弱偏向 recent，之后逐渐翻转为 uniform；四个
context 均未通过原始完整门槛。因此不能继续使用
“History-Critical / Recent-Critical”二分类叙述。

### 2.3 静态候选地图可复现

地图只用 seed replicate 0 发现：对每个 layer/head，在 64 prompts 和四个
timestep 上取 QK policy margin 中位数，再选每层 top 4。seed replicate 1
只用于复现检查：

- top-4 重合 `112/120`；
- `23/30` 层完全相同；
- layer-wise median Jaccard 为 `1.0`，mean Jaccard 为 `0.8978`。

冻结文件：

- `configs/head_maps/v152_qk_history_critical_top4_seed0.csv`
- `configs/head_maps/v152_qk_history_critical_bottom4_seed0_control.csv`
- `configs/head_maps/v152_qk_history_critical_random4_seed2026_control.csv`
- `configs/head_maps/v152_qk_history_critical_manifest.json`

每张控制地图均为每层 4 个 label-10、总计 `120/240`，不存在数量混杂。

### 2.4 与 PF 分类不是同一成员集合

QK-top 120 个 head 与 PF 的交叉表为：

| PF 类别 | QK-top | Default |
|---|---:|---:|
| Wave | 77 | 79 |
| Anchor | 24 | 148 |
| Veil | 19 | 13 |

因此该地图不是 PF Anchor 的换名复用。它测量的是“分散历史相对近期历史的
QK 兼容优势”，而不是 PF 的时间 logit 符号类别。

## 3. v153 生成假设

History-Critical 和 Default 使用同样的总预算、sink 和原始位置：

| 路由 | Cache | FFE |
|---|---|---:|
| History-Critical | `sink1 + TemporalPrototype4 + recent4` | 9 |
| Default | `sink1 + recent8` | 9 |

`TemporalPrototype4` 保存四个真实帧的 K/V，不做 K/V 平均；动态 cache 只有
一个 owner，禁止与 recent 重叠，并保留原始 RoPE 位置。该路径来自此前已
通过结构审计的 role-memory 实现，避免重新引入 merge/stride 多边形噪声。

## 4. v153 七个单视频 cell

全部使用同一条 Qwen MovieBench prompt、seed 0、30 秒视频。

| Cell | Head membership | label-10 | label-11 | 目的 |
|---|---|---|---|---|
| `qk_top4_prototype4_default_recent8` | QK top4/layer | Prototype4 | recent8 | 主候选 |
| `qk_bottom4_control_prototype4_default_recent8` | QK bottom4/layer | Prototype4 | recent8 | 反向成员控制 |
| `qk_random4_control_prototype4_default_recent8` | random4/layer | Prototype4 | recent8 | 数量匹配控制 |
| `legacy_v98_membership_prototype4_default_recent8` | old-v98 304/56 | Prototype4 | recent8 | 旧成员参考 |
| `qk_top4_all_recent8_control` | QK map | recent8 | recent8 | 无长期 memory 控制 |
| `qk_top4_all_prototype4_control` | QK map | Prototype4 | Prototype4 | all-head memory 控制 |
| `legacy_v98_prototype4_retrieval1_age24_reference` | old-v98 304/56 | Prototype4 | Retrieval1 | 已知可用链路 sanity check |

这里没有 PF-native cell，也没有 ABA。v153 只回答“profiling 得到的成员是否能
迁移到持续生成”这一件事。

## 5. 服务器运行

```bash
cd /path/to/training-free
git pull
conda activate longlive

export PF_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export SINGLE_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export SINGLE_PROMPT_INDEX=0

bash scripts/run_v153_history_critical_transfer.sh preflight
bash scripts/run_v153_history_critical_transfer.sh screen
```

默认使用 GPU `0-6` 并行运行全部七个 cell。也可以分批：

```bash
V153_MODE=membership GPU_LIST=0,1,2,3 \
  bash scripts/run_v153_history_critical_transfer.sh screen

V153_MODE=controls GPU_LIST=0,1 \
  bash scripts/run_v153_history_critical_transfer.sh screen

V153_MODE=reference GPU_LIST=0 \
  bash scripts/run_v153_history_critical_transfer.sh screen
```

查看状态和打包日志：

```bash
bash scripts/run_v153_history_critical_transfer.sh status
bash scripts/run_v153_history_critical_transfer.sh package
```

## 6. 必须检查的信息

1. 七个视频均为 477 帧、16 fps，不能出现多边形噪声、静态画面或提前终止。
2. head-map audit 必须覆盖全部 360 个 head；`diagnostics/*.policy.json` 必须覆盖
   预设的 5 个 trace layers，并验证其中每个 head 的 sink/recent/middle 数量。
3. 日志必须包含 `legacy_pf_labels=false`、`exclusive_owner=true` 和
   `[HistoryPolarityPolicy]`。
4. QK 三张地图必须分别报告 `120/240`，并通过 manifest SHA256 检查。
5. 人工 review 优先比较身份、背景、运动幅度、运动自然度和后半程放大/漂移。

若任一非 reference cell 出现多边形噪声，先按实现问题处理，不能把它解释为
head 分类失败。

## 7. 决策门槛

v153 不是统计结论，但负责阻止无效配置进入大规模生成：

- **推进**：QK-top 无结构错误，且盲审不差于 bottom/random，至少在身份或
  长程背景上表现出一致优势；随后运行 16 prompts，并比较 SF、QK-top、
  bottom、random 和已知 reference。
- **仅保留 profiling 结论**：QK-top 与 random/bottom 无可重复差异；不再把
  v152 写成生成方法贡献。
- **回到实现审计**：reference 正常但多个 Prototype/recent cell 同时出现
  噪声，说明新地图路由仍有实现或 trace 契约问题。

## 8. 当前可写与不可写的论文表述

当前可以写为实验假设：DiT head 的历史策略需求是非对称的，少量高 QK-margin
head 对分散历史有稳定需求，而剩余 head 未形成统一的近期类别。分类标准、
成员和 PF 均不同。

当前不能写“该分类提升了长视频质量”“形成天然二分类”或“在线动态路由已经
有效”。这些结论必须等待 v153 生成迁移以及之后的多 prompt 评测。
