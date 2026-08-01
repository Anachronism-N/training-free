# 159: v153 结果分析与 v154 History-Critical 跨 Prompt 验证

日期：2026-08-01

## 1. v153 能说明什么

v153 的七个单视频 cell 全部完成，且满足以下工程契约：

- 477 帧、16 FPS、完整解码；
- 无多边形噪声、提前终止或静态输出；
- `exclusive_owner=true`、`legacy_pf_labels=false`；
- 主候选有 2400 条 policy records；
- 5 个采样层覆盖 60 个 layer/head pair；
- 主候选有 200 条 TemporalPrototype 更新记录；
- 所有 policy、role-event 和 cache budget audit 通过。

因此，QK-top 120-head 地图能够稳定接入持续生成，之前 merge 路径的多边形噪声
不能再用于否定该分类。

但 v153 没有人工排序或质量指标。它只通过了“实现可用性”门控，尚未证明：

1. QK-top 优于 QK-bottom；
2. QK-top 优于 count-matched random；
3. 改善不是依靠降低运动；
4. 单视频现象能够跨 prompt 复现。

所以当前不能直接进入 128-prompt 主实验，下一步是 v154 的 16-prompt 配对验证。

## 2. Manifest 可移植性修复

v153 首次在 Linux 服务器执行时发现 `report.json` 和 PF labels 的 SHA 与 Windows
工作区不同。内容没有改变，差异来自 CRLF/LF 换行。

v152 单侧分析 manifest 已升级到 version 2：

- JSON/CSV 文本移除 BOM，并统一为 LF 后计算 SHA；
- PF labels 和三张二分类地图解析为 30x12 整数矩阵后再规范化计算 SHA；
- gzip artifact 保留 raw-file SHA；
- `--check` 比较规范化内容，不再要求工作区原始字节换行一致。

三张 head map 的矩阵及 SHA 均未改变。

## 3. v154 核心问题

> 在 cache 路由、预算、prompt 和 seed 完全相同时，v152 QK-top head membership
> 是否比 bottom-tail 和 count-matched random membership 更适合保留分散历史？

这轮不再搜索新 cache。主候选和两个分类控制统一使用：

```text
label 10 / History-Critical: sink1 + TemporalPrototype4 + recent4 = 9 FFE
label 11 / Default:          sink1 + recent8 = 9 FFE
```

## 4. 冻结 Prompt Suite

使用 Qwen Rewrite MovieBench-128 中此前 v116 的 16 条多样性索引：

```text
0, 1, 4, 7, 13, 15, 17, 24, 33, 47, 61, 67, 75, 84, 109, 124
```

覆盖人物/动物身份、多主体、快速运动、旋转与 FPV 镜头、背景演化、场景切换和
主体变形。

冻结文件：

- `prompts/moviegen_128_qwen_v154_diverse16.txt`
- `prompts/moviegen_128_qwen_v154_diverse16.json`

源 Qwen-128 canonical SHA256：
`99468409fe54322bc383376e6037196e922cfbae47814a7a4e51740ee0571281`。

## 5. 八个配对方法

| Method | Membership | label-10 route | label-11 route | 作用 |
|---|---|---|---|---|
| `sf_native` | 无分类 | SF native | SF native | 原生基线 |
| `ours_qk_top4` | QK top4/layer | Prototype4 | recent8 | 主候选 |
| `ours_qk_bottom4_control` | QK bottom4/layer | Prototype4 | recent8 | 反向控制 |
| `ours_qk_random4_control` | random4/layer | Prototype4 | recent8 | 数量控制 |
| `ours_all_recent8_control` | QK map | recent8 | recent8 | 无分散历史 |
| `ours_all_prototype4_control` | QK map | Prototype4 | Prototype4 | all-head memory |
| `ours_legacy_membership` | old-v98 304/56 | Prototype4 | recent8 | 旧分类参考 |
| `ours_legacy_reference` | old-v98 304/56 | Prototype4 | Retrieval1(age<=24) | v125 已知参考 |

共 8x16=128 个配对视频。若设置可信的 v125 reuse root，SF 和历史参考的 32 个
视频会按原始 source index 严格复用，只生成剩余 96 个视频。

## 6. 四节点生成

四个节点使用同一共享 `OUT_ROOT`。node 0 先运行 preflight，其他节点可以随后
并行启动；非零 rank 会等待 node 0 写入冻结 contract。

```bash
cd /path/to/training-free
git pull
conda activate longlive

export REPO_ROOT="$PWD"
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt

# 可选。目录内必须有 v125 的 published_manifest.json。
export V154_REUSE_V125_ROOT="$PWD/runs/v125_moviebench128_main"
```

每个节点分别设置 rank：

```bash
export NODE_RANK=<0|1|2|3>
bash scripts/run_v154_history_critical_moviebench16.sh preflight
bash scripts/run_v154_history_critical_moviebench16.sh generate
```

若默认 v125 路径没有 `published_manifest.json`，runner 会重新生成 SF/reference；
若显式设置了错误或不完整的 reuse root，则直接失败，不会静默混用视频。

所有节点完成后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v154_history_critical_moviebench16.sh audit
NODE_RANK=0 bash scripts/run_v154_history_critical_moviebench16.sh blind
```

完整性 audit 要求 8 个方法各有 16 个视频、128 个 publish marker、完全一致的
prompt/seed/checkpoint/map contract，且任何 task-level failure 都会让节点失败。

## 7. VBench-Long

本轮计算八个维度：

```text
subject_consistency
background_consistency
temporal_flickering
motion_smoothness
overall_consistency
dynamic_degree
aesthetic_quality
imaging_quality
```

VBench 不能直接读取生成目录中的 `000000.mp4`。先在 node 0 物化为
`000000-0.mp4`，并冻结 prompt-correct comparison manifest：

```bash
NODE_RANK=0 bash scripts/run_v154_vbench_long.sh prepare
```

随后四个节点都执行 `split`。该步骤将每个 30 秒视频只切分一次，避免 64 个并行
评测任务竞争同一个 `split_clip` 目录：

```bash
export NODE_RANK=<0|1|2|3>
export NUM_NODES=4
bash scripts/run_v154_vbench_long.sh split
```

四个节点各有 16 个“方法 x 指标”任务，每张卡顺序执行两个任务：

```bash
export NODE_RANK=<0|1|2|3>
export NUM_NODES=4
bash scripts/run_v154_vbench_long.sh preflight
bash scripts/run_v154_vbench_long.sh eval
```

全部完成后在 node 0 汇总：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v154_vbench_long.sh collect
```

输出包括：

```text
<RUN_ROOT>/metrics/vbench_long_summary.{json,csv,md}
<RUN_ROOT>/analysis/v154_vbench_analysis.{json,md}
```

自动分析单独报告 history consistency、visual quality、temporal quality 和
dynamic degree，防止以冻结运动换取表面一致性。

每个指标单独保存 `job_contract.json`、`prompt_mapping.json`、`results.json`、
`done.json` 和 `run.log`。其中 prompt mapping 必须精确覆盖当前 16 条 Qwen
prompt；不再依赖 VBench 默认 prompt 顺序。重复执行 `eval` 只会恢复通过完整
hash/coverage 检查的任务。

## 8. 盲审

`blind` 会生成 128 个匿名 hardlink/symlink，并对每条 prompt 独立随机化方法
顺序：

```text
<RUN_ROOT>/blind_review/reviewer/videos/
<RUN_ROOT>/blind_review/reviewer/v154_review_sheet.csv
<RUN_ROOT>/blind_review/private/v154_blind_key.json
```

Reviewer 只接触 `reviewer/`。填完表格后执行：

```bash
python scripts/analyze_v154_blind_review.py \
  --review-sheet "$RUN_ROOT/blind_review/reviewer/v154_review_sheet.csv" \
  --blind-key "$RUN_ROOT/blind_review/private/v154_blind_key.json" \
  --output-root "$RUN_ROOT/analysis"
```

分析会输出 QK-top 相对七个方法的逐 prompt W/T/L、均值差和 prompt bootstrap
95% CI。

## 9. 推进门槛

QK-top 进入 128-prompt 需要同时满足：

1. 所有视频、policy trace 和 role-memory trace 无结构错误；
2. 相对 bottom 和 random，盲审至少 10/16 prompts 不劣；
3. QK-top 严重失败不超过 1 条；
4. 身份/背景平均改善为正；
5. motion 人工评分平均退化不超过 0.25；
6. VBench history consistency 高于 bottom/random；
7. visual/temporal quality 基本不退化；
8. dynamic degree 相对两个 membership control 的下降不超过 0.03。

若 QK-top 不优于 bottom/random，则 v152 仍可保留为 profiling 负/部分正结果，
但不能作为生成方法的 head-classification 贡献。若 QK-top 通过而 all-prototype
更好，则需要进一步比较效率和 cache budget；若 QK-top 同时优于 all-head，才
能支持“选择性历史保留”这一更强故事。
