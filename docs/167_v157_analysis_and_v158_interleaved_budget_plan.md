# 167: v157 结果复核与 v158 嵌套层预算实验

日期：2026-08-02

## 1. 本轮状态

v157 已完成：

- 8 methods x 16 prompts = 128 videos；
- 64 个新生成、64 个复用；
- generation audit、package、blind package 均完成；
- VBench core-9 为 72/72 tasks，`missing=[]`；
- blind review sheet 有 128 行，但所有人工评分字段均为空。

因此当前状态是：**自动指标 screen 通过，human promotion 未决**。

## 2. 结果复核

### 2.1 主要点估计

| Method | Dynamic | Flicker | Smoothness | Quality Score |
|---|---:|---:|---:|---:|
| SF | .64167 | .96804 | .98218 | 83.04 |
| early10 | .77500 | .95811 | .97905 | 83.83 |
| middle10 | .77500 | .96132 | .98003 | 84.24 |
| late10 | .75833 | .95882 | .97988 | 83.68 |
| interleaved10 | .79167 | .96230 | .98150 | 84.54 |
| all reservoir | .83333 | .95468 | .97708 | 83.72 |
| QK top | .72083 | .96346 | .98166 | 83.76 |
| all recent8 | .73333 | .95941 | .97960 | 83.60 |

interleaved10 是预注册 primary，也是最好的点估计，但不是唯一通过者。

### 2.2 冻结 gate 的实际结果

| Candidate | Dynamic vs recent | Temporal vs all-reservoir | History vs recent | Temporal vs recent | Visual vs recent | Pass |
|---|---:|---:|---:|---:|---:|---|
| early10 | +.04167 | +.00270 | -.00003 | -.00092 | +.00222 | False |
| middle10 | +.04167 | +.00480 | +.00396 | +.00117 | +.00500 | True |
| late10 | +.02500 | +.00347 | +.00177 | -.00015 | -.00142 | True |
| interleaved10 | +.05833 | +.00603 | +.00429 | +.00240 | +.00578 | True |

复合指标口径：

```text
history  = mean(subject, background, overall consistency)
temporal = mean(temporal flickering, motion smoothness)
visual   = mean(aesthetic quality, imaging quality)
```

旧版结果文档曾把单项 flicker/imaging 写到 composite 栏；冻结分析 JSON 和 gate
未受影响，`docs/166_v157_run_results.md` 已纠正。

## 3. 可以与不可以得出的结论

可以得出：

- reservoir 的收益/代价受 layer placement 影响；
- 10/30 层的 reservoir 可以保留明显 dynamic gain，同时恢复部分 temporal
  stability；
- middle、late、distributed 三种放置都可能有效；
- 这属于 cache allocation，不支持恢复 QK head taxonomy。

不能得出：

- interleaved10 是唯一最优层布局；
- dynamic gain 一定代表人眼更好的运动；
- layer 1/4/7/... 是功能性 motion layers；
- 16 prompts、单 seed 足以支持扩大到 128 prompts；
- 未填写的人评可以由 VBench 代替。

主要风险是：样本仅 16 prompts、单 generation seed；Dynamic Degree 接近
clip-level binary；多个 layer route 同时过 gate；人评缺失。因此下一轮必须是
条件式、预注册的预算确认，不做新的 placement search。

## 4. v158 假设与实验网格

Primary hypothesis：

> interleaved8 用比 v157 interleaved10 少 20% 的 reservoir layers，仍保留
> v157 Pareto improvement。

嵌套 maps：

| Budget | Reservoir layers | Heads | Status |
|---:|---|---:|---|
| 6 | 1,7,13,16,22,28 | 72 | exploratory lower bound |
| 8 | 1,4,7,13,16,22,25,28 | 96 | preregistered primary |
| 10 | 1,4,7,10,13,16,19,22,25,28 | 120 | exact v157 reference |
| 12 | 0,1,4,7,10,13,16,19,22,25,28,29 | 144 | exploratory upper bound |

所有集合严格嵌套，且每个 selected layer 的 12 heads 全部使用
`sink1 + TemporalReservoir4 + recent4`；其他层统一为 `sink1 + recent8`。

八方法共 128 videos：

- 新生成 48：interleaved6/8/12；
- 复用 v157 80：SF、interleaved10、middle10、all-reservoir、recent8；
- 四节点各 32 tasks；
- 6/12 只报告 dose curve，不能事后替换 8-layer primary。

## 5. v158 冻结判定

interleaved8 必须同时通过原 v157 五个 gate：

- dynamic vs recent `>= +.02`；
- temporal vs all-reservoir `>= +.003`；
- history vs recent `>= -.002`；
- temporal vs recent `>= -.004`；
- visual vs recent `>= -.01`。

还必须相对 exact interleaved10 reference 非劣：

- dynamic `>= -.02`；
- temporal `>= -.002`；
- history `>= -.002`；
- visual `>= -.005`。

Blind promotion 只比较 primary interleaved8 与 interleaved10/recent8：

- primary severe failures `<= 1`；
- overall noninferior prompts `>= 10/16`；
- mean(identity, background) delta `>= -.125`；
- motion delta `>= -.25`。

middle10 和 all-reservoir 是 contextual controls，不参与 v158 primary human gate。

## 6. 启动约束

2026-08-03 协议修订：考虑到完整 128-video 人评成本，v158 现在接受两种冻结
授权，详细依据、边界和操作见 `docs/169_v157_metric_screened_review_and_v158_amendment.md`。
原完整盲审仍然有效；新增的 64-video metric-screened confirmation 只授权 v158
16-prompt pilot。

v158 CPU preflight 已通过合同、map、reuse source 和 shard 检查，但按预注册决策
返回：

```text
[v158-preflight] HOLD ... blind=missing
contract_sha256=0b8e31963735a9c178ea2871d1c7d4f0dedae4395b65171173f3410485afc5a1
```

原始路径可检查冻结的 v157 blind report，且必须满足：

```text
experiment = v157_layer_gated_moviebench16_blind_review
primary = ours_layer_interleaved10_reservoir4
prompt_count = 16
human_promotion_gate = true
```

修订路径检查 `v157_metric_screened_confirmation_report.json`、固定四方法、64 条视频、
`metric_screened_confirmation_gate=true` 以及全部源证据哈希。两种授权均缺失或失败
时 GPU launch 会硬阻断。通过后另写
`contracts/v157_human_authorization.json`，避免运行时授权悄悄修改实验合同。

## 7. 代码入口

```text
scripts/build_v158_interleaved_budget_maps.py
scripts/run_v158_interleaved_budget_moviebench16.py
scripts/run_v158_interleaved_budget_moviebench16.sh
scripts/prepare_v158_blind_review.py
scripts/analyze_v158_blind_review.py
scripts/prepare_v158_vbench_comparison.py
scripts/prepare_v158_vbench_splits.py
scripts/run_v158_vbench_long.py
scripts/run_v158_vbench_long.sh
scripts/analyze_v158_vbench.py
tests/test_v158_interleaved_budget_experiment.py
```

预检：

```bash
PYTHON_BIN=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python \
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_v158_interleaved_budget_moviebench16.sh preflight
```

只有任一种 v157 人工授权通过后，四个节点才分别运行：

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_v158_interleaved_budget_moviebench16.sh generate
```

## 8. 下一步顺序

1. 完成 doc 169 的 64-video blind confirmation，或原 128-video 完整 blind review；
2. 若所选 human gate 失败，停止 v158 GPU 生成并分析失败维度；
3. 若通过，运行 v158 48 个新视频、audit、blind、core-9；
4. 只有 v158 metric 与 human gate 都通过，才考虑更多 prompts/seeds；
5. head profiling 的后续设计见
   `docs/166_head_profiling_classification_results_and_usage.md` 第 9-12 节。
