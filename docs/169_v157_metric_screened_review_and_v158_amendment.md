# 169: v157 指标预筛选盲审与 v158 协议修订

日期：2026-08-03

## 1. 决策

可以先用已经完成的 VBench-Long core-9 结果筛选方法，再做较小规模的人评，
但 VBench 不能替代人工确认。为降低 128 条完整盲审的工作量，本轮新增一份冻结的
64-video 协议修订：16 prompts x 4 methods，每条视频只评 4 个连续量和 1 个严重
失败标记。

这份修订只授权 **v158 的 16-prompt layer-budget pilot**。它不等价于原 v157
128-video `human_promotion_gate`，也不能单独支持“v157 已通过完整人工评价”或直接
扩大到 128 prompts 的结论。

## 2. 为什么选择这四个方法

筛选证据固定为 v157 已完成的 core-9：8 methods x 9 dimensions = 72/72 tasks，
`missing=[]`。选择规则在看人评分数前冻结为：

1. 取 v157 冻结 candidate ranking 中前两名且通过 metric gate 的 layer route；
2. 加入 all-reservoir 与 all-recent8 两个机制端点；
3. 保留全部 16 prompts，不按 prompt 或视频质量二次挑选。

最终四个方法为：

| Role | Method |
|---|---|
| primary | `ours_layer_interleaved10_reservoir4` |
| strongest layer-route control | `ours_layer_middle10_reservoir4` |
| high-dynamic endpoint | `ours_all_reservoir4_reference` |
| recent-only endpoint | `ours_all_recent8_reference` |

盲审 key 同时冻结 VBench summary、VBench analysis、published manifest 与 prompt
manifest 的路径和 SHA-256。分析和 v158 启动时都会重新计算这些证据；方法、排序或
源文件发生变化都会拒绝授权。

未完成的 semantic dimensions 不参与选择。它们此前因远程节点缺少 DINO/CLIP、
Detectron2、FairScale 或自定义 prompt auxiliary info 而失败，不能把缺失项当作零分，
也不能用 7/9 已完成维度的旧 84% 状态代替当前冻结 core-9。

## 3. 盲审目录与填写方式

盲审包已经生成：

```text
runs/v157_layer_gated_moviebench16/full8/metric_screened_review64/
├── reviewer/
│   ├── REVIEW_INSTRUCTIONS.md
│   ├── v157_metric_screened_review.csv
│   └── videos/                         # 64 个匿名硬链接
└── private/
    └── v157_metric_screened_blind_key.json
```

评审时只打开 `reviewer/`，不要查看 `private/`。按 CSV 行打开 `videos/<video>`，
填写以下字段：

- `identity_continuity_-2_to_2`
- `background_continuity_-2_to_2`
- `motion_quality_-2_to_2`
- `overall_preference_-2_to_2`
- `severe_failure_0_or_1`
- `notes`（可空）

四个评分必须是 `[-2, 2]` 内的整数：2 优秀、1 良好、0 可接受或优劣混合、-1 较差、
`-2` 严重失败。`severe_failure=1` 只用于不可用视频，例如崩坏、持续严重伪影、黑屏或
长时间冻结。未评项目必须留空，不能用 0 表示“尚未评价”。允许并列，不要求每个
prompt 选出唯一赢家。

## 4. 冻结确认门控

primary 必须同时相对三个 controls 满足：

- primary severe failures `<= 1/16`；
- overall noninferior prompts `>= 10/16`；
- `mean(identity delta, background delta) >= -0.125`；
- motion mean delta `>= -0.25`。

这里的 delta 均为逐 prompt 的 `primary - comparator`，overall 的 noninferior
包含并列。三个 controls 全部满足才输出
`metric_screened_confirmation_gate=true`。门控字段使用新名称，防止与原始
`human_promotion_gate` 混淆。

## 5. 运行流程

重新生成或校验冻结盲审包：

```bash
bash scripts/run_v157_metric_screened_review.sh prepare
```

人工完成 CSV 后生成报告：

```bash
bash scripts/run_v157_metric_screened_review.sh analyze
```

输出为：

```text
runs/v157_layer_gated_moviebench16/full8/analysis/
├── v157_metric_screened_confirmation_report.json
└── v157_metric_screened_confirmation_report.md
```

然后检查 v158：

```bash
PYTHON_BIN=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python \
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_v158_interleaved_budget_moviebench16.sh preflight
```

门控通过后，四个节点分别运行：

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_v158_interleaved_budget_moviebench16.sh generate
```

修订后的 v158 默认目录是
`runs/v158_interleaved_budget_moviebench16/metric_screened64`。旧 `full8` 目录保留
原始合同，不覆盖、不混写。v158 仍接受通过的原 128-video 完整盲审报告；如果两份
报告都不存在则默认等待 64-video 报告。

## 6. 实现与验证

新增：

```text
scripts/prepare_v157_metric_screened_review.py
scripts/analyze_v157_metric_screened_review.py
scripts/run_v157_metric_screened_review.sh
tests/test_v157_metric_screened_review.py
```

v158 runner 会冻结实际使用的授权报告及 SHA-256 到
`contracts/v157_human_authorization.json`。测试覆盖 64 行匿名网格、证据绑定、门控
阈值、原完整盲审兼容，以及篡改证据时拒绝启动。完整 128-video 路径也必须包含
已填写 review sheet 与 blind key 的哈希，并由 analyzer 重算得到完全相同的报告；
只有布尔值、没有评分来源的手写 bypass 报告不会被接受。

## 7. 当前状态

- 64-video reviewer package 已成功生成，使用 64 个硬链接；当前评分为 `0/64`；
- 相关 v154/v157/v158 回归测试为 `26 passed`；
- v158 修订目录 CPU preflight 的合同 SHA-256 为
  `a1b03d0f9b5feef5ea8a23b329406b4c0a570cea0e47f12c20a957385396b691`；
- 当前 preflight 正确返回
  `HOLD ... human_confirmation=gate_or_provenance_failed`；
- 原 analysis 目录存在一份注明 `User-authorized bypass`、只有布尔 gate 而没有评分
  来源的文件。代码不会删除它，但已明确拒绝将其作为 v158 授权。
