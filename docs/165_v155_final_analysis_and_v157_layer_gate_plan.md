# 165: v155 最终分析与 v157 Layer-Gated Reservoir 计划

日期：2026-08-02

## 1. 本轮实验状态

本轮有完整结果的是 v155，不是 v156：

- v155 生成：7 methods × 16 prompts = 112 videos，audit PASS；
- v155 VBench core-9：63/63 tasks 完成，`missing=[]`；
- v155 blind review：112 个匿名视频已准备，但 8 个评分字段仍全部为空；
- v156：只有 CPU preflight 和冻结合同，没有生成视频、node summary、audit、
  blind 或 VBench 结果，不能把它写成已完成实验。

非 core 的 7 个 VBench semantic dimensions 仍未完成。这些维度依赖额外模型
和 benchmark-specific auxiliary labels，不能用于任意 MovieBench prompts 的有效
结论。core-9 已足够执行 v155 预声明决策规则。

## 2. v155 完整 core-9 结果

| Method | Subject | Background | Flicker | Smooth | Overall | Dynamic | Aesthetic | Imaging |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SF native | .96868 | .96103 | .96804 | .98218 | .23314 | .64167 | .61629 | .68914 |
| QK top reservoir | .96959 | .96233 | .96346 | .98166 | .23824 | .72500 | .61661 | .70672 |
| QK bottom reservoir | .96891 | .96019 | .96127 | .97989 | .23741 | .76250 | .61965 | .71494 |
| QK random reservoir | .96959 | .96194 | .96423 | .98308 | .23794 | .70833 | .62167 | .71093 |
| all-head reservoir | .96457 | .95842 | .95468 | .97708 | .24142 | .83333 | .61824 | .69684 |
| QK top prototype | .97164 | .96377 | .96344 | .98175 | .23166 | .67917 | .62202 | .69579 |
| all recent8 | .96857 | .96121 | .95941 | .97960 | .23087 | .73333 | .61968 | .70790 |

官方可计算的 Quality Score 最高是 QK-bottom reservoir `84.01`，随后是
QK-random `83.90`、QK-top `83.79`、all-reservoir `83.72`、recent8 `83.60`、
prototype `83.42`、SF `83.05`。Semantic 和 Total 因完整 semantic contract
缺失而保持 `n/a`。

## 3. 定量解释

### 3.1 QK-top 相对 SF

从冻结分析中的派生指标看，QK-top reservoir 相对 SF：

```text
history consistency  +0.00244
visual quality       +0.00895
temporal quality     -0.00255
dynamic degree       +0.08333
```

这说明稀疏 reservoir 路由能提高运动和部分视觉/一致性指标，但存在轻微 temporal
stability 代价。它支持 cache 机制有用，不足以支持 QK 分类有用。

### 3.2 QK membership gate

QK-top 相对 bottom：

```text
history consistency  +0.00122
visual quality       -0.00563
temporal quality     +0.00198
dynamic degree       -0.03750
```

dynamic delta 低于预声明 non-inferiority 下限 `-0.03`，bottom gate 失败。

QK-top 相对 random：

```text
history consistency  +0.00023
visual quality       -0.00464
temporal quality     -0.00109
dynamic degree       +0.01667
```

差异很小，random 的 subject consistency 几乎与 top 完全相同，且 random 的
visual quality 更高。top 没有表现出稳定、独特的 membership 优势。

因此冻结 `metric_promotion_gate=false`。结论是：

> Cache useful, classifier unsupported.

v152 QK top4/layer 可以保留为稳定的 profiling observation，但不能作为生成
方法贡献或功能性 head taxonomy。

### 3.3 Reservoir 与 recent8

all-reservoir 相对 all-recent：

```text
history consistency  +0.00125
visual quality       -0.00626
temporal quality     -0.00362
dynamic degree       +0.10000
```

这是当前最可用的机制信号：把 reservoir 扩到所有 heads 会获得明显运动增益，
同时带来可测的 temporal/visual 代价。这是 Pareto allocation 问题，而不是
head 分类问题。

all-reservoir 相对 SF 的 dynamic 增益为 `+0.19167`，但 flicker 从 `.96804`
降到 `.95468`，smoothness 从 `.98218` 降到 `.97708`。它定义了高运动、低稳定
端点；recent8 和 SF 定义较稳定端点。

### 3.4 Reservoir 与 Prototype

同样 QK-top membership 下，reservoir 相对 Prototype：

```text
history consistency  +0.00104
visual quality       +0.00275
temporal quality     -0.00003
dynamic degree       +0.04583
```

reservoir 比 Prototype 更能保留运动，且 aggregate visual/temporal 没有明显
恶化。但 Prototype 的 subject/background 单项更高，所以不能声称 reservoir
统一改善 identity retention。

## 4. 人工盲评状态

`v155_review_sheet.csv` 有 112 行，以下字段当前均为 0/112 已填写：

- identity continuity；
- background continuity；
- motion quality；
- artifact free；
- late stability；
- prompt fidelity；
- overall preference；
- severe failure。

因此没有 human promotion result。它不改变 objective membership gate 已失败
的事实，但仍值得完成，用于判断 reservoir 的运动增益是否伴随人眼可见的身份
漂移、背景漂移或后半段崩坏。

## 5. 为什么当前不建议运行 v156

v156 精确复现 frame-117 profiling context 的固定 `uniform8` anchors，但其启动
条件原本是 v155 subject/background 与盲评出现明确 top-specific advantage。
现在完整 core-9 没有提供这种证据，盲评又尚未评分。

此外 v152 原始 oracle policy-choice gate 已失败，v154 和 v155 的 top/bottom/
random 生成结果也连续不支持静态 membership。继续运行 v156 只会对一个已多次
失败的静态分类做更精确的实现复核，信息价值低于直接研究 cache placement。

所以 v156 代码保留用于审计和可复现性，但状态设为 hold，不扩大、不启动。

## 6. 下一步：v157 Layer-Gated Reservoir

### 6.1 假设

> Reservoir 的运动收益和 temporal stability 代价在 transformer 深度上并非
> 均匀分布；只在一组层启用 reservoir，可以保留一部分 all-reservoir 的运动
> 增益，同时恢复 temporal stability。

该实验不使用新 head classifier。每个 layer candidate 都选择完整的 10 层，
即 `10 × 12 = 120 heads`，严格匹配 v152 QK-top 的总 head 数。四个方法使用
完全相同的 `sink1 + reservoir4 + recent4` 策略，其他层统一使用
`sink1 + recent8`，唯一变量是 layer placement。

### 6.2 冻结 layer maps

| Map | Reservoir layers | Heads |
|---|---|---:|
| early10 | 0-9 | 120 |
| middle10 | 10-19 | 120 |
| late10 | 20-29 | 120 |
| interleaved10 | 1,4,7,10,13,16,19,22,25,28 | 120 |

early/middle/late 三组互斥并覆盖 30 层；interleaved 是跨深度分布对照。

### 6.3 八方法网格

| Method | Source | Purpose |
|---|---|---|
| SF native | reuse v155 | native endpoint |
| early10 reservoir | new | early layer placement |
| middle10 reservoir | new | middle layer placement |
| late10 reservoir | new | late layer placement |
| interleaved10 reservoir | new | distributed placement, blind primary |
| all reservoir | reuse v155 | maximum-motion endpoint |
| QK-top reservoir | reuse v155 | rejected-classifier reference |
| all recent8 | reuse v155 | local-history endpoint |

总计 8×16=128 videos，其中 64 新生成、64 严格复用。四节点各 32 tasks，实际
每节点只有 16 个新生成 tasks。

### 6.4 冻结 metric gate

每个 layer candidate 必须同时满足：

- dynamic 相对 recent8 至少 `+0.02`；
- temporal quality 相对 all-reservoir 至少恢复 `+0.003`；
- history consistency 相对 recent8 不低于 `-0.002`；
- temporal quality 相对 recent8 不低于 `-0.004`；
- visual quality 相对 recent8 不低于 `-0.01`。

任一候选通过才允许进入确认阶段。interleaved10 是预声明 blind primary，并
必须同时对比其他三个 layer routes、all-reservoir 与 recent8；若 VBench 事后
选择其他层段，该结果只能视为探索发现，必须另做确认，不能 post hoc 替换
primary。

## 7. 运行入口

CPU preflight：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
PYTHON_BIN=python NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v157_layer_gated_moviebench16.sh preflight
```

四节点分别设置 `NODE_RANK=0/1/2/3`：

```bash
export NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
export V157_REUSE_V155_ROOT=$PWD/runs/v155_profile_aligned_moviebench16/full7
nohup bash scripts/run_v157_layer_gated_moviebench16.sh generate \
  > runs/v157_layer_gated_moviebench16/node${NODE_RANK}.log 2>&1 &
```

生成后：

```bash
NODE_RANK=0 bash scripts/run_v157_layer_gated_moviebench16.sh audit
bash scripts/run_v157_layer_gated_moviebench16.sh blind
bash scripts/run_v157_layer_gated_moviebench16.sh package
```

VBench 使用已有离线缓存：

```bash
NODE_RANK=0 bash scripts/run_v157_vbench_long.sh prepare
# 四节点分别 split/eval；失败只补 missing
NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v157_vbench_long.sh split
NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v157_vbench_long.sh eval
NODE_RANK=<0|1|2|3> NUM_NODES=4 bash scripts/run_v157_vbench_long.sh resume-missing
# rank0 收集合法 core-9
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v157_vbench_long.sh collect-core
```

## 8. 停止条件与边界

- 如果四个 layer routes 都未通过 metric gate，停止 reservoir placement 线；
- 如果只有 all-reservoir 增加运动，而人评显示明显闪烁、身份或背景崩坏，不把
  dynamic 增益作为改进；
- 如果 layer routes 几乎无差异，不声称层功能分类；
- 如果 interleaved 同时通过 metric 和 blind gate，再做更小的 layer-count
  budget sweep，而不是直接扩大到 128 prompts；
- 所有结论仅属于 cache allocation，不恢复 QK head taxonomy 声称。

GPU 占卡、代理和管理员许可约束沿用 `docs/164`：当前代码不自动恢复历史占卡
守护进程，实验结束后的占卡操作必须符合 `GPU占卡.md` 的现行规则。

## 9. 代码与验证状态

已实现：

- 4 张冻结 layer maps 与语义 manifest 检查；
- v157 生成、严格 v155 复用、publish audit、blind package；
- VBench prepare/split/status/resume-missing/collect-core；
- 多候选 Pareto gate 与预声明 interleaved blind primary；
- v155 core-9 结果哈希和 `classifier unsupported` 决策写入实验合同。

验证结果：

```text
82 passed  # v97-v157 相关策略、缓存所有权和实验合同扩大回归
python -m py_compile: PASS
bash -n: PASS
git diff --check: PASS
v157 CPU preflight: PASS
rank0 tasks: 32
new videos: 64
reuse: true
```

冻结 preflight contract SHA256：
`080af362da01995fd3356ac90d85986163e648fe3a25b35dfd80b6855ff35767`。

当前执行环境没有可用 NVIDIA driver，且 SSH socket 受限，因此本轮没有在
四个远程节点启动 v157 生成或 GPU 占卡进程。
