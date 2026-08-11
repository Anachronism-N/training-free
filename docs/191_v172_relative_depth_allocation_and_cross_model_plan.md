# v172：归一化深度分配、人工规则边界与跨模型验证计划

日期：2026-08-11

## 1. 先回答当前的三个核心问题

### 1.1 `cache useful, classifier unsupported` 到底是什么意思

这句话需要拆成两个相互独立的命题。

**命题 A：cache operator 是否会产生有用的轨迹级作用。**

v155 中，把相同 reservoir history 分配给所有 head，确实能够明显提高
Dynamic Degree；v157 进一步表明，把 history cache 放在不同深度会改变运动与
时序稳定性的 Pareto 关系。随后 v159--v166 通过 atomic motion pair、clean-KV
motion signature 和多尺度检索，在 Middle10 上得到当前开发集最好的
v166 配置。因此，历史 cache、motion episode 和它们的分配位置不是“完全无效”。

**命题 B：某个静态 classifier 是否能预测哪些 head 最应该使用该 operator。**

v152 的 QK top4 membership 跨 seed 很稳定，但 v155 中它没有优于等数量的
bottom/random：

| 配置 | Quality | Dynamic |
|---|---:|---:|
| QK top reservoir | 83.79 | 72.50 |
| QK bottom reservoir | 84.01 | 76.25 |
| QK random reservoir | 83.90 | 70.83 |
| all-head reservoir | 83.72 | 83.33 |

所以当前证据支持的是：

> 被测试的 cache operator 能改变并部分改善生成轨迹，但 QK-top 这个静态成员
> 预测器没有证明自己比 bottom/random 更会选择受益 head。

它不等于“所有分类都永远失败”，也不等于“当前 cache 已经是最终论文方法”。
它只否定了已经测试过的静态 membership claim，不能通过事后换名字撤销。

### 1.2 之前的 profiling 结论还能否继续深挖

可以，但应继续研究**连续 propensity、条件作用和 operator-aligned causal
response**，不应继续在同一开发集上扫描阈值寻找一个好看的二分类。

仍然值得保留的结论包括：

1. v136 的 old-vs-recent temporal rank 很稳定；
2. v138 的 self-history specificity 是可复现连续信号；
3. v145 的 K/V/policy propensity 中，K-selection 最稳定；
4. v147--v152 证明部分排名具有局部因果响应，但静态成员效用没有转移到长轨迹；
5. v157 证明 cache placement 存在明显 layer effect；
6. v166 证明 clean-KV motion episode recall 在冻结开发集上有用。

后续应把这些分数作为协变量，研究：

```text
offline propensity
    x layer normalized depth
    x denoising timestep
    x AR age/state
    x prompt-switch state
    x actual cache operator
```

这比再从某个连续分数中强制切两类更有希望得到可迁移机制。

### 1.3 我们能否像 PF 一样使用人为规则

可以。人为设计不是学术问题，**事后为结果调规则并隐瞒过程**才是问题。

PF 的分类也包含算法设计选择，但其规则对应明确的 Q-K 时间拓扑，并通过对应
cache operator 和消融进行验证。我们可以提出自己的规则，但至少应满足：

1. 规则具有可解释的自然坐标、零点或统计假设；
2. 在查看目标生成结果前冻结；
3. 同时报告附近 threshold、dose 或 quota 的完整敏感性；
4. 使用 count-matched bottom/random/placement 对照；
5. 在 held-out prompt/seed 上验证；
6. 跨模型时直接应用同一规则，不根据第二模型结果重新挑层；
7. 若生成成员效用未通过，只能称 profiling diagnostic，不能命名功能 head。

合法表述示例：

> 我们预先定义模型归一化中心三分之一为 episodic-cache 候选层，并验证其
> dose、placement 和跨模型迁移。

当前不合法的表述：

> 第 10--19 层天然就是长期记忆层。

## 2. 为什么不能继续把 Middle10 当作方法定义

当前 Self-Forcing/Pyramid-Forcing 工程底座有 30 层、每层 12 heads。
`Middle10` 实际是第 10--19 层共 120 heads，不是“只有第 10 层”。

v157 使用同一个 Reservoir4 operator、同样 120 heads 得到：

| Placement | Quality | Dynamic |
|---|---:|---:|
| Early10 | 83.83 | 77.50 |
| Middle10 | 84.24 | 77.50 |
| Late10 | 83.68 | 75.83 |
| Interleaved10 | 84.54 | 79.17 |
| All30 | 83.72 | 83.33 |

这证明 layer placement 会影响结果，但没有证明 10 层是自然类别。v159 换成
motion-pair operator 后 Middle10 又优于 interleaved10，进一步说明 placement
与 cache operator 存在交互，而不是固定的“第十层长期记忆机制”。

因此 v166 的 Middle10 目前只具有以下地位：

```text
在 30 层 SF 开发模型上冻结的、operator-specific 工程配置
```

它不能直接外推到 24、32 或 40 层 DiT。

## 3. v172 的新实验定义

### 3.1 方法定义改为归一化深度

对具有 `L` 层的 DiT，定义层中心坐标：

```text
u_l = (l + 0.5) / L,  l = 0, ..., L-1
```

层预算由 `round_half_up(L * fraction)` 确定。中心带选择与 `u=0.5`
最近的连续层；interleaved 使用等宽深度 bin 的中心。这样 30 层模型上的
Center-1/3 恰好是原来的 10--19 层，但绝对层号不再属于方法定义。

### 3.2 冻结的 dose 与 placement grid

当前 30 层模型对应：

| 方法 | 层数 | 当前绝对层 | 回答的问题 |
|---|---:|---|---|
| Center-1/6 | 5 | 12--16 | 5 层是否已足够 |
| Center-1/4 | 8 | 11--18 | 较小中心带是否进入平台期 |
| Center-1/3 | 10 | 10--19 | 精确复用 v166 reference |
| Center-1/2 | 15 | 7--21 | 增大中心带是否继续受益 |
| Early-1/3 | 10 | 0--9 | 等数量早期位置对照 |
| Late-1/3 | 10 | 20--29 | 等数量晚期位置对照 |
| Interleaved-1/3 | 10 | 1,4,...,28 | 等数量分散位置对照 |
| All-layer | 30 | 0--29 | 最大 operator exposure 上界 |

这直接回答“为什么 10 层而不是 5、8 或 15 层”，但不会假定答案一定是 10。

### 3.3 所有方法只改变 cache placement

被选中的层完全复用 v166：

```text
sink1 + reservoir2 + recalled atomic motion pair2 + recent4
```

未选中的层：

```text
sink1 + recent8
```

所有配置固定：

- 相同 16 条 MovieBench-Qwen diverse prompts；
- 相同 prompt-index seed；
- 相同 checkpoint、30 秒长度与生成参数；
- 相同 MultiScaleMotion archive、admission、age、retrieval 和 fallback；
- 相同最大 9 full-frame-equivalent read budget；
- 相同 exclusive dynamic owner 和 RoPE；
- 不运行 PF baseline；
- 默认不进行人工 review。

Center-1/3 和 SF 直接复用 v166 视频，确保参考结果 byte-level 不重新抽样。
本轮共发布 9×16=144 个视频，其中复用 32 个，新生成 112 个。

## 4. v172 假设与可证伪结果

### H1：cache 收益具有归一化深度 dose curve

比较 Center-1/6、1/4、1/3、1/2。输出每条 prompt 的 paired delta、bootstrap
CI 和 prompt-level Spearman，不只报告最佳点。

可能结果：

- 小带宽已进入平台：优先保留最小稳定 fraction，再跨模型确认；
- 随 fraction 单调增加：说明更像连续 exposure，不支持离散 memory layer；
- 存在中间峰值：说明有 operator-specific Pareto optimum，但仍不是自然类别；
- 无稳定趋势：不再讲精确层数故事，转入在线/因果 layer allocation。

### H2：相同层数下，位置本身有作用

比较 Early/Center/Late/Interleaved-1/3。只有 cache placement 不同。

- 若 Center 在 paired quality/identity/dynamic 上稳定占优，才保留
  “normalized central band” 假设；
- 若 Interleaved 更好，说明覆盖整个深度比中心语义更合理；
- 若四者不可区分，则 layer classifier 没有得到支持；
- 若 All-layer 最好，则结论仍是 operator useful，而不是 layer classifier useful。

### H3：开发集最优不能直接成为通用规则

分析代码只输出完整 Pareto set，不自动选论文主方法。即使 Center-1/3 最好，
也必须在不同深度模型上直接复用 `center 1/3` 才能声称跨模型适配性。

## 5. 服务器运行指令

### 5.1 前置检查

v172 需要完整 v166 结果和 mechanism report：

```bash
export REPO_ROOT=/path/to/training-free
export V172_REUSE_V166_ROOT=${REPO_ROOT}/runs/v166_multiscale_motion_moviebench16/full8

test -s ${V172_REUSE_V166_ROOT}/published_manifest.json
test -s ${V172_REUSE_V166_ROOT}/contracts/experiment.json
test -s ${V172_REUSE_V166_ROOT}/automated_screen/multiscale_motion_trace.json
python ${REPO_ROOT}/scripts/build_v172_relative_depth_maps.py --check
```

若最后一个 v166 mechanism 文件缺失，但 trace 已存在，先在 v166 run root 上运行：

```bash
OUT_ROOT=${V172_REUSE_V166_ROOT} \
  bash ${REPO_ROOT}/scripts/run_v166_multiscale_motion_moviebench16.sh mechanism
```

### 5.2 四节点、32 卡生成

四个节点分别设置 `NODE_RANK=0,1,2,3`：

```bash
export REPO_ROOT=/path/to/training-free
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7
export V172_REUSE_V166_ROOT=${REPO_ROOT}/runs/v166_multiscale_motion_moviebench16/full8

bash ${REPO_ROOT}/scripts/run_v172_relative_depth_moviebench16.sh preflight
bash ${REPO_ROOT}/scripts/run_v172_relative_depth_moviebench16.sh generate
```

节点 0 preflight 会先写冻结 contract；其他节点可随后启动，或四节点同时启动并
等待同一 contract。全部完成后在节点 0：

```bash
export NODE_RANK=0
bash ${REPO_ROOT}/scripts/run_v172_relative_depth_moviebench16.sh audit
```

预期：

```text
methods=9
published videos=144
new videos=112
reused videos=32
failures=0
```

### 5.3 VBench-Long core-9

节点 0 先准备 comparison：

```bash
export NODE_RANK=0
bash ${REPO_ROOT}/scripts/run_v172_vbench_long.sh prepare
```

四节点分别执行 split 和 eval：

```bash
export NUM_NODES=4
export NODE_RANK=0
export GPU_LIST=0,1,2,3,4,5,6,7

bash ${REPO_ROOT}/scripts/run_v172_vbench_long.sh split
bash ${REPO_ROOT}/scripts/run_v172_vbench_long.sh eval
```

完成后在节点 0：

```bash
export NODE_RANK=0
bash ${REPO_ROOT}/scripts/run_v172_vbench_long.sh collect
```

关键输出：

```text
runs/v172_relative_depth_moviebench16/full8/metrics/vbench_core9_summary.json
runs/v172_relative_depth_moviebench16/full8/analysis/v172_depth_metrics.json
runs/v172_relative_depth_moviebench16/full8/analysis/v172_depth_metrics.md
```

JSON 包含：

- official VBench Quality Score；
- mutually exclusive identity/background、temporal、semantic、visual、dynamic；
- 每个候选相对 SF 和 Center-1/3 的 prompt-paired bootstrap；
- dose monotonicity；
- count-matched placement 表；
- exact aggregate Pareto set；
- duplicate custom-prompt ViCLIP 审计。

## 6. 跨模型地图与当前代码边界

map builder 已经支持任意层数和每层 head 数。例如 40 层、16 heads：

```bash
python scripts/build_v172_relative_depth_maps.py \
  --write \
  --num-layers 40 \
  --num-heads 16 \
  --output-dir /tmp/v172_40x16_maps
```

它会使用同一 fraction、rounding、center 和 interleaved 规则，而不是复制
10--19 层。

但必须明确：**当前生成 runner 仍只接入 30×12 的 SF/PF 工程底座。**
builder 支持其他 shape 不等于 cache 已经移植到第二个模型。论文要声称
cross-model，仍需把相同 clean-KV descriptor、motion archive 和 cache owner
接入第二个 AR video DiT，并在不改 fraction 的情况下运行。

建议第二模型阶段只运行：

```text
native baseline
center-1/3 frozen transfer
v172 development Pareto 中最小 fraction（若与 1/3 不同）
interleaved-1/3 placement control
all-layer upper bound
```

不能在第二模型上重新扫 fraction 后只报告最优值。

## 7. v172 之后更强的分类标准

归一化深度解决了“绝对第 10--19 层不可迁移”，但仍是人工 allocation rule，
不是最终的机理分类。若 v172 不能给出稳定 placement，下一步应做
**operator-aligned causal history leverage profiling**：

1. 在同一个 native hidden state 上冻结 query、history 和噪声；
2. 对每层做 equal-budget `recent-only` 与 `episodic-motion` replay；
3. 记录 token-count-normalized old-history addressing、attention output delta、
   downstream X0 leverage 和跨 timestep 符号；
4. 先在 layer 内对 head score 去除 layer mean，再分解 layer 与 head-within-layer；
5. 使用 prompt/seed/timestep block bootstrap 和 permutation null；
6. 以 `CI lower bound > 0` 加 FDR 控制定义候选，不固定每层 top-k；
7. 在完整生成中用 bottom/random/count-matched 反证 membership；
8. 在第二模型上重新运行同一 profiler，而不是迁移绝对 head id。

这里的自然零点是“episodic history 相对等预算 recent history 没有正向
causal leverage”，比按中位数强制二分更有机理含义。即使如此，在通过完整
trajectory transfer 前也只能称 history-leverage candidate，不能直接命名
identity/motion/memory heads。

## 8. 当前论文故事允许写到哪里

在 v172 和跨模型结果出来前，最稳妥的技术主线仍是：

1. training-free clean-KV multi-scale motion signature；
2. atomic motion episode admission and recall；
3. fixed-budget sink/reservoir/episode/recent heterogeneous memory；
4. operator-aware temporal allocation，而不是 PF Anchor/Wave/Veil 路由；
5. 完整可审计的 cache ownership、budget、selection 和 fallback。

v172 若通过，只能增加：

> normalized-depth episodic allocation 在同一模型上显示稳定的 dose/placement
> 规律。

只有第二模型无调参迁移通过后，才能进一步写：

> 该 allocation rule 不依赖某个模型的绝对层号，并具有跨架构深度适配性。

如果 v172 的结果支持 All-layer 或不同 placement 同样好，则不强讲 layer
classification；应诚实转向 online state-conditioned allocation 或
operator-aligned causal profiler。这样不会损失 v166 cache/motion episode 的技术点，
也避免为了故事把未确认的层类别写成事实。
