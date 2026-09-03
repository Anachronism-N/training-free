# v183 指标修正与以 SF 为主基线的 v201 实验

> 日期：2026-09-03
>
> 状态：代码完成，等待 v189/v200 结果门控
>
> 目标：不要求超过 PF；只要相对 canonical Self-Forcing 有可复现的显著提升，即可进入论文确认实验

## 1. 最新同步后的结论

最新 `main@0e039256` 重新使用正确的 torchvision RAFT 评测了 v183 的
Dynamic Degree。四种方法的 128 prompts、每条 15 clips，共 7,680 个 clip 全部得到
`Dynamic Degree=1.0`。旧 RAFT 只加载了 18/169 个权重，因此旧的约
`0.4-0.6` Dynamic Degree 及由它计算的旧 Quality 均不可继续使用。

将每条 prompt 的 Dynamic Degree 替换为正确值后，聚合结果为：

| 方法 | 旧 Quality | 修正后 Quality | 相对 SF |
|---|---:|---:|---:|
| `sf_native` | 81.9118 | **86.4751** | 0 |
| `all_recent` | 82.4544 | 86.3566 | -0.1184 |
| `rccp_matched` | 82.2901 | 86.1482 | -0.3268 |
| `all_coverage` | 83.1119 | 86.1688 | -0.3062 |

因此当前可验证结论是：

1. v183 没有方法在修正后超过 SF；
2. 旧“all-Coverage 明显提升运动和 Quality”的证据失效；
3. Dynamic Degree 在当前 30 秒 MovieBench 上饱和，不能排序方法，也不能支持运动论断；
4. v183 不否定后续 Head x Phase x AR-Horizon 假设，因为它测试的是旧静态 map 和不同路由。

修正分析不会覆盖任何旧结果：

```bash
bash scripts/run_v202_v183_metric_correction.sh analyze
bash scripts/run_v202_v183_metric_correction.sh show
```

输出为：

```text
runs/v180_rccp_fresh128/recovery_v183/analysis/
  v202_v183_corrected_evidence.json
  v202_v183_corrected_evidence.md
```

## 2. 当前方法假设

当前最值得验证的方法不再是一个永久的 head 二分类，而是条件化路由：

```text
R(AR horizon, denoising call, layer, head) in {Recent, Coverage}
```

两个缓存读视图具有相同的 9 FFE 预算：

```text
Recent   = sink1 + recent8
Coverage = sink1 + structured-middle4 + recent4
```

`structured-middle4` 由 v189 分别为 Landmark 或 Retrieval 构造。Clean pass 始终走
Recent；只有 noisy denoising calls 按冻结 map 在两种读视图间切换。也就是说，方法不靠
增加读取 token 获益，核心假设是：长期历史何时、由哪些 layer/head 读取，会随 AR
外推长度和 denoising phase 改变。

证据链分三层：

1. **v189 profiling**：在 active trajectory 始终使用 Recent 的条件下，用
   representation-complete Union shadow teacher 测量 Coverage 相对 Recent 的 residual
   优势；
2. **v200 cross-fit**：用 64 discovery prompts 选择路由，在 32 validation prompts
   检验 horizon-conditioned selector 是否优于相同曝光量的 static selector；最后 32 条
   generation holdout 不参与选择；
3. **v201 generation**：在 holdout 上直接检验候选是否优于原生 SF，并把 static 和
   horizon-shift 对照作为独立机制归因。

只有 v200 输出
`advance_head_phase_horizon_to_runtime_design` 的 operator 才进入 v201。当前 GitHub
尚未包含 v189/v200 最终结果，所以现在不能声称分类机制已经成立。

## 3. v201 rev2 的方法矩阵

v201 rev2 新增一次共享的 canonical SF，并对每个通过 v200 的 operator 运行五个缓存方法：

| 方法 | 目的 |
|---|---|
| `sf_native` | 论文主基线，使用原始 Self-Forcing repo/config |
| `<op>_all_recent` | 等预算局部缓存对照 |
| `<op>_all_coverage` | 全单元长期历史端点 |
| `<op>_static_top10` | 固定 Head x Phase、等曝光量对照 |
| `<op>_horizon_top10` | 主候选 |
| `<op>_horizon_shift_top10` | 相同 mask 与曝光量，但 AR 对齐错误 |

一个 operator 共 192 个视频，两个 operator 共 352 个视频。所有视频使用冻结的 32 条
holdout prompts、seed `20100`、约 30 秒；不包含 PF，也暂不包含 ABA。

SF 分支在独立的 Self-Forcing 目录运行。脚本会清空所有实验环境变量，审计会拒绝 SF
日志中出现 cache runtime marker 或 schedule trace，避免把“SF baseline”意外跑成实验方法。

## 4. 晋级标准

### 4.1 论文主效应

主问题是 `<op>_horizon_top10` 是否优于 `sf_native`。晋级只要求：

1. 至少一个非 Dynamic 的主指标有正向 paired 支持；
2. 其余质量、身份、时序、语义和视觉轴在 full 与 late-half 上通过开发期
   non-inferiority；
3. 自动 temporal safety 不发现明显冻结、跳变或闪烁。

显著性使用 prompt-paired bootstrap CI 和同一 family 的 BH 修正。若相对 SF 的 95% CI
下界大于 0 且 `q<=0.10`，可进入 fresh-128 论文确认；机制对照未显著时，只能写“方法有效，
具体 horizon 归因尚未确认”，但不会抹掉真实的 SF 主效应。

### 4.2 防止 Dynamic Degree 再次污染

v201 仍报告官方 Quality，但主效应使用
`quality_without_dynamic_degree`：沿用官方归一化和权重，只把所有方法的 Dynamic
Degree 固定为 1.0。这样 Dynamic Degree 即使饱和或 RAFT provenance 异常，也不能通过
Quality 间接影响晋级。

### 4.3 机制归因

只有同时优于 `static_top10` 和 `horizon_shift_top10`，才支持
“Head x Phase x AR-Horizon 条件化选择”这一机制。`all_recent` 和 `all_coverage` 用于解释
operator 与曝光量，不再作为论文主效应的硬阻断条件。

## 5. 自动连续运动诊断

v203 在相同视频上执行 camera-compensated dense flow：先拟合并移除全局 affine camera
motion，再测主体局部 residual motion、活动区域、后半段 motion ratio、最长低运动区间和
加速度异常。它能区分“相机动但主体冻结”和真实局部运动。

该诊断不使用 GPU 生成，也不把自己包装成方法贡献。自动分析通过后最多给出 4 条定点
review；失败时无需人工盲审大量视频。对 v201 的质量配对优先使用
`quality_without_dynamic_degree`，不会退回被 Dynamic Degree 污染的 Quality。

## 6. 服务器运行顺序

### 6.1 前置 profiling 与零 GPU 门控

若 v189 尚未完成，先按 `docs/208_v189_structured_head_phase_profiling.md` 运行并上传小
结果。然后在任一 CPU 节点执行：

```bash
bash scripts/run_v200_head_phase_horizon_audit.sh preflight
bash scripts/run_v200_head_phase_horizon_audit.sh analyze
bash scripts/run_v200_head_phase_horizon_audit.sh show
```

只有 `show` 输出允许 runtime design 时继续。

### 6.2 准备和 smoke

```bash
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh preflight
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0 \
  bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh audit-smoke
```

rev2 默认写入全新目录
`runs/v201_head_phase_horizon_sf_screen`，不会与旧 v201 manifest 或视频混合。

### 6.3 四节点 32 卡生成

在四个节点分别执行，`NODE_RANK=0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0..3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh generate32
```

可用 `METHODS=method1,method2` 分批生成，已经完成的视频会被严格续跑。分批期间不要执行
最终 audit；全部 manifest 方法完成后再执行：

```bash
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh status
NODE_RANK=0 bash scripts/run_v201_head_phase_horizon_screen_32gpu.sh audit-screen
```

### 6.4 VBench 与 paired decision

```bash
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh prepare

# 四节点分别执行
NUM_NODES=4 NODE_RANK=<0..3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v201_vbench_long.sh split
NUM_NODES=4 NODE_RANK=<0..3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v201_vbench_long.sh eval

# node 0
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh temporal
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v201_vbench_long.sh decision
```

### 6.5 连续运动诊断

四节点可与其他 GPU 任务并行执行 CPU motion extraction：

```bash
NUM_NODES=4 NODE_RANK=<0..3> V193_WORKERS=8 \
  bash scripts/run_v201_vbench_long.sh motion-compute
```

全部完成后在 node 0：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v201_vbench_long.sh motion-status
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v201_vbench_long.sh motion-collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v201_vbench_long.sh motion-analyze
```

## 7. 下一步决策

1. v200 不通过：不生成 v201，不再人为寻找“更好看”的分类阈值；保留 operator-level
   cache 研究。
2. v201 对 SF 无正向信号：停止该 selector，不扩大 128 prompts。
3. v201 对 SF 有方向性但 CI 跨 0：补一个 seed 的同一 32-prompt 配对实验。
4. v201 对 SF 显著提升：立即在未使用的新 128 prompts 上做确认；不要求超过 PF。
5. SF 主效应成立但 static/shift 归因不成立：方法效果可保留，论文中弱化 horizon 机理，
   继续用更直接的 operator/phase 解释。
6. SF 主效应和机制归因同时成立：形成最完整故事，即预算固定的结构化长期记忆，通过
   Head x Denoising Phase x AR Horizon 条件路由改善 training-free 长视频外推。

当前最重要的缺口不是再添加 cache trick，而是取得未泄漏 holdout 上相对 SF 的配对正
结果。只要该结果显著且自动安全诊断没有退化，就足以开始论文写作；PF 只需作为相关工作
和可选上下文，不是当前方法成立的必要条件。
