# v183 结果审计与 v184 去噪阶段 Coverage 实验

## 1. 当前仓库与证据状态

本轮同步到远端提交 `e801f3ee`。v183 已完成 v180 的四种方法、128 prompts、
VBench-Long core-9 共 36 个评测任务。视频生成与评测文件完整，但旧 v178
membership gate 是无效占位结果，因此以下统计只能用于方法开发，不能证明 RCCP
head membership。

| 方法 | Quality | Identity/background | Temporal | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| SF native | 81.9118 | 0.97028 | 0.98243 | 0.23434 | 0.64648 | 0.40677 |
| strict-five RCCP | 82.2901 | 0.97017 | 0.97754 | 0.23467 | 0.65027 | 0.49844 |
| all-Recent | 82.4544 | **0.97094** | 0.97905 | **0.23483** | **0.65148** | 0.49271 |
| all-Coverage | **83.1119** | 0.96925 | 0.97783 | 0.23350 | 0.65087 | **0.60260** |

关键 paired 结果：

- strict-five 相对 all-Recent：Quality `-0.1643`，Identity `-0.00077`，Dynamic
  `+0.00573`。五个静态 Coverage heads 没有产生可检测的生成收益，应停止这条
  static strict-five 路线。
- all-Coverage 相对 all-Recent：Quality `+0.6576`，Dynamic `+0.1099`，但
  Identity `-0.00169`。Coverage 是明显的运动干预，但不是全面占优的方法。
- all-Recent 相对 SF：Quality `+0.5426`、Dynamic `+0.0859`，Identity 的置信区间
  跨零。all-Recent 是下一轮更合理的内部控制组。

`overall_consistency` 与 `temporal_style` 完全相同是当前 VBench wrapper 的已知
实现结果；官方 Quality 不重复使用二者，因此这不是本轮新 bug。

## 2. 逐 prompt 异质性

all-Coverage 相对 all-Recent 的 128-prompt 结果为：

- Dynamic：59 胜、57 平、12 负；
- Identity：44 胜、84 负；
- Quality：76 胜、52 负；
- Dynamic delta 与 Identity delta 的相关系数仅 `-0.088`；
- Dynamic delta 与 Quality delta 的相关系数为 `0.837`，说明官方 Quality 的提升
  很大程度来自 Dynamic Degree 权重，不能解释为全面视觉改善。

更重要的是，all-Recent 的基线 Dynamic 与 Coverage 的 Dynamic 增益相关系数为
`-0.350`：

| all-Recent 状态 | prompts | Coverage dDynamic | Coverage dIdentity |
|---|---:|---:|---:|
| Dynamic `<= 0.25` | 50 | `+0.1667` | `-0.00119` |
| Dynamic `>= 0.75` | 49 | `-0.0068` | `-0.00151` |

这说明 Coverage 对已经有充足运动的样本基本没有价值，却仍可能损伤身份。下一步
不应继续固定全程读取 Coverage，而应研究干预发生在去噪过程的哪个阶段，并最终
发展为按状态触发的读取策略。

## 3. v184 假设

四步 Self-Forcing 每个 AR block 运行四次 noisy denoising forward，再运行一次
clean forward 写入规范 KV。v184 检验以下可证伪假设：

> 长期 Coverage 在高噪声阶段更可能影响全局位移和运动规划；低噪声阶段继续读取
> 长期历史可能干扰身份、纹理和局部时序细化。只在早期 noisy calls 读取
> Coverage，可能保留运动增益并减小身份损失。

这目前是实验假设，不是已成立的 diffusion 机理结论。`early2` 与等剂量的
`late2` 是关键因果对照。

## 4. 共享双缓存状态

所有 v184 方法执行完全相同的缓存更新规则：

```text
Recent read   = sink1 + recent8                       = 9 FFE
Coverage read = sink1 + reservoir4 + recent4          = 9 FFE
Clean read    = Recent for every method
```

每次 clean update 同时维护 Recent、Coverage 和未读取的 Episode shadow bank；
noisy calls 只改变读取视图，不写长期 memory。Coverage 使用
`TemporalReservoirStrategy(capacity=4, seed=2026)`，目的是先把它作为已经由 v183
证实的中性运动干预，隔离 timestep 效应。本轮不把随机 Reservoir 声称为最终缓存
创新；若阶段路由成立，下一轮再在获胜 schedule 下比较 landmark、prototype、
retrieval 等确定性 operator。

clean pass 固定 Recent 很重要：它避免不同方法在 canonical KV commit 时直接使用
不同读取算子，使 v184 更接近“仅改变 noisy trajectory 的调用阶段”实验。不同方法
生成的 latent 仍会导致 KV 内容不同，这是端到端干预必然产生的结果。

运行时对每个 head 强制检查：

- 总读取不超过 9 FFE；
- Recent 不允许出现 middle anchor，dynamic 不超过 8 FFE；
- Coverage 的 reservoir 不超过 4 FFE，dynamic 不超过 4 FFE；
- clean call 的 effective policy 必须为 Recent；
- trace 必须覆盖 layers 0/10/20/29 和 noisy calls 0/1/2/3。

## 5. 五个方法

| 方法 | noisy call 0 | call 1 | call 2 | call 3 | clean |
|---|---|---|---|---|---|
| `all_recent` | R | R | R | R | R |
| `coverage_early1` | C | R | R | R | R |
| `coverage_early2` | C | C | R | R | R |
| `coverage_late2` | R | R | C | C | R |
| `all_coverage_noisy` | C | C | C | C | R |

其中 R/C 分别表示等预算 Recent/Coverage readout。`early1/early2` 给出早期剂量，
`early2/late2` 在相同两次 Coverage 读取下只改变阶段，`all_coverage_noisy` 是干预
上界，不是默认最终方法。

## 6. Prompt 与规模

- source：用户提供的 128 条 Qwen-rewritten MovieGen prompts；
- development subset：预先固定 `source_index = 2 + 4k, k=0..31`；
- 32 prompts，120 latent frames，约 30 秒，seed 0；
- 5 methods，共 160 个视频；
- 不运行 PF、不运行 ABA，也不重复旧 strict-five 消融。

系统采样覆盖完整 source index 范围，规则在看到 v184 视频和指标前冻结。它是开发
筛选，不是论文最终测试集。

## 7. 服务器执行命令

先在 node 0 准备并运行五方法单 prompt smoke：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v184_denoise_phase_screen_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v184_denoise_phase_screen_32gpu.sh preflight
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v184_denoise_phase_screen_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v184_denoise_phase_screen_32gpu.sh audit-smoke
```

`audit-smoke` 通过即可开始 screen32，不需要先细排五个视频。若自动媒体解码、
schedule trace 或 9-FFE contract 失败，停止并上传 log/trace；不要继续生成。

四节点分别设置 `NODE_RANK=0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v184_denoise_phase_screen_32gpu.sh generate32
```

完成后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v184_denoise_phase_screen_32gpu.sh status
NODE_RANK=0 bash scripts/run_v184_denoise_phase_screen_32gpu.sh audit-screen
NODE_RANK=0 bash scripts/run_v184_vbench_long.sh prepare
```

四节点准备 splits 并运行 5 x 9 个 core-9 jobs：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v184_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v184_vbench_long.sh eval
```

最后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v184_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v184_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v184_vbench_long.sh decision
```

## 8. 自动决策与人工 review

主要指标为 Quality、Identity/background、Dynamic Degree 和 Temporal mechanics。
阶段候选只有同时满足以下开发 gate 才能晋级：

- 位于四指标 Pareto front；
- 相对 all-Recent，Quality 不下降；
- Identity delta 不低于 `-0.001`；
- Dynamic delta 至少 `+0.02`；
- Temporal delta 不低于 `-0.002`。

这些数值是开发筛选容忍区，不是论文显著性或 non-inferiority margin。分析器同时
输出 bootstrap CI、sign test、BH q-value、early2-vs-late2 和 early2-vs-early1。
自动结论不依赖人工 review；最多给出四组 identity-motion 冲突样本，只有候选晋级
后才需要针对性查看。

## 9. 不同结果下的下一步

### early1/early2 晋级且优于 late2

支持“早期去噪阶段是更合适的长期历史注入点”。下一轮在获胜 schedule 下替换
Reservoir 为 deterministic Coverage operator，再把最终候选与 all-Recent、SF 放到
未参与筛选的 fresh 128 prompts 上确认。论文故事可以围绕 phase-conditioned
long-history exposure，而不是静态 head taxonomy。

### late2 更优

否定当前高噪声运动规划假设。保留 timestep routing 事实，但必须重新解释为后期
细化需要历史参照，并检查 Identity/Temporal 是否真的非退化，不能反向改写预设
假设。

### 稀疏阶段均失败，但 all-Coverage 仍增加运动

说明 Coverage 是有效 actuator，但固定 call schedule 无法解决身份代价。下一步应
使用当前 clean-KV/latent motion proxy 构建在线 motion-deficit gate，只在运动不足的
AR state 激活 Coverage。v171 的旧 deficit gate 作用在另一种 motion-pair selector 上
且已失败，不能直接当作该新 gate 的正证据。

### all-Coverage 也不再增加运动

v183 的效果没有在新 prompt/runtime contract 下复现，停止 timestep 路线，先检查
双缓存实现与 generation/evaluation provenance，不进行阈值扫描。

## 10. 当前论文边界

目前可诚实写出的结论是：全头长期 Coverage 能调节运动，但存在身份代价；静态
strict-five RCCP membership 没有转化为生成收益。v184 代码是寻找新方法核心的因果
实验，不是已经确认的论文方法。ABA、跨模型、最终 128 benchmark 和确定性缓存
operator 都应等待 v184 的自动指标结论。
