# v209：SF 协议、预算归因与论文收束

日期：2026-09-15。代码起点：`22df9503`。
分支：`codex/v178-v179-causal-validation`。

## 1. 最新进展的含义

本轮拉取到的更新主要是数值修复和多节点脚本，尚未包含新的 v207 最终 parity/生成指标。
不能把提交了修复等同于服务器验证通过。

最新修复包括：

- PF pipeline 的 timestep buffer 改为保留 warped timestep 的 dtype，避免把 937.5 截成 937。
- Adaptive 等长序列可以重新组织为 dense attention，与 SF 使用相同入口。
- 增加 float64 reference RoPE 和 math attention 诊断路径。
- parity 默认同一张 GPU 串行运行。

这些事实说明此前的 native-SF 差值混入了实现差异。旧 v201 仍是旧系统的真实负结果，但不能由此
单独判定 head/phase/horizon selector 必然无效；同样也不能由离线 proxy 为正推断视频必然改善。

用户提供的外部 `226_sf_baseline_related_work_and_failure_audit.md` 作为审计线索使用。
本文的执行顺序和代码根据当前仓库重新确定，不沿用外部文档中已与仓库冲突的编号。

## 2. 本轮确定的问题

### 2.1 SF baseline 的名称需要明确

本项目 SF config 显式设置 local window=21，没有设置 sink；wrapper 默认 sink=0。
因此此前的 `sf_native` 实际是 `FIFO21`。`sink1+recent20` 是另一种同预算协议。
官方 SF 在所审计 commit 的配置没有显式设 local window，不能仅凭配置名推断长视频协议：
[官方配置，commit 33593df3](https://raw.githubusercontent.com/guandeh17/Self-Forcing/33593df3e81fa3ec10239271dd2c100facac6de1/configs/self_forcing_dmd.yaml)。

后续保留两个 21-FFE SF 对照。不能因为某个 baseline 较弱就只保留它。

### 2.2 9-frame parity 没覆盖满窗

原 v207 只测试 9 latent frames，尚未触发 21-frame eviction。FIFO21 与 sink1+recent20 在此时
可以有完全相同的成员，因此旧短测无法验证两者在长程上的关系。

### 2.3 Math backend 不等于正式生成 backend

当前 v207 parity 打开 reference math attention，正式生成没有打开。即使 math 通过，也只支持
那个 backend 下的数值结论。v209 显式分开 `production` 和 `reference`；只有 production
canary 能允许本轮正式生成。

### 2.4 修正 dense trace 的帧号

旧 dense trace 将窗口一律记为连续 recent ids，在 sink>0 且发生 eviction 后会错误描述首帧。
本轮加入只用于 trace 的 frame-id sidecar，跟踪每次写入、驱逐、noisy overwrite 和 clean overwrite。
不修改实际 K/V 或 attention。测试覆盖 21/13/9 的满窗与重复写入，包含不能被 block size=3 整除的 13。

## 3. 已实现的两组实验

### A. 两个 prompt 的数值 canary

从 v207 systematic32 中取 local indices 0、16，对应 MovieGen source indices 1、65。
每次生成 30 latent frames，即 10 个 AR blocks，足以覆盖首次满窗以及后续三次 21-frame eviction。
这是约 7.3 秒的数值测试，不做 VBench，不要求人工观看；正式视觉实验仍为 30 秒。

默认 production backend，同一节点、同一张 GPU 串行：

| 组 | Run | 目的 |
|---|---|---|
| Native | FIFO21 A/B | 原生重复性底线 |
| Native | Sink1+Recent20 A/B | 保留 sink 后的重复性 |
| Native | Sink1+Recent12 | 13 FFE 成员和边界 |
| Native | Sink1+Recent8 | 9 FFE 成员和边界 |
| Native | Sink3+Recent6 | 同预算 sink 大小变化 |
| Runtime | PF plain FIFO21 | 与同协议 SF 比较 |
| Runtime | PF plain Sink1+Recent20 | 与同协议 SF 比较 |
| Runtime | Adaptive Recent21 retrieval | 与原生 Sink1+Recent20 比较 |
| Operator | Adaptive Recent9 landmark A/B | all-recent 自重复 |
| Operator | Adaptive Recent9 retrieval | 未读取的 middle 算子是否影响输出 |

Native 部分为 7 runs × 2 prompts；其余为 6 runs × 2 prompts。可先完成 Native 并提交报告，
再完成 Runtime/Operator。RNG、pipeline、attention/cache trace 都沿用统一接口。
默认缓存只存固定样本，完整 pipeline tensors 用于逐步比较，不默认保存 layer0 的完整大 K/V。

分析器按真实执行顺序定位首处分叉，流式加载两个 event 的 tensor，避免一次读入全部 trace。
不使用 MP4 文件 SHA 作为数值 parity：编码差异不能证明 latent 不同。

自动判定分别输出：

```text
native_budget_screen_ready
pf_adaptive_runtime_ready
unread_operator_isolation_pass
```

PF/Adaptive 失败不会阻止已经通过自身检查的原生 SF 预算实验。
Reference math 的结果只能用于定位，始终不能使 production readiness 变为 true。

当前未实现“关闭 middle update”的第四个 operator canary。如果 landmark/retrieval 确实出现
超出自重复底线的分叉，再按首个事件定位到更新或读取，补最小干预；不预先修改 active cache。

### B. Native SF 的 32-prompt 预算阶梯

所有方法都运行在同一个 SF runtime、同 checkpoint、同 prompt/seed、同 production backend：

| Method key | 组成 | 总预算 |
|---|---|---:|
| `sf_fifo21` | 最近 21 帧 | 21 |
| `sf_sink1_21` | 首帧 + 最近 20 帧 | 21 |
| `sf_sink1_13` | 首帧 + 最近 12 帧 | 13 |
| `sf_sink1_9` | 首帧 + 最近 8 帧 | 9 |
| `sf_sink3_9` | 最初 3 帧 + 最近 6 帧 | 9 |

使用 Qwen MovieGen systematic32，source indices=1+4k，seed=20700，120 latent frames、477 decoded
frames、16 FPS，5×32=160 个约 30 秒视频。所有方法 compile_ffn=false、batch VAE、固定原生 SF
few-step 路径。没有 middle archive、head selector 或 dynamic-RoPE 重映射。

这能单独回答 sink 与局部上下文压缩的作用。它不能把旧 v201 的多因素差异拆成可直接相加的数字。
同一节点轮换方法执行顺序，所有节点均运行所有方法，不把方法固定绑定到节点。

## 4. 服务器立即执行

请勿在已有推理进程中途更新它正在使用的 checkout。新实验使用本次提交的新 checkout 或等待旧进程退出。
旧 v207 trace 保留原来源，修改 trace 代码后不要把旧目录当作新 trace 续跑。

```bash
git pull --ff-only origin codex/v178-v179-causal-validation
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
python -m pytest -q tests/test_v209_sf_protocol.py tests/test_v207_sf_parity.py

NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh prepare
NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary-native
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary
cat runs/v209_sf_protocol_budget/canary_production/analysis/decision.md
```

Native readiness=true 后，四个节点分别执行，NODE_RANK 改为各自的 0/1/2/3：

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v209_sf_protocol.sh generate32
```

Runtime/Operator canary 仍在刚才的节点和 GPU0 上顺序执行，不能把它们与另一张 GPU 的 native
轨迹配对。可在原生预算实验结束后执行，或预先为其保留该 GPU：

```bash
NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary-runtime
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary
```

也可以一次用 `canary` 完成全部 13 个 runs。完成的 jobs 自动跳过，失败 job 单独重做。
Reference 定位需要重新运行一整组同 backend 轨迹，输出在另一个 scope：

```bash
V209_BACKEND=reference NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary
V209_BACKEND=reference NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary
```

默认 checkpoint 和 Qwen prompt 路径与 v207 一致。可用 `SHARED_CHECKPOINT`、
`V209_SOURCE_PROMPTS`、`V209_OUT_ROOT`、`CONDA_SH`、`CONDA_ENV` 覆盖；完整配置写入 `inputs/configs/`。
首次 prepare 计算 checkpoint SHA；续跑通过路径、大小、mtime 检查复用 SHA，避免每张卡重复扫描大文件。

## 5. 评测与需要返回的小文件

所有节点生成完成后，节点0发布；再四节点 split，全部结束后四节点 eval：

```bash
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh vbench-prepare

# Below: set NODE_RANK independently on each node.
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v209_sf_protocol.sh vbench-split
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v209_sf_protocol.sh vbench-eval

# After all evaluation jobs finish, node 0 only:
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh vbench-collect
bash scripts/run_v209_sf_protocol.sh package
```

`vbench-resume-missing` 只计算缺失指标；`vbench-collect` 自动补时序失败诊断和 paired 分析。
VBench root 可通过 `VBENCH_ROOT` 覆盖，模型缓存使用已有 `VBENCH_CACHE_DIR`。

主要返回：

```text
inputs/manifest.json
canary_production/analysis/decision.json
canary_production/analysis/decision.md
canary_production/jobs/*/*/metrics.jsonl
screen32/analysis/v209_budget.json
screen32/analysis/v209_budget.md
screen32/analysis/v209_paired.csv
screen32/audits/efficiency.json
screen32/metrics/vbench_core9_summary.json
screen32/metrics/temporal_diagnostics.csv
```

`package` 输出 `v209_small_artifacts.tar.gz`，不包含 MP4 或 tensor `.pt`。
完整 trace 留在服务器；仅在首处分叉需要深入检查时提供对应的一两个 tensor。

每个 prompt 额外记录 pipeline 用时、peak allocated/reserved、GPU 型号/UUID、torch/CUDA、实际
sink/window、实际 warped timestep 列表和命令参数。计时含 text encoding、DiT、cache update、VAE，
不含模型加载和 MP4 编码。只有 screen32 的无 trace 测量适合效率表，canary 时间不参与比较。

paired analysis 用 prompt 作统计单位，报告 full/early/late、95% bootstrap CI、win rate、BH q-value。
DD 单列；不把饱和的 DD 作为提升。core-9 不是完整官方 Semantic/Total Score，不补造缺失维度。
本轮自动输出预算非劣与失败定位，不要求全量人工 review。

## 6. 论文目标与下一步方法

现有证据允许先写问题定义、相关工作、评测协议及离线发现，但暂不足以写“新方法已经优于 SF”。
不要求 SOTA，也不要求每一项指标都提高。目标应是相对可靠 SF 对照，有可重复的主体/画质收益，
局部时序没有实质退化，并透明报告效率代价和其他指标的 trade-off。

参考相关工作的实验覆盖面，而非直接复用其聚合分数：
[Pyramid Forcing](https://github.com/if-lab-pku/Pyramid-Forcing)、
[Deep Forcing](https://github.com/cvlab-kaist/DeepForcing)、
[Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing)。
以 128 prompts、30/60 秒、匹配设置和效率表为后续确认框架；PF 不强制在本轮重跑。
外部审计的仓库可运行性和数值表属于特定版本记录，不能直接当成当前仓库状态或本项目实验结果。

最精简的后续候选是保留 SF 完整局部上下文的历史修正：

```text
y_local = Attention(Q, local21)
y_aug   = Attention(Q, local21 + selected_history)
delta   = y_aug - y_local
y_out   = y_local + alpha * bounded(delta)
```

这个候选还没有在本轮实现或证明有效。它比直接加 normalized(y_history) 更容易明确干预含义：
修正量来自历史加入前后的变化，alpha=0 保留原分支。`bounded` 约束相对 local 输出的 RMS 幅度，
只保证张量干预受限，不保证感知效果单调改善。clean pass 首先保持原生读法。

下一轮最小 screen 应包含：两个 SF21 对照、gate0、正确 history、等预算 random history，
alpha 只在开发集取少量预设值。先固定 early noisy calls；只有 paired 结果支持后，才补
phase-shift/head/horizon 消融。否则继续声称三维 selector 是核心贡献会超过目前证据。

论文方法故事最终取决于实验：

1. 若 21→9 有明显损失，动机是保留局部连续性，再以受限历史修正提供长期信息。
2. 若预算损失很小而 Adaptive 不等价，先解决运行路径或 RoPE 差异。
3. 若历史修正稳定胜过两个 SF 对照，冻结方法，补一个机制对照与真正未参与历史调参的确认集。
4. 若仅有离线结构而无视频收益，不能把结构发现当成方法改善的证据。

当前优先完成这 160 条归因视频和两条 prompt 的数值诊断；暂不扩大到未经归因的 v208 768 视频。
