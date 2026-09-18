# v214：六节点48卡，补齐剩余96条prompt

## 1. 为什么新增这一轮

之前并不是所有实验只使用单节点：baseline/gate0/smoke在rank0，
v213主实验本来就分布到六节点。但v213只有32个prompt bundle，
每个bundle内部七方法串行，所以最多占32张卡，无法充分利用48张卡。

本轮直接补齐MovieGen-128里v213未覆盖的96条：

- v213：零基source为`3,7,11,...,127`，共32条，保留已有结果。
- v214：零基source满足`source % 4 != 3`，共96条，不包含任何v213 source。
- 相同七方法、相同seed规则`21300+source`、相同30秒生成协议。
- 每节点16条prompt，每卡2条prompt，每条prompt的七方法在同一物理卡依次运行。
- 新增`96×7=672`条视频；两轮完整后共`128×7=896`条不同主视频。

这比为了占卡临时增加大量不明确的小技巧更直接：
需要判断当前主候选是否能在更广的prompt范围内稳定改善SF，
以及强度/phase变化是否具有一致收益，而非依赖32条开发样本。

**这仍是开发矩阵扩展，不是独立冻结最终方法后的确认实验。**
96条有历史研究使用记录，不宣称是从未看过的测试集。
如果根据这128条选出新赢家，后续仍需补相应机制对照与独立seed。

## 2. 方法矩阵与待回答的问题

全部使用120 latent frames，477输出帧，16FPS，约29.8秒，480×832。
local预算包括当前块；不把21帧说成21历史帧加3当前帧。

| 方法 | Local | 历史选择 | alpha | 去噪phase | 问题 |
|---|---|---|---:|---|---|
| sf_fifo21 | FIFO21 | 无 | 0 | 原生 | 是否改善较强的SF对照 |
| sf_fifo25 | FIFO25 | 无 | 0 | 原生 | 单纯增加局部容量是否已经足够 |
| fifo_correct | FIFO21 | 内容选择4 | .02 | 0 | 预先指定的主候选能否推广 |
| fifo_random | FIFO21 | 随机4 | .02 | 0 | 内容选择是否有额外价值 |
| fifo_e1_a010 | FIFO21 | 内容选择4 | .10 | 0 | 提高修正强度是否更好 |
| fifo_e2_a002 | FIFO21 | 内容选择4 | .02 | 0、1 | 延长早期介入是否更好 |
| fifo_full_a002 | FIFO21 | 内容选择4 | .02 | 0、1、2、3 | 持续介入是否损害局部生成/运动 |

LPHC保存12帧archive，选择最多4帧参与受限历史修正，不替换原生FIFO21。
模型、attention算子和修正公式不因扩大prompt数量而改变。
不新增PF生成，不做ABA，不新增head分类假设。
SF25并非严格显存/FLOPs匹配；phase比较同时改变累计干预量和计算量。
如果新赢家是`.10/e2/full`，需要补其匹配的随机历史对照，不能借用`.02/e1`机制证据。

## 3. 六节点任务分配

| NODE_RANK | V214_NODE_ADDRESS | 本节点source模式 | 主视频数 |
|---:|---|---|---:|
| 0 | 28.216.19.213 | `0+8k, k=0..15` | 112 |
| 1 | 28.216.19.143 | `1+8k, k=0..15` | 112 |
| 2 | 28.216.19.137 | `2+8k, k=0..15` | 112 |
| 3 | 28.216.19.225 | `4+8k, k=0..15` | 112 |
| 4 | 28.216.18.144 | `5+8k, k=0..15` | 112 |
| 5 | 28.216.18.136 | `6+8k, k=0..15` | 112 |

每节点GPU0..7全部使用。GPU g分别执行该节点第g和第g+8条prompt，
每卡14条主视频。七方法顺序按prompt轮换，不把某一种方法固定到某个节点。
这是48个独立推理进程，不是让单条视频跨48卡做DDP。
尾部会自然出现空闲卡，不承诺从头到尾100%利用率。

在prepare时写入`inputs/placement.json`；`schedule`可直接显示48个槽位及其source。
恢复运行保持同一prompt/seed/物理GPU；完整job校验后跳过。
缺少或损坏的job保留原日志并隔离重跑，不覆盖旧结果。

### 与已运行实验的关系

- v213已经开始：继续使用原冻结checkout与输出目录，先完成占卡的生成，再启动v214。
- v213已经生成完：直接启动v214，不重复那32条；两轮评测不要与新生成争用同一批GPU。
- v213尚未开始：可以先启动v214充分使用48卡，之后补v213的32条；不能把v214单独称为完整128条结果。
- v212仍在占卡：同样先确认释放，不能直接重叠启动48个新进程。

不同实验root的文件锁不是全局GPU预约系统，脚本不会自动SSH登录或停止已有任务。
如果实际六节点IP与表格不同，先更新授权配置并冻结新commit，不要伪造地址绕过校验。

## 4. 准备代码与共享路径

rank0在现有仓库中fetch，然后固定一个commit，六节点必须使用同一个值：

```bash
git fetch origin
export V214_COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
echo "$V214_COMMIT"
```

在每个节点自己的仓库中，将上面输出的完整commit设为`V214_COMMIT`，
创建独立worktree，不在运行中的checkout里pull：

```bash
git fetch origin
git worktree add --detach /tmp/training-free-v214-${V214_COMMIT:0:8} "$V214_COMMIT"
cd /tmp/training-free-v214-${V214_COMMIT:0:8}

export V214_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v214_runs/v214_${V214_COMMIT:0:8}_sixnode
export GPU_LIST=0,1,2,3,4,5,6,7
export V214_SOURCE_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export WAN_MODEL=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B
export VBENCH_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench

# Change these two values on each node using the table above.
export NODE_RANK=0
export V214_NODE_ADDRESS=28.216.19.213
```

以上模型与prompt沿用已有文件，不重复下载。默认conda环境为`longlive`。
输出目录必须是六节点可见的同一共享路径，所有输入会冻结并记录hash。

## 5. rank0前置检查

官方SF目录可复用docs/232中已经准备的干净固定版本；只复用源码与模型，
不能拿绑定v213输入的gate文件冒充本轮的检查结果。
没有该目录时先按docs/232执行clone与固定commit步骤。

```bash
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3

bash scripts/run_v214_experiment.sh prepare
bash scripts/run_v214_experiment.sh schedule
bash scripts/run_v214_experiment.sh baseline
bash scripts/run_v214_experiment.sh gate0
bash scripts/run_v214_experiment.sh smoke
```

- baseline在source0/64上对比官方SF、本地SF、官方重复，共6条短轨迹。
- gate0在同两条上比较native与alpha0，共4条短轨迹。
- smoke是source0的七方法完整30秒bundle，直接复用到672条主视频。
- 两种数值检查失败会阻止批量生成，不需要人工review短轨迹。

这段只用rank0的一张卡是为了先发现基础错误，**不是主实验的资源安排**。
GPU实际数值一致性仍需服务器验证，本地没有Torch/GPU环境。

## 6. 六节点同时生成

rank0完成上一步后，在**六个节点各执行一次**，保持各自不同的rank/IP：

```bash
bash scripts/run_v214_experiment.sh generate96
```

任一节点可以查看所有方法的完成度：

```bash
bash scripts/run_v214_experiment.sh status
```

预期每个方法最终为`96/96 validated`，日志路径为：

```text
$V214_OUT_ROOT/jobs/screen96/<method>/source_<id>/stdout.log
$V214_OUT_ROOT/jobs/screen96/<method>/source_<id>/stderr.log
$V214_OUT_ROOT/jobs/screen96/<method>/source_<id>/trace.jsonl
$V214_OUT_ROOT/jobs/screen96/<method>/source_<id>/done.json
```

LPHC日志保留layer/phase、历史预算、去重、修正幅度及source身份审计。
同一卡上两条prompt顺序执行；不同卡并行。当前代码没有时限保证：
实际时长取决于单卡速度，主生成阶段每卡约是14次完整视频推理的工作量。

## 7. 六节点评测与汇总

主生成全部完成后，rank0发布，检查完整配对、视频hash与输入一致性：

```bash
bash scripts/run_v214_experiment.sh publish
```

六节点各执行split，全部完成后再进入评测：

```bash
bash scripts/run_v214_experiment.sh split
# Wait until split has completed on all six nodes.
bash scripts/run_v214_experiment.sh preflight
bash scripts/run_v214_experiment.sh eval
```

中断时使用`eval-missing`；完成后rank0执行：

```bash
bash scripts/run_v214_experiment.sh collect
bash scripts/run_v214_experiment.sh analyze
bash scripts/run_v214_experiment.sh package
```

评测沿用core-9、full/early/late窗口、配对CI与运动诊断，已经改为验证完整96条。
不把core-9当成完整官方Semantic/Total；DD全1的问题仍需单独定位。
人工review队列最多4组异常配对，完整告警全部保留，不要求审完672条。
耗时含模型加载/VAE/编码，显存是采样的进程显存，不冒充纯DiT吞吐或严格峰值。

优先回传`v214_small_artifacts.tar.gz`，包含：

```text
inputs/manifest.json
inputs/placement.json
decisions/sf_upstream_gate.json
decisions/gate0.json
evaluation/metrics/vbench_core9_summary.json
evaluation/metrics/temporal_diagnostics.csv
evaluation/analysis/v214_remaining96.json
evaluation/analysis/v214_remaining96.md
evaluation/analysis/v214_remaining96.csv
```

## 8. 分析与收束边界

先分别看v213的32条和v214的96条，判断趋势是否一致。
本次交付独立96条的完整分析，不自动合并旧32条报告或更改旧决定。
之后合并128条前，需要核对模型/推理算子、方法配置、seed规则、prompt文本和评测fingerprint。
不能简单把32条均值与96条均值各占一半；最终应使用128条逐prompt记录计算配对统计。

主候选固定为`.02/e1`，其余强度/phase是探索变体。
若主候选稳定改善FIFO21且没有明显运动退化，可收束方法并补60秒验证。
若只有新变体有效，补匹配机制对照与新seed；若随机历史同样好，缩小内容检索贡献。
如果连FIFO25都不能区分，要结合成本与长时稳定性解释，不能只靠模块命名。
不因为扩大样本就自动输出`paper_claim_ready=true`。

本轮只修改实验调度与规模相关的发布、评测、分析，不修改SF/LPHC模型前向。
已有v212/v213 checkout不升级、不迁移输出目录，完成的视频不重复生成。

本地检查：v209至v214相关协议回归共**109 passed, 2 skipped**，
两项Torch测试因本地无Torch而跳过；Python编译与三个Bash入口语法检查通过。
48槽位各两prompt、无source重叠、96条统计、配对完整性和旧版32条兼容均有测试。
未在本机运行GPU生成、FlashAttention或实际VBench，服务器前置检查不能省略。
