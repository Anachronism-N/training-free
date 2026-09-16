# v210：六节点 A800 的效果优先实验计划

日期：2026-09-16。阅读并同步到：`5286d7f5`。
分支：`codex/v178-v179-causal-validation`。

**定位：先找到可复现的生成收益，再写论文。本文是实验计划，不是实验结果。**
本次不改变推理算法；第一批复用 v209/v207，新的历史修正分支尚需实现。
六个节点已确认，每节点卡数、显存和可用时长待确认。下文的 48 卡分配仅以每节点 8 卡为例。

## 1. 当前判断

### 1.1 最新提交不是新的正结果

本次拉取包含 reference flags 保留、v209 landmark canary 不再误传 retrieval archive 参数，
以及节点专用启动脚本。没有新增 v207/v209 完整 canary 或视频评测报告。
不能据此声称 runtime 已验证等价，也不能把旧结果当作修复后结果。

当前最重要的实测证据：

| 对比 | Quality without DD | Temporal | Semantic | Visual |
|---|---:|---:|---:|---:|
| v201 retrieval all-Recent - SF | -0.1513 | -0.005875 | +0.010282 | +0.009502 |
| v201 retrieval horizon - SF | -0.1822 | -0.006141 | +0.008345 | +0.009353 |
| v201 retrieval horizon - matched all-Recent | -0.0309 | -0.000266 | -0.001938 | -0.000149 |

这是均值，不代表每个正数都统计显著。特别是 all-Recent 已有类似的语义/画质变化，
所以不能将这些变化直接解释为历史检索收益。v201 的正式结论仍是没有建立相对 SF 的改善。

另一个值得继续的证据是：v198 的 60 秒 retrieval 相对 9-FFE all-Recent 有质量/身份收益，
但没有通过时序安全检查。它支持继续研究受控历史读取，不支持直接扩大那个配置。

来源：

- `runs/v201_head_phase_horizon_sf_screen/screen32/analysis/v201_head_phase_horizon_comparisons.csv`
- `runs/v201_head_phase_horizon_sf_screen/screen32/analysis/v201_head_phase_horizon.md`
- `runs/v198_audited_long60/analysis/v198_long60_operator.md`
- `docs/228_v209_sf_protocol_budget_and_paper_convergence.md`

### 1.2 本轮不再追求什么

- 不要求 SOTA，也不要求所有指标同时第一。
- 不继续把静态 head 分类作为必须成功的前提，不再大规模重跑 profiling。
- PF 不作为本轮必须生成的比较方法；使用其基础代码仍保留出处。
- 暂不跑 ABA、跨模型大矩阵、几十个压缩/缓存排列组合。
- 不以提高 DD 常数、改变评测权重或挑选几个好视频作为成功。

目标是：**相对可信 SF 对照，有稳定且可解释的质量或长程保持收益，运动/时序代价可接受，
且至少一个新增机制具有 matched control 支持。** 这是项目的晋级标准，不是录用保证。

## 2. 主方法方向：保护局部连续性，控制历史干预

暂用工作名 **Local-Preserving Historical Correction，LPHC**，不是最终论文标题或新颖性声明。

核心假设：远期历史可能改善外观和语义，但替换过多近期上下文、或在细节去噪阶段强制读历史，
会破坏运动连续性。与其先定义更多 head 类别，不如先控制历史对生成轨迹的作用。

保留两条路线，而不是把所有技巧叠起来：

### A. 现成代码路线：21-FFE phase retrieval

复用 v207：

```text
Local:     sink1 + recent20                  = 21 FFE
Retrieval: sink1 + retrieved4 + recent16      = 21 FFE
Archive:   12 FFE，独立于 attention read budget
```

重点候选是 early1 和 early2：四次 noisy calls 中，分别只在第 0 次、或第 0/1 次读历史。
clean commit 始终用 Local。MMR 检索、clean descriptor、exact K/V 沿用已有实现。

优点是代码已有，最快验证“恢复预算后效果是否改善”。局限是仍替换四帧 recent，
并不等于完整保留 SF21，也不能凭与 SF 同为 21 FFE 就宣称数值等价。

### B. 新实现路线：完整 SF21 + 限幅历史修正

这一分支尚未实现。首先只使用原生 SF runtime，完整保留其局部 KV 和位置处理：

```text
y_local = Attn(Q, K_local21, V_local21)
y_aug   = Attn(Q, concat(K_local21, K_history4), concat(V_local21, V_history4))
delta   = y_aug - y_local

scale   = min(1, rms(y_local) / max(rms(delta), eps))
y_out   = y_local + alpha * scale * delta
```

这里 `alpha` 是干预强度上限，不是“最多替换多少帧”。按样本/head 在 token/channel 维计算 RMS，
使用 FP32 累积，避免少量大值直接主导比例。输出投影之前的相对 RMS 干预不超过 alpha；
这不是最终视频质量的理论保证。`alpha=0` 必须直接走原函数，不计算附加 attention。

第一版约束：

1. Local21 不压缩、不重排、不更换 RoPE；新 archive 不能成为第二个 active cache owner。
2. 只归档 clean、已生成的完整历史帧，最多 12 FFE；初版不合成 prototype、不稀疏 token。
3. 读取最多 4 个已离开 Local 的历史帧，排除重叠；不足 4 帧就读可用数量，不复制填满。
4. 第一个 noisy call 之前，用上一个已完成 block 的 clean descriptor 检索；block 内冻结选择。
   不使用当前 block 尚未生成的 clean 结果作为查询，避免未来信息泄漏。
5. 历史 K 只保留一种明确表示：原生 SF 的原位置 RoPE K，或从 pre-RoPE K 精确重建同一表示。
   第一版不混用 saved/dynamic representations，也不再做历史位置重映射。
6. clean commit 保持原生 SF；初版只干预 early1/early2，先使用全部 heads。
7. 首先固定 FIFO21 作为主体，再检查同一模块是否兼容 Sink1+Recent20；两个 SF 对照都保留。

**成本必须明示：**这不是 21-FFE 等预算方法。启用时额外计算一个 25-FFE attention，
同时已有 21-FFE local attention；最多还持有 12-FFE archive。FFE 不能直接当作 latency。
需要 SF25 容量对照和实际 seconds/video、peak VRAM。SF25 不能完全匹配双分支算量，不能冒称 FLOP-matched。

与旧 additive compressed-V 的区别是：不直接加独立 historical output，不丢局部窗口，
修正来自同一 Q/Local 在有无历史时的差，并有零强度旁路及干预上限。
旧分支发生过 ghosting，因此这里仍必须验证，不能因为换公式就假定已经解决。

## 3. 第一批：现在已有代码，先运行这些

### 3.1 小规模 production canary

复用 v209 两条 prompt、30 latent frames 的数值检查。它不是 30 秒视觉筛选，
而是覆盖满窗/驱逐的短数值检查；不做 VBench、不要求人工观看。

- Native pass 后允许原生预算实验，不等 PF/Adaptive。
- PF/Adaptive production pass 后才允许 A 路线大批生成。
- 原有 v207 9-frame math parity 是额外诊断，不能代替这一检查。
- 只复用同 runtime/config/backend/checkpoint 合约的已有报告；不修改旧报告让它变绿。
- 发现分叉只回传首个失败位置，停止重复整个长审计。若 PF 路径持续阻塞，开发优先级转向原生 SF 的 B 路线。

### 3.2 六节点分配：两个独立 worker pool

假设每节点 8 卡，物理节点记为 N0...N5：

| 节点 | Pool 内 NODE_RANK | 内容 | 规模 |
|---|---|---|---:|
| N0、N1 | 0、1，NUM_NODES=2 | v209 原生 SF 预算阶梯 | 5 x 32 x 30s = 160 videos |
| N2...N5 | 0...3，NUM_NODES=4 | v207 等预算 phase retrieval | 9 x 32 x 30s = 288 videos |

每个 pool 内所有方法经过所有节点，不将某个方法固定在某台机器上。
两个 pool 的 SF 视频不能直接合并成一个更大的统计样本；各自在匹配 pool 内配对分析。
v209 sink1 对照可提供补充归因，但不能假装与另一 pool 在同一 GPU 上配对运行。

v209 五个方法：FIFO21、Sink1+Recent20、Sink1+Recent12、Sink1+Recent8、Sink3+Recent6。
原生 SF 的 sink3 不等于历史 Adaptive sink3 warm-start 路径，不复活旧失败配置。

v207 九个方法：

| 方法 | 角色 |
|---|---|
| sf_native | 同一 pool 的 SF FIFO21 对照 |
| recent21 | Adaptive 同预算 local 对照 |
| retrieval21_early1 | 主要效果候选 |
| retrieval21_early2 | 主要效果候选 |
| retrieval21_full | 暴露量上界，可作为候选但不能预设 early 更优 |
| retrieval21_late2 | equal-dose phase 对照 |
| landmark21_early2 | 内容选取方式对照 |
| recent13 | 小预算 local 对照 |
| retrieval13_early2 | 效率候选 |

为何暂保留九个而不是只跑五个：现有 v207 audit/评测合约要求九方法齐全。
`METHODS` 只能减少生成，不能自动建立合法的五方法评测合约。不要只生成五个后强行 collect。
若算力时间有限，应先完成 v209，再提交独立的精简筛选实现，而非删除 manifest 中的方法。

当前首批上限 448 条 30 秒视频，另加 canary/smoke；已经完成且合约相同的 jobs 自动跳过。
若每节点不足 8 卡，修改 GPU_LIST，不变造卡数；两个 pool 的节点映射和 GPU 数须在续跑期间固定。
不要直接设 NUM_NODES=6 运行 32-prompt 的逐方法分片：48 workers 中有16个会没有任务。

### 3.3 第一批命令

所有节点使用同一冻结 checkout。共享路径、checkpoint、prompt 内容必须一致。
若之前的 v209 root 来自不同推理脚本版本，保留原目录，使用新 root；不要覆盖旧 manifest。

```bash
git pull --ff-only origin codex/v178-v179-causal-validation

# Set these same roots on every node; examples are new campaign directories.
export V209_OUT_ROOT="$PWD/runs/v210_campaign/native_v209"
export V207_OUT_ROOT="$PWD/runs/v210_campaign/phase_v207"
export SF_PARITY_REFERENCE_ATTENTION=0

# N0 only. Keep production canaries on this same physical GPU.
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh prepare
NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary-native
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary
cat "$V209_OUT_ROOT/canary_production/analysis/decision.md"

# Still N0/GPU0, finish before occupying that GPU with generation.
NODE_RANK=0 GPU_LIST=0 bash scripts/run_v209_sf_protocol.sh canary-runtime
NODE_RANK=0 bash scripts/run_v209_sf_protocol.sh analyze-canary

# N0 and N1 respectively: NODE_RANK=0 and 1.
# Requires native_budget_screen_ready=true.
NODE_RANK=0 NUM_NODES=2 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v209_sf_protocol.sh generate32
```

如果 Native 已通过而 runtime canary 较慢，可先在 N1 开始它自己的 native shard，
待 N0/GPU0 的 canary 完成后再启动 N0 shard。不要在同一 GPU 同时启动两种任务。

A 路线必须先确认 `pf_adaptive_runtime_ready=true`、production、当前 manifest 匹配，
并确认 `unread_operator_isolation_pass=true`。当前 v207 launcher 只自动检查自己的旧 parity 报告，
**尚未自动绑定 v209 gate，因此下面的 production 检查不能省略。**

```bash
# N2, acting as phase pool rank 0. Only after the production checks above.
# The old launcher still requires its own short parity report.
GPU_LIST=0,1,2 bash scripts/run_v207_sf_parity.sh prepare
GPU_LIST=0,1,2 bash scripts/run_v207_sf_parity.sh run
GPU_LIST=0,1,2 bash scripts/run_v207_sf_parity.sh analyze

NODE_RANK=0 NUM_NODES=4 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh prepare
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh smoke
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v207_context_budget_phase_screen_32gpu.sh audit-smoke

# N2...N5 respectively: NODE_RANK=0...3, NOT physical node numbers 2...5.
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v207_context_budget_phase_screen_32gpu.sh generate32
```

generation 完成后按 docs/228 的 v209 评测命令执行，将 NUM_NODES 改为2；
按 docs/225 的 v207 audit-screen/评测命令执行，NUM_NODES 保持4。
不要在同一张 GPU 上并发 inference 和 VBench。split 可提前预热环境，但发布需等待对应完整方法合约。

通用脚本没有必要调用节点专用的 `run_v209_canary_node25.sh`：后者包含特定路径及 GPU holder 管理。

默认服务器输入：

```text
Prompts:
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
SF checkpoint:
/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
Conda:
/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
Environment: longlive
```

沿用已配置的 Wan base/text encoder/VAE 与 VBench 模型缓存；本轮不引入新生成模型权重。
实际路径以各 manifest 和本地配置为准，prepare/preflight 失败时先修路径，不跳过检查。

## 4. 第二批：新的局部保护候选，分批实现后再启动

第一批运行期间完成 B 路线。不是在当前还没有结果时预先确定其优于 A。

### 4.1 先提交一个完整可跑单元

实现顺序：

1. 原生 SF hook、独占 archive、零强度旁路、clean/noisy 隔离、跨 prompt reset。
2. 两 prompt 的 gate0 数值测试：先有 archive 更新但不读，再与完全关闭模块比较；
   没有这个检查，不批量跑非零强度。
3. 一个30秒视频验证内存非空、历史年龄、干预幅度、无 NaN/预算违规；自动检查为主。
4. 提交推送 runnable 的 8-prompt runner、评测、日志和说明，再补32/128规模。每次推进冻结新目录。

### 4.2 8 prompts x 30s 的效果初筛

使用固定 source indices `1,17,33,49,65,81,97,113`，包含在 systematic32 中。
它是确定性预筛，不声称语义上已分层均衡；保存全部 prompt 原文并报告主题分布。
新 runner 必须以 source index 定义有效 seed，例如 `21000 + source_index`，
不能因从8扩到32而改变同一 prompt 的 seed。

| Cell | Local | History | Noisy calls | alpha |
|---|---|---|---|---:|
| SF FIFO21 | 原生21 | 无 | 无 | 0 |
| SF sink1_21 | sink1 + recent20 | 无 | 无 | 0 |
| SF FIFO25 | 原生25 | 无 | 无 | 0 |
| LPHC E1-02 | 原生21 | retrieval4 | 0 | .02 |
| LPHC E1-05 | 原生21 | retrieval4 | 0 | .05 |
| LPHC E1-10 | 原生21 | retrieval4 | 0 | .10 |
| LPHC E2-10 | 原生21 | retrieval4 | 0,1 | .10 |
| LPHC Full-10 | 原生21 | retrieval4 | 0,1,2,3 | .10 |

共64条。alpha 是开发超参，不宣称来自某个物理定律。先看效果再缩减范围，不做巨大的交叉网格。
8 prompts 只排除灾难性退化/显然无效，不承担显著性或最终排名结论。

最多保留2个非零候选，扩到32 prompts；三个 SF 对照也补齐，共160条，
若同代码/有效 seed/输入合约不变，复用已经生成的40条，共补120条。
门限/模型代码改变则新开实验，不将旧轨迹伪装为当前方法。

## 5. 第三批：最多三项增益小实验

仅在 A 或 B 至少有一个生成侧候选可用后开展。先独立比较，只有各自收益成立才组合。
默认16 prompts x 30s；不重跑完整 PF、CEMR/v78 大矩阵。

| 技巧 | 具体改动 | 依据和风险 |
|---|---|---|
| 平滑 phase 强度 | `[1,.5,0,0]` 的相对强度，而非 early2 两步同强度 | 寻找语义收益与细节保留的平衡；将强度变化与 phase 对齐分开，胜出后补同剂量 late 对照 |
| 保均值的弱 V 方差校准 | 只在历史读取副本上，用 recent 的逐通道标准差做10%插值；比例先截到[.8,1.25]，保留历史均值，原 archive 不变 | 旧 reset-fixed 三 prompt 有正向信号，但样本很小，可能冻结/重影；不得沿用 full/mean moment transport |
| 读取置信度退避 | 历史与当前状态不匹配时将干预降到0，fallback 保持完整 Local；开发集固定阈值 | 针对老状态重放/主体放大；记录拒绝比例，不能靠几乎永不启用获得“安全”结果 |

这些是待实现候选，不是现有 CLI 开关清单。最后一项只在日志显示错误召回时优先；
不同时增加年龄阈值、margin、EMA、head mask 等多个难以归因的门控。
校准用 current recent 的已提交 clean 值，不能读取当前尚未完成的 clean latent。

历史依据见 docs/120；v78 在纠错后没有稳健超过 PF，CEMR 的已有正向证据主要来自 ABA。
所以二者不排在当前单 prompt 主线之前。

## 6. 晋级、确认与最少人工 review

### 6.1 开发与确认分开

- 8-prompt 初筛：排除错误、明显闪回/噪声/冻结，不以一个分数的小差值淘汰全部方案。
- 32-prompt 开发：固定指标报告、paired CI、full/late-half；候选排名使用预先选定的
  `quality_without_dynamic_degree`，接近时检查 subject consistency 和效率。
- 不要求32样本的每一项 CI 都显著。均值正向但 CI 不确定的配置可进入一次确认，不能提前宣布成功。
- 旧 v207 自动 gate 保持原样。新筛选与旧 gate 不同须另写决策 manifest，不能回改历史 pass/fail。

确认实验的主要终点是 full-video quality without DD；VBench 各原始维度全部报告。
正式比较的改善需 paired 95% CI 下界>0，并超过事先冻结的实际效应门槛。
拟用 quality 0.25分（0--100尺度）作为开发目标，**这是项目选择，不是领域公认门槛**；
最终阈值在确认集运行前冻结，并报告原始 delta/CI，不能只报告通过与否。

时序/identity 非劣范围可先沿用 v207 的 -0.0030/-0.0015，报告范围敏感性；
这些也不是“正确无误”的常数。真正严重的重复主体、噪声、闪回或静止不能由加权总分抵消。
若只在 identity 上获益，应另行冻结身份保持主张和独立确认，不事后改主要终点来宣布全局提升。

### 6.2 主方法冻结后再花大算力

1. 冻结一个版本，MovieGen-Qwen128，30秒：SF FIFO21、SF Sink1+Recent20、Ours。
   B 路线额外 SF25 容量对照；A 路线额外 Adaptive Recent21 matched-runtime 对照。
   两条路线均按最终只选一条、四方法共512条规划，不将两个候选都塞入主表后再选赢家。
2. 同样配置做60秒；可以先32条排除时长特有错误，再补齐128，保持每条有效 seed 不变。
3. 至少32 prompts x 第二个seed，对所有主要对照同样运行。不能仅给 ours 增加seed挑最好结果。
4. 加一组未参与这轮搜索的新 prompt 确认（建议64条，来源和去重记录预先固定）。
   旧 MovieGen128 已多轮使用，不能称为 fresh；旧32之外96也只是相对这轮未用。
5. 补主要机制对照：去掉 history、正确 history vs 等预算 random/固定历史、early vs equal-dose late、
   B 路线的限幅开/关。只针对最终方法，不再展开所有历史版本。

128及跨seed的新 runner 需继承输入/实际有效seed/后端/权重/输出的绑定。
现有 v208 可复用大部分评测代码，但不能不改代码就运行 B 路线或新增原生 sink1/SF25 对照。
旧视频只有在运行合约确实一致时才能复用，不能只因 prompt 文本相同就软链接进正式表。

### 6.3 自动化与人工工作量

- 必查：解码长度/FPS/分辨率、输出缺失、NaN、各 prompt 重置、actual cache ids/预算/时序、干预实际启用率。
- VBench-Long 原始各维度 + full/early/late 的配对差；DD 饱和时只单列，不据此证明运动提升。
- 运动补充：camera-compensated motion、motion coverage、时间跳变/停滞；不能将 flow 大等同于运动好。
- 合并指标只能使用真实已有维度。core-9 不是完整官方 Semantic/Total，不填补缺失分数。
- 开发每批最多4个自动异常样本供定位；没有异常不强制看全部视频。
- 最终冻结时另抽8个预先随机选定的配对短片，加最多4个异常样本。报告抽样方式，
  不能只看最好样本，也不把这个小 QC 说成正式 human preference study。

## 7. 六节点使用与进度回传

不承诺未经测量的“10小时全部跑完”。先记录每条30秒视频实际耗时及评测 GPU-hours。
若每节点8卡，第一批两 pool 的纯生成理想用时约为：

```text
native pool: 160 / 16 * median(t_native_30s)
phase pool:  288 / 32 * median(t_phase_30s)
wall time:   max(two pools) + loading/straggler + evaluation + diagnostics
```

使用实际测量再留至少20%时间处理失败和评测。不要根据输出视频30秒推断推理30秒。
先完成的 pool 在其当前所有分片结束后转去评测或新分支，不中途改变已有 pool 的分片数量。
后续64/128等规模可用六节点统一分片；新 runner 优先采用 prompt-method 作业队列，避免小样本空闲。

每个批次结束即推送小文件，不等待全部6节点结束：

- frozen input/config manifest、代码commit/hash、checkpoint/prompt hash、实际seed和GPU信息；
- canary decision 与首处分叉摘要；
- 每个方法完成数量、失败日志、实际cache及gate汇总；
- VBench逐prompt分数、paired结果、motion诊断、每视频时间与显存；
- 一页结论：可用/不确定/淘汰、下一批运行什么。

原始 MP4、大tensor、模型不上传 GitHub。原始数据留服务器，按报告相对路径可定位。
正在运行的 checkout 不原地 pull；新代码推送后使用独立 worktree/checkout，再启动新目录。
也不使用分支名作为唯一运行身份，因为分支会移动。

## 8. 最后怎样形成有说服力的方法叙述

可以积极组织叙述，但技术贡献要由对应结果支撑：

1. **问题：历史有帮助，却会干扰局部生成。** 用 matched-budget 与完整局部窗口实验建立，
   不把旧 runtime bug 写成模型的普遍机制。
2. **方法：局部生成主通路与历史修正解耦。** 不是声称首次使用 long-term memory；
   重点是具体修正形式、可控干预和不牺牲原生 local context。
3. **调度：在有用的 denoising 阶段施加适量历史。** early/late/full 对照支持后才强调 phase-specific。
4. **结果：更好的长程画质/保持能力，同时维持合理运动，并明确计算代价。**
   如果只增加预算有效而选择/限幅无收益，应回到方法设计，不把容量收益归功于分类器。

以下内容是借鉴，不单列为我们的新发明：anchor/recent 分解、历史检索、MMR、RMS限幅、
残差插值、denoising phase 概念。潜在贡献在于针对 AR 漂移的具体组合和经过验证的干预机制。
投稿前仍需针对最终算法做逐项近邻方法核对；当前不能保证该组合尚无人提出。

相关工作的原始代码与本轮用途（2026-09-16检查）：

- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)：生成基座与主对照。
- [Pyramid Forcing](https://github.com/if-lab-pku/Pyramid-Forcing)：已有 cache/RoPE 基础代码来源；
  不重命名其三分类作为新贡献，不要求每轮重跑它。
- [Deep Forcing](https://github.com/cvlab-kaist/DeepForcing)：Deep Sink 与 Participative Compression
  是相关基线；最终冻结后可补一个相同checkpoint的外部方法比较，优先复用可核验旧结果。
- [Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing)：借鉴 scene memory 生命周期，
  不是当前残差分支的已复现代码。官方 README 目前注明代码临时撤回，不能把远程 main 当作现成可跑仓库。

不把“超过SF”说成“超过所有方法”，也不把工程修复、阈值调参本身凑成若干新贡献。
我们优先选择效果与简洁性兼顾的版本，不为保留某个历史名词牺牲生成质量。

## 9. 实现状态与近期顺序

| 项目 | 当前状态 | 下一步 |
|---|---|---|
| v209 native canary / budget32 / VBench | 已实现 | 可立即在服务器运行并回传 |
| v207九配置生成/评测 | 已实现，有既有旧gate | 必须额外确认v209 production结果后运行 |
| 六节点2+4 pool配置 | 现有参数可支持 | 按本文映射启动，固定分片 |
| LPHC原生SF21残差与gate0 | 尚未实现 | 下一次代码交付的第一优先级 |
| 8→32→128统一source-index seed与新方法评测 | 尚未实现 | 随LPHC分批提交，不伪用旧v208合约 |
| 三个增益trick | 尚未实现 | 基础候选可用后逐个提交 |
| 论文正文、ABA、跨模型大规模实验 | 暂缓 | 方法收益确认后再安排 |

**现在最值得返回的不是几百个新视频，而是 production canary 报告；Native通过即可启动第一批预算实验。
与此同时推进LPHC，避免所有方法开发被PF runtime问题串行阻塞。**
