# v217：64卡新seed与随机历史复验，服务四页论文收束

## 1. 本次同步与写作判断

2026-09-20执行`git fetch origin`。工作分支最新为`d544c5e8`，
结果分支`worktree-v210-lphc`仍为`5bd33b67`；没有新增v215/v216结果。
`origin/main`最新结果为9月7日的历史分析，不是这一轮的新结果。
不能把“已提供实验代码”说成“实验已经有效”。

当前最新完整结果仍是v213：224条完成，官方SF与alpha0对齐通过；
full-phase的官方Quality开发均值提升0.133分，CI[-0.073,+0.340]，
内容检索相对随机历史的优势尚未建立。原始分析见docs/234。

**可以现在进入四页稿件的结构、方法公式、图表和已确认事实的整理阶段。**
不要求SOTA、所有指标胜出、所有旧开发gate通过或完整head分类理论。
但目前还不能选定并写死“总体质量稳定提升”这一结果主张。v215未上传时也不能臆测其赢家。
若只改善后段主体一致性，就围绕这一项有限收益写，公开运动/画质代价，不改称总体领先。
CI跨零可作为初步观察报告，不自动判不能投稿，也不称统计显著。

本次再次核对[ICASSP提交门户](https://cmsworkshops.com/ICASSP2027/papers.php)，
北京时间截止列为9月24日20:00；仍按9月23日内部提交安排。
篇幅和作者/AI辅助规则沿用docs/235，不等待所有可选实验完成才开始组织材料。

## 2. 64卡的优先级

| 优先级 | 实验 | 新视频量 | 用途 |
|---|---|---:|---|
| 已在运行 | v215七方法48 prompts | 不重复已有任务 | 一次选定方法与主要收益 |
| P0 | v216：SF与一个冻结方法，80 prompts | 160条30秒 | 主要确认；与原48条形成有标注的128条覆盖 |
| P1，可选 | v217：SF/冻结方法/匹配random，64 prompts，新seed | 192条30秒 | 稳定性与检索机制，正好64个三方法bundle |
| 暂缓 | 去clip、60秒、跨模型、PF、ABA、新分类器 | 0 | 除非主结论确实缺这项证据 |

v216入口已经在docs/236中实现，本轮继续复用；不另开一个重复的80条主实验。
v217在v216选型冻结后即可冻结，不读取v216的成绩来选择方法/指标/样本。
同一批64卡先完成v216生成和所需评测，再运行v217，不能直接重叠占卡。
如果v216已经足够支持有限的四页主张，优先写作，v217不是新的硬门槛。
如果v215完全没有可接受信号，不为了填满卡盲目启动两轮。

## 3. v217到底改变什么

从v216的80个有序source中固定取`source_list[floor(i*80/64)]，i=0..63`。
这是在查看确认结果之前确定的64条，不按视频效果筛选；它们都不在v215的48条选型样本内。
seed从`21500+source`改为`21600+source`，这是新噪声重复，不是64个历史从未见过的新prompt。

| 方法 | 与冻结方法的关系 |
|---|---|
| sf_fifo21 | 相同权重/生成配置/seed的原生SF |
| ours_correct | 完全继承v216所选的v215方法，alpha/phase/descriptor/预算不变 |
| ours_random | 只把retrieval_mode改为random，其余全部相同 |

若选full，则random也用full；若选headwise/centered，则random仍保留同样descriptor构建，
只是随机选帧不读分数。不能用e1的random来证明full的内容选择有效。
random沿用独立CPU generator与同一eligible池，不消费去噪的全局随机状态。
SF、ours和random每个source在同一物理GPU串行运行，方法顺序按node/GPU轮换。

模型算子、缓存、RoPE及原有debug输出均未修改。
本轮只补实验脚本，因此不会改变v216要求“与v215相同推理算子”的条件。
新旧已运行checkout仍不更新，已完成视频不重生成。

## 4. 64卡任务分配

固定`j=0..63`分配`node=j%8, gpu=j//8`。
每节点8条prompt、24条视频，每卡恰好一个三方法bundle，合计192条。
首方法的轮换为`(node+gpu)%3`，每节点三种首方法数量为2/3/3，不将方法绑定某个节点。
smoke提前完成source1的三条完整视频，计入192条，主阶段跳过其已验证任务。

v216则仍为每节点10个双方法bundle、20条视频；它的80条分配已覆盖64个槽位。
两轮都不是单视频跨64卡DDP。

评测沿用每方法9个维度，v217共27个维度任务，八节点分配；
不承诺评测阶段64卡全部满载，也不为利用率临时改写VBench算子。
按v213每视频约25至28分钟估算，v217主生成约三次串行推理量级，约75至85分钟，
但前置检查、加载、共享负载、评测不包含在此估计，不作为时限保证。

节点IP沿用v216冻结的八节点清单。`configs/v216_nodes.example.json`最后两项仍为空，
必须填真实IP，若前六节点也变化一并更改。freeze之后不能偷偷更换节点。
本地不会自动SSH或启动远程GPU任务。

## 5. 先执行v216

按docs/236设置v215原始结果root、八节点清单、选定候选、主要指标及理由，执行：

```bash
# rank0, after v215 collect/analyze is complete
bash scripts/run_v216_experiment.sh freeze
bash scripts/run_v216_experiment.sh prepare
bash scripts/run_v216_experiment.sh schedule
bash scripts/run_v216_experiment.sh baseline
bash scripts/run_v216_experiment.sh gate0
bash scripts/run_v216_experiment.sh smoke
# on each of eight nodes, after the gates pass
bash scripts/run_v216_experiment.sh generate80
```

服务器CPU-Torch检查及publish/split/eval/collect/analyze完整命令见docs/236。
候选未定、结果未完整、节点未填时不要猜值启动，也不要拿旧不同seed的视频冒充配对控制。

## 6. v217独立checkout与冻结

在现有仓库fetch，记录同一commit；八节点各用独立checkout，不pull正在运行的v215/v216：

```bash
git fetch origin
export V217_COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
# Share the printed full commit with every node; do not resolve different commits later.
echo "$V217_COMMIT"
git worktree add --detach /tmp/training-free-v217-${V217_COMMIT:0:8} "$V217_COMMIT"
cd /tmp/training-free-v217-${V217_COMMIT:0:8}

export V217_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v217_runs/v217_${V217_COMMIT:0:8}_replicate64
export V217_V216_ROOT="$V216_OUT_ROOT"
export GPU_LIST=0,1,2,3,4,5,6,7
export V217_SOURCE_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export WAN_MODEL=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B
export VBENCH_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3
# Set these per node using the exact frozen v216 node order.
export NODE_RANK=0
export V217_NODE_ADDRESS=28.216.19.213
```

若当前shell没有`V216_OUT_ROOT`，把`V217_V216_ROOT`设为实际已freeze的v216 root。
`freeze`只复制选择合同及v215的四个小文件快照，不依赖v216生成/评测完成，不看其效果。
它拒绝再次选择候选/指标/节点；同一root重跑freeze只允许完全一致的输入。

rank0在已配置longlive环境先检查，再执行：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
python -m pytest -q tests/test_lphc.py tests/test_lphc_native_integration.py \
  tests/test_lphc_head_descriptors.py tests/test_v216_confirmation.py tests/test_v217_replication.py

bash scripts/run_v217_experiment.sh freeze
bash scripts/run_v217_experiment.sh prepare
bash scripts/run_v217_experiment.sh schedule
bash scripts/run_v217_experiment.sh baseline
bash scripts/run_v217_experiment.sh gate0
bash scripts/run_v217_experiment.sh smoke
```

baseline为source1/65的新seed官方SF、本地SF、官方重复；gate0验证冻结方法的alpha0。
已有v216 gate绑定旧seed/root，不能直接复制来冒充v217检查。
freeze/prepare可提前做；baseline及之后需要确认GPU已经释放。

## 7. 八节点生成、评测与返回材料

前置通过且设备可用后，每个节点执行：

```bash
bash scripts/run_v217_experiment.sh generate64
```

rank0确认三个方法各`64/64 validated`，然后发布：

```bash
bash scripts/run_v217_experiment.sh status
bash scripts/run_v217_experiment.sh publish
```

八节点分别切片，所有切片完成后分别评测：

```bash
bash scripts/run_v217_experiment.sh split
# Wait for all eight split workers to complete.
bash scripts/run_v217_experiment.sh preflight
bash scripts/run_v217_experiment.sh eval
```

中断使用`eval-missing`。完成后rank0：

```bash
bash scripts/run_v217_experiment.sh collect
bash scripts/run_v217_experiment.sh analyze
bash scripts/run_v217_experiment.sh package
```

回传`v217_small_artifacts.tar.gz`。日志在`jobs/replicate64/<method>/source_<id>/`，
含stdout/stderr、trace和done；descriptor分数、实际选帧、eligible池、phase与修正量沿用原日志。
重点产物为`evaluation/analysis/v217_seed_random.{json,md,csv}`，以及冻结的`inputs/selection.json`。
发布时继续要求与v215同一个评测fingerprint，不静默改评测口径。

## 8. 怎么据此写四页

v216是主要确认。v217继承同一个主要指标/窗口，ours-vs-SF回答新seed是否保留收益，
ours-vs-random为次级机制对照。二者分别给配对均值、CI和胜率，不合并成有利的新分数。

- ours优于SF，也优于random：可以支持这个有限任务上的历史选择与修正故事。
- ours优于SF但不优于random：可以讲历史修正，不把内容检索列为已证实贡献。
- 只改善后段某项：限定主张，正文同时给全段画质与运动，不要求所有维度胜出。
- v216/v217方向不一致：报告seed敏感性，不把同prompt两seed当128条独立prompt来缩小区间。
- 没有可接受收益：不靠最好的几个视频强写效果论文；可调整为经验观察，但不保证适合该venue或能录用。

v217不自动汇总跨seed显著性，也不对v216的128条开发+确认汇总再累加64条“独立样本”。
其完整表格足够人工解释seed一致性。研究范围小，可以不写大而全的机理理论。
默认人工总预算仍最多6组：v216为主；v217最多输出2组新风险候选，替换原队列中的低优先项，
不是要求再审一套192条视频。它们是诊断，不称为正式用户偏好研究。

写作保留一个核心公式、一张主表、小消融表及少量示例；
不要求PF、跨模型、60秒全部补齐后才投稿。已有作者负责正文，代码/数值需核查，
遵循ICASSP对AI辅助的要求。本机没有Torch/GPU，未声称完成实际推理或VBench运行。

## 9. 本地验证记录

本轮相关回归共**164 passed, 3 skipped**；skip来自本机没有Torch。
Python编译与共享/v217两个Shell入口的语法检查通过。
覆盖64槽位、每节点方法顺序轮换、64条固定采样、所有四种可继承候选的匹配random、
seed/配置/选型不可变、节点授权、旧v216兼容、评测规模以及分析文件输出。
新调度的首次测试发现方法首顺序会与节点绑定，已在本地修正并由回归验证；
这是未发布v217开发代码的问题，不是对已有v213/v215结果提出新的否定证据。
未安装或运行模型环境；服务器仍需先通过Torch测试与baseline/gate0再开始主生成。
