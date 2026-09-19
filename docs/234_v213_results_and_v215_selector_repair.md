# v213结果、写作判断与v215定向检索实验

## 1. 本次同步与写作判断

本次合并`origin/worktree-v210-lphc`至`5bd33b67`：补齐v209 native gate与v213结果。
v213结果目录为`artifacts/experiment_results/v213_generation_a39f503a_eval_45e79ec1`。
生成使用training-free commit `a39f503a8cb17e88a517e06012f48f97d8d8af21`；
目录中的`45e79ec1`是VBench仓库版本，不是training-free版本。
当前快照没有v212/v214结果，不能据此判断它们成功或失败。

**结论：可以开始写背景、方法假设、实现和实验协议；暂不宜冻结一篇主张稳定改善SF的效果论文。**
不需要SOTA，也不要求每项指标都赢。但是目前连较小、稳定、可复现的优势仍未建立，
问题不仅是某个gate过严。已有强度/phase变体的均值都很小，配对区间跨零，
检索也没有证明比随机历史更有效。应先用定向实验提高信号，而非先写好成功结论再找支持。
不能保证中稿；最有助于投稿的是可信基线、真实收益、与收益相对应的机制证据及清晰边界。

### v213主要结果

32个prompt、7方法、224条30秒视频；同prompt、同seed、同物理GPU配对。
下表Quality及其差值均为百分制分数/百分点，不是相对百分比：

| 方法 | 官方公式Quality | 相对SF21差值 | 差值95% CI | 固定DD诊断差值 | 运动/边缘自动告警prompt数 |
|---|---:|---:|---|---:|---:|
| SF FIFO21 | 81.6692 | 0 | - | 0 | - |
| SF FIFO25 | 81.7694 | +0.1002 | 本表未列 | -0.2844 | - |
| pooled/e1/.02 | 81.7787 | +0.1095 | [-0.1856, +0.3910] | -0.0347 | 3/32 |
| pooled/e1/.10 | 81.6945 | +0.0253 | [-0.3489, +0.3756] | -0.0388 | 2/32 |
| pooled/e2/.02 | 81.7587 | +0.0895 | [-0.2083, +0.3550] | +0.0093 | 5/32 |
| pooled/full/.02 | 81.8024 | +0.1331 | [-0.0734, +0.3399] | +0.0210 | 4/32 |

- pooled/e1/.02相对匹配random：官方Quality仅+0.0283，CI[-0.1794,+0.2416]。
- ID/背景、语义、运动等分项也没有给出稳定全面提升的证据，不能把画面稳定直接等同身份收益。
- 告警是自动代理信号，不是人工确认的视频失败。DD均值略升与个别视频后期运动退化可以同时成立。
- 官方SF固定版本`33593df3`在source3/67上的对照与自身重复通过；每组比对4805项。
  alpha=0每组比对5346项通过。证据覆盖这些canary与采样检查，不是所有视频逐tensor完全相同。
- v209新上传native gate通过，支持继续使用更强的FIFO基线；不是方法有效的证据。

可复核摘要：`artifacts/experiment_results/v215_planning/v213_compact_evidence.{json,md}`。
复核脚本验证239个小文件hash关联、224个完成receipt、逐prompt均值及Quality分解。
Windows checkout的CRLF仅在能精确还原原SHA时作为传输换行处理，逐文件记录，不改任何数值。
没有本地视频、原始tensor trace或VBench原始parts，不能声称重新验证了媒体内容/所有评测计算。

## 2. 评测需要澄清的点

v213的DD为约0.385至0.435，不再是之前的全1。其官方Quality已经存在于JSON，不能忽略。
固定DD诊断只是把官方加权公式中的DD固定为1，**不是官方Quality，也不是完整Total**。
差值满足`delta_official = delta_fixed_DD + (100/13)*delta_DD`。

VBench checkout为dirty，但同轮fingerprint固定：不能仅因dirty就判实验无效；
同样也不能仅凭hash就知道它与早期DD全1时的代码/权重差异。
应回传实际patch、改变的runtime源码及实际使用的RAFT模型信息。
core-9不覆盖完整Semantic/Total，不能编造这两项或与论文完整Total直接比较。
`temporal_style`与`overall_consistency`在当前custom-prompt路径重复，已有分组避免重复计权。

新增导出器会拒绝与comparison manifest不一致的评测checkout，只导出源码/patch及模型SHA，不上传权重。
`--raft-checkpoint`必须由运行者按`dynamic_degree.py`及评测日志确认；声明路径不等于证明实际加载。
v213 frozen checkout仍在时优先导出一份，v215发布comparison后再导出一份：

```bash
python scripts/export_v215_vbench_provenance.py \
  --vbench-root "$VBENCH_ROOT" \
  --comparison-manifest "$V215_OUT_ROOT/evaluation/vbench_comparison/comparison_manifest.json" \
  --raft-checkpoint "$RAFT_CHECKPOINT" \
  --output "$V215_OUT_ROOT/evaluation/vbench_runtime_provenance.json"
```

若旧checkout已改变，不要绕过校验或把当前源码冒充旧源码；保留原fingerprint，明确缺失边界。
导出后检查其中没有凭据，再上传小文件。

## 3. 新实验只解决两个明确问题

### 假设A：原检索统计过度混合了head

现有descriptor对V的空间token和head两个维度一起求均值/标准差，得到256维统计量。
这种统计在head置换下不变，可能抹掉“同一head对应什么特征”的信息。
这是代码可确认的信息损失，**不是已确认的实现bug，也不是已证明的性能瓶颈**。

新变体保留head索引。对每个历史帧f、head h，在空间维求统计：

```text
d[f,h] = normalize(concat(mean_spatial(V[f,h]), std_spatial(V[f,h])))
q[h] = normalize(mean_frames(d[previous_clean_block,h]))
score_headwise(f) = mean_h cosine(d[f,h], q[h])
```

每个head参与打分，但最终按层选相同的4帧供所有head读取，不做head路由/二分类。
30层分别计算本层descriptor，不混层；仅使用已完成clean refresh的V，不窥视未来。

第二个变体在当前eligible历史池内计算公共分量`c[h]=mean_f d[f,h]`，
用`d[f,h]-c[h]`与`q[h]-c[h]`重新归一化并计算平均head cosine。
意图是降低公共背景/均值方向的支配，让内容差异更有区分度；也可能放大噪声，因此必须与未中心化版本对照。
有效norm阈值`1e-6`仅避免除零，不是head分类阈值。退化head不投票；全部退化时分数为0，
按frame id确定性打破平局，不引入隐式随机或改动去噪RNG。

### 假设B：full-phase的微小收益是否只是额外历史暴露

补`fifo_full_random`，与full/.02逐项匹配，只把内容排序替换为同池等数量随机选帧。
原e1随机对照不能证明full的内容选择有效。
e1的三个descriptor变体可共享同一个random对照，因为随机选择不读取descriptor数值，
archive构造/eligibility/数量/seed均相同；新增CPU-Torch测试验证其payload与RNG一致性。

## 4. 完全不变的缓存与修正算子

- Local：原生SF FIFO21，sink0，21包含当前3帧块，不是21历史加3当前。
- Archive：最多12帧clean pre-RoPE K/V；沿用first/last加确定性hash优先级保留规则。
- Eligible：archive去掉与当前local重叠的帧，最多选4帧；块内冻结选择，后续phase不重新排序。
- History RoPE：沿用已有原始位置sidecar及读出流程，不换representation、不压缩K/V。
- 修正：保留原生local attention输出，额外读local+history并计算差值；逐head RMS限幅后乘alpha加入。
- e1为去噪phase0，full为0/1/2/3；clean refresh不施加历史修正。

本轮只改选择分数。无新增sink、PF头分类、motion bank、ABA或CFG技巧，防止同时改多个机制。
方法名里的`correct`沿用旧代码命名，仅表示内容分数检索，不代表有正确历史的oracle标注。
descriptor计算传至CPU进行小矩阵比较会同步设备，需保留耗时记录；不能预先声称提速。
历史缓存额外内存仍存在。v213观测进程峰值为LPHC 33240MiB、SF21 26776MiB，
约24%差异；这是采样进程显存，不是精确CUDA峰值/FLOPs等预算证明。

## 5. 冻结的v215矩阵与48卡安排

| 方法 | descriptor/检索 | phase | alpha | 作用 |
|---|---|---|---:|---|
| sf_fifo21 | 原生，无历史修正 | - | 0 | 强SF基线 |
| fifo_correct | pooled内容分数 | e1 | .02 | 旧机制配对对照 |
| fifo_random | 同池随机4 | e1 | .02 | 三个e1选择器的共同随机对照 |
| headwise_correct | 保留head特征 | e1 | .02 | 新选择器A |
| centered_correct | 保留head并去公共分量 | e1 | .02 | 新选择器B |
| fifo_full_a002 | pooled内容分数 | full | .02 | 旧轮相对较好候选 |
| fifo_full_random | 同池随机4 | full | .02 | full匹配机制对照 |

48条来自128条Qwen改写prompt：`{3,7,...,127}`与`{0,8,...,120}`的有序并集。
每方法48条，总计336条，120 latent frames/477 decoded frames/16FPS，约29.8秒，480×832。
固定`seed=21500+source_index`；这是新的配对seed，不能把v213的视频当成本轮匹配SF。
48条均为开发样本，不宣称新测试集。已有不同seed结果保留，新的同root完整任务会校验后跳过。

| NODE_RANK | V215_NODE_ADDRESS | prompt bundles | 主视频 |
|---:|---|---:|---:|
| 0 | 28.216.19.213 | 8 | 56 |
| 1 | 28.216.19.143 | 8 | 56 |
| 2 | 28.216.19.137 | 8 | 56 |
| 3 | 28.216.19.225 | 8 | 56 |
| 4 | 28.216.18.144 | 8 | 56 |
| 5 | 28.216.18.136 | 8 | 56 |

按source有序位置`j`分配`node=j%6, gpu=floor(j/6)`，每卡一条prompt的七方法串行，
方法顺序按prompt轮换。不是48卡DDP。smoke提前完成source0，因此主阶段该槽位可直接跳过。
按v213约25至28分钟/视频粗估，每卡七条约3小时；另加前置检查与VBench，不承诺完成时限。
节点数不是吞吐保证，共享存储和模型加载会影响时长。

## 6. 可执行命令

**不要在仍运行v212/v213/v214的checkout上pull。** 若v214在运行，让其完成并保留视频；
若尚未启动，优先本轮，而不是先把旧弱候选扩到672条。新旧实验不要同时占同一批GPU。
以下节点地址若已变化，先更改授权配置并冻结新commit，不伪造环境变量。

rank0在现有仓库中取得一次commit，六节点使用同一个完整值：

```bash
git fetch origin
export V215_COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
echo "$V215_COMMIT"
```

每个节点在自己的仓库中设置该`V215_COMMIT`，创建独立worktree：

```bash
git fetch origin
git worktree add --detach /tmp/training-free-v215-${V215_COMMIT:0:8} "$V215_COMMIT"
cd /tmp/training-free-v215-${V215_COMMIT:0:8}
export V215_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v215_runs/v215_${V215_COMMIT:0:8}_sixnode
export GPU_LIST=0,1,2,3,4,5,6,7
export V215_SOURCE_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export WAN_MODEL=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B
export VBENCH_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3
# Change these two values on each node, using the table above.
export NODE_RANK=0
export V215_NODE_ADDRESS=28.216.19.213
```

默认conda环境longlive，全部模型沿用已有文件。官方SF目录准备见docs/232。
rank0先在已配置环境运行CPU-Torch回归，**这些测试失败不要开始生成**：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
python -m pytest -q tests/test_lphc.py tests/test_lphc_native_integration.py \
  tests/test_lphc_head_descriptors.py tests/test_v215_lphc.py \
  tests/test_v210_postprocessing.py tests/test_v211_postprocessing.py

bash scripts/run_v215_experiment.sh prepare
bash scripts/run_v215_experiment.sh schedule
bash scripts/run_v215_experiment.sh baseline
bash scripts/run_v215_experiment.sh gate0
bash scripts/run_v215_experiment.sh smoke
```

baseline：source3/67，官方/本地/官方重复，共6条短轨迹。
gate0：同两条，SF与pooled/headwise/centered的alpha0检查，共8条短轨迹，SF参考去重共享。
smoke：source0七方法完整30秒，计入336条，不重复。前置数值/trace失败会阻止批量生成。
前置只用rank0一张卡，主实验才用六节点；不会要求人工审这些短轨迹。

前置通过后，**六节点各执行一次**：

```bash
bash scripts/run_v215_experiment.sh generate48
```

生成全完成，rank0发布；然后六节点各自split，全部split完成后再评测：

```bash
# rank0 only
bash scripts/run_v215_experiment.sh status
bash scripts/run_v215_experiment.sh publish
# each of the six nodes
bash scripts/run_v215_experiment.sh split
# Wait until all six splits complete, then on each node:
bash scripts/run_v215_experiment.sh preflight
bash scripts/run_v215_experiment.sh eval
```

中断用`eval-missing`。评测完成rank0汇总，按第2节导出评测provenance，再打包：

```bash
bash scripts/run_v215_experiment.sh collect
bash scripts/run_v215_experiment.sh analyze
bash scripts/run_v215_experiment.sh package
```

## 7. 日志与返回材料

每方法应显示`48/48 validated`。日志位于：

```text
$V215_OUT_ROOT/jobs/screen48/<method>/source_<id>/stdout.log
$V215_OUT_ROOT/jobs/screen48/<method>/source_<id>/stderr.log
$V215_OUT_ROOT/jobs/screen48/<method>/source_<id>/trace.jsonl
$V215_OUT_ROOT/jobs/screen48/<method>/source_<id>/done.json
```

新日志含`descriptor_mode`、排序后的frame id及分数、有效head数、top4边界margin、score spread、
是否真的存在超过4帧的选择空间；原有local/archive/eligible/selected/phase/RMS修正日志保留。
只在实际排序时保存score，而不是每个full phase伪造一份新排名。
审计核对候选集合、选中top4、有限分数、desc排序与mode，不满足时任务不能标成功。
`calls_with_real_choice`很低说明大多时候选满所有eligible帧，不能解释为语义检索有效。
全0或很小margin提示排序退化；不是自动失败，也不能当作分类明确的证据。

回传`v215_small_artifacts.tar.gz`，主要包括：

- `inputs/manifest.json`、两种gate、全部done和trace日志；
- `evaluation/metrics/vbench_core9_summary.{json,csv}`与temporal diagnostics；
- `evaluation/analysis/v215_selector_phase.{json,md,csv}`；
- `evaluation/vbench_runtime_provenance.json`。

人工review最多4组自动异常配对，不要求浏览全部336条；其余逐prompt数值与所有告警仍完整保存。
运行失败先回传stdout/stderr/trace，禁止通过删告警、改receipt或改seed获得通过。

## 8. 评判与后续收束

本轮预先指定**官方公式Quality为主要排序指标**，固定DD分数仍作为质量诊断，
subject/background、语义、运动、early/late分项同时检查；这不是追溯修改v213的固定DD主指标。
统计单位prompt，候选对SF的三个Quality检验做BH，配对CI和胜率全部公开。
当前`.10`百分点开发晋级线是资源筛选启发式，不是理论阈值或论文充分条件。

1. 新选择器同时优于SF、pooled和random，且没有靠退化运动换稳定：冻结候选，再做更广prompt/new seed确认、60秒外推及必要消融。
2. 新选择器只优于SF、不优于random：可保留历史修正有效的方向，不能声称内容选择带来收益。
3. 只有full有效：优先解释phase/累计历史修正，不宣称head分类贡献；必须结合本轮full-random。
4. 两类选择器仍无信号：停止这一局部改造族的无差别扩展，回看历史候选在已对齐SF上的收益，或重新设计干预位置。不能用“快写论文”代替证据。

真正可形成论文的候选故事是：**保留局部生成能力，用对应head表征选择非局部历史，并限制历史对当前去噪的干预。**
此时head对应性只是检索技术点，不是证明存在某种固有head类别。
具体哪些点能作为贡献由匹配对照决定；缓存+检索+限幅的组合也需要明确与已有工作的关系，
不能把常见组件重命名就宣称首次提出。现在不写成功摘要，不承诺SOTA或中稿。

## 9. 本地检查边界

本地仅运行纯Python/NumPy协议、数学与统计检查，以及语法检查；无Torch、GPU、FlashAttention、模型推理或实际VBench。
Windows没有`fcntl`，两个旧postprocessing测试需在服务器Linux环境运行。
CPU-Torch测试已补写，需按第6节在服务器执行，不能把本地skip说成通过。
本地相关回归为**131 passed, 3 skipped**；三个skip为Torch不可用。
所有本轮Python文件编译检查及四个Bash入口的语法检查通过。
