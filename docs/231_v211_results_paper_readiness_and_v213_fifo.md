# v211 结果、写作判断与 v213 FIFO 复验

日期：2026-09-18。同步了 `origin/worktree-v210-lphc` 的 `948da2de`。
交付分支仍为 `codex/v178-v179-causal-validation`。v212 结果尚未上传。

## 1. 能否进入论文写作

**可以先写问题定义、方法、实验协议；还不能把方法有效性当作已经确认的主结论。**
目前不需要追求 SOTA，也不需要所有指标全面获胜，但至少需要：

1. 在公平的、较强的原生 SF 对照上有可重复且有实际意义的收益。
2. 关键机制有直接对照，知道收益来自什么，而不只是重新命名缓存。
3. 不能主要依赖减少运动、个别幸运 prompt 或一项饱和指标。
4. 有清晰的成本、失败场景及适用条件；不把开发集筛选当作独立确认。

v210 的 FIFO 弱修正有积极信号，但只有8条开发prompt。
v211 确认了 sink-local 上的部分恢复，**没有确认优于较强的 FIFO21**。
所以现在把“显著优于 SF”“身份长期保持已解决”“通用 head 分类”写成事实仍不合适。
这不等于要求每个指标都显著，也不等于方法没有继续验证的价值。

投稿目标可以指导实验取舍和表述重点，不能保证中稿；当前最重要的是补齐主张对应的证据。

## 2. 最新结果到底说明了什么

原始来源：
`artifacts/experiment_results/v211_generation_9a1c352b_eval_3093c028/`。
生成commit `9a1c352b`，评测commit `3093c028`。
32/32条主视频、36/36个core-9评测任务完成，原始provenance检查通过。
本地只有小文件，不能据此宣称重新核验了服务器MP4或看过所有视频。

| v211，8 prompts | Quality，DD固定 | Subject | Temporal组 | Overall consistency |
|---|---:|---:|---:|---:|
| SF FIFO21 | 85.28290 | .96893 | .98116 | .20959 |
| SF sink1+recent20 | 84.07838 | .95800 | .97710 | .21778 |
| sink-local + early1 .02 | 84.44567 | .96182 | .97718 | .21841 |
| sink-local + early1 .10 | 84.09040 | .96088 | .97511 | .21512 |

小强度 `.02`：

- 相对同结构 sink SF：Quality **+.367284**，paired bootstrap CI `[+.135023,+.659998]`。
- 相对 FIFO21：Quality **-.837227**，CI `[-1.129839,-.525660]`。
- 相对 sink SF，identity/background组 +.003243，visual组 +.007740；均有正向开发证据。
- 相对 FIFO21，subject/temporal/visual仍较低，semantic均值较高但区间跨0。
- 相对 sink，2/8条被运动规则标记：source **35、99**；相对 FIFO 是1/8条边缘密度标记。
  这是需要定位的自动告警，不等于已目视确认的崩坏，也不能因为整体均值好而删除。

`.10` 相对 sink 的Quality只增加+.012019，区间很宽；没有扩大强度有益的证据。
原始结论仍为 `stop_v211_no_eligible_lphc_candidate`，不重写旧门槛。
逐对照门槛拆解另存于`artifacts/experiment_results/v213_v211_posthoc_explanation.json`，
标记为事后解释，不改变原始判定。

综合v209/v210/v211：

- v209 32条上，FIFO21 的Quality也高于sink1_21；sink不是无代价的增强。
- v210 FIFO `.02` 对 FIFO21 是+.232029，但它与v211使用不同prompt和seed。
  **不能用两轮差值直接证明FIFO修正优于sink修正**；这个交互由v212同批次比较。
- v211的失败不只是门槛过严：相对FIFO的Quality确有负向开发证据。
- 但也不是所有历史修正失效：相对sink的小强度恢复是真实的观察结果。

注意：这里的Quality按已有官方归一化公式、将DD固定为1计算，是开发诊断量。
当前DD全为1，因此它与本轮原始Quality数值相同；这不能证明运动足够丰富。
core-9不等于完整官方Semantic/Total，subject consistency也不等于人物身份识别。

## 3. 当前最可行的方法假设

暂用描述性名称：**局部保留的受限历史修正（LPHC）**，不是已经成立的新颖性结论。

1. **保护局部生成路径。** 保留原生SF的FIFO21读取，不以历史帧挤占这21帧。
2. **分离长期历史的影响。** 对同一个Q，计算local以及local+最多4历史帧的attention输出差，
   把该差作为小幅修正，而不是直接替换local输出。
3. **限制修正强度和介入时机。** 每层每头RMS限幅；主候选只在每个AR块第一次去噪调用介入。
4. **历史选择。** 每层archive保存最多12个完整clean、pre-RoPE K/V帧；用V统计descriptor选历史，
   与local去重，块内固定选择。随机对照使用同样候选池、预算和强度。

实际公式为：

```text
y_local = Attention(Q, K_local, V_local)
y_aug   = Attention(Q, [K_local; K_history], [V_local; V_history])
delta   = y_aug - y_local
scale   = min(1, RMS(y_local) / max(RMS(delta), eps))
y_out   = y_local + alpha * scale * delta
```

RMS沿token和通道维计算，保留batch/head维。因此每次调用的修正RMS不超过
`alpha * RMS(y_local)`，**不代表整个视频累计扰动或像素误差有这个上界**。
alpha=0走精确native bypass，不执行额外attention或检索。

当前archive保留仍是首/末帧加确定性hash-priority，不是学到的身份bank，也不是运动检测器。
所有head都可用同一机制，没有已经验证的新二分类。历史读取4帧不等于总显存只多4帧：
还保存12帧archive以及临时张量，且启用调用会多做一次attention。

可发展的论文故事是：**长视频需要历史，但替代局部上下文与不受控地引入历史都有风险；
因此将“维持局部生成”与“从历史获得修正”分开。**
必须由SF21/SF25、正确/随机历史、不同phase/alpha的对照支撑，不能靠名称就宣称创新。
若随机历史同样有效，应缩小或删除“内容检索贡献”，而不是换指标掩盖。

## 4. 下一轮实验：v213

v212继续按docs/230执行；不要改已运行的checkout或用本轮门槛重新判v211。
v213不引入新的attention算子，只复用已经实现的e1/e2/full及alpha设置。

固定32条Qwen改写MovieGen：零基source=`3+4k`，与v212相同，
有效seed改为`21300+source_index`。120 latent -> 477帧，16FPS，约29.8秒。
这是**同prompt的第二seed复验**，不是32条从未见过的新prompt，也不是跨模型验证。

| 方法 | Local | 历史 | alpha | 去噪调用 | 要回答的问题 |
|---|---|---|---:|---|---|
| sf_fifo21 | FIFO21 | 无 | 0 | native | 较强原生SF |
| sf_fifo25 | FIFO25 | 无 | 0 | native | 增加局部容量是否已经足够 |
| fifo_correct | FIFO21 | 内容选择4 | .02 | 0 | v212主候选的第二seed |
| fifo_random | FIFO21 | 随机4 | .02 | 0 | 内容选择是否有额外收益 |
| fifo_e1_a010 | FIFO21 | 内容选择4 | .10 | 0 | FIFO中较强修正是否更有效 |
| fifo_e2_a002 | FIFO21 | 内容选择4 | .02 | 0、1 | 延长早期介入能否改善效果 |
| fifo_full_a002 | FIFO21 | 内容选择4 | .02 | 0、1、2、3 | 全程介入是否造成过度影响 |

总计224条主视频。gate0是source3/67各native与alpha0，共4条短数值轨迹。
smoke是source3的七方法30秒bundle，直接作为主实验结果复用，不另生成同seed同配置视频。
跨v212不复用视频，因为seed不同；这里重新生成SF是必要的配对对照，不是无意义重复。

**解释边界：**e1/e2/full同时改变介入时段、累计干预量和计算量。
因此该组能比较部署策略，不能单独证明“早期head具备唯一的长期记忆功能”。
SF25是容量参考而非严格总显存/FLOPs匹配。若`.10`成为新赢家，其检索贡献还需对应`.10`随机对照；
本轮`.02`随机对照不能替它证明这一点。

## 5. 六节点命令

启动补充：新checkout先按[docs/232](232_v213_sf_baseline_alignment_and_launch.md)
准备固定commit的干净官方SF，设置`UPSTREAM_SF_ROOT`，并在prepare后运行baseline。
这是数值实现检查，不增加主实验方法，不修改旧v212结果。

仍用原六节点白名单，无SSH自动启动：

| Rank | V213_NODE_ADDRESS |
|---:|---|
| 0 | 28.216.19.213 |
| 1 | 28.216.19.143 |
| 2 | 28.216.19.137 |
| 3 | 28.216.19.225 |
| 4 | 28.216.18.144 |
| 5 | 28.216.18.136 |

每个prompt的七方法在同一物理GPU串行执行，方法顺序轮换。
默认GPU_LIST=0..7时只使用32张卡，不声称48卡全满。
**不要在已有任务使用的GPU上并行启动。** 不同实验root的锁不提供跨实验GPU预约。
如果v212按默认分配运行且实际确认6/7号GPU空闲，可在prepare之前统一设置`GPU_LIST=6,7`，
让v213用12张卡跑多个prompt bundle；各节点都必须使用同样冻结的列表。
prepare之后不能改槽位列表，重跑同一prompt也不能换物理GPU。

```bash
# All nodes use the same immutable commit, separate from any running checkout.
git fetch origin
COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
git worktree add --detach /tmp/training-free-v213-${COMMIT:0:8} "$COMMIT"
cd /tmp/training-free-v213-${COMMIT:0:8}

export V213_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v213_runs/v213_${COMMIT:0:8}_sixnode
export NODE_RANK=0
export V213_NODE_ADDRESS=28.216.19.213
export GPU_LIST=0,1,2,3,4,5,6,7
# Set this to the already configured VBench checkout and caches on the server.
export VBENCH_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench

# Rank0 only:
bash scripts/run_v213_experiment.sh prepare
bash scripts/run_v213_experiment.sh baseline
bash scripts/run_v213_experiment.sh gate0
bash scripts/run_v213_experiment.sh smoke

# All nodes, with their own rank/IP, after smoke completes:
bash scripts/run_v213_experiment.sh generate32
bash scripts/run_v213_experiment.sh status

# Rank0 after all generation completes:
bash scripts/run_v213_experiment.sh publish
# All nodes; wait for all splits before evaluation:
bash scripts/run_v213_experiment.sh split
bash scripts/run_v213_experiment.sh preflight
bash scripts/run_v213_experiment.sh eval
# Interrupted evaluation only:
bash scripts/run_v213_experiment.sh eval-missing
# Rank0 after evaluation:
bash scripts/run_v213_experiment.sh collect
bash scripts/run_v213_experiment.sh analyze
bash scripts/run_v213_experiment.sh package
```

模型、环境、Qwen文件沿用v212。`V213_SOURCE_PROMPTS/SHARED_CHECKPOINT/WAN_MODEL`可在prepare前覆盖；
评测缓存用`VBENCH_CACHE_DIR/TORCH_HUB_DIR/VBENCH_RUNTIME_HOME`指向已有位置，避免重复下载。
中断后重跑同一命令，验证完成的job跳过；异常job隔离并保留日志。
v213 wrapper显式选择campaign；v212默认仍为原方案，不能共享输出root。

## 6. 自动分析和最小人工工作

主要回传：

```text
inputs/manifest.json
decisions/sf_upstream_gate.json
decisions/gate0.json
jobs/screen32/<method>/source_<id>/done.json
jobs/screen32/<method>/source_<id>/trace.jsonl
evaluation/metrics/vbench_core9_summary.json
evaluation/metrics/temporal_diagnostics.csv
evaluation/analysis/v213_seed_phase.json
evaluation/analysis/v213_seed_phase.md
evaluation/analysis/v213_seed_phase.csv
v213_small_artifacts.tar.gz
```

- full/early/late按prompt配对；给出均值、CI、胜率、负向prompt数量。
- 新增leave-one-prompt-out最小均值，显示正收益是否可能被单个样本撑起；不是新的晋级门槛。
- full主指标保留4个候选的单侧sign-test与BH校正，不把逐帧/逐clip当独立样本。
- `.02/e1`是预先指定的复验候选，其他三种是探索变体，不能事后都称预注册确认。
- 所有NI项目分为支持非劣、支持退化、证据不确定，不把后者写成确定失败。
- 沿用v212开发均值规则：Quality +.10、full/late组均值在容差内、运动guard通过，
  只建议进入更大规模确认，**不会输出paper_claim_ready=true**。这些容差是工程开发阈值，非理论常数。
- 保存每层每去噪phase的历史非空调用数与修正幅度；trace仍验证预算、完整调用、去重及phase边界。
- 汇总同GPU端到端耗时比与采样显存，不冒充纯DiT吞吐，不据此挑选“效果赢家”。
- `review_queue`最多4组异常配对视频，以告警数量、较差质量差值排序并去重prompt；完整告警仍保留。
  不要求盲审整套视频，也不自动挑最好看的展示样例。无告警时不强制人工review。

v212完成后，可追加第二seed汇总，无需生成视频：

```bash
python scripts/analyze_v213_lphc.py --run-root "$V213_OUT_ROOT" \
  --v212-report "$V212_OUT_ROOT/evaluation/analysis/v212_matched_history.json" \
  --v212-comparison "$V212_OUT_ROOT/evaluation/vbench_comparison/comparison_manifest.json" \
  --v212-inputs "$V212_OUT_ROOT/inputs/manifest.json"
bash scripts/run_v213_experiment.sh package
```

另写`v213_seed_phase_with_replication.*`，不覆盖第一次报告。
核对prompt、seed、模型、推理代码、配置和评测fingerprint；允许实验脚本/日志代码增加，
不允许模型算子变化后仍冒称同方法seed复验。小文件hash链不等于重新访问旧服务器视频。
**两seed仍是32个prompt cluster，不是64条独立prompt。** CI先对同prompt的两个seed取均值，再重采样prompt。

## 7. 如何尽快收束

| 结果 | 下一步 | 论文主张边界 |
|---|---|---|
| v212/v213 `.02`都稳定改善FIFO | 冻结方法，128 prompts 30s主实验；优先分析开发32之外的96；再做固定32条60s | 可以围绕相对SF的稳定收益写，不要求SOTA |
| `.10/e2/full`更好 | 冻结最多一个新变体，补其强度/phase匹配的随机历史和第二seed | 新变体不能借用`.02`机制证据 |
| 正确与随机历史相近 | 简化为受限历史修正，决定是否删除内容选择主张 | 不能宣称有效检索器 |
| 只胜过sink，不胜过FIFO | 不提升为主方法；转向改进原生FIFO | 不能挑较弱SF来构造主要收益 |
| FIFO25效果相当或更好 | 先比较成本及60s长时稳定性，再决定保留方法 | 不能仅以历史模块名字解释必要性 |
| 整体仍无收益，或靠运动下降 | 暂停扩成128或写确定性效果结论，定位少量失败样本后重新设计 | 不扩大无效搜索、不调整报告以隐藏负结果 |

128/60s与跨模型验证在此文是后续计划，不是本次已经交付或已经运行的结果。
128中开发32之外的96也有历史研究使用记录，不能称为从未见过的benchmark。
ABA仍后置，PF不新增生成；先把SF上的核心效果与机制做清楚。

## 8. 实现与检查范围

新增v213冻结配置、runner入口、分析器和测试；复用v212执行/发布/评测路径，通过显式campaign参数隔离。
没有修改LPHC数学公式、SF模型前向、既有v210/v211结果和原判定。
同步的v211修正将alpha0门禁依据从MP4容器hash改为pre-VAE轨迹张量精确一致，
视频仍做有效性验证；这是合理地区分容器封装差异与数值轨迹差异，不是放宽数值相等标准。

本地仅运行无PyTorch测试、Python编译及Bash语法检查。实际GPU、FlashAttention、VAE和VBench需服务器执行。
本次可运行的协议与回归测试为 **80 passed, 1 skipped**。
新增同步的`test_v211_postprocessing.py`依赖Linux的`fcntl`，在Windows无法收集，未声称本地通过。
建议服务器先运行：

```bash
python -m pytest -q tests/test_v213_lphc.py tests/test_v212_lphc.py \
  tests/test_v211_postprocessing.py tests/test_lphc.py tests/test_lphc_native_integration.py
```

新checkout要求baseline、gate0与完整30秒smoke通过后再批量运行；
无法在本机验证的部分不宣称“绝不会出错”。补充检查与命令见docs/232。
