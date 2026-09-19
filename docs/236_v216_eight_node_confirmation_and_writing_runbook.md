# v216：八节点64卡、冻结方法确认与四页写作

## 1. 三小时后同步的实际情况

按用户要求等待后，于北京时间2026-09-20约05:42执行`git fetch origin`。
远程仍为主工作分支`850086c6`、结果分支`worktree-v210-lphc`的`5bd33b67`，没有新增v215结果。
不能把未上传理解为运行失败，也不能把本轮代码交付说成v215有效性的证明。
最新完整实验仍是v213：224条完成，SF/alpha0检查通过，但总体收益与检索优势仍不确定。
具体数值和复核边界见docs/234，四页内容组织见docs/235。

**写作判断：现在可开始结构、公式说明、图表模板和已确认事实的整理；不必等所有指标胜出。
积极效果的主张应等v215确定一个方向，再以v216确认这一项收益。**
不把`paper_claim_ready=false`或所有旧开发gate作为投稿资格门槛。
如果只改善后段主体一致性，就限定主张并公开运动/画质代价，不包装成普遍优于SF。

2026-09-20再次核对[官方提交门户](https://cmsworkshops.com/ICASSP2027/papers.php)：
北京时间截止仍列为9月24日20:00，内部目标9月23日完成上传；不等待继续延期。
作者负责撰写和核对正文，AI用于结构讨论、代码/图表辅助与编辑，遵守该页的使用规则。

## 2. 已实现什么

本轮不修改SF模型、历史修正算子或head descriptor，只实现选型之后的实验闭环：

- 读取已完成v215的manifest、配对report、summary与全部336个completion receipt。
- 人工选一个候选并记录理由，不由脚本自动挑“最好看的”指标或视频。
- 冻结候选、一个主要指标/时间窗、八个节点IP、模型/配置/源码与评测版本。
- 补齐不在v215选型中的80条MovieGen prompt，只生成SF和冻结候选，160条30秒视频。
- 八节点生成、切片、评测、重试、统计、至多6组人工检查和小文件打包。
- 输出独立的80条确认结果，以及包含48条开发数据的128条汇总，标签严格区分。

**代码可执行，但当前尚不能填写最终候选。需要v215完整报告，以及新增两节点的真实接口IP。**
`freeze`只冻结选择，不占GPU；`prepare`校验输入，数值检查完成后才能批量生成。
原v215继续使用原checkout、六节点入口和原输出目录，不改成八节点、不重生成。

## 3. 范围与配对

| 项目 | v216 |
|---|---|
| Prompt | 128条Qwen改写MovieGen中，排除v215的48条后的80条 |
| 视频 | 每条prompt原生SF FIFO21与一个方法，共160条 |
| Seed | `21500+source_index`，与v215相同规则 |
| 长度/分辨率 | 120 latent frames，477 decoded frames，16FPS，约29.8秒，480×832 |
| 方法名 | `sf_fifo21`、`ours_correct`；后者准确指向selection.json中的v215配置 |
| 不做 | 新head分类、PF、ABA、跨模型或60秒大矩阵 |

可选候选仅为`fifo_correct`、`headwise_correct`、`centered_correct`、`fifo_full_a002`。
alpha/phase/descriptor/cache预算从原v215配置继承，不提供确认阶段继续调参入口。

冻结时二选一主要假设：

1. `official_quality_score / full`：综合Quality改善。
2. `subject_consistency / late_half`：后段主体一致性改善，不等于人物身份识别准确率。

其余指标保留描述性报告，不根据80条结果反复更换主指标。两个选项不能同时算各自独立主检验。
late_half沿用15段clip中的后8段，不声称刚好后15秒。
置信区间跨零时仍可报告初步观察，不自动拒绝写作，也不使用“统计显著提升”的措辞。

## 4. 64卡任务分配

按80条有序source的序号j分配：`node=j%8, gpu=(j//8)%8`。
每节点10条prompt、20条视频，GPU0/1各两个prompt bundle，GPU2至7各一个。
同prompt两方法在同一物理卡串行；每节点恰好5条SF先跑、5条方法先跑。
合计64个槽位，16个槽位各4条视频、48个槽位各2条视频。smoke已完成的一个bundle直接复用。

这是任务并行，不是单视频64卡DDP。恢复时不换物理GPU，已验证任务自动跳过。
不同输出root的锁不是全局GPU调度器，不在仍占用的节点上重叠启动新实验。
按上一轮约25至28分钟/视频估计，主要生成约两波/四次推理的量级，实际仍受共享负载影响。

VBench目前按`2 methods × 9 dimensions = 18`个任务分配到八节点，
所以评测不保证64卡全部忙。为填满卡而拆分/改变评测实现不属于本轮范围。

节点按`NODE_RANK=0..7`排序，模板在`configs/v216_nodes.example.json`。
前六项是此前使用的地址，后两项留空，**不是可直接启动的完整清单**。
若前六节点也变化，应一并更新。清单必须是八个互不重复的接口IP；不接受空值、主机名或回环地址。
运行时还会核对地址是否存在于该机器实际网络接口，不能只设置一个假的环境变量通过。

## 5. 独立checkout与环境

在现有仓库fetch一次并记录commit，八节点使用同一个完整值，不更新仍运行的v215：

```bash
git fetch origin
export V216_COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
echo "$V216_COMMIT"
```

每节点自己的仓库创建独立worktree：

```bash
git fetch origin
git worktree add --detach /tmp/training-free-v216-${V216_COMMIT:0:8} "$V216_COMMIT"
cd /tmp/training-free-v216-${V216_COMMIT:0:8}
export V216_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v216_runs/v216_${V216_COMMIT:0:8}_confirm80
export GPU_LIST=0,1,2,3,4,5,6,7
export V216_SOURCE_PROMPTS=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt
export SHARED_CHECKPOINT=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt
export WAN_MODEL=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B
export VBENCH_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3
```

设置各自`NODE_RANK`及`V216_NODE_ADDRESS`，例如原rank0未变化时：

```bash
export NODE_RANK=0
export V216_NODE_ADDRESS=28.216.19.213
```

v216要求模型、推理配置及`src/`和`third_party/Self-Forcing/`的实际源码与v215相同。
本轮只改实验脚本所以可满足；若服务器私自改过模型算子，需解释并重新设计实验，不能绕过检查。

## 6. v215完成后，rank0冻结

先把以下变量设置为实际值；候选不设默认赢家，节点清单位于共享路径：

```text
V216_V215_ROOT     已完成且执行过collect/analyze的v215原始输出根目录
V216_NODES_FILE    已填完整的八IP JSON数组文件路径
V216_CANDIDATE     从四个候选中选一个
V216_RATIONALE     选择原因、依据的v215指标与已知取舍
```

原始服务器输出是证据来源，不用Git换行转换后的零散文件冒充原始hash链。
主要指标默认官方Quality；若选择后段主体一致性，在freeze之前明确设置：

```bash
# Only for the late-subject hypothesis; omit both for full official Quality.
export V216_PRIMARY_METRIC=subject_consistency
export V216_PRIMARY_WINDOW=late_half
```

随后rank0执行：

```bash
bash scripts/run_v216_experiment.sh freeze
bash scripts/run_v216_experiment.sh prepare
bash scripts/run_v216_experiment.sh schedule
```

冻结产物包括`inputs/selection.json`、`inputs/development/`的四个小文件快照、
`inputs/manifest.json`与`inputs/placement.json`。其他节点从同一共享输出读取，无需重复选择候选。
冻结后改候选、指标或节点会失败；不能在看到确认结果后换一个名字继续称独立确认。

rank0在longlive环境运行CPU-Torch检查，再做短轨迹检查和复用型smoke：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
python -m pytest -q tests/test_lphc.py tests/test_lphc_native_integration.py \
  tests/test_lphc_head_descriptors.py tests/test_v216_confirmation.py
bash scripts/run_v216_experiment.sh baseline
bash scripts/run_v216_experiment.sh gate0
bash scripts/run_v216_experiment.sh smoke
```

baseline在source1/65上核对官方/本地/官方重复；gate0对同两条验证alpha0；
smoke为source1的完整双方法30秒，计入160条。检查不需要新增人工短视频review。

## 7. 八节点生成与评测

前置通过后，八节点各运行一次：

```bash
bash scripts/run_v216_experiment.sh generate80
```

结束后rank0检查两方法各`80/80 validated`并发布：

```bash
bash scripts/run_v216_experiment.sh status
bash scripts/run_v216_experiment.sh publish
```

发布要求VBench fingerprint与v215一致。不要在期间更新评测仓库；
若确实修复评测，应统一重评而不是混合两个口径，视频可以复用。

八节点分别执行split；全部完成后再分别preflight/eval：

```bash
bash scripts/run_v216_experiment.sh split
# Wait for split completion on all eight nodes.
bash scripts/run_v216_experiment.sh preflight
bash scripts/run_v216_experiment.sh eval
```

中断可用`eval-missing`。全部评测完成后，rank0执行：

```bash
bash scripts/run_v216_experiment.sh collect
bash scripts/run_v216_experiment.sh analyze
bash scripts/run_v216_experiment.sh package
```

## 8. 结果怎么看、回传什么

运行日志位于`jobs/confirm80/<method>/source_<id>/`：
stdout/stderr、trace.jsonl、done.json，继承v215的排名/选帧/RMS/阶段诊断。
新增选择合同记录原候选与主要假设；日志打印source、seed、GPU、耗时、显存和skip状态。
缓存错误仍阻断；自动运动告警只要求定向检查，不直接判定整种方法失败。

回传`v216_small_artifacts.tar.gz`。主要分析产物：

- `evaluation/analysis/v216_confirmation.json`与`.md`：80条主要确认与所有取舍；
- `v216_confirmation80.csv`：逐指标/时间窗的配对均值、CI、胜率等；
- `v216_selection_included128.json`：48条开发加80条确认，以样本数加权，不把两组均值各占一半；
- 主要指标最多6组检查：高风险2组、中位数附近2组、最大改善/退化各1组，重复source合并；
- full/early/late、运动告警、时间/显存与完整来源合同保留，不伪造正式偏好研究。

128条汇总使用旧48条的逐prompt统计，不重新生成/复制其视频；明确`selection_included=true`。
core-9不是完整官方Semantic/Total；Quality是百分点，其他分项维持原始量纲。
论文可以正文给80条确认主表、128条作为有标注的补充，或同表分面，不能统称128条独立测试。

## 9. 何时进入写作收束

只需围绕一个有用收益形成有限主张，不要求所有指标都显著。
如果选定主指标改善、运动/画质取舍可接受，立即收束到四页写作；
若区间较宽，标为初步观察，不用一句“未过gate”否定所有研究价值。
若主张对应的80条表现反向，不能继续用开发集最好的几条视频支撑同样结论。

本轮默认不增加新seed、60秒或去clip消融。v215本身已有selector/random/phase对照，
先复用；只有最后的论文核心主张确实缺证据时再补一项小消融，不再为了占满64卡添加方法。

本地只进行纯Python协议/分析回归和语法检查；没有GPU推理、Torch前向或实际VBench验证。
本次回归结果为151 passed、3 skipped（缺少Torch）；Python编译与两个Shell入口语法检查通过。
正确性检查保留，效果筛选门槛不冒充会议录用要求。测试覆盖64槽位均衡、旧六节点兼容、
输入/选型/模型/配置漂移、实际接口授权、80条主指标与128条加权汇总、CLI输出和评测参数。
