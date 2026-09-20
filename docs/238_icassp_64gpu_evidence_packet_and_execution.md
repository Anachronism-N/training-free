# ICASSP收束：64卡执行顺序与自动论文证据包

日期：2026-09-20。承接docs/235、236、237；本轮不新增生成方法、不重跑已有视频。

## 1. 实际拉取状态与写作判断

本次执行`git fetch origin`并核对远端ref，`worktree-v210-lphc`仍是`5bd33b67`，
即v213结果；当前实现分支已经包含它，合并返回`Already up to date`。
没有看到v215/v216/v217的新结果。实现分支已有v216和v217，不再重复编写另一个同类生成campaign。
目前代码继续推送到`codex/v178-v179-causal-validation`，未把结果分支强行覆盖成实现分支。

**可以进入四页论文的结构、方法与图表准备，不必等待所有可选实验。**
但尚不能根据未上传结果确认赢家。最新可分析的v213仍是：

- 224条视频完成，官方SF与alpha0对照通过。
- full-phase官方Quality相对SF均值+0.133分，95% CI[-0.073,+0.340]。
- 检索相对随机历史的优势仍不明确。

不要求所有指标提升，也不把CI跨零自动视为不能投稿；把已有信号写成初步观察即可。
若v215显示后半段主体一致性有改善，就在开发阶段选定这一有限主张，冻结后确认，
同时披露全段质量与运动，不必硬写“总体质量领先”。
四页的核心可继续是“保留局部生成、以受限历史差值修正长程生成”；
不需要把head分类、检索、phase、clip全部列成已证实贡献。

官方提交门户本次仍显示北京时间**9月24日20:00**截止；内部以9月23日上传为目标。
来源：[ICASSP 2027提交门户](https://cmsworkshops.com/ICASSP2027/papers.php)。
篇幅、作者及AI辅助政策见docs/235，不把四页误认为不需要正确基线的低标准通道。

## 2. 八节点、每节点八卡的确定安排

| 顺序 | 方法与prompt | 新主视频数 | 每节点工作 | 是否必须 |
|---|---|---:|---|---|
| 当前 | 完成v215，48条、七方法 | 不重复完成项 | 沿用原六节点冻结配置 | 选型所需 |
| P0：v216 | SF与一个冻结方法，80条30秒 | 160 | 每节点10个双方法bundle、20条视频 | 主要确认 |
| P1：v217 | SF/同一方法/匹配random，64条，新seed | 192 | 每节点8个三方法bundle、24条视频 | 有时间再做 |
| 不默认新增 | 60秒、PF、ABA、跨模型、新head分类 | 0 | - | 不阻塞四页写作 |

v216的80条是128条中排除v215开发48条后的集合；不是历史从未见过的新数据。
布局为64个GPU槽位，其中16个槽位执行两条prompt，其余48个执行一条。
v217布局恰好64个槽位各一条prompt。每个prompt的方法在同一卡串行，顺序轮换；不是单视频DDP。
smoke已完成的bundle在主生成中跳过，不承诺始终64卡100%占用。

先完成P0生成和评测，再启动P1，不让不同root的任务抢同一张卡。
按旧轮每条30秒视频约25至28分钟粗估，v216最忙卡约四次推理、v217约三次，
但不包含前置检查、VBench、加载和共享存储等待，不作为完成时限保证。
评测按方法/维度分发，v216有18个、v217有27个维度任务，不为填满64卡改写评测算法。

### 节点信息还需要什么

`configs/v216_nodes.example.json`有原六个IP，最后两个为空。
需要填八台实际网络接口IP，若原六台也变了就全量更新，按rank0..7排序。
不能猜新IP或使用相同hostname代替。代码会在freeze检查八个唯一IP，并在各节点检查真实网卡归属。
IP清单放共享输出/配置目录，不改正在运行的v215工作树。

## 3. 本轮补的代码

新增`scripts/export_lphc_paper_evidence.py`，不需要GPU、Torch或重新解码视频。
它读取已完成的compact artifacts，支持三个阶段：

1. **只有v215**：列出四个候选对SF、对匹配random的综合Quality/后半段subject两种开发证据，不自动指定赢家。
2. **加v216**：输出80条确认主表，以及单独标注含开发数据的128条加权表，不把48条和80条各占一半。
3. **再加v217**：分别列SF效应与random机制对照，并在共同64条上计算两seed的prompt级统计。

两seed分析先对同一prompt的两次差值取平均，再对64个prompt做bootstrap；
不能把v216的80条加v217的64条当144个独立prompt，也不能把64条两seed叫128个独立prompt。
v216剩余16条不进入这张双seed表，仍完整保留在80条主结果中。
该分析只涉及这两个seed，不证明对所有随机种子均稳定；次级指标保留描述性定位。

新工具核对汇总、输入、comparison、完成receipt、selection和跨轮评测/模型一致性，
检查逐prompt数据有限、均值对应、Quality与原始分项一致。
但它**没有重新核验媒体内容、原始tensor或原始VBench计算**，不会把小文件校验写成全面GPU验证。
Windows Git导致的CRLF只在能还原原SHA时记录为传输换行，不改任何数值。

另修正`prepare_v216_confirmation.py`：冻结的主要指标若含NaN/Inf/非数值，明确拒绝，
不能因为长度是48就当作可分析的完整结果。这是输入有效性检查，不是提高投稿效果门槛。

## 4. 现在可以做什么

v215保持原checkout继续运行，不pull。以下工具在新的分析checkout运行即可，不改变原输出：

```bash
git fetch origin
export PAPER_COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
git worktree add --detach /tmp/training-free-paper-${PAPER_COMMIT:0:8} "$PAPER_COMMIT"
cd /tmp/training-free-paper-${PAPER_COMMIT:0:8}
```

完整v215结果ready后，在有NumPy/PyYAML的已配置Python环境中：

```bash
python scripts/export_lphc_paper_evidence.py \
  --v215-root "$V216_V215_ROOT" \
  --output-root "$PAPER_EVIDENCE_ROOT/v215_development"
```

`V216_V215_ROOT`是完整v215 root，`PAPER_EVIDENCE_ROOT`是新的共享分析输出目录，需设置实际路径。
没有完整结果时工具会报缺失，而不是填0或自动忽略失败方法。
开发brief回来后选定一个候选和一个主要指标/窗口，记录取舍，再按docs/236执行v216：

```bash
# Set V216_OUT_ROOT, V216_V215_ROOT, V216_NODES_FILE, candidate,
# endpoint, rationale, model paths and per-node identity as in docs/236.
# rank0
bash scripts/run_v216_experiment.sh freeze
bash scripts/run_v216_experiment.sh prepare
bash scripts/run_v216_experiment.sh schedule
bash scripts/run_v216_experiment.sh baseline
bash scripts/run_v216_experiment.sh gate0
bash scripts/run_v216_experiment.sh smoke
# Each of the eight nodes, after the checks pass:
bash scripts/run_v216_experiment.sh generate80
```

按docs/236执行publish/split/preflight/eval/collect/analyze。
优先回传v216，不为等可选v217推迟主要结果分析。
v217全部命令见docs/237，它继承v216选择，不再选新赢家。

## 5. 自动形成表格与最小review队列

v216完成后：

```bash
python scripts/export_lphc_paper_evidence.py \
  --v215-root "$V216_V215_ROOT" --v216-root "$V216_OUT_ROOT" \
  --output-root "$PAPER_EVIDENCE_ROOT/v216_confirmation"
```

若也完成v217，则增加其root，写到新目录：

```bash
python scripts/export_lphc_paper_evidence.py \
  --v215-root "$V216_V215_ROOT" --v216-root "$V216_OUT_ROOT" \
  --v217-root "$V217_OUT_ROOT" \
  --output-root "$PAPER_EVIDENCE_ROOT/v216_v217_joint"
```

主要产物：

- `evidence.md/json`：开发与确认分开、选定主指标、SF/random效应、双seed表及边界。
- `main_tables.csv`：原始core-9及官方Quality，保留全部方法，不只导出获胜分项。
- `development_options.csv`：选型依据，允许判断局部收益，但不追溯改写v215的预设主指标。
- `review_queue.json`：总计最多6组，优先风险，其次中位附近，再考虑极端差值。
- `review_queue_ids.json`：本次排队的ID，**不表示已经完成review**。

已经看过的pair可将其ID记录为JSON字符串数组，通过`--reviewed-ids 实际文件路径`传入。
脚本从6组总预算扣除已review数量，而不是每次再要求6组；不重复已完成pair。
ID绑定comparison、方法对和source，新seed的视频不会误当作旧视频。
这种review用于排错与定性解释，不据此报告用户偏好胜率。

输出只写新文件或完全相同内容，不覆盖上一次brief。不同阶段用不同分析目录。
返还这些小文件即可；无需为了分析上传几百条MP4。
若在本机分析从Git拉取的compact结果，完整receipt等小文件需同时存在；不能只给一张平均分表。

## 6. 四页文章的最终取舍

- 主要确认有明确收益：聚焦该项收益写方法文章，运动/画质/显存同时说明，不要求全分项或SOTA。
- SF效应好但random一样好：讲局部保留的历史修正，不把语义检索写成已证实贡献。
- 只有开发集信号、确认不明确：限定为初步结果；不因为想投稿就改统计单位或隐藏确认结果。
- 没有方法赢家：仍可准备方法和实验结构，不能假装已确认head分类或整体领先。

主文只需一个核心公式、一张主表、少量匹配消融与定性示例。
v217不成为新硬门槛，60秒/PF/ABA不默认追加。
作者负责正文撰写与核查，工具提供结构、代码与数据辅助，按docs/235中的会议规则执行。

## 7. 验证边界

新工具用合成fixture测试，不把这些测试数据导出为实验结果。
覆盖原始分项与Quality、48/80加权、同64条跨seed、来源/方法不一致拒绝、NaN拒绝、
Windows换行、全局6组review及已看视频跳过。
本轮未改变模型、cache、RoPE或生成算子；不需要因此重生成v215。
本机没有Torch/GPU，服务器仍需执行原有Torch与数值对齐检查。
本轮相关回归为**174 passed, 3 skipped**，skip为缺少Torch；新代码Python编译检查通过。
没有在本机生成视频或运行实际VBench，不把合成测试的正负效应当成方法结论。
