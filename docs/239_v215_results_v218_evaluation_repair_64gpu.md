# v215结果、v218评测修复与ICASSP四页收束

## 1. 当前判断

2026-09-20同步了`origin/worktree-v210-lphc`的`a988f0e6`，补齐v215分数。
结果目录：`artifacts/experiment_results/v215_generation_850086c6_eval_45e79ec1`。
336条视频完成，官方SF对齐和alpha0检查通过，方法trace审计通过。
`850086c6`是生成代码，`45e79ec1`是VBench版本，两者不能混淆。
当前没有新上传的v216/v217结果，不根据缺少上传推断运行失败。

**可以立即开始四页论文的结构、方法说明、图表准备；效果结论尚需最后一轮确认。**
不要求SOTA，不要求每个指标都胜出，也不以所有开发gate通过为投稿条件。
但不能把不可信的评测、开发集选型或微小波动写成确定的全面提升。
本轮不扩展head分类、不增加PF/ABA/60秒矩阵，优先修复评测并确认一个有限收益。

推荐暂定方法：`headwise_correct`，对应**保留head对应关系的检索 + phase0小幅历史修正**。
推荐新确认主指标：`imaging_quality/full`。这是看过开发集后的选择，
不是v215原先预设的主指标；v215仍保留原分析口径，新主指标仅在v216生成之前冻结。
如果v216已经冻结或开始运行，不覆盖其selection，不事后把新指标改称预设主指标。

## 2. 结果及边界

48条Qwen改写MovieGen prompt，约30秒，同prompt/seed/物理GPU配对。
下面排除已发现实现问题的Dynamic Degree；固定DD的Quality只是诊断，不是官方成绩。

| 方法 | 固定DD诊断Quality | Subject | Background | Imaging | Aesthetic |
|---|---:|---:|---:|---:|---:|
| SF FIFO21 | 85.58510 | .969681 | .962116 | .680141 | .590344 |
| pooled/e1/.02 | 85.53780 | .969009 | .961260 | .678883 | .588695 |
| random/e1/.02 | 85.55109 | .969800 | .961740 | .680060 | .590133 |
| **headwise/e1/.02** | **85.68704** | .969860 | .961557 | **.684518** | .590827 |
| centered/e1/.02 | 85.57511 | .969576 | .961481 | .681515 | .590071 |
| pooled/full/.02 | 85.52895 | .969098 | .962020 | .681679 | .589170 |
| random/full/.02 | 85.70291 | .971010 | .962444 | .683769 | .591472 |

- Headwise相对SF的Imaging均值差约`+.004377`，换成百分制是`+0.438`百分点。
- 固定DD诊断Quality差`+0.101944`，配对95% CI约`[-0.048821,+0.269554]`，不能称统计显著。
- 旧report没有单独Imaging的逐prompt区间。本轮补算该项，不把组合visual指标的区间冒充它。
- Subject变化很小，Background下降，不支持“身份保持显著优于SF”。Subject也不是人物身份识别准确率。
- Full-random在部分指标上更好，必须保留在开发结果和机制讨论里；不能只展示内容检索胜出的列。
- Headwise有7/48个自动时序代理告警。告警不是已确认的视频失败，不能直接说7条不可用。

可复核分析：`artifacts/experiment_results/v218_planning/v215_results_review.{json,md}`。
本地核对小文件hash关联、完成receipt、配对统计和实际评测源码，没有本地重新推理/重算VBench。

## 3. 发现的评测问题

上传的`evaluation/vbench_runtime_provenance.json`记录了实际修改版DD源码：

1. 优先调用`torchvision.models.optical_flow.raft_large(weights='DEFAULT')`。
2. 仍将VBench读入的RGB float `0..255`直接传入，没有应用该权重要求的归一化。
3. 异常回退路径使用另一套原生RAFT并以`strict=False`加载权重。

[Torchvision官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)
要求其权重配套预处理将输入缩放到`[-1,1]`；DEFAULT还不是VBench原生的固定checkpoint。
[固定版本VBench DD](https://raw.githubusercontent.com/Vchitect/VBench/45e79ec14e69a2187202c675d2dbce1a71843d53/vbench/dynamic_degree.py)
使用原生RAFT、20次迭代和原本阈值；[原生RAFT.forward](https://raw.githubusercontent.com/Vchitect/VBench/45e79ec14e69a2187202c675d2dbce1a71843d53/vbench/third_party/RAFT/core/raft.py)
在内部进行`2*(x/255)-1`。两种backend不能只替换模型而沿用同一输入处理。

因此，当前DD与由它参与计算的官方公式Quality不宜用于论文定论。
这不意味着全部非DD指标都失效，也不证明生成代码有bug，更不保证修复后我们会变好。
现有日志不能确定每次究竟进入哪个分支；记录的checkpoint路径只是声明，不等于实际加载证明。
不能进一步断言旧实验一定加载了错误权重。新日志会记录实际模型类、权重SHA和strict加载结果。

代码只拦截**已经确认有问题的源码SHA**，不是把所有dirty评测仓库都判无效，
也不是自动认证其他任意SHA正确。旧结果不删除；放入独立新目录统一重评七方法。
原v213若有相同源码SHA，其含DD结论同样应暂缓，不能与新口径直接混合。

## 4. 方法故事压缩为一条主线

**Local-preserving history correction：保留模型可靠的局部生成路径，只用历史提供有界的辅助修正。**

1. **局部路径不被历史替换。** 原生SF FIFO21、sink0；21 FFE包含当前3帧块。
   存最多12帧clean历史K/V，排除与local重复帧，选最多4帧形成额外读出。
   使用既有first/last加确定性hash保留，不把这个archive规则重新声称为本轮发明。
2. **保留head对应关系的内容检索。** 每层每head分别计算clean V的空间均值/标准差，
   与上一已完成块比较，再平均head cosine分数选帧；避免先把不同head混为一个描述符。
   每层最后选同一组帧供该层heads读取，**不是将head分成类别，也没有证明功能性head分类**。
3. **限制历史干预的幅度和阶段。** 计算`delta = attention(local+history)-attention(local)`，
   按head用local RMS限制delta后乘`.02`加入原输出；只在去噪phase0使用，clean refresh不修正。
   因而校正量RMS不超过对应local输出约2%，是算子约束，不是保证视频质量的理论证明。

写作可突出“局部生成保真与历史信息利用之间的冲突”，把head-preserving descriptor作为实用实现贡献。
暂时不能写“检索已证明优于随机”“新head分类”“解决身份遗忘”“不损伤运动”或“全面超过SF”。
缓存、检索和RMS限幅各自不是天然全新概念；新颖性要说明组合方式、具体读出与phase约束，
并引用对应先前工作，不能仅改名声称首创。

四页建议：约0.75页问题/相关工作，1.25页方法/图，1.5页实验，0.5页局限与结论。
正文主表SF/Ours及关键取舍，开发消融压缩为一张小表；不堆砌全部历史失败轮次。
匹配random若持平，就把“内容选择优势”降为待验证假设，不隐藏该控制或把它变成无关实验。

## 5. 八节点64卡实验优先级

| 批次 | 内容 | 新生成视频 | 分配 | 解决的问题 |
|---|---|---:|---|---|
| P0 / v218 | 已有7方法×48视频，重新评core-9 | **0** | 63个method×metric任务分到8节点63个槽位 | 修正DD，补逐prompt Imaging，统一评测 |
| P1 / v216 | 冻结headwise vs SF，剩余80 prompt | **160** | 每节点10个双方法bundle，共20视频 | 确认有限画质收益；与开发48组成有标注的128覆盖 |
| P2 / v217 | 同一方法/SF/匹配random，固定64 prompt、新seed | **192** | 每节点8个三方法bundle，共24视频 | seed稳定性、随机历史是否足够 |
| 暂缓 | 新候选、分类、PF、ABA、60秒、跨模型 | 0 | - | 避免投稿前重新扩大研究问题 |

P0按dimension-major顺序：`node=i%8, gpu=i//8`，共63项，node0-6各8项、node7有7项。
不同指标耗时不同，不保证全程63卡满载。P1/P2使用同prompt同GPU串行配对，不是单视频64卡DDP。
P1评测18项、P2评测27项，允许空闲；不为利用率擅自拆分指标或改变聚合规则。
先完成P0，再冻结P1。P2在P1选择固定后即可准备，但不要与占用相同GPU的任务重叠。
P1已有足够局部证据时即可写作，不把P2设成投稿硬门槛。

不再新增几十种trick。若只剩有限时间，优先顺序为**可信主表 > 匹配random > 新seed > 更长/跨模型**。
截止前即便未完全占优，也可报告有限收益与代价；不能承诺录用或将区间跨零称显著。

## 6. P0启动：不改仍在运行的目录

以下在Linux服务器执行。本机没有推理环境，未执行GPU部分。
八节点共用同一个新commit和共享输出根。**不要在旧v215/v216/v217运行checkout直接pull。**

```bash
git fetch origin
export EXP_COMMIT=$(git rev-parse origin/worktree-v210-lphc)
echo "$EXP_COMMIT"  # Record once; use this exact value on all eight nodes.
git worktree add --detach /tmp/training-free-v218-${EXP_COMMIT:0:8} "$EXP_COMMIT"
cd /tmp/training-free-v218-${EXP_COMMIT:0:8}
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"

export V218_SOURCE_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v215_runs/v215_850086c6_sixnode
export V218_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v218_runs/v215_${EXP_COMMIT:0:8}_v218_eval
export VBENCH_ROOT=/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench
export V218_VBENCH_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v218_runtime/VBench_${EXP_COMMIT:0:8}
export VBENCH_CACHE_DIR=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/vbench
export GPU_LIST=0,1,2,3,4,5,6,7
```

额外必填：

```text
V218_NODES_FILE       共享JSON文件，按rank0..7写八个实际网卡IP
NODE_RANK            当前节点0..7
V218_NODE_ADDRESS    JSON中当前rank对应的实际网卡IP
TORCH_HUB_DIR        之前评测使用的完整torch hub cache
VBENCH_RUNTIME_HOME  之前评测使用的完整runtime home cache
```

现有完整cache应有`$VBENCH_CACHE_DIR/dino_model/facebookresearch_dino_main`和
`$VBENCH_RUNTIME_HOME/.cache`；入口使用它们定位DINO源码和DreamSim权重，
不再误用新worktree下不存在的`runs/`。非标准布局可显式设置`VBENCH_DINO_REPO`、`DREAMSIM_CACHE`。

`configs/v216_nodes.example.json`后两个IP留空，不能直接用于运行。
程序核对实际网络接口；不要填猜测IP。原六个节点若更换，也需要更新。
新输出以`v215_`开头是有意的：复用原v215生成身份，v218只改变评测，不伪造新生成campaign。
source目录中的原始videos/receipts不能移走。不要使用Git下载的小文件包代替原始生成目录。

rank0先运行：

```bash
python -m pytest -q tests/test_v218_evaluation_repair.py
bash scripts/run_v218_experiment.sh runtime
bash scripts/run_v218_experiment.sh probe
bash scripts/run_v218_experiment.sh prepare
bash scripts/run_v218_experiment.sh schedule
```

`runtime`要求原VBench与v215记录一致，复制到独立目录，只恢复原生DD严格加载，保留其他兼容性修补。
同时校验整个原生RAFT Python backend与固定版本一致。原目录漂移时停止，不能跳过指纹检查。
`probe`在一张GPU严格加载原生权重，输出相同图像/右移4像素的合成光流诊断。
理想表现是相同图像近零、右移图像x方向为正；这些是诊断，不是方法效果门槛。
`prepare`逐视频核对SHA，保留原generation目录引用，另建新评测；不改旧分数、视频、completion receipt。

八节点各执行：

```bash
bash scripts/run_v218_experiment.sh split
# Barrier: wait for all eight nodes to finish splitting.
bash scripts/run_v218_experiment.sh preflight
bash scripts/run_v218_experiment.sh eval
```

中断后同节点同配置重新`eval`或`eval-missing`，均跳过已验证成功的metric job。
全部结束后rank0：

```bash
bash scripts/run_v218_experiment.sh status
bash scripts/run_v218_experiment.sh collect
bash scripts/run_v218_experiment.sh analyze
bash scripts/run_v218_experiment.sh package
```

不只更新DD均值：统一重评core-9，避免混合新旧运行合同；代价只有评测，没有新生成。
原时序诊断CSV因视频字节不变而复用，并重绑到新comparison manifest。

### 6.1 严格RAFT加载失败时

先保存错误日志，**不使用`strict=False`，不回退torchvision，不覆盖共享旧权重**。
需要原生RAFT `raft-things.pth`，不是仅文件名相同的torchvision checkpoint。
VBench固定版本[模型下载逻辑](https://raw.githubusercontent.com/Vchitect/VBench/45e79ec14e69a2187202c675d2dbce1a71843d53/vbench/utils.py)
给出原生models.zip链接。可下载到全新目录：

```bash
set -euo pipefail
export NEW_RAFT_DIR=/apdcephfs_gy2/share_302533218/cedricnie/v218_models/raft_${EXP_COMMIT:0:8}
mkdir -p "$NEW_RAFT_DIR"
test ! -e "$NEW_RAFT_DIR/models.zip"
curl --fail --location https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip --output "$NEW_RAFT_DIR/models.zip"
unzip -n "$NEW_RAFT_DIR/models.zip" -d "$NEW_RAFT_DIR"
python scripts/prepare_v218_model_cache.py \
  --source-cache "$VBENCH_CACHE_DIR" \
  --output-cache "${VBENCH_CACHE_DIR}_v218_${EXP_COMMIT:0:8}" \
  --raft-checkpoint "$NEW_RAFT_DIR/models/raft-things.pth"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}_v218_${EXP_COMMIT:0:8}"
bash scripts/run_v218_experiment.sh probe
```

若下载站点不可达，由服务器管理员取得相同原生权重，使用最后三个命令。
overlay只新建符号链接，其他评测模型沿用现有cache；记录SHA但不将任意文件自动认证为正确权重。
所有节点改为相同overlay路径，严格probe通过后才执行prepare。成功prepare后不能再换checkpoint。
如果probe已经成功写出receipt，不要删除receipt再换权重继续同一run；新输出目录才代表新评测条件。

## 7. P1启动：80条确认，不重复48条

P0完成后，继续用同一份新checkout。以下只适用于**尚未冻结的新v216**：

```bash
export V216_V215_ROOT="$V218_OUT_ROOT"
export V216_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v216_runs/v216_${EXP_COMMIT:0:8}_imaging80
export V216_NODES_FILE="$V218_NODES_FILE"
export V216_NODE_ADDRESS="$V218_NODE_ADDRESS"
export V216_CANDIDATE=headwise_correct
export V216_PRIMARY_METRIC=imaging_quality
export V216_PRIMARY_WINDOW=full
export V216_RATIONALE='v215 development imaging signal; limited visual-quality claim, no established ID or overall superiority'
export VBENCH_ROOT="$V218_VBENCH_ROOT"
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3

# rank0 only
bash scripts/run_v216_experiment.sh freeze
bash scripts/run_v216_experiment.sh prepare
bash scripts/run_v216_experiment.sh schedule
python -m pytest -q tests/test_lphc.py tests/test_lphc_native_integration.py tests/test_lphc_head_descriptors.py
bash scripts/run_v216_experiment.sh baseline
bash scripts/run_v216_experiment.sh gate0
bash scripts/run_v216_experiment.sh smoke
```

模型/提示默认沿用原路径，无须新下载生成模型：

| 项目 | 默认路径 |
|---|---|
| SF权重 | `/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt` |
| Wan模型目录 | `/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B` |
| 128 prompt | `/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt` |

检查通过后，八节点分别`bash scripts/run_v216_experiment.sh generate80`。
rank0执行`status`和`publish`，八节点分别`split`，等待全部切片完成，再分别`preflight`、`eval`。
最后rank0依次`collect`、`analyze`、`package`。更详细的操作和恢复见[docs/236](236_v216_eight_node_confirmation_and_writing_runbook.md)，
但**本文件的候选、Imaging主指标、新评测路径取代其中旧的待选型说明**。

已经在跑v216/v217：不要停止生成、pull旧checkout、覆盖selection或重新生成已完成视频。
先回传`inputs/selection.json`及运行状态。v218当前只实现v215的评测分叉，
不声称可直接用于任意v216/v217；若已有冻结run用了旧评测，另行设计只重评的迁移，不改变其主指标。

## 8. 可选P2与回传

v217沿用冻结方法、主指标和八节点清单。固定取80条中的64条，seed为`21600+source_index`，
与P1的`21500+source_index`不同；三方法为SF/Ours/仅随机化选帧的Ours。
完整入口在[docs/237](237_v217_64gpu_replication_and_icassp_closure.md)，必须沿用修正后的VBench/cache路径。
设置`V217_V216_ROOT=$V216_OUT_ROOT`、新`V217_OUT_ROOT`、各节点`V217_NODE_ADDRESS`，
rank0 `freeze/prepare/baseline/gate0/smoke`，八节点`generate64`，然后按同样发布/切片/评测顺序。

优先回传`v218_small_artifacts.tar.gz`，随后`v216_small_artifacts.tar.gz`，有P2再回传v217包。
关键文件包括：

- `v218_runtime.json`、`raft_probe.json`、`evaluation_repair.json`：实际模型、评测源码与媒体绑定。
- `evaluation/analysis/v215_selector_phase.json`：重评结果，新增Imaging逐prompt/early/full/late比较。
- `evaluation/analysis/v216_confirmation.json`：80条冻结假设确认、全部取舍和风险。
- `v216_selection_included128.json`：48开发+80确认，明确含选型数据，不称128条独立测试。
- `jobs/.../stdout.log`、`stderr.log`、`trace.jsonl`、`done.json`：复用已有debug，不增加GPU profiling。

论文表格工具[docs/238](238_icassp_64gpu_evidence_packet_and_execution.md)继续使用，但其输入应是新评测根。
core-9不等于完整官方Semantic/Total，不补造缺失分数。共用人工检查预算最多6对，
先看自动风险/中位样本/极端差异，不要求逐一盲审全部视频，也不将诊断称为用户研究。

## 9. 写作决策与截止

若80条Imaging保持正向、其他质量/运动代价可接受，就围绕有限的画质改善完成方法短文。
不要求所有指标显著，不要求全部旧gate通过；区间宽则写初步证据，不写统计显著。
若80条主指标反向，不能继续用48条最好的列支持同一成功结论。
若匹配random持平，弱化内容选择的必要性，保留局部路径与受限历史修正的实际证据。

2026-09-20核对[官方提交入口](https://cmsworkshops.com/ICASSP2027/papers.php)：北京时间9月24日20:00截止。
[Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)允许4页技术内容，
可选第5页仅用于参考文献、资助和伦理声明；其旧日期表与提交入口不一致，按更新的入口安排，内部9月23日上传。
作者负责正文与结果核验；遵循官方LLM使用政策，由作者撰写并核对论文，工具辅助实验、结构和编辑。
目标是提高投稿质量，不承诺录用，不将四页当作省略关键负面对照或评测错误的理由。

## 10. 本轮实现与验证范围

新增独立RAFT runtime、严格GPU probe、只读生成目录复用、64槽位评测计划和可选权重cache overlay。
补Imaging逐prompt统计及可冻结确认端点；旧协议默认指标不变。
修复多节点`eval-missing`入口，使用原本可恢复的`eval`调度，不误入仅支持单节点的legacy分支。
已确认无效的DD源码指纹不能冻结新确认或导出正式论文结果；旧结果仍可用诊断脚本查看。

本地仅纯Python/NumPy协议回归、文件/哈希/符号链接测试、编译和Shell语法检查。
本地Windows无创建符号链接权限，两项相关集成测试需在Linux运行；不能把skip称为已验证通过。
最终回归为**185 passed、5 skipped**；另3项skip因本机无Torch。Python编译及两个Shell入口语法检查通过。
**没有运行GPU推理、真实RAFT前向或VBench评测；需在服务器执行probe、数值检查和baseline/gate0。**
本轮不修改`src/`及Self-Forcing推理算子，保持旧生成视频可复用。
