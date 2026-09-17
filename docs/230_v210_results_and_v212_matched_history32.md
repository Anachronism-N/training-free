# v210 结果分析与 v212 同局部窗口历史对照

日期：2026-09-18。同步来源：`origin/worktree-v210-lphc`，先到 `dc3f7452`，
交付前再次同步到补齐v209结果的 `0ff76ce4`。
当前交付分支：`codex/v178-v179-causal-validation`。

## 1. 最新进度

- v210 LPHC 已实现，8 prompts x 8 methods 的生成和 VBench core-9 已完成。
- 小文件已上传 `artifacts/experiment_results/v210_generation_789d8604_eval_08fe9d72/`。
- v211 已实现 sink1 + recent20 上的两种强度，8条 prompt；本次同步没有其完整结果。
- v209的32-prompt原生预算实验也已上传；160条视频、45个core-9评测job完成。
- docs/229 中“LPHC尚未实现”已经过时。此轮不再实现第二个残差模块，也不重复旧8条视频。

## 2. v209 与 v210 的实际结论

### 2.1 最新v209支持保留FIFO对照

| Native SF配置 | Quality w/o DD | Temporal | Semantic |
|---|---:|---:|---:|
| FIFO21 | 86.09030 | .98042 | .23310 |
| sink1+recent20 | 85.66665 | .97705 | .23599 |
| sink1+recent12 | 85.41467 | .97406 | .23951 |
| sink1+recent8 | 85.41543 | .97438 | .23968 |
| sink3+recent6 | 85.42393 | .97259 | .24203 |

sink1_21相对FIFO21的quality差为-.423646，paired CI=[-.786235,-.071239]；
temporal也较低。semantic均值较高但CI跨0，不能说其语义收益已确认。
这说明加入sink或压缩局部窗口并非免费收益，不能只留下更弱的sink基线。
这些原生配置比较不是我们的新方法效果，也不能用于代数修正旧v201结果。

v209 native production canary通过；PF/Adaptive部分在上传包里明确写的是`not run`。
因此当前**未证明PF/Adaptive production parity**，不是“测试后已经失败”。
v212完全走native SF，不需要为此等待或重跑v207/PF。

来源：`artifacts/experiment_results/v209_generation_86f10607/screen32/analysis/v209_budget.md`
及同目录`v209_paired.csv`。

### 2.2 v210不是全部生成失败

原始自动结论是 `stop_v210_no_eligible_lphc_candidate`，保留不改。
但必须区分它的几个条件，不能把汇总 fail 解释为每个方案都出现视觉噪声。

相对**同局部窗口的 SF FIFO21**，全视频 paired 数字为：

| 候选 | Quality w/o DD 差值 | 95% bootstrap CI | Temporal差值 | Semantic差值 | 运动诊断 |
|---|---:|---|---:|---:|---|
| early1, alpha=.02 | +0.232029 | [+.039312,+.471845] | +.001180 | -.001424 | 通过，0/8异常 |
| early1, alpha=.10 | +0.234146 | [+.072703,+.404651] | +.001314 | -.000796 | 通过，0/8异常 |

这些是经过小样本、多候选筛选的开发结果，不是确认性显著提升。它们值得扩样，不值得直接进入论文结论。

重要细节：

1. **五个 LPHC 相对 FIFO21 的运动诊断全部通过。** 总表的 temporal fail 来自另一组 sink1 对照。
2. .02/.10 的 FIFO 配对主要卡在 full/late 的语义 NI 区间。例如 .02 full semantic
   CI=[-.005302,+.001918]，跨越 NI margin=-.003。这表示证据不足，不是证明退化超过该门槛。
3. Sink1 SF 自身已有不同的权衡：其 subject/temporal 数值低于 FIFO，semantic较高。
   FIFO上的修正不保证复制 sink 所带来的运动/语义行为。仍应报告它，但不能把两个协议的差都归因于检索。
4. 旧分析器虽写了“significance不用于选择”，实际仍以多个 NI 的 CI 下界决定资格；
   “不要求显著胜出”不等于“不受区间不确定性影响”。8样本全部指标过门较难。
5. DD仍为1，不能用于证明运动提升；subject consistency也不能等同于真正的身份识别。
6. 旧耗时差异很大，部分 early1 比 full 更慢。只有现有小文件无法确定是负载、I/O还是实现开销，
   不能据此得出 early1 本身效率更差。本轮不以旧耗时排名候选。

可复算的逐条件解释在 `artifacts/experiment_results/v212_v210_posthoc_explanation.json`。
它标明 `posthoc_explanation_only=true`、`original_decision_changed=false`，不是重投票放行v210。

## 3. 这轮要回答的三个问题

1. 小幅正向效果能否从8 prompts扩展到32，而不是继续调整公式？
2. 收益来自检索选择，还是随便添加一些历史也能得到相似结果？
3. FIFO与sink local两个起点是否产生不同的历史修正收益？

不加新的head分类、压缩算子或EMA。保留现有LPHC公式、12帧archive、最多4帧读取、early1、alpha=.02。
选择较小强度的理由是v210中效果接近.10且干预更小，不是声称.02理论最优。

**实现细节需要准确表述：**当前archive是首/末帧加确定性hash-priority保留，descriptor来自V的统计，
不是物体身份检测器。此轮正确/随机使用同样的保留规则、候选资格、数量上限和干预强度。
两条生成轨迹分叉后，历史内容会不同，因此这不是在完全相同tensor轨迹上的单步反事实实验。

## 4. v212矩阵

| 方法 | Local | 历史选择 | alpha / phase | 角色 |
|---|---|---|---|---|
| sf_fifo21 | 最近21 | 无 | 0 | FIFO局部对照 |
| sf_sink1_21 | sink1+recent20 | 无 | 0 | sink局部对照 |
| sf_fifo25 | 最近25 | 无 | 0 | 容量参考，不是严格FLOP匹配 |
| fifo_correct | 最近21 | 内容选择4 | .02 / early1 | 主要候选 |
| fifo_random | 最近21 | 随机选择4 | .02 / early1 | FIFO机制对照 |
| sink_correct | sink1+recent20 | 内容选择4 | .02 / early1 | 主要候选 |
| sink_random | sink1+recent20 | 随机选择4 | .02 / early1 | sink机制对照 |

32条source index=`3+4k`，零基索引，有效seed=`21200+source_index`。
这些prompt不与v210本轮的8条重叠，但历史研究曾使用MovieGen128，不能称为全新未见测试集。
与v211 prompt有重叠，但seed不同，不拼接两轮成同一32-prompt表。
120 latent frames -> 477帧、16 FPS、832x480，约29.8秒，沿用“30秒”协议。

主实验224条视频。smoke就是source3的完整七方法bundle，其输出随后直接复用。
gate0另有两条prompt、两种local协议、各native/zero，共8条30-latent-frame短数值轨迹。

## 5. 六节点执行方式

沿用v211的六节点白名单，不访问19.69或19.70：

| NODE_RANK | V212_NODE_ADDRESS |
|---:|---|
| 0 | 28.216.19.213 |
| 1 | 28.216.19.143 |
| 2 | 28.216.19.137 |
| 3 | 28.216.19.225 |
| 4 | 28.216.18.144 |
| 5 | 28.216.18.136 |

每个prompt的七个方法都在**同一物理GPU串行**运行，方法顺序随prompt轮换。
32个prompt分给六节点，默认每节点5或6张卡工作；其余卡不自动抢占或启动占卡程序。
这是为减少method/GPU混杂主动保留的空位，不声称用满48卡。

新实验使用独立冻结checkout，不能在正在运行v211的目录中pull。无需SSH自动部署，
每台节点手动启动自己的rank；脚本检查实际网卡IP，不仅信任环境变量。
prepare绑定完整commit及runtime哈希，后续文档/代码更新也不会悄悄改变这个实验的身份。

```bash
# Fetch in a non-running checkout; create/deploy the same immutable commit on all nodes.
git fetch origin
COMMIT=$(git rev-parse origin/codex/v178-v179-causal-validation)
git worktree add --detach /tmp/training-free-v212-${COMMIT:0:8} "$COMMIT"
cd /tmp/training-free-v212-${COMMIT:0:8}

# All nodes use the SAME shared absolute output root, not node-local /tmp.
export V212_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v212_runs/v212_${COMMIT:0:8}_sixnode
export NODE_RANK=0
export V212_NODE_ADDRESS=28.216.19.213
export GPU_LIST=0,1,2,3,4,5,6,7

# Rank0 only. gate0 runs on GPU_LIST's first GPU; do not shorten GPU_LIST after prepare.
bash scripts/run_v212_experiment.sh prepare
bash scripts/run_v212_experiment.sh gate0
bash scripts/run_v212_experiment.sh smoke

# Each node separately, with its OWN NODE_RANK and V212_NODE_ADDRESS.
bash scripts/run_v212_experiment.sh generate32
bash scripts/run_v212_experiment.sh status
```

本轮没有新的模型依赖。默认SF checkpoint、Qwen prompt、Wan权重沿用v211，
可通过`SHARED_CHECKPOINT`、`V212_SOURCE_PROMPTS`、`WAN_MODEL`设置；prepare后不可改。
环境默认`longlive`，支持`CONDA_SH/CONDA_ENV`覆盖。所有节点须有相同配置路径。

generation失败重跑同一命令：已验证的job跳过，不完整job移到本实验quarantine，保留原日志。
同prompt换物理GPU会拒绝续跑，避免把这种变化藏起来。跨v210/v211结果不自动导入v212。

## 6. 评测与小文件回传

所有节点生成完成后：

```bash
# Rank0: media, source, trace and same-GPU/serial checks; read-only publication.
bash scripts/run_v212_experiment.sh publish

# All six nodes: split. Wait until all split jobs finish before eval.
bash scripts/run_v212_experiment.sh split
bash scripts/run_v212_experiment.sh preflight
bash scripts/run_v212_experiment.sh eval
# Interrupted metric jobs only:
bash scripts/run_v212_experiment.sh eval-missing

# Rank0 after all metrics finish:
bash scripts/run_v212_experiment.sh collect
bash scripts/run_v212_experiment.sh analyze
bash scripts/run_v212_experiment.sh package
```

`VBENCH_ROOT`指向已配置的VBench checkout，`VBENCH_CACHE_DIR`指向已有模型缓存。
与旧流程一样，core-9按原prompt映射评测，保存各维度原值；不冒充完整官方Semantic/Total。
数据/推理runtime/VBench fingerprint均绑定，缺少维度或混合来源就停止，不产生一个看似完整的表。

重要输出：

```text
inputs/manifest.json
decisions/gate0.json
jobs/screen32/<method>/source_<id>/invocation.json
jobs/screen32/<method>/source_<id>/trace.jsonl
jobs/screen32/<method>/source_<id>/done.json
evaluation/metrics/vbench_core9_summary.json
evaluation/metrics/temporal_diagnostics.csv
evaluation/analysis/v212_matched_history.json
evaluation/analysis/v212_matched_history.md
evaluation/analysis/v212_comparisons.csv
v212_small_artifacts.tar.gz
```

done中包含逐层early调用数、历史非空次数、平均/最大修正幅度、平均/最大历史年龄。
正确/随机分支的实际执行计数分别验证，不能把“开关已设置”当成“实际起效”。
计时为单进程wall time，包含加载、推理、VAE、编码；nvidia-smi轮询显存是process peak近似，
不是CUDA allocated/reserved精确测量，不能直接作为纯DiT吞吐。
package不打包MP4/PT/权重，不改写旧实验目录。先上传gate0或已完成的指标小文件也可以。

## 7. 决策如何改进

v210/v211旧门禁不改。v212在新数据生成前明确采用新的**开发规则**：

- FIFO候选的主要对照是FIFO21；sink候选主要对照是sink1_21。
- 另一个SF、SF25、random的全部差值仍报告，不删除不利对照。
- full quality w/o DD均值提升至少.10，full/late各组均值在原有NI范围内，matched运动诊断通过，
  则标记“可进入独立确认”，不是“论文已成功”。.10只是开发阈值。
- 同时列出每项CI：支持非劣、支持退化、区间不确定。不把“无法证明非劣”直接写成“证明有害”。
- correct-random均值至少+.05只是检索的方向性支持，报告CI；若随机同样好，不声称内容选择是贡献。
- 两个主要quality检验报告BH校正，其余window/指标用作解释。DD不参与晋级，计时不用于挑选赢家。

论文前仍需冻结方法、第二seed及更大样本确认。若FIFO候选只胜过FIFO却不如sink，要明确trade-off；
若所有历史方案都不如SF25，优先解释容量与成本，不包装成优秀检索方法。

这轮默认无需全量人工review。先分析指标与异常定位；只有出现冲突才选最多4条配对视频，
不能由少量“最好看的例子”覆盖总体负结果。

## 8. 本次交付与验证

- 新增v212冻结协议、同GPU生成、gate0、smoke复用、断点续跑、发布、VBench及配对分析。
- v211 trace auditor新增显式`allow_random=False`参数；旧v211仍禁止random，只有v212对照显式允许。
- 修复一个旧v210测试fixture在Windows以文本恢复YAML造成CRLF哈希变化的问题，推理代码不变。
- README更新为真实进度，补充可复算的v210逐对照失败解释。
- 本地只跑无PyTorch协议测试和静态检查。PyTorch/张量集成及真实GPU推理未验证，必须服务器gate0+smoke。

服务器建议先执行：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
python -m pytest -q tests/test_v212_lphc.py tests/test_v211_lphc_protocol.py \
  tests/test_lphc.py tests/test_lphc_native_integration.py
```

当前最有希望的路线仍是弱历史修正，但不能跳过“随机历史是否一样好”这一判断。
如果v212成立，论文重心才可从分类转向局部保护、受限历史干预及其适用条件。
