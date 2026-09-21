# v216结果、v219机制补全与四页论文收束

## 1. 本轮同步与写作决定

2026-09-21在`worktree-v210-lphc`快进到`13b62640`：

- `64f5c72a`：v218修正评测已完成，没有重新生成原336条视频。
- `13b62640`：v216的80条SF/Ours确认完成，共160条30秒视频。
- 目前未见v217上传结果；未上传不等于没运行。

**建议现在进入四页论文写作，同时并行完成一轮冻结方法的机制补全。**
不再以所有指标领先、所有旧gate通过或SOTA为前提，也不继续搜索新head分类。
现在可组织成一篇有限收益的实证方法短文，但收益小、区间宽，不能保证录用。
主要风险是收益幅度和机制证据，不是“必须多跑几个大表才有投稿资格”。

写作与实验并行：先完成问题定义、方法图、公式和实验协议；正文暂用明确的均值与区间，
不要先写“显著优于SF”“改善人物身份”再等待实验凑齐。

## 2. 当前有哪些可信结果

### 2.1 评测修复完成

v218严格使用原生RAFT；权重SHA与原记录一致，说明没有证据指向“旧checkpoint一定错了”。
合成相同图像平均光流幅度约0.0279，右移4像素得到x中位约4.0249、幅度约4.0274，
符合诊断预期。恢复了输入归一化/后端/严格加载的一致性，但不是方法优越性的证据。
不要再运行docs/239的P0，也不要下载/替换当前已通过的RAFT权重。

### 2.2 80条确认与128条汇总

| Cohort | 方法 | 官方公式Quality | Imaging | Aesthetic | Subject | Background | Dynamic |
|---|---|---:|---:|---:|---:|---:|---:|
| 确认80 | SF | 81.544057 | .685255 | .592273 | .970766 | .963805 | .427500 |
| 确认80 | Ours | 81.633525 | .686486 | .594693 | .970672 | .963877 | .429167 |
| 含选型128 | SF | 81.519056 | .683343 | .591662 | .970349 | .963188 | .441667 |
| 含选型128 | Ours | 81.574701 | .685747 | .593212 | .970377 | .963007 | .438021 |

v216冻结主指标是`imaging_quality/full`，不是看到结果后选Aesthetic：

- 80条Imaging：`+0.001231`，即百分制`+0.1231`百分点；95% CI `[-0.002066,+0.004589]`，55% prompt正向。
- 80条Quality：`+0.089468`百分点，95% CI `[-0.080498,+0.262995]`。
- 128条Imaging约`+0.2404`百分点，Quality约`+0.0556`百分点；包含48条选型数据，不称独立128条确认。
- Subject基本持平略负；不能讲已证明身份收益，Subject也不等于人物重识别准确率。
- 80条Dynamic小幅正向，128条小幅负向；不能声称已证明提高运动或完全无运动代价。
- v218开发48条中，headwise相对random的Imaging差`+0.004597`，CI `[+0.000429,+0.009109]`；
  相对pooled差也正向。但这些是看过开发数据后的机制信号，需要在新seed上确认，不能冒充预注册独立结论。

成像、审美、语义与部分时序均值正向，可以作为“局部质量改善的初步证据”。
没有必要因区间跨零而停止写作，也不能把它改称“统计显著”。

### 2.3 成本与失败信息不能省略

v216方法的中位同prompt耗时比约`1.0854`，即约8.5%额外wall time；
最大采样进程显存约33,244 MiB，对照26,776 MiB，约增加6.3 GiB。
这些包含加载、VAE、编码和共享负载，不是纯DiT吞吐或CUDA allocated峰值。
每条视频约67分钟的本轮中位耗时明显高于旧轮次，不能沿用25分钟估计作为保证。

6/80条出现自动时序代理告警，不等于6条人工确认失败。继续使用最多6对的总人工检查预算。
不需要逐一盲审所有视频；但也不能把未review称为视觉已验证。

### 2.4 上传证据边界

本轮复核publication manifest、小文件哈希、160个completion receipt、配对GPU、冻结指标和128条逐prompt算术。
新摘要在`artifacts/experiment_results/v219_planning/v216_review.{json,md}`。
当前上传包缺少v216的`decisions/sf_upstream_gate.json`与`decisions/gate0.json`；
这不表明服务器没通过，而是本地不能复核这两个receipt。只需回传小文件，**不用重生成**。
视频、原始VBench结果parts与完整trace未上传，所以没有本地重新计算视觉指标。

## 3. 四页的方法故事

建议暂称 **Local-Preserving History Correction (LPHC)**，标题不加入显著提升或身份保持承诺。
故事重点：直接用历史替换局部上下文可能扰动模型本来的生成路径；采用受控辅助读出，
在不训练的条件下把有限历史作为小幅残差修正。

1. **保留原生局部路径。** SF FIFO21，21 FFE包含当前3帧；不修改其主要attention输出。
   独立archive最多12帧clean K/V，排除local重叠后读最多4帧。
2. **保留head对应关系的检索。** 每head分别统计clean V的空间均值/标准差，与上一已完成块比较，
   平均同head的cosine分数后按层选帧。这是descriptor设计，不是head功能分类。
3. **受限的历史修正。** `attention(local+history)-attention(local)`按head相对local RMS限幅，
   再乘`.02`；只作用于phase0，clean refresh不注入。限幅不意味着对视频质量的理论保证。

新颖性应落在具体的组合方式和受限历史读出，而不是声称缓存、检索或限幅概念首次出现。
与SF/PF/EF等借鉴工作的关系保留在引用与实现说明中。当前没有证据支持“新功能性head分类”这个故事。

正文建议一张方法图、一张SF/Ours主表、一张紧凑机制表、一张时间趋势或定性图。
大规模历史探索不堆入正文，也不拿几个最漂亮样例替代总体结果。

## 4. 最后一轮：v217和v219二选一

用户已确认v217尚未启动：**本次直接执行v219，不再另外启动v217。**

| 运行状态 | 下一步 | 新生成 |
|---|---|---:|
| v217未启动 | 优先v219：SF/Ours/random/pooled，64 prompt，新seed | 256条 |
| v217正在运行或完成 | 保留v217，不启动v219，不丢弃已生成视频 | 沿用其192条 |
| 截止前时间不足 | 先完成已有80/128主表和零生成分析，保留探索性结论 | 0 |

**v219不是又一轮选最优方法。** 方法、alpha、phase、缓存、prompt集和主要指标不再调参。
只增加一个最直接的消融：同样的cache和修正，将headwise descriptor改为pooled descriptor。
它是v217尚未启动时的四组替代方案，不与v217合并为“两个独立seed”；两者seed和64个source完全相同。
已经跑v217时，开发48条的pooled对照足够作为有标注的探索性证据，不为补一列重复192条。

### 四个配置

| Key | 配置 | 作用 |
|---|---|---|
| `sf_fifo21` | 原生SF FIFO21 | 主要基线 |
| `ours_correct` | 冻结headwise/e1/.02 | 唯一主方法 |
| `ours_random` | 仅将同池选帧改为random | 内容选择是否有帮助 |
| `pooled_correct` | 仅将descriptor改为pooled | 保留head对应关系是否有帮助 |

SF对比仍是主比较；random/pooled是两个次要机制比较，不可用机制显著替代SF主比较。
pooled分支另有alpha0 gate，与方法alpha0共同对齐SF。
随机seed使用独立控制RNG；不改变生成噪声seed。已有archive的first/last+hash保留不变。

### 八节点分配

64个固定source来自v216的80条：按有序序号`floor(i*80/64)`采样，非按成绩筛选。
seed固定`21600+source_index`；v216为`21500+source_index`。
`node=j%8, gpu=j//8`，每卡一个四方法bundle，同prompt四方法串行。
每节点8个prompt、32条视频，方法顺序轮换，每种方法恰好两次在节点内先跑。
共256条30秒视频、64个GPU槽位，不是单视频64卡DDP。

按本轮约67分钟/视频估计，主生成四次串行约4.5小时，另加baseline/gate/smoke与评测，预留6至8小时；
共享负载可能显著改变时间，不能保证。smoke已完成的四条直接计入256条。
core-9评测共36个method×dimension任务，八节点各4或5项，不要求评测时64卡全部满载。

## 5. v219运行命令

以下只在**v217尚未启动**时运行。沿用v218修正后的评测，不修改旧运行目录。

```bash
git fetch origin
export EXP_COMMIT=$(git rev-parse origin/worktree-v210-lphc)
echo "$EXP_COMMIT"  # Distribute this exact value once to all eight nodes.
git worktree add --detach /tmp/training-free-v219-${EXP_COMMIT:0:8} "$EXP_COMMIT"
cd /tmp/training-free-v219-${EXP_COMMIT:0:8}
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
export GPU_LIST=0,1,2,3,4,5,6,7
export V219_V216_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v216_runs/v216_64f5c72a_eightnode_imaging80
export V219_OUT_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_${EXP_COMMIT:0:8}_mechanism64
export VBENCH_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v218_runtime/VBench_cf9b3b45
export VBENCH_CACHE_DIR=/apdcephfs_gy2/share_302533218/cedricnie/model_cache/vbench
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3
export TORCH_HUB_DIR=/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/torch_hub
export VBENCH_RUNTIME_HOME=/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/dreamsim_home
```

最后两个cache路径来自实际v216合同：它们是节点本地路径，请先确认每台仍存在。
若已清理，用完整现有cache的实际路径，不下载到新worktree后盲目重跑。
非标准布局可设置`VBENCH_DINO_REPO`和`DREAMSIM_CACHE`。
生成模型和MovieGen128 Qwen prompt路径沿用docs/239默认值，默认120 latent frames/约29.8秒/16FPS/480×832。

每节点设置自身rank，IP必须在本机真实接口上。冻结合同继承v216已确认清单：

| Rank | IP |
|---:|---|
| 0 | 28.216.19.213 |
| 1 | 28.216.19.143 |
| 2 | 28.216.19.137 |
| 3 | 28.216.19.225 |
| 4 | 28.216.18.144 |
| 5 | 28.216.18.136 |
| 6 | 28.216.19.69 |
| 7 | 28.216.17.70 |

例如rank0：

```bash
export NODE_RANK=0
export V219_NODE_ADDRESS=28.216.19.213
# rank0 only
python -m pytest -q tests/test_lphc.py tests/test_lphc_native_integration.py tests/test_lphc_head_descriptors.py tests/test_v219_closure.py
bash scripts/run_v219_experiment.sh freeze
bash scripts/run_v219_experiment.sh prepare
bash scripts/run_v219_experiment.sh schedule
bash scripts/run_v219_experiment.sh baseline
bash scripts/run_v219_experiment.sh gate0
bash scripts/run_v219_experiment.sh smoke
```

若节点更换，先讨论迁移而不是伪造旧IP绕过实际接口检查。不要在读到新结果后换主要指标。
gate通过后八节点各运行一次：

```bash
bash scripts/run_v219_experiment.sh generate64
```

结束后rank0：

```bash
bash scripts/run_v219_experiment.sh status
bash scripts/run_v219_experiment.sh publish
```

八节点分别执行：

```bash
bash scripts/run_v219_experiment.sh split
# Wait until split finishes on ALL eight nodes.
bash scripts/run_v219_experiment.sh preflight
bash scripts/run_v219_experiment.sh eval
```

rank0收尾：

```bash
bash scripts/run_v219_experiment.sh collect
bash scripts/run_v219_experiment.sh analyze
bash scripts/run_v219_experiment.sh package
```

中断重试同一命令，验证通过的任务跳过；不清空目录、不改变GPU placement、不更新运行checkout。
回传`v219_small_artifacts.tar.gz`，其中包括源commit、节点/GPU UUID、配置、trace审计、三组比较与成本。
descriptor分数、候选/实际选帧、eligible池、修正RMS、phase和异常沿用现有trace，无新增GPU profiling。

## 6. 不占GPU的补充分析：现在就可做

新时间曲线工具读取已完成VBench原始parts，不推理、不重新评测、不改旧report。
它核对冻结输入、媒体/完成记录和metric结果hash，但允许新分析代码读取旧生成checkout的产物；
这条只读路径不能用于跳过生成时的runtime冻结。

在有完整原始parts的服务器执行：

```bash
python scripts/analyze_lphc_timecourse.py \
  --campaign v216 --run-root "$V219_V216_ROOT" \
  --output-root "$V219_V216_ROOT/evaluation/timecourse_v219" --plot
```

输出九个原始指标的15个有序clip曲线、逐prompt配对bootstrap区间、
每个指标内覆盖15个clip的max-deviation band，以及CSV/JSON/PNG/PDF。
`--plot`需要matplotlib，不装图形库也能去掉该参数得到数值。
时间单位为有序评测clip，不假定每段都恰好2秒；不平滑、不删除失败prompt、不只画正向维度。
区间按整个prompt轨迹重采样，不把15个片段当成独立样本。
这是事后诊断，不是新主检验；也没有跨指标/跨对照的同时覆盖保证。

v219完成后可同样用`--campaign v219 --run-root "$V219_OUT_ROOT"`生成三种对照曲线。
已有80条的“后半段优势减前半段优势”已经按prompt计算，并保存在本轮摘要中；
Imaging约`+0.001907`，CI `[-0.003662,+0.007301]`，仍不支持确定的长程改善或phase因果解释。

## 7. 论文证据包与最少人工工作

现在就可以在服务器导出完整80/128表，不用等v219：

```bash
export V218_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/v218_runs/v215_cf9b3b45_v218_eval
python scripts/export_lphc_paper_evidence.py \
  --v215-root "$V218_ROOT" --v216-root "$V219_V216_ROOT" \
  --output-root "$V219_V216_ROOT/paper_evidence_v219"
```

回传此证据包，并补v216的两个`decisions/*.json`。该导出入口仍要求实际gate文件，
不会因为本地小文件包缺失就默认成功。视频不必上传。
v219完成后增加`--v219-root "$V219_OUT_ROOT"`并换新output-root；若完成的是v217，就使用`--v217-root`。
两个选项互斥，不能把相同seed的两个campaign当独立复验。
已有人工检查ID用`--reviewed-ids`传入，跨实验最多6对；新轮仅新增至多2个高风险候选，
不用再审全部256条。不要把自动生成的队列ID当成已经review的ID。

四页主表可列80确认，并另标注128汇总；新seed的同64条结果及prompt-cluster区间单独展示。
若Ours对SF正向、对random/pooled持平：只保留受限历史修正的收益，弱化检索必要性。
若三项均正向：支持更完整的机制故事，但次要对照仍不冒充独立主检验。
若新seed反向：如实呈现不稳定性，不能选择性丢弃该seed；可收束为探索性短文或重新评估投稿定位。

## 8. 截止与停止规则

2026-09-21再次核对[官方入口](https://cmsworkshops.com/ICASSP2027/papers.php)：北京时间9月24日20:00截止。
按4页技术内容准备，可选第5页仅参考文献/资助/伦理声明。内部目标9月23日完成可提交版本。
作者负责正文撰写、核验和提交，工具辅助代码、实验、结构与编辑，遵循官网LLM政策。

今天同步写方法与主表、启动一轮v217或v219；次日汇总机制/时间趋势，随后冻结结果进入排版。
**不再把“再试一点可能更好”作为无限延后写作的理由，也不靠只展示有利片段包装成功。**
四页允许问题更窄，不要求大而全的理论、跨模型、60秒、PF或ABA；关键是主张与证据相符。

## 9. 实现边界

本轮新增v219协议、四组同卡调度、pooled零校正检查、三种独立比较、证据包兼容和零GPU时间曲线。
不改变SF/LPHC算子、archive预算或v216冻结方法，不覆盖已有生成或评测结果。
本地仅做纯Python/NumPy与语法测试，没有A800推理或VBench实跑；服务器需先跑baseline/gate0。
最终回归**197 passed、5 skipped**，其中3项缺Torch、2项缺Windows符号链接权限。
Python编译、共享Shell入口及v219入口语法检查通过；没有把跳过的测试计为成功。
