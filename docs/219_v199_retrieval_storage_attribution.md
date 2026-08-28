# v199：Retrieval 存储预算归因与下一阶段实验

> 日期：2026-08-29  
> 状态：生成、严格审计、自动评测与容量选择代码均已完成
> 人工 review：不需要

## 1. 本次同步结论

拉取全部远端分支后没有新结果提交：`origin/main` 仍为 v186 的两组
128-prompt、60 秒日志，实验分支仍停在 v198 评测代码。GitHub 与本地均未出现
v198、v189 或更后阶段的新结果，因此目前不能给出新的效果结论。

现有实验链路中，v190 已经实现 all-head、Head-only、Phase-only、membership shift、
phase shift 和 dense-phase 控制。继续新增相似的 Head x Phase 路由不会增加可识别证据。
当前明确但尚未闭合的混杂因素是：

- all-Recent 每次读取并保存 `sink1 + recent8 = 9 FFE`；
- v186 Retrieval 每次最多读取 `sink1 + retrieved4 + recent4 = 9 FFE`；
- 但 Retrieval 默认维护 12 FFE 的候选 archive，总存储为 17 FFE。

因此 v198 即使显示 Retrieval 有优势，也还不能判断优势来自内容寻址检索，还是来自额外
保存了更多历史帧。

## 2. v199 的严格变量控制

四种方法全部用当前同一 commit 重新生成，不混用旧 runtime：

| 方法 | Sink | Middle read | Recent | Archive storage | 总读取上限 | 总存储 |
|---|---:|---:|---:|---:|---:|---:|
| `all_recent` | 1 | 0 | 8 | 0 | 9 | 9 |
| `retrieval_archive4` | 1 | 4 | 4 | 4 | 9 | 9 |
| `retrieval_archive8` | 1 | 4 | 4 | 8 | 9 | 13 |
| `retrieval_archive12` | 1 | 4 | 4 | 12 | 9 | 17 |

关键对照是 `retrieval_archive4` 与 `all_recent`：两者总读取和总存储都为 9 FFE，唯一
变化是局部八帧窗口被替换为“recent4 + 由 archive maintenance 保留的四个远期帧”。
若前者仍有收益，不能再把结果解释成单纯增加存储。`archive8/12` 用于判断更大的检索
候选池是否必要；三种 Retrieval 的每次读取始终最多四帧。

这是开发期配置归因，不直接形成论文主结果。v199 只在 v198 输出以下任一结果时授权：

```text
promote_retrieval_operator_to_selective_routing_validation
noninferior_but_no_clear_long_history_gain
```

如果 v198 明确拒绝 Retrieval，脚本会阻止 32-prompt 生成。单 prompt smoke 不受此 gate
限制，可提前验证 runtime。

## 3. Runtime 与 debug

新增参数：

```text
--pyramidkv_semantic_retrieval_archive_capacity {4|8|12}
```

该参数只改变 exact-frame 候选 archive 的容量，不改变 Retrieval read capacity。容量小于
read capacity 时立即报错。启动日志必须出现：

```text
[SemanticRetrievalArchive] labels=21 read_capacity=4 archive_capacity=N \
read_budget_unchanged=true exact_frame_storage=true
```

每个视频还记录 layer `0,10,20,29`、head `0,6,11` 的抽样 policy trace。审计自动检查：

- `SemanticRetrievalStrategy.capacity == 4`；
- `archive_capacity` 与方法名一致；
- archive 实际 resident frames 不超过容量；
- `sink + middle + recent <= 9 FFE`；
- `cache_contract_pass == true`；
- all-Recent 中没有 Retrieval strategy；
- 957 帧完整解码、16 FPS、832x480；
- 没有 traceback/OOM/assert、缺视频或跨方法完全相同的 MP4。

## 4. 服务器先行命令

使用分支 `codex/v178-v179-causal-validation`。先完成 v198 的 `decision`，或只执行 smoke。

Node 0：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh preflight
```

四卡 smoke：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3 \
  bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh audit-smoke
```

期望输出：

```text
[v199-smoke] PASS methods=4 prompts=1 read_budget<=9
```

v198 gate 允许后，四个节点分别执行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh generate32
```

每张卡依次生成同一 prompt 的四种方法，共 128 个新视频。完成后 Node 0：

```bash
NODE_RANK=0 bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh status
NODE_RANK=0 bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v199_retrieval_storage_attribution_32gpu.sh package
```

不需要人工观看视频。只有自动评测发现指标冲突或时序异常时，后续分析器才会列出不超过
四条定位样本。

## 5. 当前优先级

1. 先运行 v198，对已经生成的 512 个视频完成审计与指标计算；这一步不生成新视频。
2. profiling 主线仍从 v189 开始，服务器可用 `run_v196_campaign_frontier.sh show` 确认。
3. v198 至少非劣时运行 v199，消除 Retrieval 的额外存储混杂。
4. v199 选出的最低充分 archive 容量再用于 v189/v190 的 Retrieval operator；不根据
   人工观感调整容量。

v199 不替代 Head x Denoising Phase 的分类验证。它只保证后续分类实验使用的长期历史
算子具有可解释、可归因的存储预算。

## 6. 自动评测与容量选择

生成审计通过后，Node 0：

```bash
NODE_RANK=0 bash scripts/run_v199_vbench_long.sh prepare
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v199_vbench_long.sh preflight
```

四个节点分别执行切片与 core-9：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v199_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v199_vbench_long.sh eval
```

Node 0 汇总；`collect` 自动计算 temporal jump、低运动区间、后段运动衰减和画面异常：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v199_vbench_long.sh status
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v199_vbench_long.sh collect
```

相机补偿局部运动是 CPU/共享存储任务。四个节点分别运行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> V199_CAMERA_WORKERS=8 \
  bash scripts/run_v199_vbench_long.sh camera-compute
```

最后 Node 0：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v199_vbench_long.sh camera-status
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v199_vbench_long.sh camera-collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v199_vbench_long.sh decision
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v199_vbench_long.sh package
```

容量选择不使用一个可事后调权重的总分。每个候选首先相对 all-Recent 独立通过：

- full 与 late-half Quality 的 paired bootstrap CI 下界不低于 `-0.25`；
- identity/background 不低于 `-0.002`；
- temporal mechanics 不低于 `-0.004`；
- visual quality 不低于 `-0.005`；
- 自动 temporal guard 通过。

这些是开发期非劣容忍区，不是论文显著性阈值。通过者中默认选择 archive 最小的方法。
只有更大容量相对当前较小容量在所有主轴继续非劣，并且至少一个 full/late 主轴的 CI 下界
大于 0 且 BH `q <= 0.10`，才升级容量。

可能输出：

| Recommendation | 含义 |
|---|---|
| `use_archive4_storage_matched_retrieval` | 等总存储下 Retrieval 有正向信号，优先进入 v189/v190 |
| `archive4_noninferior_but_no_retrieval_gain` | 最小缓存安全，但没有选择 Retrieval 的证据 |
| `use_retrieval_archive8_extra_storage_required` | 需要 8 FFE 候选池，论文必须披露额外存储 |
| `use_retrieval_archive12_extra_storage_required` | 默认 12 FFE archive 才充分，存储混杂未消除 |
| `reject_retrieval_under_current_runtime` | 三种容量均未通过安全非劣 gate |

`Dynamic Degree` 若为常数或全 1，只记录 ceiling non-regression，不参与“运动提升”判断。
相机补偿光流只用于开发期方向判断，不直接当作论文指标。分析器不要求人工 review；仅在
自动异常出现时列出最多四个 debug prompt。

需要推送的小文件：

```text
runs/v199_retrieval_storage_attribution/inputs/
runs/v199_retrieval_storage_attribution/audits/*.json
runs/v199_retrieval_storage_attribution/published_manifest.json
runs/v199_retrieval_storage_attribution/vbench_comparison/comparison_manifest.json
runs/v199_retrieval_storage_attribution/metrics/*.{json,csv,md}
runs/v199_retrieval_storage_attribution/analysis/v199_retrieval_storage.{json,md}
runs/v199_retrieval_storage_attribution/camera_motion/*/analysis/*.json
runs/v199_retrieval_storage_attribution/evidence_manifest.json
```

无需上传 MP4、VBench clips 或 `vbench_long_parts/`。
