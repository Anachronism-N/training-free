# v116 MovieGenBench-16 初评与指标运行手册

日期：2026-07-27

状态：代码已实现；必须先完成人工 v115 单视频筛选，才允许启动。

## 1. 目的与边界

v115 用一个固定 prompt 快速排除多边形噪声、重复主体、运动冻结和无效
cache。v116 不重新搜索全部配置，只把人工 review 后的少量候选放到冻结的
MovieGenVideoBench-16 子集上做初步排序。

v116 回答三个问题：

1. 单 prompt 上干净的缓存是否能跨身份、多人、快速运动、镜头运动和场景演化
   保持稳定；
2. 二角色路由是否优于相同预算的 role-neutral/local control；
3. 视觉差异较小时，DINO、drift、motion、background、loop 和 VBench-Long
   是否给出一致证据。

16 prompts 只用于快速选择主方法，不足以支撑最终论文主表。候选确定后仍需在
MovieGenVideoBench-128 上与 SF、PF、Echo-Forcing 及最终消融做同 seed 对比。

## 2. 冻结 prompt 子集

源文件：

```text
third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num128.txt
SHA256: 926fdba7b8c325960b1cdeca559cd7ed2bbc475875e74f7a04476c32637a77f9
```

冻结清单为 `prompts/moviegenbench_diverse16.json`。使用的 0-based 源索引为：

```text
0, 1, 4, 7, 13, 15, 17, 24, 33, 47, 61, 67, 75, 84, 109, 124
```

覆盖内容包括人和动物身份、多主体、快速运动、旋转/跟踪/FPV 镜头、拥挤背景、
季节变化、场景切换与主体变形。runner 会同时核对 prompt 数量和源文件 SHA；
任何源文件漂移都会直接停止。

## 3. 方法选择

先检查 v115 的全部视频和 `analysis/role_memory_summary.md`。只有满足以下条件的
候选才能进入 v116：

- 视频为 477 帧、16 FPS、832 x 480；
- 无 polygon/color-block artifact、持续双主体或明显后段冻结；
- policy trace 中 `exclusive_owner=true`，304/56 membership 正确；
- sink/middle/recent 无重叠，实际读 token 不超过 9 frame-equivalents；
- 新 memory 确实发生 admission/replacement/compression，而不是退化为 recent；
- 相对 `support_landmark4_suppress_recent8` 或 `all_recent8` 至少有可解释优势。

默认四方法集合是：

```text
prototype_motion1
snapshot_motion1
control_landmark_recent
control_all_recent8
```

它只是推荐的低风险起点，不代表预先选定结果。若 v115 选择不同配置，应显式
设置 `V116_METHODS`。所有可用 key 可通过以下命令查看：

```bash
python scripts/run_v116_role_memory_diverse16.py --list-methods
```

推荐保持“两个候选 + 两个对照”。至少一个对照必须完全不依赖 304/56 路由；
当前 `control_all_recent8` 满足该要求。方法顺序也是冻结 contract 的一部分，
四个节点必须完全一致。

## 4. 四节点生成

四个节点都切到同一 commit，并使用共享 `OUT_ROOT` 所在文件系统：

```bash
cd /path/to/training-free
git pull

export REPO_ROOT="$PWD"
export PF_CHECKPOINT="$PWD/third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"
export V115_PROMOTION_APPROVED=1
export V116_METHODS="prototype_motion1,snapshot_motion1,control_landmark_recent,control_all_recent8"
export NUM_NODES=4
export GPU_LIST="0,1,2,3,4,5,6,7"
```

每个节点设置不同 rank：

```bash
NODE_RANK=<0|1|2|3> \
python scripts/run_v116_role_memory_diverse16.py generate
```

四方法时共有 `4 x 16 = 64` 个 30 秒任务，每节点 16 个任务，每张 GPU 顺序运行
2 个。输出目录由有序方法集合的 SHA 自动决定；默认集合为：

```text
runs/v116_role_memory_diverse16/m4_7c75817c92a2
```

不要在同一输出目录中改变方法顺序、prompt 文件、head map、checkpoint 或代码。
runner 会冻结完整 contract、implementation hashes 和 304/56 map hash。

## 5. 发布与完整性审计

所有节点完成后，在 node 0 使用相同的 `V116_METHODS`：

```bash
NODE_RANK=0 NUM_NODES=4 \
python scripts/run_v116_role_memory_diverse16.py audit
```

审计通过后生成：

```text
<RUN_ROOT>/published_manifest.json
<RUN_ROOT>/published/<method>/000000.mp4 ... 000015.mp4
<RUN_ROOT>/published_indexed/<method>/000000-0_v116.mp4 ... 000015-0_v116.mp4
<RUN_ROOT>/prompts/moviegenbench_diverse16.txt
```

两套目录都是指向同一源视频的 hardlink 或 symlink，不重复存储视频：

- `published` 用于 VBench-Long；
- `published_indexed` 使用仓库综合评测器要求的
  `<prompt>-<sample>_<suffix>.mp4`，防止 prompt 错配。

任何缺失、多余、零字节、marker/contract 不一致或链接指向错误都会使 audit
失败，指标脚本只接受成功的 `published_manifest.json`。

## 6. Trace 汇总

生成完成后先汇总实际缓存行为：

```bash
export RUN_ROOT="$PWD/runs/v116_role_memory_diverse16/m4_7c75817c92a2"
python scripts/analyze_v115_role_memory_traces.py --run-root "$RUN_ROOT"
```

重点检查：

- Prototype：span、medoid、represented frame count、压缩/创建/淘汰次数；
- Snapshot：relevance、uniqueness、utility、spacing/replacement gate；
- Retrieval：archive 大小、eligible/gated/top-k、相似度和 MMR；
- Sparse：每 snapshot token 数、实际 keep ratio、token score 范围；
- Motion pair：候选语义相似度、运动阈值、pair bank 与替换原因；
- 所有方法的实际 token-equivalent read budget、overlap 和 contract failure。

出现“新缓存从未更新”“始终只选 block 尾帧”“prototype 全部 count=1”时，
即使视频看起来正常也不能把对应机制写成有效贡献。

## 7. VBench-Long

每个节点保留第 4 节环境变量并设置自己的 rank：

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
bash scripts/run_v116_vbench_long.sh eval
```

等待四个节点全部完成后，在 node 0 汇总：

```bash
NODE_RANK=0 NUM_NODES=4 \
bash scripts/run_v116_vbench_long.sh collect
```

计算六个维度：

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
motion_smoothness
dynamic_degree
```

结果位于：

```text
<RUN_ROOT>/metrics/vbench_long_summary.{json,csv,md}
```

## 8. 八项辅助诊断

四节点分别运行：

```bash
NODE_RANK=<0|1|2|3> NUM_NODES=4 \
bash scripts/run_v116_aux_metrics.sh eval
```

全部完成后在 node 0 汇总：

```bash
NODE_RANK=0 NUM_NODES=4 \
bash scripts/run_v116_aux_metrics.sh collect
```

该链路计算 DINO consistency、long-term drift、RAFT motion smoothness、
ArcFace identity、flicker、CLIP alignment、background consistency 和 loop
diagnostic。结果位于：

```text
<RUN_ROOT>/metrics/auxiliary_summary.{json,csv,md}
```

指标不能替代人工 review。尤其是 DINO 较高但 motion/dynamic degree 明显下降，
可能只是冻结；loop 较低但 identity/background 较差，也不能视为改进。

## 9. 人工 Review 表

每条 prompt 盲评以下项目，使用 `-2/-1/0/+1/+2` 相对最强对照打分：

| 项目 | 观察内容 |
|---|---|
| ID continuity | 人/动物/物体身份、服饰、颜色、形状 |
| background continuity | 地点、布局、远景和光照是否漂移 |
| motion quality | 动作是否连续、有幅度且不冻结 |
| camera consistency | 跟踪、旋转和 FPV 是否平滑 |
| duplicate/artifact | 双主体、多边形、色块、局部撕裂 |
| long-term drift | 15-30 秒是否显著劣化 |
| prompt fidelity | 复杂动作、场景演化与变形是否完成 |

必须同时记录失败 prompt 的 source index 和失败发生时间，不能只给总胜率。

## 10. 选择规则与下一步

优先选择满足以下条件、且故事最简洁的方法：

1. artifact hard gate 全通过；
2. 至少 10/16 prompts 不劣于两个对照，严重失败不超过 1 条；
3. identity/background 改善不能依靠明显降低 motion；
4. role-conditioned 候选优于或至少稳定匹配相同 memory 的 all-head control；
5. trace 能证明两个角色实际读取了不同的、符合定义的历史。

若 `prototype_motion1` 最稳，论文主方法可表述为：

```text
History-polarity head partition
+ Supportive temporal prototype memory
+ Suppressive coherent motion bridge with a larger recent window
```

若 `snapshot_motion1` 更好，应把 snapshot 明确写为受 Echo-Forcing 启发的缓存
组件，创新重点放在二角色发现、角色条件化使用和统一预算/生命周期，而不是声称
relevance/uniqueness snapshot 本身原创。

Retrieval 或 sparse 只有在无 artifact 且跨 prompt 稳定时才能升级为主方法；
否则记录为负结果或消融。最后进入 MovieGenVideoBench-128 时，复用已有且
prompt/seed/checkpoint/frame contract 完全相同的 SF/PF/Echo 视频，不重复生成。
