# v115 双角色内容驱动缓存设计与单视频筛选

日期：2026-07-27

状态：代码已实现，等待服务器单视频实验和人工 review。

## 1. 当前结论与本轮问题

我们继续固定旧 v98 的 304/56 二分类：

```text
History-Supportive:  304 heads
History-Suppressive: 56 heads
```

v111 并非只测试了 Suppressive。已经完成的 Supportive 配置包括：

- `recent8`
- `landmark4`
- `motion_pair2`
- `landmark2 + motion_pair1`

已经完成的 Suppressive 配置包括：

- `recent8`
- `landmark4`
- `motion_pair2`

单 prompt 人工 review 的当前证据是：

1. `Supportive=landmark4` 最稳定。
2. `all_motion_pair2` 会出现后段双主体和运动下降。
3. `Supportive=landmark4, Suppressive=recent8` 与
   `Supportive=landmark4, Suppressive=motion_pair2` 都干净。
4. 单个 prompt 无法证明 Suppressive motion memory 有独立收益。
5. Supportive 尚未系统测试压缩、快照和受限检索。

因此 v115 不再重复 v111 视频，而是补齐两类 head 的 cache 机制。目标不是
在单 prompt 上宣布最优，而是找出：

- 没有多边形噪声、重复主体和运动冻结的可用机制；
- 在视觉接近时最容易形成清晰论文因果链的机制；
- 值得进入 MovieBench 多样性 16-prompt 评测的 3 至 4 个方法。

## 2. 不变的正确性约束

所有 v115 方法共享以下结构：

```text
sink + middle memory + recent
```

每个 head 只有一个 `HeadComposition` 所有者。旧 dynamic history 不会与
显式 middle memory 同时生效。

所有更新只使用 committed clean K/V。每个角色在每层共享选帧描述子，但每个
head 保存自己的 K/V。所有 anchor 保留原始 position sidecar，读取时重新做
dynamic RoPE，并排除 sink/recent 的时间重叠。

三种等预算形式为：

```text
sink1 + middle4 + recent4              = 9 frame equivalents
sink1 + middle2 + recent6              = 9 frame equivalents
sink1 + 4 * 0.75 sparse + recent5      = 9 frame equivalents
```

审计按实际 token 数检查 9-frame-equivalent 上限，不能用 slot 数掩盖稀疏
snapshot 的预算差异。

## 3. 共享 clean-KV 描述子

角色 `r` 在时间 `t` 的描述子为：

```text
z_t^r = normalize([
  mean_spatial,heads(K_t^r),
  mean_spatial,heads(V_t^r),
  std_spatial,heads(V_t^r)
])
```

相邻运动分数为：

```text
m_t^r =
  mean((V_t^r - V_(t-1)^r)^2)
  / clamp(mean((V_t^r)^2 + (V_(t-1)^r)^2), eps)
```

这两个量每层、每角色只计算一次。所有同角色 head 使用相同的 admission 或
retrieval 决策，避免“同一角色每个 head 随机选择不同帧”造成不可审计的 cache。

## 4. 新增 middle memory

### 4.1 Temporal Prototype Memory

这是当前优先级最高的 Supportive 候选。

相邻帧只有同时满足以下条件才进入同一 prototype：

```text
cosine(z_t, prototype_centroid) >= 0.985
m_t <= online_q70(last_32_motion_edges)
time is contiguous
```

每个连续语义段维护：

- `start_t`, `end_t`
- normalized descriptor centroid
- 被代表的帧数
- 区间内最大 motion
- 一个真实帧 medoid 的完整 K/V 和 position

centroid 在线更新，但 cache 不平均 K/V。若新帧比旧 medoid 更接近更新后的
centroid，只把 medoid 替换成该真实帧。因此它借鉴“相邻冗余时间融合”的思想，
但避免 PF Merge 或直接 K/V averaging 曾出现的合成纹理和位置不合法风险。

四个 prototype 满后，保留首个长期参考，并优先淘汰与其他 prototype 最冗余、
覆盖时间短且 motion 较弱的区间。

可写故事：

> Supportive heads 不需要周期性采样，而需要对长期语义状态做事件分段和压缩。
> 连续冗余帧被一个真实 medoid 表示，语义或运动边界触发新 prototype。

### 4.2 Relevance-Uniqueness Snapshot Memory

每个 clean block 只选择一个完整帧 snapshot。候选分数为：

```text
relevance_t  = cosine(z_t, z_block_endpoint)
uniqueness_t = 1 - mean_j cosine(z_t, z_snapshot_j)

utility_t =
  0.75 * minmax(relevance_t)
  + 0.25 * minmax(uniqueness_t)
  + endpoint_bonus
```

snapshot bank 为 4 帧。bank 满后，候选只有显著优于低 utility snapshot，或旧
snapshot 超过 24 帧未更新时才替换。

该机制直接受 Echo-Forcing coherent relevance/uniqueness snapshot 启发。
我们的实验差异是：

- 在旧 v98 二角色 head 内分别选择；
- 作为 per-head KV middle cache，而不是 scene pool；
- 不依赖 prompt 指定 recall id；
- 不做跨帧空间 token stitching；
- 固定 sink/middle/recent token 预算。

它是可用候选，但如果最终采用，论文必须明确引用 Echo-Forcing，不能把
relevance/uniqueness snapshot 本身声明为原创。

### 4.3 Bounded Semantic Retrieval

每个 clean block 最多向 archive 加入一个候选。archive 最多保存
`max(8, 3 * read_k)` 个真实帧；首尾帧受保护，冗余且低 utility 的中间帧优先
淘汰。

读取时先排除 sink 与 recent，再按当前 clean query 做：

```text
relevance = cosine(z_query, z_archive)
MMR =
  0.80 * unit(relevance)
  + 0.20 * diversity_to_selected
```

只读 top-2 或 top-4。完整 archive 永远不会直接注入 attention。

该机制借鉴 LongLive-RAG 的“当前描述子检索非近期历史”思想，但不使用其
learned retrieval autoencoder，也不复制 CPU memory 实现。

这是高风险探索项，不是默认主方法。历史 ProbeCache 已经显示直接历史检索可能
保持 DINO identity，却产生背景幻觉、重复主体和多边形噪声。v115 的限制只是
降低风险，不能提前假设它有效。

### 4.4 Sparse Snapshot 75%

先按 4.2 选择 snapshot，再保留 75% spatial tokens：

```text
token_score =
  0.60 * normalized spatial deviation
  + 0.40 * normalized adjacent V change
```

保留 token 的一半用于均匀空间覆盖，另一半取最高 token score。所有 token
保留原始 position。四个 sparse snapshot 的中期预算约等于 3 个完整帧，因此
搭配 `recent5`。

该项借鉴 `docs/flash_vareason.md` 中的相邻冗余、唯一性和 core-token 思想，
但没有复制音频模型、公式或代码。视频生成 attention 对空间稀疏历史更敏感，
该项可能产生多边形纹理，因此只作为高风险实验。失败时必须记录为负结果，不能
隐去。

### 4.5 Motion Pair 1 + Recent 6

Suppressive 的轻量 motion cache 只保留一个语义一致的高运动相邻帧对：

```text
sink1 + one adjacent pair + recent6
```

与 v111 的 `motion_pair2 + recent4` 相比，它减少远期运动事件，增加近期连续
上下文。这样更符合 Suppressive 中 Veil-like heads 较多、对旧历史注入敏感的
诊断，同时仍保留一个明确的运动方向证据。

这不是 cyclic 或固定 lag。pair 由 clean-V motion quantile 和 semantic gate
在线触发。

## 5. v115 16-cell 单视频矩阵

统一使用 MovieGenVideoBench prompt 0、seed 0、120 latent output frames、
477 decoded frames、16 FPS、约 30 秒。

### 5.1 Supportive sweep

Suppressive 固定为最安全的 `sink1 + recent8`。

| Cell suffix | Supportive middle | 目的 |
|---|---|---|
| `support_prototype4_suppress_recent8` | prototype4 | 测试安全的语义段 medoid 压缩 |
| `support_snapshot4_suppress_recent8` | snapshot4 | 测试完整帧 relevance/uniqueness snapshot |
| `support_retrieval2_suppress_recent8` | retrieval2 + recent6 | 测试保守检索 |
| `support_retrieval4_suppress_recent8` | retrieval4 + recent4 | 检索强度对照 |
| `support_sparse75_suppress_recent8` | sparse snapshot4 + recent5 | 测试 token 压缩风险 |

应与已有 `support_landmark4_suppress_recent8` 比较，不重新生成该视频。

### 5.2 Suppressive sweep

Supportive 固定为 v111 最稳的 `landmark4`。

| Cell suffix | Suppressive middle | 目的 |
|---|---|---|
| `support_landmark4_suppress_prototype2` | prototype2 + recent6 | 中期语义压缩 |
| `support_landmark4_suppress_snapshot2` | snapshot2 + recent6 | 两个唯一性快照 |
| `support_landmark4_suppress_retrieval2` | retrieval2 + recent6 | 保守非近期 recall |
| `support_landmark4_suppress_motion_pair1` | motion pair1 + recent6 | 近期连续性加单个运动事件 |
| `support_landmark4_suppress_sparse75` | sparse snapshot4 + recent5 | Suppressive token 压缩压力测试 |

应与已有 `support_landmark4_suppress_recent8` 和
`support_landmark4_suppress_motion_pair2` 比较。

### 5.3 Joint candidates and same-route controls

| Cell suffix | Supportive | Suppressive |
|---|---|---|
| `support_prototype4_suppress_motion_pair1` | prototype4 | motion pair1 |
| `support_snapshot4_suppress_motion_pair1` | snapshot4 | motion pair1 |
| `support_retrieval2_suppress_motion_pair1` | retrieval2 | motion pair1 |
| `support_sparse75_suppress_motion_pair1` | sparse75 | motion pair1 |
| `all_prototype4_control` | prototype4 | prototype4 |
| `all_snapshot4_control` | snapshot4 | snapshot4 |

same-route control 用于判断视频改善是否真的依赖 304/56 路由，而不是任何 head
统一使用新 memory 都能得到。

## 6. 服务器运行

先更新并检查代码：

```bash
cd /path/to/training-free
git pull

python -m pytest -q \
  tests/test_v115_role_memory_strategy_contract.py \
  tests/test_v115_runner_contract.py \
  tests/test_v115_trace_analysis.py \
  tests/test_v111_role_event_cache_contract.py \
  tests/test_v111_runner_contract.py

export PF_CHECKPOINT="$PWD/third_party/Pyramid-Forcing/checkpoints/self_forcing_dmd.pt"
export OUT_ROOT="$PWD/runs/v115_role_memory_cache_1video"
```

四节点共享同一个 `OUT_ROOT`，每个节点分别设置 rank：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3 \
python scripts/run_v115_role_memory_cache_1video.py all
```

16 个 cell 在四节点上各分到 4 个。每个 cell 只生成一个视频。不要为了占满
32 卡重复 seed 或 prompt。

如果需要分阶段先跑低风险项：

```bash
python scripts/run_v115_role_memory_cache_1video.py support \
  --gpu-list 0,1,2,3,4

python scripts/run_v115_role_memory_cache_1video.py suppress \
  --gpu-list 0,1,2,3,4

python scripts/run_v115_role_memory_cache_1video.py joint \
  --gpu-list 0,1,2,3
```

每个 mode 使用独立 frozen contract。不要修改同一 `OUT_ROOT` 下已经完成 cell
的配置。

完成后运行：

```bash
python scripts/analyze_v115_role_memory_traces.py \
  --run-root "$OUT_ROOT"
```

## 7. 必须观察的日志和 trace

### 7.1 通用 hard gates

任一条件成立立即判为实现失败：

- `Traceback`、OOM 或 trace writer error；
- `[HistoryPolarityPolicy]` 缺失；
- `legacy_pf_labels=false` 或 `exclusive_owner=true` 缺失；
- runtime head count 不是 `10:304,11:56`；
- sink/middle/recent overlap；
- 实际 token 超过 9-frame-equivalent；
- role context 的 head ids 与 map 不一致；
- video 不是 477 帧、16 FPS、832 x 480；
- 背景多边形噪声、色块扩散或持续重复主体。

### 7.2 Prototype

检查：

- `prototype_spans`
- `prototype_medoid_ids`
- `prototype_counts`
- `compressed_count`, `created_count`, `evicted_count`
- 每帧 action 的 semantic similarity、motion 和 online threshold

预期：

- `prototype_counts` 中应出现大于 1 的值；
- 不能所有帧都新建 prototype，否则压缩没有生效；
- 不能长时间只有一个 prototype，否则可能过度压缩和冻结运动；
- medoid 必须位于对应 `[start_t, end_t]`。

### 7.3 Snapshot

检查：

- 每个 block 的 `candidate_scores`
- relevance、uniqueness 和 utility
- `snapshot_admit`、`spacing_gate`、`replacement_gate`
- victim 与 stale refresh
- snapshot token count

若所有候选 utility 完全相同，说明描述子或归一化可能失效。

### 7.4 Retrieval

检查：

- archive occupancy 和 eviction；
- eligible/gated 数量；
- selected frame ids、similarity、MMR；
- top-2 与 top-4 是否真的有不同 read 数量；
- retrieved frame 是否落入 recent；
- 检索相似度很低时是否仍强制 recall。

若 top-4 视频出现噪声而 top-2 干净，应优先保留 top-2 作为检索强度上限，不要
把问题归因于 head 分类。

### 7.5 Sparse snapshot

检查：

- role trace 中 `token_score_summary`；
- 每个 snapshot token count 是否等于 `ceil(0.75 * frame_seqlen)`；
- 原始 position 数量是否匹配 token 数；
- policy trace 的实际 token 总预算；
- 空间纹理、轮廓和背景是否出现栅格或多边形噪声。

### 7.6 Motion pair1

检查：

- `pair_capacity=1`
- bank 中仅一个相邻 pair；
- pair endpoint 相差 1；
- motion threshold、semantic score 和 replacement reason；
- recent frame count 为 6，不是 4。

## 8. 人工 review 与晋级规则

把 v115 16 个视频与以下 v111 旧视频放在一起盲审：

- `all_recent8`
- `all_landmark4`
- `support_landmark4_suppress_recent8`
- `support_landmark4_suppress_motion_pair2`

每个视频记录：

| 维度 | 记录 |
|---|---|
| 0-10s / 10-20s / 20-30s identity | 1-5 |
| subject count | stable / transient duplicate / persistent duplicate |
| motion amount | frozen / reduced / normal / excessive |
| motion plausibility | 1-5 |
| background/layout | 1-5 |
| polygon/grid artifact | none / mild / severe |
| first failure time | seconds |
| overall rank | 1-N |

晋级优先级：

1. 任何明显多边形噪声直接淘汰。
2. identity、motion、background 均不差于 `landmark4+recent8`。
3. 若视觉相近，优先选择机制最简、因果对照完整、与 PF 差异清楚的方法。
4. `prototype+motion1` 是默认故事候选。
5. `snapshot+motion1` 是第二候选，但与 Echo-Forcing 的关系更近。
6. retrieval 只有在无幻觉且明显有收益时保留。
7. sparse75 只有在完全无空间 artifact 时保留。
8. 最多选择 3 个 candidate 加 1 个 control 进入 16-prompt。

## 9. 结果分支

### A. Prototype 可用

主方法优先为：

```text
History-polarity binary heads
+ Supportive temporal prototype memory
+ Suppressive coherent motion pair and enlarged recent window
```

论文主张聚焦“二角色 head 与内容驱动双时间尺度 memory 的耦合”，不再依赖 PF
三分类、stride、cyclic 或 Merge。

### B. Snapshot 可用但 Prototype 较差

可使用 role-conditioned coherent snapshot，但必须把 Echo-Forcing 列为直接机制
先例。创新只能来自二角色分配、固定预算 cache ownership 和单 prompt 长外推
场景中的组合及实验证据。

### C. 只有 Landmark + Recent 可用

不要把新压缩/检索写成贡献。可以保留：

- 旧 v98 二角色分类的诊断；
- Supportive content-driven landmark；
- Suppressive enlarged recent；
- role-neutral、随机和 inverted map 对照。

但该故事的创新强度有限，需要尽快做 16-prompt 证据，再判断是否回到 v78 或
PF-based 最优结果。

### D. Retrieval 或 sparse 出现 artifact

记录负结果并关闭对应路径。不要继续扩大 prompt 数量，也不要用平均指标掩盖
人工不可用的视频。

## 10. 论文故事候选

当前最清楚的候选故事不是“我们修改了 PF cache”，而是：

> 长视频 AR attention heads 对历史的作用方向不同。Supportive heads 需要跨
> 时间维持可复用的语义状态，Suppressive heads 对过期外观更敏感，更依赖近期
> 动态证据。我们用 history-polarity 形成二角色划分，并在同一固定 token 预算
> 下，为两类 head 构造内容驱动的 prototype/motion cache，从而避免周期性采样
> 和合成 K/V 压缩。

可以写成三个技术点，但必须由后续实验支持：

1. **History-polarity head partition**：二分类标准、304/56 membership 和与
   PF Anchor/Wave/Veil 的 post-hoc cross-tab。
2. **Role-conditioned dual-timescale cache**：Supportive 长期语义 prototype，
   Suppressive 近期窗口加单个 coherent motion event。
3. **Budgeted clean-KV event update**：clean-only admission、真实帧 medoid、
   token 精确预算、exclusive ownership、原始 position sidecar 和可审计更新。

旧 v98 分数不是 shift-invariant，因此第 1 点目前仍是 frozen diagnostic map。
论文最终声称“发现新的 head taxonomy”前，必须补独立校准、阈值稳定性、
random/inverted/count-matched 对照。缓存效果可以先验证，但不能把分类缺陷藏在
cache 结果中。

## 11. 参考工作与学术边界

| 工作 | 借鉴内容 | 当前实现的关键差异 | 禁止 claim |
|---|---|---|---|
| [Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | SF/PF runtime、ragged per-head cache、dynamic RoPE、sink/middle/recent infrastructure | 旧 v98 两角色 map；prototype/snapshot/retrieval/motion event；候选无 PF stride/cyclic/Merge | 不能把 per-head heterogeneous cache 或 PF runtime 说成原创 |
| [Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) | coherent relevance/uniqueness snapshot、scene memory 思路 | 当前 snapshot 是角色内 per-head KV middle；无 scene pool、无 prompt recall id、无 token stitching | 不能把 relevance/uniqueness snapshot 本身说成原创 |
| [LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG) | query-to-history descriptor retrieval、recent exclusion | bounded candidate archive、MMR top-k、无 learned AE、无 CPU history 实现 | 不能把 top-k historical retrieval 说成原创 |
| `docs/flash_vareason.md` | 相邻冗余压缩、唯一性和 core-token 概念 | 视频 KV 使用真实 medoid；sparse 分支保留原 position，不复制音频模型或公式 | 不能把 Flash-VAReason 的音频压缩贡献移作本工作贡献 |
| Head Forcing / Forcing-KV | head heterogeneity 与 head-specific memory 先例 | 当前二角色标准、membership、任务和 cache 路由不同 | 不能声称首次发现 head heterogeneity 或首次做 head-wise cache |

所有最终论文文字必须区分：

- borrowed prior idea；
- 本项目的 adaptation；
- 本项目真正由实验支持的新组合或新机制；
- 尚未验证的假设。

代码位置：

```text
third_party/Pyramid-Forcing/pyramidkv/role_memory.py
third_party/Pyramid-Forcing/pyramidkv/role_event.py
third_party/Pyramid-Forcing/pyramidkv/factory.py
third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py
third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py
scripts/run_v115_role_memory_cache_1video.py
scripts/analyze_v115_role_memory_traces.py
```
