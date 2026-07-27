# v117：v115 结果后的候选收缩与 cache 实验优先级

日期：2026-07-27

状态：v115 人工 review 已完成；v116 MovieBench-16 代码已按本计划更新。

## 1. 对 v115 结论的精确修正

v115 的单 prompt 结果支持以下结论：

1. `Supportive=Landmark4` 是当前最稳定的长期缓存。
2. `Supportive=Prototype4` 也可用，明显优于 Supportive-Retrieval、
   Supportive-Sparse75，并值得保留为替代主线。
3. 在 Landmark4 支撑下，Suppressive 的 Recent8、Motion-pair1、
   Motion-pair2、Prototype2、Snapshot2、Retrieval2 和 Sparse75 均未出现立即
   崩溃。
4. 第 3 点只表示这些配置通过了一个 prompt 的可用性筛选，不表示它们视觉等价，
   也不表示 Suppressive cache 无效。人工 review 已观察到差异，但单样本不足以
   稳定命名这些差异。
5. Supportive-Retrieval 和 Supportive-Sparse75 的失败不能直接外推到
   Suppressive-Retrieval/Sparse75。后者只影响 56 个 heads，而且在 Landmark4
   支撑下生成干净，因此仍应进入多 prompt 指标比较。

因此，下一步不是继续做大量单 prompt cache 组合，也不是现在验证 all-head
control，而是固定 Supportive，放大 Suppressive cache 的统计差异。

## 2. v116 默认九方法矩阵

统一使用冻结的 MovieGenVideoBench-16、seed 0、30 秒视频。

### 2.1 Suppressive 主比较

| Method key | Supportive | Suppressive | 主要观察 |
|---|---|---|---|
| `landmark_recent8` | Landmark4 + Recent4 | Recent8 | 无 middle 的局部参考 |
| `landmark_motion2` | Landmark4 + Recent4 | Motion-pair2 + Recent4 | 较强运动事件记忆 |
| `landmark_motion1` | Landmark4 + Recent4 | Motion-pair1 + Recent6 | 较弱运动事件与更多近期上下文 |
| `landmark_prototype2` | Landmark4 + Recent4 | Prototype2 + Recent6 | 中期语义段压缩 |
| `landmark_snapshot2` | Landmark4 + Recent4 | Snapshot2 + Recent6 | relevance/uniqueness 快照 |
| `landmark_retrieval2` | Landmark4 + Recent4 | Retrieval2 + Recent6 | 当前状态相关的非近期读取 |
| `landmark_sparse75` | Landmark4 + Recent4 | 4 x 75% sparse + Recent5 | 小比例 heads 的 token 压缩 |

这七项只改变 Suppressive cache。其余 prompt、seed、head map、Supportive cache、
总 token 预算和代码路径保持一致。

### 2.2 Supportive 替代候选

| Method key | Supportive | Suppressive | 目的 |
|---|---|---|---|
| `support_prototype_recent` | Prototype4 + Recent4 | Recent8 | Landmark 与 Prototype 的直接候选比较 |
| `prototype_motion1` | Prototype4 + Recent4 | Motion-pair1 + Recent6 | 检查 Prototype 与轻量运动缓存的组合 |

Snapshot4、Retrieval4 和 Sparse75 不再作为 Supportive 主候选，因为 v115 已给出
一致的放大、双主体或冻结负证据。

## 3. 指标如何对应人工差异

不能用单一总分选方法。需要联合以下证据：

| 现象 | 主要指标 | 必须排除的伪改进 |
|---|---|---|
| ID/脸部保持 | DINO、ArcFace、subject consistency | 靠冻结或主体持续放大获得高分 |
| 背景稳定 | background consistency、drift | 背景固定但动作消失 |
| 动作连续 | motion smoothness、flicker | smooth 但 dynamic degree 过低 |
| 动作幅度 | VBench dynamic degree、人工 review | 高频抖动被误判为动态 |
| 重复/回环 | loop score、后半段人工 review | 主体复制后仍有高 DINO |
| Prompt 完成度 | CLIP、人工事件检查 | 只保留主体但不执行复杂动作 |

`scripts/analyze_v116_candidate_metrics.py` 以 `landmark_recent8` 为参考做逐 prompt
配对比较。每项指标报告：

- raw mean delta；
- 方向归一化后的 mean/median improvement；
- 16 prompts 的 wins/ties/losses；
- 固定 seed bootstrap 95% 区间。

这能把“看起来不完全一样但难描述”转成可定位到 prompt 和指标的证据。

## 4. 立即运行顺序

### 4.1 先恢复 v115 审计状态

8 个缺 marker 的视频不需要重生成。先只做审计：

```bash
python scripts/recover_v115_done_markers.py \
  --run-root "$PWD/runs/v115_role_memory_cache_1video" \
  --dry-run

python scripts/recover_v115_done_markers.py \
  --run-root "$PWD/runs/v115_role_memory_cache_1video"
```

只有现有视频、日志、policy trace、role-event trace 和冻结 contract 全部通过时，
第二条命令才补 marker。缺 trace 的 cell 保持失败状态；其视频可用于人工参考，
但不能作为论文机制证据。该工具不会启动推理。

### 4.2 四节点生成 v116

四个节点使用完全相同的方法字符串：

```bash
export V115_PROMOTION_APPROVED=1
export V116_METHODS="landmark_recent8,landmark_motion2,landmark_motion1,landmark_prototype2,landmark_snapshot2,landmark_retrieval2,landmark_sparse75,support_prototype_recent,prototype_motion1"
export NUM_NODES=4
export GPU_LIST="0,1,2,3,4,5,6,7"

NODE_RANK=<0|1|2|3> \
python scripts/run_v116_role_memory_diverse16.py generate
```

共 144 个视频，每节点 36 个任务。默认输出：

```text
runs/v116_role_memory_diverse16/m9_7a14c511d500
```

生成、发布审计、VBench-Long 和辅助指标的完整命令见
`docs/116_v116_moviebench16_evaluation_runbook.md`。

## 5. 还需要哪些 cache 策略实验

### P0：现在必须做

只有当前 v116 Suppressive 七路比较。它直接决定：

- Suppressive 是否只需 Recent；
- 是否需要显式 motion event；
- 语义压缩是否优于运动缓存；
- Snapshot/Retrieval/Sparse 在少量 heads 上是否仍有价值。

在该结果出来前继续组合 cache，会把 Supportive 和 Suppressive 的作用重新混杂。

### P1：主候选确定后，只调一个容量轴

根据赢家选择一组，不要全部运行：

- Motion 胜出：`pair0/recent8`、`pair1/recent6`、`pair2/recent4`；
- Prototype 胜出：`prototype1/recent7`、`prototype2/recent6`、
  `prototype4/recent4`；
- Snapshot 胜出：`snapshot1/2/4`，并保持总 token 预算；
- Retrieval 胜出：先测 `top1/recent7` 对 `top2/recent6`，不再测试 top4；
- Sparse 胜出：只测 `90%` 对 `75%`，并检查 dynamic degree。

这是 cache 容量和 recent 窗口的因果消融，也是论文需要的预算曲线。

### P2：更新生命周期

只对 P1 赢家测试：

1. 当前内容驱动更新；
2. 加入最大年龄刷新；
3. 去掉 replacement/novelty gate。

历史 v78 说明适度 update budget 和 max-age 可能减小 temporal jump，但不能直接
把 PF 的 per-head transition mask 接到当前共享角色 memory 上：它会让同一角色
的 heads 使用不同 bank，破坏共享选择假设。若采用 transition，必须先实现
role-synchronized commit，再做实验。

### P3：Sink 与长期生命周期

当前候选统一使用固定 `sink1`，并由 Landmark/Prototype middle 提供长期状态。
主候选确定后可做一次预算匹配的：

```text
fixed sink1
vs sink3 with two fewer middle/recent frame-equivalents
vs sink1 + age-refreshed adaptive landmark
```

不要同时改变 sink、middle 类型和 recent 数量。更新 sink 是高风险操作，必须保留
frame 0 或独立身份参考，否则可能把长期身份一起刷新掉。

### P4：更长视频和 A-B-A

顺序必须是：

1. MovieBench-16 选定 cache；
2. 该 cache 做 60/90 秒单 prompt；
3. 同一实现做 A-B-A，并与 Echo-Forcing 对比；
4. 最后 MovieBench-128 主表。

A-B-A 可以增加 scene-boundary 时的 bank archive/restore，但不能为了该任务改变
单 prompt 主方法。单 prompt 长视频仍是论文第一任务。

## 6. 暂时不做的实验

- Supportive Retrieval top1/recency bias：v81/v82 和 v115 已有重复的非 ID
  幻觉、放大和降运动风险。
- Supportive Sparse90：75% 已冻结，且 token 稀疏不是当前质量瓶颈。
- 更多 Snapshot admission 阈值：先看 56-head Suppressive-Snapshot 的多 prompt
  结果。
- 任意 Landmark/Prototype/Snapshot 三者混合：缺少单因素解释。
- all-head、random、inverse、类别数量匹配：这些是主方法确定后的分类消融，
  不是当前候选筛选。

## 7. 论文故事的当前候选

在 v116 结果前，最稳妥的故事骨架是：

1. **History-polarity head discovery**：用与 PF 三分类不同的二元历史响应标准，
   找到长期 Supportive 与历史抑制/响应型 heads。
2. **Role-conditioned temporal memory**：Supportive 读取长期语义 Landmark 或
   Prototype；Suppressive 在相同总预算下选择更局部的 state/motion memory。
3. **Content-driven lifecycle**：所有 middle memory 只由 clean K/V 的语义、
   运动、唯一性或检索信号更新，不使用 stride/cyclic/Merge。
4. **Evidence-driven cache choice**：v116 的七路 Suppressive 比较决定第 2 点最终
   使用 Recent、Motion、Prototype、Snapshot、Retrieval 还是 Sparse，而不是
   先写故事再挑结果。

若 Snapshot/Retrieval 被选中，必须引用 Echo-Forcing/LongLive-RAG；若使用
Prototype，需要说明它受时序压缩思想启发但采用真实帧 medoid、没有 KV averaging。
不得把相关工作的基础组件重新命名后声称原创。
