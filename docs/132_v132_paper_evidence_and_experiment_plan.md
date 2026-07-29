# v132：最终方法证据、32 卡实验计划与论文并行安排

Date: 2026-07-29

Status: 主方法暂时冻结；代码已覆盖现有结果配对统计、head partition 审计、
二分类与双缓存消融、30/60 秒 VBench-Long、以及盲评包生成。本计划优先补齐
论文必需证据，不再继续无边界搜索 cache 配置。

## 1. 当前暂定主方法

当前主方法固定为：

```text
History-Supportive / History-Suppressive 二分类

Supportive (304 heads):
  sink1 + TemporalPrototype4 + recent4

Suppressive (56 heads):
  sink1 + Retrieval1(age <= 24 latent frames) + recent7

两类均使用：
  clean K/V admission
  exact-frame K/V（不平均或合成 K/V）
  original temporal-position sidecar
  exclusive cache ownership
  9 full-frame-equivalent read budget
```

对应方法 key：

```text
ours_prototype_retrieval_age24
```

这就是 v125/v129 中已经完成 128 prompts × 30 s 评测的配置，不添加
confidence gate，也不添加 MotionPair 或 Value calibration。

v129 的关键结果如下：

| Method | Dynamic Degree | Motion Smoothness | Overall Consistency | Quality Score |
|---|---:|---:|---:|---:|
| `sf_native` | 43.28 | 98.57 | 23.47 | 81.62 |
| `deep_forcing` | 55.52 | 98.27 | 23.81 | 82.25 |
| `rolling_forcing` | 34.58 | 98.80 | 24.07 | 82.09 |
| `longlive` | 41.98 | 98.69 | 24.14 | 82.16 |
| **`ours_prototype_retrieval_age24`** | **61.72** | 98.38 | 23.70 | **82.95** |
| `ours_confidence_motion` | **62.19** | 98.31 | 23.69 | 82.88 |

因此，现有 128-prompt 结果足以：

1. 冻结一个效果较好的主方法；
2. 开始写 Introduction、Method、Related Work 和主结果表；
3. 支持“相对 SF 兼顾质量与运动，而不是通过冻结视频维持一致性”的初步叙述。

但它还不足以完成投稿，因为尚缺：

1. per-prompt paired statistics；
2. head partition 与双缓存的因果消融；
3. 60 秒外推；
4. Semantic Score / Total Score；
5. 对 identity、背景、运动与 artifact 的盲评证据。

## 2. 可以写入论文的三个技术点

### 2.1 History-polarity binary head discovery

对第 \(h\) 个 head，离线统计其对历史 token 的归一化 signed Q-K mass：

```text
rho_h = median_records(
          sum(history QK logits)
          / sum(abs(history QK logits))
        )
```

然后使用自然零点进行分类：

```text
rho_h >= 0  -> History-Supportive
rho_h <  0  -> History-Suppressive
```

最终得到 304/56 个 heads。该分类不读取 PF 的 Anchor/Wave/Veil 标签。
PF 标签只能用于分类完成后的解释性 cross-tab：

| PF class（仅 post-hoc） | Supportive | Suppressive |
|---|---:|---:|
| Anchor | 169 | 3 |
| Wave | 133 | 23 |
| Veil | 2 | 30 |

这说明二分类找到的是一个不同问题下的功能轴：历史信息对该 head 的净支持方向，
而不是复现 PF 的三类时间采样模式。不能写成“PF 三分类的重命名”，也不能声称
首次发现 attention-head heterogeneity。

当前 score 仍存在 common-logit shift 不变性不足的理论局限。因此论文中更稳妥的
表述是 **an empirically discovered history-polarity partition**，而不是普适的
head taxonomy。阈值稳定性、random、all-head 和 inverted 对照用于限定这一 claim。

### 2.2 Role-conditioned dual-timescale memory

Supportive heads 使用长期 TemporalPrototype：

1. clean K/V descriptor 相似度不低于 0.985；
2. 相邻帧连续；
3. motion 不高于在线 motion history 的 70% 分位数；
4. 同时满足时，将新帧压入相邻 temporal segment；
5. 每个 segment 只保留最接近 descriptor centroid 的真实帧 medoid；
6. 容量超过 4 时，按 novelty、segment duration 和 motion utility 淘汰，
   并优先保护最早 prototype。

Suppressive heads 使用有界内容检索：

1. 每个 clean block 只向有限 archive 接纳一个 relevance/novelty 较高的真实帧；
2. recent 与 sink 在读取时被显式排除，防止同一帧重复占预算；
3. query 由当前 clean K/V descriptor 产生；
4. 只读取一个最相关历史帧；
5. `age <= 24` latent frames，即约 6 秒，避免后期强制注入过旧外观；
6. 当前主方法不使用 confidence abstention；v129 表明该 gate 并未稳定改善主配置。

两类 memory 的区别不是简单 stride/cyclic，也不沿用 PF 的三路 cache。长期
Supportive heads 获得跨段结构摘要；较少的 Suppressive heads 获得当前内容相关、
但受年龄约束的中期状态。

### 2.3 Auditable fixed-budget lifecycle

所有方法均遵守同一 9-FFE 读取预算，并显式实现：

```text
sink owner + middle-memory owner + recent owner = one exclusive role owner
```

PF legacy dynamic history 不能与新 middle memory 同时生效。每个 middle item
保留原始 K/V、原始时间位置和来源类型，不执行 KV averaging。代码输出每层、
每角色的 admission、compression、eviction、retrieval age、cache composition、
budget overflow 和 ownership violation 日志。

这部分可作为实现可靠性和训练免费部署机制的技术点，但不能把 dynamic RoPE、
sink/recent 或 per-head ragged cache 本身声称为原创；这些基础设施来自
Self-Forcing/Pyramid-Forcing 代码路径。

## 3. 不进入当前主方法的小机制

### Confidence-gated retrieval

v129 中 `cosine >= 0.55` 几乎不拒绝候选，真正起作用的是 margin gate。相对无 gate：

- `confidence_recent` 的 Dynamic Degree 和 Quality Score 均下降；
- `confidence_motion` 只获得很小的运动变化，复杂度与收益不匹配。

因此 gate 保留为诊断/补充实验，不放入 Method 主线。

### MotionPair

MotionPair 候选说明运动 companion 可以改变 dynamics，但当前
`prototype_retrieval_age24` 已获得最高 Quality Score 和接近最高 Dynamic Degree。
在没有 identity 盲评证明 MotionPair 更优前，不再增加主方法组件。

### Historical Value calibration

v129 non-cache add-on 只完成小规模筛选。只有在 16 prompts 上同时提升 identity，
且 Dynamic Degree、连续 flow、曝光和 artifact 不退化时，才允许作为 optional
refinement；否则不进入论文主方法。

### A-B-A / scene switching

当前论文的关键任务是单 prompt 长视频外推。A-B-A、Echo-Forcing 对照和
scene-memory reset 全部延后，只在主实验结束后仍有空余 GPU 时运行。

## 4. P0：立即运行，不占生成 GPU

以下三项可以与任何生成实验并行。

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull origin codex/v98-correctness-fixes

python scripts/run_v132_existing_paired_analysis.py
python scripts/summarize_v132_head_partition.py
python scripts/prepare_v132_blind_review.py
```

输出：

```text
runs/v129_paper_comparison_30s/metrics/v132_paired/
  paired_metrics.json
  paired_metrics.md

runs/v132_head_partition_evidence/
  head_partition_report.json
  head_partition_report.md

runs/v132_blind_review16/
  public/
  private/
  source_subset/selection_manifest.json
```

盲评默认比较：

```text
sf_native
deep_forcing
ours_prototype_retrieval_age24
ours_confidence_motion
```

`private/` 中包含解盲信息。评分完成前不要查看或分发该目录。

## 5. P1：先做 16-prompt correctness screen

默认只筛四个论文必需消融：

| Key | 唯一变化 | 回答的问题 |
|---|---|---|
| `random_binary` | 304/56 数量相同，membership 随机 | head membership 是否携带信号 |
| `all_supportive` | 360 heads 全部读 Prototype | 二角色异构性是否必要 |
| `no_prototype` | Supportive 改为预算匹配 recent | 长期 prototype 是否必要 |
| `no_retrieval` | Suppressive 改为预算匹配 recent | 有界检索是否必要 |

四种方法 × 16 prompts = 64 个 30 秒视频，在 32 卡上每卡两个。

所有节点使用相同环境，仅修改 `NODE_RANK=0..3`。先在 node 0 执行 preflight，
再启动其余节点。

```bash
export REPO_ROOT="$PWD"
export NUM_NODES=4
export GPU_LIST=0,1,2,3,4,5,6,7
export V132_SCOPE=screen16
export V132_METHODS=random_binary,all_supportive,no_prototype,no_retrieval

# node 0 first, then all nodes
export NODE_RANK=<0|1|2|3>
python scripts/run_v132_binary_memory_ablation.py preflight
python scripts/run_v132_binary_memory_ablation.py generate
```

全部完成后仅在 node 0：

```bash
export NODE_RANK=0
python scripts/run_v132_binary_memory_ablation.py audit
```

人工 review 的 hard gate：

1. 任意 polygon noise、背景几何块、全局色偏或 cache jump，立即拒绝；
2. 检查 15–30 秒的 identity、服饰/物体几何、背景漂移；
3. 检查有效主体/镜头运动，不能以冻结换一致性；
4. 检查循环、第一姿态返回、突然放大和旧场景回闪；
5. 检查 `diagnostics/` 中 ownership、budget、source count 和位置 sidecar。

如果四项均干净，直接进入 full128；肉眼差异很小时不再增加一轮 16-prompt
VBench，因为最终 128-prompt paired evaluation 更有论文价值。

`inverted_binary` 与 `all_suppressive` 已有代码，但不是首批必需项。只有首批
screen 干净且仍有资源时再补：

```bash
export V132_METHODS=inverted_binary,all_suppressive
export V132_SCOPE=screen16
```

## 6. P2：两组 16 卡并行生成

screen 通过后，将四个节点分成两组。这样 30 秒消融和 60 秒确认可以同时完成，
并为论文写作保留等待时间。

### 6.1 物理 node 0–1：四个消融 full128

两台机器内部重新编号为 `NODE_RANK=0,1`：

```bash
export NUM_NODES=2
export NODE_RANK=<0|1>
export GPU_LIST=0,1,2,3,4,5,6,7
export V132_SCOPE=full128
export V132_METHODS=random_binary,all_supportive,no_prototype,no_retrieval

python scripts/run_v132_binary_memory_ablation.py preflight
python scripts/run_v132_binary_memory_ablation.py generate
```

它与 screen16 使用同一 frozen contract 和输出 root：

```text
runs/v132_binary_memory_ablation30/controls4_790fe870ce2c/
```

已经完成的 16 prompts 会按 marker/contract 校验后跳过，而不是重新生成。

完成后在该组 rank 0：

```bash
export NODE_RANK=0
python scripts/run_v132_binary_memory_ablation.py audit
python scripts/prepare_v132_ablation_comparison.py
```

assembler 直接复用 v125 的 `sf_native` 和主方法视频，不重新生成它们。最终
比较含 6 个方法、768 个引用/新生成视频：

```text
sf_native
ours_main
random_binary
all_supportive
no_prototype
no_retrieval
```

### 6.2 物理 node 2–3：SF / Ours 的 60 秒确认

这两台机器也独立编号为 `NODE_RANK=0,1`：

```bash
export NUM_NODES=2
export NODE_RANK=<0|1>
export GPU_LIST=0,1,2,3,4,5,6,7

python scripts/run_v132_moviebench128_60s.py preflight
python scripts/run_v132_moviebench128_60s.py generate
```

它只生成：

```text
sf_native
ours_prototype_retrieval1_age24
```

共 256 个 60 秒视频。完成后在该组 rank 0：

```bash
export NODE_RANK=0
python scripts/run_v132_moviebench128_60s.py audit
python scripts/prepare_v132_long60_comparison.py
```

输出：

```text
runs/v132_moviebench128_60s/ours1_bd5b61eefc10/
runs/v132_main_60s_comparison/
```

## 7. P3：VBench-Long

生成完成后可重新使用四节点共同评测。

### 7.1 30 秒消融

所有节点：

```bash
export NUM_NODES=4
export NODE_RANK=<0|1|2|3>
export GPU_LIST=0,1,2,3,4,5,6,7
export COMPARISON_ROOT="$PWD/runs/v132_binary_memory_ablation_comparison_30s/controls4_790fe870ce2c"
export V132_METRIC_PROFILE=core

bash scripts/run_v132_ablation_vbench.sh split
bash scripts/run_v132_ablation_vbench.sh preflight
bash scripts/run_v132_ablation_vbench.sh eval
```

仅 node 0：

```bash
export NODE_RANK=0
bash scripts/run_v132_ablation_vbench.sh collect
python scripts/run_v132_ablation_paired_analysis.py \
  --comparison-root "$COMPARISON_ROOT"
```

### 7.2 60 秒主比较

所有节点：

```bash
export NUM_NODES=4
export NODE_RANK=<0|1|2|3>
export GPU_LIST=0,1,2,3,4,5,6,7
export COMPARISON_ROOT="$PWD/runs/v132_main_60s_comparison"
export V132_METRIC_PROFILE=core

bash scripts/run_v132_long60_vbench.sh split
bash scripts/run_v132_long60_vbench.sh preflight
bash scripts/run_v132_long60_vbench.sh eval
```

仅 node 0：

```bash
export NODE_RANK=0
bash scripts/run_v132_long60_vbench.sh collect
```

30 秒视频被切为 15 个 2 秒 clips；60 秒视频切为 30 个 clips。split manifest
记录 comparison hash、VBench commit、prompt count 和 clip count，避免误用旧缓存。

### 7.3 现有 v129 的 Semantic Score / Total Score

这一项不重新生成视频。若模型已经缓存，它的优先级高于 A-B-A：

```bash
export NUM_NODES=4
export NODE_RANK=<0|1|2|3>
export GPU_LIST=0,1,2,3,4,5,6,7
export COMPARISON_ROOT="$PWD/runs/v129_paper_comparison_30s"
export V129_METRIC_PROFILE=semantic_extension

bash scripts/run_v129_vbench_long.sh preflight
bash scripts/run_v129_vbench_long.sh eval
```

仅 node 0：

```bash
export NODE_RANK=0
bash scripts/run_v129_vbench_long.sh collect
```

## 8. 结果选择标准

不能只看一个总分，也不能因为某方法运动更低而误判为 identity 更好。

主结果至少联合：

1. official VBench Quality Score、Semantic Score、Total Score；
2. Dynamic Degree 与 Motion Smoothness；
3. subject/background/overall consistency；
4. 128 prompts 的 paired bootstrap CI 与 paired permutation test；
5. 盲评 identity、background、motion、artifact 和 overall rank；
6. cache diagnostics 中的实际 composition、age、update 和 violation。

VBench subject consistency 在现有方法间接近饱和，因此它不能单独证明 identity。
论文可写“提高 motion-quality trade-off”，但在盲评前不能声称 identity 显著优于
所有 baseline。

消融的判定：

| 结果 | 可支持的结论 |
|---|---|
| 主方法显著优于 `random_binary` | history-polarity membership 有效 |
| 主方法优于 `all_supportive` | 二角色异构 routing 有效 |
| 主方法优于 `no_prototype` | long-term temporal prototype 有效 |
| 主方法优于 `no_retrieval` | bounded content recall 有效 |
| random/all-head 与主方法无差别 | 缩小 head-classification claim，主讲 dual memory |
| Prototype/Recall 消融无差别 | 不能把对应组件列为独立贡献 |

不存在“必须每个指标都显著更优”的要求，但主方法至少应满足：

```text
Quality Score 不低于最佳外部 baseline
Dynamic Degree 明显高于 SF
identity/background 盲评不退化
无 polygon noise 或 cache correctness failure
```

## 9. 10 小时与论文写作并行安排

建议顺序：

| 时间段 | GPU 实验 | 同时进行的论文工作 |
|---|---|---|
| 0–0.5 h | P0 + screen16 启动 | 固定符号、方法图和 Related Work 结构 |
| 0.5–1.5 h | screen16 完成并人工 hard gate | 写二分类与双 memory 算法 |
| 1.5–7.5 h | 16 卡 full128 ablation + 16 卡 60 s | 写 Introduction、Implementation、现有 v129 主表 |
| 7.5–10 h | 四节点 VBench；优先 ablation core，再 60 s / semantic | 填 ablation/60 s 表，写 limitation |

实际生成速度若较慢，优先级固定为：

```text
paired/head audit
> full128 four-way ablation
> 60-second SF/Ours
> existing-v129 semantic extension
> optional inverted/all-suppressive
> A-B-A
```

不要等待所有实验才开始写论文。以下内容现在已经冻结，可以直接写：

1. 问题定义：training-free long autoregressive extrapolation 的
   identity/motion trade-off；
2. history-polarity score 与二分类规则；
3. Prototype 与 bounded Retrieval 的精确定义；
4. fixed budget、clean K/V、exclusive ownership 和复杂度；
5. v129 128-prompt 30 秒主结果；
6. 与 PF、Echo-Forcing、LongLive-RAG、Deep/Rolling Forcing 的区别与引用。

结果未回来前仅保留占位符：

```text
Table: paired significance
Table: binary routing / memory component ablation
Table: 60-second VBench-Long
Table: semantic and total score
Table: blind human review
```

## 10. 必须保留的学术边界

1. Self-Forcing/Pyramid-Forcing 提供 backbone/runtime 和基础 cache
   infrastructure，必须明确引用。
2. PF 的三类 head、stride/cyclic/merge 不是本文贡献，主方法也不使用这三路
   middle policy。
3. Retrieval 的一般思想已有 LongLive-RAG 等先例；本文贡献必须落在二角色
   routing、exact budget、age-bounded one-frame recall 及其组合证据上。
4. coherent snapshot/compression 思想需要引用 Echo-Forcing、Flash-VAReason
   等相关工作；不能把通用压缩或检索重新命名后声称首次提出。
5. PF labels 只允许出现在 post-hoc analysis，不得写成分类训练标签。
6. 若 random/all-head ablation 不支持 head membership，必须主动缩小 claim，
   不能选择性隐藏负结果。
