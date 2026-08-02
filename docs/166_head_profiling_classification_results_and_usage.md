# 166: Head Profiling 分类结果、证据边界与应用方案

日期：2026-08-02

## 1. 结论先行

历史 head profiling 实验产生了多组可复现的分数、排名和候选分组，
但截至 v155，**没有一套静态 head 分类已经同时通过以下四层证据**：

1. 跨 prompt/seed 的观测稳定性；
2. 在 count-matched random、bottom 和 PF-matched 对照下的 head-selective
   因果效应；
3. 干预类型特异性，即分类不只是测量“容易被扰动”的 head；
4. 30 秒实际生成中的成员特异质量改善。

因此，当前不能把任何历史分类直接写成“identity head”、“motion
head”、“history-critical head”或“recent-critical head”，也不应把它们
作为当前生成方法的硬路由。

仍然可用的资产是：

- **连续、层内相对的 propensity score**，用于实验分层、协变量和
  构造 top/bottom/random 反证对照；
- **很强的 layer effect**，它直接支持 v157 优先检验 cache placement，
  而不是再次硬分 head 语义类别；
- **K-selection 是目前最稳定的观测轴**，可以作为后续层内
  因果试验的预注册排名，但不能单独决定 cache policy；
- **QK top4 的稳定成员集**，已是一个重要的负结果和回归对照：
  它稳定，但在 v155 中不比 bottom/random 更有用；
- PF Anchor/Wave/Veil 和 legacy 304/56 可以继续作为外部标签、
  sanity reference 或消融对照，不能作为新功能分类的证据。

一句话总结：

> Profiling 保留了可复现的排名和实验分层，但没有保留可部署的
> 静态 head taxonomy。当前应用对象是 cache allocation，不是 head 命名。

## 2. 如何判断一个“分类结果”是否可用

历史文档中的 `candidate`、`screening_pass`、`causal` 和
`generation transfer` 不是同一层级的证据。本文统一使用以下四级：

| 级别 | 回答的问题 | 允许的用法 | 不允许的用法 |
|---|---|---|---|
| L1 观测稳定 | 这个分数在 prompt/seed 间是否重现 | 排名、分层、协变量 | 命名功能 head |
| L2 一步因果 | 扰动该组 head 是否更改下游 X0 | 机制候选 | 宣称视频质量更好 |
| L3 干预特异 | 效应是否对预言的 K/V/policy 干预特有 | 功能假设 | 无对照地设计硬路由 |
| L4 轨迹效用 | 持续生成是否优于 bottom/random/all-head | 方法组件 | 越过盲评/质量 gate 扩展 |

稳定不等于有用，局部因果效应也不等于生成质量改善。v152
QK top4 正是典型：成员集非常稳定，但 v155 的 trajectory-level
membership gate 失败。

## 3. 历史证据总表

| 来源 | 产生的分类/排名 | 最强证据 | 最终状态 | 当前用法 |
|---|---|---|---|---|
| PF | Anchor 172 / Wave 156 / Veil 32 | 外部已有算法标签 | 参考系 | 运行对照、post-hoc cross-tab |
| v97/v98 legacy | Supportive 304 / Suppressive 56 | 零阈值与附近阈值稳定 | 旧分组，未证明 membership | sanity/reference，不用于新 claim |
| v136 | history-supportive 49 / recent-preferred 311；long-range 46 | temporal gate PASS | 连续 temporal diagnostic | 轨迹年龄/历史倾向协变量 |
| v138 | self-history-specific 201 / no-preference 159 | specificity gate PASS | 仅跨视频匹配信号 | retrieval 诊断，不是 identity class |
| v140-v141 | prompt-sensitive threshold / A-B-A switch score | 部分连续 rank 稳定 | 两个固定分类 gate 均 FAIL | 保留连续 prompt modulation |
| v142 | best equal-budget output policy | prompt-policy modulation PASS | static policy 与 online opportunity FAIL | 仅同状态 policy diagnostic |
| v143 | multi-axis clustering | 层内剩余特征 23 个 | `no_stable_k` | 否定固定 cluster taxonomy |
| v144 | camera 48, action 42, identity 32, scene 31 | raw descriptor 稳定 | 207/360 unresolved，语义类无效 | 只用于说明 layer/state 结构 |
| v145 | 16 factor-axis + 51 state candidates | 跨 family/seed 重现 | L1 连续轴 | 分层、排名、构造因果对照 |
| v147 | full-semantic K 排名 | ranked downstream effect PASS | L2；Q retrieval/selectivity FAIL | 机制先验，不是 retrieval 路由 |
| v148 | K/V/policy top-bottom | axis effect 都 PASS，仅 K PF-independent PASS | 特异性全 FAIL | K 作为较强连续先验 |
| v149-v151 | calibrated/signed/static groups | 部分 susceptibility | leverage/specificity/group gate FAIL | 不能硬切静态类 |
| v152 | QK top4/layer | 112/120 跨 seed 重合 | 原对称 gate FAIL；单侧候选 | 已验证的负对照 |
| v153 | QK top/bottom/random 生成路由 | 7/7 结构正常 | 只证明代码可运行 | 不是质量证据 |
| v154/v155 | QK 成员实际生成 | 16 prompts、完整 core-9 | top 不优于 bottom/random | `classifier unsupported` |

## 4. 各套分类的具体结果

### 4.1 PF Anchor / Wave / Veil

当前 `best_labels.csv` 中 360 个 head 的分布是：

| PF 类别 | 标签 | Heads | 原始解释 |
|---|---:|---:|---|
| Wave | -1 | 156 | QK 时间符号/周期结构 |
| Anchor | 1 | 172 | 较稳定的正向历史相互作用 |
| Veil | 2 | 32 | 稀疏或负向历史相互作用 |

这套标签有明确的 PF 算法和 cache operator，因此可以作为已有方法
基线。但在本项目中，它们不是 identity/action/motion 语义类，也不是
新 profiling 结果的 ground truth。

允许用法：

- PF-native 生成基线；
- 检查新排名是否仅重现 PF 成员；
- 在同 PF class 内做 top/bottom 对照，排除 PF label 混杂。

不允许用法：把 PF 标签当作本项目新分类的验证标签，或将
Wave/Anchor/Veil 直接改名为 motion/identity/background heads。

### 4.2 v97/v98 legacy 304/56 分组

当前保留的 `legacy_v98_absolute_sign_304_56.csv` 使用：

```text
rho_h = median_records(
          sum(history QK logits) / sum(abs(history QK logits))
        )

rho_h >= 0 -> Supportive
rho_h <  0 -> Suppressive
```

冻结计数为 304/56。附近阈值较稳定：阈值 `-0.05/+0.05` 分别仅改变
2 个 head，`-0.1/+0.1` 分别改变 5/3 个 head。

但它与 PF 高度重合：

| PF 类别 | Supportive | Suppressive |
|---|---:|---:|
| Anchor | 169 | 3 |
| Wave | 133 | 23 |
| Veil | 2 | 30 |

零阈值 membership 与 `(Anchor + Wave) / Veil` 一致率为 `0.9222`。
这说明它主要是 history-logit polarity 参考，不是已证明的独立功能轴。

另外，该 absolute-sign score 对 common logit shift 不具有不变性。历史上它和
Prototype/Retrieval cache 组合曾得到较好的 v129 整体指标，但 v132 的
random/all-head/no-component 严格成员消融只完成生成与 audit，没有一份可用
的最终对照评分证明 304/56 membership 本身带来收益。

因此当前只将该地图用作：

- 已知可运行 cache 链路的 sanity reference；
- 与 PF 的 post-hoc cross-tab；
- 旧方法回归测试。

不再将它当作新论文的功能 head partition。

### 4.3 v136-v142：可复现的时间/状态诊断，不支持固定 prompt/policy 类

v136 在冻结 v134 profiles 上分析 prompt 与 temporal 轴：

| 分组 | 计数 | Gate |
|---|---:|---|
| prompt-conditional / invariant | 1 / 359 | prompt axis FAIL |
| history-supportive / recent-preferred | 49 / 311 | temporal axis PASS |
| long-range / local-or-mixed | 46 / 314 | temporal diagnostic |
| age-invariant | 360 | 无可用年龄类别 |

middle/recent score 的 split-half Spearman 为 `.9959`，bootstrap-reliable
fraction 为 `.9917`；但 prompt CPHI 的自然零阈值只产生 1 个正类 head。
所以 v136 支持稳定 temporal ranking，不支持 prompt-head taxonomy。

v138 比较 self history 与词汇相似/固定 offset 的 wrong-video history：

- self-history-specific: 201；
- no-self-history-preference: 159；
- split-half Spearman: `.9711`；
- bootstrap-reliable fraction: `.9472`；
- history-specificity gate: PASS；
- order-axis gate: FAIL。

这是一个真正可复现的 retrieval/specificity 信号，但 wrong-video 对照同时改变
了 identity、scene、action 和 trajectory，所以不能将 201 个 head 命名为
identity-memory heads。reverse/phase/freeze/value-mismatch 也只是 attention-level
sensitivity，没有转化为生成路由证据。

v140 对 prompt sensitivity 的 raw/query/native/key-adjusted 四个零阈值做 held-out
family 检验，四者均失败。例如 raw CPHI 的 rank Spearman 为 `.7428`，
但 validation 仅 3/348 个正类；query-adjusted 有 240/348 个正类，但
label agreement 只有 `.7443`。这说明可复现的 rank 不自动给出稳定阈值。

v141 使用真实 A-B-A prompt switch。exact-shadow parity 为 0，switch response
大于 paraphrase response，但 discovery-validation Spearman/label agreement 只有
`.5558/.6695`，full prompt-switch gate FAIL。它保留的是连续 switch
diagnostic，不是固定 switch-responsive 成员集。

v142 直接测量 equal-budget history policy 对每个 head 投影输出的近似误差。
其 policy rank 的 split Spearman/label agreement 高达 `.9661/.9167`，但：

- static policy gate: FAIL；
- online-policy opportunity: FAIL，validation static-policy regret median 为 0；
- prompt-policy modulation: PASS；
- persistent-A selectivity: FAIL。

这再次说明“最优 policy label 跨 split 一致”不等于存在有价值的
static/online routing opportunity。可保留的只是 prompt 会对 policy-error vector 产生
小而可测的调制。

### 4.4 v143：静态聚类没有通过

v143 在 layer-residual 坐标中接受了 23 个 split-stable feature，但
`k=2..6` 均未通过完整稳定性 gate：

- selected k: `None`；
- status: `no_stable_k`；
- threshold sensitivity gate: `false`。

这是一个有用的否定结果：强制把 360 个 head 分成固定 2-6 类没有
统计支持。即使 `k=2` 的 split agreement 为 `.9556`，其 bootstrap ARI
只有 `.0914`，不能因为一项稳定性较高就忽略整体 gate。

### 4.5 v144：语义标签是描述性的，不是功能类

v144 曾给出下列 dominant-factor 计数：

| 观测标签 | Heads |
|---|---:|
| camera | 48 |
| action | 42 |
| identity | 32 |
| scene | 31 |
| unresolved | 207 |

这些数字不能当作 head taxonomy，原因是：

- `57.5%` head unresolved；
- 包含 unresolved 的 dominant-label split agreement 仅 `.4556`；
- 在两个 split 都 resolved 的 64 个 head 中，只有 19 个标签相同，
  即 `.2969`；
- 85 个 split-stable feature 中 77 个是 raw perturbation response，只有
  8 个是 semantic-minus-seed-control；
- layer 本身解释了大量方差，例如 `seed_control.key_shift` 的 layer
  eta-squared 约为 `.52`。

可用结论不是“找到了四种语义 head”，而是：

```text
offline head propensity x online prompt/episode/timestep state
```

其中 K-selection 的状态稳定性最强，而语义命名不成立。

### 4.6 v145：可使用的跨 seed 连续轴

v145 使用 16 prompt families、2 seeds 和 5 variants，共 160 profiles。
它发现 16 个跨 family/seed 重现的 factor-axis candidate：

```text
paraphrase:    q_shift, k_shift, value_scale_shift, policy_shift
identity:      value_scale_shift, policy_shift
scene:         q_shift, k_shift, v_shift, value_scale_shift, policy_shift
full_semantic: q_shift, k_shift, v_shift, value_scale_shift, policy_shift
```

另有 51 个 state-specific candidate。筛选门槛同时要求 family split Spearman、
seed replicate Spearman、same-factor direction cosine 和 cross-factor margin。

对后续实验最重要的 full-semantic 轴为：

| 轴 | Family split Spearman | Seed Spearman | 解释 |
|---|---:|---:|---|
| K shift | .9908 | .9923 | 最稳定的历史 K-selection propensity |
| V shift | .9370 | .9466 | 可复现的 value response |
| Policy shift | .8824 | .8919 | 可复现的 policy susceptibility |

这些是当前最值得保留的 profiling 结果，但正确形式是每个 head 的
连续分数，而不是 top4/bottom4 硬标签。硬切只能在预注册的因果实验中
作为对照构造方法。

### 4.7 v147-v148：排名有局部因果意义，但功能特异性不足

v147 使用 v145 full-semantic K-shift 排名：

- native replay parity: PASS；
- ranked heads downstream effect: PASS；
- Q retrieval rescue: FAIL；
- Q retrieval head selectivity: FAIL。

因此可以说 K 排名与下游敏感性有关，但不能说这组 head 适合
Q-retrieval cache。

v148 进一步对 K/V/policy 做 axis-matched intervention：

| Gate | K | V | Policy |
|---|---:|---:|---:|
| axis-matched causal effect | PASS | PASS | PASS |
| PF-independent effect | PASS | FAIL | FAIL |
| intervention specificity | FAIL | FAIL | FAIL |

K 是唯一一个在同 PF label 层内对照后仍保留效应的轴。这使它比 V
和 policy 更适合做后续 prior。但三轴的 intervention specificity 全部失败，
意味着它们更像一般 susceptibility/leverage 排名，不是可命名的单一功能。

v148 dose 试验中，V 在 dose 3/4 有分离且 dose4 超过 dose1；policy
在 dose 3/4 有分离但无稳定 dose growth；K 只在 dose2 通过。这些仍是
一步诊断，不是生成路由证据。

### 4.8 v149-v151：静态分组确认失败

v149 在等投影 RMS 强度下分离 susceptibility 和 downstream leverage。
结果中 K 在部分 context 仍有 susceptibility，但所有 K/V/policy leverage
均未通过，且 core calibration integrity 失败。

v150 使用 top4/bottom4/middle4 和 8 张 count-matched random maps，检验
policy group：

- count-matched group effect: FAIL；
- intervention specificity: FAIL；
- final policy group: FAIL；
- 多个 random map 也显示正效应，说明扰动响应不是 top4 独有。

v151 的 signed scene-policy source screen 通过，其跨 family/seed 稳定性为：

```text
family split Spearman = .7457
seed Spearman         = .7898
validation high/low  = 2.3001
```

但这仅是 source L1 screen。下游的 scalar low-tail 和 signed policy 两条分支
均未确认，所有 context 又因严格 2% calibration 契约而不完整。未 gate
的诊断数据还显示 signed-high 的 downstream leverage 低于 middle。

这一证据链直接否定：“因为静态排名稳定，所以应将 top heads 分配给
特定 cache policy”。

### 4.9 v152：QK high-tail 稳定，但完整分类 gate 失败

v152 在 native state 上冻结 selector，对同一组 head 比较等预算：

```text
uniform8 = 4 个均匀分布旧帧 + 4 个最近帧
recent8  = 8 个最近帧
```

原始预注册的对称分类 gate 结果是：

```text
oracle policy choice              FAIL
QK policy choice                  FAIL
QK beats count-matched random     FAIL
QK matches oracle                PASS
online QK candidate confirmed     FAIL
old-mass baseline                 FAIL
```

其后单侧重分析发现，高 QK-margin 组的 uniform8 优势在 t1000/t750/t500
通过单侧门槛，并在 t750/t500/t250 优于 random。但低分组不稳定偏向
recent8，因此只能构造 `high-tail candidate / Default`，不能构造
`History-Critical / Recent-Critical` 二分类。

从 discovery seed 0 冻结每层 top4，在 validation seed 1 上：

- 120 个候选中重合 112 个；
- 23/30 层完全相同；
- layer-wise median Jaccard `1.0`；
- mean Jaccard `.8978`。

QK-top 与 PF 的 cross-tab 是：

| PF 类别 | QK-top | Default |
|---|---:|---:|
| Wave | 77 | 79 |
| Anchor | 24 | 148 |
| Veil | 19 | 13 |

因此这张地图不是 PF Anchor 的重命名。但“不同于 PF”也不等于“生成
中有用”。

### 4.10 v153-v155：从可运行到成员假设被否证

v153 的 7 个单视频 cell 全部结构正常，只能证明 top/bottom/random maps
可以通过相同 cache 实现而不出现 polygon noise、提前结束或 owner 冲突。
这是 correctness gate，不是 science gate。

v154/v155 才是 trajectory-level membership 检验。v155 完整 core-9 中：

```text
QK-top vs bottom:
  history consistency  +0.00122
  visual quality       -0.00563
  temporal quality     +0.00198
  dynamic degree       -0.03750

QK-top vs random:
  history consistency  +0.00023
  visual quality       -0.00464
  temporal quality     -0.00109
  dynamic degree       +0.01667
```

top 相对 bottom 的 dynamic delta 越过冻结 non-inferiority 下限 `-0.03`，
且相对 random 没有稳定独特优势。最终结论是：

> Cache useful, classifier unsupported.

因此 v152 top4 现在的正确地位是：

- 一个复现性很高、但成员效用被否证的分类实例；
- 后续新 runner 的 regression/reference method；
- 论文中“稳定性不代表功能性”的负结果。

它不再是一个等待扩展的主方法。

## 5. 现在如何应用 profiling 结果

### 5.1 应用一：用连续分数做实验分层，不做永久标签

保留每个 `(layer, head)` 的 v145 K/V/policy 分数和 v152 QK margin。
在后续分析中将它们用作：

- 连续协变量；
- 层内 rank percentile；
- 构造 top/bottom/random 的预注册对照；
- 评估不同 seed/context 的 rank recurrence。

分析时必须使用 within-layer residual/rank，不能在全部 360 个 head 上直接排序，
否则会把 layer effect 误写成 head type。

### 5.2 应用二：先找到有用的 cache placement，再问 head 差异

v155 的可用机制结果是 all-reservoir 相对 recent8 提高 dynamic，但损失
temporal/visual stability。这首先是一个资源分配问题。

v157 因此先比较：

```text
early 10 layers
middle 10 layers
late 10 layers
interleaved 10 layers
```

四组都是 120 heads，cache policy 完全一样。该实验不使用 profiling
membership，可以先回答 reservoir 收益/代价在深度上的分布。

只有 v157 出现通过 metric 与 blind gate 的 layer route，才值得在该层段
内部再检验 head score。

### 5.3 应用三：若 v157 通过，用 K score 做层内二阶反证

后续不应直接重跑 QK top4。更合理的预注册设计是：

1. 冻结 v157 胜出的 layer band，不允许看到 head 结果后换层；
2. 使用 v145 full-semantic K-shift 的层内排名，因为它的跨 seed
   重现最强，且 v148 的 PF-independent effect 是唯一通过的轴；
3. 对胜出层段构造 top4、bottom4 和至少 4 张层内 count-matched
   random4 maps；
4. 所有组使用相同 sink/reservoir/recent 预算和 owner；
5. 同时保留 all-head-in-band 与 all-recent 对照；
6. 仅当 top 同时优于 bottom 和 random ensemble，且没有损失 v157 的
   layer-level Pareto 优势时，才能说 K ranking 带来成员信息。

这个设计检验的是“已知有用层段中的连续 K propensity”，不是恢复
identity/scene 类别。

### 5.4 应用四：只在新 oracle 成立后研究 online state gate

v144/v145 都指向 state modulation，v152 也显示 QK high-tail 在 timestep 间
比较稳定。但 v152 的 policy-choice oracle 失败，所以现在不应立即部署
online QK routing。

仅当某个新 layer/cache 实验先显示真正的 policy opportunity，才可以重启
online gate。届时必须：

- selector 只在 native/frozen state 上计算；
- 两个 policy replay 使用 byte-identical scores 和 head ids；
- 分开 oracle opportunity 与 cheap proxy 两个 gate；
- 报告 shared candidate bank 的内存和 attention 成本；
- 使用 hysteresis/置信度避免 block-to-block 路由抖动。

### 5.5 应用五：profiling 结果作为论文的负结果和设计动机

当前最完整、最可辩护的论文叙述是：

1. 观测 descriptor 可以跨 prompt/seed 稳定；
2. 稳定的 head ranking 可以有局部因果效应；
3. 但这些效应不一定具有 intervention specificity，也不一定转化为
   trajectory-level membership advantage；
4. 相比固定 head taxonomy，cache mechanism 和 layer/state allocation 是更有效的
   研究对象。

这不是“profiling 失败所以没有结果”，而是一条完整的因果验证链，
它防止把 reproducibility 误当成 utility。

## 6. 当前明确禁止的应用

- 不用 v144 camera/action/identity/scene 标签决定 cache route；
- 不把 v145 top4 直接命名为 K-memory heads 或 policy heads；
- 不把 v152 bottom/default 称为 recent-preferring/Recent-Critical；
- 不把 QK-top 成员稳定性当成生成效用；
- 不用 PF overlap 证明新分类正确；
- 不仅与 SF 比较就宣称 membership，必须有 bottom/random/all-head
  的数量和 cache 匹配对照；
- 不在读取生成结果后重新选 layer、threshold 或 top-k；
- 不将 one-step X0 effect 直接表述为 identity、background 或 motion 改善。

## 7. 建议保留的资产

| 资产 | 用途 |
|---|---|
| `docs/results/v136_multi_axis_head_discovery/` | prompt axis 失败与 temporal axis 通过的起点 |
| `docs/results/v138_history_interventions/` | self-history specificity 与 order-axis 负结果 |
| `docs/results/v141_full_prompt_switch_profile/` | A-B-A 连续 prompt-switch diagnostic |
| `docs/results/v142_output_causal_profile/` | equal-budget output-policy 近似误差与 gate |
| `docs/results/v145_crossed_seed_head_profile/` | 当前最主要的连续 K/V/policy 排名来源 |
| `docs/results/v147_causal_transport_profile/` | 排名有下游效应、但 retrieval 失败的证据 |
| `docs/results/v148_axis_causal_profile/` | K PF-independent 与 intervention-specificity 边界 |
| `docs/results/v150_policy_group_confirmation/` | top/bottom/random 静态组失败证据 |
| `docs/results/v152_online_policy_profile/` | frozen selector、oracle/proxy 和 recurrence 结果 |
| `configs/head_maps/v152_qk_history_critical_manifest.json` | 稳定但已被否证的 QK membership 回归对照 |
| `configs/head_maps/legacy_v98_absolute_sign_304_56.csv` | legacy method/sanity reference |
| `docs/results/v155_profile_aligned_moviebench16/vbench_core9_summary.json` | 最终 trajectory membership 否证证据 |

## 8. 与当前实验的关系

v157 故意不使用上述任何 head classifier。它使用四个 count-matched layer
maps 检验 reservoir placement，直接响应以上证据：

- layer effect 强；
- reservoir 机制有用；
- static membership 无效；
- 需要在 motion 收益与 temporal stability 之间寻找 Pareto allocation。

因此实验顺序应保持为：

```text
v157 layer placement（已完成 core-9）
    -> interleaved10/middle10/late10 均通过自动指标 screen
    -> v157 人工盲评仍未填写，不能宣称最终 promotion
    -> v158 嵌套 6/8/10/12 层预算代码已准备
       -> 仅在 v157 blind gate 通过后允许 GPU 生成
```

不应跳过盲评直接启动 v158，也不应再对另一套静态 head map 做
128-prompt 扩展。

## 9. v136-v142 到底如何分类，阈值是否可能有问题

### 9.1 分类标准与阈值来源

| 实验 | 连续量 | 硬分类规则 | 阈值含义 |
|---|---|---|---|
| v136 prompt | semantic/history residual 相对 paraphrase residual 的 log-ratio `P_h` | `P_h > 0` | 语义扰动是否超过同义改写噪声 |
| v136 temporal | old-vs-recent centered QK margin `T_h` | `T_h >= 0` | 中远历史 logit 是否高于 recent4 |
| v136 long-range | old attention mass 相对 uniform baseline 的 excess `L_h` | `T_h>=0 and L_h>=0` | 是否同时有 old-logit 与 old-mass 证据 |
| v138 specificity | self-history effect 减去 wrong-history effect | `score > 0` | 自己历史是否优于错误视频历史 |
| v138 order | reverse/phase/freeze/value mismatch 相对 old-history effect | GMM 结构 + 重现性 gate | 是否存在可复现的 order-sensitive 子群 |
| v140 | raw/query/native/key-adjusted prompt score | primary zero；Otsu/GMM/percentile 仅诊断 | 检验 v136 失败是否只是阈值选择问题 |
| v141 | A-B-A switch/paraphrase log-ratio，再减 direct-Q log-ratio | primary zero | prompt switch 对 history use 的额外影响 |
| v142 | recent/uniform/boundary policy 的 projected-output approximation error | 每个 head 取最小误差 policy | 不是单一数值阈值，而是 argmin policy label |

这些 zero threshold 大多不是为了得到某个预期类数而设，而是有可解释的
null：语义效应不超过 paraphrase、self history 不优于 wrong history、old history
不优于 recent。它们比按中位数强制二分更合理，但仍可能受 score centering、
分母尺度和测量噪声影响，所以 zero 失败不能单独证明连续信号不存在。

### 9.2 v136 prompt 失败是否只是 zero 设错

有这种可能，但现有证据不支持把另一个阈值直接当成修复：

- v136 `cphi_score` 的 split-half Spearman 为 `.8163`，说明排名很稳定；
- 其整体分布偏负，zero 只产生 `1/360` 个正类；
- GMM2/Otsu 的诊断阈值约为 `-.351/-.478`，确实可切出更大的组；
- 但阈值有组数不等于标签可跨 prompt family 转移；
- v140 对 raw、query-adjusted、native-adjusted、key-adjusted 四种定义做了
  discovery/validation 审计，所有 zero taxonomy gate 均失败；
- raw rank 可复现但 validation 正类仍近乎退化；query-adjusted 虽产生约
  `240/348` 正类，label agreement 只有约 `.744`，低于冻结 `.80` 门槛。

因此最准确的说法是：v136 发现了稳定连续排名，但自然 zero 不适合做二分类；
现有 Otsu/GMM/percentile 也没有 held-out 证据足以替代它。阈值可以改变类数，
尚无证据表明它能产生可靠且有用的功能类。

### 9.3 prompt 扰动是否不够大

v134/v136 的 semantic-vs-paraphrase 对比可能存在信号幅度不足和直接 Q/K
变化混杂，因此这个担忧是合理的。但 v141 已经做了更强的真实 A-B-A full-prompt
switch：

- switch residual median `.00633`，高于 local paraphrase `.00370`；
- zero split 非退化，validation 有 `189/348` 个正类；
- exact-shadow parity 为 0，说明实现没有伪差；
- 但 discovery/validation Spearman 只有 `.5558`，label agreement `.6695`。

所以“扰动太弱”不是完整解释。把扰动放大后信号确实增强，但固定 membership
仍不稳定，说明 prompt-history interaction 很可能同时依赖 identity/scene、AR
episode、frame 和 denoising timestep。进一步无限放大 prompt 差异还会引入新
问题：语义、场景、动作和构图一起改变，得到的是 generic susceptibility，
而不是可命名的语义功能。

### 9.4 v138-v142 的其他可疑点

- v138 wrong-video donor 同时改变 identity、scene、action 和 trajectory；
  `201/159` specificity split 可复现，但不能区分到底对什么内容 specific。
- v138 reverse/phase/freeze 是 cached feature 干预，只测 attention output；
  order ranking 可复现，但 GMM 子群结构失败，且未直接证明 final X0 或视频运动。
- v141 每个 full prompt switch 仍是复合语义变化；它排除了“扰动太小”，没有
  排除 factor entanglement。
- v142 的 policy label 很稳定，但 validation static regret median 为 0，说明
  多个 policy 的误差面可能很平。此时 argmin 标签可以稳定，却没有足够 policy
  opportunity；改分类阈值不能创造缺失的因果效应。

## 10. 其他分类标准是否也有同类问题

有，但严重程度不同：

| 分类 | 阈值/扰动风险 | 已有反证 | 当前判断 |
|---|---|---|---|
| PF Anchor/Wave/Veil | 类由 PF 算法拓扑定义，不是本项目阈值 | 未证明对应语义或本项目最优 cache | 合法外部标签，不是 ground truth |
| legacy 304/56 | zero score 不抗 common logit shift，且与 PF 高重合 | 缺少完整 random/all-head trajectory membership 证据 | 只作 legacy reference |
| v143 cluster | `k=2..6`、feature rho `.3/.5/.7` 都可能影响结果 | bootstrap ARI/min-class/threshold sensitivity 未共同通过 | 没有稳定静态 cluster |
| v144 semantic argmax | 强制 dominant factor，且 `207/360` unresolved | resolved head 跨 split 同标签仅 `19/64` | 描述性标签，不是功能类 |
| v145 top/bottom rank | `.3` Spearman、`.05` cosine、`.02` margin 是 screening 门槛 | v147-v155 连续做了 causal/random/trajectory 验证 | 连续分数可用，硬组未确认 |
| v152 per-layer top4 | 固定 quota 会强制每层选 4 个，即使层内分布无 gap | 112/120 复现，但 v155 top 不胜 bottom/random | 稳定负对照，不是部署 map |

特别需要避免两个推理错误：

1. “换一个阈值也许会成功”只能构成新假设，不能改写已冻结的失败结果；
2. “某个 top-k 没通过”不严格否定所有可能的 `k`，但在已经做过 dose、random、
   intervention-specificity 和 trajectory transfer 后，继续 post-hoc 搜索 `k`
   的假阳性风险很高，必须使用新 held-out prompts/seeds 重新预注册。

## 11. v145 以后完整证据链、结论与剩余问题

### 11.1 已经得到的结论

| 版本 | 最强正结果 | 关键失败 | 可允许结论 |
|---|---|---|---|
| v145 | 160 profiles；16 factor-axis、51 state candidates 跨 family/seed screen | observational，未校正到生成效用 | K/V/policy 连续 propensity 可复现 |
| v147 | ranked heads 的 downstream effect PASS | Q retrieval rescue/selectivity FAIL | 排名有局部因果信息，retrieval 设计不成立 |
| v148 | K/V/policy matched G1 全 PASS；K 的 PF-independent PASS | 三轴 intervention specificity 全 FAIL | K 是最强连续轴，但可能仍是 generic history susceptibility |
| v149 | K raw susceptibility 有部分正结果 | calibration G0 FAIL，4/8 degenerate layer cells | 只能作诊断，不能确认 leverage |
| v150 | native replay 与结构审计通过 | top4 不胜 bottom/middle/8-random；specificity/strength FAIL | 静态 policy top4 被否证 |
| v151 | signed source screen PASS | 无 intact context；scalar/signed downstream gates 全 FAIL | source ranking 仍是 L1，不可部署 |
| v152 | QK high-tail 单侧 uniform8 preference、membership 跨 seed 很稳定 | 原 oracle、random、完整 candidate gate FAIL | 可形成单侧候选，不能形成 high/low 对称 taxonomy |
| v153 | 7 cells 结构正确 | 单 prompt、无统计质量结论 | runner/map 可用 |
| v154-v155 | 完整多 prompt 生成和 core-9 | top 不胜 bottom/random | cache useful, classifier unsupported |
| v157 | 3/4 layer routes 过自动 metric screen | 16 prompts、1 seed、blind 未评分 | layer allocation 有效，唯一最优未证明 |

### 11.2 v145 自身还可能有哪些问题

v145 的门槛本来就是 discovery screen，较宽松：family/seed Spearman 至少 `.3`、
direction cosine 至少 `.05`、cross-factor margin 至少 `.02`。仍需注意：

- 同时筛选大量 factor/axis/state 组合，存在 multiple-comparison 风险；
- Q/K/V/policy 特征高度相关，16 个 candidate 不等于 16 个独立机制；
- layer effect 很强，必须坚持 within-layer residual/rank；
- 16 个 prompt families 仍来自同一套构造模板，外部 prompt-domain transfer 未证；
- 每个 factor 的文本变化可能同时改变多个视觉属性；
- 后续多为 frame117 的 one-step replay，不能替代完整 trajectory；
- v149-v151 calibration integrity 不完整，使 leverage 负结果中混有数值无效 cell；
- 但 v150 的 count-matched random 失败和 v155 trajectory 失败不依赖“换 v145
  screening threshold”即可消除，因此不能把全部负结果归因于 calibration。

### 11.3 仍值得做的已有数据分析

在新增 GPU profiling 前，优先做一次 threshold-free audit：

1. 对 v136/v140/v141 的连续 score 计算 family/seed block bootstrap、permutation
   null 和跨 context rank curve，不输出主二分类；
2. 对 zero/Otsu/GMM/percentile 画完整 threshold-stability surface，报告 class
   size、Jaccard、held-out AUC/continuous correlation，不挑最佳点；
3. 对 v145 的全部 candidate 做 Benjamini-Hochberg FDR 和 effective-rank/相关簇
   分析，避免把相关轴重复计数；
4. 用 cross-fit 回归检验 v145 continuous score 能否预测 v147 的 per-head local
   response；v147/v148 的 final-X0 只有 group intervention，必须在 group/dose
   层面分析，不能伪造 per-head `R^2`；
5. 分解 layer、head-within-layer、prompt family、seed、timestep 的方差，回答主要
   不稳定性到底来自哪里。

这一步主要消耗 CPU，能直接回答“阈值错了还是对象本身连续/状态依赖”。

其中第一版 v145 threshold-free/FDR audit 已实现并运行：

```text
packaged counterfactual factor-axis tests = 20
within-layer permutation + BH-FDR pass    = 20/20
feature-correlation effective rank        = 4.431
```

这补强了两个结论：层内连续排名确实非常稳定，不是由少数偶然 threshold
产生；但 20 个显著 feature 只有约 4.4 的相关矩阵有效秩，说明它们高度冗余，
不能解释为 20 种独立 head 功能。该审计先做 within-layer median residual，
所以其 zero split 是“相对本层中位数”，不是 v136 原始 semantic null；它适合
检验排名/阈值稳定性，不得用来恢复功能 taxonomy。

这里的 20/20 也不推翻 v145 原报告只有 16 个 factor-axis candidates：新审计
只要求 rank reproduction + FDR，而原 screen 还要求 delta-direction cosine 与
cross-factor specificity margin。少掉的 4 个主要是“排名稳定但方向/因素特异性
不足”，恰好说明 reproducibility 和 mechanism specificity 必须分开。

产物位于：

```text
docs/results/head_profile_threshold_free_audit/
  audit_report.json
  audit_summary.md
  feature_fdr_audit.csv
  feature_correlations.csv
  threshold_stability.csv
```

## 12. 是否还需要新的 profiling 实验

需要，但只建议一个针对现有证据缺口的实验，不再做无约束 taxonomy 搜索。

### 12.1 推荐：layer-conditioned matched-factor causal profile

启动条件：先完成 v157 blind；若人评否决 reservoir placement，则不为该机制继续
做 head profiling。若通过，在 v157 已通过的固定 layer route 内进行：

- primary analysis 使用连续 K propensity，不以 top-k label 为主要结论；
- discovery/validation 使用不同 prompt families 和不同 seeds；
- donor 分别匹配 `identity-only`、`scene-only`、`action-only`、
  `temporal-order-only`，避免 v138 wrong-video 的复合混杂；
- 每种干预至少有 null、弱、中、强四档 dose，检验 monotonic slope，而不是只看
  一个阈值两侧；
- 在相同 projected perturbation norm 下记录 local output 与 final X0，分开
  susceptibility 和 leverage；
- 预先固定 layer route、score、dose、prompts、seeds 和 multiplicity correction；
- 只有连续 score 在 held-out 上预测 causal slope，且优于 layer-matched random，
  才进入 trajectory routing。

### 12.2 不建议继续的 profiling

- 再对 360 heads 做 k-means/GMM 并命名 identity/motion/scene；
- 在现有结果上寻找一个能让 top-k 通过的 threshold 或 `k`；
- 只增加 prompt 文本差异而不做 factor-matched control；
- 只测 attention output，不测 final X0 或 trajectory；
- 在 v157 blind 未完成时并行扩大 reservoir 生成规模。

### 12.3 当前优先级

```text
P0  填写并分析 v157 blind review
P1  对 v136-v145 做 threshold-free/FDR/cross-fit CPU 重分析
P2  blind 通过后运行 v158 6/8/10/12 nested budget sweep
P3  根据 v158 结果决定是否做 layer-conditioned matched-factor profiling
P4  只有 P3 因果与 held-out gate 通过，才重启 head-selective trajectory routing
```

我们仍有时间做 profiling，但应把时间用于消除可识别的混杂和验证连续预测，
而不是增加另一套未经因果确认的静态分类。
