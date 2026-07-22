# AAAI Provisional Title and Abstract

> 用途：在方法尚未完全圈定时完成 AAAI paper registration。
> 原则：固定研究问题与论文主线，不锁死 HREM-v2、episode/head 公式或具体实现名称；不写尚未获得的实验结果。

## 1. 推荐注册标题

```text
When to Remember: Selective Memory Recall for Training-Free Long Video Generation
```

中文参考：

```text
何时回忆：面向免训练长视频生成的选择性记忆召回
```

这个标题适合作为当前占位标题，原因是：

- `Selective Memory Recall` 能覆盖当前 episode selection、head routing、cache selection 和后续可能的 token/frame-level 变体；
- 没有写死 `HREM`、`head-role`、`episodic`、`dual-evidence` 或具体 backbone；
- 即使最终从两级 gate 调整为统一 uncertainty/relevance controller，标题仍然成立；
- `When to Remember` 保留清晰问题意识，后续可以自然升级为 `When and Where to Remember`。

## 2. 可直接用于注册的英文摘要

> Autoregressive video diffusion models can generate high-quality clips efficiently, yet extending them to long horizons remains difficult. Their bounded temporal caches gradually discard early visual evidence, while directly retaining or retrieving more history can introduce stale content, suppress motion, and increase memory cost. We study selective memory recall for training-free long video generation. Our framework augments a pretrained autoregressive generator with a bounded side memory and separates two decisions that are often conflated: which historical content is relevant to the current generation state, and where that content should influence attention. Historical representations are compactly maintained, retrieved using evidence from the current context, and injected through a gated auxiliary attention path that preserves the native short-term cache. When recall evidence is weak or conflicting, the model abstains and falls back to its original computation. This design requires neither additional training nor changes to model parameters, and supports multiple memory organization, retrieval, and routing strategies within a common formulation. We evaluate the framework on long-horizon scene return and general long-video generation, measuring identity and background consistency, temporal dynamics, visual quality, computational overhead, and failure rates. Our study aims to clarify when external memory benefits autoregressive video generation and when selective abstention is necessary.

当前版本为 201 词。它是一份完整、无占位符、无虚构结果的 registration abstract。

## 3. 中文对照

自回归视频扩散模型能够高效生成高质量短视频，但将其扩展到长时间范围仍然困难。有限的时序缓存会逐渐丢弃早期视觉证据，而直接保留或检索更多历史又可能引入过期内容、抑制运动并增加存储开销。我们研究面向免训练长视频生成的选择性记忆召回。该框架在预训练自回归生成器之外增加一个有界侧记忆，并分离两个通常被混合处理的决策：哪些历史内容与当前生成状态相关，以及这些内容应在何处影响 attention。历史表征被紧凑维护，依据当前上下文证据进行检索，并通过保留原生短期缓存的门控辅助 attention 路径注入。当召回证据较弱或相互冲突时，模型放弃召回并回退到原始计算。该设计不需要额外训练或修改模型参数，并可在统一框架中容纳多种记忆组织、检索与路由策略。我们将在长时场景回访和通用长视频生成任务上进行评估，同时测量身份与背景一致性、时序动态、视觉质量、计算开销和失败率。该研究旨在厘清外部记忆何时能改善自回归视频生成，以及何时必须采用选择性拒绝机制。

## 4. 摘要的可修改结构

当前摘要可拆成四个稳定块和两个待更新块：

| Block | 当前内容 | 后续处理 |
|---|---|---|
| Problem | bounded cache 遗忘早期信息 | 建议保留 |
| Tension | 更多历史会产生 stale content、motion suppression 和成本 | 建议保留 |
| Core idea | selective memory recall | 建议保留，作为论文主线 |
| Method | bounded side memory + selection/injection separation + abstention | 根据最终方法细化 |
| Evaluation | scene return + general long video metrics | 根据实际 benchmark 更新 |
| Results | 当前不写结果 | 实验后加入数值和统计结论 |

如果最终保留 HREM-v2，可将方法部分替换为：

```text
The method first admits a non-recent episode using semantic and visual-query
evidence, and then estimates continuous per-head recall gates from the temporal
persistence of archived keys and values and the drift of current queries.
```

如果最终不保留 head-role，而改为统一 confidence controller，可替换为：

```text
The method estimates the relevance and reliability of historical memory under
the current generation state and uses the resulting uncertainty to control
both retrieval and injection.
```

## 5. 实验完成后的结果句模板

不要把下列占位句提交为当前摘要。完成多 prompt、多 seed 实验后，用真实数字替换方括号，再放到摘要倒数第二句。

```text
Across [BENCHMARKS] and [NUMBER] generation settings, the proposed method
improves [PRIMARY LONG-TERM CONSISTENCY METRIC] by [VALUE] over [BASELINE],
while preserving [MOTION/QUALITY METRIC] and adding only [MEMORY/LATENCY COST].
Ablations show that [MEMORY SELECTION COMPONENT] and [INJECTION/ROUTING
COMPONENT] provide complementary gains, and that abstention prevents
degradation on [NON-RETURN OR AMBIGUOUS CASES].
```

如果结果尚不支持全面提升，应使用更保守的版本：

```text
Experiments show that selective recall improves long-range scene recovery in
settings where relevant historical evidence is available, while abstention is
important for limiting degradation under ambiguous or non-returning contexts.
```

## 6. 方法收敛后的标题选项

### 6.1 当前 HREM-v2 最终成立

```text
When and Where to Remember: Factorized Episodic Recall for Training-Free Long Video Generation
```

强调 episode admission 与 head admission 两级结构。

### 6.2 最终以 evidence/uncertainty 为核心

```text
Evidence-Guided Selective Memory for Training-Free Long Video Generation
```

适合 selector 与 routing 最终被统一为 evidence controller。

### 6.3 最终以 cache/memory 系统贡献为主

```text
Adaptive Memory Routing for Training-Free Long-Horizon Video Generation
```

适合方法从 episodic return 扩展到一般长时一致性。

### 6.4 最终结果只支持场景回访

```text
Selective Episodic Recall for Training-Free Long Video Generation
```

范围最清晰，也最不容易过度声称。

## 7. 当前不建议写入标题或摘要的内容

- 不在标题中使用 `Head-Role`，除非 head ablation 明确成立；
- 不使用 `Dual-Evidence`，除非 A-B-C-A/A-B-C-B 多候选选择通过；
- 不使用 `Long-Term Consistency` 作为唯一主张，除非通用 32-prompt/VBench 成立；
- 不写 `state-of-the-art`、`significant`、`robust`、`first`；
- 不写具体提升百分比，直到多 seed paired statistics 完成；
- 不把显式 `||` prompt boundary 描述为 automatic episode discovery。

## 8. 注册后建议保留的修改记录

每次修改标题或摘要时记录：

```text
date:
method commit:
title:
abstract word count:
new claim added:
supporting experiment:
claim removed or weakened:
reason:
```

当前注册版本建议标记为 `AAAI-registration-v0`。待 Stage-1、multi-candidate 和多 seed 结果完成后，再分别形成 `v1-method-specific` 和 `v2-result-complete`。
