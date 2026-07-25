# Training-Free Long Video Related Work, Code Provenance, and Claim-Safety Ledger

## 2026-07-25 v95 provenance update

The current candidate is defined in
`docs/95_post_v93_dual_axis_phase_cache.md`. The new
`pyramidkv/prompt_warmup.py` implementation was written in this repository; no
source code was copied from the works below.

| v95 mechanism | Local implementation | Nearest prior ideas | Attribution boundary |
|---|---|---|---|
| PF Anchor/Wave/Veil steady-state reads | vendored PF plus existing local integration | [Pyramid Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | borrowed base, never our contribution |
| paired prompt-intervention head response | local profiler and `build_prompt_contrastive_head_maps.py` | head specialization in PF, Forcing-KV, Head Forcing | project-designed measurement; do not claim discovery of head heterogeneity |
| prompt-role startup history shield | `pyramidkv/prompt_warmup.py` | head-specific caches, attention sink, local/anchor/memory lifecycles | project-designed control; novelty claim remains conditional on full review |
| deterministic staggered history release | `pyramidkv/prompt_warmup.py` | asynchronous/staggered cache updates generally | project-designed implementation, not proof of first use |
| noisy/clean trust-qualified middle promotion | `pyramidkv/transition.py` | novelty/episodic updates in Head Forcing and cache lifecycle work | claim only the exact trajectory-trust admission if ablations pass |
| weak prompt-role priority after trust gates | `pyramidkv/transition.py` plus lifecycle-role CSV | semantic memory in SWIFT/MemFlow and role-dependent cache control | project-designed combination; must beat random and inverse controls |

Additional reviewed repositories and papers:

- [Forcing-KV](https://arxiv.org/abs/2605.09681): static/dynamic heads and
  hybrid compression. The local `third_party/Forcing-KV` remains an empty
  placeholder.
- [Head Forcing](https://arxiv.org/abs/2605.14487): local/anchor/memory heads,
  hierarchical episodic memory, dynamic updates, and head-wise RoPE.
- [LongLive](https://github.com/NVlabs/LongLive): attention sink, KV recache,
  streaming long tuning, and later long-video infrastructure.
- [MemFlow](https://github.com/KlingAIResearch/MemFlow): trained
  prompt-conditioned historical-frame and token retrieval.
- [DummyForcing](https://github.com/csguoh/DummyForcing): head/context
  allocation through dummy-head analysis.

Prohibited claim for v95: "we discover head specialization," "we introduce
head-aware memory," "we are first to update memory by novelty," or "PF's
three-class cache is our method." Any paper claim must name Self-Forcing and
Pyramid Forcing as the generator/base cache and report v95 as the added
phase/lifecycle controller.

> 更新日期：2026-07-23
>
> 目的：记录本项目使用、移植、参考和待评估的论文与代码，约束论文叙事和代码复用，降低错误归属、遗漏引用、代码许可证违规和学术不端风险。
>
> 范围：当前 `third_party/`、Commit Forcing 主路径以及历史
> LifeCache/CEMR/HREM 原型。
>
> 注意：本文是研究与工程台账，不替代学校、会议或律师对许可证和科研伦理的正式意见。

## 0. 不可违反的原则

我们可以调整论文故事，使其聚焦真正经过实验支持的贡献，但不能通过改名、重新包装或省略引用，把已有机制描述为自己的原创。

必须遵守：

1. 论文中使用过的机制，无论是否直接复制代码，都要引用最接近的已有工作；
2. 直接移植或改写的代码必须记录原仓库、原文件、许可证和本地修改位置；
3. 没有明确许可证的仓库只可用于阅读和比较，默认禁止复制代码；
4. 通用组件的重新实现不自动构成方法创新，例如 archive、top-k retrieval、EMA、RoPE 重映射、独立 memory attention 和 sigmoid gate；
5. “组合从未出现过”必须经过系统文献检索和近邻方法对比，不能仅凭当前 `third_party/` 判断；
6. 不能用单 prompt、单 seed、挑选样例或不公平 baseline 支撑论文主张；
7. 不能使用 `first`、`首次`、`首个`、`novel head-aware memory` 等强表述，除非投稿前完成检索并保留检索记录；
8. 若近邻工作已经覆盖某个贡献点，应主动收缩 claim，而不是只更换术语。

本文使用五种来源标签：

| 标签 | 含义 | 论文和代码要求 |
|---|---|---|
| **PORTED** | 本地代码从第三方实现移植或明显改写 | 文件内注释、论文引用、许可证/NOTICE 均需保留 |
| **PROJECT-INTERNAL PORT** | 从本仓库早期原型迁移到当前主路径 | 记录原型文件与 commit；不要误归因给承载原型的第三方论文 |
| **INSPIRED** | 使用了论文思想，但当前实现不是逐行移植 | 论文中引用，并明确实现差异 |
| **PROJECT-DESIGNED** | 当前组合、公式或控制结构由本项目设计 | 仍需引用组成它的已有思想；不能自动声称世界首次 |
| **CANDIDATE** | 尚未进入当前 HREM-v2 主路径 | 只有真正实现并消融后才能写入 method contribution |

## 1. 当前 HREM-v2 的机制来源图

| HREM-v2 组件 | 当前代码 | 来源标签 | 最接近的相关工作 | 当前可接受的归属 |
|---|---|---|---|---|
| AR backbone、rolling K/V、clean/noisy forward | `third_party/Self-Forcing/` | **PORTED / BASE** | Self-Forcing | 明确写成基于 Self-Forcing 的推理时扩展 |
| 多段 prompt schedule、boundary conditional switch | `third_party/Self-Forcing/pipeline/causal_inference.py` | **PROJECT-INTERNAL PORT** | 本项目先前集成在 vendored PF tree 的 A-B-A 原型 | 不归因给 PF 论文；PF 仍作为承载原型的第三方基座引用 |
| episode id 与 prompt descriptor 向 archive 传播 | 同上 `_set_memory_episode` | **PROJECT-INTERNAL PORT + MODIFIED** | 本项目先前 PF-tree structured-memory 原型 | 记录内部迁移历史；masked descriptor 与 sidecar 为后续修改 |
| clean pre-RoPE K/V sidecar archive | `src/lifecycle_kv/episodic_archive.py` | **PROJECT-DESIGNED COMPOSITION** | MemRoPE、Echo-Forcing、LongLive-RAG、Deep/Rolling Forcing | 不把 archive 或 pre-RoPE 单独声称为创新 |
| episode-balanced coverage maintenance | `EpisodicArchive._budget_indices` | **PROJECT-DESIGNED IMPLEMENTATION** | scene memory、sink/coverage、retrieval memory | 可写成本项目实现细节，不能写成首个 bounded archive |
| semantic + visual-QK non-recent episode admission | `src/lifecycle_kv/role_episodic.py` | **PROJECT-DESIGNED COMBINATION** | Echo-Forcing、LongLive-RAG、IAMFlow、SWIFT | 当前最可能成为贡献之一，但必须与四个近邻逐项比较 |
| top-k frame retrieval | `src/lifecycle_kv/attention_fusion.py` | **COMMON / INSPIRED** | LongLive-RAG、Echo-Forcing 及一般 RAG | 只作为实现组件，不作为创新点 |
| per-head K/V persistence + query-stability gate | `src/lifecycle_kv/role_episodic.py` | **PROJECT-DESIGNED FORMULA** | Pyramid-Forcing、Forcing-KV、MotionCache | 可主张在线证据公式与验证框架；必须引用 head specialization 先例 |
| absolute/relative/hybrid role calibration | 同上 | **PROJECT-DESIGNED ITERATION** | 排名门控、online routing 的一般思想 | 只有实验显示稳定选择性和质量收益后才能成为贡献 |
| independent memory attention | `src/lifecycle_kv/attention_fusion.py` 与 SF bridge | **PROJECT-DESIGNED IMPLEMENTATION / COMMON PATTERN** | 通用 multi-branch/gated attention；未发现从当前近邻仓库直接移植 | 可写成本项目实现选择，不把并行 attention 概念本身声称为原创 |
| confidence/head/alignment bounded convex fusion | 同上 | **PROJECT-DESIGNED COMBINATION** | gated residual/convex fusion 的一般方法 | 可作为安全实现，不宜作为核心理论创新 |
| fail-closed native fallback 与因果 trace | HREM bridge + analyzer | **PROJECT-DESIGNED CONTROL PROTOCOL** | selective prediction、abstention 的一般思想 | 可主张可审计工程和因果实验设计，不声称 abstention 概念原创 |
| sparse token-wise 3D RoPE helper | `third_party/Self-Forcing/wan/modules/causal_model.py` | **INSPIRED / LEGACY** | MemRoPE | 文件已注明来源；当前 P0 `position_mode=none` 并未使用该 helper |
| scene pool 历史原型 | `src/lifecycle_kv/episodic_pool.py` | **INSPIRED / LEGACY** | Echo-Forcing | 文件已注明 inspired by Echo；不是当前 HREM-v2 主贡献 |
| 固定 PF head labels 历史原型 | `src/lifecycle_kv/head_roles.py` 等 | **PORTED DATA / LEGACY** | Pyramid-Forcing、Forcing-KV | 当前 P0 不使用固定 labels；旧消融必须引用来源 |

### 1.1 Vendored tree 不是纯净 upstream snapshot

`third_party/` 下的代码不能一律理解为未经修改的上游源码。Git 历史显示：

- `5175099` 将 Pyramid-Forcing 源码 vendored 到本仓库；
- 之后本项目在该树上加入 structured visual memory、full-frame history routing、confidence routing、coverage archive、local-RoPE readout、content-addressed retrieval、A-B-A benchmark 和 dual-cue retrieval；
- 当前 Self-Forcing HREM bridge 的一部分是从上述“本项目修改后的 PF tree”迁移，而不是从 canonical Pyramid-Forcing upstream 迁移；
- canonical PF 论文贡献仍是 offline head classification、per-head pyramidal cache policy 和 ragged-cache attention。

关键内部演化提交如下。这里的提交只能证明本项目的实现顺序，不能替代公开文献的新颖性检索：

| Commit | 在 vendored PF tree 中加入的项目内部实现 |
|---|---|
| `3bccd41` | query-conditioned structured memory |
| `f4e35a8` | structured-memory clean pass 与实现收敛 |
| `088ec1c` | full-frame history 与稳定 head routing |
| `5ebbb2a` | dual-cue retrieval |
| `0766772` | coverage-aware archive 与 abstention |
| `972a94b` | local-RoPE memory readout |
| `874d80a` | content-addressed retrieval |
| `ea19b1c` | A-B-A benchmark 与场景回返协议 |

随后，`ab13365` 将 HREM-v2 主路径迁入 Self-Forcing，`f19a6bd` 增加诊断输出，`768c704` 增加在线 role calibration。以上哈希应在重写历史或迁移仓库前保留可追溯记录。

因此必须同时记录两层 provenance：

```text
canonical PF upstream -> vendored PF base (Apache-2.0 + NOTICE)
training-free project commits -> structured-memory prototype in vendored PF
project-internal port -> current Self-Forcing HREM bridge
```

Git 历史可以证明本项目内部实现的演化顺序，但不能单独证明学术新颖性；新颖性仍取决于公开文献和同期工作检索。

### 1.2 已确认的直接代码来源

当前人工审计在主仓库中发现以下显式来源标记：

1. `third_party/Self-Forcing/pipeline/causal_inference.py` 的 `_set_memory_episode` 从本项目先前的 vendored-PF structured-memory prototype 迁移；
2. 同文件的 `A || B || A` prompt 拆分和场景切换控制流从本项目先前的 vendored-PF A-B-A prototype 迁移；
3. `third_party/Self-Forcing/wan/modules/causal_model.py` 中历史 LifeCache sparse 3D RoPE helper 标记为 based on MemRoPE；
4. `src/lifecycle_kv/episodic_pool.py` 标记为 inspired by Echo-Forcing scene pool；
5. `src/lifecycle_kv/head_roles.py` 和旧 LifeCache manager 可读取 Pyramid-Forcing CSV 或 Forcing-KV 风格标签。

前两项可作为本项目内部实现演化的一部分，但不能误写成 PF 论文贡献；第 3-5 项涉及明确外部思想、代码或数据来源，不能写成 HREM-v2 原创。所有派生代码仍应保留底层第三方许可证、NOTICE 和学术引用。

### 1.3 当前未发现逐行移植、但有明确思想影响的部分

- `episodic_archive.py` 没有发现直接复制第三方 scene-pool 类的标记，但其设计受到 scene memory、bounded cache、complete-frame memory 和 pre-RoPE storage 的共同影响；
- `select_dual_evidence_episode` 的当前公式由本项目组合，但“语义选择历史”“视觉检索历史”“排除近期窗口”都有强先例；
- `compute_head_role_evidence` 的 K/V persistence、query stability 和 calibration 公式由本项目实现，但“head 功能分化”和“motion/static 分工”已有 Pyramid-Forcing 与 Forcing-KV；
- `fuse_parallel_attention` 的多级门控和 exact native fallback 是本项目的安全组合，但独立 attention、confidence gate 和 residual/convex fusion 都不是新概念。

因此建议统一写法为：

> We build on training-free cache, scene-memory, retrieval, and head-specialization findings, and propose a factorized admission mechanism that jointly audits historical episode eligibility and online head eligibility before a bounded side-branch readout.

不要写：

> We introduce the first episodic KV archive / the first head-aware memory attention / a novel pre-RoPE memory.

### 1.4 2026-07-23 research reset：Commit Forcing

`docs/73_lifecache_v3_screen_results.md` 已经证明：LifeCache-v3 的
side-memory fusion 虽然机械生效，但没有可见收益；online head routing 也没有优于
all-head。上面的 HREM-v2 来源图因此保留为历史台账，不再代表推荐投稿方法。

当前新增路径的来源如下：

| Commit Forcing 组件 | 当前代码 | 来源标签 | 最接近工作 | 允许的归属 |
|---|---|---|---|---|
| SF backbone、rolling cache、四步 denoising | `third_party/Self-Forcing/` | **PORTED / BASE** | Self-Forcing | 明确写为 SF 推理扩展 |
| reference denoise -> re-noise -> native-context denoise | `pipeline/causal_inference.py` | **INSPIRED / INDEPENDENT REIMPLEMENTATION** | Pathwise TTC | 必须引用 TTC；fixed-origin 只能作为 prior baseline |
| denoising prediction disagreement reliability | `src/lifecycle_kv/commit_forcing.py` | **PROJECT-DESIGNED FORMULA** | diffusion uncertainty/consistency 的一般思想 | 可作为候选公式，需证明能预测后续失败 |
| origin/trusted state admission 与更新 | 同上 | **PROJECT-DESIGNED COMBINATION** | TTC fixed anchor、Echo snapshots、一般 bounded memory | 可主张 reliability-gated state commit 组合，不能把 anchor/FIFO 单独写成创新 |
| clean pre-RoPE full-frame K/V snapshot | 同上与 `causal_model.py` | **PROJECT-DESIGNED IMPLEMENTATION / COMMON** | MemRoPE、cache memory work | 只作为实现选择 |
| adjacent reference RoPE | `build_reference_cache` | **PROJECT-DESIGNED IMPLEMENTATION / COMMON** | PF、MemRoPE、Echo 的 position-safe 思想 | 必须引用 position-safe memory 先例，不把 re-RoPE 本身作为贡献 |
| strict trace 与 native fail-off | analyzer 与 env gating | **PROJECT-DESIGNED CONTROL PROTOCOL** | 一般可审计实验 | 可作为 reproducibility 贡献，不是生成算法创新 |

`src/lifecycle_kv/commit_forcing.py` 没有从 Pathwise TTC 仓库复制源码。
截至审计时，其公开仓库只有 `README.md` 和 assets；本地实现根据论文公开流程独立完成。
如果后续 upstream 发布源码，必须重新做逐文件碰撞和许可证审计。

## 2. 与当前论文故事碰撞最高的工作

### 2.1 Self-Forcing

- 论文：[Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009)
- 原始代码：[guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- 本地代码：`third_party/Self-Forcing/`
- 许可证：本地顶层 Apache-2.0。
- 我们使用：backbone、checkpoint、rolling K/V、causal attention、clean/noisy forward 和推理入口。
- 论文定位：必须作为 base model 和主 baseline；HREM-v2 是 inference-time extension，不是新的 AR generator training 方法。

### 2.2 Pyramid-Forcing

- 论文：[Head-Aware Pyramid KV Cache Policy for High-Quality Long Video Generation](https://arxiv.org/abs/2605.13111)
- 原始代码：[if-lab-pku/Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing)
- 本地代码：`third_party/Pyramid-Forcing/`
- 许可证：Apache-2.0，且带 `NOTICE`。
- 我们直接使用 PF 上游的 head-specialization/cache-policy 基座与 `best_labels.csv` 历史消融；head specialization、per-head 异构 cache 和 ragged-cache attention 是重要设计参照。多 prompt schedule 和 episode sidecar 是本项目在 vendored PF tree 中加入的原型，不是 PF 上游贡献。
- 关键差异：PF 离线分类 head 并分配 `[sink + middle + recent]` cache 策略，再通过 ragged-cache attention 读取异构 cache；HREM-v2 尝试在线估计当前 readout 的 head eligibility，并在显式 episode admission 后读取独立历史 archive。HREM 的 side readout 不应归因给 PF。
- 风险：如果 online gate 没有稳定选择性或没有优于固定 PF labels，则不能把 head-aware 部分作为贡献。
- 必做实验：`PF fixed labels`、`all heads`、`online role gate` 三者同预算比较。

### 2.3 Echo-Forcing

- 论文：[Echo-Forcing: A Scene Memory Framework for Interactive Long Video Generation](https://arxiv.org/abs/2605.16003)
- 原始代码：[mingqiangWu/Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing)
- 本地代码：`third_party/Echo-Forcing/`
- 许可证：MIT。
- 我们参考：preserve/recall/forget 场景语义、scene pool、绝对 frame/spatial sidecar 和 difference-aware decay；旧 `episodic_pool.py` 明确受其启发。
- 关键差异候选：HREM-v2 不依赖手写 recall id，先做可拒绝的 non-recent episode admission，再做 per-head admission；current P0 不使用 Echo 的 decay 或 scene-pool 代码。
- 风险：Echo 已经明确覆盖 scene recall。论文不能把“保存旧场景并在返回时召回”作为新问题或核心创新。
- 必做实验：相同 multi-prompt protocol 下与 Echo-Forcing 比较，并报告自动 route、motion 和边界伪影。

### 2.4 LongLive-RAG

- 论文：[LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation](https://arxiv.org/abs/2606.02553)
- 原始代码：[qixinhu11/LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG)
- 本地代码：`third_party/LongLive-RAG/`
- 许可证：Apache-2.0。
- 我们参考：把长视频历史访问表述为 retrieval、latent descriptor、recent-window exclusion、CPU history 和临时 recall context。
- 关键差异候选：HREM-v2 的单位是受 episode sidecar 约束的 K/V frame，并在 retrieval 前做 episode admission、在 retrieval 后做 head admission；当前没有使用 LongLive-RAG 的 retrieval AE 或 CPU memory 实现。
- 风险：top-k historical retrieval、temporary history injection 和 recent exclusion 不能作为我们的创新。
- 必做实验：同 archive budget 下与 LongLive-RAG 或其 retrieval policy 比较；若无法完整运行，至少做 policy-level ablation。

### 2.5 SWIFT

- 论文：[SWIFT: Prompt-Adaptive Memory for Efficient Interactive Long Video Generation](https://arxiv.org/abs/2605.09442)
- 原始代码：[ShanwenTan/SWIFT](https://github.com/ShanwenTan/SWIFT)
- 本地代码：`third_party/SWIFT/`
- 许可证：本地顶层 Apache-2.0。
- 与我们重合：多 prompt interactive generation、prompt-adaptive memory、semantic injection、head-wise injection 和 segment-level anchors。
- 当前差异候选：HREM-v2 关注返回到非近期 episode 的显式历史选择，并用当前 Q 与历史 K/V 动态证据做两级 admission；SWIFT 更强调 prompt switch 的高效语义注入与动态窗口。
- 风险：这是当前最容易被遗漏、也最可能削弱“semantic + head-aware”新颖性的近邻之一。投稿前必须精读论文公式和实现，不能只根据 README 写差异。
- 必做动作：建立逐项 comparison table，并加入 SWIFT baseline 或最接近的 semantic-injection ablation。

### 2.6 Forcing-KV

- 论文：[Forcing-KV: Hybrid KV Cache Compression for Efficient Autoregressive Video Diffusion Models](https://arxiv.org/abs/2605.09681)
- 原始代码：[zju-jiyicheng/Forcing-KV](https://github.com/zju-jiyicheng/Forcing-KV)
- 本地状态：`third_party/Forcing-KV/` 当前为空目录，不能声称已完成本地源码审查或复现。
- 许可证：本地无文件，且空目录；在确认 upstream license 前禁止复制代码。
- 我们参考：static/dynamic head 的功能差异和 head-specific cache policy。
- 关键差异候选：Forcing-KV 主要做 head-specific KV compression/acceleration；HREM-v2 尝试做历史 episode readout eligibility。
- 风险：若论文将 K/V persistence 直接解释成 static/dynamic head 分类，必须承认 Forcing-KV 的先例，并证明我们的在线证据、任务和干预对象不同。

### 2.6b Head Forcing

- 论文：[Head Forcing: Long Autoregressive Video Generation via Head Heterogeneity](https://arxiv.org/abs/2605.14487)
- 项目页：[Head Forcing](https://jiahaotian-sjtu.github.io/headforcing.github.io/)
- 本地状态：未 vendored；截至 2026-07-24 项目页仍标记 code coming soon，因此没有完成源码审查，也没有复制其实现。
- 已公开机制：local / anchor / memory head 分类、针对不同 head 的 KV 策略、memory head 的 hierarchical episodic memory、dynamic episodic update 和 head-wise RoPE re-encoding。
- 与 v86 的高碰撞点：head heterogeneity、不同生命周期、novelty / dynamic update 都已有明确先例。
- 当前必要区别：v86 不增加 episodic read path，不把 head 分类本身作为贡献；它用已有 noisy/clean pass 的一致性和 last-admitted shock 控制 PF middle state promotion，role 只调整写入时钟。
- 风险：如果实验不能证明 noisy-clean trust admission 或 role-conditioned promotion 独立于 PF/Head Forcing 带来收益，就应只保留 v78，并放弃 head-role 主贡献。

### 2.7 Pathwise Test-Time Correction

- 论文：[Pathwise Test-Time Correction for Autoregressive Long Video Generation](https://arxiv.org/abs/2602.05871)
- 原始代码页：[xbxsxp9/Pathwise_TTC](https://github.com/xbxsxp9/Pathwise_TTC)
- 本地状态：未 vendored；截至 2026-07-23，公开仓库只有 README 和 assets，未发现可复制实现。
- 我们直接参考：低噪声阶段 reference-conditioned prediction、恢复到同一噪声等级、再用原 AR context denoise。
- 必须作为 baseline：固定 initial/origin reference 的 correction。
- 候选差异：我们用现有多步 denoising prediction disagreement 做 state admission，并维护 bounded origin + online trusted state bank；reference 不再固定为 initial frame。
- 决定性风险：如果 dynamic hybrid 不优于 fixed origin，可靠状态提交的新增贡献不成立；不能把 TTC 改名后作为本项目方法。

### 2.8 Future Forcing

- 论文：[Future Forcing: Future-aware Training-free KV Cache Policy for Autoregressive Video Generation](https://arxiv.org/abs/2605.30083)
- 本地状态：未 vendored；当前只做论文级审计。
- 其核心先例：利用 canonical pre-RoPE query 的近似稳定性估计 future query，并做 future-aware cache eviction/merge。
- 与当前方法差异：Commit Forcing 的判断对象是“generated state 是否可长期提交”，证据是同一 block 的 denoising trajectory，作用位置是 pathwise sampling correction；当前不预测 future query，也不做 token eviction/merge。
- 禁止 claim：不能把 query stationarity、future proxy、future-aware cache policy 或 eviction/merge 归为本项目原创。

## 3. 完整 third-party 论文与代码台账

“许可证”只表示当前本地顶层文件的人工检查结果；未检测到不等于没有许可证，使用前必须再次核验 upstream。

| 本地目录 | 论文 | 原始代码 | 本地状态/许可证 | 与本项目的关系 |
|---|---|---|---|---|
| `Self-Forcing` | [Self Forcing](https://arxiv.org/abs/2506.08009) | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | 已 vendored；Apache-2.0 | **直接基座和修改目标** |
| `Causal-Forcing` | [Causal Forcing](https://arxiv.org/abs/2602.02214), [Causal Forcing++](https://arxiv.org/abs/2605.15141) | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) | 已 vendored；Apache-2.0 | 替代高动态 backbone 和跨基座验证 |
| `RollingForcing` | [Rolling Forcing: Autoregressive Long Video Diffusion in Real Time](https://arxiv.org/abs/2509.25161) | [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) | 已 vendored；**仅学术用途自定义许可证** | rolling window、attention sink、完整结构 anchor |
| `DeepForcing` | [Training-Free Long Video Generation with Deep Sink and Participative Compression](https://arxiv.org/abs/2512.05081) | [cvlab-kaist/DeepForcing](https://github.com/cvlab-kaist/DeepForcing) | 已 vendored；Apache-2.0 | deep sink、参与式 compression、固定预算对照 |
| `Pyramid-Forcing` | [Head-Aware Pyramid KV Cache Policy](https://arxiv.org/abs/2605.13111) | [if-lab-pku/Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | 已 vendored；Apache-2.0 + NOTICE | **上游基座与高碰撞近邻；本项目内部原型曾以该 vendored tree 为宿主，并非从上游直接移植 HREM 控制流** |
| `Forcing-KV` | [Forcing-KV](https://arxiv.org/abs/2605.09681) | [zju-jiyicheng/Forcing-KV](https://github.com/zju-jiyicheng/Forcing-KV) | **空目录；未完成本地审查；license 未核验** | static/dynamic head 先例 |
| 未 vendored | [Head Forcing](https://arxiv.org/abs/2605.14487) | [项目页](https://jiahaotian-sjtu.github.io/headforcing.github.io/)；code coming soon | 未复制代码；发布后需补做 license/commit 审计 | **local/anchor/memory head、hierarchical episodic memory 和 novelty update 的高碰撞先例** |
| `MemRoPE` | [MemRoPE](https://arxiv.org/abs/2603.12513) | [YoungRaeKimm/MemRoPE](https://github.com/YoungRaeKimm/MemRoPE) | 已 vendored；Apache-2.0 | pre-RoPE memory、online position indexing；legacy helper 来源 |
| `LongLive-RAG` | [LongLive-RAG](https://arxiv.org/abs/2606.02553) | [qixinhu11/LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG) | 已 vendored；Apache-2.0 | **retrieval history 高碰撞近邻** |
| `Echo-Forcing` | [Echo-Forcing](https://arxiv.org/abs/2605.16003) | [mingqiangWu/Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) | 已 vendored；MIT | **scene recall 高碰撞近邻；旧原型来源** |
| `IAMFlow` | [Advancing Narrative Long Video Generation via Training-Free Identity-Aware Memory](https://arxiv.org/abs/2605.18733) | [Eddie0521/IAMFlow](https://github.com/Eddie0521/IAMFlow) | 已 vendored；MIT；NOTICE 提醒部分文件可能为 CC-BY-NC-SA-4.0 | entity/state memory、VLM/LLM validation、NarraStream-Bench |
| `infinity-rope` | [Infinity-RoPE](https://arxiv.org/abs/2511.20649) | [yesiltepe-hidir/infinity-rope](https://github.com/yesiltepe-hidir/infinity-rope) | 已 vendored；Apache-2.0 | position extrapolation 与长期 action consistency |
| `FreePCA` | [FreePCA](https://arxiv.org/abs/2505.01172) | [JosephTiTan/FreePCA](https://github.com/JosephTiTan/FreePCA) | 已 vendored；**未检测到顶层 license** | PCA/low-rank long-short consistency；只读参考 |
| `DiT-Extrapolation` | [RIFLEx](https://arxiv.org/abs/2502.15894), [UltraViCo](https://arxiv.org/abs/2511.20123) | [thu-ml/DiT-Extrapolation](https://github.com/thu-ml/DiT-Extrapolation) | 已 vendored；Apache-2.0 | RoPE frequency 与长度外推；可与 memory position 组合 |
| `FreeLOC` | [Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](https://arxiv.org/abs/2603.25209) | [Westlake-AGI-Lab/FreeLOC](https://github.com/Westlake-AGI-Lab/FreeLOC) | 已 vendored；Apache-2.0 | layer probing、relative-position/context OOD 修正 |
| `MIGA` | [Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](https://arxiv.org/abs/2605.18233) | [XiaokunFeng/MIGA](https://github.com/XiaokunFeng/MIGA) | 已 vendored；**未检测到顶层 license** | self-reflection、long-range frame guidance；不同生成范式 |
| `LongVideoSparseAttention` | [LVSA](https://arxiv.org/abs/2605.31057) | [JiusiServe/LongVideoSparseAttention](https://github.com/JiusiServe/LongVideoSparseAttention) | 已 vendored；Apache-2.0 | rotating global anchors、固定稀疏预算、VQeval |
| `MotionCache` | [Motion-Aware Caching for Efficient AR Video Generation](https://arxiv.org/abs/2605.01725) | [MAC-AutoML/MotionCache](https://github.com/MAC-AutoML/MotionCache) | 已 vendored；**未检测到顶层 license** | motion importance 可用于诊断；主要是 denoising acceleration |
| `FlowCache` | [Flow Caching for Autoregressive Video Generation](https://arxiv.org/abs/2602.10825) | [mikeallen39/FlowCache](https://github.com/mikeallen39/FlowCache) | 已 vendored；**未检测到顶层 license** | chunkwise cache reuse 和固定预算 compression；主要是效率 |
| `SWIFT` | [Prompt-Adaptive Memory for Efficient Interactive Long Video Generation](https://arxiv.org/abs/2605.09442) | [ShanwenTan/SWIFT](https://github.com/ShanwenTan/SWIFT) | 已 vendored；Apache-2.0 | **semantic/head-wise memory 高碰撞近邻** |
| 未 vendored | [Pathwise TTC](https://arxiv.org/abs/2602.05871) | [xbxsxp9/Pathwise_TTC](https://github.com/xbxsxp9/Pathwise_TTC) | README/assets；未复制代码；license 需在源码发布后重审 | **当前 path correction 的最近先例与固定首帧 baseline** |
| 未 vendored | [Future Forcing](https://arxiv.org/abs/2605.30083) | 未记录未经核验的代码链接 | 论文级审计 | future-query cache policy 高碰撞先例 |

## 4. 后续可使用的工作及优先级

### P0：先完成新颖性和正确性审计，不立即融合更多模块

#### SWIFT mechanism collision audit

可做：

- 精读 semantic injection、head-wise injection、segment anchors 的公式和代码；
- 对比它是否已经形成“prompt evidence -> head-wise historical access”；
- 将差异落实到输入证据、选择对象、作用位置和 abstention，而不是只比较名称。

不可做：

- 直接把 SWIFT 的 semantic injection 政名为 episode admission；
- 未引用地使用其 head mask、窗口调度或 anchor 公式。

#### Echo-Forcing / LongLive-RAG baseline

优先做同 prompt、同帧数、同 backbone 或可解释 backbone 差异的 baseline。它们比继续增加内部消融更能决定论文是否有独立空间。

### P1：在 head-role P0 通过后考虑

#### MemRoPE：pooled-grid position legality

可能用途：为 stride-4 archive 定义真实 pooled-grid coordinates，在独立 branch 中做 bounded online RoPE indexing。

安全边界：

- 必须把 MemRoPE 作为直接技术先例；
- 如果移植频率 gather 或 index mapping，代码中标记来源并遵守 Apache-2.0；
- 贡献只能是“position mapping 与 factorized admission 的组合及其新场景”，不能是 online RoPE indexing 本身。

#### LongLive-RAG：CPU offload 和 descriptor retrieval

可能用途：扩展 archive 时长、减少 GPU resident memory、建立 learned/latent descriptor 对照。

安全边界：其 optional retrieval AE 涉及额外训练，不能在论文中继续统称为完全 training-free component；应把 base generator 无训练和 retriever 训练状态分别报告。

#### IAMFlow：entity-aware evaluation

可能用途：增加主体身份、属性和 narrative transition 指标；采用 NarraStream-Bench 或类似 entity-state annotation。

安全边界：IAMFlow 的 entity memory 是强相关贡献，不应直接融合后仍把 identity memory 归为自己；其 NOTICE 还提示部分代码可能有非商业条款。

### P2：性能和长时外推增强

#### LVSA

- 可采用 VQeval 检查“高一致性但实际冻结/循环”的假增益；
- 可把 rotating anchors 作为固定预算 attention baseline；
- 不应在 HREM 因果链尚未成立时加入 sparse attention，否则难以归因。

#### FreeLOC / RIFLEx / Infinity-RoPE

- 适合处理长时位置 OOD 和层敏感性；
- 可用 layer probing 选择 HREM active layers，但需说明 probing 来源；
- 不可把 position correction 带来的提升归给 episodic recall。

#### DeepForcing / RollingForcing

- 可作为 fixed-budget sink/compression baseline；
- 可用于证明 HREM 的收益不是仅因保留第一帧或更大历史预算；
- RollingForcing 许可证限定学术用途，不得默认用于商业或 production。

#### FlowCache / MotionCache

- 主要解决 denoising activation/cache reuse 的推理效率，不等于长期 K/V recall；
- 可在方法稳定后作为正交加速层；
- MotionCache 的 motion importance 可以作为诊断对照，但若进入 head gate，必须引用且重新定义消融，不能把其思想归为 HREM 原创。

#### FreePCA / MIGA

- FreePCA 可探索 archive low-rank compression；MIGA 可提供 long-range frame guidance/self-reflection 对照；
- 两者本地未检测到顶层 license，现阶段禁止复制代码，只可根据论文重新实现并引用；
- 两者与当前 AR K/V sidecar 的范式差异较大，优先级低于直接近邻 baseline。

## 5. 可调整但合规的论文故事

调整故事是允许的，前提是新的故事由我们的实现、实验和差异支撑，而不是隐藏来源。

### Story A：Factorized admission（当前首选，条件式）

适用条件：

- dual evidence 在多候选 episode 中明显优于 prompt-only/QK-only；
- online head gate 具有稳定选择性，并优于 all-head 与 PF fixed labels；
- motion 和 return 至少不形成明显 trade-off 退化。

安全 claim：

> We propose and evaluate a factorized admission view of historical recall: episode eligibility and per-head eligibility are estimated separately before bounded memory readout.

必须引用：Echo-Forcing、LongLive-RAG、SWIFT、Pyramid-Forcing、Forcing-KV。

### Story B：Selective episodic recall with abstention

适用条件：episode gate 成立，但 head gate 无稳定收益。

做法：删除“head-role contribution”，将 head gate 降为负结果或辅助分析，聚焦 non-recent multi-cue admission、wrong-episode safety 和 native fallback。

风险：与 Echo/SWIFT/LongLive-RAG 更接近，必须证明 multi-candidate return、明确 abstention 和严格 causal controls 带来独立价值。

### Story C：Diagnosing historical memory in long AR video

适用条件：完整方法收益不稳定，但 oracle、wrong/shuffled、motion freeze 和 gate trace 形成扎实诊断结论。

做法：转为 empirical/analysis paper，贡献是可复现的因果分解、失败模式和 evaluation protocol，而不是宣称一个 SOTA memory method。

安全性：这是比强行包装负结果更可信的路线，但需要更大规模实验和公开可复现分析。

### Story D：Position-legal factorized memory

适用条件：实现 pooled-grid position 后，position legality 是主要收益来源，且与 MemRoPE/FreeLOC 有清楚区别。

风险：MemRoPE 已覆盖 pre-RoPE memory 和 online indexing。不能把 position legality 本身作为新颖性，只能讨论它与 episode/head admission 的交互。

## 6. 建议和禁止使用的论文措辞

| 不建议/禁止 | 原因 | 建议替代 |
|---|---|---|
| “We introduce episodic KV memory.” | Echo、LongLive-RAG、MemRoPE 等已有历史记忆 | “We build an episode-indexed sidecar to study factorized admission.” |
| “We are the first to retrieve historical frames.” | LongLive-RAG 等直接覆盖 | “We condition retrieval on an admitted non-recent episode.” |
| “We discover head specialization.” | PF、Forcing-KV 已系统研究 | “Motivated by prior head specialization findings, we estimate online eligibility.” |
| “Our novel independent memory attention” | 独立/并行 attention 不是新概念 | “We use a separate bounded readout to preserve the native branch.” |
| “Our novel pre-RoPE memory” | MemRoPE 已明确覆盖 | “Following position-safe memory principles, we store pre-RoPE payloads.” |
| “HREM preserves motion.” | 当前尚无充分结果 | “We test whether selective head admission reduces all-head motion degradation.” |
| “significantly improves” 用于单 seed | 统计不成立 | 报告 paired mean、CI、seed 数和 effect size |
| “ours” 指代整个 SF/PF 派生 pipeline | 归属不清 | “Self-Forcing + our HREM module” 或明确组件名 |

## 7. 代码和实验合规工作流

### 7.1 引入第三方代码前

1. 记录论文、repository URL、commit/tag 和访问日期；
2. 检查顶层 LICENSE、NOTICE、子目录 SPDX 和模型权重 license；
3. 无许可证时默认不复制；联系作者或只根据论文独立实现；
4. 新建 provenance entry，列出 upstream file -> local file 映射；
5. 在移植代码块附近写简短来源注释，不删除原版权/SPDX；
6. 通过 git commit 保留移植和后续修改的历史。

### 7.2 写论文前

1. 为每个方法公式建立“最近先例”引用，不只在 Related Work 笼统引用；
2. 对 PF、Echo、LongLive-RAG、SWIFT、Forcing-KV 做逐项 mechanism table；
3. 图示必须重新绘制并引用思想来源，不能临摹对方配色、结构和 caption；
4. 文字必须独立撰写，不能近义改写论文摘要或 method 段落；
5. 对第三方结果标明“reported by prior work”，本项目结果标明复现实验设置；
6. 对复现失败、环境差异和缺失 baseline 如实报告；
7. 所有主结论对应预先定义的 metric、prompt suite 和多 seed 统计。

### 7.3 投稿前强制审计

- [ ] 当前 bibliography 包含本文列出的所有直接使用和高碰撞工作；
- [ ] `third_party/` 每个实际使用目录都有 LICENSE/NOTICE 审核记录；
- [ ] 所有 ported code 均有文件级或代码块级来源注释；
- [ ] Forcing-KV 空目录没有被描述为“已本地复现”；
- [ ] current HREM P0 没有被错误描述为使用 MemRoPE position mode；
- [ ] 旧 Echo/PF 原型结果没有被混入 HREM-v2 贡献；
- [ ] 所有 baseline 使用相同 prompt、seed、frame count、reset 和预算，或清楚解释差异；
- [ ] 没有用单样例“显著”、没有隐去失败 prompt；
- [ ] 标题、摘要和 Introduction 没有未经验证的 `first`/SOTA claim；
- [ ] 至少一名未参与实现的合作者按本文台账检查归属和引用；
- [ ] 按 AAAI 最新的 generative-AI、code attribution 和 reproducibility policy 再做一次正式检查。

## 8. 当前结论

LifeCache-v3 的完整 screen 和人工 review 已否定“继续微调 side-memory
fusion/head routing 就能获得可见提升”的当前假设。当前最可辩护、但仍待验证的
潜在贡献改为：

```text
denoising-trajectory disagreement
  -> reliability-gated generated-state admission
  -> bounded origin + trusted state lifecycle
  -> temporary pathwise correction with native recent-context recovery
```

其中 pathwise correction 明确继承 Pathwise TTC 思想；候选新增点只能是
reliability-gated state commit 及其受控生命周期。下一轮必须依次回答：

1. fixed-origin TTC-style correction 在当前 SF base 上是否真的改善可见质量；
2. dynamic hybrid 是否稳定优于 fixed origin，而不是只优于 native；
3. reliability 是否预测未来 block 的质量下降；
4. 在同 prompt、frame、seed 和 checkpoint 下是否能与官方 PF 公平比较；
5. 若任一步不成立，停止包装该贡献并按 docs/74 的 fallback 收缩。

Head/layer 分类不再是 P0 贡献。只有新的跨 prompt/seed counterfactual
evidence 证明其稳定有效后才允许恢复。这样可以避免在无收益时继续靠术语区分 PF，
并确保每个 claim 都由代码、实验和正确引用支撑。
