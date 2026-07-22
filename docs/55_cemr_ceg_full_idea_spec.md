# CEMR + Contrastive Episode Gate：完整方法规格与论文定位

> 状态：go for method development；no-go for stable superiority或submission claim。
> 本文是当前 idea 的权威描述，覆盖历史 v3–v64 的所有结论与失败，并冻结下一阶段实验门槛。
> 任何与本文冲突的旧文档段落，以本文为准。

---

## 1. 一句话定位

在 training-free 长视频扩散中，提出一个**与 native recent-cache 正交的、bounded、content-addressable 的 episodic visual memory**，通过 **contrastive episode selection + within-episode visual retrieval + decoupled memory attention** 解决“远期事件被淘汰后无法按内容重新访问”的问题。它不重新定义 PF 的 retention 策略，也不把历史 K/V 拼回原生 softmax。

---

## 2. 问题与动机

### 2.1 现有 training-free 路径的局限

- PF / RollingForcing / Forcing-KV 解决“有限 native cache 中保留什么 token”，本质是 **retention**。
- LongLive-RAG 类 retrieval 把历史 token 拼回原 self-attention，改变 softmax 分母，造成冻结、重影、错误回放。
- AR 视频扩散的 causal latent trajectory 有强烈场景惯性；简单切换 prompt 不足以让模型离开当前场景。
- Mean-pooled prompt descriptor 不能理解否定语；A2 描述里出现的“B 物体 no longer visible”反而让 A2 与 B 更接近。

### 2.2 核心问题

> 如何在不训练检索器、不修改原生局部 attention、不无限增长 memory 的条件下，**按 episodic content 重新访问已被 native cache 淘汰的远期视觉事件**？

### 2.3 关键观察（来自 v62–v64 诊断）

1. Normal dual retrieval 在 P1 A2 段从未选择 A1：
   `A1 mass = 0, B mass = 0.600, current-A2 mass = 0.400`。
2. A1 oracle 强制只读 episode 0，full return margin 从 `-0.3504` 提升到 `+0.1428`；oracle 的 effective fusion weight 反而更低。**主因是 episode selection，不是融合强度或 archive 丢失。**
3. A1 payload 始终存在，`missing_payload = 0`；coverage archive 没有把 A1 清空。
4. `cos(B, A2) = 0.8541 > cos(A1, A2) = 0.7823`；否定语义让 mean-pooled descriptor 方向错误，这是原 Dual-Cue convex blend 失败的直接原因。
5. `recent_exclude = 4` 不足以防止 A2 后段自我回读早期 A2；必须用显式 episode 排除。

---

## 3. 与 PF 的机制边界

| 维度 | PF | CEMR + CEG |
|---|---|---|
| 解决的问题 | 有限 cache 中保留什么 | 远期 episode 如何按内容重新访问 |
| 决策时序 | offline static head label | online episode-level decision |
| 单位 | token | clean full-frame episode |
| Native attention | 改写 cache 内容 | 完全不变 |
| Memory branch | 无 | 独立 memory attention |
| Scene/episode provenance | 无 | 显式 episode ID + prompt descriptor sidecar |
| False-memory 控制 | 无 | current/previous exclusion + fail-closed abstain |

**可主张的差异**：CEMR 是 orthogonal episodic-memory add-on over PF（或任何 recent-cache backend）。
**不可主张**：backend-independent（尚未在 native Self-Forcing / Causal-Forcing 上验证）。

---

## 4. 方法完整规格

### 4.1 总体流程

```text
native recent-cache backend (PF / SF / CF)
        │
        ▼
clean pass of each completed block
        │   写入完整 H×W K/V 帧
        ▼
bounded episodic archive
        │   query-independent coverage maintenance
        │   episode id / interval / prompt descriptor sidecar
        ▼
online episode selection (CEG)
        │   contrastive ranking over historical episodes
        │   current episode 硬排除；previous winner reject
        │   metadata/payload 缺失 fail-closed
        ▼
within-episode visual retrieval
        │   仅在选定 episode 内用 visual score 做 top-k / softmax
        ▼
independent memory attention
        │   独立 logits/softmax，不进入 native softmax
        ▼
confidence / alignment / head-mask controlled fusion
        │   x = (1-w) * x_native + w * x_memory
        ▼
output
```

### 4.2 Clean episodic archive

- 每个 completed clean block 写入完整空间 K/V 帧，不存 noisy denoising eviction。
- Archive 有固定预算 `archive_max_frames`（实验用 24）。
- 写入时同时保存：
  - episode id；
  - frame interval `[start, end]`；
  - 该 block 的 normalized prompt descriptor（mean-pooled UMT5）；
  - clean K/V。
- 维护策略：query-independent **greedy k-center coverage** over frame descriptors，保留端点帧。
- 严格不变量：episode id / interval / prompt descriptor sidecar 必须与 K/V 严格同长；append、coverage prune、reset、device 迁移均同步；shape 异常在修改实例 payload 前 raise。

### 4.3 Contrastive Episode Gate (CEG)

对每个历史 episode `e`，令：

```text
t_cur   = normalized current-segment prompt descriptor
t_prev  = normalized previous-segment prompt descriptor
t_e     = normalized archived prompt descriptor of episode e

s_cur(e)  = cos(t_cur,  t_e)
s_prev(e) = cos(t_prev, t_e)
L(e)      = s_cur(e) - s_prev(e)
```

候选集构造：

```text
C = { e in archived_episodes :
      remote_time_mask(e) and e != current_episode_id }
```

**关键 correctness invariant**：

- current episode 从候选构造阶段即硬排除，不受 `recent_exclude` 影响；
- previous episode 保留在竞争中以暴露其高分；若 winner == previous，则 abstain；
- metadata 缺失、shape 异常、winner 无 surviving payload → fail-closed abstain，不回退 visual-all。

#### 4.3.1 Admission 策略

两种可审计策略，默认均 `off`：

- `contrastive_strict`：accept iff `max L > 0` 且 winner != previous。
- `contrastive_relative`：accept iff winner != previous 且 `L(winner) > L(previous)`，不要求绝对正值。

**结构性冲突（已在 P1 实测）**：

```text
L(A1) = 0.7823 - 0.7931 = -0.0108
L(B)  = 0.8541 - 1.0000 = -0.1459
```

- strict 预期 abstain；
- relative 预期选 A1，winner-vs-previous gap ≈ `+0.1351`。

**不允许**：
- 从 P1 单例拟合新阈值；
- 宣称 relative 能防 A-B-C false recall（无 A-B-C 安全集验证）。

### 4.4 Within-episode visual retrieval

CEG accept 后：

- `allowed_episode_id = winner`；
- 仅在该 episode 的 surviving 帧上计算 visual score：

```text
v_h(m) = cos(mean_q_h, mean_K_mh)
```

- shared top-k = `TopK_m mean_h v_h(m)`；
- frame weights = `softmax(v_h / tau_visual)` 仅在 winner episode 内；
- confidence / margin / entropy 仅基于 episode 内 visual 分布；
- CEG active 时 **关闭** semantic frame prior blend，`combined_scores == visual_scores`。

### 4.5 Independent memory attention & fusion

- Native PF attention 保持不变。
- Memory branch 独立：

```text
x_native = attention(q, native_k, native_v)
x_memory = memory_attention(q, archive_k, archive_v)
x        = (1 - w) * x_native + w * x_memory
```

- `w = gate × confidence × alignment × memory_head_mask`；
- convex mode 下 `w` 再 clamp 到 `[0, 1]`；
- 任意 abstain 时 `w = 0`，输出 bitwise 等于 native。

### 4.6 A2-only activation

为支持干净的因果消融，CEG 可仅在 `current_episode_id >= activation_episode` 时激活，默认 `activation_episode = 1`。设为 2 时，A1/B 段完全走 normal readout，只有 A2 启用 gate。这是后续 held-out 因果实验的必要控制。

### 4.7 Validity-aware A-B-A benchmark

- 评估器按实际 block schedule 计算 scene window，不使用硬编码百分比；
- baseline (PF/no-memory) 的 B formation 必须先通过 validity gate（人工 + 可选 CLIP/DINO），否则该样本不定义 episodic-return estimand；
- baseline valid 但 candidate B 失败 → 计入 candidate failure rate；
- return metric 与 failure rate 必须联合报告，不允许只报 return；
- 旧 v59/v60/v61 中 B 未形成的样本已被剔除，不再作为 episodic recall 证据。

---

## 5. 当前实验证据

### 5.1 普通 32-prompt（v55，seed 0，无 prompt prior）

```text
PF:     subject 0.97761  bg 0.96548  aest 0.64681  imag 0.72195  motion 0.98718  dyn 0.58750
CEMR:   subject 0.97882  bg 0.96671  aest 0.64739  imag 0.72667  motion 0.98747  dyn 0.56250
```

5/6 小幅正向，Dynamic `-0.025`。**这不是 Dual-Cue 证据**（v55 没有启用 prompt prior），只支持基础 archive/readout 的温和趋势，且未跨 seed 稳健。

### 5.2 线性 Dual-Cue v61 多 seed 统计（已降级为失败基线）

3 prompts × 3 seeds 配对：

| 指标 | mean Δ | bootstrap 95% CI | exact p | wins |
|---|---:|---:|---:|---:|
| Full return margin | `+0.01696` | `[-0.1045, 0.1401]` | `0.867` | 4/9 |
| Background margin | `+0.01046` | `[-0.0502, 0.0786]` | `0.785` | 5/9 |
| A1–A2 | `+0.00370` | `[-0.0757, 0.0831]` | `0.875` | 5/9 |
| B–A2 (lower=better) | `-0.01325` | `[-0.0997, 0.0796]` | `0.809` | 5/9 |

VBench seed 1/2：

```text
overall +0.00385（主要来自 Dynamic +0.04444）
imaging -0.01555, aesthetic -0.00782, background -0.00125
```

**结论**：不能声称 stable / robust / significant improvement。

### 5.3 旧 A-B-A benchmark 多数无效

人工审查 + scene-only CLIP：

- P0 transition 最好；
- P1 边缘且 seed 依赖；
- P2 多数未离开 A，DINO 高相似度来自场景持续 / 冻结 / 近回放。

强 hard-cut prompt + CFG3 可让 P1/P2 都形成 B，但 CFG3 严重过饱和，仅作 stress test。

### 5.4 P1 有效案例的 oracle 诊断（v62 / v64）

| 方法 | Full margin | BG margin | A1 mass | B mass | current mass |
|---|---:|---:|---:|---:|---:|
| PF | `+0.0073` | `-0.1132` | — | — | — |
| Normal dual | `-0.3504` | `-0.5220` | 0.000 | 0.600 | 0.400 |
| A1 oracle (prior on) | `+0.1428` | `-0.1133` | ≈1.0 | 0 | 0 |
| A1 oracle (prior off) | **`+0.2060`** | **`+0.0967`** | ≈1.0 | 0 | 0 |

- Oracle effective fusion weight 反而低于 normal → 排除“融合不够强”；
- `missing_payload = 0` → 排除 coverage 丢失；
- **主因是 episode selection**。

### 5.5 CEG 开发集实测（v63b）

| 方法 | Full margin | BG margin |
|---|---:|---:|
| CEG strict | `+0.0469` | `-0.0979` |
| CEG relative | `-0.3967` | `-0.3998` |

- Trace 验证 relative winner = episode 0，A1 mass ≈ 1，B/current mass = 0；
- 但端到端 return 未复制 oracle；
- relative 从 B 段开始改变 trajectory，与 oracle 不是单因素对照；
- 78 个匹配 A2 layer/block 中 selected intervals `0/78` 完全一致。

**只能判定**：正确 episode 是必要但不充分；不能判 CEG 整体有效或无效。

### 5.6 v64 frame-prior 2×2（复合路径依赖诊断，非纯因果）

| Episode gate | Frame prior | Full margin | BG margin |
|---|---|---:|---:|
| Oracle | on | `0.1428` | `-0.1133` |
| Oracle | off | **`0.2060`** | **`0.0967`** |
| Relative | off | `-0.3967` | `-0.3998` |
| Relative | on | `-0.2337` | `-0.1638` |

- Prior 对 relative 有部分修复，对 oracle 反而恶化；
- 78 个匹配 layer/block selected intervals 仍 `0/78` 完全一致；
- **保留为复合路径依赖诊断**：relative 从 B 段改变了 trajectory / archive，不能作为固定 payload 上的纯因果消融；
- v65b A2-only 四格已替代其作为下一阶段判别依据（见 5.7 / 第 6 节）。

### 5.7 v65b A2-only 四格（checksum 不一致，仅观察不归因）

在 P1 strong、seed0、120f、no-CFG、`activation_episode=2` 下重跑四格，所有公共参数与 v64 一致
（coverage archive24、top3、recent exclude4、prompt prior .50、convex gate .075、confidence .25、
temperature .3、layers 15:21、static routing、trace enabled、warmup6）。四组均完成、产出 477 帧 MP4、
完整 A2 trace；admission rate 96.06%–97.44%。

| Episode gate | Frame prior | Full margin | BG margin |
|---|---|---:|---:|
| Oracle | on | **`0.2650`** | **`0.1908`** |
| Oracle | off | `-0.3712` | `-0.4130` |
| Relative | off | **`0.3118`** | **`0.2030`** |
| Relative | on | `-0.2935` | `-0.3550` |

数值与 v64 完全反转：v64 中 oracle-off 与 relative-on 占优，v65b 中 oracle-on 与 relative-off 占优；
relative-off 的 full margin `0.3118` 甚至超过 oracle-on 的 `0.2650`。

**checksum 审计结论**：四组在 A2 transition（episode_id=2, frame=81）的 memory-layer archive K/V
checksum、archive intervals、archive episode ids **全部不一致**；latest clean-block input checksum
也不一致（`7e3680d4` / `b6637b7a` / `27d53a5d` / `c3335a7d`）。

因此 **v65b 不能做因果归因**：relative 从 B 段开始改变了 trajectory / archive / clean input，
四组并非同 A1/B trajectory 上的纯 A2-only 干预。relative-off 数值占优同样不能归因于 frame prior，
因为 archive 内容本身已经不同。v64 与 v65b 的数值反转进一步确认端到端生成使该 2×2 不是固定 payload
消融；需要冻结 archive 的离线 readout 才能隔离 episode decision 与 within-episode scorer。

---

## 6. v65b A2-only 四格结果（观察性，非因果）

目的：在同版本、同 A1/B trajectory 下，隔离 episode decision 与 within-episode scorer。

设计：

| 组 | episode 选择 | frame prior | activation |
|---|---|---|---|
| 1 | forced A1 oracle | on | A2-only |
| 2 | forced A1 oracle | off | A2-only |
| 3 | relative CEG | on | A2-only |
| 4 | relative CEG | off | A2-only |

**前提**：四组 A2 开始前的 archive intervals / episode ids / K/V checksum / clean-block input 必须一致；否则只报告观察，不做因果归因。

**结果**：

- 四组均完成、产出 477 帧 MP4、完整 A2 trace；admission rate 96.06%–97.44%。
- DINO full/background margin（validity-aware，PF baseline-admitted）：

| Episode gate | Frame prior | Full margin | BG margin |
|---|---|---:|---:|
| Oracle | on | **`0.2650`** | **`0.1908`** |
| Oracle | off | `-0.3712` | `-0.4130` |
| Relative | off | **`0.3118`** | **`0.2030`** |
| Relative | on | `-0.2935` | `-0.3550` |

- 数值与 v64 完全反转（v64 oracle-off / relative-on 占优，v65b oracle-on / relative-off 占优）。

**checksum 审计**：四组在 A2 transition（episode_id=2, frame=81）的 memory-layer（15–20）archive
K/V weighted_checksum、archive intervals、archive episode ids **全部不一致**；latest clean-block input
checksum 也不一致（`7e3680d4` / `b6637b7a` / `27d53a5d` / `c3335a7d`）。例：layer 15 archive_k
checksum 分别为 `09dc2d26` / `39f152dd` / `9d1cba95` / `b70210d5`。

**判别**：

- 落入“checksum 不一致 → 端到端生成本身使单因素隔离不可行”分支；
- 不做因果归因；relative-off 数值占优同样不能归因于 frame prior；
- 下一步需要冻结 archive 的离线 readout，才能隔离 episode decision 与 within-episode scorer。

**运行产物**：

```
runs/v35_pf_value_refresh/20260721_v65b2_a2only_{oracle,relative}_prior{on,off}_s0/
  pf_refresh_v65b2_a2only_*/0-0_ema.mp4
  pf_refresh_v65b2_a2only_*/episodic_trace.jsonl
  pf_refresh_v65b2_a2only_*/run.log
  pf_refresh_v65b2_a2only_*/run_meta.txt
runs/v65b_p1_a2only_2x2_manifest.json
runs/v65b_p1_a2only_2x2_metrics.json
```

---

## 7. 论文 claim 边界（当前可写 vs 不可写）

### 7.1 当前可写

- 提出 training-free bounded coverage archive + decoupled memory attention + contrastive episode retrieval 的组合。
- 在 controlled A-B-A 上通过 A1 oracle 证明长视频失败的主要可定位原因是 episode mis-selection，而非 archive 丢失或融合强度不足。
- 线性 Dual-Cue convex blend 在 3 seeds × 3 prompts 上不显著，应作为失败基线。
- Mean-pooled prompt descriptor 不能理解否定，导致 A2 与 B 语义距离反常。
- Validity-aware A-B-A benchmark 是必要的；旧 aggregate 把场景冻结误判为 recall。

### 7.2 当前不可写

- stable / robust episodic return across seeds / scenes；
- significantly outperforms PF；
- Pareto improvement；
- false-memory prevention / protection；
- correct memory causally superior to wrong / shuffled（多 seed 未做）；
- backend-independent / general to SF / CF；
- semantic / entity memory；
- first full-frame retrieval / scene recall。

### 7.3 投稿门槛（预注册，未完成不投）

在 ≥12 个 held-out、validity-passed A-B-A triplets、≥3 seeds 上：

1. **Episode selection**：CEG 在 A2 选择 A1 ≥ 10/12；选择 B = 0/12；A-B-C 安全集 abstain ≥ 5/6。
2. **Return preservation**：paired `ΔA1-A2` 95% cluster-bootstrap lower bound ≥ -0.01。
3. **Leakage reduction**：paired `ΔB-A2` upper bound ≤ 0。
4. **Return margin**：paired `Δmargin` 95% CI lower bound > 0，≥ 9/12 triplet mean wins，≥ 24/36 triplet-seed wins。
5. **灾难尾部**：`Δmargin vs PF <= -0.05` 不超过 2/36。
6. **Specificity**：CEG 显著优于 shuffle-previous control（CI > 0）。
7. **质量安全**：subject/background/aesthetic/imaging/motion 任何维度均值不得低于 PF 超过 0.005；Dynamic 不得低于 PF 超过 0.02；人工盲审严重 flashback / 额外肢体 / 背景硬切率不增加 > 5pp。
8. **Cost**：报告 latency / peak VRAM / archive memory / descriptor scan / readout overhead。
9. **Backend**：在非 PF backend（native Self-Forcing 或 Causal-Forcing）复现核心机制。
10. **人工盲评**：artifact rate 不增加。

所有门槛 / 矩阵在生成前冻结；失败则保留为负结果，不对 λ、LR threshold、triplet 子集做事后搜索。

---

## 8. Review 指南

### 8.1 普通长视频候选（30s）

```text
runs/REVIEW_v53_candidate_30s/
  0_pf_vs_ours.mp4
  1_pf_vs_ours.mp4
  2_pf_vs_ours.mp4
```

### 8.2 当前 episodic return 数值最好的诊断（oracle，非可部署方法）

```text
runs/v35_pf_value_refresh/20260721_v64_p1_oracle_prioroff_s0/
  pf_refresh_v64_p1_oracle_prioroff_s0/0-0_ema.mp4
```

### 8.3 对照组

- PF baseline：
  `runs/v35_pf_value_refresh/20260721_v62_validity_strong_nocfg_s0/.../0-0_ema.mp4`
- Normal dual（错误检索）：
  `runs/v35_pf_value_refresh/20260721_v62_p1_normal_trace_s0/.../0-0_ema.mp4`
- A1 oracle（initial）：
  `runs/v35_pf_value_refresh/20260721_v62_p1_oracle0_trace_s0/.../0-0_ema.mp4`
- CEG strict / relative：
  `runs/v35_pf_value_refresh/20260721_v63b_p1_ceg_{strict,relative}_s0/.../0-0_ema.mp4`

人工重点：B gym 是否完全形成；A2 是否残留白墙/蓝垫/攀爬架；主体 clone；衣着瞬变；A2 是否真正回到开放屋顶。

---

## 9. 代码与 Git 状态

当前未 commit/push 的 tracked 改动（15）：

```text
docs/52_preliminary_paper_candidate.md
docs/53_cemr_paper_outline.md
docs/54_dual_cue_cemr_result.md
scripts/evaluate_aba_return.py
scripts/run_v35_pf_value_refresh.sh
src/lifecycle_kv/attention_fusion.py
tests/test_structured_memory_readout.py
third_party/Pyramid-Forcing/inference.py
third_party/Pyramid-Forcing/pipeline/causal_diffusion_inference.py
third_party/Pyramid-Forcing/pipeline/causal_inference.py
third_party/Pyramid-Forcing/pipeline/pyramidkv_config.py
third_party/Pyramid-Forcing/pyramidkv/adaptive_cache.py
third_party/Pyramid-Forcing/tests/test_adaptive_cache.py
third_party/Pyramid-Forcing/tests/test_pipeline_config.py
third_party/Pyramid-Forcing/wan/modules/attention/core.py
```

Untracked 但应提交（6）：

```text
docs/55_cemr_ceg_full_idea_spec.md            （本文）
prompts/aba_validity_strong_p1_dev.txt
prompts/aba_validity_strong_p1p2_dev.txt
prompts/aba_validity_v62_cfg3_manifest.json
prompts/aba_validity_v62_nocfg_manifest.json
tests/test_evaluate_aba_return.py
```

不提交：`runs/` 下所有 mp4 / log / trace / metrics JSON。

联合测试历史最高 `86 passed`；checksum 修复后需重跑完整套件再统一 commit + push。

---

## 10. 下一步执行顺序

1. 修复 boundary summary helper API 缺口，恢复完整联合测试。
2. v65b A2-only 四格生成 + checksum 审计 + 因果分析。
3. 若 CEG 机制成立：构造 ≥12 held-out validity-passed triplets + A-B-C 安全集，预注册后多 seed 复验。
4. 若 CEG 仍失败：定位 episode 内 frame weighting / payload / temporal readout 瓶颈，设计 episode-conditioned readout，不再调 episode score。
5. 最终方法固定后：matched 32-prompt 普通 long video、非 PF backend、成本、人工盲评。
6. 全部通过投稿门槛后，统一 commit + push + 撰稿。
