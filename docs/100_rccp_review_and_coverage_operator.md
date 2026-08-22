# RCCP 方案回顾与 Coverage Operator 探索

## 0. 重要澄清：实际对比的是什么

### 0.1 Pyramid-Forcing (PF) 的真正默认配置

PF 的默认配置（`pyramid-forcing.yaml` + `best_labels.csv`）使用 **三种 head 分类策略**：

| PF Label | 含义 | Head 数量 | Cache 策略 |
|----------|------|-----------|------------|
| `-1` (osc) | 振荡头 | 156 | `sink1 + cyclic(period=6, bucket=4) + recent4` |
| `1` (sta+) | 稳定+ | 172 | `sink3 + stride(interval=6, cap=4) + recent4` |
| `2` (sta-) | 稳定- | 32 | `sink3 + merge(patch=2, cap=4) + recent4` |

PF 的 middle 策略是 **cyclic/stride/merge** 三种，由 `best_labels.csv` 的 head 分类驱动。这是 PF 论文的方法。

### 0.2 我们实验中实际使用的配置

**所有实验（v181-v184）都没有使用 PF 的默认配置**。我们用了一套独立的 label 系统：

| 我们的 Label | 含义 | Cache 策略 | 启用方式 |
|-------------|------|------------|----------|
| `20` (recent) | 纯近期 | `sink1 + recent8` | `--pyramidkv_cache_compatibility_policy` |
| `21` (coverage) | 覆盖 | `sink1 + middle4 + recent4` | 同上 + `--pyramidkv_cache_compatibility_coverage_policy` |

当传入 `--pyramidkv_cache_compatibility_policy` 时，PF 原生的 label 系统（-1/1/2 + cyclic/stride/merge）被**完全绕过**，替换为我们的 20/21 label 系统。

### 0.3 各实验的实际配置

| 实验 | Method | Head Config | Cache 策略 | 实际是什么 |
|------|--------|-------------|-----------|-----------|
| v181/v183 | `all_recent` | 全 20 (360) | 全 sink1+recent8 | **非 PF 默认！纯 recent baseline** |
| v181/v183 | `rccp_matched` | 5×21 + 355×20 | 5 heads coverage + 355 recent | RCCP 选择性 coverage |
| v181/v183 | `all_coverage` | 全 21 (360) | 全 sink1+middle4+recent4 | 全覆盖 |
| v182 | `strict5_retrieval` | 5×21 + 355×20 | 5 heads coverage(retrieval) | RCCP + retrieval operator |
| v184 | `all_coverage_retrieval` | 全 21 (360) | 全 coverage(retrieval) | 全覆盖 + retrieval |

### 0.4 核心问题

**我们没有与 PF 论文的方法对比。** 我们的 "all_recent" baseline 是一个我们自己构造的简化策略（纯 sink1+recent8），不是 PF 的 `best_labels.csv`（cyclic/stride/merge 三策略）。PF 默认配置比 pure recent 复杂得多，可能效果也更好。

**正确的 PF baseline 应该是**：不传 `--pyramidkv_head_config_path` 和 `--pyramidkv_cache_compatibility_policy`，让 PF 用 yaml 中的默认 `best_labels.csv` 配置。

### 0.5 与 Self-Forcing (sf_native) 的关系

`sf_native` 是完全不同的 pipeline（Self-Forcing，非 Pyramid-Forcing），不使用 PyramidKV cache 压缩。它是作为另一个 baseline 存在的，不是 PF 的 baseline。

---

## 1. 研究背景

训练自由的长视频生成面临一个核心矛盾：**随着视频长度增加，KV cache 持续增长导致显存压力，同时生成质量（尤其是运动动态性）显著退化**。Self-Forcing 通过 next-frame 自回归生成长视频，但其原生 full cache 策略在长视频（60s+）中动态性衰减严重。Pyramid-Forcing (PF) 引入了 PyramidKV 分块缓存，通过 head 分类（osc/sta+/sta-）驱动 cyclic/stride/merge 三种 middle 策略压缩 cache，但仍未解决动态性退化问题。

**核心观察**：不同 attention head 对 cache 压缩的敏感度不同。一些 head 保留近期信息更有效（recent），而另一些 head 需要更广泛的覆盖（coverage: sink1 + middle4 + recent4）。如果能为每个 head 选择最优的 cache 策略，或许能在保持质量的同时提升动态性。

---

## 2. 之前的方案：RCCP（Residual Cache Compatibility Profiling）

### 2.1 方案设计

RCCP 的核心思路是**通过 profiling 为每个 attention head 分配最优的 cache 策略**：

1. **Calibration 阶段**（v173）：对 360 个 head（30 层 × 12 head）逐一计算 recent vs coverage vs episode 三种策略的 log error，找到每个 head 的"最佳策略"。
2. **Statistical Gating**：通过 8 个统计 gate（BH q-value、bootstrap CI、win fraction、AR stability 等）筛选出"显著支持 coverage 策略"的 head。
3. **v173 结果**：6 个 head 被 profiled 为 coverage（分布在 layer 0, 5, 6, 8, 23），其余 354 个为 recent。
4. **v176/v177 严格筛选**：进一步收紧到 5 个 head（layer 0/head 10, layer 5/head 3, layer 6/head 6, layer 8/head 6, layer 23/head 2），形成 `rccp_matched` 方案。
5. **Generation 验证**（v178-v181）：用这 5 个 head 的 coverage map 在 30s 和 60s 视频上生成，与 all_recent（PF 默认）和 all_coverage 对比。

### 2.2 实验结果

#### v181 60s Long Stress（128 prompts × 3 methods，VBench-Long Core-9）

| 指标 | sf_native | rccp_matched (5 heads) | all_recent (PF) | all_coverage |
|------|------|------|------|------|
| dynamic_degree ↑ | 0.33 | 0.56 | 0.51 | — |
| imaging_quality ↑ | 0.676 | 0.686 | 0.686 | — |
| identity_background ↑ | 0.963 | 0.961 | 0.962 | — |

**Paired Analysis（v183, 30s, 128 prompts）**：

| 对比 | dynamic_degree delta | 显著性 |
|------|------|------|
| rccp vs all_recent | +0.006 | **不显著** (q=1) |
| all_coverage vs all_recent | +0.110 | **显著** (q=2e-7) |
| rccp vs all_coverage | -0.104 | **显著劣势** |

#### Decision
- v181: `long_horizon_rccp_not_confirmed`（两个 seed 一致）
- v183: `stop_static_strict5_and_revisit_operator`
- 正式 RCCP membership claim **不被允许**

### 2.3 为什么舍弃 RCCP

RCCP 方案在实验中暴露了三个根本问题：

**问题 1：选择性 coverage 的增量价值极小**

RCCP 的核心假设是"只有少数关键 head 需要 coverage，其余保持 recent"。但实验表明：
- rccp_matched（5 heads coverage）vs all_recent（0 heads coverage）的 dynamic_degree 差值仅 +0.006（30s），统计不显著
- 真正的动态性提升来自 PF pipeline 本身（all_recent 0.51 vs sf_native 0.33 = +55%），而非 head 选择

**问题 2：全覆盖（all_coverage）反而更优**

all_coverage（360 heads 全用 coverage）在动态性和质量上都显著优于 rccp_matched：
- dynamic_degree: 0.60 vs 0.50（+20%）
- official_quality: 83.1 vs 82.3（+0.8）
- sparse_vs_dense_coverage 对比中，rccp 在动态性和质量上都显著劣势

这说明 **RCCP 的"选择性 head"路线本身是错误的**——coverage 策略应该应用到所有 head，而非少数。

**问题 3：Profiled head 不比随机选择更好**

v183 明确指出："cannot establish that RCCP chose better heads than count/layer-matched alternatives"。即 RCCP 精心 profiled 的 5 个 head，并不比随机选 5 个 head 做得更好。这动摇了整个 profiling 方法论的基础。

**问题 4：Identity 轻微但显著退化**

rccp_matched 在 late_half 的 identity_background 有统计显著的退化（-0.004），导致 quality+identity gate 未通过。虽然退化很小，但在严格的 gate 标准下无法通过。

**根本原因**：RCCP 试图在"保持 cache 压缩"和"提升动态性"之间找平衡点（只给少数 head coverage），但实验证明这个平衡点不存在——要么全覆盖（动态性最高、质量略升），要么全 recent（PF 默认）。中间路线反而两边都不讨好。

---

## 3. 当前方案：Coverage Operator 探索（v182）

### 3.1 方案转向

基于 RCCP 的失败，我们转向 **Coverage Operator 探索**：
- **不再纠结"哪些 head 需要 coverage"**（RCCP 的 profiling 路线）
- **关注"coverage 策略本身如何更好地工作"**（operator 设计）

核心问题变为：**当所有 head 都用 coverage 策略时，不同的"中间帧选择方法"如何影响生成质量？**

### 3.2 Coverage Operator 定义

Coverage 策略的 cache 结构为 `sink1 + middle4 + recent4`（共 9 frames）。其中 sink1 和 recent4 是固定的，**关键在于 middle4 如何选择**——即从历史帧中选择哪些 4 帧作为"中间覆盖"。

v182 测试了 5 种 middle4 选择策略：

| Operator | middle4 选择策略 | 直觉 |
|----------|------|------|
| **all_recent** | 无 coverage（PF 默认，sink1+recent8） | 基线，不使用 coverage |
| **strict5_reservoir** | Reservoir sampling（均匀随机采样） | 公平但无优先级 |
| **strict5_landmark** | 选择"地标帧"（场景变化最大的帧） | 保留关键转折 |
| **strict5_prototype** | 选择"原型帧"（聚类中心） | 保留代表性内容 |
| **strict5_retrieval** | 基于 retrieval 的选择 | 语义相关性 |

注意：这 5 种策略都用相同的 5 个 RCCP profiled head（v177 strict5），但 v182 的 decision 是 `reprofile_structured_coverage_operator`——意味着如果选定 operator，需要用该 operator 重新 profiling head 选择。

### 3.3 v182 实验结果

**VBench-Long Core-9（16 prompts × 5 methods, 30s 视频）**：

| Operator | dynamic_degree ↑ | imaging_quality ↑ | aesthetic_quality ↑ |
|----------|------|------|------|
| all_recent (PF) | 0.688 | 0.708 | 0.616 |
| strict5_reservoir | 0.738 | 0.703 | 0.618 |
| strict5_landmark | 0.742 | 0.709 | 0.622 |
| strict5_prototype | 0.717 | 0.706 | 0.618 |
| **strict5_retrieval** | **0.771** | 0.710 | 0.620 |

**关键发现**：
1. **所有 coverage operator 都优于 all_recent 的动态性**（+4% 到 +12%）
2. **strict5_retrieval 最佳**：dynamic_degree 0.771 vs 0.688（+12%），且质量基本持平
3. **strict5_landmark 次之**：0.742，质量指标略优
4. **Pareto front**：strict5_landmark 和 strict5_retrieval 在非劣解前沿

### 3.4 下一步方向

v182 的 decision 是 `reprofile_structured_coverage_operator`，意味着：
1. 选定最佳 operator（retrieval 或 landmark）
2. 用该 operator 重新 profiling head 选择（可能不再是 5 个 head，而是更多或全部 head）
3. 在长视频（60s）上验证

---

## 4. 方案演进总结

```
v173: RCCP Calibration — 6 heads profiled as coverage
  ↓
v176/v177: Strict Superset — 收紧到 5 heads (strict5)
  ↓
v178-v181: Generation 验证 — RCCP vs all_recent vs all_coverage
  ↓ 结论: RCCP 增量极小，all_coverage 更优，profiling 不比随机好
  ↓
v182: Coverage Operator 探索 — 5 种 middle4 选择策略
  ↓ 结论: retrieval > landmark > reservoir > prototype > all_recent
  ↓
下一步: 用最佳 operator reprofile，在 60s 视频上验证
```

### 核心教训

1. **Profiling 路线（RCCP）失败**：选择性 coverage 不如全覆盖，精心 profiled 的 head 不比随机选择更好
2. **Operator 路线（v182）有前景**：不同的 coverage operator 确实带来差异化效果，retrieval 策略显著提升动态性
3. **动态性提升的来源**：主要来自 PF pipeline + coverage operator 设计，而非 head 选择
4. **未来方向**：从"哪些 head"转向"如何做 coverage"，聚焦 operator 设计与优化

---

## 5. 与 Pyramid-Forcing (PF) 的关系（已澄清）

### 关键区分

- **Pyramid-Forcing**：一种长视频生成 pipeline（分块生成），使用 PyramidKV 压缩 cache
- **PyramidKV 默认配置**：通过 `best_labels.csv` 把 360 个 head 分为 osc(-1)/sta+(1)/sta-(2) 三类，分别用 cyclic/stride/merge 三种 middle 策略
- **我们的 cache_compatibility_policy**：一套独立的 label 系统（20=recent/21=coverage/22=episode），**绕过** PF 原生的 head 分类

### 实际对比关系

| 对比 | 含义 | 是否公平 |
|------|------|----------|
| 我们 vs all_recent | coverage operator vs 纯 recent | **不公平** — all_recent 不是 PF 默认 |
| 我们 vs sf_native | PF pipeline vs Self-Forcing pipeline | 不同 pipeline，参考性对比 |
| 我们 vs PF 默认 | coverage operator vs cyclic/stride/merge | **未做** — 需要补跑 |

### 下一步必须补做

1. **生成真正的 PF baseline**：用 `best_labels.csv` + 不传 `--pyramidkv_cache_compatibility_policy`
2. **在相同 prompt set 上对比**：PF baseline vs 我们的最优 coverage operator
3. **确认 PF 的 head 分类是否本身已经解决了动态性问题**

---

## 附录：实验版本索引

| 版本 | 内容 | 规模 | 结论 |
|------|------|------|------|
| v173 | RCCP Calibration | 360 heads profiling | 6 heads coverage |
| v176 | Superset RCCP | 扩展 profiling | 收紧 gate |
| v177 | Strict Superset | 严格筛选 | 5 heads (strict5) |
| v178 | Holdout Generation | 30s, 128 prompts | 初步验证 |
| v180 | Fresh128 | 30s, 512 videos | 4 methods 对比 |
| v181 | Long Stress | 60s, 576 videos | not_confirmed |
| v182 | Structured Coverage | 30s, 80 videos | retrieval 最佳 |
| v183 | v180 Recovery | 30s, 512 videos | stop_static_strict5 |
