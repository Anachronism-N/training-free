# SF 验证实验矩阵规划

> 状态：只读规划文档，未 commit/push。所有 phase 的门槛在生成前冻结，失败则保留为负结果。
> 本文是 SF backend 上验证 CEMR+CEG 独立增量的权威矩阵，引用 `docs/55_cemr_ceg_full_idea_spec.md` 第 7.3 节投稿门槛。
> 任何与 `docs/55` 冲突的旧 PF-only 结论，在 SF 阶段以本文为准。

---

## 0. 背景与定位

### 0.1 为什么需要 SF 验证

消融1（v66，`runs/v66_pf_uniform_ablation_metrics.json`）已证明在 PF backend 上，**PF head 分类是当前效果的主要来源**：

- `with_head`（PF native + head routing）：P1 full return margin `+0.0073`，B 形成通过 validity gate。
- `uniform`（去掉 head 分类，所有 head 同等 retention）：P1 full margin `-0.2210`，B 段被红色脚手架 + 室内白墙污染，A2 灾难性崩坏（starburst sky / 棋盘格地面），validity=false。

即 PF 的 retention 策略本身已经吃掉了 baseline 的主要信号，当前所有 CEMR/CEG 实验都在 `with_head` 之上，无法独立证明 archive+CEG 有 **超出 PF retention 的增量**。`docs/55` 第 7.2 节已明确将 "backend-independent" 列为不可写 claim。

为满足 `docs/55` 第 7.3 节门槛 9（"在非 PF backend 复现核心机制"），必须把 CEMR+CEG 移植到 native Self-Forcing，并在 SF baseline 之上验证独立增量。general-purpose-4 正在执行 task #29（archive + CEG + readout 注入），本文档冻结移植完成后的验证矩阵。

### 0.2 已有 SF 资产

| 资产 | 路径 | 说明 |
|---|---|---|
| SF native A-B-A 120f | `runs/sf_native_aba_120f/{0,1,2}-0_ema.mp4` | A-B-A 强 prompt，3 条，seed 0，**未跑 strong P1/P2 manifest** |
| SF native 普通 120f | `runs/sf_native_120f/{0,1,2}-0_ema.mp4` | 普通长视频，3 条，seed 0 |
| Strong A-B-A prompt（P1+P2） | `prompts/aba_validity_strong_p1p2_dev.txt` | P1 hard-cut rooftop\|\|gym\|\|rooftop；P2 hard-cut cafe\|\|street\|\|cafe |
| Strong P1 dev prompt | `prompts/aba_validity_strong_p1_dev.txt` | 单 P1，v62/v63b/v64/v65b 专用 |
| Validity evaluator | `scripts/evaluate_aba_return.py` | 支持 manifest、baseline admission、candidate failure accounting |
| v62 PF baseline metrics | `runs/v62_validity_{nocfg,cfg3}_metrics.json` | P1 no-CFG B formation pass；P2 no-CFG fail；CFG3 P1/P2 都 pass 但画质退化 |
| v65b A2-only 4-cell | `runs/v65b_p1_a2only_2x2_metrics.json` | checksum 不一致，仅观察 |
| SF pipeline 入口 | `third_party/Self-Forcing/inference.py` | 当前未含 `structured_memory` / `pyramidkv_*` flag（待 task #29 注入） |
| Run wrapper | `scripts/run_v35_pf_value_refresh.sh` | PF 专用，SF 需独立 wrapper 或扩展该脚本 |

### 0.3 已知 PF 上的关键数值（用于 SF baseline 校准）

| 方法 | P1 full margin | P1 BG margin | Validity |
|---|---:|---:|---|
| PF no-CFG (v62) | `+0.0073` | `-0.1132` | pass |
| PF CFG3 (v62) | `+0.2042` | `+0.0103` | pass（画质退化） |
| A1 oracle prior-on (v64) | `+0.1428` | `-0.1133` | pass |
| A1 oracle prior-off (v64) | `+0.2060` | `+0.0967` | pass |
| CEG strict (v63b) | `+0.0469` | `-0.0979` | pass |
| CEG relative (v63b) | `-0.3967` | `-0.3998` | pass |
| PF uniform head (v66, ablation1) | `-0.2210` | `-0.0726` | **fail** |

---

## 1. 总体执行顺序与 go/no-go gate

```text
Phase 1: Smoke test ──[gate G1]──> Phase 2: A-B-A validity baseline
                                            │
                                            ├─[G2a]─> Phase 3: SF + memory 主对照
                                            │              │
                                            │              └─[G2b]─> Phase 4: SF + CEG
                                            │                              │
                                            │                              └─[G3]─> Phase 5: 多 prompt 多 seed
                                            │                                              │
                                            │                                              └─[G4]─> Phase 6: 普通长视频
                                            │
                                            └─(G2 fail)─> STOP / 回退到 fallback 路径
```

**Gate 定义（任何 gate 失败则停止后续 phase，进入第 8 节 fallback）：**

| Gate | 位置 | 条件 | 失败后果 |
|---|---|---|---|
| G1 | Phase 1 后 | gate=0 bitwise 等价；gate>0 不 NaN/不崩 | 移植有 bug，回 task #29 修复，不进入 Phase 2 |
| G2 | Phase 2 后 | SF native 在 strong P1 上 B formation pass（与 PF 同 validity gate） | SF baseline 不成立，CEMR+CEG 在 SF 上无 estimand，进入 fallback 8.1 |
| G3 | Phase 3+4 后 | oracle 相对 SF native 有正 full margin；normal 不显著差于 SF native；CEG relative winner=A1 且 full margin 接近 oracle | CEMR+CEG 在 SF 上无独立增量，进入 fallback 8.2/8.3 |
| G4 | Phase 5 后 | 满足 `docs/55` 第 7.3 节门槛 1-7、10 | 不投，保留为负结果 |
| G5 | Phase 6 后 | 满足 `docs/55` 第 7.3 节门槛 7、8、9 | 不投，保留为负结果 |

---

## 2. Phase 1: Smoke test（移植完成后立即跑）

### 2.1 目的

验证 task #29 的 SF 移植不破坏 native SF 输出，且 memory branch 在 gate>0 时不崩溃。

### 2.2 实验矩阵

| Cell | Backend | Memory | Gate | Prompt | Frames | Seed | 产出 |
|---|---|---|---|---|---|---|---|
| 1a | SF native | off | — | `prompts/aba_validity_strong_p1_dev.txt` 第 0 行 | 30 | 0 | `runs/sf_smoke/gate0/sf_native_30f/0-0_ema.mp4` |
| 1b | SF + archive+CEG | on，gate=0 | 0 | 同上 | 30 | 0 | `runs/sf_smoke/gate0/sf_memory_30f/0-0_ema.mp4` |
| 1c | SF + archive+CEG | on，gate>0 | 0.075（与 PF v62 对齐） | 同上 | 30 | 0 | `runs/sf_smoke/gate_pos/sf_memory_30f/0-0_ema.mp4` |

### 2.3 通过标准（G1）

- **1a vs 1b bitwise 等价**：逐帧 PSNR ≥ 60 dB，或 `torch.allclose` 在 bfloat16 容差内。这是 `docs/55` 第 4.5 节 "任意 abstain 时 w=0，输出 bitwise 等于 native" 的直接验证。
- **1c 不 NaN / 不崩**：视频完整解码 30f，所有 latent 有限，无 NaN/Inf；admission rate > 0（证明 memory branch 真的被调用）。
- **不要求 1c 有视觉增量**，只要不崩就过 G1。

### 2.4 资源

- 1 GPU，3 cells × 30f ≈ 3 × 2 min = 6 min 生成 + 5 min bitwise 校验。
- 可与 Phase 2 的 SF native baseline 准备并行。

---

## 3. Phase 2: A-B-A validity baseline

### 3.1 目的

在 strong P1/P2 上确认 SF native 的 B formation 能力，建立 SF baseline 的 validity 集合。这是后续所有 SF episodic-return 实验的 estimand 前提。

### 3.2 实验矩阵

| Cell | Backend | CFG | Prompt | Frames | Seed | 复用/新建 |
|---|---|---|---|---|---|---|
| 2a | SF native no-memory | off (CFG=1) | `prompts/aba_validity_strong_p1p2_dev.txt` | 120 | 0 | **需新建**（现有 `runs/sf_native_aba_120f` 未确认是否用 strong p1p2 prompt，需核对 `run_meta.txt`；若不是则重跑） |
| 2b | SF native no-memory | CFG=3 | 同上 | 120 | 0 | 新建 |

**核对动作**：先读 `runs/sf_native_aba_120f` 同目录的 `run_meta.txt` 或 `run.log`，确认 prompt 文件 hash。若已用 strong p1p2 prompt 且 seed=0、120f，直接复用 2a；否则重跑。

### 3.3 评估

用 `scripts/evaluate_aba_return.py` + 新建 manifest `prompts/sf_aba_validity_phase2_manifest.json`：

```json
{
  "baseline": "sf_native_nocfg",
  "latent_frames": 120,
  "scene_ranges": {"a1": [0,42], "b": [42,81], "a2": [81,120]},
  "windows": {"a1": [0.35,0.8], "b": [0.55,0.9], "a2": [0.20,0.85]},
  "methods": {
    "sf_native_nocfg": "runs/sf_phase2/sf_native_nocfg_s0",
    "sf_native_cfg3": "runs/sf_phase2/sf_native_cfg3_s0"
  },
  "samples": [
    {"id":"p1_s0","prompt_index":0,"seed":0,
     "validity": {
       "sf_native_nocfg": {"pass": null, "source":"manual", "evidence":""},
       "sf_native_cfg3":  {"pass": null, "source":"manual", "evidence":""}
     }},
    {"id":"p2_s0","prompt_index":1,"seed":0,
     "validity": {
       "sf_native_nocfg": {"pass": null, "source":"manual", "evidence":""},
       "sf_native_cfg3":  {"pass": null, "source":"manual", "evidence":""}
     }}
  ]
}
```

人工审查 manifest 中 `pass` 字段（参考 `docs/55` 第 4.7 节 validity 标准：B 场景必须人工确认形成 + scene-only CLIP/DINO 分离）。

### 3.4 通过标准（G2）

- **G2a (必需)**：SF native no-CFG 在 P1 上 B formation pass。如果 SF no-CFG 本身 B 不形成，则用 CFG3 版本作为 SF baseline（对应 PF CFG3 路径），但需在 G2b 记录 CFG3 画质退化。
- **G2b (记录)**：若 CFG3 才能让 B 形成，记录画质退化程度（与 PF CFG3 同样的过饱和问题），并在后续 phase 的 fusion weight 上更保守。
- **G2c (P2 optional)**：P2 在 PF 上 no-CFG fail、CFG3 pass。SF 上 P2 若 no-CFG 也 fail，则后续主对照只在 P1 上做；若 SF P2 no-CFG pass，则 P2 也纳入主对照（更强的 estimand）。

### 3.5 资源

- 1-2 GPU（2a、2b 可并行），2 cells × 120f ≈ 2 × 8 min = 16 min 生成 + 10 min DINO 评估 + 20 min 人工 validity 审查。
- 与 Phase 1 并行。

### 3.6 决策树

```text
2a P1 pass? ─yes─> SF baseline = sf_native_nocfg，进入 Phase 3
            ─no──> 2b P1 pass? ─yes─> SF baseline = sf_native_cfg3（记录退化），进入 Phase 3
                              ─no──> G2 fail，进入 fallback 8.1
```

---

## 4. Phase 3: SF + memory 主对照

### 4.1 目的

在 valid P1（Phase 2 确定的 SF baseline 集合）上，验证：
1. SF + A1 oracle 是否相对 SF native 有正 full margin（证明 archive+readout 在 SF 上有信号，复现 PF v62 的 oracle 增量）。
2. SF + memory(normal dual) 是否不显著差于 SF native（证明 normal 路径不破坏 SF baseline，对应 PF 上 normal dual 失败的对照）。

### 4.2 实验矩阵

| Cell | Backend | Episode selection | Frame prior | Activation | Prompt | Frames | Seed |
|---|---|---|---|---|---|---|---|
| 3a (baseline) | SF native | — | — | — | strong P1 (p1p2 dev 第 0 行) | 120 | 0 |
| 3b (oracle) | SF + archive | forced A1 oracle | off | A2-only (`activation_episode=2`) | 同上 | 120 | 0 |
| 3c (normal dual) | SF + archive | normal dual retrieval (query) | off | A2-only | 同上 | 120 | 0 |

参数对齐 PF v62/v64 oracle 配置（`docs/55` 第 5.4 节）：
- `MEMORY_STORAGE_MODE=archive`, `MEMORY_ARCHIVE_MAX_FRAMES=24`, `MEMORY_ARCHIVE_POLICY=coverage`
- `MEMORY_TOP_K_FRAMES=3`, `MEMORY_RECENT_EXCLUDE_FRAMES=4`
- `MEMORY_SELECTION_POLICY=query`, `MEMORY_FUSION_MODE=convex`, `MEMORY_READOUT_MODE=clean_only`
- `MEMORY_GATE=0.075`, `MEMORY_CONFIDENCE=0.25`, `MEMORY_TEMPERATURE=0.3`
- `MEMORY_LAYER_START=15`, `MEMORY_LAYER_END=21`, `MEMORY_WARMUP_BLOCKS=6`
- `MEMORY_HEAD_ROUTING=static`（SF 无 PF head label，此 flag 在 SF 上语义待 task #29 确认；若 SF 移植使用 all-head，则记录为已知差异）
- `MEMORY_EPISODE_GATE_MODE=off`（3c 用 normal dual，不是 CEG）
- `MEMORY_ORACLE_EPISODE_ID=0`（3b 强制 episode 0 = A1）
- `MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2`（A2-only）

### 4.3 评估

新建 manifest `prompts/sf_phase3_manifest.json`，baseline=`sf_native`，samples 为 Phase 2 valid 的 P1（+ P2 if G2c pass）。运行 `scripts/evaluate_aba_return.py`，产出 `runs/sf_phase3_metrics.json`。

### 4.4 通过标准（G3a）

- **3b oracle full margin > 3a SF native full margin**（paired，单 prompt 单 seed 仅观察，正式统计在 Phase 5）。若 oracle 无正增量，说明 archive+readout 在 SF 上无信号，进入 fallback 8.2。
- **3c normal dual full margin 不显著差于 3a SF native**（容差 ±0.05）。若 normal 大幅差于 SF native（如 PF 上 normal `-0.3504` vs PF `+0.0073`），说明 memory branch 在 SF 上有破坏性副作用，需排查 fusion weight 或 head routing。

### 4.5 资源

- 1-2 GPU（3b、3c 可并行），2 cells × 120f ≈ 16 min 生成 + 10 min DINO + 15 min trace 审查。
- 依赖 Phase 2 完成（G2 pass）。

---

## 5. Phase 4: SF + CEG

### 5.1 目的

在 valid P1 上验证 CEG 在 SF 上能否复制 oracle：
- CEG strict vs CEG relative vs SF native。
- A2-only activation（避免 B 段 trajectory 差异，对应 `docs/55` 第 4.6 节）。

### 5.2 实验矩阵

| Cell | Backend | Episode selection | Frame prior | Activation | Prompt | Frames | Seed |
|---|---|---|---|---|---|---|---|
| 4a (baseline) | SF native | — | — | — | strong P1 | 120 | 0 |
| 4b (oracle) | SF + archive | forced A1 oracle | off | A2-only | 同上 | 120 | 0 |
| 4c (CEG strict) | SF + archive | contrastive_strict | off | A2-only | 同上 | 120 | 0 |
| 4d (CEG relative) | SF + archive | contrastive_relative | off | A2-only | 同上 | 120 | 0 |

参数同 Phase 3，但：
- `MEMORY_EPISODE_GATE_MODE=contrastive_strict`（4c）/ `contrastive_relative`（4d）
- `MEMORY_EPISODE_FRAME_PRIOR_MODE=off`（4c、4d 均 off，对应 v65b relative-off 占优格）
- `MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2`（A2-only）
- `MEMORY_TRACE_ENABLED=1`，产出 `episodic_trace.jsonl` 供 episode selection 审计。

### 5.3 评估

- 同 Phase 3，manifest `prompts/sf_phase4_manifest.json`，产出 `runs/sf_phase4_metrics.json`。
- **额外 trace 审计**：读取 4c/4d 的 `episodic_trace.jsonl`，确认：
  - 4c (strict)：预期 abstain（`docs/55` 第 4.3.1 节，P1 上 `L(A1)=-0.0108<0`）；
  - 4d (relative)：winner=episode 0 (A1)，A1 mass ≈ 1，B mass = 0（复现 PF v63b trace）。

### 5.4 通过标准（G3b）

- **4d relative winner=A1**（trace 验证）。若 winner != A1，CEG selection 在 SF 上失败。
- **4d relative full margin 接近 4b oracle**（容差 ±0.05）。若 relative 远低于 oracle（如 PF v63b relative `-0.3967` vs oracle `+0.1428`），CEG 仍未复制 oracle，进入 fallback 8.3。
- **4c strict full margin 不显著差于 4a SF native**（strict 预期 abstain，应 bitwise 等价 baseline；若不等价，排查 abstain 路径）。

### 5.5 资源

- 2-3 GPU（4b/4c/4d 可并行；4a 复用 Phase 3 的 3a），3 cells × 120f ≈ 24 min 生成 + 15 min DINO + 20 min trace 审计。
- 依赖 Phase 3 完成（G3a pass）。

---

## 6. Phase 5: 多 prompt 多 seed

### 6.1 目的

在 ≥12 held-out validity-passed A-B-A triplets、3 seeds 上预注册复验，满足 `docs/55` 第 7.3 节门槛 1-7、10。

### 6.2 Triplet 构造

构造 ≥12 held-out triplets，4 类各 3 条（与 `docs/55` 第 7.3 节门槛一致）：

| 类别 | 描述 | 示例 |
|---|---|---|
| C1: 同主体强场景切换 | A 主体在场景 X → B 场景 Y → A2 回到 X | P1 parkour rooftop\|\|gym\|\|rooftop |
| C2: 同动作硬负例 | A 动作在场景 X → B 同动作场景 Y（易混淆）→ A2 回到 X | 跑步公园\|\|跑步街道\|\|公园 |
| C3: 视觉相近语义不同 | A 场景 X → B 视觉相近但语义不同场景 Y → A2 回到 X | 暖色咖啡馆\|\|暖色书店\|\|咖啡馆 |
| C4: A2 paraphrase | A 场景 X → B 场景 Y → A2 用不同措辞描述回到 X | 测试 prompt descriptor 对 paraphrase 鲁棒性 |

**held-out 要求**：12 条 triplets 不得与 dev set（`prompts/aba_validity_strong_p1p2_dev.txt` 的 P1/P2）重复。建议构造后冻结到 `prompts/sf_phase5_heldout_12.txt` 并 commit hash 记录。

### 6.3 实验矩阵

| Cell | Backend | 方法 | Seeds | Triplets | 总样本 |
|---|---|---|---|---|---|
| 5a | SF native | — | {0,1,2} | 12 | 36 |
| 5b | SF + archive | A1 oracle, A2-only | {0,1,2} | 12 | 36 |
| 5c | SF + archive | CEG relative, A2-only | {0,1,2} | 12 | 36 |
| 5d (control) | SF + archive | shuffle-previous control | {0,1,2} | 12 | 36 |

参数同 Phase 4 的 4d（relative-off, A2-only）。5d 是 `docs/55` 第 7.3 节门槛 6 的 specificity control。

### 6.4 评估

- 每条 triplet × seed 需人工 validity（B formation pass）。validity 字段在 manifest 中预填 `null`，生成后人工填。
- `scripts/evaluate_aba_return.py` 产出 per-triplet-seed metrics。
- **cluster bootstrap**：以 triplet 为 cluster，10000 次 resample，计算 95% CI（对应 `docs/55` 第 7.3 节 "cluster-bootstrap lower bound"）。

### 6.5 通过标准（G4，预注册）

满足 `docs/55` 第 7.3 节门槛 1-7、10：

1. **Episode selection**：CEG (5c) 在 A2 选择 A1 ≥ 10/12；选择 B = 0/12；A-B-C 安全集 abstain ≥ 5/6。
2. **Return preservation**：paired `ΔA1-A2` 95% cluster-bootstrap lower bound ≥ -0.01。
3. **Leakage reduction**：paired `ΔB-A2` upper bound ≤ 0。
4. **Return margin**：paired `Δmargin` 95% CI lower bound > 0，≥ 9/12 triplet mean wins，≥ 24/36 triplet-seed wins。
5. **灾难尾部**：`Δmargin vs SF native <= -0.05` 不超过 2/36。
6. **Specificity**：CEG (5c) 显著优于 shuffle-previous control (5d)，CI > 0。
7. **质量安全**：subject/background/aesthetic/imaging/motion 任何维度均值不得低于 SF native 超过 0.005；Dynamic 不得低于 SF native 超过 0.02；人工盲审严重 flashback / 额外肢体 / 背景硬切率不增加 > 5pp。
10. **人工盲评**：artifact rate 不增加。

**门槛冻结**：所有阈值在生成前冻结，失败则保留为负结果，不对 triplet 子集、λ、LR threshold 做事后搜索。

### 6.6 资源

- 4 方法 × 12 triplets × 3 seeds = 144 runs。
- 单 run 120f ≈ 8 min，4 GPU 并行 ≈ 144 × 8 / 4 = 288 min ≈ 5 小时生成。
- DINO 评估 144 视频 ≈ 30 min。
- 人工 validity 36 个视频（12 triplets × 3 seeds，但每个 triplet-seed 需看 3 段）≈ 2-3 小时。
- 人工盲评（门槛 10）≈ 1-2 小时。
- **依赖 Phase 4 完成（G3b pass）**。

---

## 7. Phase 6: 普通长视频

### 7.1 目的

验证 SF + 最终方法（Phase 5 胜出配置）在普通长视频上不降质，满足 `docs/55` 第 7.3 节门槛 7、8、9。

### 7.2 实验矩阵

| Cell | Backend | 方法 | Prompts | Frames | Seed |
|---|---|---|---|---|---|
| 6a | SF native | — | `third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt` (32) | 120 (≈30s) | 0 |
| 6b | SF + 最终方法 | Phase 5 胜出 | 同上 | 120 | 0 |

若 Phase 5 未决出胜出配置，则用 CEG relative (A2-only) 作为默认候选。

### 7.3 评估

- VBench-Long（`scripts/aggregate_v55_and_vbench.sh` 的 SF 版本，需新建 `scripts/aggregate_sf_vbench.sh`）。
- 6 维度：subject_consistency, background_consistency, aesthetic_quality, imaging_quality, motion_smoothness, dynamic_degree。
- 产出 `runs/vbench_long/sf_phase6_{native,ours}/results_eval_results.json`。

### 7.4 通过标准（G5）

满足 `docs/55` 第 7.3 节门槛 7、8、9：

7. **质量安全**：subject/background/aesthetic/imaging/motion 任何维度均值不得低于 SF native 超过 0.005；Dynamic 不得低于 SF native 超过 0.02。
8. **Cost**：报告 latency / peak VRAM / archive memory / descriptor scan / readout overhead。
9. **Backend**：在非 PF backend（native Self-Forcing）复现核心机制。**Phase 6 本身就是门槛 9 的最终验证。**

### 7.5 资源

- 2 cells × 32 prompts = 64 runs，4 GPU 并行 ≈ 64 × 8 / 4 = 128 min ≈ 2 小时生成。
- VBench 6 维度评估 64 视频，2 GPU 并行 ≈ 1-2 小时。
- **依赖 Phase 5 完成（G4 pass）**。

---

## 8. 风险与 fallback

### 8.1 SF native B formation 比 PF 差（G2 fail）

**场景**：SF native no-CFG 和 CFG3 在 strong P1 上 B 都不形成。

**原因**：SF 的 causal latent trajectory 场景惯性可能比 PF 更强（`docs/55` 第 2.1 节），hard-cut prompt 不足以让 SF 离开 A 场景。

**fallback**：
1. **更强 prompt**：在 `||` 分隔符前后加更强烈的视觉冲突描述（如 "completely different lighting, color palette, and geometric layout"）。
2. **更高 CFG**：尝试 CFG=5 或 CFG=7（记录画质退化）。
3. **改用 Causal-Forcing**：`third_party/Causal-Forcing/` 作为第三个 backend，CF 的 cache 策略与 SF 不同，可能更易切换场景。
4. **降级 claim**：若所有 backend 都无法在 no-CFG 下 B formation，则 CEMR+CEG 的 episodic-return claim 仅在 CFG3 stress test 下成立，对应 `docs/55` 第 5.3 节 "CFG3 严重过饱和，仅作 stress test" 的限制。论文 claim 降级为 "在 CFG3 stress test 下的 episodic return"，不满足门槛 9 的 native 复现。

### 8.2 SF + memory 在 valid P1 上无增量（G3a fail: oracle 无正增量）

**场景**：3b oracle full margin ≤ 3a SF native full margin。

**原因**：
- SF 的 native cache 策略已经隐式保留了 A1 信息，archive 是冗余的；
- SF 的 attention 路径与 PF 不同，memory branch 注入点（layer 15-21）在 SF 上不是正确的 readout 层；
- RoPE 处理不当（`docs/self_forcing_patch_skeleton.md` 提到的 causal_rope_apply）。

**fallback**：
1. **layer sweep**：在 SF 上跑 layer {5-10, 10-15, 15-20, 20-25, 25-30} 的 oracle sweep，找到有正增量的 layer 区间。
2. **readout mode sweep**：`clean_only` vs `all` vs `noisy_only`。
3. **fusion weight sweep**：gate {0.05, 0.10, 0.15, 0.20, 0.30}。
4. **若所有 sweep 都无增量**：说明 CEMR 在 SF 上无独立信号，论文 claim 限制为 "PF-specific episodic memory"，不满足门槛 9，不投。

### 8.3 SF + CEG 仍无法复制 oracle（G3b fail: relative 远低于 oracle）

**场景**：4d relative winner=A1（trace 正确）但 full margin 远低于 4b oracle。

**原因**：这是 PF v63b 的同一现象（`docs/55` 第 5.5 节 "正确 episode 是必要但不充分"）。在 SF 上若复现，说明问题不在 episode selection，而在 within-episode visual retrieval 或 fusion。

**fallback**：
1. **冻结 archive 离线 readout**（`docs/55` 第 6 节下一步）：固定 A1/B trajectory，在 A2 注入不同 episode decision，隔离 episode decision vs within-episode scorer。
2. **frame prior on/off**：跑 4d 的 frame prior on 格（对应 v65b 4-cell），但需先确认 checksum 一致性（v65b 在 PF 上 checksum 不一致，无法归因）。
3. **episode-conditioned readout**：`docs/55` 第 10 节下一步 4，设计根据 episode id 调整 readout 的机制，不再调 episode score。
4. **降级 claim**：若 CEG 始终无法复制 oracle，则论文只 claim oracle 诊断结论（"episode mis-selection 是主要可定位原因"），不 claim CEG 是可部署方法。这仍满足 `docs/55` 第 7.1 节 "当前可写" 的 oracle 诊断 claim，但不满足门槛 1-6 的 CEG claim。

---

## 9. 资源规划汇总

### 9.1 GPU 需求与时间

| Phase | Runs | GPU | 生成时间 | 评估时间 | 人工时间 |
|---|---|---|---|---|---|
| 1 Smoke | 3 | 1 | 6 min | 5 min bitwise | 0 |
| 2 Validity baseline | 2 | 1-2 | 16 min | 10 min DINO | 20 min validity |
| 3 SF + memory 主对照 | 3 | 1-2 | 16 min | 10 min DINO | 15 min trace |
| 4 SF + CEG | 4 | 2-3 | 24 min | 15 min DINO | 20 min trace |
| 5 多 prompt 多 seed | 144 | 4 | 288 min (≈5h) | 30 min DINO | 2-3h validity + 1-2h 盲评 |
| 6 普通长视频 | 64 | 4 | 128 min (≈2h) | 1-2h VBench | 0.5h 抽查 |
| **总计** | 220 | — | ≈9h 生成 | ≈3h 评估 | ≈5-6h 人工 |

### 9.2 并行性

| 并行组 | Phases | 条件 |
|---|---|---|
| A | Phase 1 + Phase 2 准备 | 无依赖，移植完成后立即开始 |
| B | Phase 2 评估 + Phase 3 生成 | Phase 2 生成完成后，Phase 3 可在另一 GPU 开始（但 G2 决策需等 Phase 2 评估） |
| C | Phase 3 + Phase 4 生成 | 若 G3a 乐观预期，可提前启动 Phase 4 生成（但 G3a 决策需等 Phase 3 评估） |
| D | Phase 5 4 方法并行 | 4 GPU 各跑 1 方法 × 12 triplets × 3 seeds |
| E | Phase 6 2 方法并行 | 2 GPU 各跑 1 方法 × 32 prompts |

**保守串行**：Phase 1 → 2 → 3 → 4 → 5 → 6，总 ≈ 9h 生成 + 3h 评估 + 6h 人工 ≈ 18h wall clock（单 GPU 串行生成）。
**乐观并行**：4 GPU 并行，生成阶段 ≈ 9h / 4 ≈ 2.5h，总 ≈ 8-10h wall clock。

### 9.3 Go/No-Go Gate 汇总

| Gate | Phase 后 | 失败后果 | 是否停止 |
|---|---|---|---|
| G1 | Phase 1 | 移植 bug，回 task #29 | 停止，修复后重跑 Phase 1 |
| G2 | Phase 2 | SF baseline 不成立 | 停止 Phase 3+，进入 8.1 |
| G3a | Phase 3 | archive+readout 无信号 | 停止 Phase 4，进入 8.2 |
| G3b | Phase 4 | CEG 无法复制 oracle | 停止 Phase 5，进入 8.3 |
| G4 | Phase 5 | 不满足投稿门槛 1-7、10 | 停止 Phase 6，保留为负结果 |
| G5 | Phase 6 | 不满足投稿门槛 7、8、9 | 不投，保留为负结果 |

---

## 10. 代码路径与依赖

### 10.1 现有代码（复用）

| 用途 | 路径 | 说明 |
|---|---|---|
| Validity evaluator | `scripts/evaluate_aba_return.py` | 直接复用，新建 SF manifest |
| Strong A-B-A prompt | `prompts/aba_validity_strong_p1p2_dev.txt` | P1/P2 hard-cut |
| 32-prompt 普通 long video | `third_party/Pyramid-Forcing/prompts/MovieGenVideoBench_num32.txt` | VBench-Long 标准 32 条 |
| VBench-Long eval | `$ROOT/../research_sprint/bench_baselines/VBench/vbench2_beta_long/eval_long.py` | 6 维度评估 |
| Run wrapper (PF) | `scripts/run_v35_pf_value_refresh.sh` | 参考，SF 需独立 wrapper |
| Aggregation (PF) | `scripts/aggregate_v55_and_vbench.sh` | 参考，SF 需独立脚本 |

### 10.2 待 task #29 产出的代码

| 用途 | 预期路径 | 依赖 |
|---|---|---|
| SF pipeline 注入 | `third_party/Self-Forcing/pipeline/causal_diffusion_inference.py` 或 `causal_inference.py` | archive commit、\|\| schedule、scene 切换 |
| SF attention 注入 | `third_party/Self-Forcing/wan/modules/attention/core.py` 或 `causal_model.py` | memory branch、fusion |
| SF run wrapper | `scripts/run_sf_validation.sh`（新建） | 对齐 `run_v35_pf_value_refresh.sh` 的 env flag 接口 |
| SF aggregation | `scripts/aggregate_sf_vbench.sh`（新建） | 对齐 `aggregate_v55_and_vbench.sh` |

### 10.3 新建 manifest 清单

| Manifest | Phase | 说明 |
|---|---|---|
| `prompts/sf_aba_validity_phase2_manifest.json` | 2 | SF native P1/P2 validity |
| `prompts/sf_phase3_manifest.json` | 3 | SF + memory 主对照 |
| `prompts/sf_phase4_manifest.json` | 4 | SF + CEG |
| `prompts/sf_phase5_heldout_12.txt` | 5 | 12 held-out triplets（冻结） |
| `prompts/sf_phase5_manifest.json` | 5 | 多 prompt 多 seed |
| `prompts/sf_phase6_manifest.json` | 6 | 32-prompt 普通 |

---

## 11. 与 `docs/55` 门槛的映射

| `docs/55` 第 7.3 节门槛 | 本文验证 Phase | 备注 |
|---|---|---|
| 1. Episode selection | Phase 5 (5c) | CEG 在 A2 选择 A1 ≥ 10/12 |
| 2. Return preservation | Phase 5 | `ΔA1-A2` lower bound ≥ -0.01 |
| 3. Leakage reduction | Phase 5 | `ΔB-A2` upper bound ≤ 0 |
| 4. Return margin | Phase 5 | `Δmargin` CI > 0, ≥ 9/12, ≥ 24/36 |
| 5. 灾难尾部 | Phase 5 | `Δmargin <= -0.05` ≤ 2/36 |
| 6. Specificity | Phase 5 (5c vs 5d) | CEG 显著优于 shuffle-previous |
| 7. 质量安全 | Phase 5 + Phase 6 | VBench 6 维度 + 人工盲审 |
| 8. Cost | Phase 6 | latency / VRAM / archive memory |
| 9. Backend | Phase 2-6 全程 | **本文核心：SF native 复现** |
| 10. 人工盲评 | Phase 5 | artifact rate |

---

## 12. 执行检查清单

- [ ] task #29（SF 移植）完成，gate=0 bitwise 等价 native SF
- [ ] Phase 1 smoke test 通过 G1
- [ ] Phase 2 SF native validity baseline 通过 G2
- [ ] Phase 3 SF + memory 主对照通过 G3a
- [ ] Phase 4 SF + CEG 通过 G3b
- [ ] Phase 5 多 prompt 多 seed 通过 G4
- [ ] Phase 6 普通长视频通过 G5
- [ ] 所有 manifest 冻结 commit hash
- [ ] 人工 validity 与盲评记录归档
- [ ] 满足 `docs/55` 第 7.3 节全部门槛后，统一 commit + push + 撰稿
