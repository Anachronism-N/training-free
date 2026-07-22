# Head-Role-Aware KV Retention for Training-Free Long Video Generation

> 状态：详细设计阶段，未 commit/push。
> 本文定义 Echo-Forcing + Head-Role-Aware Memory 的完整方法、实现、实验规划。
> 相比 CEMR+CEG（docs/55），本方法的创新点在于 **利用 attention head 的自然功能分化**，无需外部标签、无需 episode archive。

---

## 0. 为什么放弃 CEMR+CEG 而转向这个方向

### 0.1 CEMR+CEG 在 SF 上的问题

SF 验证 Phase 1-4（docs/56）揭露出三个根本性问题：

1. **SF native 场景惯性太强**：P2（cafe→street→cafe）的 hard-cut 完全失败，模型无法离开 cafe。这意味着我们的 episodic-return estimand 只在特定 prompt 上成立。

2. **Memory branch 无条件破坏了 B-formation**：即使 episode gate 100% 正确，convex fusion 也会在 abstain 时衰减 native attention，6 层累积后完全杀死 B 场景。

3. **CEG 正确但不充分**：即使 gate 决策完美（trace 验证），B-formation 仍然无法恢复。correct episode 是 necessary but not sufficient（同 PF v63b 结论）。

### 0.2 方向重新评估

| 维度 | CEMR+CEG (旧) | Head-Role-Aware (新) |
|---|---|---|
| Backend | Self-Forcing（场景惯性太强） | **Echo-Forcing**（已有 scene pool + preserve/recall/forget 基础设施） |
| Head 标签 | PF pre-trained（不可移植） | **在线 Q/K 统计（universal）** |
| Memory 粒度 | Episode-level frame archive | **Per-head token retention** |
| 集成复杂度 | pipeline + causal_model + attention_fusion（4 文件, 500+ 行） | **pipeline single-file（~100 行核心）** |
| 实验 gate | 6 phases, 10 gates, 220 runs | **4 cells, ~16 runs** |
| 讲故事难度 | backend-independent claim 不能写 | **head specialization is universal** |

### 0.3 为什么 Echo-Forcing 更适合当前阶段

Echo-Forcing 已有：
- **Scene transition 系统**（smooth/hardcut/recall），直接产生可观的 A-B-A 场景切换
- **Scene pool**（压缩存储、语义召回），可复用为 per-head 记忆的基础设施
- **Working long-video pipeline**（672 帧, MovieGenBench 32 prompts），可直接对比 VBench

我们只需要在它的 pipeline 上加 **~100 行代码** 改造 compress + decay 路径，不需要侵入 causal model。

---

## 1. 核心创新：Attention Head 的自然功能分化

### 1.1 假设

视频扩散 transformer 的 attention head **天然分化** 为四种功能角色（无需训练、无需外部标签）：

| Head 类型 | 关注内容 | 典型特征 | 示例（rooftop→gym→rooftop） |
|---|---|---|---|
| **Layout (结构)** | 物体轮廓、空间布局 | 空间局部 + 时序稳定 | 屋顶护栏的形状、建筑结构 |
| **Texture (纹理)** | 颜色、材质、细节 | 空间局部 + 时序变化 | 蓝天/健身房蓝色垫子的颜色 |
| **Motion (运动)** | 时序动态、光流 | 空间全局 + 时序稳定 | 人物的跑酷动作流 |
| **Dynamic (动态)** | 场景切换、变化 | 空间全局 + 时序变化 | scene transition 的 token 重分配 |

### 1.2 关键洞察

传统方法（CEMR+CEG, Echo-Forcing native）对所有 head 使用相同的 cache 策略 → "一刀切"：
- 要么保留太多（Layout 头需要旧信息，但 Texture 头也被保留了 → B 场景被 A 场景的纹理污染）
- 要么保留太少（Texture 头需要刷新，但 Layout 头被清除了 → A2 无法回归 A 的外观）

**Head-role-aware 策略**：Layout 头保留、Texture 头刷新、Motion 头部分保留。

### 1.3 为什么这个假设是合理的

1. **Transformer head specialization 已被广泛验证**：BERT（Clark et al. 2019）、GPT（Elhage et al. 2021）、ViT（Darcet et al. 2023）都确认 attention head 有功能分化。

2. **视频扩散 transformer 的 spatial-temporal 结构**：同一个 30×52 spatial grid 上，某些 head 天然关注局部邻域（texture），某些关注全局（motion），这是 3D RoPE + causal attention 的自然结果。

3. **PF 的 head role labels（best_labels.csv）已经间接验证**：LAYOUT/MOTION/RECALL/WAVE 分类在 PF 上有效，说明 head specialization 确实存在。

---

## 2. 方法：Head-Role-Aware KV Retention

### 2.1 总体流程

```
Scene A 生成（第 1 次 inference）
    │
    ├─> 在第 k 个 denoising block 的 forward_pass 中
    │   收集 Q/K tensors（L15-L20 层，仅需 ~3 帧）
    │
    ├─> 计算：per-head Spatial Locality (SL) 和 Temporal Persistence (TP)
    │
    ├─> 分类 heads：Layout | Texture | Motion | Dynamic
    │
    └─> 存储分类结果 → head_classification_cache（Dict[layer_id, Tensor[12]])
    
Scene B 生成（transition）
    │
    ├─> KV_scene_control 使用 per-head 策略：
    │   - Layout heads: 从 scene pool 加载 A 的 KV → 缓慢 decay
    │   - Texture heads: 不加载旧 KV → 快速 decay → 仅滑动窗口
    │   - Motion heads: 保留最近 N 帧 → 中等 decay
    │   - Dynamic heads: native Echo-Forcing 行为
    
Scene A2 生成（return）
    │
    ├─> Layout heads 的缓慢 decay 使 A1 的 structure 仍有残留
    │    → 拉回 A 的外观但不污染 B
    │
    └─> Texture heads 的快速 refresh 使 B 的纹理被洗掉
```

### 2.2 Head 分类算法

#### 输入

对于每层 l ∈ [L_classify_start, L_classify_end]（默认 15-20）：

- `Q_tensors`: list of [B, num_frames, frame_seqlen, num_heads, head_dim]（来自 forward pass）
- `K_tensors`: same shape

在 **scene A** 的 `_context_forward` 中（timestep-0 的 forward_pass）收集。

#### 2.2.1 Spatial Locality Score (SL)

对于每层 l 的每个 head h：

```
SL_{l,h} = mean_{token i, frame f} [ 
    mean_{j in spatial_neighbors(i, 30x52 grid)} [cos_sim(Q_i^{f}, K_j^{f})] 
]
```

spatial_neighbors(i, 30×52_grid) 取 i 的 4-邻域（上下左右）。

**直觉**：如果 head 主要关注空间邻域 → SL 高 → texture/layout 特征。如果 head 关注远距离 token → SL 低 → motion/global 特征。

#### 2.2.2 Temporal Persistence Score (TP)

对于每层 l 的每个 head h：

```
TP_{l,h} = mean_{token i, frame f in [0, N-1]} [cos_sim(K_i^{f}, K_i^{f+1})]
```

帧 f 和 f+1 取同一 spatial position 的 K embedding。

**直觉**：如果 head 的 K 在时间上变化小 → TP 高 → layout/motion（稳定特征）。如果 K 频繁变化 → TP 低 → texture/dynamic（细节变化）。

#### 2.2.3 分类决策

四象限映射：

```
                    TP 高 (temporally stable)
                         │
          MOTION         │        LAYOUT
     (global + stable)   │   (local + stable)
                         │
SL 低 ──────────────────┼────────────────── SL 高
                         │
          DYNAMIC        │        TEXTURE
    (global + variable)  │   (local + variable)
                         │
                    TP 低 (temporally variable)
```

实现：对每层的 12 个 head，用 SL 和 TP 做 k-means (k=4)，然后根据 centroid 位置映射标签。

或者更简单：对 SL 和 TP 各取 median 作为分界线，形成 2×2 四象限。

### 2.3 Per-Head Retention 策略

每个 head 类型在 scene transition 时有不同的 KV cache 行为：

| 参数 | Layout | Texture | Motion | Dynamic |
|---|---|---|---|---|
| **Scene pool recall** | **YES**（加载旧场景 layout KV） | NO（全新生成） | NO | NO |
| **Sliding window** | full（21 帧） | short（3 帧） | medium（7 帧） | full（native） |
| **Decay rate** (per block) | slow（γ=0.95） | fast（γ=0.5） | medium（γ=0.8） | native（Echo-Forcing 默认） |
| **Old memory weight** | 0.3（30% 旧 layout） | 0.0 | 0.0 | 0.0 |

#### 2.3.1 Scene pool recall（Layout 头）

在 `KV_scene_control` 中，layout head 的 K/V 从 scene pool 中加载最匹配的旧场景：

```python
# 对 layout heads: 加载旧场景的 compressed KV
for layer_id in active_layers:
    layout_mask = head_classification[layer_id] == LAYOUT  # [12] bool tensor
    old_k = scene_pool.get_best_match().compressed_k[layer_id]  # [tokens, 12, 128]
    old_v = scene_pool.get_best_match().compressed_v[layer_id]
    # 仅 layout heads 加载
    cache["k"][:, :old_tokens, layout_mask] = old_k[:, layout_mask]
    cache["v"][:, :old_tokens, layout_mask] = old_v[:, layout_mask]
```

#### 2.3.2 滑动窗口 + Decay（Texture/Motion 头）

扩展 `local_token_weights` 从 `[kv_cache_size]` 到 `[kv_cache_size, num_heads]`：

```python
# Per-head decay rates: [num_heads]
# Layout=0.95, Texture=0.5, Motion=0.8, Dynamic=native
per_head_gamma = torch.zeros(12, device=k.device)
per_head_gamma[layout_mask] = 0.95
per_head_gamma[texture_mask] = 0.50
per_head_gamma[motion_mask] = 0.80
per_head_gamma[dynamic_mask] = 1.00  # no decay

# Apply per-head decay
weights = cache["local_token_weights"]  # [kv_cache_size, 12]
weights[decay_mask] *= per_head_gamma  # broadcast: [N_tokens, 12] * [12]
```

#### 2.3.3 Sliding window truncation

在 attention 读取 KV cache 时，按 head 类型截断：

```python
# Per-head window sizes: [12]
per_head_window = torch.tensor([...], device=k.device)  # in tokens
# Layout: full 32760, Texture: 3*1560=4680, Motion: 7*1560=10920, Dynamic: full

# CausalWanSelfAttention.forward 中：
valid_tokens = torch.arange(kv_cache_size, device=k.device) >= start_idx
valid_tokens = valid_tokens[:, None] & (torch.arange(kv_cache_size)[None] < per_head_window[None])  # [kv_cache_size, 12]
```

### 2.4 简化实现路径

为减少侵入性，**Phase 1 只实现 decay + window truncation**，不修改 scene pool recall。decay 和 window 两类 head 即可验证核心假设：

- **Layout/Structure heads**：长 window + 慢 decay → 保留跨场景一致性
- **Detail/Refresh heads**（Texture+Dynamic）：短 window + 快 decay → 场景切换时快速刷新

scene pool recall 留到 Phase 2。

---

## 3. 实现计划

### 3.1 文件变更

| 文件 | 改动 | 行数 |
|---|---|---|
| `src/lifecycle_kv/head_classifier.py` | **新建**：HeadClassifier 类，SL/TP 计算，k-means 分类 | ~80 |
| `third_party/Echo-Forcing/pipeline/causal_inference.py` | 集成：添加 head 收集、分类调用、per-head decay/window | ~120 |
| `scripts/run_ef_head_role.sh` | **新建**：实验 wrapper | ~100 |
| `docs/57_head_role_aware_memory_spec.md` | **本文** | — |

### 3.2 head_classifier.py API

```python
class HeadClassifier:
    def __init__(self, num_heads=12, num_layers=30, classify_layers=(15, 21)):
        self.classifications = {layer: torch.zeros(num_heads, dtype=torch.long) 
                                for layer in range(*classify_layers)}
    
    def collect_qk(self, layer_id: int, q: Tensor, k: Tensor):
        """收集一次 forward pass 的 Q/K [B, frames, seqlen, num_heads, dim]"""
        
    def compute_sl(self, grid_h=30, grid_w=52) -> Dict[int, Tensor]:
        """计算 Spatial Locality Score per-head, per-layer → {layer_id: [12]}"""
        
    def compute_tp(self) -> Dict[int, Tensor]:
        """计算 Temporal Persistence Score per-head, per-layer → {layer_id: [12]}"""
        
    def classify(self, method="kmeans") -> Dict[int, Tensor]:
        """分类: 0=Layout, 1=Texture, 2=Motion, 3=Dynamic → {layer_id: [12]}"""
```

### 3.3 Echo-Forcing 集成点

修改 `causal_inference.py`:

```python
class CausalInferencePipeline:
    def __init__(self, ...):
        # ... 现有代码 ...
        self.head_classifier = None
        self.head_role_enable = os.environ.get("HEAD_ROLE_ENABLE", "0") == "1"
    
    def _init_head_classifier(self):
        from lifecycle_kv.head_classifier import HeadClassifier
        self.head_classifier = HeadClassifier(
            classify_layers=(
                int(os.environ.get("HEAD_ROLE_LAYER_START", "15")),
                int(os.environ.get("HEAD_ROLE_LAYER_END", "21"))
            )
        )
    
    def _context_forward(self, ...):  # line ~1627
        # 在 A 场景首次 context forward 时收集 Q/K
        if self.head_classifier and not self.head_classified:
            # hook into the generator's forward to collect Q/K
            # 或直接在 _context_forward 返回时捕获
            pass
    
    def _apply_decay(self, ...):  # line ~820
        # 替换 uniform decay 为 per-head decay
        if self.head_role_enable and self.head_classified:
            per_head_weights = self.head_classifier.get_decay_weights()
            # ... per-head decay logic ...
```

### 3.4 环境变量

```bash
HEAD_ROLE_ENABLE=1                          # Master switch
HEAD_ROLE_LAYER_START=15                    # 分类层起始
HEAD_ROLE_LAYER_END=21                      # 分类层结束
HEAD_ROLE_LAYOUT_DECAY=0.95                 # Layout 头 decay 率
HEAD_ROLE_TEXTURE_DECAY=0.50               # Texture 头 decay 率
HEAD_ROLE_MOTION_DECAY=0.80                # Motion 头 decay 率
HEAD_ROLE_LAYOUT_WINDOW=21                 # Layout 头 window（frame 数）
HEAD_ROLE_TEXTURE_WINDOW=3                 # Texture 头 window
HEAD_ROLE_MOTION_WINDOW=7                  # Motion 头 window
```

---

## 4. 实验矩阵

### 4.1 Phase 1: Smoke Test + Baseline（6 runs）

| Cell | Backend | Head Role | Scene Mode | Prompt | Frames | Seed |
|---|---|---|---|---|---|---|
| 0a | Echo-Forcing native | off | smooth | P1: rooftop\|\|gym\|\|rooftop | 120 | 0 |
| 0b | Echo-Forcing native | off | hardcut | P1 | 120 | 0 |
| 0c | Echo-Forcing native | off | recall | P1 | 120 | 0 |
| 1a | Echo-Forcing + HeadRole | fixed split | smooth | P1 | 120 | 0 |
| 1b | Echo-Forcing + HeadRole | fixed split | hardcut | P1 | 120 | 0 |
| 1c | Echo-Forcing + HeadRole | statistical | smooth | P1 | 120 | 0 |

**Cell 0a-c**：Echo-Forcing 三种模式在 A-B-A 上的 baseline（验证 EF 能否处理 episodic return）。

**Cell 1a-b**：固定 head split（heads [0:4]=layout, [4:8]=texture, [8:12]=motion）确认 per-head 策略至少不 bad。

**Cell 1c**：统计分类是否比固定 split 更好。

### 4.2 Phase 2: 分类质量审计

- 可视化 per-layer head classification（SL×TP scatter plot）
- 检查跨 layer 的一致性（layout heads 是否集中在相同 layer）

### 4.3 Phase 3: Per-Head Decay Sweep（可选）

| Cell | Layout γ | Texture γ | Motion γ |
|---|---|---|---|
| 3a | 0.90 | 0.30 | 0.70 |
| 3b | 0.95 | 0.50 | 0.80 |
| 3c | 0.98 | 0.70 | 0.90 |

### 4.4 Phase 4: 多 Prompt 验证（如果 Phase 1 通过）

用 MovieGenVideoBench 32 prompts，在 120f 上对比：
- Echo-Forcing native（hardcut/smooth/recall）
- Echo-Forcing + HeadRole（best from Phase 1 decay sweep）

VBench 6 维度 + 人工视角验证。

---

## 5. 预期结果与故事线

### 5.1 预期发现

1. **Head specialization 在视频 DiT 中存在**：不同层的 head 有明显的 SL/TP 分化
2. **Layout head 的 slow decay 改善 episodic return**：A2 段更好地回到 A 的外观
3. **Texture head 的 fast decay 减少 B 污染**：B 场景更干净（不被 A 纹理污染）
4. **Motion head 的中等 window 保持运动连续性**：无运动闪烁

### 5.2 论文故事线

**标题候选**：*Training-Free Head-Role-Aware KV Retention for Long-Horizon Autoregressive Video Generation*

1. **Motivation**: 长视频生成的两个矛盾需求 → preserving episodic consistency (layout recall) vs refreshing scene identity (texture refresh)
2. **Observation**: Attention heads naturally specialize → some focus on structure, others on texture/motion
3. **Method**: Simple online Q/K statistics → 2D classification (SL × TP) → per-head KV retention
4. **Experiments**: Echo-Forcing + HeadRole on Wan2.1-T2V → A-B-A return improvement + VBench quality

### 5.3 与 CEMR+CEG 的区别（如何避免重复）

| 维度 | CEMR+CEG | Head-Role-Aware |
|---|---|---|
| **粒度** | Episode-level (coarse frame selection) | Per-head token retention (fine-grained) |
| **标签** | PF external head labels | Online Q/K statistics (universal) |
| **Memory** | Episodic archive (new data structure) | Echo-Forcing scene pool (reuse) |
| **门控** | Contrastive episode gate (CEG) | Per-head decay rates (simple scalar) |
| **复杂度** | 500+ lines, 4 files | ~100 lines, 1 file |

---

## 6. 风险与 Fallback

### 6.1 Head 分类失败（无明显的 SL/TP 分化）

**场景**：12 个 head 的 SL/TP 分布几乎均匀，无法区分角色。

**Fallback**：
1. 回退到固定比例分组（Cell 1a/1b）
2. 如果固定分组也无效果 → head specialization 假设在 Wan DiT 上不成立
3. 回到 CEMR+CEG 的 SF validation 或直接用 PF results 投稿

### 6.2 Per-head decay 破坏生成质量

**场景**：per-head decay 导致 attention 分布异常，产生漂白/模糊/伪影。

**Fallback**：
1. 调低 decay strength（γ 更接近 1.0）
2. 只在特定层（如 L15-L21）应用，其余层保持 native
3. 仅应用 window truncation（不 decay），证明 window 至少正确

### 6.3 Echo-Forcing 的 scene transition 在 A-B-A 上不 work

**场景**：Echo-Forcing 的 smooth/hardcut 无法在 P1 (rooftop→gym→rooftop) 上形成有效的 B 场景。

**Fallback**：
1. 尝试更强的 hardcut 参数（更大的 rope_jump）
2. 使用 Echo-Forcing 的 recall 模式手动绑定 A→B→A token 序列
3. 回退到 PF backend（已有 A-B-A work 的证据）

---

## 7. 执行检查清单

- [ ] `src/lifecycle_kv/head_classifier.py` 实现完成
- [ ] Q/K 收集 hook 在 Echo-Forcing pipeline 中可用
- [ ] SL/TP 计算验证（unit test）
- [ ] k-means 分类可视化（scatter plot）
- [ ] `_apply_decay` per-head 版本
- [ ] `local_token_weights` 扩展为 [kv_cache_size, num_heads]
- [ ] Per-head window truncation 在 CausalWanSelfAttention 中
- [ ] Cell 0a-c baseline runs
- [ ] Cell 1a-c head role runs
- [ ] VBench evaluation
- [ ] 论文 outline
