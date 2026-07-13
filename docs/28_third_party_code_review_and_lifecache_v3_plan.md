# Third-party 代码审查与 LifeCache-v3 推进方案

> 更新时间：2026-07-13  
> 目标：基于仓库 `third_party/` 中已经 vendored 的相关工作源码，重新审视 LifeCache 当前实现为何没有获得提升，并给出可以直接落地的代码级改造路线。  
> 核心结论：**当前实验尚不足以否定“历史 KV 召回”本身；更可能的问题是当前 LifeCache 使用了缺失准确时空坐标的 sparse-token memory，并且没有真正实现结构化记忆、per-head 路由和严格的固定预算对照。**

---

## 1. 本次代码审查范围

本次重点阅读了以下第三方实现：

```text
third_party/RollingForcing/
third_party/DeepForcing/
third_party/Pyramid-Forcing/
third_party/MemRoPE/
third_party/LongLive-RAG/
third_party/Echo-Forcing/
third_party/IAMFlow/
third_party/Causal-Forcing/
third_party/Self-Forcing/
```

同时参考了仓库 README 中声明的 Forcing-KV 上游源码：

```text
zju-jiyicheng/Forcing-KV
```

说明：当前 GitHub connector 未能解析仓库内 `third_party/Forcing-KV/wan/modules/causal_model_forcingkv.py` 的本地路径，因此 Forcing-KV 部分使用 README 中对应的 canonical upstream 实现进行代码核对。后续本地开发时应再次确认 vendored 版本与 upstream commit 是否一致。

当前 `third_party/` inventory 中没有明确列出 WorldKV 和 OmniMem 源码目录，因此本文不会把它们当作已完成的本地代码审查对象。若后续将其加入 `third_party/`，应补充独立审查记录。

---

## 2. Executive conclusion

当前 LifeCache-v2 的主要问题不是“bank 没有写入”或“recall 没有发生”。这些问题已经基本修复。真正的问题更接近：

```text
1. 被淘汰 token 的绝对 frame / spatial position 没有被正确保存；
2. arbitrary sparse tokens 被 padding 成伪造的完整 frame grid 后重新 RoPE；
3. RecallResult 没有完整传播 temporal/spatial metadata；
4. current query、recent K、recalled K 的诊断空间并不一致；
5. region bias 实际没有送入 attention logits；
6. head-aware routing 仍是“按 layer 多数投票”，不是真正 per-head；
7. 当前 recall 是额外 append，而不是固定总预算下重新分配；
8. 当前 A-B-A 评估不是严格的 prompt schedule，且主要依靠 MP4 文件大小判断。
```

而 `third_party/` 中已经存在几乎完整的解决参考：

```text
Echo-Forcing:
  abs_frame_idx / spatial_idx sidecar
  raw K
  synchronized roll / drop
  scene-aware selection and decay

MemRoPE:
  compressed_temporal_indices
  compressed_spatial_indices
  vectorized token-wise 3D RoPE
  sink + compressed + recent fixed layout

LongLive-RAG:
  complete-frame CPU offload
  clean latent descriptor
  recent-window exclusion
  fixed total attention budget

Pyramid-Forcing:
  30 × 12 head labels
  cyclic / stride / merge policies
  true head-dependent cache composition

Forcing-KV:
  spatial / temporal head split
  separate K/V views and separate attention
  structured patch-chunk compression

IAMFlow:
  full-frame KV archive
  entity-aware / visual-aware frame selection
  dynamic memory allocation
```

因此下一阶段不应继续在现有 sparse-token path 上微调 `topk / beta / layer count`，而应进入 **LifeCache-v3：Structured Archival Memory + Head-aware Retention**。

---

## 3. 当前 LifeCache 实现中的关键缺陷

### 3.1 被淘汰 token 的 frame position 是 cache slot，不是绝对历史 frame

当前 Self-Forcing 接入在 eviction 时近似使用：

```python
token_indices = torch.arange(
    sink_tokens,
    sink_tokens + num_evicted_tokens,
)
frame_positions = token_indices // frame_seqlen
```

这里得到的是当前 KV cache 内部槽位对应的局部 frame，而不是这些 token 实际生成时的全局 frame。

例如：

```text
生成 frame 0 时写入槽位 0
生成 frame 24 时，旧 frame 被滚出，但它仍可能被记录为 frame_position=0
生成 frame 48 时，下一批被滚出的 token 仍可能被记录为 frame_position=0
```

结果是 bank 中多个不同时刻的 token 共享错误时间标签。之后所谓：

```python
rel = current_frame - old_frame
```

并没有使用真实 `old_frame`。

### 3.2 缺失 spatial position

Wan 的 RoPE 是三轴的：

```text
temporal + height + width
```

当前 TokenSet 只保存粗略的 `frame_positions`，没有保存：

```text
spatial_idx
h_idx
w_idx
```

因此 arbitrary top-k sparse token 被召回时，无法恢复其原始空间位置。

### 3.3 Sparse token padding 会伪造空间坐标

当前代码将 recalled sparse tokens 顺序写入：

```python
rk_padded[:n] = active_k[idx]
```

然后调用针对完整 `[F,H,W]` 网格的 RoPE 函数。

这意味着原本来自不同位置的 token：

```text
(frame=10, h=18, w=7)
(frame=24, h=2,  w=31)
(frame=53, h=9,  w=4)
```

可能被重新解释为：

```text
(frame=0, h=0, w=0)
(frame=0, h=0, w=1)
(frame=0, h=0, w=2)
```

所以当前实验不是严格意义上的 RoPE-safe sparse recall。

### 3.4 Recall metadata 在中间层丢失

当前 `RecallResult` 没有完整携带：

```text
frame_positions
spatial_positions
rope_mode
source token absolute positions
```

`ActiveCacheComposer` 创建 `recall:view` 时也没有为它恢复这些字段，缺失位置最后会被填成 `-1`。

因此 attention 层即使看见 `CacheRegion.RECALL`，也不知道每个 recalled token 的真实来源位置。

### 3.5 当前 QK ratio 不能代表真实 attention mass

当前诊断使用 cosine Q-K proxy，并且：

```text
q: pre-RoPE query
recalled K: pre-RoPE raw K
recent K: 普通路径中的 post-RoPE cache K
```

这不是同一个表示空间，也不是模型真实执行的：

```text
scaled dot product + positional rotation + bias + softmax
```

所以 `recall/recent QK ratio = 3.5x` 不能严格证明模型真实 attention 对 recall 分配了 3.5 倍权重。

### 3.6 region bias 尚未真正生效

`ActiveCacheView` 虽然构造了 `region_bias`，但 LifeCache 最终仍直接调用：

```python
attention(roped_query, active_k, active_v)
```

没有把 `view.region_bias` 加入 logits。因此此前：

```text
region_bias_beta = 0.00 vs 0.05
```

不能被视为有效 ablation。

### 3.7 当前不是严格的 head-aware memory

目前逻辑是：统计一个 layer 中 motion/wave heads 是否超过一半，再决定整个 layer 是否使用 recall。

这与 Pyramid-Forcing / Forcing-KV 的 per-head cache 完全不同：

```text
当前 LifeCache:
  one layer -> one active K/V view -> all heads shared

真正 head-aware:
  spatial/layout heads -> structured historical memory
  temporal/motion heads -> recent/dynamic memory
  generic heads -> native recent
```

### 3.8 random recall ablation 也没有完全保持 position consistency

当前 random recall 将 recall region 的 K/V 替换成随机 bank token，但没有同步替换对应的：

```text
frame_positions
spatial_positions
source_set_ids
rope metadata
```

因此 random 和 QK recall 的差异可能被错误位置映射掩盖。

### 3.9 Timestep filter 仍然是硬编码实验逻辑

Pipeline 中当前使用固定阈值：

```python
if capture_ts > 650:
    continue
```

这不是通用 capture policy，也不能确保选中的就是正确 clean/stable memory。应改为显式配置：

```yaml
capture_timestep_mode: last_step
capture_timestep_values: [625]
```

并将 capture policy 从 pipeline 逻辑中解耦。

---

## 4. RollingForcing：保留结构化 anchor，而不是 arbitrary sparse token

### 4.1 代码机制

RollingForcing 的核心 cache 结构是：

```text
[first block anchor] + [rolling recent window]
```

第一块 K 以 raw/unrotated 形式保存，后续 attention 时再将 anchor 重新 RoPE 到当前 working window 前面的合法位置。

生成时构造：

```text
anchor_cache_key
working_cache_key
current roped_key
```

然后拼接：

```text
[anchor | working | current]
```

### 4.2 对 LifeCache 的启示

RollingForcing 没有把第一块拆成 arbitrary top-k sparse token，而是保留完整 block，因此：

```text
空间结构保持
frame boundary 清楚
RoPE grid 合法
V 与 K 对齐
```

### 4.3 可以直接借鉴的代码设计

LifeCache 的第一个 oracle 应改为：

```text
完整历史 frame/block recall
```

而不是继续测试 arbitrary sparse top-k。

建议新增：

```python
@dataclass
class StructuredMemoryBlock:
    block_id: str
    layer_id: int
    k_raw: torch.Tensor           # [T,H,D]
    v: torch.Tensor               # [T,H,D]
    abs_frame_idx: torch.Tensor   # [T]
    spatial_idx: torch.Tensor     # [T]
    num_frames: int
    frame_seq_length: int
```

第一版保存一个完整 latent frame：

```text
1560 tokens per enabled layer
```

只在单层运行，控制显存。

---

## 5. DeepForcing：只修 temporal phase，保留原空间结构

### 5.1 代码机制

DeepForcing 提供 `_rope_time_delta_mul_()`，只旋转 RoPE 中的 temporal channels：

```text
time channels: apply delta rotation
height channels: unchanged
width channels: unchanged
```

它根据当前 recent tail 推导 sink 应映射到的位置，然后对 sink K 做时间轴 delta rotation。

### 5.2 对 LifeCache 的启示

如果 memory block 本身保持完整 frame 空间结构，就不需要重新生成全部 3D RoPE。可以：

```text
保留原本正确的 H/W phase
只修 temporal phase
```

这比当前“把 sparse tokens 放进假 grid，再重做完整 RoPE”安全得多。

### 5.3 推荐使用条件

```text
完整 frame / 完整 patch grid memory：
  可采用 temporal-only delta rotation

任意 sparse tokens：
  必须保存每个 token 的 t/h/w，并做 token-wise 3D RoPE
```

---

## 6. MemRoPE：当前最应该直接移植的实现

### 6.1 最关键代码

MemRoPE 已经实现：

```python
causal_rope_apply_with_spatial_indices(
    x,
    ...,
    compressed_temporal_indices,
    compressed_spatial_indices,
    ...,
)
```

它明确保存并使用：

```text
compressed_temporal_indices
compressed_spatial_indices
```

随后逐 token 计算：

```python
comp_h = comp_local // w
comp_w = comp_local % w

freq_temporal = freq_t[all_temporal]
freq_height = freq_h[all_h]
freq_width = freq_w[all_w]
```

最后拼成每个 token 自己的三轴 frequency，并向量化应用 RoPE。

### 6.2 这正好解决 LifeCache 当前问题

当前 LifeCache 需要移植的不是 MemRoPE 的全部 cache policy，而是：

```text
1. temporal/spatial sidecar
2. token-wise frequency gather
3. raw K read-time RoPE
4. block-relative / growing 两种映射策略
```

### 6.3 建议新增函数

文件：

```text
third_party/Self-Forcing/wan/modules/causal_model.py
```

新增：

```python
def causal_rope_apply_sparse_3d(
    x: torch.Tensor,            # [B,T,H,D]
    freqs: torch.Tensor,
    temporal_idx: torch.Tensor, # [B,T] or [T]
    spatial_idx: torch.Tensor,  # [B,T] or [T]
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    ...
```

实现直接参考 MemRoPE 的 frequency gather，不再 padding 到完整 grid。

### 6.4 映射策略

建议支持：

```yaml
rope_position_mode: absolute_clamp
rope_position_mode: relative_clamp
rope_position_mode: block_relative
rope_position_mode: temporal_delta_only
```

第一轮 oracle：

```text
完整 frame + temporal_delta_only
```

第二轮 sparse：

```text
token-wise 3D relative_clamp
```

---

## 7. Echo-Forcing：位置 sidecar、scene memory 和 decay 的最佳本地参考

### 7.1 Echo 已经维护完整 sidecar

Echo-Forcing 的 cache 中明确存在：

```text
k_raw
abs_frame_idx
spatial_idx
history_k
history_v
history_abs_frame_idx
history_spatial_idx
```

它在 roll/drop 操作时同步移动：

```text
K
V
raw K
absolute frame index
spatial index
local token weights
local decay mask/rate
```

这就是当前 LifeCache 缺失的基础设施。

### 7.2 应直接复用的设计原则

对所有 cache mutation，必须提供统一 helper：

```python
def roll_cache_and_metadata(...):
    roll(k)
    roll(v)
    roll(k_raw)
    roll(abs_frame_idx)
    roll(spatial_idx)
```

禁止不同 tensor 分散手动滚动，否则位置一定会逐渐失配。

### 7.3 Echo 的 scene transition 也解决评估问题

Echo pipeline 会解析 scene segments，并按 scene 切换 prompt。它不是把 A-B-A 写在一条长文本里期待模型自行执行，而是显式构造 scene boundaries。

LifeCache 应借鉴：

```text
Scene 0: prompt A, N blocks
Scene 1: prompt B, M blocks
Scene 2: prompt A, K blocks
```

这才是严格 A-B-A memory benchmark。

### 7.4 Preserve / recall / forget

Echo 使用：

```text
scene pool
prompt feature
similarity-based recall
old memory token weights
decay mask/rate
transition eviction
```

这些机制不应立即全部移植，但可以作为 LifeCache-v3 后期的 `rho` 与 scene gate 参考。

---

## 8. LongLive-RAG：完整 frame recall、clean latent descriptor 和固定预算

### 8.1 Memory unit 是完整 frame

LongLive-RAG 在 eviction 时将每个完整 frame 的 raw K/V 拆开并 offload 到 CPU：

```text
cpu_k_frames
cpu_v_frames
```

检索时重新取回完整 frame，而不是 arbitrary token。

### 8.2 Descriptor 来自 clean generated latent

它在每个生成 block 完成后，从 `denoised_pred` 构造 latent descriptor：

```text
avg_pool latent
or trained latent AE
```

检索 descriptor 与 K/V payload 解耦：

```text
latent descriptor 决定取哪个 frame
完整 raw KV 负责真正 attention
```

这比当前直接使用当前 q 对历史 K 做全局 token top-k 更稳定。

### 8.3 Recent exclusion

LongLive-RAG 显式排除最近刚 evict 的若干 frame：

```text
recent_exclude
```

避免 retrieval 只是重复 local window 附近内容。

LifeCache 应新增：

```yaml
exclude_recent_history: true
min_recall_gap_frames: 21
```

### 8.4 固定总预算

LongLive-RAG 使用：

```python
local_budget = max_attention_size
             - sink_tokens
             - memory_size * frame_seqlen
```

也就是说 memory 占用多少，就从 local context 中让出多少，总 attention length 保持固定。

当前 LifeCache 是：

```text
native recent + extra recall
```

这不是公平 cache allocation 对照，也可能导致 softmax dilution。

### 8.5 LongLive-RAG 可改进的地方

其代码将 retrieved memory frame 的 temporal position统一映射为 0。LifeCache 不应直接复制这一点，应使用：

```text
完整 frame sequential slots
或 relative-clamp slots
```

例如选中 K 个历史 frame：

```text
memory positions = [0, 1, ..., K-1]
recent positions = [K, ..., local_range-1]
```

这样保留 memory frame 之间的顺序。

---

## 9. Pyramid-Forcing：不要再使用 layer majority routing

### 9.1 Head labels 的真实语义

Pyramid 的 `best_labels.csv` 是 30 × 12：

```text
-1 = oscillating
 1 = stable compact
 2 = stable sparse
```

推荐策略：

```text
-1: cyclic middle cache
 1: stride sampling
 2: merge
```

并为每类 head 分配不同 sink/middle/recent 组合。

### 9.2 LifeCache 当前实现与其差异

当前：

```text
某层 motion heads 超过一半 -> 整层不 recall
否则 -> 整层所有 heads 都 recall
```

Pyramid：

```text
同一层的 12 个 heads 可以使用 12 个不同 cache policy
```

因此不能用当前 layer-only ablation 排除 motion-head pollution。

### 9.3 推荐最小 per-head 实现

不要第一步就实现 ragged K/V kernel，可先用统一 K/V union + per-head additive mask：

```text
K union = [historical_memory | recent]

layout/stable heads:
  historical + recent

motion/oscillating heads:
  recent only
```

构造：

```python
head_bias: [1, num_heads, 1, K]
```

对于不允许访问 historical memory 的 head：

```python
head_bias[..., historical_range] = -inf
```

然后使用 SDPA 或支持 additive mask 的 attention。

### 9.4 路径加载必须 fail loudly

当前 head labels 相对路径存在解析风险。下一版必须：

```text
1. 以 repo root 为基准解析；
2. assert 文件存在；
3. assert shape == [30, 12]；
4. 启动时打印三类 head 数量；
5. trace 每层 head policy。
```

禁止静默 fallback 到空 role map。

---

## 10. Forcing-KV：真正的 spatial/temporal head split

### 10.1 代码机制

Forcing-KV 会抽取两组 heads：

```text
spatial heads
temporal heads
```

分别构造：

```text
k1/v1 = spatial sink + spatial cache + current spatial
k2/v2 = temporal sink + dynamic memory + temporal cache + current temporal
```

分别运行 attention，再 scatter 回原始 head 维度。

### 10.2 对 LifeCache 的直接启示

长期 scene/frame recall 应优先注入：

```text
layout / spatial / stable heads
```

而不是所有 heads。

motion/temporal heads需要的是：

```text
recent motion
boundary continuity
novel dynamic patches
```

不应被旧场景 frame 大量污染。

### 10.3 Structured patch compression

Forcing-KV 不做任意 token top-k，而是把 frame 分成规则 patch chunks，比较相邻 frame 的 chunk similarity，保留变化更大的 structured chunks。

LifeCache 的 sparse memory 可以从：

```text
arbitrary token top-k
```

改成：

```text
regular patch block top-k
```

例如每帧划分：

```text
2 × 2
4 × 4
8 × 8
```

每个 block 内 token 保持原空间结构和顺序。

---

## 11. IAMFlow：语义检索应在结构化 frame memory 之后加入

### 11.1 Memory unit 是 frame

IAMFlow 的 `FrameInfo` 同时维护：

```text
frame id/path
prompt id
associated entities
semantic score
visual score
pixel frame
all-layer frame KV
```

它从被淘汰 chunk 中选择一个完整 frame，并把该 frame 的 KV 存档。

### 11.2 动态 memory allocation

它使用类似 set-cover 的贪心逻辑，为当前所需 entities 选择尽量少但覆盖完整的 memory frames。

这对多角色场景很有价值：

```text
只召回覆盖当前角色/物体的 frame
```

### 11.3 当前阶段怎么借鉴

第一阶段不要接入 LLM/VLM；先实现 training-free descriptor：

```text
prompt embedding
latent avg-pool
DINO/CLIP frame embedding（可选）
```

当 full-frame oracle 验证有效后，再加入：

```text
entity registry
frame archive
semantic + visual fused score
```

---

## 12. Causal-Forcing：更合适的 clean capture hook

Causal-Forcing pipeline 显式维护：

```text
positive KV cache
negative KV cache
```

并在每个生成 block 完成后使用 timestep 0 重新运行 clean context 来更新 cache。

这说明若把 LifeCache 迁移到 Causal-Forcing：

```text
1. capture 应发生在 clean rerun 产生的 cache mutation；
2. positive / negative cache 必须明确处理；
3. 不能只 patch conditional branch 而忽略 CFG 对齐；
4. framewise 模式可提供更细粒度 scene memory。
```

相比 Self-Forcing 当前只能在 denoising eviction 抓取 K/V，Causal-Forcing 的 clean rerun 更适合构建稳定 archival memory。

但迁移 CF 不应早于结构化 memory oracle，否则只是把同一个错误 sparse recall 换到另一个 backbone。

---

## 13. LifeCache-v3 推荐架构

### 13.1 三层 memory system

```text
Tier 0 — Native Recent Window
  完整 recent K/V
  负责局部运动与短时连续性

Tier 1 — Head-aware Smart Retention
  Pyramid / Forcing-KV 风格
  针对 stable/layout heads 保留 anchor/stride/merge history
  针对 temporal heads 保留 recent/dynamic chunks

Tier 2 — Archival Structured Memory
  Echo / LongLive-RAG / IAMFlow 风格
  保存完整 frame、规则 patch block或 scene chunk
  只在明确 scene revisit / semantic match 时召回
```

### 13.2 新 memory 数据结构

建议新增：

```python
@dataclass
class StructuredMemoryBlock:
    block_id: str
    layer_id: int
    source_chunk_id: int

    k_raw: torch.Tensor
    v: torch.Tensor

    abs_frame_idx: torch.Tensor
    spatial_idx: torch.Tensor

    memory_unit: str              # full_frame / patch_grid / sparse_token
    num_frames: int
    frame_seq_length: int

    descriptor: torch.Tensor | None
    prompt_descriptor: torch.Tensor | None
    motion_descriptor: torch.Tensor | None

    entity_ids: tuple[str, ...]
    scene_id: str | None

    retention_score: float
    capture_timestep: float | None
    access_count: int
    last_used_step: int
```

### 13.3 为什么保留 TokenSet

`TokenSet` 可以继续作为 attention 前的 active view 数据结构，但 archival bank 不应只存 TokenSet。

建议分离：

```text
StructuredMemoryBlock = persistent storage
TokenSet = temporary selected view
RecallView = attention-time assembly
```

---

## 14. 核心代码改造

### 14.1 新增 position sidecar

修改：

```text
third_party/Self-Forcing/wan/modules/causal_model.py
```

在 KV cache 中增加：

```python
kv_cache["k_raw"]
kv_cache["abs_frame_idx"]
kv_cache["spatial_idx"]
```

写入新 token：

```python
abs_token = current_start + torch.arange(num_new_tokens)
abs_frame = abs_token // frame_seqlen
spatial = abs_token % frame_seqlen
```

注意：如果 `current_start` 已经是 token offset，上式成立；如果是 frame offset，需要统一转换并加断言。

所有 roll/evict 操作必须同步处理 sidecar。

### 14.2 移植 MemRoPE sparse 3D RoPE

新增：

```python
causal_rope_apply_sparse_3d(...)
```

输入必须包含真实：

```text
temporal_idx
spatial_idx
```

不允许使用缺失值 `-1`。

### 14.3 RecallResult metadata propagation

修改：

```text
src/lifecycle_kv/recall.py
```

新增：

```python
frame_positions: torch.Tensor | None
spatial_positions: torch.Tensor | None
rope_modes: list[str] | None
source_set_ids: list[str]
source_positions: torch.Tensor | None
```

所有 filter/top-k/random replacement 必须同步 index metadata。

### 14.4 ActiveCacheView

修改：

```text
src/lifecycle_kv/active_cache.py
```

新增：

```python
abs_frame_idx
spatial_idx
head_access_mask
```

并禁止自动把缺失 recall position 填成 `-1` 后继续 attention。应直接：

```python
raise RuntimeError("recalled token position metadata missing")
```

### 14.5 固定总预算 composer

新增 mode：

```yaml
active_cache_budget_mode: append
active_cache_budget_mode: replace_recent
```

推荐默认实验：

```text
replace_recent
```

满足：

```text
num_recent_after_crop + num_memory = native_max_tokens
```

### 14.6 Per-head access mask

使用 Pyramid labels：

```text
-1 oscillating: recent only
1 stable compact: memory + recent
2 stable sparse: memory + recent, lower budget
```

第一版不做 ragged kernel，构造 `[H,K]` mask 即可。

### 14.7 实际 attention mass diagnostic

在 K 完成 RoPE 后，采样少量 query/head：

```python
logits = torch.einsum("qhd,khd->hqk", q_sample, k_sample) / sqrt(d)
logits += head_bias
weights = softmax(logits, dim=-1)
```

记录：

```text
attention_mass_recent
attention_mass_memory
attention_mass_by_head_role
```

不要继续使用跨 RoPE 空间的 cosine proxy 作为最终证据。

---

## 15. 必做 Oracle 实验

### O0：Native baseline

```text
原生 Self-Forcing
```

### O1：完整 frame oracle recall

```text
保存 scene A 中一个完整 latent frame 的 raw K/V
保存真实 abs_frame/spatial position
第二次 A 开始时固定召回
不做 retrieval，不做 compression
只在 layer 29 的 stable heads 生效
```

目的：回答“完全正确的历史 KV 是否能帮助 scene revisit”。

### O2：结构化 patch grid

比较：

```text
full frame
4×4 patch grid
8×8 patch grid
```

每个 patch block 保持连续空间 token。

### O3：当前 arbitrary sparse top-k

作为旧实现对照。

### O4：budget-neutral

比较：

```text
append memory
replace recent with equal-size memory
```

### O5：head routing

比较：

```text
all heads
stable heads only
spatial heads only
```

### O6：prompt schedule

显式运行：

```text
A: frames 0–29
B: frames 30–69
A: frames 70–119
```

不要只使用一条包含“后来回到”的长 prompt。

---

## 16. 评估协议

禁止再使用 MP4 文件大小作为主要质量结论。

至少记录：

```text
1. A1–A2 scene similarity
2. DINO/CLIP frame similarity
3. subject identity similarity
4. background/layout similarity
5. object reappearance accuracy
6. temporal flicker
7. dynamic degree
8. paired human preference
```

对于厨房 prompt，人工检查项：

```text
蓝色橱柜是否恢复
红杯是否恢复
黄色外套是否保持
桌面布局是否一致
是否错误带回花园元素
```

每种方法至少：

```text
3 prompts × 3 seeds
```

---

## 17. 分阶段实施计划

### Phase 0：Correctness infrastructure

```text
[ ] 建立 k_raw / abs_frame_idx / spatial_idx sidecar
[ ] 将所有 cache roll/drop/write 统一成 helper
[ ] 添加 metadata length/value assertions
[ ] 修复 RecallResult metadata propagation
[ ] 删除 sparse padding fake-grid RoPE path
[ ] 移植 MemRoPE token-wise 3D RoPE
```

验收：

```text
每个 recalled token 都有合法 t/h/w
random recall 同步替换 metadata
position trace 中无 -1
```

### Phase 1：Full-frame oracle

```text
[ ] 添加 StructuredMemoryBlock
[ ] 保存完整 historical frame raw KV
[ ] 第二个 A 开始时 deterministic recall
[ ] 仅 layer29 + stable heads
[ ] fixed-budget replace mode
```

验收：

```text
完整 frame 能合法进入 attention
actual attention mass 可测
A-B-A 有可解释变化
```

### Phase 2：Structured compression

```text
[ ] 规则 patch-grid selection
[ ] full-frame / grid / sparse 对比
[ ] recent exclusion
[ ] clean latent descriptor
```

### Phase 3：True head-aware memory

```text
[ ] 修复 Pyramid labels loading
[ ] 构造 per-head access mask
[ ] stable/spatial heads 使用 archival memory
[ ] oscillating/temporal heads 保持 recent/dynamic memory
```

### Phase 4：Semantic gating and forgetting

```text
[ ] prompt/latent descriptor retrieval
[ ] scene gate
[ ] retention score rho
[ ] difference-aware decay
[ ] entity-aware frame selection（可选）
```

### Phase 5：Causal-Forcing integration

```text
[ ] 复用 clean timestep-0 capture
[ ] 对齐 positive/negative KV cache
[ ] framewise A-B-A benchmark
```

---

## 18. 决策门槛

### 情况 A：Full-frame oracle 有提升

结论：

```text
recall-after-loss 本身可行；当前失败来自 sparse/position/selection 实现。
```

继续：

```text
structured compression + semantic retrieval + head gating
```

### 情况 B：Full-frame recall 无提升，但 smart retention 有提升

结论：

```text
该 backbone 更依赖持续保留，而不是晚期重新注入。
```

转向：

```text
Pyramid/Forcing-KV retention + archival recall hybrid
```

### 情况 C：Full-frame recall 和 smart retention 都无提升

结论：

```text
当前 benchmark/backbone 的主要瓶颈可能不是历史 KV。
```

再考虑：

```text
Causal-Forcing
更长视频
更强 scene-revisit benchmark
world-state / latent-level memory
```

### 情况 D：Attention mass 很低

处理：

```text
head mask / region bias / retrieval gating
```

### 情况 E：Attention mass 很高但质量不提升

处理：

```text
检查 V 的有效性
检查完整结构
检查是否召回错误 scene
检查过强 historical domination
```

---

## 19. 对现有 docs/27 的修订建议

`docs/27_lifecache_final_analysis.md` 当前把：

```text
fundamental mechanism mismatch
SF bottleneck is not old-K/V loss confirmed
```

写得过于确定。

在完成以下实验前，不应视为 final conclusion：

```text
accurate t/h/w sidecar
true sparse 3D RoPE
actual attention mass
full-frame oracle
budget-neutral recall
true per-head routing
controlled prompt schedule
```

建议将其改名或标注为：

```text
Interim negative result for current sparse-token LifeCache implementation
```

更准确的结论应是：

> 当前 LifeCache-v2 的 arbitrary sparse-token recall 在现有实现和评估下没有观察到稳定提升；由于时空 metadata、sparse RoPE、head routing 和评估协议尚不充分，尚不能据此否定 structured historical KV recall。

---

## 20. 下一轮 coding agent checklist

```text
[ ] Read docs/28 before coding.
[ ] Port Echo-style abs_frame_idx/spatial_idx sidecars.
[ ] Centralize cache write/roll/drop metadata operations.
[ ] Port MemRoPE causal_rope_apply_with_spatial_indices logic.
[ ] Add causal_rope_apply_sparse_3d to Self-Forcing.
[ ] Add spatial_positions to TokenSet.
[ ] Add frame/spatial/rope metadata to RecallResult.
[ ] Ensure top-k and random recall index all metadata together.
[ ] Raise error when recalled position contains -1.
[ ] Add StructuredMemoryBlock persistent storage class.
[ ] Add full-frame archival write path.
[ ] Add deterministic full-frame oracle recall mode.
[ ] Add recent-window exclusion.
[ ] Add active_cache_budget_mode=replace_recent.
[ ] Verify total active token count equals native budget.
[ ] Fix Pyramid head-role path resolution.
[ ] Assert head label matrix shape is 30x12.
[ ] Add per-head historical access mask.
[ ] Route stable/spatial heads to memory, motion heads to recent.
[ ] Make region/head bias enter real attention logits.
[ ] Add sampled actual attention mass diagnostic.
[ ] Add explicit A-B-A prompt schedule.
[ ] Run O0–O6 oracle matrix.
[ ] Evaluate with scene/identity metrics, not file size.
[ ] Only after oracle success, add descriptor/rho/entity memory.
```

---

## 21. Final recommendation

下一步最值得做的不是继续增大 `recall_top_tokens`，也不是直接迁移到 Causal-Forcing，而是完成一个最小但严格的闭环：

```text
Echo-style accurate position sidecar
+ MemRoPE token-wise 3D RoPE
+ LongLive-RAG full-frame archival recall
+ Pyramid/Forcing-KV per-head access
+ fixed total cache budget
+ controlled A-B-A prompt schedule
```

最小目标可以概括为：

> **在 layer 29 的 stable heads 上，以固定总预算召回一个具有真实 t/h/w 坐标的完整历史 frame，并验证第二次 scene A 的恢复是否优于 native Self-Forcing。**

这个实验通过后，LifeCache 才适合继续推进 semantic retrieval、soft forgetting、entity memory 和多层扩展。若这个 oracle 仍完全无效，再讨论转向 pure smart-retention 或其他 backbone，结论才足够可信。
