# LifeCache Phase 0 正确性闭环与 Full-Frame Oracle 推进路线

> 更新时间：2026-07-14  
> 适用仓库：`Anachronism-N/training-free`  
> 当前基线：Self-Forcing + LifeCache-v3 Phase 0  
> 目标：先完成可验证的 metadata / RoPE / configuration correctness，再进入 full-frame oracle 与后续 structured memory 实验。

---

## 1. 当前判断

最近两次提交已经修复了部分关键问题：

```text
fb1b3f7  Phase 0A/0B：重写 sparse 3D RoPE，并修复 metadata prefix-slice

d341553  Phase 0A/0C：compression metadata 同步，并修复 head-role 路径
```

但当前实现仍未形成完整闭环。现在不应继续运行新的长视频 `top-k / layer / beta / random recall` sweep，也不应直接将当前负结果解释为“历史 KV recall 无效”。

当前阶段必须先跨过四个门槛：

```text
Phase 0.0：代码可以正常初始化和运行
Phase 0A：K/V 与全部 metadata 全链路严格对齐
Phase 0B：sparse 3D RoPE 数值正确
Phase 0C：配置值与真实执行行为一致
```

上述门槛通过后，再进入：

```text
Phase 1：Clean Full-Frame Oracle
```

---

## 2. 当前阻塞问题

### 2.1 LifeCache manager 可能无法初始化

`third_party/Self-Forcing/scripts/lifecache_manager.py` 当前最终执行：

```python
return cls(runtime, num_layers=num_layers, head_roles=head_roles)
```

但 head-role 路径修改后，`runtime = LifeCacheRuntime(runtime_config)` 可能已被遗漏。

必须恢复：

```python
runtime = LifeCacheRuntime(runtime_config)
return cls(runtime, num_layers=num_layers, head_roles=head_roles)
```

同时，role distribution 不应直接对 `HeadRole` 枚举排序：

```python
sorted(role_counts.items(), key=lambda item: item[0].value)
```

### 2.2 `compress_attention_participation()` 使用未定义 metadata

当前返回 TokenSet 时引用：

```python
frame_positions=selected_fp
spatial_positions=selected_sp
```

但函数没有定义这两个变量。

应统一 `compress_attention_participation()` 与 `compress_qk_proxy()` 的接口：

```python
frame_positions: torch.Tensor | None = None
spatial_positions: torch.Tensor | None = None
```

并使用同一个 `positions` 同步选择：

```python
selected_fp = optional_index_select(frame_positions, positions)
selected_sp = optional_index_select(spatial_positions, positions)
```

### 2.3 TokenSet 仍会丢失 spatial metadata

`TokenSet` 已有 `spatial_positions` 字段，但当前：

```text
__post_init__
clone_with_tokens()
to_device()
```

没有完整验证或传播它。

必须增加：

```python
if self.spatial_positions is not None:
    if self.spatial_positions.ndim != 1:
        raise ValueError("spatial_positions must be 1D")
    if self.spatial_positions.numel() != self.k.shape[0]:
        raise ValueError("spatial_positions length must match token count")
    if self.spatial_positions.min() < 0:
        raise ValueError("spatial_positions must be non-negative")
```

`clone_with_tokens()`：

```python
spatial_positions=(
    self.spatial_positions.index_select(0, token_positions)
    if self.spatial_positions is not None
    else None
)
```

`to_device()`：

```python
spatial_positions=(
    self.spatial_positions.to(device)
    if self.spatial_positions is not None
    else None
)
```

### 2.4 RecallResult 尚未传播位置 metadata

当前 `RecallResult` 没有：

```text
frame_positions
spatial_positions
rope_modes
```

因此 compression 虽然保存了正确位置，recall top-k 后仍会丢失。

建议扩展为：

```python
@dataclass
class RecallResult:
    k: torch.Tensor | None
    v: torch.Tensor | None
    token_indices: torch.Tensor | None
    token_scores: torch.Tensor | None
    frame_positions: torch.Tensor | None
    spatial_positions: torch.Tensor | None
    rope_modes: list[str]
    token_sets: list[TokenSet]
    source_set_ids: list[str]
    source_positions: torch.Tensor | None
    set_scores: torch.Tensor | None
```

以下操作必须同步索引所有字段：

```text
candidate-set filtering
min-token-score filtering
top-k token selection
random recall replacement
budget cropping
device transfer
```

### 2.5 recall:view 仍可能填入 `-1`

`ActiveCacheComposer` 创建 `recall:view` 时，应直接写入：

```python
frame_positions=recall_result.frame_positions
spatial_positions=recall_result.spatial_positions
rope_mode="pre_rope"
```

对 `CacheRegion.RECALL`，不允许缺失位置后继续 fallback：

```python
if recalled_position_missing:
    raise RuntimeError("recalled token position metadata missing")
```

Phase 0 阶段禁止继续使用：

```python
temporal_idx = zeros(...)
spatial_idx = arange(...)
```

因为这会重新伪造 sparse token 的时空位置。

---

## 3. Sparse 3D RoPE 的剩余问题

### 3.1 complex frequency 处理方向已修正

Wan 的 `freqs` 已经是 complex polar tensor。正确实现应直接：

```python
x_complex * token_freqs
```

而不是再次对 `freqs` 调用 `view_as_complex()`。

这一点已在最新提交中修正，但仍需 parity test 验证。

### 3.2 temporal mapping 仍不是相对当前 query 的 clamp

当前近似逻辑：

```python
temporal_idx = historical_frame.clamp(0, TR - 1)
```

会把绝对历史帧直接压入 `[0, TR-1]`，但 query 仍可能使用绝对位置 70、90、120。

例如：

```text
query frame      = 70
historical frame = 60
memory position  = 20
实际相对距离     = 50
```

这并没有把 query-memory 距离控制在训练窗口内。

在保留 native query/recent K 不变的模式下，建议：

```python
distance = (
    current_start_frame - historical_frame
).clamp(0, TR - 1)

mapped_memory_position = current_start_frame - distance
```

得到：

```text
current=70, historical=60 -> mapped=60
current=70, historical=30 -> mapped=50
current=70, historical=0  -> mapped=50
```

从而保证：

```text
query_position - mapped_memory_position <= TR - 1
```

### 3.3 H/W 不能硬编码

禁止继续固定：

```python
grid_h=60
grid_w=104
```

应从当前 `grid_sizes` 读取：

```python
grid_h = int(grid_sizes[0, 1])
grid_w = int(grid_sizes[0, 2])
```

### 3.4 必须增加 sparse/full parity test

这是进入任何视频实验前的硬门槛。

```python
full_roped = causal_rope_apply(
    full_raw_k,
    grid_sizes,
    freqs,
    start_frame=start_frame,
)

chosen = torch.randperm(full_raw_k.shape[1])[:256]

sparse_roped = causal_rope_apply_sparse_3d(
    full_raw_k[0, chosen],
    freqs,
    temporal_idx[chosen],
    spatial_idx[chosen],
    grid_h=H,
    grid_w=W,
    clamp_temporal=TR,
)

torch.testing.assert_close(
    sparse_roped,
    full_roped[0, chosen],
    rtol=1e-4,
    atol=1e-4,
)
```

测试覆盖：

```text
单帧完整 grid
多帧完整 grid
随机 sparse token
连续 patch block
不同 H/W
float32
bf16
batch=1
```

---

## 4. 配置真实性问题

### 4.1 recall_top_tokens 可能仍被默认 RegionBudget 覆盖

配置中的：

```yaml
recall_top_tokens: 32
```

可能被 `HeadRole.LAYOUT` 的默认：

```python
RegionBudget(recall=512)
```

覆盖。

建议只保留一个权威值：

```python
effective_recall_tokens = min(
    budget.recall,
    self.recall_config.top_tokens,
)
```

trace 必须记录：

```text
configured_recall_tokens
effective_recall_tokens
actual_recalled_tokens
```

### 4.2 max_frame_distance 尚未贯穿调用链

需要完整传递：

```text
causal_model.current_start_frame
-> runtime.compose_active_cache
-> ActiveCacheComposer.compose
-> recall_tokens
-> retrieve_token_sets
```

否则 near-only / far-only 实验不可信。

### 4.3 region bias 尚未进入真实 attention

在 additive bias path 未实现前：

```yaml
region_bias_beta: 0.0
```

如果用户设置非零，应直接警告或报错：

```text
region_bias_beta > 0, but LifeCache attention bias path is disabled
```

### 4.4 head-role 加载需要 fail-fast

仅打印 warning 不够。Phase 0 / oracle 模式下应要求：

```python
assert len(head_roles) == 30 * 12
```

并打印：

```text
loaded_head_roles=360
layout=...
wave=...
generic=...
```

---

## 5. 建议的提交拆分

### Commit 1：Restore runnable state

修改：

```text
third_party/Self-Forcing/scripts/lifecache_manager.py
src/lifecycle_kv/compression.py
```

任务：

```text
[ ] 恢复 runtime 初始化
[ ] 修复 HeadRole 排序
[ ] 修复 compress_attention_participation 未定义 metadata
[ ] 增加 from_env smoke test
[ ] 增加 compile/import test
```

验收：

```text
LIFECACHE_ENABLE=1 可成功初始化
head roles loaded=360
无 NameError
无未定义变量
```

### Commit 2：Metadata correctness

修改：

```text
src/lifecycle_kv/tokenset.py
src/lifecycle_kv/compression.py
src/lifecycle_kv/recall.py
src/lifecycle_kv/active_cache.py
src/lifecycle_kv/runtime.py
```

任务：

```text
[ ] TokenSet 验证 spatial_positions
[ ] clone_with_tokens 传播 spatial_positions
[ ] to_device 传播 spatial_positions
[ ] RecallResult 增加 frame/spatial/rope metadata
[ ] recall filter/top-k 同步索引所有 metadata
[ ] recall:view 写入真实 metadata
[ ] random recall 同步替换所有 metadata
[ ] recall region 缺失位置时 fail-fast
```

验收不变量：

```python
assert len(k) == len(v)
assert len(k) == len(token_indices)
assert len(k) == len(frame_positions)
assert len(k) == len(spatial_positions)
assert frame_positions.min() >= 0
assert spatial_positions.min() >= 0
assert spatial_positions.max() < grid_h * grid_w
```

### Commit 3：RoPE correctness

修改：

```text
third_party/Self-Forcing/wan/modules/causal_model.py
tests/test_sparse_3d_rope.py
```

任务：

```text
[ ] relative-to-current temporal mapping
[ ] dynamic grid_h/grid_w
[ ] 删除 synthetic fallback
[ ] sparse/full parity test
[ ] 多帧、随机 token、patch block 测试
[ ] trace mapped distance
```

验收：

```text
invalid position count = 0
max mapped relative distance <= TR - 1
sparse/full parity test passes
```

### Commit 4：Configuration truthfulness

任务：

```text
[ ] recall_top_tokens 真正控制数量
[ ] max_frame_distance 真正过滤候选
[ ] head-role count fail-fast
[ ] region bias 未实现时禁止非零配置
[ ] capture timestep policy 配置化
[ ] trace 最终 effective config
```

trace 至少包含：

```text
effective_recall_budget
effective_max_frame_distance
loaded_head_roles
enabled_layers
actual_active_tokens
actual_recalled_tokens
```

---

## 6. Phase 1：Clean Full-Frame Oracle

上述四个提交完成后，再实现 full-frame oracle。

### 6.1 为什么先做 full-frame oracle

它绕开：

```text
noisy eviction capture
QK compression
semantic retrieval
random top-k
sparse structure loss
错误 candidate ranking
```

直接回答：

> 给 Self-Forcing 一个完整、干净、有真实 t/h/w 坐标的历史 frame，它是否能帮助 scene revisit？

### 6.2 memory capture

在 clean-context refresh 后直接保存：

```python
start = local_end_index - current_num_frames * frame_seq_length
end = local_end_index

block = StructuredMemoryBlock(
    layer_id=layer_id,
    k_raw=kv_cache["k_pre_rope"][:, start:end].clone(),
    v=kv_cache["v"][:, start:end].clone(),
    abs_frame_idx=...,
    spatial_idx=...,
    num_frames=current_num_frames,
    frame_seq_length=frame_seq_length,
)
```

不要从 noisy eviction payload 重建 oracle memory。

### 6.3 建议配置

```text
Backbone: Self-Forcing
Layer: 29 only
Memory: one complete latent frame
Memory tokens: 1560
K: raw/pre-RoPE
V: aligned full-frame V
Compression: none
Retrieval: deterministic
Anchor: disabled
Motion memory: disabled
Region bias: 0
Seeds: 0, 1, 2
```

### 6.4 Prompt schedule

不要再使用一条自然语言长 prompt 模糊描述“后来返回”。

显式切换 condition：

```text
A：frames 0-29
B：frames 30-69
A：frames 70-119
```

第二个 A 开始时触发 deterministic historical-frame injection。

### 6.5 Oracle 实验矩阵

| 实验 | Memory | Budget | Head access |
|---|---|---|---|
| O0 | 无 | native | all |
| O1 | full frame | append | layer 29 all heads |
| O2 | full frame | replace recent | layer 29 all heads |
| O3 | full frame | replace recent | stable heads only |
| O4 | structured patch grid | replace recent | stable heads only |
| O5 | current sparse top-k | replace recent | stable heads only |

O2 固定总预算：

```text
20 recent frames + 1 historical frame
= 21-frame native attention budget
```

如果 online-local RoPE 同时改变 query/recent positions，必须增加：

```text
O0b：online-local RoPE without memory
```

用于分离 memory 收益与 RoPE 策略收益。

---

## 7. 必须记录的诊断

### 7.1 Metadata trace

```text
recalled_token_count
frame_position_min/max
spatial_position_min/max
unique_source_frames
invalid_frame_position_count
invalid_spatial_position_count
```

要求：

```text
invalid_frame_position_count = 0
invalid_spatial_position_count = 0
```

### 7.2 实际 attention mass

不要继续以 pre-RoPE cosine proxy 作为最终证据。

在真实 RoPE 和真实 active K/V 完成后，采样少量 query/head：

```python
logits = torch.einsum("qhd,khd->hqk", q_sample, k_sample) / math.sqrt(d)
weights = torch.softmax(logits, dim=-1)

memory_mass = weights[..., memory_mask].sum(dim=-1)
recent_mass = weights[..., recent_mask].sum(dim=-1)
```

记录：

```text
attention_mass_memory
attention_mass_recent
attention_mass_by_head_role
```

### 7.3 输出质量

禁止再以 MP4 文件大小作为主要质量结论。

至少记录：

```text
A1-A2 DINO/CLIP similarity
subject identity similarity
background/layout similarity
object reappearance accuracy
temporal flicker
dynamic degree
paired human preference
```

每个设置至少：

```text
3 prompts x 3 seeds
```

---

## 8. 实验前硬性验收门槛

只有全部满足后，才运行新的长视频实验：

```text
[ ] LifeCache manager 正常初始化
[ ] head roles loaded == 360
[ ] compression metadata 长度一致
[ ] TokenSet clone/device transfer 不丢 spatial metadata
[ ] RecallResult 携带真实 frame/spatial positions
[ ] recall:view 不存在 -1 position
[ ] sparse/full RoPE parity test 通过
[ ] mapped relative distance <= TR-1
[ ] effective recall budget 等于配置
[ ] max_frame_distance 确实过滤候选
[ ] region bias 未伪装为有效实验
[ ] LifeCache disabled 与 native 输出一致
[ ] recall budget=0 与 native 输出一致
```

---

## 9. 决策门槛

### 情况 A：Full-frame oracle 有提升

结论：

```text
历史 KV recall 本身可行；此前失败主要来自 sparse structure、position、selection 或 routing。
```

继续：

```text
structured patch compression
recent-window exclusion
semantic/latent descriptor retrieval
per-head access
scene gating
soft forgetting
```

### 情况 B：Full-frame recall 无提升，但 smart retention 有提升

结论：

```text
当前 backbone 更依赖持续保留，而不是晚期重新注入。
```

方向：

```text
Pyramid/Forcing-KV smart retention
+ archival memory for explicit scene revisit
```

### 情况 C：Full-frame recall 与 smart retention 都无提升

再考虑：

```text
Causal-Forcing
latent-level memory
world-state memory
更长、更严格的 benchmark
```

### 情况 D：Memory attention mass 很低

优先排查：

```text
head access mask
additive bias
gating
retrieval timing
```

### 情况 E：Memory attention mass 很高但质量无提升

优先排查：

```text
V 是否有效
是否召回错误 scene
是否需要完整 frame/patch structure
历史信息是否压制当前 motion
```

---

## 10. 下一轮 Coding Agent Checklist

```text
[ ] Restore runtime initialization in lifecache_manager.py.
[ ] Fix HeadRole sorting by enum value.
[ ] Fix compress_attention_participation metadata interface.
[ ] Add LifeCache manager smoke test.
[ ] Add spatial_positions validation to TokenSet.
[ ] Propagate spatial_positions in clone_with_tokens and to_device.
[ ] Extend RecallResult with frame_positions and spatial_positions.
[ ] Synchronize recall filtering/top-k across all metadata.
[ ] Propagate recall metadata into recall:view.
[ ] Remove synthetic position fallback for recalled tokens.
[ ] Synchronize random recall K/V and all metadata.
[ ] Implement relative-to-current temporal mapping.
[ ] Read grid_h/grid_w dynamically from grid_sizes.
[ ] Add sparse/full 3D RoPE parity tests.
[ ] Make recall_top_tokens control effective recall count.
[ ] Pass current_frame and max_frame_distance through the full call chain.
[ ] Disable unsupported nonzero region_bias_beta.
[ ] Fail fast when head role count != 360.
[ ] Trace all effective configuration values.
[ ] Add StructuredMemoryBlock.
[ ] Capture clean full-frame raw K/V after context refresh.
[ ] Add deterministic full-frame oracle injection.
[ ] Add append and replace_recent modes.
[ ] Add explicit A-B-A prompt schedule.
[ ] Add actual post-RoPE attention-mass diagnostic.
[ ] Evaluate scene revisit with semantic/identity metrics, not file size.
```

---

## 11. Final recommendation

下一步最合理的路径是：

```text
Restore runnable state
-> complete metadata correctness
-> prove sparse 3D RoPE parity
-> make configuration truthful
-> run clean full-frame oracle
-> only then add structured compression and semantic retrieval
```

当前最核心的正确性目标是：

> 一个 recalled token 从 capture、compression、bank、retrieval、top-k、device transfer、active composition、position remap 到 attention 的全过程中，K、V、frame position、spatial position 和 RoPE 始终对应同一个原始 token。

只有该不变量被单元测试和 trace 同时证明，后续视频结果才具备可解释性。