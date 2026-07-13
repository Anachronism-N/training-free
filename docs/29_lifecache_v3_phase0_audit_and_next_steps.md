# LifeCache-v3 Phase 0 审查与下一步推进方案

> 更新时间：2026-07-14  
> 当前基线提交：`fc199c84abb3f600c27c4394cb8638ab62896c70`  
> 当前状态：Phase 0 已开始实现 absolute frame position、spatial position 和 sparse 3D RoPE，但尚未达到可以支撑正式视频实验的正确性标准。

---

## 1. Executive summary

当前最新提交已经加入：

```text
1. 基于 global_end_index 推导 absolute frame position；
2. spatial_positions sidecar；
3. MemRoPE-style sparse 3D RoPE；
4. pipeline -> runtime -> bank -> active view 的部分 metadata 传播；
5. 删除旧的 sparse-token fake-frame padding 路径。
```

方向是正确的，但当前实现仍存在若干会直接使实验结论失效的关键问题：

```text
P0. compression top-k 后 K/V 与 frame/spatial metadata 没有同步选择；
P0. TokenSet.clone_with_tokens()/to_device() 会丢 spatial_positions；
P0. RecallResult 仍未传播 frame/spatial metadata；
P0. sparse 3D RoPE 的 complex frequency 使用和 tensor reshape 存在错误风险；
P0. 当前 temporal mapping 是 absolute clamp，不是 relative-clamp；
P0. recall metadata 使用 fp[:n]，而不是 fp[idx]；
P1. recall_top_tokens 很可能被 RegionBudget 覆盖，历史 budget ablation 可能未生效；
P1. max_frame_distance 没有真正传入 retrieval；
P1. region_bias_beta 没有进入真实 attention logits；
P1. Pyramid head-role path 很可能解析失败且静默 fallback；
P1. random recall 只替换 K/V，没有同步替换位置 metadata；
P2. 当前仍没有 full-frame oracle、固定预算 replacement 和实际 attention mass 诊断。
```

因此下一步不应继续直接生成更多长视频，而应先完成：

```text
Phase 0A：metadata alignment
Phase 0B：sparse 3D RoPE parity
Phase 0C：configuration truthfulness
Phase 1：full-frame oracle
```

---

## 2. 当前最新实现已经完成的部分

最新提交 `fc199c84` 的主要改动包括：

```text
src/lifecycle_kv/tokenset.py
  + spatial_positions

src/lifecycle_kv/active_cache.py
  + ActiveCacheView.spatial_positions

src/lifecycle_kv/runtime.py
  + on_kv_evicted(spatial_positions=...)

third_party/Self-Forcing/pipeline/causal_inference.py
  + 从 payload 读取 spatial_positions

third_party/Self-Forcing/wan/modules/causal_model.py
  + 基于 global/local end index 推导 absolute historical token index
  + frame_positions = abs_token_indices // frame_seqlen
  + spatial_positions = abs_token_indices % frame_seqlen
  + causal_rope_apply_sparse_3d(...)
```

这一步修正了旧版中最严重的两个问题：

```text
旧版：frame position 是 cache slot position；
新版：frame position 尝试使用 absolute global token position。

旧版：sparse tokens 被 padding 成伪造 full frame；
新版：尝试逐 token gather t/h/w RoPE frequency。
```

但是上述改动尚未在整个数据生命周期中闭环。

---

## 3. P0 问题一：压缩后的 K/V 与 metadata 不对齐

### 3.1 当前逻辑

`compress_qk_proxy()` 中：

```python
positions = select_topk_tokens(scores, config.topk, config.min_tokens)
selected_k = k.index_select(0, positions)
selected_v = v.index_select(0, positions)
```

返回的 `TokenSet` 只包含压缩后的：

```text
K
V
token_indices
importance_score
```

但是不包含：

```text
frame_positions
spatial_positions
```

随后 runtime 在 compression 完成后执行：

```python
token_set.frame_positions = frame_positions
token_set.spatial_positions = spatial_positions
```

这里的 `frame_positions/spatial_positions` 仍然是压缩前的完整长度。

### 3.2 实际错误

例如：

```text
num_evicted_tokens = 4680
compression_topk = 512

TokenSet.k.shape[0] = 512
TokenSet.frame_positions.shape[0] = 4680
TokenSet.spatial_positions.shape[0] = 4680
```

而且 metadata 是在 `TokenSet.__post_init__()` 之后赋值，因此构造时的长度检查无法发现。

### 3.3 必须修改

为所有 compression 接口增加 metadata 参数：

```python
def compress_qk_proxy(
    *,
    ...,
    frame_positions: torch.Tensor | None = None,
    spatial_positions: torch.Tensor | None = None,
) -> TokenSet:
    ...
```

并使用同一个 `positions`：

```python
selected_frame_positions = (
    frame_positions.index_select(0, positions)
    if frame_positions is not None
    else None
)

selected_spatial_positions = (
    spatial_positions.index_select(0, positions)
    if spatial_positions is not None
    else None
)
```

构造 TokenSet 时直接传入：

```python
return TokenSet(
    ...,
    frame_positions=selected_frame_positions,
    spatial_positions=selected_spatial_positions,
)
```

runtime 中删除：

```python
token_set.frame_positions = frame_positions
token_set.spatial_positions = spatial_positions
```

所有 metadata 必须在构造 TokenSet 时一次性确定。

### 3.4 验收断言

```python
assert token_set.num_tokens == token_set.frame_positions.numel()
assert token_set.num_tokens == token_set.spatial_positions.numel()
```

---

## 4. P0 问题二：TokenSet 生命周期仍会丢失 spatial_positions

当前 `TokenSet` 已有：

```python
spatial_positions: Optional[torch.Tensor]
```

但以下两个方法没有完整传播：

```text
clone_with_tokens()
to_device()
```

### 4.1 clone_with_tokens 修复

必须增加：

```python
spatial_positions=(
    self.spatial_positions.index_select(0, token_positions)
    if self.spatial_positions is not None
    else None
)
```

### 4.2 to_device 修复

必须增加：

```python
spatial_positions=(
    self.spatial_positions.to(device)
    if self.spatial_positions is not None
    else None
)
```

### 4.3 __post_init__ 增加检查

```python
if self.spatial_positions is not None:
    if self.spatial_positions.ndim != 1:
        raise ValueError("spatial_positions must be 1D")
    if self.spatial_positions.numel() != self.k.shape[0]:
        raise ValueError("spatial_positions length must match token count")
    if int(self.spatial_positions.min()) < 0:
        raise ValueError("spatial_positions must be non-negative")
```

对 frame_positions 也增加非负约束。

---

## 5. P0 问题三：RecallResult 仍未传播位置 metadata

当前 `RecallResult` 仍只有：

```python
k
v
token_indices
token_scores
token_sets
source_set_ids
source_positions
set_scores
```

缺少：

```python
frame_positions
spatial_positions
rope_modes
```

### 5.1 数据传播要求

`recall_tokens()` 中，构造 selected set union 时应同步拼接：

```python
all_frame_positions = torch.cat([
    s.frame_positions.to(q.device)
    for s in selected_sets
], dim=0)

all_spatial_positions = torch.cat([
    s.spatial_positions.to(q.device)
    for s in selected_sets
], dim=0)
```

如果任何 selected set 缺少 metadata：

```python
raise RuntimeError(
    "recalled TokenSet is missing frame/spatial metadata"
)
```

不能再使用 `-1` fallback。

### 5.2 min_token_score 过滤必须同步

当前过滤：

```python
scores = scores[keep_mask]
all_k = all_k[keep_mask]
all_v = all_v[keep_mask]
all_indices = all_indices[keep_mask]
```

还必须：

```python
all_frame_positions = all_frame_positions[keep_mask]
all_spatial_positions = all_spatial_positions[keep_mask]
all_source_positions = all_source_positions[keep_mask]
```

### 5.3 top-k 必须同步

```python
positions = torch.topk(...).indices
```

返回：

```python
RecallResult(
    ...,
    frame_positions=all_frame_positions.index_select(0, positions),
    spatial_positions=all_spatial_positions.index_select(0, positions),
    rope_modes=[...],
)
```

### 5.4 ActiveCacheComposer 修复

创建 `recall:view` 时必须传入：

```python
frame_positions=recall_result.frame_positions
spatial_positions=recall_result.spatial_positions
rope_mode="pre_rope"
source_positions=recall_result.source_positions
```

禁止创建一个没有 source metadata 的临时 TokenSet。

---

## 6. P0 问题四：当前 sparse 3D RoPE 需要重写并做 parity test

### 6.1 当前风险

现实现中：

```python
freq_t = freqs_split[0][t_idx]
freq_h = freqs_split[1][h_idx]
freq_w = freqs_split[2][w_idx]
freq_i = torch.cat([freq_t, freq_h, freq_w], dim=-1)
```

随后又对 `freq_i` 调用：

```python
torch.view_as_complex(...)
```

但 Wan 原有 `freqs` 已经是可以直接与 complex representation 相乘的 complex frequency tensor。

因此不能把已是 complex 表示的 frequency 再 reshape 成 real-pair 后 `view_as_complex()`。

### 6.2 推荐实现

```python
def causal_rope_apply_sparse_3d(
    x: torch.Tensor,          # [T,H,D] or [B,T,H,D]
    freqs: torch.Tensor,
    temporal_idx: torch.Tensor,
    spatial_idx: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, T, H, D = x.shape
    c = D // 2
    freq_t, freq_h, freq_w = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3],
        dim=1,
    )

    t_idx = temporal_idx.long()
    h_idx = (spatial_idx // grid_w).long()
    w_idx = (spatial_idx % grid_w).long()

    token_freqs = torch.cat(
        [
            freq_t.index_select(0, t_idx),
            freq_h.index_select(0, h_idx),
            freq_w.index_select(0, w_idx),
        ],
        dim=-1,
    )  # [T,D/2], complex

    x_complex = torch.view_as_complex(
        x.float().reshape(B, T, H, -1, 2)
    )

    out_complex = x_complex * token_freqs.view(1, T, 1, -1)
    out = torch.view_as_real(out_complex).flatten(-2)

    out = out.type_as(x)
    if squeeze_batch:
        out = out.squeeze(0)
    return out
```

具体维度应与仓库真实 `freqs.dtype/freqs.shape` 核对，不应依赖猜测。

### 6.3 必须新增 parity test

新增：

```text
tests/test_sparse_3d_rope.py
```

测试流程：

```python
full_roped = causal_rope_apply(
    full_raw_k,
    grid_sizes,
    freqs,
    start_frame=start_frame,
)

chosen_positions = torch.tensor([...])

sparse_roped = causal_rope_apply_sparse_3d(
    full_raw_k[0, chosen_positions],
    freqs,
    temporal_idx=full_temporal_idx[chosen_positions],
    spatial_idx=full_spatial_idx[chosen_positions],
    grid_h=H,
    grid_w=W,
)

expected = full_roped[0, chosen_positions]

assert torch.allclose(
    sparse_roped.float(),
    expected.float(),
    atol=1e-4,
    rtol=1e-4,
)
```

测试至少覆盖：

```text
单 token
同 frame 多 spatial token
跨 frame sparse token
batch=1
不同 frame start offset
bf16/fp32 tolerance
```

### 6.4 验收门槛

只有当 parity test 通过，才能声称 sparse 3D RoPE 正确。

---

## 7. P0 问题五：temporal mapping 不是 relative-clamp

当前代码近似执行：

```python
temporal_idx = absolute_frame_position.clamp(0, TR - 1)
```

这只是 absolute clamp。

例如：

```text
TR = 21
historical frame = 35 / 50 / 80
mapped position = 20 / 20 / 20
```

而 query 仍然在当前 absolute position，例如 70。

最终：

```text
query position = 70
memory position = 20
relative distance = 50
```

并没有把相对距离限制到训练窗口。

### 7.1 推荐模式一：native_relative_clamp

保留 native query 和 recent K 不变，只将 memory 映射到 query 附近：

```python
distance = (
    current_start_frame - historical_frame
).clamp(0, TR - 1)

mapped_memory_position = current_start_frame - distance
```

例如：

```text
current = 70
history = 60 -> distance 10 -> mapped 60
history = 30 -> distance 20 -> mapped 50
history = 0  -> distance 20 -> mapped 50
```

这样 memory-query 相对距离最大为 20。

### 7.2 推荐模式二：online_local

更严格的做法：

```text
memory raw K
recent raw K
query raw Q
```

在 attention 前统一重新编码到局部 window：

```text
old memory -> oldest legal local position
recent frames -> preserve relative spacing
query -> newest local position
```

这个模式更接近 MemRoPE/LongLive-RAG，但改动更大。

### 7.3 必须保留 baseline

配置应支持：

```yaml
memory_rope_policy: absolute_clamp      # current buggy baseline, only for regression
memory_rope_policy: native_relative_clamp
memory_rope_policy: online_local
```

默认实验使用：

```text
native_relative_clamp
```

---

## 8. P0 问题六：metadata 必须使用 idx，而不是前缀切片

当前 attention path 先找到 recall token：

```python
idx = is_recall.nonzero(as_tuple=True)[0]
```

但读取 metadata 时使用：

```python
fp[:idx.shape[0]]
sp[:idx.shape[0]]
```

这只有在 active view 永远是：

```text
[recall | recent]
```

时才成立。

一旦出现：

```text
[anchor | recall | recent]
```

就会读取 anchor metadata。

必须改成：

```python
recall_fp = fp.index_select(0, idx)
recall_sp = sp.index_select(0, idx)
```

并断言：

```python
assert recall_fp.numel() == idx.numel()
assert recall_sp.numel() == idx.numel()
```

---

## 9. P1 问题一：recall_top_tokens 配置可能未生效

当前配置：

```yaml
recall_top_tokens: 32
```

但 `ActiveCacheComposer` 实际构造 recall config 时使用：

```python
top_tokens=budget.recall
```

`HeadRole.LAYOUT` 默认：

```python
RegionBudget(recall=512)
```

因此实际可能始终 recall 512 tokens。

### 9.1 修改方式

推荐统一成唯一配置源：

```python
effective_recall_budget = self.recall_config.top_tokens
```

或者：

```python
effective_recall_budget = min(
    budget.recall,
    self.recall_config.top_tokens,
)
```

### 9.2 Trace 必须记录

```text
configured_recall_top_tokens
effective_recall_budget
actual_recalled_tokens
```

此前 32/64/256 的 ablation 在确认 effective budget 前，不应继续引用为有效结论。

---

## 10. P1 问题二：max_frame_distance 没有真正进入 retrieval

runtime 初始化 RecallConfig 时传入：

```python
max_frame_distance=config.max_frame_distance
```

但 composer 重新构造 RecallConfig 时没有复制该字段，同时也没有传 `current_frame` 给 `recall_tokens()`。

因此 near-only/far recall 的历史实验很可能没有真正按距离过滤。

### 10.1 修改链路

```text
causal_model.py current_start_frame
  -> runtime.compose_active_cache(current_frame=...)
  -> ActiveCacheComposer.compose(current_frame=...)
  -> recall_tokens(current_frame=...)
  -> retrieve_token_sets(current_frame=...)
```

### 10.2 Trace

```text
configured_max_frame_distance
effective_max_frame_distance
candidate_sets_before_distance_filter
candidate_sets_after_distance_filter
recall_distance_min
recall_distance_mean
recall_distance_max
```

---

## 11. P1 问题三：region bias 尚未进入 attention

`ActiveCacheView.region_bias` 已经构造，但最终调用仍然是：

```python
attention(roped_query, active_k, active_v)
```

没有 additive mask。

因此此前：

```text
region_bias_beta=0.0
region_bias_beta=0.05
```

不是有效 ablation。

### 11.1 当前建议

在 full-frame oracle 前：

```yaml
region_bias_beta: 0.0
```

避免配置造成误解。

### 11.2 后续接入

可以复用现有 HCP 的 SDPA 路径：

```python
qh = roped_query.permute(0, 2, 1, 3)
kh = active_k.permute(0, 2, 1, 3)
vh = active_v.permute(0, 2, 1, 3)

bias = view.region_bias.view(1, 1, 1, -1)

out = F.scaled_dot_product_attention(
    qh,
    kh,
    vh,
    attn_mask=bias,
)
```

首先验证 beta=0 与原 attention 输出一致，再做 bias sweep。

---

## 12. P1 问题四：Pyramid head roles 很可能未加载

配置中的路径：

```yaml
head_roles_path: third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
```

当前 manager 将相对路径拼到 config 所在目录：

```python
os.path.join(dirname(config_path), head_roles_path)
```

最终可能变成：

```text
configs/lifecache/third_party/Pyramid-Forcing/...
```

而不是仓库根目录：

```text
third_party/Pyramid-Forcing/...
```

文件不存在时当前逻辑静默返回空 map。

### 12.1 修改

新增 repository root resolution：

```python
repo_root = Path(__file__).resolve().parents[3]
path = (repo_root / configured_path).resolve()
```

更稳妥的是配置中显式使用 repo-root-relative path，并统一 helper。

### 12.2 强制验证

```python
expected = num_layers * num_heads
if len(head_roles) != expected:
    raise RuntimeError(
        f"expected {expected} head roles, loaded {len(head_roles)}"
    )
```

同时打印：

```text
Loaded head roles: 360
LAYOUT: ...
WAVE: ...
MOTION: ...
GENERIC: ...
UNKNOWN: ...
```

不能继续静默 fallback。

---

## 13. P1 问题五：random recall 必须同步替换 metadata

当前 random recall 只做：

```python
view.k[recall_pos] = all_k[rand_idx]
view.v[recall_pos] = all_v[rand_idx]
```

但没有替换：

```text
frame_positions
spatial_positions
source_set_ids
source_positions
rope_mode
```

结果是：随机 K/V 会使用原 QK recall token 的位置 metadata，实验不公平。

### 必须修改

构建全 bank union：

```python
all_k
all_v
all_frame_positions
all_spatial_positions
all_source_set_ids
all_source_positions
```

随机抽样后同步写入所有字段。

建议不要直接修改 `view.k/view.v`，而是生成新的 `RecallResult` 或新的 TokenSet，确保 metadata 结构不被破坏。

---

## 14. Phase 0A：metadata correctness 提交计划

### 修改文件

```text
src/lifecycle_kv/tokenset.py
src/lifecycle_kv/compression.py
src/lifecycle_kv/recall.py
src/lifecycle_kv/active_cache.py
src/lifecycle_kv/runtime.py
```

### Checklist

```text
[ ] TokenSet.__post_init__ 检查 frame/spatial metadata 长度与范围
[ ] clone_with_tokens 传播 spatial_positions
[ ] to_device 传播 spatial_positions
[ ] compression top-k 同步选择 frame/spatial positions
[ ] runtime 不再 post-hoc 覆盖 metadata
[ ] RecallResult 增加 frame_positions/spatial_positions/rope_modes
[ ] score filter/top-k 同步 index 所有 metadata
[ ] recall:view 写入真实 metadata
[ ] random recall 同步替换全部 metadata
[ ] 任何 recalled position 缺失时直接报错
```

### 验收

```text
K/V/frame/spatial/source 长度完全一致
invalid frame position = 0
invalid spatial position = 0
spatial_position < H*W
```

---

## 15. Phase 0B：RoPE correctness 提交计划

### 修改文件

```text
third_party/Self-Forcing/wan/modules/causal_model.py
tests/test_sparse_3d_rope.py
```

### Checklist

```text
[ ] 重写 causal_rope_apply_sparse_3d
[ ] 不再对已是 complex 的 freqs 重复 view_as_complex
[ ] grid_h/grid_w 从 grid_sizes 动态读取
[ ] 使用 fp.index_select(0, idx)
[ ] 删除 synthetic position fallback
[ ] 增加 native_relative_clamp
[ ] 可选增加 online_local
[ ] 添加 sparse/full parity test
[ ] 添加不同 start_frame 的 parity test
```

### 验收

```text
sparse token-wise RoPE == full-grid RoPE 后再采样相同 token
```

数值标准：

```python
atol <= 1e-4
rtol <= 1e-4
```

bf16 可根据实际数值放宽，但必须记录。

---

## 16. Phase 0C：configuration truthfulness 提交计划

### 修改文件

```text
src/lifecycle_kv/runtime.py
src/lifecycle_kv/active_cache.py
third_party/Self-Forcing/scripts/lifecache_manager.py
third_party/Self-Forcing/wan/modules/causal_model.py
configs/lifecache/*.yaml
```

### Checklist

```text
[ ] recall_top_tokens 成为真实有效预算
[ ] max_frame_distance 传入 current_frame
[ ] head_roles path 使用 repo-root resolution
[ ] head role 数量错误时直接报错
[ ] region_bias 未接入时禁止非零配置或发出强 warning
[ ] capture_clean_only 改为符合事实的 capture policy
[ ] trace 记录所有 effective config
```

### Trace 字段

```text
configured_recall_top_tokens
effective_recall_budget
configured_max_frame_distance
effective_max_frame_distance
loaded_head_roles
enabled_layers
actual_active_tokens
actual_recalled_tokens
memory_rope_policy
region_bias_applied
```

---

## 17. Phase 1：完整 frame oracle

Phase 0A/B/C 通过后，再做完整 frame oracle。

### 17.1 为什么必须做 oracle

当前 sparse-token path 同时包含：

```text
capture quality
compression
retrieval
metadata
RoPE
attention routing
```

任何一个失败都会导致无提升。

完整 frame oracle 通过 deterministic recall 去除：

```text
QK compression
retrieval error
random token structure
noisy eviction selection
```

只回答：

> 一个完全正确、完整、有真实 t/h/w 坐标的历史 KV frame，在 scene revisit 时是否有帮助？

### 17.2 建议新增结构

```python
@dataclass
class StructuredMemoryBlock:
    block_id: str
    layer_id: int
    k_raw: torch.Tensor
    v: torch.Tensor
    frame_positions: torch.Tensor
    spatial_positions: torch.Tensor
    num_frames: int
    frame_seq_length: int
    capture_step: int
    capture_timestep: float | None
```

第一版只保存：

```text
one complete latent frame
layer 29 only
```

### 17.3 从 clean context 获取

不要依赖 denoising eviction。

在 clean-context refresh 完成后，直接从：

```python
kv_cache["k_pre_rope"]
kv_cache["v"]
```

截取刚生成的完整 frame：

```python
start = local_end_index - frame_seq_length
end = local_end_index
```

同时构造真实：

```text
frame_positions = current frame id
spatial_positions = 0 ... frame_seq_length-1
```

### 17.4 Oracle 配置

```yaml
lifecache:
  oracle_mode: full_frame
  oracle_layer: 29
  oracle_num_frames: 1
  oracle_recall_step: <second A start>
  compression: none
  retrieval: deterministic
  anchor_enabled: false
  motion_enabled: false
  region_bias_beta: 0.0
```

---

## 18. 固定预算实验

当前 LifeCache 使用：

```text
recent native cache + extra memory
```

这会改变总 token 数和 softmax denominator。

必须增加：

```yaml
active_cache_budget_mode: append
active_cache_budget_mode: replace_recent
```

### replace_recent 模式

例如 native 21 frames：

```text
20 recent frames + 1 historical frame = 21 total frames
```

约束：

```python
assert active_k.shape[0] == native_recent_k.shape[0]
```

建议正式结论以 `replace_recent` 为主，`append` 只用于确认 memory 是否有任何因果作用。

---

## 19. Head-aware oracle

当前 routing 仍是 layer-level majority。

Phase 1 至少比较：

```text
O1 all heads
O2 stable/layout heads only
O3 motion/wave heads recent only
```

第一版可复用现有 HCP SDPA mask：

```text
memory tokens 对 stable/layout heads 可见
memory tokens 对 motion/wave heads设为 -inf
recent tokens 对所有 heads 可见
```

无需立刻实现 ragged per-head K/V。

---

## 20. Prompt schedule

禁止继续仅使用一条包含“后来回到”的长 prompt 作为唯一 scene-revisit 证据。

实现显式 schedule：

```text
A: frames 0–29
B: frames 30–69
A: frames 70–119
```

每段单独更新 conditional embedding。

Oracle recall 在：

```text
第二个 A 开始时
```

被 deterministic 触发。

建议第一批 prompt：

```text
A1 kitchen with blue cabinets + red cup
B1 outdoor garden
A2 same kitchen

A1 white laboratory + blue control panel
B1 dark hallway
A2 same laboratory

A1 dog with red collar near fountain
B1 trees/park
A2 same fountain
```

---

## 21. 实际 attention mass 诊断

当前 QK cosine proxy 不能作为最终证据。

在 K 完成真实 RoPE，并形成最终 active cache 后，采样少量 query/head：

```python
q_sample = q_roped[:, sampled_q, sampled_heads]
k_sample = active_k[:, :, sampled_heads]

logits = torch.einsum(
    "bqhd,bkhd->bhqk",
    q_sample,
    k_sample,
) / math.sqrt(head_dim)

weights = logits.softmax(dim=-1)
```

记录：

```text
attention_mass_memory
attention_mass_recent
attention_mass_memory_by_head
attention_mass_memory_by_role
```

采样建议：

```text
16–32 query tokens
2–4 heads
每 5–10 个生成 block 记录一次
```

避免全 attention map 显存爆炸。

---

## 22. 正式实验矩阵

### O0 Native

```text
Native Self-Forcing
```

### O1 Full-frame append

```text
1 full historical frame
append to native recent
all heads
```

目的：确认 history memory 是否有任何因果作用。

### O2 Full-frame replace

```text
1 full historical frame
replace 1 recent frame
fixed total budget
all heads
```

### O3 Full-frame stable heads

```text
fixed total budget
memory only visible to stable/layout heads
```

### O4 Structured patch grid

```text
full frame
4x4 patch blocks
8x8 patch blocks
```

### O5 Sparse top-k

```text
当前 QK sparse token method
但使用完全正确 metadata 和 sparse 3D RoPE
```

### O6 Random sparse

```text
与 O5 相同 budget
同步替换 K/V/frame/spatial/source metadata
```

---

## 23. 评估协议

不再使用 MP4 文件大小作为主要质量结论。

至少记录：

```text
1. A1–A2 DINO/CLIP scene similarity
2. subject identity similarity
3. background/layout similarity
4. key object reappearance accuracy
5. temporal flicker
6. dynamic degree
7. paired human preference
8. runtime / peak VRAM / active token count
```

人工检查：

```text
蓝色橱柜是否恢复
红杯是否恢复
人物外套颜色是否保持
控制台/喷泉等关键物体是否回归
是否错误带回 B 场景元素
是否因 memory 注入导致冻结或闪烁
```

每种方法至少：

```text
3 prompts x 3 seeds
```

Oracle correctness 阶段可先用单 seed，正式结论必须多 seed。

---

## 24. 决策门槛

### 情况 A：Full-frame oracle 有提升

结论：

```text
recall-after-loss 可行；
当前历史失败主要来自 sparse structure、position、retrieval 或 routing。
```

继续：

```text
structured patch compression
recent-window exclusion
latent/scene descriptor retrieval
semantic gating
rho/soft forget
```

### 情况 B：Full-frame recall 无提升，但 smart retention 有提升

结论：

```text
Self-Forcing 更依赖持续保留，而不是晚期重新注入。
```

方向：

```text
Pyramid/Forcing-KV smart retention
+ archival recall only for explicit revisit
```

### 情况 C：Full-frame recall 和 smart retention 都无提升

再考虑：

```text
Causal-Forcing
latent-level memory
world-state memory
更长、更严格 scene revisit benchmark
```

### 情况 D：实际 memory attention mass 很低

处理：

```text
head mask
region bias
memory gating
```

### 情况 E：attention mass 很高但质量不提升

处理：

```text
检查 V 有效性
检查是否 recall 错 scene
检查 historical domination
从 sparse token 转为完整 frame/patch context
```

---

## 25. 下一轮 coding agent checklist

```text
[ ] Read docs/28 and docs/29 before coding.
[ ] Add spatial metadata validation to TokenSet.__post_init__.
[ ] Propagate spatial_positions in clone_with_tokens and to_device.
[ ] Pass frame/spatial metadata into compression functions.
[ ] Index compression metadata with exactly the same top-k positions.
[ ] Remove runtime post-hoc metadata assignment.
[ ] Extend RecallResult with frame_positions/spatial_positions/rope_modes.
[ ] Apply min-score and top-k indices to all recall metadata.
[ ] Build recall:view with real metadata.
[ ] Make random recall replace all metadata together.
[ ] Remove all -1/synthetic position fallback for recalled tokens.
[ ] Rewrite causal_rope_apply_sparse_3d using native complex freqs correctly.
[ ] Add sparse/full-grid RoPE parity tests.
[ ] Use dynamic grid_h/grid_w from grid_sizes.
[ ] Replace fp[:n] with fp.index_select(0, idx).
[ ] Implement native_relative_clamp mapping.
[ ] Make recall_top_tokens control actual recall count.
[ ] Thread current_frame into distance filtering.
[ ] Fix head role path resolution and assert 30x12 roles.
[ ] Disable or warn on region_bias until it enters SDPA logits.
[ ] Add effective config fields to trace.
[ ] Add StructuredMemoryBlock.
[ ] Add clean full-frame archival capture.
[ ] Add deterministic full-frame oracle recall.
[ ] Add append and replace_recent cache budget modes.
[ ] Add actual post-RoPE attention mass diagnostic.
[ ] Add explicit A-B-A prompt schedule.
[ ] Run O0-O6 only after Phase 0 tests pass.
```

---

## 26. Final recommendation

当前最短、最可信的推进路径不是继续调：

```text
recall_top_tokens
region_bias_beta
layer count
random recall
```

而是完成一个严格闭环：

```text
metadata alignment
+ sparse/full RoPE parity
+ truthful effective config
+ clean full-frame oracle
+ fixed total budget
+ controlled A-B-A schedule
```

最关键的 Phase 0 验收目标是：

> 从一个完整历史 frame 中任意选择 sparse tokens，`causal_rope_apply_sparse_3d` 的结果必须与“先对完整 frame 使用原生 RoPE，再选择相同 tokens”的结果数值一致；同时 K/V/frame/spatial/source metadata 长度必须全程一致且没有 `-1`。

在该条件满足前，继续生成更多视频只会产生更多无法解释的负结果。Phase 0 通过后，优先运行 layer 29、one-full-frame、fixed-budget 的 deterministic oracle，验证 scene revisit 是否真正受益。
