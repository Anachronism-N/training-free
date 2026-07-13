# LifeCache Phase 0 完成条件与 Full-Frame Oracle 推进方案

> 更新时间：2026-07-14  
> 对应最新实现：`fc199c84 feat: LifeCache v3 Phase 0 — position sidecar + sparse 3D RoPE`  
> 目标：重新核对 Phase 0 是否真正完成，并给出下一轮代码修改、单元测试、配置修复、Oracle 实验和决策门槛。

---

## 1. Executive summary

当前最新提交已经开始实现以下关键能力：

```text
1. 使用 global token counter 计算历史绝对 frame position；
2. 增加 spatial_positions；
3. 增加 sparse token-wise 3D RoPE；
4. 从 pipeline → runtime → bank → active view 传播空间位置；
5. 删除旧的 sparse-token fake-frame padding 作为主路径。
```

这些改动方向正确，但当前代码仍有多个会让实验结论失效的问题：

```text
P0-1. compression top-k 后 K/V 与 frame/spatial metadata 没有同步索引；
P0-2. TokenSet clone/to_device 仍会丢失 spatial_positions；
P0-3. RecallResult 仍未传播 frame_positions / spatial_positions / rope_mode；
P0-4. recall:view 仍可能得到 -1 metadata，随后进入 synthetic fallback；
P0-5. sparse 3D RoPE 的 complex frequency 处理方式存在 shape/type 风险；
P0-6. 当前 temporal mapping 是 absolute clamp，不是真正 relative-clamp；
P0-7. metadata 读取使用 fp[:n]，而不是 fp[idx]；
P0-8. recall_top_tokens 可能被 RegionBudget.recall 覆盖；
P0-9. max_frame_distance 没有贯穿到 retrieve_token_sets；
P0-10. region_bias_beta 仍没有进入真实 attention logits；
P0-11. Pyramid head role 路径很可能仍然解析错误并静默 fallback；
P0-12. random recall 只替换 K/V，没有同步替换位置和 source metadata。
```

因此当前阶段不应继续扩大：

```text
recall budget sweep
region bias sweep
layer count sweep
max distance sweep
random recall
更长视频
Causal-Forcing integration
```

当前最短、最科学的推进路径是：

```text
Phase 0A: metadata alignment
Phase 0B: sparse 3D RoPE parity
Phase 0C: configuration truthfulness
Phase 1: clean full-frame oracle
```

---

## 2. 当前最新实现中已经完成的部分

### 2.1 绝对位置计算已经开始改正

eviction path 已不再直接使用 cache slot 作为历史 frame，而是根据：

```python
evict_start_token = (
    kv_cache["global_end_index"].item()
    - kv_cache["local_end_index"].item()
    + sink_tokens
)
abs_token_indices = torch.arange(
    evict_start_token,
    evict_start_token + num_evicted_tokens,
)
frame_positions = abs_token_indices // frame_seqlen
spatial_positions = abs_token_indices % frame_seqlen
```

这比旧版：

```python
frame_positions = token_indices // frame_seqlen
```

更接近真实历史位置。

### 2.2 `TokenSet` 已加入空间位置字段

当前已有：

```python
frame_positions: Optional[torch.Tensor]
spatial_positions: Optional[torch.Tensor]
```

### 2.3 attention path 已新增 sparse 3D RoPE

当前新增：

```python
causal_rope_apply_sparse_3d(...)
```

理论上允许 arbitrary sparse tokens 使用各自独立的 temporal/spatial index，而不再伪造成完整 frame grid。

### 2.4 pipeline 已传递 spatial positions

当前 payload 已包含：

```text
frame_positions
spatial_positions
```

pipeline 也会传给：

```python
runtime.on_kv_evicted(...)
```

这些都是正确方向。

---

## 3. P0 问题一：compression 之后 metadata 与 K/V 错位

### 3.1 当前逻辑

`compress_qk_proxy()` 内部执行：

```python
positions = select_topk_tokens(scores, config.topk, config.min_tokens)
selected_k = k.index_select(0, positions)
selected_v = v.index_select(0, positions)
```

但它没有接收：

```text
frame_positions
spatial_positions
source absolute token positions
```

随后 runtime 在 TokenSet 构造完成后执行：

```python
token_set.frame_positions = frame_positions
token_set.spatial_positions = spatial_positions
```

此时 `token_set.k` 已经是 top-k 长度，而位置 tensor 仍是原始 evicted token 长度。

### 3.2 典型错误

```text
num_evicted_tokens = 4680
compression_topk   = 512
len(token_set.k)   = 512
len(frame_positions) = 4680
len(spatial_positions) = 4680
```

这会造成 recalled K/V 和位置 metadata 完全错位。

### 3.3 必须修改

修改：

```text
src/lifecycle_kv/compression.py
src/lifecycle_kv/runtime.py
```

建议新增辅助函数：

```python
def _select_optional(
    value: torch.Tensor | None,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if value is None:
        return None
    positions = positions.to(value.device)
    return value.index_select(0, positions)
```

修改压缩接口：

```python
def compress_qk_proxy(
    *,
    ...,
    frame_positions: torch.Tensor | None = None,
    spatial_positions: torch.Tensor | None = None,
    rope_mode: str = "pre_rope",
) -> TokenSet:
    scores = qk_proxy_scores(q, k)
    positions = select_topk_tokens(scores, config.topk, config.min_tokens)

    return TokenSet(
        ...,
        k=k.index_select(0, positions),
        v=v.index_select(0, positions),
        token_indices=token_indices.index_select(0, positions),
        frame_positions=_select_optional(frame_positions, positions),
        spatial_positions=_select_optional(spatial_positions, positions),
        rope_mode=rope_mode,
    )
```

runtime 中删除：

```python
token_set.frame_positions = frame_positions
token_set.spatial_positions = spatial_positions
```

metadata 必须在构造 TokenSet 时一次性进入，以便触发 `__post_init__()` 校验。

---

## 4. P0 问题二：TokenSet 的 clone/device 传播不完整

当前 `clone_with_tokens()` 和 `to_device()` 已传播 `frame_positions`，但没有完整传播 `spatial_positions`。

### 4.1 必须修改

文件：

```text
src/lifecycle_kv/tokenset.py
```

`clone_with_tokens()`：

```python
spatial_positions=(
    self.spatial_positions.index_select(0, token_positions)
    if self.spatial_positions is not None
    else None
),
```

`to_device()`：

```python
spatial_positions=(
    self.spatial_positions.to(device)
    if self.spatial_positions is not None
    else None
),
```

### 4.2 增加严格校验

```python
if self.spatial_positions is not None:
    if self.spatial_positions.ndim != 1:
        raise ValueError("spatial_positions must be 1D")
    if self.spatial_positions.numel() != self.k.shape[0]:
        raise ValueError("spatial_positions length must match token count")
    if self.spatial_positions.numel() > 0 and self.spatial_positions.min() < 0:
        raise ValueError("spatial_positions must be non-negative")
```

对于 pre-RoPE historical token，建议进一步要求：

```python
if self.rope_mode == "pre_rope":
    if self.frame_positions is None or self.spatial_positions is None:
        raise ValueError("pre_rope historical memory requires frame/spatial positions")
```

---

## 5. P0 问题三：RecallResult 没有传播位置 metadata

### 5.1 当前缺失

当前 `RecallResult` 没有：

```text
frame_positions
spatial_positions
rope_modes
```

在 set selection、min-token-score filter 和 token top-k 后，只同步更新 K/V/index。

### 5.2 必须修改

文件：

```text
src/lifecycle_kv/recall.py
```

扩展：

```python
@dataclass
class RecallResult:
    k: torch.Tensor | None
    v: torch.Tensor | None
    token_indices: torch.Tensor | None
    token_scores: torch.Tensor | None
    token_sets: list[TokenSet]
    source_set_ids: list[str] = field(default_factory=list)
    source_positions: torch.Tensor | None = None
    set_scores: torch.Tensor | None = None
    frame_positions: torch.Tensor | None = None
    spatial_positions: torch.Tensor | None = None
    rope_modes: list[str] = field(default_factory=list)
```

拼接 selected sets：

```python
all_frame_positions = torch.cat([
    s.frame_positions.to(q.device)
    for s in selected_sets
], dim=0)

all_spatial_positions = torch.cat([
    s.spatial_positions.to(q.device)
    for s in selected_sets
], dim=0)

all_rope_modes = [
    s.rope_mode
    for s in selected_sets
    for _ in range(s.num_tokens)
]
```

要求：

```python
for s in selected_sets:
    if s.rope_mode == "pre_rope":
        if s.frame_positions is None or s.spatial_positions is None:
            raise RuntimeError(...)
```

### 5.3 所有 filter/top-k 必须同步

min-score filter：

```python
all_frame_positions = all_frame_positions[keep_mask]
all_spatial_positions = all_spatial_positions[keep_mask]
all_rope_modes = [m for m, keep in zip(all_rope_modes, keep_mask.tolist()) if keep]
```

top-k：

```python
selected_frame_positions = all_frame_positions.index_select(0, positions)
selected_spatial_positions = all_spatial_positions.index_select(0, positions)
selected_rope_modes = [all_rope_modes[int(i)] for i in positions.tolist()]
```

返回结果必须保证：

```python
assert result.k.shape[0] == result.frame_positions.numel()
assert result.k.shape[0] == result.spatial_positions.numel()
assert len(result.rope_modes) == result.k.shape[0]
```

---

## 6. P0 问题四：recall:view 仍未携带真实位置

当前 composer 构造：

```python
TokenSet(
    set_id="recall:view",
    k=recall_result.k,
    v=recall_result.v,
    ...,
)
```

但没有设置：

```text
frame_positions
spatial_positions
rope_mode
source_positions
```

### 6.1 必须修改

文件：

```text
src/lifecycle_kv/active_cache.py
```

```python
recall_rope_modes = set(recall_result.rope_modes)
if len(recall_rope_modes) != 1:
    raise RuntimeError(
        f"mixed rope modes in one recall view: {recall_rope_modes}"
    )

recalled.append(
    TokenSet(
        set_id="recall:view",
        ...,
        frame_positions=recall_result.frame_positions,
        spatial_positions=recall_result.spatial_positions,
        rope_mode=next(iter(recall_rope_modes)),
        source_positions=recall_result.source_positions,
    )
)
```

### 6.2 禁止 recall metadata fallback

当前 composer 对缺失 metadata 填 `-1`。

更安全的规则：

```python
if s.region == CacheRegion.RECALL:
    if s.frame_positions is None or s.spatial_positions is None:
        raise RuntimeError("recalled token position metadata missing")
```

对于 recent/anchor region 可以暂时填 `-1`，但 attention 中只能对 recall mask 对应位置读取 recall metadata。

---

## 7. P0 问题五：random recall 必须同步替换所有 metadata

当前 random recall 只替换：

```text
K
V
```

没有替换：

```text
frame_positions
spatial_positions
source_set_ids
source_positions
rope_mode
```

因此 random recall ablation 不是 position-consistent 的严格对照。

### 必须修改

构造全 bank tensor 时同时构造：

```python
all_k
all_v
all_frame_positions
all_spatial_positions
all_source_set_ids
all_source_positions
all_rope_modes
```

应用同一个 `rand_idx`：

```python
view.k[recall_pos] = all_k[rand_idx]
view.v[recall_pos] = all_v[rand_idx]
view.frame_positions[recall_pos] = all_frame_positions[rand_idx]
view.spatial_positions[recall_pos] = all_spatial_positions[rand_idx]
```

混合 pre/post-RoPE memory 时禁止 random replacement，除非同时区分 rope mode。

---

## 8. P0 问题六：当前 sparse 3D RoPE 需要重新实现与验证

### 8.1 当前风险

现有 Wan `freqs` 在原生 `causal_rope_apply()` 中直接与 complex `x_i` 相乘：

```python
x_i = x_i * freqs_i
```

因此 `freq_i` 已是 complex frequency tensor。

当前 sparse path 再调用：

```python
torch.view_as_complex(freq_i...reshape(..., 2))
```

可能造成：

```text
1. 对已经是 complex 的频率重复 view_as_complex；
2. reshape 维度不合法；
3. token/head/channel 维度被错误重排；
4. 结果与原生 full-grid RoPE 不一致。
```

### 8.2 推荐实现

```python
def causal_rope_apply_sparse_3d(
    x: torch.Tensor,              # [T,H,D] or [B,T,H,D]
    freqs: torch.Tensor,
    temporal_idx: torch.Tensor,   # [T]
    spatial_idx: torch.Tensor,    # [T]
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    elif x.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(...)

    B, T, H, D = x.shape
    c = D // 2
    freq_t, freq_h, freq_w = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3],
        dim=1,
    )

    t_idx = temporal_idx.long().to(x.device)
    s_idx = spatial_idx.long().to(x.device)
    h_idx = s_idx // grid_w
    w_idx = s_idx % grid_w

    token_freqs = torch.cat([
        freq_t.index_select(0, t_idx),
        freq_h.index_select(0, h_idx),
        freq_w.index_select(0, w_idx),
    ], dim=-1)  # [T,D/2], already complex

    x_complex = torch.view_as_complex(
        x.float().reshape(B, T, H, -1, 2)
    )

    out_complex = x_complex * token_freqs[None, :, None, :]
    out = torch.view_as_real(out_complex).flatten(-2).type_as(x)

    return out.squeeze(0) if squeeze_batch else out
```

### 8.3 禁止 synthetic fallback

当前无有效 metadata 时 fallback 到：

```python
temporal_idx = zeros
spatial_idx = arange
```

这会重新制造伪坐标。

必须改为：

```python
raise RuntimeError(
    "LifeCache recall contains invalid temporal/spatial positions"
)
```

---

## 9. Sparse 3D RoPE parity test

这是 Phase 0 的核心验收测试。

### 9.1 测试目标

对于完整 raw K：

```text
方案 A：先对完整 frame 用原生 full-grid RoPE，再选 sparse tokens；
方案 B：先选相同 sparse raw tokens，再用 sparse 3D RoPE；
```

两者应数值一致。

### 9.2 单元测试伪代码

新增：

```text
tests/test_sparse_3d_rope.py
```

```python
def test_sparse_rope_matches_full_grid_rope():
    B = 1
    F = 3
    H_grid = 4
    W_grid = 5
    num_heads = 2
    head_dim = 12
    T = F * H_grid * W_grid

    raw = torch.randn(B, T, num_heads, head_dim)
    grid_sizes = torch.tensor([[F, H_grid, W_grid]])
    freqs = make_test_freqs(...)

    full = causal_rope_apply(
        raw,
        grid_sizes,
        freqs,
        start_frame=7,
    )

    chosen = torch.tensor([0, 3, 8, 17, 21, 35, 49])
    abs_token = 7 * H_grid * W_grid + chosen
    t_idx = abs_token // (H_grid * W_grid)
    spatial_idx = abs_token % (H_grid * W_grid)

    sparse = causal_rope_apply_sparse_3d(
        raw[:, chosen],
        freqs,
        temporal_idx=t_idx,
        spatial_idx=spatial_idx,
        grid_h=H_grid,
        grid_w=W_grid,
    )

    torch.testing.assert_close(
        sparse,
        full[:, chosen],
        rtol=1e-4,
        atol=1e-4,
    )
```

还需要测试：

```text
single token
single frame
duplicate temporal position
different h/w
batch size 1
BF16 input
invalid spatial position raises error
metadata length mismatch raises error
```

### 9.3 验收

```text
parity test 全部通过
任何 recalled token 不允许走 synthetic fallback
```

在此之前不运行正式视频实验。

---

## 10. P0 问题七：temporal mapping 目前不是 relative-clamp

当前路径近似：

```python
temporal_idx = frame_positions.clamp(0, TR - 1)
```

这是绝对 frame clamp，不是相对 query 的距离映射。

### 10.1 正确的 native-relative-clamp

保持 native query 和 recent post-RoPE K 不变，仅为 historical memory 分配与 query 最大距离不超过训练范围的位置：

```python
distance = (
    current_start_frame - historical_frame
).clamp(0, TR - 1)

mapped_memory_position = current_start_frame - distance
```

例如：

```text
current query frame = 70
historical frame 60 -> distance=10 -> mapped=60
historical frame 30 -> distance=20 -> mapped=50
historical frame 0  -> distance=20 -> mapped=50
```

### 10.2 配置命名

建议明确两种模式：

```yaml
rope_remap_policy: native_relative_clamp
rope_remap_policy: online_local
```

`native_relative_clamp`：

```text
query/recent 保持原生绝对位置；
memory 被映射到 query 之前的合法相对位置。
```

`online_local`：

```text
query、recent raw K、memory raw K 一起重映射到统一的局部坐标系。
```

后者更接近 MemRoPE/LongLive-RAG，但会改变 baseline，必须增加：

```text
online-local without memory
```

作为控制组。

---

## 11. P0 问题八：使用 metadata 时必须按 recall index 读取

当前 attention path：

```python
idx = is_recall.nonzero(as_tuple=True)[0]
```

之后却使用：

```python
fp[:idx.shape[0]]
sp[:idx.shape[0]]
```

这仅在 recall 永远位于 active cache 最前面时成立。

一旦 active order 为：

```text
[anchor | recall | recent]
```

就会读取 anchor 的 metadata。

必须改成：

```python
recall_fp = fp.index_select(0, idx)
recall_sp = sp.index_select(0, idx)
```

同时增加：

```python
assert recall_fp.numel() == idx.numel()
assert recall_sp.numel() == idx.numel()
```

---

## 12. P0 问题九：配置值没有真实控制实验

### 12.1 recall_top_tokens 被 RegionBudget 覆盖

配置：

```yaml
recall_top_tokens: 32
```

但 composer 中实际使用：

```python
top_tokens=budget.recall
```

`HeadRole.LAYOUT` 默认：

```python
recall=512
```

所以旧的：

```text
32 / 64 / 256 token ablation
```

可能实际全部召回 512 token。

### 修复

```python
effective_recall_budget = min(
    budget.recall,
    self.recall_config.top_tokens,
)
```

或删除重复预算来源，只保留一个。

trace 必须记录：

```text
configured_recall_top_tokens
effective_recall_budget
actual_recalled_tokens
```

### 12.2 max_frame_distance 未贯穿调用链

当前 runtime 创建的 `RecallConfig` 含 `max_frame_distance`，但 composer 重建配置时未复制该字段，且没有把 `current_frame` 传入 recall。

必须贯穿：

```text
attention current_start_frame
→ runtime.compose_active_cache(current_frame=...)
→ composer.compose(current_frame=...)
→ recall_tokens(current_frame=...)
→ retrieve_token_sets(current_frame=...)
```

trace：

```text
effective_max_frame_distance
candidate_sets_before_distance_filter
candidate_sets_after_distance_filter
selected_distance_min/max/mean
```

### 12.3 region bias 尚未生效

`ActiveCacheView.region_bias` 已构造，但 attention 最终仍调用：

```python
attention(roped_query, active_k, active_v)
```

没有 additive mask。

现阶段建议：

```yaml
region_bias_beta: 0.0
```

同时在 config loader 中警告：

```text
region_bias_beta > 0 but real additive-bias attention path is disabled
```

等 full-frame oracle 跑通后，再复用 HCP 的 SDPA path 让 bias 进入 logits。

### 12.4 head role 路径解析错误风险

当前相对路径基于 config 文件目录拼接：

```python
join(dirname(config_path), head_roles_path)
```

配置中写：

```yaml
head_roles_path: third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
```

实际可能被解析为：

```text
configs/lifecache/third_party/Pyramid-Forcing/...
```

而不是 repo root 下的 `third_party/`。

必须：

```python
repo_root = Path(__file__).resolve().parents[3]
role_path = (repo_root / configured_path).resolve()
```

并验证：

```python
if len(head_roles) != 30 * 12:
    raise RuntimeError(
        f"expected 360 Pyramid head labels, got {len(head_roles)}"
    )
```

打印并 trace：

```text
head_roles_path_resolved
head_roles_loaded
layout_count
wave_count
motion_count
generic_count
unknown_count
```

---

## 13. 建议拆分为四个提交

## Commit A：metadata correctness

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
[ ] compression top-k 同步索引 frame/spatial metadata
[ ] TokenSet clone/to_device 传播 spatial_positions
[ ] TokenSet 增加严格 metadata 校验
[ ] RecallResult 新增 frame_positions/spatial_positions/rope_modes
[ ] min-score filter/top-k 同步所有 metadata
[ ] recall:view 写入真实 metadata
[ ] random recall 同步替换 metadata
[ ] 删除 recall metadata 的 -1 fallback
```

验收：

```text
任何 recall result:
len(K) == len(V)
       == len(frame_positions)
       == len(spatial_positions)
       == len(source_set_ids)

invalid recall position count = 0
```

---

## Commit B：RoPE correctness

修改：

```text
third_party/Self-Forcing/wan/modules/causal_model.py
tests/test_sparse_3d_rope.py
```

任务：

```text
[ ] 修复 sparse 3D RoPE complex frequency 处理
[ ] 使用动态 grid_h/grid_w
[ ] 使用 fp[idx]/sp[idx]
[ ] 删除 synthetic fallback
[ ] 实现 native_relative_clamp
[ ] 可选实现 online_local
[ ] 添加 full-grid vs sparse parity test
[ ] 添加 BF16/invalid metadata tests
```

验收：

```text
sparse 3D RoPE 与 full-grid RoPE 数值一致
```

---

## Commit C：configuration truthfulness

修改：

```text
src/lifecycle_kv/runtime.py
src/lifecycle_kv/active_cache.py
third_party/Self-Forcing/scripts/lifecache_manager.py
configs/lifecache/*.yaml
```

任务：

```text
[ ] recall_top_tokens 真正控制有效 budget
[ ] max_frame_distance 贯穿到 retrieval
[ ] 修复 head role 路径
[ ] 要求 30x12 labels，否则报错
[ ] region bias 未生效时警告
[ ] capture_clean_only 配置与实际逻辑统一
[ ] trace 记录所有 effective config
```

验收：

```text
配置文件值 == trace 中的 effective 值 == 实际运行值
```

---

## Commit D：clean full-frame oracle

新增建议：

```text
src/lifecycle_kv/structured_memory.py
configs/lifecache/lifecache_full_frame_oracle.yaml
prompts/aba_schedule/*.yaml
```

不要继续通过 eviction sparse bank 构造 oracle。

在 clean-context refresh 后直接捕获刚生成的干净 block：

```python
start = local_end_index - current_num_frames * frame_seq_length
end = local_end_index

block = StructuredMemoryBlock(
    block_id=...,
    layer_id=...,
    k_raw=kv_cache["k_pre_rope"][:, start:end].clone(),
    v=kv_cache["v"][:, start:end].clone(),
    abs_frame_idx=...,
    spatial_idx=...,
    capture_step=...,
)
```

这一步绕开：

```text
noisy eviction
capture timestep threshold
QK compression
set retrieval
token top-k
```

直接测试“正确历史 KV 是否有价值”。

---

## 14. StructuredMemoryBlock 建议结构

```python
@dataclass
class StructuredMemoryBlock:
    block_id: str
    layer_id: int
    head_group: str

    k_raw: torch.Tensor             # [T,H,D]
    v: torch.Tensor                 # [T,H,D]

    abs_frame_idx: torch.Tensor     # [T]
    spatial_idx: torch.Tensor       # [T]

    start_frame: int
    end_frame: int
    num_frames: int
    frame_seq_length: int

    memory_unit: str                # full_frame / patch_grid / sparse_token
    capture_step: int
    capture_timestep: float | None

    descriptor: torch.Tensor | None = None
    scene_id: str | None = None
    entity_ids: tuple[str, ...] = ()
```

第一版只需要：

```text
one frame
layer 29
all 1560 tokens
raw K + V
real temporal/spatial coordinates
```

---

## 15. Full-frame oracle 实验设计

### 15.1 核心问题

```text
给模型一个完全正确、完整、有真实 t/h/w 坐标的历史 scene-A frame，
第二次回到 scene A 时是否会比 native Self-Forcing 更一致？
```

### 15.2 固定实验条件

```text
Backbone: Self-Forcing
Enabled layer: 29 only
Memory unit: one complete latent frame
Memory tokens: 1560
Compression: none
Retrieval: deterministic
Anchor: disabled
Motion cache: disabled
Region bias: 0
Head routing: first run all heads; second run stable heads only
Seeds: 0, 1, 2
```

### 15.3 Prompt schedule

必须实现显式 condition 切换，而不是使用单条长 prompt：

```text
A: frames 0–29
B: frames 30–69
A: frames 70–119
```

示例：

```yaml
segments:
  - start: 0
    end: 29
    prompt: >
      A woman in a yellow coat stands in a small kitchen with blue cabinets,
      a red cup on a wooden table, warm indoor lighting.

  - start: 30
    end: 69
    prompt: >
      The same woman walks through a green garden with flowers and stone paths.

  - start: 70
    end: 119
    prompt: >
      The same woman returns to the original small kitchen with blue cabinets
      and the same red cup on the wooden table.
```

在第二个 A 开始时确定性注入第一个 A 的完整 memory frame。

---

## 16. Oracle 实验矩阵

| ID | Memory | Budget | Heads | 目的 |
|---|---|---|---|---|
| O0 | none | native | all | 原生基线 |
| O1 | full frame | append | all | 判断历史 KV 是否有任何因果作用 |
| O2 | full frame | replace recent | all | 固定总预算公平对照 |
| O3 | full frame | replace recent | stable/layout heads | 测试 motion-head 污染 |
| O4 | regular patch grid | replace recent | stable heads | 测试结构化压缩 |
| O5 | current sparse top-k | replace recent | stable heads | 与旧方法对照 |

### 16.1 append 模式

```text
native recent + 1560 historical tokens
```

用途仅是判断 historical memory 是否能影响结果。

### 16.2 replace_recent 模式

保持总预算不变：

```text
20 recent frames + 1 historical frame = 21 total frames
```

要求：

```text
active_tokens == native_max_tokens
```

### 16.3 stable-head 模式

第一版可使用 per-head additive mask：

```text
stable/layout heads: historical + recent
wave/motion heads: recent only
```

不要求一开始实现 ragged KV kernel。

---

## 17. 必须记录的诊断

### 17.1 Metadata trace

```text
memory_tokens
frame_position_min/max
spatial_position_min/max
invalid_frame_position_count
invalid_spatial_position_count
unique_source_frames
unique_spatial_positions
rope_mode
```

验收：

```text
invalid counts = 0
```

### 17.2 RoPE trace

```text
historical_abs_frame_min/max
mapped_temporal_position_min/max
current_query_position
relative_distance_after_mapping_min/max
spatial_h_min/max
spatial_w_min/max
```

验收：

```text
relative_distance_after_mapping <= TR - 1
```

### 17.3 Effective configuration

```text
configured_recall_top_tokens
effective_recall_budget
actual_recalled_tokens
effective_max_frame_distance
enabled_layers
head_roles_loaded
active_cache_budget_mode
actual_active_tokens
```

### 17.4 实际 attention mass

不再用 pre-RoPE cosine proxy 作为主要证据。

在 K 完成真实 RoPE、bias 完成添加后，采样：

```text
16–32 query tokens
2–4 heads
若干生成 step
```

计算：

```python
logits = torch.einsum(
    "qhd,khd->hqk",
    sampled_q,
    active_k,
) / math.sqrt(head_dim)

weights = torch.softmax(logits, dim=-1)

memory_mass = weights[..., memory_mask].sum(dim=-1).mean()
recent_mass = weights[..., recent_mask].sum(dim=-1).mean()
```

记录：

```text
attention_mass_memory
attention_mass_recent
attention_mass_by_head_role
```

---

## 18. 质量评估协议

禁止继续用 MP4 文件大小作为主要结论。

至少评估：

```text
A1–A2 DINO similarity
A1–A2 CLIP image similarity
subject identity similarity
background/layout similarity
key object reappearance accuracy
temporal flicker
dynamic degree
paired human preference
```

厨房案例人工 checklist：

```text
蓝色橱柜是否恢复
红杯是否恢复
黄色外套是否保持
木桌布局是否一致
花园元素是否错误泄漏
人物是否冻结
画面是否暗化
```

每项：

```text
0 = absent / bad
1 = partial
2 = good
```

最少：

```text
3 prompts × 3 seeds
```

---

## 19. 测试清单

### 19.1 Unit tests

```text
[ ] TokenSet metadata length validation
[ ] TokenSet clone_with_tokens propagates spatial positions
[ ] TokenSet to_device propagates spatial positions
[ ] compression top-k keeps aligned metadata
[ ] recall min-score filter keeps aligned metadata
[ ] recall top-k keeps aligned metadata
[ ] random recall replaces all metadata consistently
[ ] sparse 3D RoPE parity with full-grid RoPE
[ ] invalid recall position raises
[ ] mixed rope_mode recall raises
```

### 19.2 Integration tests

```text
[ ] trace-only output equals native
[ ] compression-only output equals native
[ ] effective recall budget matches config
[ ] max_frame_distance changes candidate count
[ ] head-role loader reads 360 labels
[ ] full-frame oracle active token count is correct
[ ] replace_recent keeps native total budget
[ ] memory attention mass can be measured
```

---

## 20. 下一轮 agent checklist

```text
[ ] Read docs/28 and docs/29 before coding.
[ ] Fix compression metadata indexing.
[ ] Fix TokenSet spatial metadata propagation.
[ ] Extend RecallResult with frame/spatial/rope metadata.
[ ] Propagate metadata through all filters and top-k.
[ ] Construct recall:view with real metadata.
[ ] Remove -1/synthetic position fallback for recall.
[ ] Fix random recall metadata replacement.
[ ] Rewrite sparse 3D RoPE using native complex freqs.
[ ] Add full-grid parity test.
[ ] Use fp.index_select(idx) and sp.index_select(idx).
[ ] Implement native_relative_clamp mapping.
[ ] Remove hardcoded grid_h=60/grid_w=104.
[ ] Make recall_top_tokens control actual budget.
[ ] Pass current_frame/max_frame_distance through retrieval.
[ ] Fix Pyramid head-role path resolution.
[ ] Assert 30x12 role matrix.
[ ] Disable or warn on unused region_bias.
[ ] Add StructuredMemoryBlock.
[ ] Add clean-context full-frame archival path.
[ ] Add deterministic scene-A recall trigger.
[ ] Add replace_recent fixed-budget mode.
[ ] Add explicit A-B-A prompt schedule.
[ ] Add actual post-RoPE attention-mass diagnostics.
[ ] Run O0–O5 only after Phase 0 tests pass.
```

---

## 21. 决策门槛

### Case A：Full-frame oracle 有提升

结论：

```text
历史 KV recall-after-loss 可行；
此前失败来自 sparse structure、position、selection 或 routing。
```

继续：

```text
regular patch-grid compression
recent-window exclusion
latent/scene descriptor retrieval
stable-head gating
rho/soft forgetting
```

### Case B：Full-frame recall 无提升，但 smart retention 有提升

结论：

```text
该 backbone 更依赖 continuous retention，而不是 late recall。
```

方向：

```text
Pyramid/Forcing-KV style smart retention
+ archival recall only for explicit scene revisit
```

### Case C：Full-frame recall 和 smart retention 都无提升

再考虑：

```text
Causal-Forcing
更长生成长度
latent-level memory
world-state memory
更严格 scene-revisit benchmark
```

### Case D：memory attention mass 接近 0

优先修复：

```text
head access mask
real region bias
memory gating
retrieval trigger
```

### Case E：memory attention mass 很高但质量不提升

优先检查：

```text
recalled V 是否有效
是否召回错误 scene
history domination
完整结构是否仍被破坏
prompt schedule 是否真实执行 A-B-A
```

---

## 22. 对现有实验结论的修订

当前可以保留的结论：

> LifeCache-v2 的 arbitrary sparse-token recall 在现有实现下没有观察到稳定提升。

当前不应保留为最终结论的说法：

```text
recall-after-loss 已被证明无效
Self-Forcing 的瓶颈不是 old K/V loss 已被确认
region bias 无效
recall token budget 不影响结果
near/far distance 不影响结果
random recall 与 QK recall 等价
```

原因是：

```text
部分 ablation 配置可能没有真实生效；
position metadata 没有全链路对齐；
sparse 3D RoPE 尚未通过 parity；
实际 attention mass 尚未测量；
A-B-A prompt 尚不是严格 schedule。
```

---

## 23. Final recommendation

当前最优先目标不是再生成更多视频，而是完成一个严格的不变量闭环：

```text
K/V top-k
→ temporal/spatial metadata 使用相同 index
→ RecallResult 全量传播
→ recall:view 不含无效位置
→ sparse 3D RoPE 与 full-grid RoPE parity
→ effective config 与 trace 一致
→ full-frame clean oracle 合法进入 attention
```

本阶段最关键验收标准是：

> **对一个完整历史 frame 随机选择任意 sparse tokens，token-wise 3D RoPE 的输出必须与“先对完整 frame 使用原生 RoPE，再选择相同 token”的输出数值一致。**

该测试通过后，再运行：

```text
layer 29
one full historical frame
fixed total budget
controlled A-B-A schedule
```

只有这个 Oracle 仍无效，才适合讨论转向 pure smart-retention、Causal-Forcing 或 latent/world-state memory。