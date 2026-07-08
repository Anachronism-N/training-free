下面给出一份可以直接交给代码 agent 的**详细优化指南**。目标不是继续扩展概念，而是把当前 `src/lifecycle_kv` 从“模块骨架”推进到“能接入 Self-Forcing、能跑 trace、能压缩 bank、能逐步打开 recall 的实验原型”。

当前仓库状态是：`TokenSet / TokenSetBank / compression / recall / anchor / motion / head_roles / active_cache / instrumentation` 已经实现，但还没有接入 `third_party/Self-Forcing`，下一步应该先做 disabled-by-default 的 Phase 0/1 接入，而不是立刻改变生成输出。

---

# 一、总体优化目标

代码 agent 的目标应该拆成 5 个阶段：

```text
Phase A: 优化 LifeCache 核心库
Phase B: 新增 LifeCacheRuntime 调度层
Phase C: Self-Forcing trace-only 接入
Phase D: compression-only 接入
Phase E: union recall 小范围启用
```

当前不要做：

```text
1. 不要引入 VLM / LLM / entity tracking。
2. 不要做 full stale / invalid state。
3. 不要直接修改所有 attention heads 的复杂 split-head kernel。
4. 不要一上来启用 region bias。
5. 不要默认改变生成输出。
```

当前应该做：

```text
1. 让 LifeCache 能在 Self-Forcing 推理时 trace K/V。
2. 让 evicted recent K/V 能变成 TokenSet。
3. 让 bank 有稳定 stats / pruning / dedup。
4. 让 Q-K proxy compression 可以不依赖真实 attention map。
5. 让 union recall 能在少量层、少量 top-k 下安全启用。
```

---

# 二、当前代码问题总览

## 1. `TokenSet` 基本正确，但缺少 source trace 信息

当前 `TokenSet` 已经是 `[tokens, heads, dim]` 的轻量 token-level K/V payload，并检查 k/v shape、token_indices 和 importance_score 长度。

问题是：recall view 生成后很难追踪 recalled token 来自哪个 source TokenSet。后续分析 recall 是否有效时，需要 source map。

## 2. `TokenSetBank` 能控制预算，但 prune 太简单

当前 bank 只有 `_sets` 和 `_by_region`，支持 add/list/prune/as_tensors。
当前 prune priority 是：

```python
quality + 0.1 * importance + 0.01 * access_count
```

问题是没有：

```text
stats()
dedup()
按 layer/head_group 的预算统计
最近使用时间更新
memory report
```

## 3. `compression.py` 只支持 exact attention participation

当前 `compress_attention_participation()` 要求输入真实 attention map，并根据 key 维度平均得到 AP score。
但 Self-Forcing 的高效 attention 路径未必能拿到 full attention weights。第一版必须支持 **Q-K proxy compression**。

## 4. `recall.py` 结构正确，但缺 source map / step / 距离过滤

当前 recall 是两阶段：

```text
TokenSet scoring -> token-level Q-K top-k
```

并且最终分数是：

```python
scores = 0.7 * qk + 0.3 * importance
```

问题是：

```text
1. access_count 增加了，但 last_used_step 没更新。
2. RecallResult 没有 source_set_ids / source_positions。
3. 没有 max_frame_distance / chunk distance 过滤。
4. 没有 fallback 空结果诊断。
```

## 5. `active_cache.py` 已有 head role budget，但现在直接 recall anchors + compressed

当前 `ActiveCacheComposer.compose()` 先加入 anchors，然后 recall 候选又是 `anchors + compressed`，这可能导致 anchor 重复进入 active cache。

另外 `_take_tokens()` 超预算时直接取前 N 个 token，而不是按 importance/motion score 取 top-k。

## 6. `instrumentation.py` 能写 JSONL，但 trace schema 还不够

当前 trace event 支持 step/layer/head/event/kv_shape/recent_span/region_mass/extra。
这可以保留，但需要规范 `extra` 的字段，让后续实验可以聚合统计。

---

# 三、文件级修改指南

下面按文件给出具体修改建议。

---

## 1. 修改 `src/lifecycle_kv/tokenset.py`

### 1.1 增加 source tracking 字段

在 `TokenSet` dataclass 中加入：

```python
source_set_id: str | None = None
source_region: CacheRegion | None = None
source_positions: torch.Tensor | None = None
```

建议放在 `region` 附近：

```python
region: CacheRegion = CacheRegion.COMPRESSED
source_set_id: Optional[str] = None
source_region: Optional[CacheRegion] = None
source_positions: Optional[torch.Tensor] = None
```

### 1.2 在 `__post_init__` 中校验 source_positions

加入：

```python
if self.source_positions is not None:
    if self.source_positions.ndim != 1:
        raise ValueError("source_positions must be a 1D tensor")
    if self.source_positions.numel() != self.k.shape[0]:
        raise ValueError("source_positions length must match token count")
```

### 1.3 修改 `clone_with_tokens()`

当前 `clone_with_tokens()` 会复制 k/v/token_indices/importance_score/motion_score。

修改为：

```python
source_positions = token_positions
if self.source_positions is not None:
    source_positions = self.source_positions.index_select(0, token_positions)
```

然后传入：

```python
source_set_id=self.source_set_id or self.set_id,
source_region=self.source_region or self.region,
source_positions=source_positions,
```

### 1.4 增加轻量工具函数

新增：

```python
def to_device(self, device: torch.device | str) -> "TokenSet":
    ...
```

用途：减少 recall / active_cache 中到处 `.to(q.device)` 的重复逻辑。

实现注意：不要 inplace 改原对象，返回新 TokenSet。

---

## 2. 修改 `src/lifecycle_kv/bank.py`

### 2.1 新增 `BankStats`

新增 dataclass：

```python
@dataclass
class BankStats:
    num_sets: int
    total_tokens: int
    sets_by_region: dict[str, int]
    tokens_by_region: dict[str, int]
    sets_by_layer: dict[int, int]
    tokens_by_layer: dict[int, int]
```

### 2.2 新增 `stats()`

实现：

```python
def stats(self) -> BankStats:
    sets_by_region = {}
    tokens_by_region = {}
    sets_by_layer = defaultdict(int)
    tokens_by_layer = defaultdict(int)

    for s in self._sets.values():
        r = s.region.value
        sets_by_region[r] = sets_by_region.get(r, 0) + 1
        tokens_by_region[r] = tokens_by_region.get(r, 0) + s.num_tokens
        sets_by_layer[s.layer_id] += 1
        tokens_by_layer[s.layer_id] += s.num_tokens

    return BankStats(...)
```

### 2.3 新增 `dedup()`

目的：避免 bank 中保存大量相似 TokenSet。

```python
def dedup(
    self,
    region: CacheRegion | None = None,
    similarity_threshold: float = 0.985,
) -> int:
    ...
```

逻辑：

```text
1. 取指定 region 的 TokenSet。
2. 按 priority 从高到低排序。
3. 遍历保留列表。
4. 如果当前 set 的 k_summary 和已保留 set 的 cosine similarity > threshold，则删除当前 set。
5. 返回删除数量。
```

注意：`k_summary` 当前 shape 是 `[heads, dim]`，可以 flatten 后 normalize。

### 2.4 修改 `prune()` 的 priority

当前 priority 只考虑 quality / importance / access_count。

建议改成：

```python
def priority(set_id: str) -> tuple[float, int, int]:
    s = self._sets[set_id]
    quality = float(s.quality_score)
    importance = float(s.importance_score.float().mean()) if s.importance_score is not None else 0.0
    recency = float(s.last_used_step)
    usage = min(float(s.access_count) / 10.0, 1.0)

    score = (
        1.0 * quality
        + 0.5 * importance
        + 0.2 * usage
    )
    return (score, int(recency), -s.num_tokens)
```

说明：`-s.num_tokens` 可让同分时优先删更大的 set，帮助控内存。

### 2.5 新增 `list_region_layer_group()`

```python
def list_region_layer_group(
    self,
    region: CacheRegion,
    layer_id: int | None = None,
    head_group: str | None = None,
) -> list[TokenSet]:
    return self.list_sets(regions=[region], layer_id=layer_id, head_group=head_group)
```

这是 runtime 中最常用的查询形式。

---

## 3. 修改 `src/lifecycle_kv/compression.py`

### 3.1 保留 exact AP，不要删除

当前 exact AP 函数对后续 head profiling 和可解释分析很有用。

### 3.2 新增 Q-K proxy score

新增：

```python
def qk_proxy_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Approximate attention participation without materialized attention maps.

    q: [query_tokens, heads_q, dim]
    k: [key_tokens, heads_k, dim]
    return: [key_tokens]
    """
```

实现：

```python
if q.ndim != 3 or k.ndim != 3:
    raise ValueError(...)

if q.shape[-1] != k.shape[-1]:
    raise ValueError(...)

qn = F.normalize(q.float(), dim=-1)
kn = F.normalize(k.float(), dim=-1)

if qn.shape[1] != kn.shape[1]:
    qn = qn.mean(dim=1, keepdim=True)
    kn = kn.mean(dim=1, keepdim=True)

sim = torch.einsum("qhd,khd->qkh", qn, kn)
scores = sim.max(dim=0).values.mean(dim=-1)
scores = scores.clamp_min(0)
return scores / scores.sum().clamp_min(1e-8)
```

这和 `recall.token_qk_scores()` 很像，后续可以复用一个 shared helper。

### 3.3 新增 `compress_qk_proxy()`

```python
def compress_qk_proxy(
    *,
    set_id: str,
    chunk_id: int,
    frame_ids: list[int],
    layer_id: int,
    head_group: str,
    k: torch.Tensor,
    v: torch.Tensor,
    token_indices: torch.Tensor,
    q: torch.Tensor,
    config: CompressionConfig,
    prompt_summary: torch.Tensor | None = None,
    visual_summary: torch.Tensor | None = None,
    quality_score: float = 1.0,
) -> TokenSet:
    scores = qk_proxy_scores(q, k).to(k.device)
    positions = select_topk_tokens(scores, config.topk, config.min_tokens).to(k.device)
    ...
```

返回 TokenSet 与 `compress_attention_participation()` 一致。

### 3.4 新增 head-aware compression 的入口，但先做 proxy

先加接口，不一定复杂实现：

```python
@dataclass(frozen=True)
class HeadAwareCompressionConfig:
    layout_topk: int = 256
    motion_topk: int = 256
    generic_topk: int = 256
```

新增：

```python
def compress_head_aware_proxy(...):
    if head_group in {"motion", "wave"}:
        use motion_score if provided else qk_proxy_scores
    elif head_group in {"layout", "anchor"}:
        combine qk_proxy + quality
    else:
        qk_proxy
```

第一版不需要完整视频理解 key token compression，但接口要预留。

---

## 4. 修改 `src/lifecycle_kv/recall.py`

### 4.1 扩展 `RecallResult`

当前字段是：

```python
k, v, token_indices, token_scores, token_sets
```

修改为：

```python
@dataclass
class RecallResult:
    k: torch.Tensor | None
    v: torch.Tensor | None
    token_indices: torch.Tensor | None
    token_scores: torch.Tensor | None
    token_sets: list[TokenSet]
    source_set_ids: list[str] | None = None
    source_positions: torch.Tensor | None = None
    set_scores: torch.Tensor | None = None
```

### 4.2 扩展 `RecallConfig`

新增：

```python
max_frame_distance: int | None = None
min_set_score: float | None = None
min_token_score: float | None = None
```

### 4.3 修改 `retrieve_token_sets()`

支持当前 frame 过滤：

```python
def retrieve_token_sets(
    token_sets,
    q,
    *,
    head_group,
    config,
    current_frame: int | None = None,
):
    ...
```

过滤逻辑：

```python
if config.max_frame_distance is not None and current_frame is not None:
    center = mean(token_set.frame_ids)
    if abs(center - current_frame) > config.max_frame_distance:
        continue
```

### 4.4 修改 `recall_tokens()`

新增参数：

```python
step: int | None = None
current_frame: int | None = None
```

更新 access：

```python
for token_set in selected_sets:
    token_set.access_count += 1
    if step is not None:
        token_set.last_used_step = step
```

构造 source map：

```python
source_set_ids = []
source_positions = []
offset = 0
for s in selected_sets:
    n = s.num_tokens
    source_set_ids.extend([s.set_id] * n)
    source_positions.append(torch.arange(n, device=q.device))
...
selected_source_ids = [source_set_ids[int(i)] for i in positions.tolist()]
selected_source_positions = all_source_positions.index_select(0, positions)
```

返回 RecallResult 时带上这些字段。

### 4.5 避免 `TokenSet(set_id=f"recall:{s.set_id}")` 的误导

当前 `token_sets` 返回的是 selected source sets 的 wrapper，实际不是 top-k token 子集。

建议改为：

```python
token_sets=selected_sets
```

真正的 recall view 应由 `ActiveCacheComposer` 统一构造，并带 source map。

---

## 5. 修改 `src/lifecycle_kv/active_cache.py`

### 5.1 新增 compose mode

新增 enum 或字符串参数：

```python
compose_mode: Literal["union", "head_role"] = "union"
```

当前阶段默认 `"union"`，不要默认复杂 head-aware。

### 5.2 recall 候选不要默认包含 anchors

当前：

```python
recall_tokens(anchors + compressed, ...)
```

改成：

```python
recall_candidates = compressed
```

可配置：

```python
include_anchors_in_recall: bool = False
```

### 5.3 `_take_tokens()` 改为按 score 裁剪

当前超预算直接取前 N 个。

改为：

```python
@staticmethod
def _take_tokens(token_sets: list[TokenSet], budget: int | None, score_name: str = "importance") -> list[TokenSet]:
    ...
```

当需要裁剪一个 TokenSet 时：

```python
if token_set.importance_score is not None:
    scores = token_set.importance_score
elif token_set.motion_score is not None:
    scores = token_set.motion_score
else:
    scores = torch.arange(token_set.num_tokens, device=token_set.k.device, dtype=torch.float32)

positions = torch.topk(scores.float(), remaining, largest=True, sorted=True).indices
```

### 5.4 添加 active cache 诊断

`ActiveCacheView` 增加：

```python
region_counts: dict[str, int] | None = None
source_set_ids: list[str] | None = None
```

或者在 `extra` 中输出。

### 5.5 region bias shape 说明

当前 `_region_bias()` 返回 `[active_tokens]` 的 bias。
Self-Forcing attention 接入时需要 reshape 为：

```python
bias = region_bias.view(1, 1, 1, K_active)
```

或者适配现有 attention logits 的 shape：

```text
[B, H, Q, K]
```

在 `ActiveCacheView` docstring 中写清楚，否则 agent 接入时容易维度错。

---

## 6. 修改 `src/lifecycle_kv/motion.py`

### 6.1 新增 token index 到 frame 的函数

```python
def token_indices_to_frames(token_indices: torch.Tensor, frame_seq_length: int) -> torch.Tensor:
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")
    return token_indices.long() // frame_seq_length
```

### 6.2 `combined_motion_score()` 增加 flicker penalty

新增参数：

```python
flicker: torch.Tensor | None = None
```

配置：

```python
flicker_weight: float = 0.20
```

公式：

```python
if flicker is not None:
    score = score - config.flicker_weight * flicker.float().to(dynamic_k.device)
```

### 6.3 dynamic_k_change 支持 previous_k shape 不完全一致

当前要求 current_k 和 previous_k shape 完全一致。

在真实 sliding window 中，previous token 数可能不同。建议新增安全模式：

```python
if current_k.shape != previous_k.shape:
    n = min(current_k.shape[0], previous_k.shape[0])
    current_k = current_k[-n:]
    previous_k = previous_k[-n:]
```

或者保留严格模式，但 runtime 里要保证 shape 对齐。

---

## 7. 修改 `src/lifecycle_kv/head_roles.py` 和 `head_profiler.py`

### 7.1 `head_roles.py` 增加 default role 获取

新增：

```python
def get_head_role(
    roles: dict[tuple[int, int], HeadRole],
    layer_id: int,
    head_id: int,
    default: HeadRole = HeadRole.GENERIC,
) -> HeadRole:
    return roles.get((layer_id, head_id), default)
```

### 7.2 `head_profiler.py` 输出可保存 JSON

给 `HeadProfile` 增加：

```python
def to_dict(self) -> dict:
    ...
```

以及：

```python
def save_profiles(profiles: list[HeadProfile], path: str | Path) -> None:
    ...
```

### 7.3 不要现在依赖 profiler 决定主实验

Phase 2 先用 generic/union。Profiler 只收集数据。Phase 4 才启用。

---

## 8. 修改 `src/lifecycle_kv/instrumentation.py`

### 8.1 新增标准 extra keys 文档

不一定改 dataclass，但加注释或 helper：

```python
def make_bank_extra(...):
    return {
        "active_tokens": ...,
        "recent_tokens": ...,
        "anchor_tokens": ...,
        "compressed_tokens": ...,
        "motion_tokens": ...,
        "recalled_tokens": ...,
        "bank_total_tokens": ...,
        "num_evicted_tokens": ...,
        "compressed_tokens_added": ...,
        "recall_top_sets": ...,
        "recall_top_tokens": ...,
        "fallback": ...,
        "latency_ms": ...,
    }
```

### 8.2 新增 trace summary 脚本

可以放在：

```text
scripts/summarize_lifecache_trace.py
```

功能：

```text
读取 JSONL
统计每 step bank token 数
统计每 layer active/recalled token 数
统计 region_mass 均值
输出 CSV 或 markdown table
```

这是做实验非常关键的脚本。

---

# 四、新增 `src/lifecycle_kv/runtime.py`

这是最重要的新增文件。

## 1. RuntimeConfig

```python
@dataclass
class LifeCacheRuntimeConfig:
    enabled: bool = False
    trace_only: bool = True

    mode: str = "union"  # union / head_role
    enable_layers: tuple[int, ...] | None = None

    compression: str = "qk_proxy"  # qk_proxy / attention_participation
    compression_topk: int = 512
    compression_min_tokens: int = 1

    recall_enabled: bool = False
    recall_top_sets: int = 4
    recall_top_tokens: int = 256
    max_frame_distance: int | None = None

    anchor_enabled: bool = False
    fixed_anchor_enabled: bool = False
    dynamic_anchor_enabled: bool = False
    anchor_budget: int = 256

    motion_enabled: bool = False
    motion_topk: int = 256

    region_bias_beta: float = 0.0

    frame_seq_length: int = 1560
    trace_path: str | None = None
```

## 2. LifeCacheRuntime

```python
class LifeCacheRuntime:
    def __init__(self, config: LifeCacheRuntimeConfig):
        self.config = config
        self.bank = TokenSetBank(...)
        self.composer = ActiveCacheComposer(...)
        self.trace = CacheTraceWriter(config.trace_path) if config.trace_path else None
        self.step = 0
        self.previous_k_by_layer = {}
```

## 3. `should_enable_layer()`

```python
def should_enable_layer(self, layer_id: int) -> bool:
    if not self.config.enabled:
        return False
    if self.config.enable_layers is None:
        return True
    return layer_id in self.config.enable_layers
```

## 4. `trace_event()`

```python
def trace_event(self, *, step, layer_id, head_id, event, kv_shape=None, extra=None):
    if self.trace is None:
        return
    self.trace.write(CacheTraceEvent(...))
```

## 5. `on_kv_evicted()`

```python
def on_kv_evicted(
    self,
    *,
    layer_id: int,
    head_group: str,
    evicted_k: torch.Tensor,
    evicted_v: torch.Tensor,
    token_indices: torch.Tensor,
    q_current: torch.Tensor,
    chunk_id: int,
    frame_ids: list[int],
):
    if not self.config.enabled:
        return None

    if self.config.trace_only and not self.config.compression:
        return None

    token_set = compress_qk_proxy(
        set_id=f"compressed:L{layer_id}:C{chunk_id}:{self.step}",
        chunk_id=chunk_id,
        frame_ids=frame_ids,
        layer_id=layer_id,
        head_group=head_group,
        k=evicted_k,
        v=evicted_v,
        token_indices=token_indices,
        q=q_current,
        config=CompressionConfig(
            topk=self.config.compression_topk,
            min_tokens=self.config.compression_min_tokens,
            region=CacheRegion.COMPRESSED,
        ),
    )

    self.bank.add(token_set)

    self.trace_event(... compressed_tokens_added=token_set.num_tokens ...)
    return token_set
```

## 6. `compose_active_cache()`

```python
def compose_active_cache(
    self,
    *,
    layer_id: int,
    q: torch.Tensor,
    native_recent_k: torch.Tensor,
    native_recent_v: torch.Tensor,
    token_indices: torch.Tensor,
    head_group: str = "generic",
    role: HeadRole = HeadRole.GENERIC,
):
    if not self.config.enabled or self.config.trace_only or not self.config.recall_enabled:
        return native_recent_k, native_recent_v, None

    recent_set = TokenSet(
        set_id=f"recent:L{layer_id}:S{self.step}",
        chunk_id=self.step,
        frame_ids=[],
        layer_id=layer_id,
        head_group=head_group,
        k=native_recent_k,
        v=native_recent_v,
        token_indices=token_indices,
        k_summary=native_recent_k.float().mean(dim=0),
        region=CacheRegion.RECENT,
    )

    compressed = self.bank.list_sets(
        regions=[CacheRegion.COMPRESSED],
        layer_id=layer_id,
    )

    anchors = self.bank.list_sets(
        regions=[CacheRegion.ANCHOR],
        layer_id=layer_id,
    )

    motion = self.bank.list_sets(
        regions=[CacheRegion.MOTION],
        layer_id=layer_id,
    )

    view = self.composer.compose(
        q=q,
        role=role,
        head_group=head_group,
        recent=[recent_set],
        anchors=anchors,
        compressed=compressed,
        motion=motion,
    )

    if view.k is None:
        return native_recent_k, native_recent_v, None

    return view.k, view.v, view
```

## 7. `advance_step()`

```python
def advance_step(self):
    self.step += 1
```

---

# 五、Self-Forcing 接入指南

当前 `docs/16` 已经指定接入点：`third_party/Self-Forcing/wan/modules/causal_model.py` 中的 `CausalWanSelfAttention.forward`，位置在 RoPE 后、attention 前。

## 1. 先不要改输出

先加：

```python
if lifecache_runtime is not None and lifecache_runtime.config.enabled:
    lifecache_runtime.trace_event(...)
```

不要改 K/V。

## 2. 需要传入 runtime

可能需要从 pipeline / model forward 一路传：

```python
lifecache_runtime=None
```

到 attention forward。

建议最低侵入式：

```python
self.lifecache_runtime = getattr(model, "lifecache_runtime", None)
```

或者在 attention module 初始化时挂载。

## 3. q/k/v shape 适配

Self-Forcing/Wan 常见 layout 可能是：

```text
[B, tokens, heads, dim]
```

而 TokenSet 需要：

```text
[tokens, heads, dim]
```

batch size 1 时：

```python
k_life = k[0] if k.ndim == 4 else k
v_life = v[0] if v.ndim == 4 else v
q_life = q[0] if q.ndim == 4 else q
```

注意：真实变量名可能是 `roped_query / roped_key / v`，要以实际代码为准。

## 4. Phase 0 patch

在 attention forward 中：

```python
if lifecache_runtime is not None and lifecache_runtime.config.enabled:
    lifecache_runtime.trace_event(
        step=lifecache_runtime.step,
        layer_id=layer_id,
        head_id=None,
        event="attention_forward",
        kv_shape=tuple(k_life.shape),
        extra={
            "q_shape": tuple(q_life.shape),
            "v_shape": tuple(v_life.shape),
            "current_start": current_start,
            "trace_only": lifecache_runtime.config.trace_only,
        },
    )
```

## 5. Phase 1 eviction hook

先不要依赖完美 eviction。第一版可以用“当前 native cache 超过 recent window 时，左侧滑出的 token 区间”作为 evicted。

伪代码：

```python
if lifecache_runtime is not None and should_compress:
    evicted_k = old_cache_k[:, evict_start:evict_end]  # adapt shape
    evicted_v = old_cache_v[:, evict_start:evict_end]
    token_indices = torch.arange(evict_start, evict_end, device=evicted_k.device)

    lifecache_runtime.on_kv_evicted(
        layer_id=layer_id,
        head_group="generic",
        evicted_k=evicted_k[0],
        evicted_v=evicted_v[0],
        token_indices=token_indices,
        q_current=q_life,
        chunk_id=current_chunk,
        frame_ids=...
    )
```

如果暂时找不到 precise eviction 位置，先在 cache refresh 后对“即将不在 local window 内的 old tokens”做 pseudo-eviction trace，不改变模型。

## 6. Phase 2 recall patch

```python
if lifecache_runtime is not None and lifecache_runtime.config.recall_enabled:
    active_k, active_v, view = lifecache_runtime.compose_active_cache(
        layer_id=layer_id,
        q=q_life,
        native_recent_k=native_k_life,
        native_recent_v=native_v_life,
        token_indices=native_token_indices,
        head_group="generic",
        role=HeadRole.GENERIC,
    )

    # restore batch dim
    active_k = active_k.unsqueeze(0)
    active_v = active_v.unsqueeze(0)

    # use active_k/active_v in attention
```

先只开后几层：

```python
enable_layers = tuple(range(num_layers - 6, num_layers))
```

---

# 六、测试指南

## 1. 单元测试

新增：

```text
tests/test_tokenset.py
tests/test_bank.py
tests/test_compression.py
tests/test_recall.py
tests/test_active_cache.py
tests/test_runtime.py
```

### `test_tokenset.py`

测试：

```text
1. k/v shape mismatch 抛错。
2. token_indices 长度不匹配抛错。
3. clone_with_tokens 后 token 数正确。
4. source_positions 正确传播。
```

### `test_bank.py`

测试：

```text
1. add/list/total_tokens 正确。
2. max_sets prune 生效。
3. max_tokens prune 生效。
4. mark_used 更新 access_count 和 last_used_step。
5. dedup 能删除相似 TokenSet。
```

### `test_compression.py`

测试：

```text
1. attention_participation_scores 支持 2D/3D/4D。
2. compress_attention_participation 输出 top-k TokenSet。
3. qk_proxy_scores 输出 [key_tokens]。
4. compress_qk_proxy 不需要 attention map。
```

### `test_recall.py`

测试：

```text
1. retrieve_token_sets 能按 group_match 选中正确 set。
2. recall_tokens 返回 top_tokens 个 token。
3. source_set_ids 和 source_positions 正确。
4. max_frame_distance 能过滤远历史。
```

### `test_active_cache.py`

测试：

```text
1. generic role 不 recall。
2. layout role 有 anchor + recall + recent。
3. motion role 有 motion + recent + tiny anchor，且无 recall。
4. budget 裁剪按 importance_score top-k。
5. region_bias shape 和 token 数一致。
```

### `test_runtime.py`

测试：

```text
1. trace_only 下 compose 返回 native K/V。
2. on_kv_evicted 能向 bank 添加 compressed TokenSet。
3. recall_enabled 时 compose 返回 active K/V。
4. bank stats 随 evicted 增长。
```

---

# 七、实验配置建议

新增配置文件：

```text
configs/lifecache/lifecache_trace_only.yaml
configs/lifecache/lifecache_compression_only.yaml
configs/lifecache/lifecache_union_recall.yaml
```

## `lifecache_trace_only.yaml`

```yaml
lifecache:
  enabled: true
  trace_only: true
  recall_enabled: false
  compression: none
  trace_path: outputs/lifecache/trace_only.jsonl
```

## `lifecache_compression_only.yaml`

```yaml
lifecache:
  enabled: true
  trace_only: true
  compression: qk_proxy
  compression_topk: 512
  recall_enabled: false
  trace_path: outputs/lifecache/compression_only.jsonl
```

## `lifecache_union_recall.yaml`

```yaml
lifecache:
  enabled: true
  trace_only: false
  mode: union
  compression: qk_proxy
  compression_topk: 512
  recall_enabled: true
  recall_top_sets: 4
  recall_top_tokens: 256
  enable_last_n_layers: 6
  region_bias_beta: 0.0
  trace_path: outputs/lifecache/union_recall.jsonl
```

---

# 八、给代码 agent 的完整任务 prompt

你可以直接把下面这段交给 agent：

```text
你正在优化仓库 Anachronism-N/training-free 中的 LifeCache-v1 原型。

请阅读：
- docs/11_lifecache_v1_design.md
- docs/12_lifecache_experiment_plan.md
- docs/16_lifecache_v1_implementation_status.md
- src/lifecycle_kv/*.py

当前目标：
不要继续扩展概念，不要引入 VLM/LLM/entity tracking，不要直接做 split-head attention。请把现有库代码优化成能接入 Self-Forcing 的实验原型。

必须完成以下修改：

1. tokenset.py
   - 给 TokenSet 增加 source_set_id、source_region、source_positions。
   - clone_with_tokens 需要正确传播 source 信息。
   - 增加 to_device() helper。

2. bank.py
   - 增加 BankStats dataclass。
   - 增加 stats()。
   - 增加 dedup(region=None, similarity_threshold=0.985)。
   - 优化 prune priority，考虑 quality、importance、access_count、last_used_step、num_tokens。
   - 增加 list_region_layer_group()。

3. compression.py
   - 保留 attention_participation_scores 和 compress_attention_participation。
   - 新增 qk_proxy_scores(q, k)。
   - 新增 compress_qk_proxy(...)，用于没有 attention map 时的 compression。
   - 预留 compress_head_aware_proxy(...) 接口，可先简单实现。

4. recall.py
   - RecallResult 增加 source_set_ids、source_positions、set_scores。
   - RecallConfig 增加 max_frame_distance、min_set_score、min_token_score。
   - recall_tokens 增加 step 和 current_frame 参数。
   - recall 时更新 access_count 和 last_used_step。
   - 返回 top-k recalled tokens 对应的 source map。
   - 不要在 recall_tokens 里构造误导性的 recall TokenSet；由 ActiveCacheComposer 构造 recall view。

5. active_cache.py
   - 增加 compose_mode: union/head_role，默认 union。
   - 默认 recall candidates 只使用 compressed，不要 anchors + compressed。
   - _take_tokens 超预算时按 importance_score 或 motion_score top-k，而不是直接取前 N。
   - ActiveCacheView 增加 region_counts/source 信息。
   - 明确 region_bias 是 [K]，接入 attention 时需要 reshape 到 [1,1,1,K]。

6. motion.py
   - 增加 token_indices_to_frames(token_indices, frame_seq_length)。
   - combined_motion_score 增加 flicker penalty 参数。
   - dynamic_k_change 支持可选安全对齐，或在 docstring 中说明必须同 shape。

7. instrumentation.py
   - 增加 make_trace_extra helper，标准化 active_tokens/recalled_tokens/bank_total_tokens/compressed_tokens_added/latency_ms 等字段。
   - 新增 scripts/summarize_lifecache_trace.py，读取 JSONL 并输出 region/token/bank 统计。

8. 新增 src/lifecycle_kv/runtime.py
   - 定义 LifeCacheRuntimeConfig。
   - 定义 LifeCacheRuntime。
   - 支持 trace_only、compression-only、union recall 三种模式。
   - 提供 on_kv_evicted()、compose_active_cache()、trace_event()、advance_step()、stats()。
   - 不依赖 Self-Forcing 具体类，只接收 q/k/v tensor。

9. 修改 __init__.py
   - 导出 LifeCacheRuntime 和 LifeCacheRuntimeConfig。
   - 导出新增 compression/retrace helper。

10. 新增 tests
   - test_tokenset.py
   - test_bank.py
   - test_compression.py
   - test_recall.py
   - test_active_cache.py
   - test_runtime.py

验证要求：
- python -m compileall src tests 通过。
- 所有不依赖 torch 外部模型的单元测试通过。
- 不修改 Self-Forcing 输出路径。
- 不默认启用 recall 或 region bias。

下一步不要求直接接入 Self-Forcing，但 runtime API 必须能让 Self-Forcing 的 CausalWanSelfAttention.forward 在 RoPE 后、attention 前调用。
```

---

# 九、验收标准

这次优化完成后，应该能达到以下状态：

```text
1. LifeCache 核心库可以被单元测试覆盖。
2. 不需要真实 Self-Forcing 就能测试 compression / recall / active cache。
3. Runtime API 足够清晰，可以接入 Self-Forcing。
4. Trace schema 固定，便于后续分析。
5. TokenSetBank 不会无限增长。
6. Recall 能追踪 source set。
7. active cache 可以先以 union 模式安全运行。
```

最关键的验收命令：

```bash
python -m compileall src tests
pytest tests/test_tokenset.py tests/test_bank.py tests/test_compression.py tests/test_recall.py tests/test_active_cache.py tests/test_runtime.py
```

---

# 十、下一轮迭代之后再做什么

代码 agent 完成上述优化后，再进入真正 Self-Forcing patch：

```text
1. 加 lifecache_enable / lifecache_trace_only config。
2. pipeline 初始化 LifeCacheRuntime。
3. CausalWanSelfAttention.forward 只 trace，不改输出。
4. 再做 compression-only。
5. 最后只在 last 6 layers 开 union recall。
```

不要跨步。现在这个项目最大的风险不是 idea 不够，而是一次性打开太多机制导致不可 debug。先让 trace 和 compression 变成稳定基础设施，再谈 recall 增益。
