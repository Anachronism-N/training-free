# LifeCache 下一阶段实验推进文档

> 目标：把当前 LifeCache 从“有代码接入但几乎无提升”的状态，推进到“能明确定位瓶颈、验证 RoPE-safe raw-K recall 是否有效、再逐步加入 AdaMem-style rho/soft-forget”的实验阶段。  
> 当前优先级：**诊断 > 修正数据路径 > 小规模验证 > 再做方法增强**。  
> 不要继续盲目堆 memory component。

---

## 0. 当前判断

当前 LifeCache 效果几乎没有提升，不能直接说明 idea 无效。更可能是以下路径没有真正成立：

```text
clean context evicted raw K/V
  -> real-query compression
  -> RoPE-safe bank storage
  -> content-aware recall
  -> legal relative RoPE remap
  -> active attention actually uses recalled tokens
  -> video quality improves
```

现在需要逐段验证这条链路。

当前最可能的问题：

```text
P0. recall 没有真的产生有效 attention mass；
P1. compression 选出来的 token 不对，因为 query 不是实际 query；
P2. captured K 是 post-RoPE old-position K，远距离 recall 位置非法；
P3. capture 发生在 noisy denoising 或被覆盖，不是 clean-context-only；
P4. near-only recall 与 native recent cache 重叠太多，本来就不应期待明显提升；
P5. 当前 prompt suite 可能没有强制 A-B-C-A / long recall 场景。
```

---

## 1. 本阶段总目标

本阶段不追求立刻 SOTA，而是回答 6 个问题：

```text
Q1. LifeCache 在 trace/compression-only 模式是否完全不改变输出？
Q2. bank 是否真的保存了 clean context 中滑出的 token？
Q3. compression 是否由真实 query 驱动，而不是 evicted_k.mean 自相似？
Q4. recalled tokens 是否真的进入 active cache，并被 attention 使用？
Q5. post-RoPE far recall 是否导致质量下降？
Q6. pre-RoPE + relative-clamp remap 是否能稳定地使用远历史？
```

只有 Q1-Q6 被清楚回答后，才值得继续加入 AdaMem 的 `rho / z / soft forget / write_or_merge`。

---

## 2. 代码修改总览

下一轮代码修改分 5 个 block。

| Block | 目的 | 是否改变输出 | 优先级 |
|---|---|---:|---:|
| B0 Trace diagnostics | 让所有失败可定位 | 否 | P0 |
| B1 Clean-only capture | 只保存 clean context memory | 否 | P0 |
| B2 Real-query compression | 用真实 q 选 token | 否 | P0 |
| B3 RoPE metadata + near-only filter | 禁止非法 post-RoPE far recall | 可选 | P1 |
| B4 Pre-RoPE bank + remap | 核心有效 recall 路径 | 是 | P1 |
| B5 AdaMem rho/z/forget | 提升 memory 选择与遗忘 | 是 | P2 |

本阶段最少应完成 B0-B4。B5 是 RoPE-safe recall 成功后再做。

---

## 3. B0：Trace diagnostics

### 3.1 目标

每次实验结束后，不能只看视频主观质量，必须能从 trace 判断：

```text
memory 是否写入；
recall 是否发生；
recall 距离多远；
recall 的 K 是 pre-RoPE 还是 post-RoPE；
attention 是否真的使用 recalled tokens；
输出无提升是因为没召回、召回错、还是召回被忽略。
```

---

### 3.2 需要新增/规范的 trace 字段

在 `CacheTraceEvent.extra` 中标准化以下字段。

#### Capture 相关

```text
capture_enabled: bool
capture_reason: str                  # clean_context / denoising / disabled
capture_step: int
layer_id: int
current_start_frame: int
num_evicted_tokens: int
num_evicted_frames_est: int
payload_type: str                     # dict / legacy_tuple
has_q_pre_rope: bool
has_q_post_rope: bool
has_k_pre_rope: bool
has_k_post_rope: bool
```

#### Compression 相关

```text
compression_mode: str                 # qk_proxy / attention_participation / none
q_source: str                         # actual_q_pre_rope / actual_q_post_rope / evicted_k_mean / unknown
compressed_tokens_added: int
compression_topk: int
qk_score_mean: float
qk_score_top: float
qk_score_std: float
selected_token_index_min: int
selected_token_index_max: int
selected_frame_min: int | null
selected_frame_max: int | null
```

#### Bank 相关

```text
bank_total_sets: int
bank_total_tokens: int
bank_pre_rope_sets: int
bank_post_rope_sets: int
bank_tokens_by_region: dict
bank_tokens_by_layer: dict
bank_mean_rho: float | null
bank_min_rho: float | null
bank_max_rho: float | null
```

#### Recall 相关

```text
recall_enabled: bool
recalled_tokens: int
recall_top_sets: int
recall_top_tokens: int
recall_candidate_sets: int
recall_selected_sets: int
recall_rope_mode: str                 # pre_rope / post_rope / mixed / none
recalled_source_distance_mean: float | null
recalled_source_distance_max: float | null
recalled_source_frame_min: int | null
recalled_source_frame_max: int | null
recalled_qk_score_mean: float | null
recalled_qk_score_top: float | null
recalled_source_set_ids_sample: list[str]
```

#### Active attention 相关

```text
active_tokens: int
recent_tokens: int
anchor_tokens: int
compressed_tokens: int
motion_tokens: int
recall_tokens: int
region_counts: dict
region_bias_beta: float
region_bias_mean: float | null
region_bias_min: float | null
region_bias_max: float | null
attention_mass_to_recent: float | null
attention_mass_to_recall: float | null
attention_mass_to_anchor: float | null
attention_mass_to_motion: float | null
```

注意：如果当前 attention kernel 不返回 attention map，则 `attention_mass_to_*` 可以先为空，但必须记录 Q-K proxy mass 或 sampled attention diagnostic。

---

### 3.3 需要新增 summarizer

新增脚本：

```text
scripts/summarize_lifecache_trace.py
```

输入：

```bash
python scripts/summarize_lifecache_trace.py \
  --trace cache_trace.jsonl \
  --out-md outputs/lifecache/trace_summary.md \
  --out-csv outputs/lifecache/trace_summary.csv
```

输出 markdown 至少包含：

```text
1. Basic run information
2. Events count by type
3. Enabled layers
4. Bank growth over time
5. Compression summary
6. Recall summary
7. Rope-mode summary
8. Distance summary
9. Per-layer active token summary
10. Failure warnings
```

### 3.4 Warning 规则

脚本中加入自动 warning：

```text
WARN_NO_BANK_GROWTH:
  bank_total_tokens 始终为 0

WARN_NO_RECALL:
  recall_enabled=True 但 recalled_tokens 总和为 0

WARN_RECALL_TOO_NEAR:
  recalled_source_distance_mean < local_attn_size / 2

WARN_FAR_POST_ROPE:
  post_rope recalled tokens 的 distance > max_post_rope_frame_distance

WARN_WRONG_QUERY:
  q_source == evicted_k_mean

WARN_ALL_LAYERS_ENABLED:
  enable_last_n_layers=6 但 trace 出现所有 30 层

WARN_TRACE_ONLY_CHANGED_OUTPUT:
  需要外部比较输出 hash 或 metric，可先人工记录
```

---

## 4. B1：Clean-context-only capture

### 4.1 为什么要做

长期 memory 应该来自 clean context refresh，而不是 denoising loop 中间噪声状态。

当前风险：

```text
noisy denoising step 触发 eviction；
_lifecache_evicted 被后续 step 覆盖；
MemoryBank 保存的不是稳定 clean K/V。
```

---

### 4.2 Runtime 修改

修改文件：

```text
src/lifecycle_kv/runtime.py
```

新增字段：

```python
self.capture_enabled: bool = False
self.capture_reason: str = ""
```

新增方法：

```python
def begin_capture(self, reason: str) -> None:
    self.capture_enabled = True
    self.capture_reason = reason
    self.trace_event(
        layer_id=-1,
        event="begin_capture",
        extra={"capture_reason": reason, "step": self.step},
    )


def end_capture(self) -> None:
    self.trace_event(
        layer_id=-1,
        event="end_capture",
        extra={"capture_reason": self.capture_reason, "step": self.step},
    )
    self.capture_enabled = False
    self.capture_reason = ""
```

### 4.3 Self-Forcing pipeline 修改

修改文件：

```text
third_party/Self-Forcing/pipeline/causal_inference.py
```

在 clean context refresh 前：

```python
if self.lifecache_manager is not None:
    self.lifecache_manager.runtime.begin_capture("clean_context")
```

运行 clean context refresh：

```python
self.generator(
    noisy_image_or_video=denoised_pred,
    conditional_dict=conditional_dict,
    timestep=context_timestep,
    kv_cache=self.kv_cache1,
    crossattn_cache=self.crossattn_cache,
    current_start=current_start_frame * self.frame_seq_length,
)
```

之后：

```python
if self.lifecache_manager is not None:
    self.lifecache_manager.runtime.end_capture()
```

### 4.4 Self-Forcing attention 修改

修改文件：

```text
third_party/Self-Forcing/wan/modules/causal_model.py
```

当前 eviction capture 条件应改为：

```python
if (
    lifecache_manager is not None
    and lifecache_manager.runtime.capture_enabled
    and num_evicted_tokens > 0
    and sink_tokens == 0
):
    ...
```

不要在 denoising loop 中捕获。

### 4.5 从单 payload 改为 list

当前如果使用：

```python
kv_cache["_lifecache_evicted"] = payload
```

改为：

```python
kv_cache.setdefault("_lifecache_evicted_list", []).append(payload)
```

pipeline 处理：

```python
payloads = cache.pop("_lifecache_evicted_list", [])
for payload in payloads:
    self.lifecache_manager.runtime.on_kv_evicted_payload(payload)
```

---

## 5. B2：Real-query compression

### 5.1 为什么要做

不能继续使用：

```python
q_proxy = evicted_k.mean(dim=0, keepdim=True)
```

这会使 compression 退化为 evicted K 自相似，而不是“当前 query 需要什么历史”。

### 5.2 Payload 格式

在 attention forward 捕获 evicted token 时，payload 应该是 dict：

```python
payload = {
    "layer_id": block_index,
    "evicted_k_pre_rope": evicted_k_pre_rope,      # [T,H,D], preferred
    "evicted_k_post_rope": evicted_k_post_rope,    # [T,H,D], fallback
    "evicted_v": evicted_v,                        # [T,H,D]
    "q_pre_rope": q_pre_rope,                      # [Q,H,D]
    "q_post_rope": roped_query[0],                 # [Q,H,D]
    "token_indices": token_indices,                # [T]
    "frame_positions": frame_positions,            # [T]
    "current_start_frame": current_start_frame,
    "capture_reason": lifecache_manager.runtime.capture_reason,
}
```

### 5.3 如何获得 pre/post-RoPE tensors

在 `causal_model.py` 中，通常存在：

```text
q                 # pre-RoPE query
k                 # pre-RoPE key
v                 # value
roped_query       # post-RoPE query
roped_key         # post-RoPE key
```

需要根据实际变量作用域确认。如果 evicted tokens 已经在 `kv_cache` 中而没有保存 pre-RoPE K，则第一版可以：

```text
evicted_k_post_rope = kv_cache["k"] old slice
q_post_rope = roped_query[0]
```

但 v2 的目标是改 cache write path：额外维护或捕获 pre-RoPE K，使：

```text
evicted_k_pre_rope 可用
q_pre_rope 可用
```

### 5.4 Runtime 新增接口

新增：

```python
def on_kv_evicted_payload(self, payload: dict) -> TokenSet | None:
    ...
```

逻辑：

```python
if payload has evicted_k_pre_rope and q_pre_rope:
    k = payload["evicted_k_pre_rope"]
    q = payload["q_pre_rope"]
    rope_mode = "pre_rope"
    q_source = "actual_q_pre_rope"
elif payload has evicted_k_post_rope and q_post_rope:
    k = payload["evicted_k_post_rope"]
    q = payload["q_post_rope"]
    rope_mode = "post_rope"
    q_source = "actual_q_post_rope"
else:
    fallback or skip
```

Then call:

```python
token_set = compress_qk_proxy(
    k=k,
    v=payload["evicted_v"],
    q=q,
    token_indices=payload["token_indices"],
    ...
)
```

Set metadata:

```python
token_set.rope_mode = rope_mode
token_set.frame_positions = payload["frame_positions"]
token_set.source_start_frame = int(payload["frame_positions"].min())
token_set.capture_step = self.step
```

Trace:

```text
q_source
rope_mode
qk_score_mean/top/std
selected_frame_min/max
```

---

## 6. B3：RoPE metadata + near-only safety

### 6.1 TokenSet 修改

文件：

```text
src/lifecycle_kv/tokenset.py
```

新增字段：

```python
rope_mode: str = "post_rope"  # "pre_rope" or "post_rope"
frame_positions: Optional[torch.Tensor] = None
source_start_frame: Optional[int] = None
capture_step: int = -1
rho: float = 1.0              # for later AdaMem-style retention
z: Optional[torch.Tensor] = None
```

`__post_init__`：

```python
if self.rope_mode not in {"pre_rope", "post_rope"}:
    raise ValueError(...)

if self.frame_positions is not None:
    if self.frame_positions.ndim != 1:
        raise ValueError(...)
    if self.frame_positions.numel() != self.k.shape[0]:
        raise ValueError(...)
```

`clone_with_tokens()`：

```python
frame_positions = None
if self.frame_positions is not None:
    frame_positions = self.frame_positions.index_select(0, token_positions)
```

`to_device()` 同步移动 `frame_positions` 和 `z`。

### 6.2 Recall filtering

文件：

```text
src/lifecycle_kv/recall.py
```

`RecallConfig` 增加：

```python
rope_safe_recall: bool = True
allow_post_rope_recall: bool = False
max_post_rope_frame_distance: int = 21
```

过滤逻辑：

```python
def is_rope_safe(ts, current_frame, config):
    if ts.rope_mode == "pre_rope":
        return True
    if ts.rope_mode == "post_rope":
        if not config.allow_post_rope_recall:
            return False
        if current_frame is None or ts.frame_positions is None:
            return False
        dist = abs(current_frame - ts.frame_positions.float().mean().item())
        return dist <= config.max_post_rope_frame_distance
```

### 6.3 near-only config

新增：

```text
configs/lifecache/lifecache_recall_near_only.yaml
```

内容：

```yaml
lifecache:
  enabled: true
  trace_only: false
  mode: union

  compression: qk_proxy
  compression_topk: 512
  compression_min_tokens: 64

  recall_enabled: true
  recall_top_sets: 2
  recall_top_tokens: 128
  max_frame_distance: 21

  rope_safe_recall: true
  allow_post_rope_recall: true
  max_post_rope_frame_distance: 21

  anchor_enabled: false
  fixed_anchor_enabled: false
  dynamic_anchor_enabled: false
  motion_enabled: false

  region_bias_beta: 0.0
  enable_last_n_layers: 6
  trace_path: outputs/lifecache/recall_near_only.jsonl
  record_latency: true

  bank_max_compressed_sets: 64
  bank_max_compressed_tokens: 65536
```

---

## 7. B4：Pre-RoPE bank + relative-clamp remap

### 7.1 为什么必须做

near-only recall 可能安全但冗余。真正可能产生提升的是远距离 recall。远距离 recall 必须使用 pre-RoPE K + read-time remap。

### 7.2 不要把 Wan RoPE 放进 generic runtime

保持：

```text
src/lifecycle_kv/runtime.py = model-agnostic
third_party/Self-Forcing/scripts/lifecache_manager.py = Self-Forcing-specific adapter
```

### 7.3 新增 RopeAdapter

文件：

```text
third_party/Self-Forcing/scripts/lifecache_manager.py
```

新增：

```python
class SelfForcingRopeAdapter:
    def __init__(self, frame_seq_length: int, temporal_range: int = 21, split_recent: int = 4):
        self.frame_seq_length = frame_seq_length
        self.temporal_range = temporal_range
        self.split_recent = split_recent

    def map_frame_positions(
        self,
        frame_positions: torch.Tensor,
        *,
        current_start_frame: int,
        num_new_frames: int,
    ) -> torch.Tensor:
        # frame_positions: [T]
        # return t_pos per frame or per token, mapped into [0, TR-1]
        newest = current_start_frame
        rel = (newest - frame_positions).clamp(0, self.temporal_range - 1)
        if self.split_recent > 0:
            is_recent = rel < self.split_recent
            rel_mapped = torch.where(
                is_recent,
                rel,
                torch.full_like(rel, self.temporal_range - 1),
            )
        else:
            rel_mapped = rel
        t_pos = (self.temporal_range - 1) - rel_mapped
        return t_pos
```

### 7.4 How to apply RoPE

In `causal_model.py`, the existing functions are available in the same module:

```text
causal_rope_apply
causal_rope_apply_pos
```

For recalled K with token-level frame positions, prefer `causal_rope_apply_pos` if it supports per-frame positions. If it expects frame-level grid, construct `t_pos` for recalled frames.

Pseudo-flow inside attention forward:

```python
active_k, active_v, view = rt.compose_active_cache(...)

if view contains pre_rope recalled tokens:
    recalled_slice = locate CacheRegion.RECALL positions
    recalled_k_raw = active_k[recalled_slice]
    frame_positions = view.frame_positions[recalled_slice]
    t_pos = rope_adapter.map_frame_positions(
        frame_positions,
        current_start_frame=current_start_frame,
        num_new_frames=num_new_frames,
    )
    recalled_k_roped = causal_rope_apply_pos(...)
    active_k[recalled_slice] = recalled_k_roped
```

If this is too complex at first, implement a simpler remap:

```text
all far recalled tokens -> temporal position 0
recent recalled tokens -> temporal position near current window
```

This is less accurate but should avoid unbounded RoPE phase.

### 7.5 Config

新增：

```text
configs/lifecache/lifecache_pre_rope_remap.yaml
```

```yaml
lifecache:
  enabled: true
  trace_only: false
  mode: union

  compression: qk_proxy
  compression_topk: 512
  compression_min_tokens: 64

  recall_enabled: true
  recall_top_sets: 4
  recall_top_tokens: 128
  max_frame_distance: null

  rope_safe_recall: true
  allow_post_rope_recall: false
  rope_remap_policy: relative_clamp
  max_post_rope_frame_distance: 21

  anchor_enabled: false
  motion_enabled: false
  region_bias_beta: 0.0
  enable_last_n_layers: 6
  trace_path: outputs/lifecache/pre_rope_remap.jsonl
  record_latency: true
```

---

## 8. B5：AdaMem-style rho / descriptor / soft forget

Only start this block after B4 works.

### 8.1 Why

AdaMem's useful ideas:

```text
raw K at write time
content-addressed recall
soft retention rho
log(rho) attention bias
write/merge/soft forget
```

For LifeCache, adopt the training-free subset first.

### 8.2 TokenSet additions

Already planned:

```python
rho: float = 1.0
z: torch.Tensor | None = None
```

Descriptor:

```python
z = normalize(k_raw.float().mean(dim=(0, 1)))  # [D]
```

### 8.3 Update retention

File:

```text
src/lifecycle_kv/bank.py
```

Add:

```python
def update_retention(
    self,
    current_descriptor: torch.Tensor,
    *,
    mu0: float = 0.001,
    beta: float = 0.01,
    xi: float = 0.3,
    rho_min: float = 1e-4,
) -> None:
    for s in self._sets.values():
        if s.z is None:
            continue
        drift = 1.0 - cosine(s.z, current_descriptor)
        mu = mu0 + beta * max(0.0, drift - xi)
        s.rho = max(rho_min, float(s.rho) * math.exp(-mu))
```

### 8.4 Write or merge

Add:

```python
def write_or_merge(
    self,
    token_set: TokenSet,
    *,
    theta_merge: float = 0.95,
    motion_low_threshold: float = 0.2,
) -> str:
    # return "merged" or "added"
```

First version: do not merge K/V tensors if token count differs.

Simpler merge:

```python
if sim >= theta_merge and token_set.motion_score_set <= motion_low_threshold:
    old.z = normalize((old.rho * old.z + token_set.rho * token_set.z) / (old.rho + token_set.rho))
    old.rho = min(1.0, old.rho + 0.1)
    old.access_count += 1
    return "merged"
else:
    self.add(token_set)
    return "added"
```

### 8.5 Recall score update

Modify set-level scoring:

```text
S_set =
    0.45 * cos(Q_bar, K_bar)
  + 0.20 * cos(z_current, z_set)
  + 0.15 * log(rho + eps)
  + 0.10 * usage
  + 0.10 * distance_score
  - 1.00 * rope_risk
```

Modify token-level score:

```text
S_token =
    0.70 * QK
  + 0.20 * importance
  + 0.10 * log(rho + eps)
```

Attention bias:

```python
memory_bias = lambda_rho * log(rho + eps)
```

Keep `lambda_rho` small first:

```text
lambda_rho = 0.05 or 0.1
```

---

## 9. Experiment suite

### 9.1 Prompt set

Use a small controlled set first.

#### Group A: native continuation

Purpose: ensure LifeCache does not damage normal generation.

```text
A1. A woman walks slowly along a city street at sunset, camera following smoothly.
A2. A dog runs across a green field, then slows down near a tree.
A3. A dancer performs continuous spinning motions on a stage.
```

#### Group B: scene revisit A-B-A

Purpose: test long recall.

```text
B1. A woman in a yellow coat stands in a small kitchen with blue cabinets and a red cup on a wooden table. She walks outside into a garden. Later she returns to the same kitchen and stands beside the same red cup.

B2. A white dog with a red collar runs through a park, disappears behind trees, then reappears near the same fountain still wearing the red collar.

B3. A robot walks in a clean white laboratory, moves into a dark hallway, then returns to the same laboratory with the same blue control panel.
```

#### Group C: hard switch

Purpose: test forgetting / avoid wrong recall.

```text
C1. A robot walks in a clean white laboratory. The scene suddenly cuts to a crowded night market with neon signs.

C2. A red kitchen scene changes to a blue ocean beach, with no return to the kitchen.
```

#### Group D: distractor scenes

Purpose: test content-addressing.

```text
D1. A person enters a red kitchen, then a blue kitchen, then returns to the red kitchen.

D2. A cat sits on a striped sofa, later appears on a similar striped bed, then returns to the original sofa.
```

### 9.2 Lengths

Run in this order:

```text
30s / 120 latent frames first
60s only after 30s is stable
120s only after 60s is stable
```

### 9.3 Seeds

Initial debugging:

```text
seed = 0 only
```

After stable:

```text
seed = 0, 1, 2
```

---

## 10. Experiment matrix

### Stage 1: safety and diagnosis

| ID | Config | Goal | Output expected |
|---|---|---|---|
| E0 | native SF | baseline | normal |
| E1 | trace-only | integration safety | same as E0 |
| E2 | compression-clean-only | bank capture safety | same as E0, bank grows |
| E3 | near-only recall | safe post-RoPE recall | stable, maybe little gain |
| E4 | far post-RoPE recall | RoPE failure stress | likely degradation |

### Stage 2: core v2

| ID | Config | Goal | Output expected |
|---|---|---|---|
| E5 | pre-RoPE + remap recall | test legal long recall | stable, maybe gain on B/D prompts |
| E6 | E5 + real-query compression | test selection quality | better than E5 if E5 used fallback |
| E7 | E6 + small rho bias | test AdaMem soft retention | less wrong recall |

### Stage 3: method enhancement

| ID | Config | Goal | Output expected |
|---|---|---|---|
| E8 | E7 + write_or_merge | budget efficiency | same/better with fewer tokens |
| E9 | E7 + per-head region bias | reduce static/motion interference | better dynamics |
| E10 | E9 + fixed anchor | identity/scene stability | better B prompts |
| E11 | E10 + motion cache | dynamic continuity | better A3/motion prompts |

---

## 11. Metrics and logging

### 11.1 Required numeric metrics

For every run:

```text
runtime seconds
peak GPU memory
bank_total_tokens final
mean recalled_tokens per enabled layer
mean recall distance
number of pre_rope recalled tokens
number of post_rope recalled tokens
```

### 11.2 Video quality metrics

Use available metrics first:

```text
VBench if already integrated
subject consistency
background consistency
dynamic degree
temporal flicker
CLIP text-video similarity if available
```

If metrics are slow, use small prompt subset and manual review first.

### 11.3 Qualitative checklist

For each video, record:

```text
Does it darken after 10s?
Does subject freeze?
Does identity drift?
Does scene revisit correctly recover old background/object?
Does wrong old scene leak into hard switch?
Does motion become static after recall begins?
```

Use 0/1/2 score:

```text
0 = bad / absent
1 = partial
2 = good
```

---

## 12. Decision tree

### Case 1: trace-only changes output

Conclusion:

```text
Integration is intrusive.
```

Action:

```text
Do not continue experiments.
Find state mutation, random seed change, cache write, or attention path difference.
```

---

### Case 2: compression-only changes output

Conclusion:

```text
Compression path is accidentally affecting generation.
```

Action:

```text
Ensure on_kv_evicted only writes bank and trace.
Ensure active attention path is not called when recall_enabled=False.
```

---

### Case 3: bank does not grow

Conclusion:

```text
Eviction capture is not triggered or clean-only capture window is wrong.
```

Action:

```text
Check capture_enabled trace.
Check num_evicted_tokens.
Check local_attn_size and max cache size.
Check whether full-attention mode prevents eviction.
```

---

### Case 4: bank grows but recall_tokens = 0

Conclusion:

```text
Recall filter too strict or wrong role/budget.
```

Action:

```text
Check role budget.
Check max_frame_distance.
Check rope_safe filter.
Check candidate set layer_id match.
```

---

### Case 5: recall_tokens > 0 but no gain

Possible causes:

```text
Near-only recall is redundant.
Recalled tokens are ignored by attention.
Retrieval selects irrelevant tokens.
Prompt suite does not require long recall.
```

Action:

```text
Check attention_mass_to_recall or QK proxy mass.
Run A-B-A scene revisit prompts.
Compare actual_q vs evicted_k.mean compression.
Increase recall distance only with pre-RoPE remap.
```

---

### Case 6: far post-RoPE recall degrades

Conclusion:

```text
RoPE mismatch confirmed.
```

Action:

```text
Disable far post-RoPE recall.
Implement pre-RoPE bank + remap.
```

---

### Case 7: pre-RoPE + remap still no gain

Possible causes:

```text
Memory is valid but not useful for prompts.
Selection is weak.
Attention ignores memory.
Self-Forcing bottleneck is not old K/V loss.
```

Action:

```text
Add rho/z descriptor scoring.
Add A-B-C-A prompts.
Add small positive recall bias.
Try Causal-Forcing integration.
```

---

## 13. Implementation checklist for next agent

Give the agent this exact checklist.

```text
[ ] Add docs/22_lifecache_next_experiment_roadmap.md reading to implementation prompt.
[ ] Add trace fields listed in Section 3.
[ ] Add scripts/summarize_lifecache_trace.py.
[ ] Add configs/lifecache/lifecache_recall_near_only.yaml.
[ ] Add configs/lifecache/lifecache_compression_clean_only.yaml.
[ ] Add configs/lifecache/lifecache_pre_rope_remap.yaml.
[ ] Add TokenSet.rope_mode.
[ ] Add TokenSet.frame_positions.
[ ] Add TokenSet.source_start_frame.
[ ] Add TokenSet.capture_step.
[ ] Add TokenSet.rho and TokenSet.z but do not use them until B5.
[ ] Add RecallResult.frame_positions and RecallResult.rope_mode.
[ ] Add RecallConfig.rope_safe_recall / allow_post_rope_recall / max_post_rope_frame_distance.
[ ] Add LifeCacheRuntime.begin_capture/end_capture.
[ ] Add LifeCacheRuntime.on_kv_evicted_payload(payload).
[ ] Change _lifecache_evicted to _lifecache_evicted_list.
[ ] Capture only when runtime.capture_enabled=True.
[ ] Capture q_pre_rope/q_post_rope in causal_model.py.
[ ] Stop using evicted_k.mean as q_proxy.
[ ] Add post-RoPE far recall filter.
[ ] Implement SelfForcingRopeAdapter.map_frame_positions.
[ ] Implement first version of pre-RoPE recalled K remap.
[ ] Run D0-D4 diagnostic experiments.
[ ] Only after D4 passes, implement rho/z/update_retention/write_or_merge.
```

---

## 14. Suggested command templates

Environment variables:

```bash
export LIFECACHE_ENABLE=1
export LIFECACHE_CONFIG=configs/lifecache/lifecache_compression_clean_only.yaml
```

Trace-only:

```bash
export LIFECACHE_CONFIG=configs/lifecache/lifecache_trace_only.yaml
python <your_self_forcing_inference_script>.py ...
```

Compression-clean-only:

```bash
export LIFECACHE_CONFIG=configs/lifecache/lifecache_compression_clean_only.yaml
python <your_self_forcing_inference_script>.py ...
python scripts/summarize_lifecache_trace.py \
  --trace outputs/lifecache/compression_clean_only.jsonl \
  --out-md outputs/lifecache/compression_clean_only_summary.md
```

Near-only recall:

```bash
export LIFECACHE_CONFIG=configs/lifecache/lifecache_recall_near_only.yaml
python <your_self_forcing_inference_script>.py ...
python scripts/summarize_lifecache_trace.py \
  --trace outputs/lifecache/recall_near_only.jsonl \
  --out-md outputs/lifecache/recall_near_only_summary.md
```

Pre-RoPE remap recall:

```bash
export LIFECACHE_CONFIG=configs/lifecache/lifecache_pre_rope_remap.yaml
python <your_self_forcing_inference_script>.py ...
python scripts/summarize_lifecache_trace.py \
  --trace outputs/lifecache/pre_rope_remap.jsonl \
  --out-md outputs/lifecache/pre_rope_remap_summary.md
```

---

## 15. Success criteria for this project phase

This phase is successful if we can produce the following table with reliable results:

| Run | Output safe? | Bank grows? | Recall > 0? | Far memory? | RoPE-safe? | Quality effect |
|---|---|---|---|---|---|---|
| trace-only | yes | no/irrelevant | no | no | n/a | same as native |
| compression-clean-only | yes | yes | no | no | n/a | same as native |
| near-only recall | yes | yes | yes | no | yes by distance | stable, maybe no gain |
| far post-RoPE recall | maybe no | yes | yes | yes | no | likely worse |
| pre-RoPE remap recall | yes | yes | yes | yes | yes | target improvement |
| pre-RoPE remap + rho | yes | yes | yes | yes | yes | target improvement + less wrong recall |

If this table cannot be filled, the project is not ready for full ablation or paper claims.

---

## 16. Paper/idea implications

If pre-RoPE remap recall works:

```text
Claim direction:
  LifeCache is a RoPE-safe content-addressed KV memory for sliding-window AR video generation.

Main contribution:
  selected raw-K historical memory + read-time relative RoPE remap + real-query token recall.
```

If rho improves results:

```text
Claim direction:
  LifeCache further uses AdaMem-style soft retention to unify recall and forgetting.
```

If nothing improves even after pre-RoPE remap:

```text
Interpretation:
  For the current Self-Forcing setting and prompt suite, missing old K/V may not be the dominant bottleneck.

Next pivot:
  stronger A-B-C-A benchmark;
  head-aware region bias;
  semantic descriptors;
  Causal-Forcing integration;
  or training-light AdaMem-style gates.
```

---

## 17. Final recommendation

The next implementation should not add VLM, entity memory, or more cache pools.

The next implementation should do exactly this:

```text
1. Make memory capture clean-only.
2. Make compression query real.
3. Add RoPE metadata.
4. Block unsafe post-RoPE far recall.
5. Implement pre-RoPE raw-K memory with relative-clamp remap.
6. Add diagnostics to prove recalled memory is used.
7. Only then add AdaMem rho/z soft retention.
```

This is the shortest path from the current no-gain state to a scientifically interpretable result.
