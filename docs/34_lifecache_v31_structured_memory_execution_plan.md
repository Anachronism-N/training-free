# LifeCache v3.1 与 Structured Historical KV Memory 推进计划

> 日期：2026-07-16  
> 目标：在不否定 historical KV cache 路线的前提下，停止无边界迭代，将当前项目收敛为一条可验证、可复现、可逐步升级的工程与实验路线。

---

## 1. 当前阶段判断

当前仓库已经完成了一个重要转折：最新 sparse-v3 首次从 trace 上证明了以下链路确实发生：

```text
历史 K/V 被淘汰
→ 写入长期 bank
→ 候选集合被检索
→ recalled tokens 被拼接到 active cache
→ attention 的 K/V 长度发生变化
```

此前多个版本存在 payload 被全部过滤、layer 级路由将 recall 关闭、frame/spatial metadata 丢失、RoPE 使用错误、配置值未真正生效等问题。因此，早期负结果不能被视为 historical KV memory 路线无效。

但当前版本仍不能被当作最终可信方法，主要原因包括：

1. 配置中的 `recall_top_tokens=32` 实际仍可能召回 512 tokens；
2. relative temporal mapping 后又被 sparse RoPE 二次 clamp；
3. 当前 bank 主要来自 denoising eviction，而不是 clean-context memory；
4. full-frame oracle 仍停留在配置层，没有独立执行路径；
5. invalid recalled metadata 只触发 warning，随后仍可能继续参与 attention；
6. `TokenSet` 的 spatial metadata 在 clone/device transfer 中仍可能丢失；
7. 当前是所有 heads 统一访问 memory，而不是真正的 per-head access；
8. append 模式改变了总 attention token 数，正式方法尚未实现固定预算；
9. 当前 A-B-A 评估仍缺少严格的 segment schedule 和因果对照。

因此，下一步不再继续叠加 scorer、bank 策略或更多超参数，而是依次完成：

```text
v3.1 correctness closure
→ 统一 RoPE 坐标契约
→ clean structured memory capture
→ deterministic full-frame oracle
→ controlled A-B-A evaluation
→ structured compression
→ per-head fixed-budget access
→ hybrid retention + archival recall
```

---

## 2. 项目主线重新定义

当前不再讨论：

```text
历史 KV cache 是否理论可行？
```

项目主问题应改为：

> 在 Self-Forcing 上，哪一种历史 KV 表示、位置编码、访问方式和预算分配，能够稳定改善长视频中的场景回访、身份保持和世界状态延续？

具体拆分为六个可验证假设：

### H1：clean historical K/V 携带可复用状态

正确 A1 场景的 clean historical K/V，在返回 A2 时，应比错误场景、随机 memory、shuffled-V memory 更有帮助。

### H2：结构化 memory unit 优于任意 sparse token

完整 frame 或规则 patch block 保留空间结构，因此应比 global arbitrary top-k tokens 更稳定。

### H3：位置编码必须与当前 query 坐标兼容

raw K、真实 temporal/spatial metadata 和 recall-time RoPE 缺一不可。

### H4：历史 memory 对不同 heads 的价值不同

layout、anchor、identity、phase heads 更可能需要历史；motion/local heads 可能受到旧状态污染。

### H5：固定预算比无约束 append 更适合作为正式方法

append 适合 oracle 因果验证；正式实验应在 recent 和 historical memory 之间重新分配原有预算。

### H6：先检索结构单元，再做单元内压缩，比全局 token top-k 更合理

正式路线应优先采用：

```text
frame/chunk retrieval
→ patch/block selection
→ token-level refinement
```

而不是直接从所有历史 token 中全局 top-k。

---

## 3. 总体推进阶段

| 阶段 | 目标 | 输出 | 是否允许跑长视频 |
|---|---|---|---|
| Stage 0 | 冻结基线与实验协议 | native/trace-only/zero-recall baseline | 仅短样例 |
| Stage 1 | 完成 sparse-v3 correctness closure | v3.1 可信 sparse baseline | 通过 gate 后允许 |
| Stage 2 | 统一 RoPE 坐标契约 | parity tests + relative mapping tests | 通过 gate 后允许 |
| Stage 3 | clean structured capture | 完整 frame memory block | 允许 oracle |
| Stage 4 | deterministic full-frame oracle | 正确/错误/random/shuffled controls | 必须执行 |
| Stage 5 | structured compression | frame → patch block → sparse token | oracle 有效后 |
| Stage 6 | per-head fixed-budget access | head mask + budget allocator | structured memory 有效后 |
| Stage 7 | hybrid memory | smart retention + archival recall | 最终方法阶段 |

---

# Stage 0：冻结可信基线

## 4.1 冻结三种基线

必须固定并保留以下三种配置：

```text
E0：Native Self-Forcing
E1：LifeCache enabled + trace_only=true
E2：LifeCache recall_enabled=true + recall_top_tokens=0
```

三者应在相同 seed、prompt、frame 数和推理参数下运行。

验收不再使用 MP4 文件大小，而是比较：

```text
latent max_abs_diff
latent mean_abs_diff
relative L2
decoded frame pixel difference
输出帧 hash 或关键帧特征差异
```

目标：

```text
E0 ≈ E1 ≈ E2
```

若 trace-only 或 zero-recall 改变输出，应先修复 integration side effect，不进入后续阶段。

## 4.2 固定实验环境

每次实验必须记录：

```text
commit SHA
config 路径
seed
prompt schedule
frame 数
分辨率
local attention size
sink size
启用层
recall budget
bank budget
RoPE policy
capture source
budget mode
```

建议所有运行目录包含：

```text
run_manifest.yaml
stdout.log
cache_trace.jsonl
metrics.json
selected_frames/
video.mp4
```

---

# Stage 1：LifeCache v3.1 Correctness Closure

## 5.1 修复真实 recall budget

当前 composer 使用 role budget，而不是配置中的 `recall_top_tokens`。

应改为：

```python
effective_recall_tokens = min(
    role_budget.recall,
    recall_config.top_tokens,
)
```

并在每次 compose 中记录：

```text
configured_recall_tokens
role_recall_budget
effective_recall_tokens
actual_recalled_tokens
```

硬性验收：

```text
配置 32 → 实际 32
配置 0  → 实际 0
配置 1560 → 实际不超过 1560
```

## 5.2 补齐 TokenSet spatial metadata

`TokenSet` 必须保证：

```text
len(K)
= len(V)
= len(token_indices)
= len(frame_positions)
= len(spatial_positions)
= len(importance_score)
```

需要修改：

```text
__post_init__
clone_with_tokens
to_device
anchor promotion
compression:none
random recall
budget cropping
```

必须增加：

```python
if spatial_positions is not None:
    assert spatial_positions.ndim == 1
    assert spatial_positions.numel() == num_tokens
    assert spatial_positions.min() >= 0
```

空间范围检查由 integration 层结合 `grid_h * grid_w` 完成。

## 5.3 invalid recall 必须 fail-fast 或回退 native

当前 warning 后继续 attention 的行为不可接受。

建议增加：

```yaml
strict_correctness: true
```

行为定义：

```text
strict=true：发现 invalid metadata 立即抛出 RuntimeError
strict=false：删除全部 invalid recalled tokens，回退 native recent cache
```

禁止：

```text
未旋转 historical K
+ post-RoPE query/recent K
```

进入同一次 attention。

## 5.4 打通 current_frame 与 max_frame_distance

调用链应改为：

```text
causal_model.current_start_frame
→ runtime.compose_active_cache(current_frame)
→ composer.compose(current_frame)
→ recall_tokens(current_frame)
→ retrieve_token_sets(current_frame)
```

需要记录：

```text
candidate frame range
filtered set count
selected set frame range
actual maximum distance
```

## 5.5 修复 random recall control

随机对照必须同步替换：

```text
K
V
frame_positions
spatial_positions
rope_mode
source_set_id
source_position
```

不能只替换 K/V 而保留原位置。

## 5.6 强制 pre-RoPE recall

当：

```yaml
allow_post_rope_recall: false
```

则所有 recalled token 的 `rope_mode` 必须为 `pre_rope`。

若任一 token 为 `post_rope`：

```text
strict=true：报错
strict=false：移除该 token
```

## 5.7 Stage 1 单元测试

新增：

```text
tests/test_tokenset_metadata.py
tests/test_recall_metadata.py
tests/test_active_cache_budget.py
tests/test_random_recall_alignment.py
tests/test_lifecache_equivalence.py
```

通过标准：

```text
configured=effective=actual
invalid frame positions=0
invalid spatial positions=0
all recalled rope_mode=pre_rope
random recall metadata 对齐
trace-only 与 native 等价
```

---

# Stage 2：统一 RoPE 坐标契约

## 6.1 第一版采用 absolute-compatible mapping

v3.1 不改变 native query 和 recent K 的坐标。

historical K 的 temporal position 定义为：

```python
distance = (
    current_start_frame - historical_frame
).clamp(0, temporal_range - 1)

mapped_absolute_position = (
    current_start_frame - distance
)
```

含义：

```text
近历史保留真实相对距离
远历史映射到训练窗口最旧的合法相对位置
query/recent 保持 native absolute coordinate
```

## 6.2 删除二次 TR clamp

`sparse_3d_rope()` 内部不再执行：

```python
temporal_idx.clamp(0, TR - 1)
```

只允许根据 frequency table 长度做安全检查：

```python
temporal_idx.clamp(0, max_temporal_frequency_index)
```

接口建议改为：

```python
causal_rope_apply_sparse_3d(
    ...,
    temporal_idx=mapped_absolute_position,
    temporal_mode="absolute",
)
```

未来若实现 local coordinate，应使用独立函数或独立枚举，禁止隐式混用。

## 6.3 必须增加 parity test

核心测试：

```python
full_roped = causal_rope_apply(
    full_raw_k,
    full_grid,
    freqs,
    start_frame=source_frame,
)

selected = random_sparse_positions

sparse_roped = causal_rope_apply_sparse_3d(
    full_raw_k[0, selected],
    freqs,
    temporal_idx=full_temporal_idx[selected],
    spatial_idx=full_spatial_idx[selected],
    grid_h=H,
    grid_w=W,
)

torch.testing.assert_close(
    sparse_roped,
    full_roped[0, selected],
    rtol=1e-4,
    atol=1e-4,
)
```

覆盖：

```text
单帧完整 grid
多帧
随机 sparse token
连续 patch block
不同 H/W
float32
bf16
```

## 6.4 增加 relative mapping test

验证：

```text
0 <= current - mapped <= TR-1
近历史距离保持不变
远历史距离截断到 TR-1
mapped position 不被二次截断
```

通过 Stage 2 前，不允许继续长视频 ablation。

---

# Stage 3：Clean Structured Memory Capture

## 7.1 新增独立模块

新增：

```text
src/lifecycle_kv/structured_memory.py
```

建议数据结构：

```python
@dataclass
class StructuredMemoryBlock:
    memory_id: str
    layer_id: int

    k_raw: torch.Tensor
    v: torch.Tensor

    frame_indices: torch.Tensor
    grid_h: int
    grid_w: int

    capture_source: str
    capture_step: int
    prompt_segment: str | None
    quality_stage: str

    descriptor: torch.Tensor | None = None
```

第一版仅支持完整 frame，不做 compression 和 retrieval。

## 7.2 从 clean-context forward 直接捕获

不能继续通过 eviction list 间接获得 oracle memory。

建议在 clean-context refresh 前设置：

```python
runtime.begin_structured_capture(
    segment_id=current_segment,
    source="clean_context",
)
```

在目标 layer 的 `qkv_fn()` 中直接捕获当次 forward 的：

```text
raw/pre-RoPE K
aligned V
完整 frame token 顺序
真实 frame index
H/W
```

捕获完成后：

```python
runtime.end_structured_capture()
```

## 7.3 去重规则

同一个：

```text
(layer_id, frame_index, capture_source, segment_id)
```

只保留一个 memory block。

不得因多次 denoising forward 重复写入相同内容。

## 7.4 保留 eviction bank 作为对照

现有 bank 不删除，但必须明确标记：

```text
capture_source=denoising_eviction
```

structured memory 标记：

```text
capture_source=clean_context
```

后续实验可直接比较：

```text
clean frame memory
vs
denoising eviction memory
```

---

# Stage 4：Deterministic Full-Frame Oracle

## 8.1 Oracle 必须独立于普通 recall

新增并解析配置：

```yaml
oracle_mode: full_frame
oracle_layer: 29
oracle_num_frames: 1
oracle_capture_segment: A1
oracle_inject_segment: A2
oracle_budget_mode: append
oracle_control: correct
```

要求 manager 对未知字段执行校验，不能静默忽略 oracle 配置。

## 8.2 Oracle 绕过以下模块

```text
QK compression
set-level ranking
token-level ranking
ordinary bank pruning
random top-k
usage score
quality score
```

执行路径：

```text
A1 clean-context full frame
→ deterministic memory selection
→ full-grid native RoPE
→ A2 attention injection
```

## 8.3 第一轮使用 append

append 的目的只是证明 memory 是否产生因果影响：

```text
[memory frame | native recent window]
```

成功后再测试固定预算：

```text
1 historical frame
+ 20 recent frames
= 21-frame native total budget
```

## 8.4 必要 controls

第一轮至少运行：

| ID | 条件 | 目的 |
|---|---|---|
| O0 | Native | 基线 |
| O1 | Correct A1 full frame | 正确历史内容 |
| O2 | Wrong B full frame | 场景特异性 |
| O3 | Random historical frame | 排除 token 数影响 |
| O4 | Correct K + shuffled V | 检验 K/V 对齐 |
| O5 | Correct K + zero V | 检验 value 贡献 |
| O6 | Correct full frame + fixed budget | 公平预算 |
| O7 | Correct full frame + clean vs eviction | 检验 memory 质量 |

每项至少 3 seeds。

## 8.5 Oracle 的有效标准

不能只看视频是否“不同”。

有效条件：

```text
Correct A1
稳定优于
Wrong B / Random / Shuffled-V / Zero-V
```

并满足：

```text
A2 场景恢复提升
身份或物体恢复提升
B 场景泄漏不增加
运动不明显冻结
整体画质不显著下降
```

---

# Stage 5：Controlled A-B-A Prompt Schedule

## 9.1 显式 segment schedule

新增：

```python
@dataclass
class PromptSegment:
    segment_id: str
    start_frame: int
    end_frame: int
    prompt: str
```

配置示例：

```yaml
segments:
  - id: A1
    start_frame: 0
    end_frame: 29
    prompt: "A woman in the original blue kitchen holding a red cup..."

  - id: B
    start_frame: 30
    end_frame: 69
    prompt: "The same woman walking in a green garden..."

  - id: A2
    start_frame: 70
    end_frame: 119
    prompt: "The same woman returns to the original blue kitchen..."
```

segment 边界必须与生成 block 对齐。

## 9.2 评价指标

至少计算：

```text
A1–A2 scene similarity
A1–A2 background similarity
subject identity similarity
red-cup reappearance
blue-kitchen recovery
B→A2 leakage
temporal flicker
dynamic degree
latent output delta
real post-RoPE memory attention mass
```

## 9.3 实际 attention mass

在 RoPE 完成后，对部分 query/head 计算：

```python
logits = q_roped @ k_roped.transpose(-1, -2) / sqrt(d)
weights = softmax(logits)

memory_mass = weights[..., memory_mask].sum(-1)
recent_mass = weights[..., recent_mask].sum(-1)
```

记录：

```text
all-head memory mass
per-head memory mass
WAVE heads memory mass
non-WAVE heads memory mass
memory mass 随时间变化
```

pre-RoPE cosine 只用于 retrieval，不作为 attention 使用程度的证据。

---

# Stage 6：Structured Compression

只有 full-frame oracle 明确有效后，才进入 compression。

## 10.1 memory unit 递进实验

固定相同 source frame 和总预算，按顺序比较：

```text
完整 frame
→ 规则 patch grid
→ 连续 patch block
→ patch-block top-k
→ arbitrary sparse token
```

目的：确定最小仍能保留收益的结构单元。

## 10.2 推荐的正式压缩路线

```text
frame-level descriptor retrieval
→ 选中相关 frame/chunk
→ frame 内 patch/block scoring
→ 保留连续空间结构
→ 必要时再做 token refinement
```

避免直接对整个历史 bank 做 global token top-k。

## 10.3 descriptor 设计顺序

优先级：

```text
1. clean latent/frame descriptor
2. K summary
3. prompt/segment descriptor
4. entity/scene descriptor
5. 多模态融合 descriptor
```

第一版应复现 LongLive-RAG 类 frame-level descriptor，而不是继续增加复杂 token scorer。

---

# Stage 7：Per-Head Fixed-Budget Access

## 11.1 第一版不需要 ragged KV

可使用统一的：

```text
[memory | recent]
```

再通过 SDPA additive mask 控制每个 head 是否访问 memory。

比较：

```text
all heads
non-WAVE heads
layout/anchor heads
profiled memory heads
```

## 11.2 固定预算分配

建议支持：

```yaml
budget_mode: append | replace_recent
memory_frames: 1
recent_frames: 20
```

正式方法采用：

```text
historical budget + recent budget = native total budget
```

## 11.3 head-specific allocation

后续可扩展：

```text
layout heads：较多 historical frames
identity heads：高分辨率主体 patches
motion heads：仅 recent window
phase heads：anchor + long-range structured memory
```

---

# Stage 8：Hybrid Memory 最终路线

当 structured recall 有效后，最终方法建议为：

```text
LifeCache-v4
=
Native Recent Window
+ Head-aware Smart Retention
+ Structured Archival Frame Bank
+ Scene-triggered Historical Recall
+ Fixed-Budget Memory Injection
```

其中：

```text
Smart Retention
解决连续演化与重要状态不被过早丢弃；

Archival Recall
解决场景回访、角色重现和远距离历史访问；

Structured Memory
保证 frame/patch 的空间关系；

Head-aware Access
避免旧状态污染 motion/local heads；

Fixed Budget
保证公平性和计算可控。
```

---

## 12. 建议的提交序列

### Commit 1

```text
fix: close sparse-v3 budget and metadata gaps
```

内容：

```text
recall_top_tokens 真正生效
spatial metadata clone/device/validation
current_frame/max_frame_distance 贯穿
random recall metadata 同步
compression:none metadata 传播
strict invalid recall behavior
pre-RoPE-only enforcement
```

### Commit 2

```text
fix: define a single historical RoPE coordinate contract
```

内容：

```text
删除 TR 二次 clamp
absolute-compatible mapping
sparse/full RoPE parity tests
relative-distance tests
明确 temporal_mode
```

### Commit 3

```text
feat: add clean structured frame memory capture
```

内容：

```text
StructuredMemoryBlock
clean-context capture hook
完整 frame K/V
真实 frame/grid metadata
capture deduplication
```

### Commit 4

```text
feat: implement deterministic full-frame oracle
```

内容：

```text
解析 oracle config
A1 capture / A2 injection
append / replace_recent
correct / wrong / random / shuffled controls
```

### Commit 5

```text
feat: add segmented A-B-A inference schedule
```

内容：

```text
PromptSegment
block-aligned condition switching
segment-aware capture/injection
run manifest
```

### Commit 6

```text
feat: add post-RoPE memory attention diagnostics
```

内容：

```text
real attention mass
per-head statistics
WAVE/non-WAVE grouping
latent output delta
```

### Commit 7

```text
feat: add structured patch memory and fixed-budget access
```

仅在 full-frame oracle 有效后实施。

---

## 13. 两周执行计划

### 第 1–2 天：Sparse v3.1 correctness

完成 Commit 1，运行 E0/E1/E2 短样例。

交付：

```text
correctness tests
配置生效 trace
latent equivalence report
```

### 第 3 天：RoPE contract

完成 Commit 2。

交付：

```text
parity tests
relative mapping tests
无二次 clamp
```

### 第 4–6 天：Clean structured capture

完成 Commit 3。

交付：

```text
完整 frame memory dump
frame/grid metadata audit
clean vs eviction 对比
```

### 第 7–9 天：Full-frame oracle

完成 Commit 4。

先运行：

```text
O0 Native
O1 Correct A1
O2 Wrong B
O3 Random frame
O4 Shuffled V
```

### 第 10 天：Prompt schedule

完成 Commit 5，确保 A-B-A 切换发生在指定帧。

### 第 11–12 天：Attention diagnostics

完成 Commit 6。

### 第 13–14 天：三 seed 完整复验

汇总：

```text
correctness
attention mass
scene/identity/object metrics
visual review
Go/No-Go conclusion
```

---

## 14. Go / No-Go 规则

### Go：继续 structured historical KV

满足：

```text
Correct A1 full-frame 在多个 seed 上稳定优于 controls；
正确 memory 的 attention mass 可观且具有 head specificity；
场景恢复、身份或物体恢复至少一项稳定提升；
运动和画质没有明显恶化。
```

后续进入：

```text
patch/block compression
frame-level retrieval
per-head access
fixed-budget allocation
hybrid retention + recall
```

### Partial Go：full-frame 有效，sparse 无效

结论：

```text
历史 KV 有效，但空间结构是必要条件。
```

后续停止 arbitrary token sparse recall，专注 frame/patch-block memory。

### Partial Go：仅部分 heads 有效

结论：

```text
历史 memory 存在明显 head specialization。
```

后续优先实现 per-head access mask。

### Backbone-specific No-Go

若 clean、完整、确定性注入的 historical frame 在 Self-Forcing 上仍无法区分正确与错误 memory，则停止在当前 backbone/当前 layer 上继续调 scorer。

下一步比较：

```text
其他 layer
Causal-Forcing
smart retention
latent-level memory
scene/world-state memory
```

该结论不否定 historical KV cache 的总体可行性，只说明当前 Self-Forcing integration 不合适。

---

## 15. 当前禁止事项

在 Stage 1–4 完成前，暂停：

```text
大规模 top-k sweep
更多 bank size sweep
region bias sweep
更多 layer 数量 sweep
复杂 semantic scorer
agent-style retrieval
Causal-Forcing 全量迁移
仅凭视频文件大小判断效果
```

这些工作会增加变量，却不能回答核心问题。

---

## 16. 下一步唯一优先任务

当前最优先的工程顺序是：

```text
1. 修复 recall 32/512 不一致
2. 修复 spatial metadata 全链路
3. 删除 temporal 二次 clamp
4. 通过 sparse/full RoPE parity test
5. 实现 clean structured frame capture
6. 实现真正的 deterministic full-frame oracle
7. 在显式 A-B-A schedule 上运行正确/错误/random/shuffled 对照
```

当前阶段的成功标准不是得到一个复杂的新模块，而是回答一个明确问题：

> 在 Self-Forcing 上，一整个 clean、位置正确、内容正确、确定性注入的历史 frame K/V，能否在返回场景时带来内容特异的收益？

只有这一问题得到可信答案后，才进入 structured compression、head-aware access 和 hybrid memory 的正式方法开发。
