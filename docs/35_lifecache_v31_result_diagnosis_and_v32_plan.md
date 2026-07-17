# LifeCache v3.1 实验诊断与 v3.2 推进计划

> 日期：2026-07-17  
> 适用仓库：`Anachronism-N/training-free`  
> 目的：记录 v3.1 的真实实验现象，解释其技术含义，并将下一阶段从“继续增强 sparse recall”收敛为“身份记忆与场景记忆解耦、clean capture、head-gated parallel injection 和亮度稳定诊断”。

---

## 1. 当前实验现象

当前 v3.1 相比 v3 的主要观察为：

```text
总体质量：几乎没有明显提升
亮度稳定性：长时间生成仍然会逐渐变暗
背景一致性：仍然存在明显背景幻觉和场景重构错误
人物身份：相比 v3 有一定改善
```

这一结果不能简单解释为 historical KV cache 无效。更合理的结论是：

> v3.1 已经从历史 K/V 中恢复出部分人物身份信息，但当前 memory unit、捕获来源、head 访问方式和 attention 注入方式不足以同时支持背景布局、场景状态和长时视觉稳定性。

因此，v3.1 应被视为一个“局部正信号 + 整体机制仍不完整”的阶段结果。

---

## 2. 当前结果说明了什么

### 2.1 历史 KV 中确实存在可复用的身份信息

人物 ID 相比 v3 有改善，说明至少以下链路已经开始产生有效贡献：

```text
历史人物相关 K/V
→ bank
→ retrieval
→ recall-time position handling
→ attention
→ 后续人物身份保持
```

这意味着当前 sparse QK memory 不应被整体推翻。它更适合被重新定义为：

```text
Identity / Salient Entity Memory
```

而不是同时承担身份、背景、布局、运动和世界状态等全部长期记忆任务。

### 2.2 当前 sparse token 选择偏向人物和显著前景

当前 set-level 和 token-level QK 评分更容易选择：

- 人脸；
- 衣服与发型；
- 高对比前景物体；
- 与当前 query 高度匹配的局部区域。

它不擅长保持：

- 房间整体几何；
- 家具之间的空间关系；
- 墙面、地面和背景纹理；
- 全局光照和色调；
- 场景拓扑与视角。

因此，“ID 有改善、背景仍幻觉”与当前 arbitrary sparse top-k 的机制是一致的。

### 2.3 背景记忆需要结构化 memory unit

背景和场景布局不是若干独立高分 token 的简单集合。它们依赖：

```text
连续 patch 区域
+ 相对空间关系
+ 全局低频结构
+ 场景色调
+ 物体布局
```

下一阶段不应继续期望单一 sparse token bank 同时解决 ID 和背景。应显式引入：

```text
完整 frame memory
或
规则 patch-block memory
```

### 2.4 持续变暗更可能来自基础稳定性与注入扰动

当前 v3/v3.1 使用 memory 与完整 recent window 的 union/append：

```text
[historical memory | native recent]
→ 同一个 attention softmax
```

这种做法会改变原生 recent attention 的 softmax 分母和竞争关系。即使 memory 只在人物身份上有帮助，也可能同时扰动：

- 全局亮度；
- 局部纹理；
- 运动连续性；
- 背景生成分布；
- latent 统计量。

此外，当前 historical V 主要来自 denoising eviction，而不一定是最终 clean-context 对应的 V。可能出现：

```text
K 足以匹配人物身份
但 V 仍带有噪声、错误背景或不稳定亮度统计
```

这可以解释“ID 略有改善，但画面仍会变暗且背景幻觉没有消失”。

### 2.5 all-head memory access 可能污染运动和局部生成 heads

为了避免此前 layer-level majority routing 将 recall 完全关闭，当前启用层被统一按 `LAYOUT` 处理。这等价于让该层所有 heads 都能访问历史 memory。

实际可能是：

```text
identity/layout heads：从历史 memory 中获益
motion/wave/local heads：被陈旧历史状态污染
```

因此，下一阶段应从 all-head access 转向 per-head gated access。

---

## 3. 当前版本不应继续做的事情

在完成下面的结构调整前，暂停：

```text
继续增大 recall top-k
继续扩大 bank
继续启用更多层
继续添加复杂 retrieval scorer
继续使用 all-head union append
继续用 denoising eviction 作为 scene memory
继续只通过最终视频主观观察判断问题
```

v3.1 已经说明，问题不再是“有没有召回足够多 token”，而是：

> memory 的职责、结构、来源、访问 head 和注入方式不正确。

---

## 4. v3.2 总体定义

下一版本建议命名为：

> **LifeCache v3.2: Dual-Channel Structured Memory with Head-Gated Injection**

核心结构：

```text
Native Recent Attention
        |
        +-------------------------------+
        |                               |
Identity Memory Branch          Scene/Layout Memory Branch
sparse clean entity KV          clean frame / patch-block KV
small budget                    structured spatial budget
identity/anchor heads           layout/phase heads
        |                               |
        +------------- gated fusion ----+
                      |
                 attention output
```

v3.2 不再使用单一 memory bank 同时承担所有长期一致性目标。

---

## 5. 双通道 memory 设计

### 5.1 Identity Memory

保留 v3.1 已经出现正信号的 sparse QK 路线，但重新限制职责。

```text
目标：人物 ID、服装、发型、显著实体一致性
来源：clean historical frame
单位：人物/显著实体 sparse tokens
预算：32–128 tokens
访问 heads：identity / anchor / selected layout heads
注入强度：低
```

第一版建议：

```yaml
identity_memory:
  enabled: true
  source: clean_context
  token_budget: 64
  layers: [29]
  access: selected_heads
  integration: parallel_gated
  gate: 0.10
```

### 5.2 Scene/Layout Memory

新增结构化场景记忆。

```text
目标：背景布局、场景回访、光照、物体关系和空间一致性
来源：clean-context forward
单位：完整 frame 或规则 patch blocks
检索单位：frame/chunk
访问 heads：layout / phase / long-range heads
```

第一阶段只实现两个 memory unit：

```text
S1：完整一帧
S2：规则 4×4 或 8×8 patch blocks
```

暂不使用全局 arbitrary sparse token 作为 scene memory。

---

## 6. 从 union append 改为 gated parallel attention

### 6.1 当前方式

```python
active_k = cat([memory_k, recent_k])
active_v = cat([memory_v, recent_v])
x = attention(q, active_k, active_v)
```

问题：

- memory 改变 recent softmax 分母；
- 无法独立控制 memory 输出幅度；
- 无法保证 `gate=0` 时严格退化到 native；
- 难以判断变暗来自 selection、V 还是 softmax 竞争。

### 6.2 建议方式

```python
x_recent = attention(
    q_roped,
    recent_k,
    recent_v,
)

x_identity = attention(
    q_roped,
    identity_k,
    identity_v,
)

x_scene = attention(
    q_roped,
    scene_k,
    scene_v,
)

x_identity = rms_match(x_identity, x_recent)
x_scene = rms_match(x_scene, x_recent)

x = (
    x_recent
    + identity_gate * identity_head_mask * x_identity
    + scene_gate * scene_head_mask * x_scene
)
```

也可以先使用更保守的凸组合：

```python
x = (
    1.0 - identity_gate - scene_gate
) * x_recent \
    + identity_gate * x_identity \
    + scene_gate * x_scene
```

### 6.3 第一轮 gate

只做小型响应曲线，不进行无边界调参：

```text
gate = 0.00
0.05
0.10
0.20
```

记录：

```text
ID 指标
背景指标
逐帧亮度
x_recent RMS
x_memory RMS
memory attention mass
```

---

## 7. Clean structured capture

### 7.1 不再依赖 eviction 构造 scene memory

scene memory 必须从 clean-context forward 中直接捕获。

在 clean forward 前：

```python
runtime.begin_clean_memory_capture(
    segment_id=current_segment,
    frame_range=current_frame_range,
)
```

在目标 layer 的 `qkv_fn()` 内直接保存：

```text
raw/pre-RoPE K
aligned V
absolute frame index
完整 spatial order
H/W grid
segment ID
capture_source=clean_context
```

### 7.2 数据结构

```python
@dataclass
class StructuredMemoryBlock:
    memory_id: str
    layer_id: int
    memory_type: str  # identity | scene

    k_raw: torch.Tensor
    v: torch.Tensor

    frame_indices: torch.Tensor
    spatial_positions: torch.Tensor
    grid_h: int
    grid_w: int

    segment_id: str
    capture_source: str
    quality_stage: str

    descriptor: torch.Tensor | None = None
```

### 7.3 硬性不变量

```text
同一 layer/frame/segment 只写一次
capture_source == clean_context
len(K) == len(V)
len(K) == len(frame_positions)
len(K) == len(spatial_positions)
完整 frame token 数 == frame_seq_length
spatial positions 覆盖合法 H×W 范围
```

---

## 8. RoPE 与基础长视频稳定性

### 8.1 Memory RoPE

historical K 使用 raw/pre-RoPE K，并在 recall 时按统一坐标契约重新编码。

禁止：

```text
relative mapping 后再次按 TR 二次 clamp
未旋转 historical K 与 post-RoPE query 混合
不同 memory branch 使用隐式不同坐标规则
```

### 8.2 基础 RoPE 稳定与 memory 解耦

Self-Forcing 本身的 rolling-window 变暗问题，应与 LifeCache 正交处理。

执行顺序应改为：

```text
native query/recent K RoPE stabilization
→ identity memory RoPE
→ scene memory RoPE
→ per-head memory access
→ gated fusion
```

不应让 FWAAR、split-window、AAR 与 LifeCache 互斥。

---

## 9. 变暗来源诊断实验

先用单一 prompt，不做 A-B-A，定位变暗来源。

| ID | 设置 | 目的 |
|---|---|---|
| D0 | Native Self-Forcing | 基础漂移 |
| D1 | Native + split-window/FWAAR，无 memory | 检查基础 RoPE |
| D2 | v3.1 union append | 当前问题复现 |
| D3 | sparse memory + parallel gate | 检查 softmax 竞争 |
| D4 | D3，仅 non-WAVE heads | 检查 head pollution |
| D5 | D4 + clean V | 检查 denoising V |
| D6 | D5 + RMS matching | 检查输出尺度 |

逐帧记录：

```text
mean luminance
RGB mean/std
latent mean/std
x_recent RMS
x_identity RMS
x_scene RMS
memory/recent attention mass
per-head memory output RMS
```

诊断规则：

```text
D1 改善 → 基础 rolling-window RoPE 是主因
D3 改善 → union softmax 竞争是主因
D4 改善 → WAVE/motion head 污染是主因
D5 改善 → denoising V 是主因
D6 改善 → memory branch 输出尺度不匹配是主因
```

---

## 10. ID 与背景分离实验

| ID | Identity Memory | Scene Memory | Head Access |
|---|---|---|---|
| M0 | 无 | 无 | native |
| M1 | sparse clean | 无 | identity heads |
| M2 | 无 | clean full frame | layout heads |
| M3 | 无 | clean patch blocks | layout heads |
| M4 | sparse clean | clean full frame | 分组 heads |
| M5 | sparse clean | clean patch blocks | 分组 heads |
| M6 | wrong identity | correct scene | controls |
| M7 | correct identity | wrong scene | controls |
| M8 | random/shuffled V | random/shuffled V | controls |

预期模式：

```text
M1：主要改善 ID
M2/M3：主要改善背景
M4/M5：同时改善 ID 与背景
wrong/random controls：不能产生同样收益
```

若结果符合，则双通道 memory 假设成立。

---

## 11. Head-gated access

第一版不实现 ragged per-head KV，可以复用同一份 memory K/V，并使用 additive mask 或输出 head mask。

建议对比：

```text
H0：all heads
H1：non-WAVE heads
H2：identity/anchor heads
H3：layout/phase heads
H4：identity heads 访问 identity memory，layout heads 访问 scene memory
```

要求：

```text
motion/wave heads 默认不访问 archival memory
query block 始终可访问 native recent
memory gate=0 时严格等价于 native
```

---

## 12. Fixed-budget 正式模式

append 仅用于 oracle 和因果验证。

正式比较必须支持 fixed budget：

```text
native total = 21 frames

example:
19 recent frames
+ 1 identity-equivalent budget
+ 1 scene frame
= 21-frame-equivalent total
```

对于 sparse identity memory，使用 token-equivalent budget；对于 full-frame scene memory，从 recent window 中移除等量 token。

同时保留：

```text
append oracle
fixed-budget official
```

两种模式独立报告。

---

## 13. 建议提交顺序

### Commit 1

```text
chore: freeze v3.1 implementation and experiment evidence
```

必须提交：

```text
实际运行代码
实际 config
run manifest
stdout/trace 摘要
关键帧
ID/背景/亮度观察
```

当前 GitHub 主分支最新代码与本次 v3.1 实验版本尚未完全对应，应先解决版本漂移。

### Commit 2

```text
fix: close remaining v3.1 correctness invariants
```

包括：

```text
configured/effective/actual recall budget 一致
删除 temporal 二次 clamp
invalid metadata fail-fast
spatial metadata clone/device propagation
pre-RoPE-only recall 检查
```

### Commit 3

```text
refactor: separate RoPE stabilization from historical memory
```

### Commit 4

```text
feat: add gated parallel historical attention
```

### Commit 5

```text
feat: capture clean identity and scene memory
```

### Commit 6

```text
feat: add dual-channel structured memory banks
```

### Commit 7

```text
feat: add head-gated fixed-budget injection
```

### Commit 8

```text
feat: add luminance and memory-branch diagnostics
```

---

## 14. 阶段验收标准

### Gate A：v3.1 correctness

```text
configured recall == effective recall == actual recall
invalid frame/spatial count == 0
all recalled K 为 pre-RoPE
RoPE parity test 通过
memory gate=0 与 native 等价
```

### Gate B：亮度稳定

至少满足：

```text
v3.2 的逐帧亮度衰减不明显劣于 native
parallel gated 不产生 union append 的额外变暗
memory branch RMS 不持续增长
```

### Gate C：内容特异性

```text
correct identity memory > wrong/random identity memory
correct scene memory > wrong/random scene memory
shuffled V 不产生同样收益
```

### Gate D：双通道有效

```text
identity branch 主要改善 ID
scene branch 主要改善背景
组合分支同时改善二者
运动和总体画质不显著下降
```

---

## 15. 当前研究结论

v3.1 不应被描述为完全失败。

更准确的表述是：

> LifeCache v3.1 已经开始从 historical KV 中恢复人物身份信息，但 arbitrary sparse、all-head、union-append、denoising-eviction 的实现无法同时保存背景空间结构，并会继续放大或无法缓解 Self-Forcing 的长时亮度漂移。

因此，下一阶段的重点不是“召回更多 token”，而是：

```text
保留 sparse ID memory 的局部收益
+ 新增 structured scene memory
+ 使用 clean capture
+ 改为 per-head gated parallel injection
+ 将基础 RoPE 稳定与 memory 正交组合
```

最终目标路线：

```text
Native Recent Window
+ Clean Sparse Identity Memory
+ Clean Structured Scene Memory
+ Head-Gated Parallel Attention
+ Fixed-Budget Allocation
+ RoPE / Luminance Stabilization
```

这应作为 LifeCache v3.2 的正式实现方向。