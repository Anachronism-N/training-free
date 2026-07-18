# LifeCache 失败审计与双骨干实验计划

> 日期：2026-07-17  
> 范围：Self-Forcing、Causal-Forcing、LifeCache v3.x

## 1. 结论

本项目的研究目标是 **training-free 长时间自回归视频外推**。任何方法结论都必须同时在以下原生骨干上验证：

- Self-Forcing：`third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt`
- Causal-Forcing：`../research_sprint/cf_checkpoints/chunkwise/causal_forcing.pt`

`../research_sprint/cf_checkpoints/chunkwise/longvideo.pt` 是经过长视频训练的上界，不属于主 training-free 对照。

此前 v3.2 从第 0 帧开始灰噪声的结果无效。它首先是 attention 路径回归，不是 historical KV 方法的负结论。

## 2. 已定位的工程根因

### P0：非启用层丢失原生 attention

LifeCache 只启用 layer 29，但 v3.2 删除了 layer 0--28 的原生 attention fallback。未启用层随后错误地复用了函数输入 `x`，导致整个网络从第一帧开始失真。

处理：恢复未启用层的原生 causal attention；仅在启用层分配 LifeCache payload 和 pre-RoPE cache。

### P0：oracle 实验被 sparse bank 污染

oracle 只在指定 recall frame 提供完整帧；其余时刻 composer 会静默回退到 sparse recall，因此所谓 oracle 结果并非单变量实验。

处理：`oracle_allow_sparse_fallback: false` 为默认值；严格 oracle 模式不再捕获无用 eviction memory。

### P0：RoPE 元数据丢失

`compression: none` 分支未保存 `frame_positions` 和 `spatial_positions`。历史 K 无法可靠地重映射到当前坐标，日志中的大量 invalid-position warning 与此一致。

处理：所有 TokenSet 路径传播显式时空坐标；strict 模式遇到无效 metadata 直接失败，非 strict 模式回退原生 attention，不再注入未正确旋转的 K。

### P0：oracle 捕获帧错位

clean block 一次包含 3 帧，旧实现总是截 cache 尾部，却把结果标成 block 第一帧。实际捕获的是第三帧。

处理：按 `target_frame - block_start_frame` 精确切片，并为该逻辑增加独立测试。

### P1：跨 prompt 状态泄漏

原 pipeline 只重置 KV index，不清理 bank/oracle/runtime state。同一进程生成多个 prompt 时，后一个视频可能读取前一个视频的历史。

处理：每次 inference 开始调用 runtime reset，清理 bank、oracle frame、step 和临时状态。

### P1：QK 评分峰值显存过高

旧实现物化 `[query_token, key_token, head]` 相似度张量。`gate020` 日志中的 6.96 GiB 分配失败来自该路径，并被多进程共享 GPU 放大。

处理：按 query chunk 计算 max similarity；诊断性 QK 默认关闭；GPU 实验不再在同卡并发。

### P1：clean-only 名称与实际来源不符

部分配置声称使用 clean memory，但 denoising eviction 仍可进入 bank。

处理：将 `capture_reason` 贯穿 pipeline/runtime；`capture_clean_only` 会真实拒绝 denoising eviction。

## 3. 参考工作的可迁移结论

从本地 AMA、RollingForcing、Pyramid-Forcing 和 TMM 综述可归纳为：

1. **先解决坐标有效性，再讨论召回质量。** 保留的 KV 只有在 query/key 的相对位置仍处于模型可解释范围时才有意义。
2. **retention 不等于 recall。** 必须分别验证“历史信息是否保存”“是否在正确时刻读出”“读错是否造成 false memory”。
3. **memory 应分层。** active recent context、archive memory 和 summary/structured memory 不应混成一个无边界 token pool。
4. **历史表示宜作为受控增量。** AMA 的 anchor-only additive attention 支持 parallel branch；直接把 memory append 到 native softmax 会改变所有 recent token 的竞争关系。
5. **评估要使用 registration-interference-return。** 连续运动 prompt 只能测漂移，不能证明长时召回。

## 4. 当前正确性状态

已经完成：

- gate=0 在 fusion helper 中直接返回 native branch，不执行 memory attention。
- head mask 固定为 `[1, 1, H, 1]`，避免 token/head 维广播错误。
- RMS matching 有上限，避免低能量 memory branch 被异常放大。
- trace path 可由环境变量按 run 隔离。
- Self-Forcing 和 Causal-Forcing 共用同一份已修复的 Wan attention 实现。
- 8 个针对性单元测试通过；Python 编译、shell 语法和 `git diff --check` 通过。

GPU 36 帧冒烟结果：

| Backbone | Native | gate=0 | 结果 |
|---|---:|---:|---|
| Self-Forcing | pass | pass | 视频正常，视觉轨迹一致 |
| Causal-Forcing chunkwise | pass | pass | 视频正常，视觉轨迹一致 |

不同 GPU/独立进程的解码视频会受 FlashAttention 非确定性影响，不能把跨进程像素差当作 gate=0 数学等价性证明。正式 Gate A 应在同 GPU、固定环境下记录 latent/attention 输出。

## 5. 后续执行顺序

### Gate A：双骨干原生等价性

- 36/120 帧，固定 prompt、seed、GPU。
- 比较 native、LifeCache enabled + gate=0。
- 检查 latent、attention 输出、首帧质量、亮度和运动轨迹。
- 任一骨干不等价，停止所有 memory sweep。

### Gate B：位置与捕获正确性

- oracle capture frame：0、30、60、90。
- recall frame：30、60、90；逐 token 检查 frame/spatial position。
- 做 correct / wrong / shuffled / zero-V 四个因果对照。
- correct recall 必须优于 wrong/shuffled，才说明 recall 携带可用时空信息。

### Gate C：诊断任务而非自然 prompt

建立三类 A-B-A prompt：

- identity return：人物离场后回归；
- scene return：相机离开场景后回访；
- object-state return：物体状态发生干扰后再次出现。

指标同时覆盖 identity、scene/layout、motion、brightness、false recall 和时间分段退化，不只看全视频平均分。

### Gate D：最小机制搜索

只在 Gate B/C 出现正因果信号后搜索：

- layer：先 29，再少量相邻层；
- gate：0.02 / 0.05 / 0.10；
- memory：full-frame oracle -> patch block -> sparse identity tokens；
- position：absolute remap 与 bounded/window-clamp 对照；
- head：all-head -> selected-head。

每个配置先在 Self-Forcing 和 Causal-Forcing 各跑 120 帧；通过后才扩到 240/600 帧。

### Gate E：training-free 长时收益

最终候选必须同时满足：

- 两个原生骨干均有改善；
- 不使用 `longvideo.pt` 作为主结果；
- 240/600 帧收益保持，而非只改善短片；
- latency、峰值显存和 bank 大小可控；
- wrong/shuffled memory 不产生同等收益。

## 6. 当前优先级

当前不应继续扩大 bank、top-k 或启用层数。最优先工作是完成 Gate A，然后用双骨干的 A-B-A oracle 因果实验判断 historical clean KV 是否真实可用。只有该问题得到肯定答案，才值得优化 retrieval 和压缩。
