# 结构化视觉记忆路线与 v3.8 伪影审计

> 日期: 2026-07-19
> 状态: v3.8 已完成并判负；结构化 memory 原型已接入 PF，真实视频因本轮 GPU 权限被拒而待跑

## 1. 对当前进度的修正

PF 的 temporal-pattern cache 和 head label 是 baseline 机制，不是我们的创新点。v3.7 的
variance-only transport 相对 reset-fixed PF 只有小幅 Pareto 改善，不能支撑顶会主贡献。
当前可人工 review 的最优结果仍是：

```text
runs/REVIEW_v37_threeway/
```

其中 `ours` 显著优于 SF native，但仍存在用户指出的公园背景分块变化、跑酷肢体液化/
重影和 ID 不稳。它们也存在于原生 SF/PF 路径中；正确表述是当前方法没有解决 baseline
的外推误差，而不是这些问题由 LifeCache 单独引入。

## 2. v3.8 针对性消融及判定

v3.8 在用户指出的前两条 prompt 上测试三种 stale-V 改动：扩大 target overlap、限制
variance ratio、用相邻 live moment 差异抑制场景冲突。结果目录：

```text
runs/v38_artifact/
  comparisons/0_fourway.mp4
  comparisons/1_fourway.mp4
  comprehensive.json
  motion_luma.json
  block_boundaries.json
```

| 方法 | DINO | BG | Flow | Luma Q4/Q1 | Low-freq boundary | Edge boundary |
|---|---:|---:|---:|---:|---:|---:|
| v3.7 baseline | 0.8705 | 0.8676 | 6.763 | +5.72% | 1.336 | 1.060 |
| overlap | 0.8584 | 0.8521 | 6.621 | -5.53% | 1.357 | **1.032** |
| bounded | 0.8592 | 0.8505 | **6.927** | +1.89% | 1.370 | 1.041 |
| transition | 0.8656 | 0.8452 | 6.322 | +0.23% | 1.363 | 1.046 |

结论：三者都没有降低低频背景块边界，且均损失 DINO/BG。overlap 只改善 edge
boundary；bounded 只改善 flow 和亮度；transition 质量更差且明显变慢。v3.8 不替代
v3.7。继续扫 stale-V 参数的预期收益已经很低，应停止把它作为主线。

新增 `scripts/evaluate_block_boundaries.py`，以 decoded block 起点为中心分别计算低频变化
与边缘变化相对普通帧间变化的比值。该指标直接覆盖人工观察到的周期性背景跳变，后续
候选必须同时报告它，不能只用 DINO 掩盖块边界。

## 3. 从 Flash-VAReason 借鉴什么

`docs/flash_vareason.md` 的可迁移部分不是把音频算法机械套到视频，而是三级信息管理：

1. Local fusion：只融合时间相邻且视觉描述相似的帧，并保留完整时间区间。
2. Fixed budget：历史长度增长时，memory 计算量保持固定。
3. Uniqueness retention：优先保留全局不可替代的帧；被删冗余帧的信息融合回核心帧，
   而不是直接丢弃。

视频生成还必须增加第四点：压缩后的历史不能重新拼回普通 self-attention 与当前局部
上下文竞争，否则仍然只是另一种 cache eviction。

## 4. 新的核心假设

**长期外推的瓶颈不只是保存了哪些历史，而是模型如何读取历史。** PF、LifeCache 和多数
方案把历史 K/V 放回 self-attention；局部动态、背景、身份和远期事件共享同一个 softmax
归一化，远期信息要么被淹没，要么被增强过度后造成冻结、重影和场景回滚。

新原型采用双路径：

```text
native PF attention(q, local/cache K,V) -> x_native
compressed visual memory                -> M_K, M_V
query-conditioned memory attention(q, M_K, M_V) -> x_memory
x = x_native + gate * confidence * alignment * RMSMatch(x_memory)
```

关键区别：

- memory 有独立 softmax，不占用 native attention 的概率质量；
- frame-level retrieval 由当前 query 动态决定，不依赖静态 identity/motion head 标签；
- cosine confidence 抑制无相关历史，输出 alignment gate 抑制与当前生成方向冲突的残差；
- clean pass 只提交完成块，读出在 update 前快照 memory，保证当前块不会通过 memory
  分支重复访问自身；
- 只在指定中层启用且使用低 gate，默认关闭时严格保持 PF 行为。

这比“PF + 一个 cache 压缩器”更强：论文问题从 cache storage 改为
**training-free structured memory construction and decoupled memory readout**。

## 5. 与 AMA head 实验的关系

AMA 已经给出两个不能忽略的反例：

- 身份保持是分布式的。阈值从 0.25 放宽到 0.15 后约 38.9% 的 head 都有弱身份信息，
  少数 identity-head 路由并不足够。
- SF/CF 曾使用 `|QK|^2` proxy，导致 360 个 head 全被判成 identity，随后全头增强
  anchor，产生背景锁死和运动退化。静态分类若度量失真，整个方案会系统性出错。

因此 head 分类降为分析工具和 ablation，不再是主方法的硬路由。后续可以研究连续的
query-memory affinity 与深度特征，但不能再把粗粒度类别名称直接当成真实机制。

## 6. 已实现内容

- `src/lifecycle_kv/structured_visual_memory.py`
  - 同空间位置的相邻帧融合；
  - mean/std frame descriptor；
  - Flash-style uniqueness score；
  - fixed budget、端点保留、冗余信息回融和时间区间追踪。
- `src/lifecycle_kv/attention_fusion.py`
  - frame retrieval prior；
  - 独立 query-conditioned memory attention；
  - confidence、RMS match 和 alignment-gated residual fusion。
- PF integration
  - CLI/config/cache/pipeline/attention 已接通；
  - 原始未旋转 Q 用于内容检索；
  - memory 只保存 clean 历史；reset 跨 prompt 清空；
  - `PYTHONPATH` 和全部实验参数写入统一脚本。

验证：root tests `20 passed`；PF cache/config/history focused tests `44 passed`；Python
compile、shell syntax、`git diff --check` 均通过。

## 7. v3.9 实验矩阵

入口：

```text
scripts/run_v39_structured_memory_smoke.sh
```

同一 prompt/seed/36 latent frames 并行生成：

| 组别 | Memory | Gate | Layers | Budget | Spatial stride |
|---|---:|---:|---:|---:|---:|
| PF control | off | 0 | - | - | - |
| memory-low | on | 0.02 | 15-20 | 4 frames | 4 |
| memory-high | on | 0.05 | 15-20 | 4 frames | 4 |

预定输出：

```text
runs/v35_pf_value_refresh/20260719_v39_pf36/pf_refresh_pf36/
runs/v35_pf_value_refresh/20260719_v39_mem002/pf_refresh_mem002/
runs/v35_pf_value_refresh/20260719_v39_mem005/pf_refresh_mem005/
```

本轮受限环境中 PyTorch 看不到 CUDA；请求宿主 GPU 权限被拒，当前目录只有失败日志，
没有可 review 视频。恢复 GPU 权限后直接运行上述入口。

判定规则：先要求运行稳定且 memory-low/high 相对 PF 不降低 flow、DINO、BG，并降低
block boundary。任一强度出现主体复制、运动拖影或场景回滚，就缩小/重构 readout，而
不是扩到长视频。通过后才运行 120 帧两类伪影 prompt，再扩到三 prompt。

## 8. 后续路线

1. v3.9 若有稳定正向：将 memory 拆成 identity/scene continuity/event 三种预算，使用
   query 动态混合，专门区分“保持人物”与“允许背景/动作演化”。
2. v3.9 若只保 ID 但加剧块边界：加入 block-boundary residual schedule，让 memory
   在新块前几帧平滑衰减，而不是改变整个 stale V 分布。
3. PF 上筛选通过后，必须移植到原生 SF 和 Causal Forcing；SF+PF 只能证明原型有效，
   不能替代用户要求的双 backbone 验证。
4. 若单 prompt 长外推仍陷入瓶颈，转入 Echo-Forcing prompt/scene switching benchmark。
   该任务可直接验证 query-conditioned retrieval 是否能选择正确事件 memory、抑制旧场景
   幻觉，且比单场景 DINO 更能区分“冻结”与“正确记忆”。
