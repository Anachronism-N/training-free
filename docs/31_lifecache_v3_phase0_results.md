# LifeCache v3 Phase 0 实验结果记录

> 2026-07-15 | 最新推理结果汇总

## v3 Phase 0 配置

使用 `configs/lifecache/lifecache_v2_optimized.yaml`（实际上 v3 Phase 0 仍沿用 v2 配置，尚未为 v3 创建独立配置）：

```yaml
lifecache:
  enabled: true
  trace_only: false
  mode: union
  compression: qk_proxy           # Q-K proxy compression
  compression_topk: 512
  compression_min_tokens: 64
  recall_enabled: true
  recall_top_sets: 1
  recall_top_tokens: 32           # 最小 recall budget
  max_frame_distance: null         # 无距离过滤
  anchor_enabled: false
  motion_enabled: false
  region_bias_beta: 0.05          # 注意：region_bias 未实际接入 attention logits
  enable_last_n_layers: 1         # 仅 layer 29
  rope_safe_recall: true
  allow_post_rope_recall: false
  rope_remap_policy: relative_clamp
  max_post_rope_frame_distance: 21
  capture_clean_only: true        # 注意：此配置实际未生效（eviction 只在 denoising 发生）
  use_real_query_for_compression: true
  head_roles_path: third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv
```

## 关键代码变更（vs v2）

| 变更 | 文件 | 说明 |
|---|---|---|
| absolute frame position | causal_model.py | 使用 global_end_index 计算真实绝对帧索引 |
| spatial_positions sidecar | causal_model.py, tokenset.py | 添加 per-token 空间坐标 |
| sparse 3D RoPE 重写 | causal_model.py | 修复 complex freqs 处理，移除 fake-grid padding |
| compression metadata 同步 | compression.py | top-k 同时索引 K/V/frame/spatial metadata |
| runtime metadata 移除 | runtime.py | 不再 post-hoc 覆盖 metadata |
| head roles path 修复 | lifecache_manager.py | 从 repo root 解析，验证 360 个 head roles |
| metadata index_select | causal_model.py | fp.index_select(0, idx) 替代 fp[:idx.shape[0]] |

## 实验结果

### Review Prompts (seed=0, 120 frames)

| 实验 | 目录 | p00 | p01 | p02 | 时间 |
|---|---|---|---|---|---|
| Native SF | `runs/sf_native_120f/` | 7.0M | 5.9M | 9.8M | 6m |
| SF + Pyramid | `runs/sf_pyramid_120f/` | 8.5M | 4.9M | 7.2M | 6m |
| **v3 Phase 0** | **`runs/sf_lifecache_v3_phase0_120f/`** | **7.0M** | **5.9M** | **9.8M** | **8m** |
| v2 optimized | `runs/sf_lifecache_v2_optimized_120f/` | 6.7M | 6.5M | 9.5M | 8m |

### A-B-A Scene Revisit (seed=0, 120 frames)

| 实验 | 目录 | p00 | p01 | p02 |
|---|---|---|---|---|
| Native SF | `runs/sf_native_aba_120f/` | 3.3M | 7.3M | 3.3M |
| QK proxy recall | `runs/sf_lifecache_aba_120f/` | 3.4M | 6.9M | 3.0M |
| Timestep-filtered | `runs/sf_lifecache_aba_clean_120f/` | 3.3M | 7.3M | 3.3M |
| Random recall | `runs/sf_lifecache_aba_random_120f/` | 3.3M | 7.3M | 3.3M |

## 结论

v3 Phase 0 文件大小与 native SF 完全一致（7.0M/5.9M/9.8M），**仍无显著优化效果**。

按 docs/29 和 docs/30 的分析，当前失败并非"历史 KV recall"这个 idea 本身无效，而是实现中存在以下未闭环的问题：

1. **P0-3**: RecallResult 仍未传播 frame/spatial metadata（recall top-k 后位置信息丢失）
2. **P0-5**: temporal mapping 是 absolute clamp 而非 relative-to-current clamp
3. **P1-1**: recall_top_tokens=32 可能被 RegionBudget(recall=512) 覆盖
4. **P1-2**: max_frame_distance 未贯穿调用链
5. **P1-3**: region_bias 未实际接入 attention logits
6. **Phase 1 未做**: 完整 frame oracle（当前仍是 sparse token recall，存在结构损失）

按 docs/30 Section 8 的硬性验收门槛，当前未满足的实验前条件：
- [ ] RecallResult 携带真实 frame/spatial positions
- [ ] recall:view 不存在 -1 position
- [ ] sparse/full RoPE parity test 通过
- [ ] mapped relative distance <= TR-1
- [ ] effective recall budget 等于配置
- [ ] max_frame_distance 确实过滤候选
- [ ] region bias 未伪装为有效实验

## 下一步

按 docs/30 Commit 4 + Phase 1：
1. 完成配置真实性修复（recall_top_tokens, max_frame_distance, region_bias）
2. 完成 RecallResult metadata 传播
3. 实现 full-frame oracle（绕过 sparse token 的所有问题）
4. 在 oracle 通过后再推进 structured compression 和 semantic retrieval

## 所有实验结果目录

```
runs/
├── sf_native_120f/                      # 原生 SF baseline
├── sf_pyramid_120f/                     # SF + Pyramid Forcing
├── sf_lifecache_v1_120f/                # v1 (旧)
├── sf_lifecache_trace_120f/             # trace-only
├── sf_lifecache_compression_120f/       # compression-only
├── sf_lifecache_recall_120f/            # union recall (旧)
├── sf_lifecache_near_recall_120f/       # near-only (未完成)
├── sf_lifecache_pre_rope_remap_120f/    # pre-RoPE remap (未完成)
├── sf_lifecache_v2_optimized_120f/      # v2 optimized
├── sf_lifecache_v3_phase0_120f/         # v3 Phase 0 (最新)
├── sf_native_aba_120f/                  # A-B-A native
├── sf_lifecache_aba_120f/               # A-B-A QK proxy
├── sf_lifecache_aba_clean_120f/         # A-B-A timestep-filtered
└── sf_lifecache_aba_random_120f/        # A-B-A random recall
```
