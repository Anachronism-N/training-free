# v129 实验进度总结（服务器被kill时的状态）

Date: 2026-07-29

## 0. 恢复进度（2026-07-29 续 — 当天恢复并推进）

服务器恢复后已在4节点(29.191.210.172/29.119.99.134/29.127.81.251/29.232.229.175,
每节点8×H20)上重新推进v129全流程。**所有生成已完成，VBench-Long评估进行中。**

### 已完成

| 步骤 | 状态 | 说明 |
|---|---|---|
| 占卡 + 挂载 | ✅ | 4节点taiji_client mount; gpu_occupier.py占卡 |
| /tmp/self_forcing_dmd.pt | ✅ | 从DeepForcing副本恢复到4节点(5.3GB) |
| ffprobe | ✅ | pip安装static-ffmpeg,软链接到env bin(4节点共享) |
| 内部方法生成 | ✅ 256/256 | 两种confidence方法各128,audit-internal通过(p126标记已修复) |
| analyze-gates | ✅ | retrieval_gate_summary.json已生成 |
| 外部方法生成 | ✅ 384/384 | deep_forcing 128(断点恢复), rolling_forcing 128, longlive 128 |
| audit-external | ✅ | published_manifest.json已生成 |
| assemble | ✅ 1024视频 | 8方法×128 prompt, comparison_manifest.json |
| vbench-split | ✅ | 4节点全部完成2秒切片 |

### VBench-Long评估（✅ 全部完成, core profile 8维度×8方法=64 jobs）

| 维度 | 完成/8 | 说明 |
|---|---|---|
| temporal_flickering | 8/8 ✅ | 完成 |
| motion_smoothness | 8/8 ✅ | 完成(AMT) |
| overall_consistency | 8/8 ✅ | 完成(ViClip) |
| dynamic_degree | 8/8 ✅ | 完成(RAFT) |
| subject_consistency | 8/8 ✅ | 完成(DINO) |
| background_consistency | 8/8 ✅ | 完成(CLIP ViT-B/32) |
| aesthetic_quality | 8/8 ✅ | 完成(LAION aesthetic + CLIP ViT-L/14) |
| imaging_quality | 8/8 ✅ | 完成(pyiqa musiq) |

**vbench-collect完成**: paper_table.md + vbench_long_summary.json 已生成。
Quality Score: ours_prototype_retrieval_age24=82.95(最高), ours_confidence_motion=82.88。
Dynamic Degree: ours_confidence_motion=62.19(最高), ours_prototype_retrieval_age24=61.72。
所有Ours方法在Quality Score和Dynamic Degree上均优于sf_native/deep_forcing/rolling_forcing/longlive。

### 关键修复（本session）

1. **LongLive 474帧**: LongLive原生输出474帧(4×N−6),非SF/PF的477帧(4×N−3)。
   在`run_v129_external_baselines.py`加`LONGLIVE_EXPECTED_FRAMES=474`按方法验证。
2. **v125 manifest旧格式**: 补充`decoded_video_contract`/`prompt_items`/`prompt_file_sha256`;
   修正assembler期望的method_keys(`ours_landmark_retrieval1_age24`等)和source_key。
3. **VBench模型下载**: 节点1-3的`wget`不走代理→DINO/CLIP/ViClip/pyiqa下载失败。
   已通过共享文件系统(`runs/vbench_models_cache/`)分发并软链接到所有节点的`~/.cache`。
   DINO/CLIP走torch.hub(urllib,走代理),ViClip/pyiqa走wget(需预缓存)。
4. **p126标记**: 内部motion方法p126的published标记size过期(3783228→3783331),
   重新publish并清理了v120遗留的indexed文件。

### GPU占用策略

- VBench评估期间,空闲GPU(mem<100MiB)用`CUDA_VISIBLE_DEVICES`指定子集运行gpu_occupier.py占卡,
  防止资源被回收。
- GPU密集型维度(subject/background/overall_consistency用DINO/CLIP/ViClip)需要GPU时,
  先`--stop`占卡程序再运行。
- 当前: node0 GPU2-7占卡(6张), node2 GPU4,6,7占卡(3张); 其余GPU运行VBench。

### 待完成

v129 core profile **全部完成**。可选后续:
1. (可选) semantic_extension profile (8个语义维度, 需umt/grit/caption模型)
2. (可选) 60秒确认实验
3. (可选) paired bootstrap置信区间 + 人工review
4. 当前32张GPU已全部占卡(每节点8×813MiB), 评估完成后保持占卡状态

## 1. 实验概述

v129是当前最新的主实验批次，基于v125的成果，目标是128个MovieBench prompt的30秒单提示词长视频外推。v129在v125基础上增加了置信度门控检索（confidence-gated retrieval）机制，并排除了PF和ABA对比。

### v129方法

两种内部候选方法（confidence-gated retrieval）：
- `ours_confidence_recent`: sink1 + gated Retrieval1 + recent7
- `ours_confidence_motion`: sink1 + gated Retrieval1 + MotionPair1 + recent5

检索门控条件：`top1 cosine >= 0.55 AND top1 cosine - top2 cosine >= 0.005`

### 完整对比表（8种方法）

| Key | 来源 | 说明 |
|---|---|---|
| sf_native | 复用v125 | Self-Forcing基线 |
| deep_forcing | 官方代码 | 外部方法 |
| rolling_forcing | 官方代码 | 外部训练系统 |
| pf_native | 复用v125 | Pyramid Forcing |
| ours_v125_best | 复用v125 | v125最优(prototype_retrieval1_age24) |
| ours_confidence_recent | v129新生成 | 置信度门控+recent |
| ours_confidence_motion | v129新生成 | 置信度门控+motion |
| ours_v125_retrieval_motion | 复用v125 | v125检索+motion |

## 2. 进度状态（被kill时）

### 2.1 内部方法生成（`runs/v129_moviebench128_30s_internal/`）

| 方法 | 完成prompt数 | 总计 | 状态 |
|---|---:|---:|---|
| ours_prototype_retrieval_conf_recent | 112 | 128 | 🔄 87.5% |
| ours_prototype_retrieval_conf_motion | 90 | 128 | 🔄 70.3% |
| **合计** | **202** | **256** | **78.9%** |

- 已生成视频：610个
- 已完成状态标记：202个 `.done.json`
- 视频按prompt编号存储，每个prompt一个目录，每个目录1个MP4
- 缺失的prompt编号为偶数（p000, p002, p004...），可能是分片策略导致某些节点负责的prompt未完成

### 2.2 外部方法生成（`runs/v129_moviebench128_30s_external/`）

| 方法 | 视频数 | 状态 |
|---|---:|---|
| deep_forcing | 0 | ❌ 仅有worker目录结构，无视频 |
| rolling_forcing | — | ⏳ 未开始 |

- 外部方法目录结构已创建（`raw/deep_forcing/worker000-004`）
- 但无实际视频生成

### 2.3 非缓存附加实验（`runs/v129_noncache_addons/`）

历史Value校准实验，5种方法×1个prompt的快速筛选：

| 方法 | 完成数 | 状态 |
|---|---:|---|
| ours_value_control | 1 | ✅ |
| ours_value_var_s025 | 1 | ✅ |
| ours_value_var_s050 | 1 | ✅ |
| ours_value_var_s050_mid | 1 | ✅ |
| ours_value_var_s050_mid_t3 | 1 | ✅ |

- 已生成视频：111个（含中间产物）
- 状态标记：37个 `.done.json`
- 筛选实验已完成，等待人工review决定是否加入主实验

### 2.4 VBench-Long评估

- ❌ 未开始
- v125的VBench-Long结果已完成（见`docs/128_v125_vbench_long_results.md`）

### 2.5 v125复用数据

v125已完成8方法×128 prompt的生成和VBench-Long评估：
- 视频总数：20,481个
- VBench-Long 6维度评估完成
- v125最优方法：`ours_prototype_retrieval1_age24`（Dynamic Degree=61.93）

## 3. 已完成的历史实验（v93-v97）

| 版本 | 状态 | 关键结论 |
|---|---|---|
| v93 MovieBench-128 | ✅ 完成 | PF/v78/veil并列(DINO~0.91) |
| v90 Priority Factorization | ✅ 完成 | age-only效果弱 |
| v96 QK-Threshold | ✅ 完成(15层bug) | 97%头为Stable，分类无效 |
| v97 QK-Threshold(修正) | ✅ 完成 | 30层修正，但二分类仍失败 |
| v98-v128 | ✅ 完成 | 逐步演进到v125/v129基础方法 |

## 4. 被kill时丢失的工作

1. **内部方法剩余54个prompt的生成**（256-202=54）
   - `ours_confidence_recent`: 缺16个prompt
   - `ours_confidence_motion`: 缺38个prompt
2. **外部方法生成**（deep_forcing + rolling_forcing，共256个视频）
3. **VBench-Long评估**（所有v129方法）
4. **结果汇总和分析报告**

## 5. 恢复方案

### 需要重新生成的部分

1. **内部方法补全**：运行`run_v129_no_pf_10h.sh generate-internal`，脚本支持断点续传（检查`.done.json`跳过已完成prompt）
2. **外部方法**：运行`run_v129_no_pf_10h.sh generate-external`
3. **VBench-Long**：运行`run_v129_vbench_long.sh`
4. **结果汇总**：运行`run_v129_no_pf_10h.sh assemble`和`analyze-gates`

### 可复用的部分

- v125的sf_native、pf_native、ours_v125_best、ours_v125_retrieval_motion视频和VBench结果
- v129已完成202个内部方法prompt
- v129非缓存附加实验的5个筛选视频

## 6. 关键文件位置

| 内容 | 路径 |
|---|---|
| v129内部方法视频 | `runs/v129_moviebench128_30s_internal/ours_only2_9cbdc4f1900b/videos/` |
| v129外部方法目录 | `runs/v129_moviebench128_30s_external/raw/` |
| v129非缓存附加 | `runs/v129_noncache_addons/screen1/` |
| v129生成日志 | `runs/v129_internal_node1.log`, `v129_internal_node2.log`, `v129_internal_node3.log` |
| v129附加实验日志 | `runs/v129_addon_screen.log` |
| v125完整结果 | `runs/v125_moviebench128_main/` |
| v125 VBench结果 | `docs/128_v125_vbench_long_results.md` |
| v129设计文档 | `docs/129_no_pf_paper_comparison_and_10h_runbook.md` |
| v129附加实验文档 | `docs/130_v129_noncache_addon_experiments.md` |
