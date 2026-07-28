# v129 实验进度总结（服务器被kill时的状态）

Date: 2026-07-29

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
