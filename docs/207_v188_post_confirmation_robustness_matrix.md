# v188 Post-confirmation Robustness Matrix

## 1. 目的与边界

v187 是当前方法第一次在完全未参与 v184/v186 开发的 128 条 prompt 上做正式确认。
v188 只在 v187 自动输出以下结论后解锁：

```text
freeze_method_for_replication_and_cross_model
```

v188 不继续搜索 cache 配置，而是补齐一篇论文在进入跨模型实验前最缺的三类证据：

1. 独立 seed 是否复现；
2. 30 秒效果能否延续到 60 秒后半程；
3. 效果是否真的来自特定 denoising phase，而不是随机长期采样或全程读取历史。

本轮不运行 PF baseline，不运行 ABA，不修改 v187 已冻结的方法。PF 仅作为当前 runtime
所在的代码仓库，不作为方法故事或必要对比。跨模型和 prompt switch 留给 v188 全部通过后
的下一阶段。

## 2. 冻结方法

候选方法仍是 v184/v186 自动选出的：

```text
phase_deterministic
```

三类 cache 方法始终保持相同读取预算：

| Readout | 构成 | Read budget |
|---|---|---:|
| Recent | sink1 + recent8 | 9 FFE |
| Coverage | sink1 + structured middle4 + recent4 | 9 FFE |

Clean call 始终读取 Recent；长期 Coverage 只改变冻结 schedule 指定的 noisy calls。
Landmark/Prototype 的 middle storage 为 4 FFE；若 v186 选中 Retrieval，则 archive storage
为 12 FFE、实际 read 仍为 4 FFE。存储和读取预算会分别报告，不能用“等读取预算”掩盖
额外 archive 存储。

## 3. Prompt 划分

v188 使用 v187 的 source index `128..255`，但不看任何指标或人工反馈。脚本对
`source_index + normalized prompt text` 使用固定 salt 做 SHA256 排序，再连续切成：

| Scope | 数量 | 排名区间 |
|---|---:|---:|
| Seed replication | 64 | 0..63 |
| Long 60s | 32 | 64..95 |
| Phase mechanism | 32 | 96..127 |

三个集合互斥且完整覆盖 128 条 prompt。这样不会按结果挑 prompt，也避免三项后续实验反复
使用同一小组 prompt。

## 4. 三个实验作用域

### 4.1 `replica64_seed20000`

固定 30 秒、seed `20000`，重新生成：

```text
sf_native
all_recent
phase_reservoir
phase_deterministic
```

共 `4 x 64 = 256` 个新视频。它与 v187 seed `10000` 的同 prompt 子集做 paired
cross-seed 分析，输出：

- 两个 seed 下每个主要指标的方法效应；
- aggregate sign 是否一致；
- per-prompt sign agreement 和 effect correlation；
- seed interaction 的 bootstrap CI；
- two-seed meta mean。

### 4.2 `long60_seed10000_32`

固定 seed `10000`、60 秒，重新生成与上面相同的四个方法，共 `4 x 32 = 128` 个新视频。

VBench-Long 会分别分析：

```text
full       clips 0..29
early_half clips 0..14
late_half  clips 15..29
```

主要结论看 `phase_deterministic - all_recent` 的 full 和 late-half 结果，同时报告相对
phase-Reservoir 和 SF 的结果。`late effect - early effect` 用于判断运动增益是否随时间衰减，
而不是只看 60 秒 aggregate。

因为该 scope 与 v187 使用相同 seed，另有自动 prefix audit：每个方法、每个视频从前
477 帧均匀采 16 帧，与 v187 对应 30 秒视频比较 MAE/PSNR。该结果只诊断“60 秒生成是否
保持相同前缀轨迹”，不参与方法晋级。

### 4.3 `mechanism32_seed10000`

核心三组在同一批次重新生成：

| Method | Noisy Coverage calls | 作用 |
|---|---|---|
| `phase_deterministic` | v184 获胜 phase | 冻结候选 |
| `opposite_phase_deterministic` | 相同 call 数、相反位置 | 等剂量 phase 反事实 |
| `all_noisy_deterministic` | 0,1,2,3 | 检验选择性暴露是否必要 |

若获胜 phase 为 `early1`，相反 phase 是 `late1`；`early2` 与 `late2` 互换。三组使用
完全相同的 operator、history policy、9-FFE read budget、prompt、seed 和节点轮换，差异
只剩 Coverage 出现在哪些 noisy calls。

SF、all-Recent、phase-Reservoir 从已审计的 v187 视频中映射同 prompt 复用，只作为上下文
参考。候选本身会在 v188 同批重新生成，因此候选与两个反事实之间不存在旧批次/新批次混杂。

该 scope 新生成 `3 x 32 = 96` 个视频，VBench 实际评测六组 `6 x 32 = 192` 个视频。

## 5. 总规模

| Scope | New videos | Evaluated videos | Duration |
|---|---:|---:|---:|
| Seed replication | 256 | 256 | 30s |
| Long 60s | 128 | 128 | 60s |
| Phase mechanism | 96 | 192 | 30s |
| **Total** | **480** | **576** | - |

VBench-Long core-9 共 `36 + 36 + 54 = 126` 个 method-dimension 任务。所有生成方法均在
四个节点上分片，节点间轮换方法顺序，避免 method 固定绑定 node 或固定绑定最早/最晚时段。

## 6. 自动审计

每个生成 scope 必须先通过：

- 视频数量、477/957 帧、16 FPS、832x480 和完整 decode；
- SF 日志不含 phase-cache route；
- cache 方法为 `360 heads` exclusive owner；
- noisy-call route 与冻结 schedule 完全一致；
- clean call 全部为 Recent；
- structured source kind、physical frame id、frame age 可复核；
- read budget 不超过 9 FFE；
- traceback、OOM、policy/trace warning、non-finite marker 为 0；
- 每个 shard 有 `videos / elapsed_seconds / frames / seed` runtime marker。

机制组复用的三个 v187 方法必须保持 source audit 哈希和视频索引映射，审计后才建立本地
hardlink/symlink。任何混合目录都会拒绝发布。

## 7. 自动 Gate

### 7.1 Seed replication

相对 all-Recent：

- Quality CI lower `>= -0.15`；
- Identity CI lower `>= -0.0015`；
- Dynamic mean `>= +0.015` 且 CI lower `>= -0.01`；
- Temporal CI lower `>= -0.003`；
- Dynamic aggregate sign 在两个 seed 一致；
- 四个主要指标至少三个 aggregate sign 一致。

相对 phase-Reservoir 另做 operator non-inferiority，不与上面的主要 gate 混合。

### 7.2 Long 60s

相对 all-Recent，full 与 late-half 同时约束 Quality、Identity、Dynamic 和 Temporal。
其中 late-half 要求 Dynamic mean `>= +0.015`，Identity CI lower `>= -0.002`，
Temporal CI lower `>= -0.004`。完整阈值保存在分析 JSON 的 `long_horizon_gate`。

### 7.3 Phase mechanism

候选分别对 `opposite_phase` 和 `all_noisy`：

1. Quality/Identity/Dynamic/Temporal 四项均满足预声明 non-inferiority；
2. Quality `+0.10`、Identity `+0.0005`、Dynamic `+0.02` 或 Temporal `+0.001`
   至少一项达到 mean gain。

两组反事实都通过才支持 `phase_specificity_supported`。32 prompts 的机制实验不声称通用
effect size，只回答 phase 位置和选择性暴露是否有可观测作用。

### 7.4 最终决策

三个 scope 都通过：

```text
advance_phase_structured_memory_to_cross_model
```

Seed + 60s 通过但 phase 反事实失败：保留效果，删除 denoising-phase 机理主张。Seed
replication 失败则停止当前冻结方法，不能通过人工挑视频救回。

## 8. 附加自动分析

每个 scope 还会按以下变量的 bottom/top quartile 报告候选相对 all-Recent 的效果：

- all-Recent baseline dynamic；
- all-Recent baseline identity/background；
- prompt word count。

这是 descriptive heterogeneity，只用于发现“高运动/低身份/复杂 prompt”中的薄弱区域，
不参与 method selection。人工 review 只在对应自动 gate 已通过时触发，三个 scope 合并后
最多 6 条，不需要盲审整批视频。

## 9. 服务器执行

### 9.1 准备与 smoke

先完成 v187 的 `collect` 和 `decision`，然后 node 0：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5 \
  bash scripts/run_v188_robustness_matrix_32gpu.sh smoke
NODE_RANK=0 NUM_NODES=1 \
  bash scripts/run_v188_robustness_matrix_32gpu.sh audit-smoke
```

Smoke 只需要自动审计，不要求人工 review。

### 9.2 32 卡生成

每个命令都在四个节点执行，分别设置 `NODE_RANK=0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v188_robustness_matrix_32gpu.sh generate-replica

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v188_robustness_matrix_32gpu.sh generate-long

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v188_robustness_matrix_32gpu.sh generate-mechanism
```

生成后 node 0：

```bash
NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh status
NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh audit-replica
NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh audit-long
NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh audit-mechanism
NODE_RANK=0 bash scripts/run_v188_vbench_long.sh prefix-audit
NODE_RANK=0 bash scripts/run_v188_vbench_long.sh efficiency
```

### 9.3 VBench-Long

每个 scope 先在 node 0 prepare：

```bash
for SCOPE in replica64_seed20000 long60_seed10000_32 mechanism32_seed10000; do
  NODE_RANK=0 bash scripts/run_v188_vbench_long.sh prepare "$SCOPE"
done
```

对每个 scope，四节点分别运行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v188_vbench_long.sh split <SCOPE>

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v188_vbench_long.sh eval <SCOPE>
```

最后 node 0：

```bash
for SCOPE in replica64_seed20000 long60_seed10000_32 mechanism32_seed10000; do
  NODE_RANK=0 bash scripts/run_v188_vbench_long.sh status "$SCOPE"
  NODE_RANK=0 bash scripts/run_v188_vbench_long.sh collect "$SCOPE"
  NODE_RANK=0 bash scripts/run_v188_vbench_long.sh decision "$SCOPE"
done

NODE_RANK=0 bash scripts/run_v188_vbench_long.sh aggregate
NODE_RANK=0 bash scripts/run_v188_robustness_matrix_32gpu.sh package
```

## 10. 推荐执行优先级

算力不足时按以下顺序：

1. `replica64_seed20000`；
2. `mechanism32_seed10000`；
3. `long60_seed10000_32`。

如果 seed replication 直接失败，可以停止后两项并回到方法优化；如果 replication 通过，
机制和长时长证据都应完成。不要先运行更多相似 cache trick，也不要在 v188 前加入 ABA，
否则方法变量和任务变量会再次混在一起。
