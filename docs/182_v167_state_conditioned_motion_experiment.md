# v167: State-conditioned, Deficit-triggered Motion Recall

## 1. 当前结论

v166 已经完成 `16 prompts x 6 methods`，机制审计通过，没有 cache
ownership、预算、atomic-pair 或多边形噪声错误。MultiScaleMotion 相对
MultiScaleDir 有稳定修复，说明运动幅度匹配有作用；但相对 DirectionMatch：

- proxy composite `-0.0055`；
- DINO `-0.0087`；
- first-last 在 13/16 prompts 上更差；
- motion smoothness 和 flicker 各在 11/16 prompts 上更好；
- flow speed 平均反而下降 `0.0806`；
- p5、p6、p8、p10 被 background-drift safety gate 标记。

因此 v166 不提升为主方法。当前问题不是 cache 没有执行，而是 motion-only
retrieval 会选择运动方向和幅度相似、但当前人物状态、视角或背景不兼容的历史
pair。

## 2. 为什么不能直接恢复 v161 StateMotion

v161 使用当前 descriptor 与候选 endpoint 的绝对 cosine，并设置
`state_similarity >= -0.25`。v166 trace 的离线审计显示，实际被选择候选的绝对
state cosine 集中在约 `0.981--0.999`：

- 该阈值几乎永远通过；
- 将绝对 state cosine 直接乘到 motion score 上，518 次可重放 retrieval 中
  选择改变次数为 0；
- 候选间微小差异主要携带 recency，容易退化成隐式近期偏置。

v167 不复用绝对阈值，也不复用 v161 的 direction/state/recency
lexicographic selector。

## 3. 方法

### 3.1 Reference-centered state residual

clean-KV descriptor 与 v166 相同，不增加 encoder 或训练：

```text
z_ref = 当前 sequence 的首个 descriptor
r_q   = normalize(z_current - z_ref)
r_i   = normalize(z_candidate_end - z_ref)
s_i   = cosine(r_q, r_i)
```

若 residual norm 接近 0，单次候选退回原始 state cosine；不会因为数值零向量
丢失整个 pair read。

对通过冻结 motion-direction gate 的 `N` 个候选：

1. 按 `s_i` 排序；
2. 保留 `ceil(N/2)` 个状态最兼容候选；
3. 在 shortlist 内最大化 v166 冻结的 multi-scale motion signature；
4. score 相同时选择更新的 pair。

这是候选内相对筛选，没有全局 state threshold、混合权重或训练参数。

### 3.2 Two-scale motion-deficit gate

每个在线 block 记录：

```text
m_local   = ||z_last - z_last-1||
m_context = ||z_last - z_first|| / block_steps
```

使用前序 block 的在线中位数作为两个基线。至少积累四次更新后，当且仅当：

```text
m_local   < median(history_local)
and
m_context < median(history_context)
```

才触发 episodic motion recall：

- motion deficit：使用 3.1 的 state-conditioned motion pair；
- motion healthy：读取最新的 direction-compatible atomic pair；
- 没有 direction-compatible pair：保持 v166 的 newest age-eligible fallback。

该 gate 不预测视频质量，只回答“当前是否需要旧运动状态”。阈值固定为两个尺度
各自的在线中位数，不在 16-prompt 开发集上扫描 ratio。

### 3.3 两个隔离方法

| Method | State shortlist | Deficit gate | 目的 |
|---|---|---|---|
| `ours_middle10_reservoir2_staterankmotion1` | 是 | 否 | 隔离状态兼容筛选 |
| `ours_middle10_reservoir2_deficitstaterankmotion1` | 是 | 是 | 隔离按需召回 |

## 4. 冻结 cache 契约

仅 Middle10 层使用实验 cache；其他层保持 `sink1 + recent8`。Middle10 每个
head 的读取为：

| Component | Stored | Read | Update |
|---|---:|---:|---|
| Sink | 1 frame | 1 frame | 固定首帧 |
| Temporal reservoir | 2 frames | 2 frames | 冻结 online reservoir |
| Motion-pair archive | 4 adjacent pairs | 1 pair / 2 frames | v166 admission |
| Recent | 4 frames | 4 frames | FIFO |

最大读取仍为九个 full-frame equivalents。两帧 motion pair 必须连续、成对写入、
成对读取；composition 是唯一 dynamic-history owner。以下内容全部冻结：

- prompt、seed、30 秒生成参数；
- Middle10 layer map；
- sink/recent/reservoir 容量；
- archive 写入、stale refresh 和 read age；
- direction floor `0.1`；
- dynamic RoPE 与 attention read budget。

## 5. 实验网格

逻辑网格为六种方法、16 prompts、共 96 个视频，但只新生成 32 个：

| Method | Source |
|---|---|
| `sf_native` | 复用 v166 |
| `ours_middle10_reservoir2_directionmatch1` | 复用 v166 |
| `ours_middle10_reservoir2_multiscalemotion1` | 复用 v166 |
| `ours_middle10_reservoir2_statemotionpair1_reference` | 复用 v166 |
| StateRankMotion | v167 新生成 16 个 |
| DeficitStateRankMotion | v167 新生成 16 个 |

不运行 PF，不运行 ABA，不扫描 cache 容量或阈值。

## 6. 服务器运行

### 6.1 拉取与 smoke

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git fetch origin
git switch codex/v167-state-conditioned-motion
git pull --ff-only origin codex/v167-state-conditioned-motion

# 默认 p6，只生成两个新方法，共两个视频。
bash scripts/run_v167_state_conditioned_motion_moviebench16.sh smoke
```

smoke 只检查：视频可解码、没有 polygon noise、主体没有立即崩坏、trace 中读取
始终为 0 或连续两帧。不要根据这两个视频选择方法。

### 6.2 四节点完整生成

四个节点分别设置 `NODE_RANK=0,1,2,3`：

```bash
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v167_state_conditioned_motion_moviebench16.sh preflight

NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v167_state_conditioned_motion_moviebench16.sh generate
```

preflight 每节点应报告 24 个逻辑任务，其中约 8 个新生成、16 个复用。全部完成后
只在 node 0：

```bash
NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v167_state_conditioned_motion_moviebench16.sh audit
bash scripts/run_v167_state_conditioned_motion_moviebench16.sh mechanism
```

### 6.3 自动代理指标

mechanism gate 为 true 后：

```bash
EVAL_GPUS=0,1,2,3,4,5 bash scripts/run_v167_automated_screen.sh all
```

该阶段不要求人工观看 96 个视频。

### 6.4 VBench-Long core-9

```bash
# node 0
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v167_vbench_long.sh prepare

# 四节点分别运行
NODE_RANK=<0..3> NUM_NODES=4 bash scripts/run_v167_vbench_long.sh split
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v167_vbench_long.sh preflight
NODE_RANK=<0..3> NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v167_vbench_long.sh eval

# node 0
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v167_vbench_long.sh collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v167_vbench_long.sh prepare-review
```

中断补缺只能单节点：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v167_vbench_long.sh resume-missing
```

`prepare-review` 自动选择最大 Quality Score 下降和“运动不下降时最大 Quality
Score 提升”两个 prompts，只比较主方法与 DirectionMatch，最多四个盲审视频。

## 7. 必须观察的 debug 信息

每次 retrieval 保存：

- raw state cosine、reference-residual state cosine 和两个 residual norms；
- 每个候选的 state rank、shortlist pass 与 motion signature；
- motion-only、state-ranked、newest 和最终选择；
- local/context magnitude、各自在线 median、ratio、warmup 和 deficit trigger；
- selection reason、selected age、fallback 和实际 read frame ids；
- archive 中真实 pair、sink/recent overlap、read budget 和 owner contract。

机制 gate 必须满足：

1. 两个方法均有完整 16-prompt trace；
2. state shortlist 至少一次改变 v166 motion-only 选择；
3. DeficitStateRank 同时执行 healthy 与 deficit 分支；
4. score、rank、shortlist、gate、counterfactual 和最终 pair 可从 trace 重算；
5. archive 不超过四对，read 只能是 0 或一个连续 pair；
6. 没有 budget violation、selected/read mismatch 或 contract failure。

## 8. 决策规则

1. **机制 gate 失败**：停止质量评测，只上传 trace、config 和日志。
2. **StateRank 不优于 MultiScaleMotion**：否定当前 residual descriptor，不继续调
   top-half 比例。
3. **StateRank 改善背景但运动下降**：检查 residual rank 是否等价于 recency；不
   直接提升方法。
4. **Deficit gate 优于 StateRank**：说明按需读取比持续注入历史 motion 更合理，
   保留完整主方法。
5. **Deficit gate 不优于 StateRank**：删除 gate，保留更简单的状态条件召回。
6. **两者均不优于 DirectionMatch**：冻结为负结果，下一步修改 descriptor，而非
   扫描混合权重、阈值或 cache 容量。

任何 16-prompt 结果都只是 adaptive development evidence。只有冻结方法后在新的
held-out prompt suite 上复验，才能写成论文主结果。
