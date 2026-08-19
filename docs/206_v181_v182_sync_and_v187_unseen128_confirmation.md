# v181/v182 同步结论与 v187 Unseen-128 Confirmation

## 1. 本轮同步状态

2026-08-20 已将远端 `main` 的以下结果合入当前开发分支：

- v181 `long60_seed0`：128 prompts、3 methods、60 秒、core-9 完整；
- v181 `long60_seed10000_64`：64 prompts、3 methods、60 秒、core-9 完整；
- v182 structured Coverage：5 methods x 16 prompts，共 80 个约 30 秒视频，
  生成和自动审计完整。

本机没有推理环境，本轮只完成结果复核、静态测试和后续服务器代码，不在本机生成
视频或运行 VBench。

## 2. v181 的新结论

### 2.1 60 秒 aggregate 指标

| Scope | Method | Quality | Identity | Temporal | Dynamic |
|---|---|---:|---:|---:|---:|
| seed 0, 128 | SF native | 80.1904 | 0.96644 | 0.98060 | 0.33151 |
| seed 0, 128 | RCCP | 81.3364 | 0.96464 | 0.97201 | 0.55573 |
| seed 0, 128 | all-Recent | **81.4070** | **0.96670** | **0.97595** | 0.50547 |
| seed 10000, 64 | SF native | 81.2477 | 0.96930 | 0.98258 | 0.36458 |
| seed 10000, 64 | RCCP | **82.2875** | 0.96900 | 0.97675 | **0.51354** |
| seed 10000, 64 | all-Recent | 82.2299 | **0.96982** | **0.97918** | 0.47969 |

RCCP 相对 all-Recent 的 paired full-video 差异：

| Scope | dQuality, 95% CI | dIdentity, 95% CI | dDynamic, 95% CI |
|---|---:|---:|---:|
| seed 0, 128 | -0.0706 `[-0.5910, 0.3501]` | -0.00206 `[-0.00335, -0.00113]` | +0.0503 `[0.0188, 0.0820]` |
| seed 10000, 64 | +0.0576 `[-0.2662, 0.3809]` | -0.00081 `[-0.00142, -0.00017]` | +0.0339 `[-0.0089, 0.0792]` |

两个作用域均给出 `long_horizon_rccp_not_confirmed`。因此可以保留的事实是：长期
Coverage 能提高运动，但冻结的静态 RCCP membership 没有稳定优于 all-Recent，且
身份/背景一致性存在可重复下降。不能继续把 v177 五头分类作为最终方法核心。

## 3. v182 当前能回答什么

v182 五组生成均通过审计：

```text
all_recent
strict5_reservoir
strict5_landmark
strict5_prototype
strict5_retrieval
```

共 80/80 个视频可完整解码；检索算子 trace 包含 19,200 条调用记录、4,800 条选中
head 记录，真实 middle readout、dynamic RoPE、9-FFE 预算和 exclusive ownership
均通过。当前仓库没有 v182 core-9 指标。

v182 使用后来已被 v183 否定的 static strict-five membership，所以其评测只能作为
算子稳定性和候选优先级证据，不能直接晋级论文方法。已有视频不应重生成，先执行：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 bash scripts/run_v182_vbench_long.sh prepare

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v182_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v182_vbench_long.sh eval

NODE_RANK=0 bash scripts/run_v182_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v182_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v182_vbench_long.sh decision
```

这些命令只评测服务器已有的 80 个视频。

## 4. 主实验顺序没有改变

当前主线仍必须按因果顺序执行：

1. **v184**：确定长期 Coverage 应注入哪些 noisy denoising calls；
2. **v186**：在同一获胜 schedule 下比较 Reservoir 与三种确定性 Coverage 算子；
3. **v187**：仅在 v184 和 v186 都自动晋级后，在未见 128 prompts 上确认。

v182 评测可以与 v184 生成并行，但 v182 结果不能绕过 v184/v186 gate。

## 5. v187 的实验问题

v187 回答两个相互独立的问题：

1. 冻结的 phase-conditioned deterministic Coverage 是否在未见 prompt 上优于
   all-Recent；
2. 确定性算子是否能在保持 phase-Reservoir 运动作用的同时改善身份、时序或总体
   质量，从而支撑“不是随机长期采样本身”的方法归因。

Prompt 使用 v180 已冻结但未参与 v184/v186 开发的 MovieGen source index
`128..255`。准备脚本会验证：

- 128 条文本完整且哈希匹配；
- 与 v186 development32 精确文本重合为 0；
- v186 decision、input manifest、generation contract、published manifest 和被选中
  算子的 audit 均未漂移；
- 只有 `advance_deterministic_operator_to_fresh128` 能解锁 v187。

## 6. v187 四个方法

四组均重新生成，不复用旧视频，固定 seed `10000`，避免旧 runtime 和新 runtime
混杂：

| Method | Noisy read | Clean read | 作用 |
|---|---|---|---|
| `sf_native` | 原生 SF | 原生 SF | 外部基线 |
| `all_recent` | sink1 + recent8 | Recent | 等预算局部控制 |
| `phase_reservoir` | 获胜 schedule 下 sink1 + reservoir4 + recent4 | Recent | 随机长期覆盖参考 |
| `phase_deterministic` | 同 schedule 下被 v186 选中的 structured middle4 | Recent | 最终候选 |

所有 phase-cache 方法读取预算均为 9 FFE。`phase_deterministic` 可自动实例化
Landmark、Prototype 或 Retrieval；若选中 Retrieval，会明确报告 12 FFE archive、
4 FFE read，不把额外存储隐藏在结果中。

四个节点会按 `NODE_RANK` 轮换方法执行顺序。每种方法仍在全部四个节点上生成，
因此方法不会固定绑定某个节点，也不会全部固定绑定同一运行时段。

## 7. v187 服务器命令

### 7.1 前置条件

先完成 v184 和 v186 的 `decision`。v186 必须输出：

```text
advance_deterministic_operator_to_fresh128
```

然后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v187_unseen128_confirmation_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v187_unseen128_confirmation_32gpu.sh preflight
```

### 7.2 四方法 smoke

只做自动解码和 trace 审计，不要求人工排序：

```bash
NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3 \
  bash scripts/run_v187_unseen128_confirmation_32gpu.sh smoke

NODE_RANK=0 NUM_NODES=1 \
  bash scripts/run_v187_unseen128_confirmation_32gpu.sh audit-smoke
```

### 7.3 32 卡生成 4 x 128

四个节点分别设置相对 `NODE_RANK=0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v187_unseen128_confirmation_32gpu.sh generate128
```

全部完成后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v187_unseen128_confirmation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v187_unseen128_confirmation_32gpu.sh audit-confirm
NODE_RANK=0 bash scripts/run_v187_vbench_long.sh prepare
```

### 7.4 VBench-Long core-9

四节点分别执行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v187_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v187_vbench_long.sh eval
```

最后在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v187_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v187_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v187_vbench_long.sh decision
NODE_RANK=0 bash scripts/run_v187_unseen128_confirmation_32gpu.sh package
```

评测共 `4 methods x 9 dimensions = 36` 个任务。

## 8. 自动审计与停止条件

生成审计要求：

- 512/512 个视频均为 477 帧、16 FPS、832x480 且可完整解码；
- SF 日志不得出现 phase-cache route；
- 三个 cache 方法必须为 360 heads 的 exclusive owner；
- noisy call 的 Recent/Coverage 路由与冻结 schedule 完全一致；
- clean call 始终读取 Recent；
- structured middle 来源、physical frame id、frame age 和 9-FFE 预算可复核；
- 任意 traceback、OOM、预算漂移或 trace warning 都阻止发布。

确认分析使用 paired 128-prompt bootstrap CI。主要 gate 在看到 v187 视频前冻结：

### 方法相对 all-Recent

- Quality CI 下界 `> 0`；
- Identity CI 下界 `>= -0.001`；
- Dynamic mean `>= +0.02` 且 CI 下界 `>= 0`；
- Temporal CI 下界 `>= -0.002`；
- 候选位于四指标 Pareto front。

### 确定性算子相对 phase-Reservoir

- Quality / Identity / Dynamic / Temporal CI 下界分别不低于
  `-0.15 / -0.0005 / -0.02 / -0.001`；
- Quality `+0.10`、Identity `+0.0005` 或 Temporal `+0.001` 的 mean gain 至少一项
  成立。

只有两组 gate 同时通过，才输出最多 6 个定向冲突样本供人工 review，并进入 seed
复现、60 秒和跨模型实验。失败时不通过盲审挑视频修改结论。

## 9. 当前论文边界

当前可写的事实仍是“长期历史读取是运动 actuator，但静态 head membership 没有
确认”。只有 v184、v186、v187 连续通过后，方法故事才能升级为：

> 在不改变 clean memory commit 的情况下，训练免费地把确定性结构化长期记忆仅暴露
> 给特定去噪阶段，从而解除局部窗口的运动衰减，并减少全程长期读取带来的身份和
> 时序代价。

这条故事与 PF 的 Anchor/Wave/Veil 三分类不同；它的核心变量是 denoising phase 和
structured Coverage operator，而不是沿用 PF 的三类 head 路由。跨模型适配性、ABA
场景切换和 60 秒最终结果仍需后续独立验证。
