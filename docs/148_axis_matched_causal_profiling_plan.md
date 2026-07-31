# v148: Axis-Matched Causal DiT Head Profiling

## 1. 这一轮解决什么问题

v145 找到了可跨 prompt family 和 seed 复现的 K、V、policy 响应排序。
v147 只选择了 `full_semantic/k_shift` 一条排序，却同时对它执行
`recent4`、Q retrieval 和 V shift。结果只能说明这组 head 可被干预，
不能确认：

1. K、V 和 policy 是否为不同功能坐标；
2. 每条坐标是否对相应干预最敏感；
3. K 轴的结果是否只是因为其 top heads 大多属于 PF Anchor；
4. top-3 的结论是否依赖人为 head 数量。

v148 不生成反事实视频。它沿原生 Self-Forcing 轨迹，在 AR frame 117
进行只读 downstream replay，以较低成本完成上述因果归因。

## 2. 冻结的三条轴

三条轴统一来自 v145 的 `full_semantic` 因素，避免改变语义因素与改变
响应模态同时发生：

| 轴 | v145 分数 | 匹配干预 | 暂定物理含义 |
|---|---|---|---|
| K | `all_k_shift_mean` | `key_shift` | 历史地址表面对 prompt 的响应 |
| V | `all_v_shift_mean` | `value_shift` | 历史内容传输对 prompt 的响应 |
| policy | `all_policy_shift_mean` | `recent4` | 历史访问策略的可塑性 |

这些都是连续排序，不预设阈值，也不宣称天然离散类别。

## 3. 三种只读干预

所有干预只重算被选 head 的 self-attention output，未选 head 完全复制
native output。干预发生在 self-attention output projection 之前，因此
影响会自然传播到后续层以及最终 flow/x0。

### 3.1 `key_shift`

- 保留 current block；
- 保留 newest recent4 的 K/V；
- 对更旧的 K 按 frame 循环移动一位；
- V 保持原位。

该操作破坏旧 K 与 V 的对应关系，同时保留 tensor 数量、head 数量和
recent context。它与 `value_shift` 对称。

### 3.2 `value_shift`

- K 完全不变；
- recent4 的 V 不变；
- 更旧的 V 按 frame 循环移动一位。

### 3.3 `recent4`

- 仅保留 newest four historical frames 和 current block；
- 直接测试移除中期历史后的下游敏感度。

## 4. Core 交叉矩阵

每层分别按每条轴排序：

- top-3；
- bottom-3；
- middle ranks 的六个 head 随机且互斥地分成 random-0 和 random-1。

四组在每一层组成 12 个 head 的完整无交叠分区。实验包含：

```text
3 ranking axes x 3 interventions x (top, bottom)
3 matched diagonal cells x (random-0, random-1)
3 PF-matched diagonal controls x (high, low)
= 30 probes
```

加 native replay 后，每个 denoising state 为 31 次 replay。只测
noisy-1000 和 noisy-500，共 62 条 downstream records/profile。

## 5. PF 类内控制

PF 标签不参与 K/V/policy 分数或 top/bottom 排名，仅用于事后控制。

每一层内：

1. 在 PF Wave、Anchor、Veil 中寻找至少有两个 head 的类别；
2. 选择该层内分数跨度最大的 PF 类别；
3. 在同一个 PF 类别中选择最高分和最低分 head；
4. 对二者执行该轴的匹配干预。

因此每层 high/low：

- PF 标签完全相同；
- head 数量相同；
- layer 完全相同；
- 唯一系统差异是我们的轴分数。

若该比较仍通过，才能认为结果不只是重新发现 PF Anchor。

真实 v145 数据生成的 PF-matched 类别数量为：

| 轴 | Wave layers | Anchor layers | Veil layers |
|---|---:|---:|---:|
| K | 12 | 18 | 0 |
| V | 15 | 14 | 1 |
| policy | 16 | 13 | 1 |

## 6. 预注册门控

所有主要门控必须在同一个 noisy context 内成立。不能用 t1000 的
top-vs-random 与 t500 的 top-vs-bottom 拼成一个 PASS。

单个 paired comparison 通过需要：

- median paired log-ratio 至少为 `log(1.05)`；
- prompt bootstrap 95% mean CI 下界大于 0，或 paired win rate 不低于
  0.65；
- 两个 seed replicate 的 prompt-level Spearman 不低于 0.30。

### G1: axis-matched causal effect

同一轴、同一匹配干预、同一 context 中同时满足：

- top > bottom；
- top > geometric mean(random-0, random-1)。

### G2: PF-independent effect

同一层、同一 PF 标签的 high-score head > low-score head。

### G3: intervention specificity

同一轴的 `top/bottom` separation 在匹配干预上，需要同时大于该轴的
另外两种非匹配干预。

G1 可以证明排序有功能意义；G2 才能排除 PF 类别混杂；G3 才允许把
K-address、V-content 或 policy-plasticity 写成不同机制。

## 7. Dose 阶段

Dose 阶段使用前 16 个 prompts 和两个 seeds，共 32 profiles。对每条轴
的匹配干预分别运行：

```text
top-1 vs bottom-1
top-2 vs bottom-2
top-3 vs bottom-3
top-4 vs bottom-4
```

每个 k 均为 equal-count comparison。只看绝对扰动随 head 数增长没有
意义；需要观察 top-k/bottom-k separation 是否在多个 k 上成立，以及
dose-4 separation 是否稳定大于 dose-1。

Dose 应在 core 至少有一条轴通过 G1 后运行。代码已提前准备，避免 core
完成后再次等待开发。

## 8. 实验规模

### Core

- prompts: MovieBench Qwen Rewrite 的多样 32 条；
- seeds: 2；
- profiles: 64；
- length: 120 latent frames，约 30 秒；
- GPUs: 4 nodes x 8 GPUs；
- 每张 GPU: 2 profiles。

### Dose

- prompts: core 的前 16 条；
- seeds: 2；
- profiles: 32；
- 每张 GPU: 1 profile。

## 9. 运行命令

### 9.1 准备与 smoke

仅在 node 0：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh smoke_core
```

必须看到：

```text
[v148-smoke] replay/cache/axis-probe contract: PASS
```

### 9.2 Core，四个节点各运行一条

```bash
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh core64
NODE_RANK=1 bash scripts/run_v148_axis_causal_profile_32gpu.sh core64
NODE_RANK=2 bash scripts/run_v148_axis_causal_profile_32gpu.sh core64
NODE_RANK=3 bash scripts/run_v148_axis_causal_profile_32gpu.sh core64
```

完成后仅在 node 0：

```bash
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh audit_core
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh analyze_core
```

### 9.3 Dose，可与 core 分开运行

四个节点各运行：

```bash
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh dose32
NODE_RANK=1 bash scripts/run_v148_axis_causal_profile_32gpu.sh dose32
NODE_RANK=2 bash scripts/run_v148_axis_causal_profile_32gpu.sh dose32
NODE_RANK=3 bash scripts/run_v148_axis_causal_profile_32gpu.sh dose32
```

完成后：

```bash
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh audit_dose
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh analyze_dose
NODE_RANK=0 bash scripts/run_v148_axis_causal_profile_32gpu.sh package
```

进度：

```bash
bash scripts/run_v148_axis_causal_profile_32gpu.sh status
```

## 10. Debug 与正确性信息

日志必须包含：

```text
[HeadProfile] begin ... downstream=1
[HeadProfile] downstream-probe ... name=... policy=...
[HeadProfile] end ... records=60
```

Smoke 和 audit 会检查：

- native replay `flow/x0 relative RMS <= 1e-4`；
- self/cross K/V pointer、version 和 cache index 未变化；
- `key_shift/value_shift` 实际移动超过一个 old frame；
- core 每个 profile 正好 62 条 downstream records；
- dose 每个 profile 正好 50 条 downstream records；
- 每个 prompt 恰有两个不同 seeds；
- 任何 traceback、OOM、assertion 或 runtime error 都会使 audit 失败。

## 11. 如何解释结果

| 结果 | 结论 |
|---|---|
| K-G1 pass，K-G2 fail | K 排名有作用，但可能只是 PF temporal class proxy |
| K-G1/G2 pass，G3 fail | 有 PF 独立的历史敏感度，但不能命名为独立 K-address 机制 |
| V-G1/G2/G3 pass | 可以支持独立 content-transport coordinate |
| policy-G1/G2/G3 pass | 可以支持 cache-policy plasticity coordinate |
| 只有某个 timestep pass | 使用 timestep-conditioned continuous routing |
| 三轴 G1 均 fail | 停止从 v145 扰动分数构造静态 head routing |

`x0_relative_rms` 是干预敏感度，不是视频质量。任何 PASS 均需后续
trajectory-level 单视频和多 prompt 评测。

## 12. 与长程 retrieval 的关系

v148 仍然只干预原生 SF 滑窗中的历史，用于干净地确认功能轴。它不回答
跨 30 秒的 recall 是否有效。

仓库已有只读 persistent K/V capture，可在早期 clean blocks 保存稀疏
K/V，并在 frame 117 读取而不修改 native cache。下一轮应单独实现：

- early/middle archive + recent4；
- archive uniform retrieval；
- archive Q retrieval；
- equal token budget；
- K、V、policy 通过 v148 的轴分别路由。

只有 v148 先确认哪个轴对应哪种干预，长程 archive 实验才有明确归因。

## 13. 对应代码

- `src/lifecycle_kv/downstream_probe.py`
- `scripts/build_v148_axis_causal_suite.py`
- `scripts/analyze_v148_axis_causal_profiles.py`
- `scripts/run_v148_axis_causal_profile_32gpu.sh`
- `tests/test_v148_axis_causal_suite.py`
- `tests/test_v148_axis_causal_analysis.py`
