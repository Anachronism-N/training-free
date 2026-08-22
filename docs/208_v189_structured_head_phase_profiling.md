# v189 Structured Head x Denoising-Phase Profiling

## 1. 最新同步结论

本轮已合入远端 `main` 的 v182 完整 core-9 结果。v182 使用旧 strict-five head map，
所以不能证明该静态分类有效；它可以回答的是 structured Coverage operator 是否值得继续：

| Method | Quality | Identity | Temporal | Semantic | Visual | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| all-Recent | 83.1411 | 0.96521 | 0.96871 | 0.23431 | 0.66232 | 0.68750 |
| strict5-Landmark | 83.7484 | **0.96535** | **0.96946** | 0.23588 | **0.66593** | 0.74167 |
| strict5-Retrieval | **83.8280** | 0.96522 | 0.96829 | **0.23757** | 0.66515 | **0.77083** |

Landmark 和 Retrieval 均在 v182 的 Pareto front：Landmark 更偏身份/时序，Retrieval
更偏质量/语义/运动。该结果支持重新 profile 这两个算子，但不允许继续复用 strict-five
membership。

## 2. 为什么组合 head 与 denoising phase

此前两条结果并不矛盾：

1. v183 表明全头 Coverage 是明显的 motion actuator，但全程使用会损伤身份；
2. v177 的静态五头分类在生成侧失败，其分类又强制 head 在所有 denoising calls 中选择
   相同算子，可能把真正的 phase-specific 单元过滤掉。

v189 因此不再为每个 head 生成一个永久标签，而是估计四维路由：

```text
R[operator, denoising_call, layer, head] in {Recent, Coverage}
```

同一个 head 可以在 call 0 使用 Coverage、在 call 1-3 使用 Recent。跨 call 一致性不再是
gate；相反，call 间差异本身是要检验的机制。

## 3. 公平的 representation-complete teacher

两个候选具有相同读取预算：

```text
Recent   = sink1 + recent8                         = 9 FFE
Coverage = sink1 + structured middle4 + recent4  = 9 FFE
Union    = sink1 + recent8 + structured middle4 <= 13 FFE
```

Union 不是生成方法，只是 shadow teacher。它必须包含两个候选实际使用的全部
`(physical frame, K representation family)`，包括 saved/time-mapped/dynamic-RoPE 的区别。
每条记录执行 `2 candidates x 12 heads = 24` 次 representation-superset 检查；任意缺失
都会直接终止任务。

Landmark 与 Retrieval 分开 profile，各自的 Union 只包含自身候选需要的 representation，
不再混入旧 Reservoir、Episode 或 motion-pair 表征。Active trajectory 始终读取 Recent，
所有 candidate/Union 仅做 shadow readout，不改变 latent、clean commit 或最终视频轨迹。

## 4. Profiling 规模

- prompts：Qwen-rewritten MovieGen 128；
- 时长参数：120 latent frames，约 30 秒，但跳过视频 decode；
- noisy calls：0/1/2/3 全部记录；
- head cells：`4 x 30 x 12 = 1440`；
- 每个 prompt/call/layer 记录 12 个 AR 位置；
- 每个算子 184,320 records，两个算子共 368,640 records；
- 固定 split：64 discovery、32 validation、32 classifier-holdout generation set；
- 为完整归档会采集全部 128 prompts 的 shadow readout，但 analyzer 不读取最后 32 条来设阈值、选 operator 或拟合 head-phase map；
- 32 卡并行：每个节点前半 GPU 跑 Landmark，后半 GPU 跑 Retrieval，两个算子各 16 卡。

## 5. 分类过程

对每个 `operator/call/layer/head` 计算：

```text
gain = log(error_Recent_to_Union) - log(error_Coverage_to_Union)
```

正值表示该 cell 使用 Coverage 更接近 representation-complete teacher。Primary
`compatible` map 使用以下预声明 gate：

- discovery mean gain `>= 0.02`，约等于至少 2% 的几何误差优势；
- validation mean gain `>= 0`；
- validation 95% CI lower `>= -0.01`，这是开发期非灾难容忍，不是论文显著性界限；
- validation prompt win fraction `>= 0.60`；
- 完整 9-FFE Coverage budget 比例 `>= 0.80`；
- Union residual energy 至少为同层中位数的 `0.10`，避免选择几乎不贡献 residual 的头。

`phase_selective` map 还要求：

- 当前 call 的 gain 相对另外三个 calls 的 discovery mean contrast `>= 0.01`；
- validation phase contrast 为正且 prompt win fraction `>= 0.55`。

它是机理消融，不替代 primary map。分析器另输出固定 `top12_discovery` map 和
`gain x win-fraction` threshold sensitivity；这些控制在看到视频前生成，不能根据人工
review 事后挑选。

同一份 discovery/validation profile 还会冻结两个不使用联合交互的 factor control：

- `head_only_compatible`：先在四个 noisy calls 上平均每个 layer/head 的 gain，再用与
  primary 相同的 gain、validation、budget 和 residual-energy gate 分类；所得 head
  membership 在四个 calls 中完全相同；
- `phase_layer_only_compatible`：先在 12 个 heads 上平均每个 call/layer 的 gain，
  通过同一组 gate 后，该 call/layer 的全部 heads 一起读取 Coverage。

它们不使用最后 32 条 generation holdout。`Head x Phase` 只有在生成侧同时优于这两个
因子化方法时，才支持“联合路由不可被单独的 head 或 denoising-phase 分类替代”。

## 6. 新增 runtime 与 debug

新增 schedule：

```text
--pyramidkv_cache_compatibility_denoise_schedule head_phase
--pyramidkv_cache_compatibility_head_phase_map <map.json>
```

map 必须严格为 `4 x 30 x 12` boolean tensor，并绑定 Coverage operator。每个 layer/call
可以同时存在 Recent heads 和 Coverage heads。Clean pass 无论 map 内容如何都强制全
Recent。

schedule trace 新增：

```text
phase_map_id
coverage_heads
recent_heads
coverage_head_indices
```

readout trace 为抽样 head 记录其真实 `effective_policy`、physical frame IDs、frame ages、
source kind 和 FFE。Runtime 会对全部 360 heads 做预算与 middle-leak 检查，而不是只检查
trace 中的抽样头。

## 7. 服务器执行

### 7.1 准备与 smoke

Node 0：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1 \
  bash scripts/run_v189_structured_head_phase_profile_32gpu.sh smoke
```

Smoke 同时运行两个算子，各一个 prompt。必须看到：

```text
[v189-smoke] PASS operators=2 records_per_operator=1440
```

不需要人工 review。

### 7.2 32 卡 profile

四个节点分别设置相对 `NODE_RANK=0,1,2,3`：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v189_structured_head_phase_profile_32gpu.sh profile128
```

每节点 GPU 0-3 属于 Landmark，GPU 4-7 属于 Retrieval。不要让不同节点使用不同的
`GPU_LIST` 排列。

完成后 Node 0：

```bash
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh status
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh audit
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh analyze
NODE_RANK=0 bash scripts/run_v189_structured_head_phase_profile_32gpu.sh package
```

## 8. 需要回传的文件

无需上传 `.pt` profiles 或视频。优先推送：

```text
runs/v189_structured_head_phase_profile/inputs/
runs/v189_structured_head_phase_profile/profile_audit.json
runs/v189_structured_head_phase_profile/analysis/analysis.json
runs/v189_structured_head_phase_profile/analysis/analysis.md
runs/v189_structured_head_phase_profile/analysis/cell_scores.csv
runs/v189_structured_head_phase_profile/analysis/factor_scores.csv
runs/v189_structured_head_phase_profile/analysis/maps/
```

若失败，再上传对应 operator 的 log 和诊断压缩包。

## 9. v190 Classifier-Holdout Causal Screen

只有 `analysis.json` 输出：

```text
advance_head_phase_maps_to_causal_screen
```

才进入未参与 map fitting 的 32-prompt generation set。v190 已实现并自动比较：

1. all-Recent；
2. all-Coverage，即四个 noisy calls 的全部 360 heads 都读取长期历史；
3. primary compatible Head x Phase map；
4. call-invariant Head-only map；
5. head-invariant Phase/Layer-only map；
6. 同 call、同 layer 数量的确定性 membership shift；
7. 将同一 membership 循环平移到其他 denoising calls 的 phase control；
8. primary 激活的相同 call/layer cells 上使用 all-head 的 dense control。

这些方法分别检验 operator utility、Head x Phase 交互、head membership、phase
membership 和稀疏暴露。具体方法集合由 v189 自动冻结；若一个 control map 与 primary、
all-Recent、all-Coverage 或已经注册的 control 完全相同，prepare 会将其删除，不会运行
无信息的重复视频，并在分析时复用其 exact-equivalent method。若 factor map 与 primary
本身相同，则联合交互不可识别，相应机制 claim 自动判定为未获得支持。

这里的 `all-Recent` 是同一 shadow-cache runtime 下的 9-FFE 局部读取控制，不等同于
原生 SF。它用于隔离 Head x Phase 路由的因果作用；只有 v190 通过后，新的 128-prompt
正式评测才加入 SF native 等外部 baseline。

v189 通过后，Node 0：

```bash
NODE_RANK=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh prepare
NODE_RANK=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh preflight

NODE_RANK=0 NUM_NODES=1 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v190_head_phase_causal_screen_32gpu.sh smoke
NODE_RANK=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh audit-smoke
```

Smoke 自动按 GPU 数分 wave，不要求 method 数小于 8，也不需要人工 review。随后四节点：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v190_head_phase_causal_screen_32gpu.sh generate32
```

生成后 Node 0：

```bash
NODE_RANK=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh status
NODE_RANK=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh audit-screen
NODE_RANK=0 bash scripts/run_v190_vbench_long.sh prepare
```

四节点分别执行：

```bash
NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v190_vbench_long.sh split

NUM_NODES=4 NODE_RANK=<0|1|2|3> GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v190_vbench_long.sh eval
```

最后 Node 0：

```bash
NODE_RANK=0 bash scripts/run_v190_vbench_long.sh status
NODE_RANK=0 bash scripts/run_v190_vbench_long.sh collect
NODE_RANK=0 bash scripts/run_v190_vbench_long.sh decision
NODE_RANK=0 bash scripts/run_v190_head_phase_causal_screen_32gpu.sh package
```

`collect` 会在缺失时自动运行 `compute_temporal_jump_diagnostic.py`，以 8-frame step
计算配对的光流、后段运动衰减、低运动区间、时序跳变和亮度/边缘异常。该结果只用于
自动拒绝重复伪影并挑选最多 4 条 review prompt，不作为论文效果指标。也可在 VBench
结束前单独运行：

```bash
NODE_RANK=0 V190_TEMPORAL_WORKERS=8 \
  bash scripts/run_v190_vbench_long.sh temporal
```

分析器会先检查 VBench Dynamic Degree 是否有方差。若所有方法、所有 prompt 都为
`1.0`，该维度只能记为运动在指标天花板处未退化，不能声称运动提升；此时 primary
还必须在质量、身份/背景或时序轴至少有一项正增益。若 Dynamic Degree 有区分度，
则仍要求相对 all-Recent 至少提升 `0.01`。

只有 primary 相对 all-Recent 通过基本质量/身份/运动/时序 gate，同时优于 Head-only、
Phase/Layer-only、membership-shift 与 phase-shift，并在减少 Coverage cell-call 暴露量
的前提下不劣于 all-Coverage，才输出：

```text
advance_head_phase_method_to_fresh128
```

人工 review 只在该 gate 通过后查看自动选择的最多四条冲突样本。

## 10. 当前论文边界

目前可以写成待验证假设：

> 不同 self-attention heads 对结构化长期历史的兼容性随 denoising phase 改变；因此，
> 训练免费的长视频外推应在 operator、head 与 phase 三个维度联合路由历史，而不是为
> head 分配永久的时间类别。

v182 只证明两个 structured operator 值得重新 profile；v189 只能证明 shadow residual
compatibility。必须通过 classifier-holdout generation set、count-matched random 和
phase-shift control 后，才能把 Head x Phase routing 写成论文贡献。
