# v208：冻结候选的 128-prompt 30/60 秒复测

日期：2026-09-15。分支：`codex/v178-v179-causal-validation`。

## 1. 本轮交付与当前结论

v201 的结论仍然是 `do_not_advance_v201_no_sf_gain`。当前没有服务器结果证明
v207 已通过数值 parity 或相对 SF 有收益。v208 代码已补齐，但执行有两个前置条件：

1. v207 数值 parity 报告为 `parity_pass=true`。
2. v207 32-prompt screen 自动选出唯一的 `retrieval21_early1` 或 `retrieval21_early2`。

任一条件不满足，`prepare` 拒绝启动。v208 不读取人工指定的获胜方法，不根据自身结果重新选型。
当前机器只做静态和 CPU 单元检查；真实 GPU 推理与 VBench 在服务器执行。

参考了用户提供的外部 `225_sf_parity_budget_selector_experiment_plan.md`，采用其中的 runtime、
预算和 selector 分开归因的思路。外部文档的编号和命令没有直接替代仓库已有实验编号。

## 2. 冻结方法与规模

| 方法 | Noisy attention | Clean refresh | 读取预算 |
|---|---|---|---:|
| `sf_native` | 原生 SF 滑窗 | 原生 SF | 21 latent-frame equivalents |
| `recent21` | sink1 + recent20 | Recent | 21 FFE |
| `ours` | 指定 early calls 使用 sink1 + retrieval4 + recent16，其余 Recent | Recent | 21 FFE |

`early1` 指四次 denoising 中的 call 0；`early2` 指 call 0、1。索引从 0 开始。
所有 360 heads 使用相同 schedule；本轮不声称发现新的 head 类别。Retrieval archive 上限为
12 个 latent-frame equivalents，读取时最多使用其中 4 帧。读取预算不等于总存储或峰值显存。

两个可独立执行的 scope：

| Scope | Prompts | Latent frames | 解码帧数 / FPS | 方法数 | 视频数 |
|---|---:|---:|---|---:|---:|
| `main30` | 128 | 120 | 477 / 16，约 29.8 秒 | 3 | 384 |
| `long60` | 128 | 240 | 957 / 16，约 59.8 秒 | 3 | 384 |

使用指定的 Qwen 改写版 MovieGen-128。seed 固定为 20800，逐 prompt reseed，同一 scope
三个方法严格配对。30/60 秒也使用相同 seed，因此 60 秒是长时外推检验，不能视为独立 seed 复现。
此 seed 与 v207 的 20700 不同。默认 4 节点，每节点 8 GPU；每个方法每卡生成 4 个视频。
不承诺两个 scope 加评测能在 10 小时内完成，应先完成 `main30`。

## 3. 128 条不是全新的确认集

v207 使用 source indices `1+4k`，共 32 条。v208 保存精确 split：

- `v207_development32`：上述 32 条，仅描述性复测。
- `v208_holdout96`：其余 96 条，用于本轮主要 paired tests。
- `all128`：完整基准汇总和整体安全检查，便于报告 benchmark 表格。

96 条只是相对 v207 选型未使用，可能已经用于 v129 等历史实验。代码明确输出
`historically_fresh_prompts_verified=false`。它们不能写成从未用于调参的新测试集。
若本轮有稳定收益，后续再补一套真正无历史重叠的 prompts；不能通过调整本轮 split 获得更好结果。

## 4. 运行顺序

### 4.1 前置检查

正在运行的 v207 不需要重启。先等待它完成数值 parity 和 32-prompt screen。
数值 parity 操作见 `docs/226_v207_sf_runtime_parity_supplement.md`；screen 见 `docs/225_v207_equal_budget_phase_retrieval_and_paper_plan.md`。

注意：现有 parity 的 9 latent frames 只覆盖前三个 AR blocks，尚未覆盖 21 帧窗口满载后的 eviction。
它是启动阶段的必要检查，不是所有长程缓存状态均已等价的证明。若启动阶段有分叉，先定位修复；
若通过，应在声称完整 SF emulation 前补满窗后的边界 trace。

等待所有相关进程退出后拉取新代码：

```bash
git pull --ff-only origin codex/v178-v179-causal-validation
```

四个节点使用同一 checkout、共享运行目录及模型文件。节点 0 执行：

```bash
NODE_RANK=0 bash scripts/run_v208_paper_confirmation_32gpu.sh prepare
```

默认读取：

```text
prompts:
/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt

checkpoint:
/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt

v207 parity:
runs/v207_context_budget_phase_recovery/parity/report/parity_report.json

v207 selection:
runs/v207_context_budget_phase_recovery/screen32/analysis/v207_context_budget_phase.json

v208 output:
runs/v208_paper_confirmation/
```

路径覆盖变量：`V208_SOURCE_PROMPTS`、`SHARED_CHECKPOINT`、`V207_OUT_ROOT`、`V207_REPORT`、
`V207_PARITY_REPORT`、`V207_PARITY_INPUTS`、`V208_OUT_ROOT`。保持 parity report 和 inputs 属于同一运行目录。
默认 conda 环境为 `longlive`；`CONDA_SH` 和 `CONDA_ENV` 可覆盖。

`prepare` 首次计算 checkpoint SHA256；后续检查复用 SHA 并验证路径、大小和 mtime。
修改任何推理代码、冻结配置、checkpoint 或 v208 脚本后，旧输出目录会被拒绝复用。

### 4.2 30 秒主复测

在四个节点分别执行，将 `NODE_RANK` 设为各自的 0、1、2、3：

```bash
NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v208_paper_confirmation_32gpu.sh generate30
```

可用 `METHODS=sf_native,recent21` 和随后 `METHODS=ours` 分批执行，所有节点必须采用相同分片设置。
完成的分片会跳过；中断但未完成的分片会清理自身视频和 trace 并重新生成，以保持日志、计时和视频对应。
默认最多重做该分片的 4 条，不重做其他已完成分片。不要在已经 audit/publish 后使用 `FORCE=1`；
需要重做已发布结果时使用新的 `V208_OUT_ROOT`，防止旧 hardlink 指向已替换的视频。

所有节点完成后，节点 0 执行：

```bash
bash scripts/run_v208_paper_confirmation_32gpu.sh status
NODE_RANK=0 bash scripts/run_v208_paper_confirmation_32gpu.sh audit30
```

audit 检查媒体、运行配置、phase route、真实 middle readout 和逐分片 provenance，随后产生轻量效率表。
可选 `smoke` 为 3 方法各一条完整 30 秒视频，不是必需的人工 review 门禁。

### 4.3 VBench-Long

先激活服务器环境，节点 0 准备评测输入：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
SCOPE=main30 NODE_RANK=0 bash scripts/run_v208_vbench_long.sh prepare
```

四节点各自运行 split，全部完成后再各自运行 eval：

```bash
SCOPE=main30 NODE_RANK=0 NUM_NODES=4 bash scripts/run_v208_vbench_long.sh split
SCOPE=main30 NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_v208_vbench_long.sh eval
```

其中 `NODE_RANK` 同样替换为所在节点编号。缺少的评测项可用 `resume-missing` 继续。
全部评测完成后节点 0 执行：

```bash
SCOPE=main30 NODE_RANK=0 bash scripts/run_v208_vbench_long.sh collect
SCOPE=main30 NODE_RANK=0 bash scripts/run_v208_vbench_long.sh decision
```

`collect` 在缺少 temporal CSV 时自动计算并绑定来源；输出 all128/development32/holdout96
的 full、early-half、late-half 配对结果。统计单位为 prompt，不能把 split clip 当独立样本。
core-9 不覆盖 VBench 的全部语义维度，所以不能把该子集标成官方完整 Semantic/Total Score。

### 4.4 60 秒与相机补偿运动

主复测完成后，资源允许再在四节点运行：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v208_paper_confirmation_32gpu.sh generate60
```

全部结束后 `audit60`，评测的全部命令把 `SCOPE=main30` 改为 `SCOPE=long60`。
两组没有跨时长结果选择逻辑；应报告全部结果，不能用 60 秒的偶然收益掩盖 30 秒失败。

相机补偿运动沿用已有 v193 入口。对应 scope 完成 `collect` 后四节点分别执行，最后节点 0 collect：

```bash
SCOPE=main30 NODE_RANK=0 NUM_NODES=4 bash scripts/run_v208_vbench_long.sh motion-compute
SCOPE=main30 NODE_RANK=0 bash scripts/run_v208_vbench_long.sh motion-collect
SCOPE=main30 NODE_RANK=0 bash scripts/run_v208_vbench_long.sh motion-analyze
```

这部分是独立运动证据，不会被 VBench 的自动支持判定替代。运动模块使用 all128 的匹配质量汇总，
用于描述性诊断；主统计支持仍由 holdout96 的规则决定。

## 5. 判定与需要观察的输出

主要统计在 holdout96 上完成，对 SF 和 Recent21 分别检查：

1. full 和 late-half 的所有预设质量轴通过非劣检查。
2. 至少一个非 DD 主轴有正 bootstrap 下界且 BH 校正 `q<=0.05`。
3. 自动时序安全检查通过，且 all128 不出现超额时序失败。

主轴：固定 DD 的质量分、身份/背景、时序机制、语义对齐、视觉质量。非劣下界沿用 v207 的
预设量级：`-0.15/-0.0015/-0.0030/-0.0030/-0.0040`。它们是操作性容差，不是理论常数；
保持与开发阶段一致，不根据 v208 结果调整。

`paper_confirmation_pass` 只表示这一冻结规则通过，不表示论文所有证据已齐。
Dynamic Degree 单独报告，不参与晋级。raw optical flow 不用于宣称运动更好。
报告最多列出 4 条 `optional_review_queue`，只针对自动定位的异常，不要求全量盲审。

重点查看：

```text
inputs/manifest.json
inputs/runtime_contract.json
main30/audits/summary.json
main30/analysis/v208_main30_confirmation.json
main30/analysis/v208_main30_confirmation.md
main30/analysis/*_comparisons.csv
main30/analysis/v208_efficiency.json
main30/metrics/temporal_diagnostics.csv
```

日志中的 `[V208Provenance]` 绑定输入和代码；`[V208Runtime]` 记录每个完整分片用时。
效率表包含启动、trace 和编码开销，没有实测 PyTorch peak allocated/reserved bytes。
论文正式效率表仍需专门的同卡显存与稳定态 latency 测试，不能拿 FFE 代替实际显存。

轻量结果打包支持只完成一个 scope：

```bash
NODE_RANK=0 bash scripts/run_v208_paper_confirmation_32gpu.sh package
SCOPE=main30 NODE_RANK=0 bash scripts/run_v208_vbench_long.sh package
```

默认不打包完整 schedule traces 和视频；审计中已保存 trace 摘要。排查特定问题时才设置
`INCLUDE_RAW_TRACES=1`。Git 只推送小型报告、manifest 和必要日志。

## 6. 外部计划中仍需按结果推进的实验

本轮交付的是 v208 复测链路，不代表下列新推理分支已经实现：

- 满窗边界 parity：前三个 block 通过后检查首次滑窗裁剪和 clean refresh。
- `Recent21 -> Recent13 -> Recent9` 预算阶梯：现有 v207 含 21/13；新的 9 FFE 同路径结果仍需补。
- 固定 9 FFE selector 归因：只有运行路径一致时才能复用旧 v201；不能跨实现直接相减。
- 保留原 SF21 context 的历史 residual addon：需要单独的 `gate=0` parity、幅度网格和效率测量。
- 真正未参与历史调参的 prompts、matched mechanism ablation、跨模型迁移。

若 v207 没有候选胜过 SF/Recent21，应优先做预算与 addon 归因，不启动这套 768 视频的完整复测。
当前不会预先把未来的 additive gain、head 分类或 horizon 机制写成论文已经成立的贡献。
