# v207 SF 数值 Parity 与预算归因补充

> 日期：2026-09-15
>
> 性质：对现有 v207 的前置门禁；参考外部实验规划后独立落实，不覆盖现有方法选择规则。
>
> 当前机器不执行 GPU 推理，服务器结果返回后再修复首个分叉。

## 1. 为什么必须补这一组实验

现有 `recent21` 与 canonical Self-Forcing 都读取 21 FFE，但二者不是同一实现：

```text
canonical SF       : dense rolling tensor cache + native block attention
PF plain SF21      : vendored PF runtime + dense rolling tensor cache
Adaptive recent21  : AdaptiveKVCache + sink1/recent20 + varlen readout
```

等 FFE 只排除了 read length，不会自动对齐 cache 顺序、块内可见性、RoPE、clean refresh、随机数
消费顺序和 attention kernel。因此 v207 的正式归因必须拆成：

1. `SF native -> PF plain SF21`：vendored runtime 是否复现原生 SF；
2. `PF plain SF21 -> Adaptive recent21`：换成 Adaptive cache 后是否仍复现；
3. 两步均通过后，才允许把 retrieval 与 Recent 的差异归因于 cache 内容和 denoising phase。

代码审计还发现一个需要由 trace 验证、不能提前写成结论的风险：native SF 对一个 3-latent-frame
AR block 执行一次 dense attention；Adaptive 的 sink-grid-decoupled 路径可能把同一 block 组织成
逐帧 varlen sequences。若三帧的块内 K 可见性不同，数值会在第一个 attention call 就分叉。

## 2. 冻结的最小实验

使用 Qwen-rewritten MovieGen-128 的 source index 3，一条 prompt、9 个 latent frames、3 个 AR
blocks、4-step denoising，seed 固定为 20703。共运行四条轨迹：

| Run | 作用 | GPU 安排 |
|---|---|---|
| `sf_native_a` | canonical SF 主轨迹 | GPU 0 |
| `sf_native_b` | 同卡重复，估计 kernel/RNG floor | GPU 0，在 A 后串行 |
| `pf_plain_sf21` | PF runtime 的 dense recent21 | GPU 1 |
| `adaptive_recent21` | v207 candidate runtime 的等预算 Recent | GPU 2 |

每条 trace 均绑定 git commit、checkpoint/config/prompt/runtime 文件 SHA256，并保存：

- 输入噪声；
- 每个 AR block、每个 denoise call 的 noisy latent、flow、x0 和 scheduler noise/output；
- 每层 cache K/V 的确定性样本、frame membership/order 和可用 RoPE position；
- 每层 attention output 的确定性样本；
- clean-cache refresh 输入与 clean flow；
- block-end 和 final latent；
- decoded video 摘要。

Pipeline tensors 保存完整值；attention/K/V 只保存均匀确定性样本，避免一次诊断产生数十 GB。
若首个分叉定位到某层，可通过 `SF_PARITY_TRACE_LAYERS=<layer>` 进行窄化复跑，再临时扩展 full dump。

## 3. 自动判定

默认数值容差是 sampled/full tensor 的 `relative L2 <= 1e-5` 且 `max abs <= 5e-4`；若 native
A/B 自重复本身更大，则使用该 floor 的 5 倍。shape、event coverage 或 cache membership/order
不一致直接失败。最终报告区分：

```text
stop_fix_vendored_runtime_before_cache_attribution
stop_fix_adaptive_recent21_before_large_scale
parity_pass_proceed_to_budget_phase_screen
```

容差不是为了把结果调成通过；它只吸收同卡 native repeat 中实际观测到的非确定性。最终 VBench
接近不能替代该门禁。

## 4. 服务器命令

```bash
git pull origin codex/v178-v179-causal-validation

bash scripts/run_v207_sf_parity.sh prepare
GPU_LIST=0,1,2 bash scripts/run_v207_sf_parity.sh run
bash scripts/run_v207_sf_parity.sh status
bash scripts/run_v207_sf_parity.sh analyze

cat runs/v207_context_budget_phase_recovery/parity/report/parity_report.md
```

`prepare` 会对大 checkpoint 计算一次 SHA256，耗时取决于共享存储带宽。中断后可直接重跑
`run`，完成的 run 有 marker 时会跳过；需要全部重做时使用：

```bash
FORCE=1 GPU_LIST=0,1,2 bash scripts/run_v207_sf_parity.sh run
```

需要反馈的最小文件：

```text
runs/v207_context_budget_phase_recovery/parity/inputs/manifest.json
runs/v207_context_budget_phase_recovery/parity/report/parity_report.json
runs/v207_context_budget_phase_recovery/parity/report/parity_report.md
runs/v207_context_budget_phase_recovery/parity/report/parity_comparisons.csv
runs/v207_context_budget_phase_recovery/parity/cache_schedule.jsonl
runs/v207_context_budget_phase_recovery/parity/logs/*.log
runs/v207_context_budget_phase_recovery/parity/traces/*/events.jsonl
runs/v207_context_budget_phase_recovery/parity/traces/*/trace_meta.json
```

`.pt` tensor trace 不应提交 Git。只有在报告无法定位时，再打包并通过共享存储提供相关层的少量
`.pt` 文件。

## 5. Parity 后补充的实验

### P1：预算阶梯

在同一个已通过 parity 的 runtime 中比较 `recent21 -> recent13 -> recent9`，32 prompts、30 秒，
回答压缩局部上下文的代价。现有 v207 已包含 21/13；9 FFE 的旧 v201 只有在 parity 修复不改变
该路径时才可作为探索性证据，否则重跑。

### P2：固定预算 selector 归因

主比较仍是 retrieval 与 matched Recent，而不是只看 retrieval 与 native SF。只有 retrieval 在至少
一个预注册非 DD 轴上有 paired gain，并通过 full/late identity、temporal 和 camera-compensated
motion safety，才保留 selector claim。Static/phase-shift/head controls 只用于决定机制故事，不用于
选择性包装失败结果。

### P3：真正的 SF addon

若等预算 replacement 无法超过 SF，但 retrieval 相对 matched Recent 有正收益，再测试：

```text
plain SF21
SF21 + structured-history attention
SF21 + bounded gated history residual
SF21 + shifted-history control
```

其中 `gate=0` 必须数值复现 plain SF21，并单独报告 latency、peak GPU memory 和额外 attention
tokens。该分支在 P0 结果返回前不并入主代码，以免同时改变 cache runtime 与 fusion 机制。

## 6. 对论文实验的影响

Parity 通过只说明比较可解释，不代表方法有效。论文最低证据仍是：fresh MovieGen-128 上相对
canonical SF 的 paired 正收益、关键轴非劣、一个 matched mechanism ablation、30/60 秒时序安全和
效率表。PF 可对齐其 128-prompt、30/60 秒、VBench-Long 与效率报告格式，但不必在每轮开发重跑；
ABA/场景切换继续作为 secondary task，主单-prompt 外推未通过前不占用主要算力。
