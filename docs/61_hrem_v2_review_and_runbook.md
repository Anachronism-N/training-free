# HREM-v2 Review Guide and Experiment Runbook

> 方法背景 review 入口。方法实现、论文故事、代码使用和原始日志判读见本文；最新实验决策见 docs/67。
> 核心实现提交基线：`f19a6bd26d6794e6b4e0919c3bcfb49d3cb7a7af`；本文档会形成后续独立提交。
> 状态：首轮 HREM-v2 与 absolute gate sweep 已完成；relative/hybrid 与 ramp 对照待运行。
> 相关工作、代码 provenance、许可证和 claim 红线见 `docs/64_related_work_code_provenance_and_claims.md`；发生冲突时以该台账为准。
> 结果更新后的实验范围校正、episode-local ramp 和最新服务器矩阵见 `docs/67_post_sweep_optimization_and_server_protocol.md`；后续运行决策以 docs/67 为准。

## 1. 一句话 idea

**HREM-v2 是一个 training-free 的分层历史召回方法：先用语义与当前 visual query 的双证据判断“应该回忆哪个非近期 episode”，再用在线 K/V persistence 与 query drift 判断“哪些 attention head 适合读取这段历史”，最后通过独立 memory-attention 分支进行有界融合；任何一级不确定都精确回退原生 Self-Forcing。**

完整名称：

```text
HREM-v2: Head-Role Evidence-gated Episodic Memory
```

可用于论文标题的表达：

```text
When to Remember and Which Heads Should Recall:
Factorized Evidence-Gated Memory for Training-free Long Video Generation
```

## 2. 为什么是这个问题

长 AR 视频中的历史记忆不是“保存得越多越好”。一次 recall 失败通常来自两个不同错误：

1. **episode 错误**：读到了当前场景或刚刚结束的场景，导致旧背景、服装和物体污染返回场景；
2. **head 错误**：即使 episode 正确，把长期 K/V 注入 motion/refresh head 仍会冻结人物、雨、蒸汽、列车或镜头运动。

已有原型给出的动机证据是：

- CEMR 的有效案例中，强制读 episode 0 的 oracle 曾将 full return margin 从 `-0.3504` 提高到 `+0.1428`，说明正确历史 payload 可能有效；
- 32-prompt 单 seed 结果中基础 memory 在 5/6 VBench 分项上有小幅正向，但 Dynamic 下降 `-0.025`，说明 all-head 历史注入可能伤害运动；
- HREM-v1 固定 head 清理确实改变了 A2 身份细节，但 return margin 从 `-0.1045` 降至 `-0.1221`，且 B 场景未形成，说明固定 head split 与原地 cache 操作不能构成可靠方法。

这些结果只构成设计动机，不能作为 HREM-v2 已有效的证据。

## 3. 方法结构

```text
clean pre-RoPE K/V
        |
        v
episode-balanced archive + episode/prompt sidecar
        |
        v
non-recent episode candidates
        |
        +-- semantic evidence from current vs archived prompt
        +-- visual evidence from current Q vs archived K
        |
        v
dual-evidence episode admission or abstention
        |
        v
top-k frame retrieval inside the admitted episode
        |
        v
online per-head K/V persistence + query stability
        |
        v
continuous head gate
        |
        v
independent memory attention
        |
        v
confidence x head gate x positive alignment x accepted mask
        |
        v
bounded fusion or exact native fallback
```

### 3.1 Episode-balanced archive

每个 active transformer layer 保存 clean-context forward 得到的 pre-RoPE K/V：

- 空间使用 stride-4 adaptive pooling；
- 默认每层最多 36 个代表帧；
- coverage eviction 优先保留不同 episode，而不是只留下近期帧；
- sidecar 同步保存 frame interval、episode id 和 masked prompt descriptor；
- archive 与 native working K/V cache 完全分离。

Archive 是必要基础设施，但不是核心创新声明。

### 3.2 Episode admission：回答“回忆哪一段”

候选 episode 必须满足：

```text
episode_id < current_episode_id
episode_id != previous_episode_id
```

当前默认从 episode 2 才允许 recall。因此 A->B 时不读 A；B->A 时 episode 0 才成为 non-recent candidate。

语义分数：

```text
s(e) = cosine(prompt_current, mean_prompt_of_episode_e)
```

visual-query 分数：

```text
v(e) = mean_top_heads max_frame_in_e cosine(q_head, k_frame_head)
```

组合分数：

```text
c(e) = sqrt(((s(e) + 1) / 2) * ((v(e) + 1) / 2))
```

默认要求 semantic winner、visual winner、combined winner 一致，并通过 semantic、visual、combined 和 margin 阈值。证据不足时不尝试 fallback episode，直接 abstain。

### 3.3 Head admission：回答“哪些 head 可以回忆”

对 admitted episode 的 top-k frame，按 head 计算：

```text
K persistence = adjacent-frame cosine of flattened spatial K
V persistence = adjacent-frame cosine of flattened spatial V
Query stability = cosine(current query, preceding-block query EMA)
Motion risk = 0.5 * (1 - V persistence) + 0.5 * (1 - Query stability)
Persistent evidence = sqrt(K persistence * V persistence)
                      * Query stability * (1 - Motion risk)
Head gate = sigmoid(sharpness * (Persistent evidence - threshold))
```

这里没有固定 `head 0:4/4:8/8:12`，也没有训练 head classifier。role code 只用于诊断，实际融合使用连续 gate。

### 3.4 独立 readout 与融合

```text
O_native = Attention(Q_rope, K_native, V_native)
O_memory = Attention(Q_raw, K_episode_pre_rope, V_episode)

w = alpha
    * retrieval_confidence
    * head_gate
    * max(cos(O_native, O_memory), 0)
    * accepted_head_mask

O_final = (1 - w) * O_native + w * rms_match(O_memory)
```

默认使用 `alpha=0.10` 的 convex fusion。被 retrieval 拒绝的单个 head 保持 native；全部拒绝或 episode abstain 时直接返回原始 `O_native`。历史 K/V 从不写回 native cache。

## 4. 与已有工作的区别

| 工作 | 借鉴 | HREM-v2 的区别 |
|---|---|---|
| Self-Forcing | AR backbone 与 rolling cache | 增加不修改 native cache 的 episodic side branch |
| Pyramid-Forcing | head specialization、per-head 异构 cache、ragged-cache attention | 不使用静态 head label；显式选择 episode 并在线计算连续 head evidence；独立 side readout 为本项目实现选择 |
| Echo-Forcing | preserve/recall/forget | 不依赖手工 recall id；episode 可因证据不足拒绝；head 不统一处理 |
| Forcing-KV | static/dynamic head 差异 | 不固定 head index，不训练分类器 |
| MemRoPE | pre-RoPE 与位置安全 | 首轮使用 position-free branch，不把旧 K 填入绝对位置 cache |
| LongLive-RAG | temporary recall view | recall 只在当前 attention 调用存在 |
| LifeCache/CEMR | bounded archive、Q-K retrieval、因果 controls | 这些降级为基础设施；核心变为 episode/head 两级准入 |

论文不能把 archive、top-k Q-K、pre-RoPE、head-aware cache 或独立 memory attention 单独写成原创。当前最可辩护的贡献是：

1. dual-evidence non-recent episode admission；
2. online functional head admission；
3. 二者组成的 factorized recall decision；
4. 可逐级证伪且 fail-closed 的 causal implementation。

投稿前仍需系统论文检索，因此当前不能使用“first”或“首次”表述。

## 5. 论文故事怎么说

### 5.1 故事主线

**Problem.** Long-video AR models forget old scene identity under bounded working caches, but indiscriminate history recall introduces stale appearance and motion freezing.

**Diagnosis.** Recall quality has two independent latent variables: whether the selected episode matches the current return, and whether each head represents persistent state or transient motion.

**Method.** HREM-v2 factorizes recall into episode admission and head admission. Semantic and visual-query evidence select a non-recent episode; online K/V/query dynamics control per-head access; a side attention branch preserves native causality and abstains under uncertainty.

**Evidence chain.** Oracle verifies payload value; episode-only verifies automatic routing; all-head vs head-aware isolates motion protection; wrong-episode and shuffled-V verify that gains come from correct historical content.

**Claim.** The intended claim is not “more memory improves long video,” but “factorizing when to recall and where to inject makes training-free episodic memory controllable and useful.”

### 5.2 四个必须验证的假设

| Hypothesis | 对应实验 | 通过条件 |
|---|---|---|
| H1：正确旧 payload 有正上界 | oracle vs native_reset | oracle return margin 明显更好 |
| H2：dual evidence 能接近 oracle route | dual_episode_only vs oracle | route 正确且缩小 return gap |
| H3：head gate 减少 all-head 的 motion 损失 | hrem_v2 vs dual_episode_only | return 不丢，motion/dynamic 改善 |
| H4：abstention 保持非回访场景安全 | no-return suite、gate=0 | 与 native_reset 基本一致 |

### 5.3 推荐论文图表

1. 方法图：archive -> episode gate -> frame retrieval -> head gate -> side attention；
2. 主表：Self-Forcing、Pyramid-Forcing、Echo-Forcing、episode-only、HREM-v2；
3. 因果表：native_reset、oracle、wrong episode、shuffled V、all heads、full；
4. 曲线：return margin 与 Dynamic/运动指标随 gate、layer、role threshold 变化；
5. 可视化：A1/B/A2 关键帧、episode score、每层 active head 数与 fusion delta。

## 6. 当前代码修改

| 文件 | 当前职责 |
|---|---|
| `src/lifecycle_kv/episodic_archive.py` | bounded episode-balanced archive、sidecar、commit、trace/debug |
| `src/lifecycle_kv/role_episodic.py` | masked prompt descriptor、dual-evidence selector、online head evidence |
| `src/lifecycle_kv/attention_fusion.py` | frame readout、confidence/entropy/margin、accepted-head-safe fusion |
| `third_party/Self-Forcing/wan/modules/causal_model.py` | pre-RoPE capture、memory readout、role gate、融合诊断 |
| `third_party/Self-Forcing/pipeline/causal_inference.py` | episode 生命周期、boundary reset、clean commit、配置 |
| `third_party/Self-Forcing/utils/wan_wrapper.py` | prompt mask 与 structured-memory 参数桥接 |
| `third_party/Self-Forcing/inference.py` | CLI 到环境变量映射 |
| `scripts/run_hrem_v2_evidence.sh` | 五 cell、三 prompt、120-frame Stage-1 矩阵 |
| `scripts/summarize_hrem_v2_trace.py` | episode route 因果审计 |
| `scripts/analyze_hrem_v2_debug.py` | archive/admission/head/fusion 根因分析 |
| `scripts/evaluate_hrem_v2.py` | A1-A2、scene separation、B-A2、return margin |
| `tests/test_episodic_archive.py` | archive budget、episode coverage、commit/reset |
| `tests/test_role_episodic.py` | episode gate、cue conflict、head evidence |
| `tests/test_structured_memory_readout.py` | readout、abstention、逐 head native fallback |
| `tests/test_hrem_debug_analyzer.py` | 诊断器正常与错误 trace |

关键提交：

```text
ab13365  feat: implement evidence-gated HREM v2
f19a6bd  feat: add auditable HREM v2 diagnostics
```

## 7. 运行前准备

### 7.1 模型位置

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

评估还需要 `~/.cache/torch/hub/` 中可用的 DINOv2，或允许评估脚本联网下载。

### 7.2 默认服务器环境

脚本默认使用：

```text
repo: /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
conda: longlive
config: third_party/Self-Forcing/configs/self_forcing_dmd.yaml
checkpoint: third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

可覆盖变量：`REPO_ROOT`、`SF_CONFIG`、`SF_CHECKPOINT`、`PROMPTS`、`OUT_ROOT`、`FRAMES`、`SEED`、`FORCE`。

## 8. 标准运行命令

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

PYTHONPATH=src pytest -q \
  tests/test_episodic_archive.py \
  tests/test_role_episodic.py \
  tests/test_structured_memory_readout.py \
  tests/test_hrem_debug_analyzer.py

bash scripts/run_hrem_v2_evidence.sh 0 1 2 3
```

四个 GPU 的分工：

```text
GPU 0: native_raw -> native_reset
GPU 1: oracle_episode0
GPU 2: dual_episode_only
GPU 3: hrem_v2
```

重新运行并覆盖已有三段视频：

```bash
FORCE=1 bash scripts/run_hrem_v2_evidence.sh 0 1 2 3
```

运行其他 seed：

```bash
SEED=1 OUT_ROOT="$PWD/runs/hrem_v2_evidence_s1" \
  bash scripts/run_hrem_v2_evidence.sh 0 1 2 3
```

## 9. 实验矩阵如何解读

| Cell | 与谁比较 | 说明 |
|---|---|---|
| `native_raw` | `native_reset` | 判断 raw SF 是否因旧 working cache 导致 B 不形成 |
| `native_reset` | 所有主方法 | 公平主 baseline；不要把 reset 收益计入 HREM |
| `oracle_episode0` | `native_reset` | 检查旧 payload 与 fusion 上界 |
| `dual_episode_only` | oracle / native_reset | 检查自动 episode selector；all heads 暴露 motion 污染 |
| `hrem_v2` | dual_episode_only / native_reset | 检查 head evidence 是否保留 return 并改善 motion |

三个默认 prompt 分别包含：陶艺工作室-地铁-陶艺工作室、天文台-温室-天文台、雨夜餐车-剧场-雨夜餐车。每个 A2 都要求恢复身份、布局和标志物，同时保留手、镜头、天气、人群等运动。

## 10. 输出位置

```text
runs/hrem_v2_evidence_s0/
|-- native_raw/*.mp4
|-- native_reset/*.mp4
|-- oracle_episode0/*.mp4
|-- dual_episode_only/*.mp4
|-- hrem_v2/*.mp4
|-- logs/*.log
|-- traces/hrem_v2.jsonl
|-- traces/hrem_v2_diagnosis.json
`-- metrics_aba.json
```

脚本会自动运行 `analyze_hrem_v2_debug.py --strict`。出现结构性 `ERROR` 时脚本返回非零，但生成的视频仍保留供排错。

## 11. 运行后必须执行的检查

### 11.1 stdout debug

```bash
grep -E '\[HREMv2\]' \
  runs/hrem_v2_evidence_s0/logs/hrem_v2.log | tail -n 160
```

重点事件：

```text
[HREMv2][boundary]  episode transition、working cache reset、archive preserved
[HREMv2][archive]   archive frames、episode survivor、K/V RMS
[HREMv2][episode]   winner、cue agreement、abstain reason
[HREMv2][retrieval] accepted heads、confidence、margin
[HREMv2][fusion]    selected frames、head gate、effective weight、delta/native
```

默认只打印 layer 15 和 20，每个 block 每类事件最多一次。完整数值保存在 JSONL。

### 11.2 route 因果审计

```bash
python scripts/summarize_hrem_v2_trace.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl --strict
```

必须满足：

- readout 只发生在 return episode 2；
- route 必须是 `current=2, previous=1, allowed=0`；
- 不允许 current、future 或 immediately previous episode；
- 至少存在一次 admitted readout。

### 11.3 根因分析

```bash
python scripts/analyze_hrem_v2_debug.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl \
  --strict \
  --json-output runs/hrem_v2_evidence_s0/traces/hrem_v2_diagnosis.json
```

主要诊断码：

| Code | 含义 | 处理 |
|---|---|---|
| `no_archive_commits` | clean K/V 未进入 archive | 检查 bridge/layer mask，不调 gate |
| `zero_archive_signal` | K/V 捕获为空或数值异常 | 检查 capture shape、RMS、clean path |
| `no_accepted_readout` | episode/retrieval 全部拒绝 | 先看 dominant abstain reason |
| `role_gate_over_suppressed` | head gate 均值低于 0.05 | 查 per-head evidence，再降低 role threshold |
| `role_gate_not_selective` | head gate 均值高于 0.90 | 提高 threshold，并比较 all-head cell |
| `memory_native_conflict` | 正 alignment 比例低于 0.10 | 查位置约定/payload/episode，禁止增大 gate |
| `fusion_effect_negligible` | median delta/native 小于 `1e-4` | 分解 confidence/head/alignment 乘积 |
| `fusion_effect_too_large` | median delta/native 大于 `0.25` | 降 gate，检查 wrong/shuffled controls |
| `causal_invariant_violation` | 读到非法 episode | 当前结果作废，先修 mask |

### 11.4 A-B-A metric

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hrem_v2.py \
  --run-root runs/hrem_v2_evidence_s0
```

指标含义：

```text
A1-A2: A2 是否恢复 A1
A1-B / scene_separation: B 是否真正形成不同场景
B-A2: A2 是否仍被 B 污染，越低越好
return_margin = A1-A2 - B-A2，越高越好
paired_delta: 相对 native_reset 的同 prompt 差值
```

先验证 B formation，再解释 return margin。B 未形成时，该 prompt 的 episodic-return estimand 无效。

### 11.5 人工视频检查

每个 prompt 必须逐段记录：

- A1：身份、背景布局、独特物体、服装和镜头方向；
- B：是否真正切换，A1 物体是否错误残留；
- A2：身份/布局/独特物体是否恢复；
- motion：人物手部、身体、镜头、雨/蒸汽/列车/蝴蝶/人群是否冻结或重复；
- artifact：重影、纹理粘连、瞬时闪烁、错误物体复制。

不能用少量漂亮关键帧替代完整视频检查。

## 12. 首轮 go/no-go 标准

按顺序执行，前一项失败时不要直接调大 gate：

1. tests 全部通过；
2. trace audit 为 0 violation；
3. archive 有非零 K/V、episode 0 在 A2 前仍有 survivor；
4. `native_reset` 的 B formation 明显优于 `native_raw`；
5. oracle mean return margin 优于 `native_reset`；
6. dual episode route 在三个 prompt 上都是 `2 -> 0`；
7. HREM active head 既不是长期 0，也不是长期全部 12；
8. full HREM 至少 2/3 prompt 的 paired return delta 为正，三 prompt mean 至少 `+0.02`；
9. full HREM 不依靠冻结 motion 获得 return 增益；
10. 通过后才进入 gate/layer/threshold 小矩阵和 seeds `0,1,2`。

## 13. 建议的快速迭代顺序

```text
Step 1: archive/route structural smoke test
Step 2: oracle payload upper bound
Step 3: dual-evidence episode selector
Step 4: head gate selectivity
Step 5: fusion magnitude
Step 6: three prompts x three seeds
Step 7: A-B-C-A / A-B-C-B multi-candidate stress test
Step 8: 32 prompts + VBench + memory/latency
```

每次只调整一个层级：

1. episode thresholds；
2. frame retrieval temperature/confidence；
3. role threshold/sharpness；
4. global gate；
5. active layers。

每轮保留 commit、完整 env、stdout、JSONL、diagnosis JSON、metrics 和视频。不要通过同时改变多个阈值来追单个样例。

## 14. Review 时需要重点质疑的地方

1. **边界依赖**：当前 episode boundary 来自 prompt 中的 `||` schedule，不是自动 scene detection；论文必须明确任务设定。
2. **selector 是否真有多候选能力**：A-B-A 中排除 previous 后可能只剩 A，必须用 A-B-C-A/A-B-C-B 证明 dual evidence 有实际选择作用。
3. **head evidence 是否优于简单置信度 gate**：需要 confidence-adaptive、functional-adaptive、fixed PF label 和 all-head 对照。
4. **position-free memory 是否丢失空间对应**：`position_mode=none` 是安全首轮配置，但可能限制布局恢复，需要与可控 local-grid/MemRoPE 方案比较。
5. **query drift 是否只是 timestep 噪声**：需要 no-query-drift 和不同 EMA decay 消融，并按 denoising timestep 查看稳定性。
6. **motion 改善是否来自更弱融合**：比较 all-head 与 HREM 时应报告 effective weight；必要时做 weight-matched control。
7. **评估是否被 DINO 偏差影响**：return metric 必须与 B validity、VBench、运动指标和人工盲评联合报告。
8. **泛化范围**：当前首先验证显式 episodic return，不应扩张为所有长视频一致性问题。

## 15. 当前可以与不能声称的内容

可以声称：

- 已提出一个 training-free、两级证据准入、fail-closed 的 episodic recall 实现；
- 已完成 Self-Forcing 的 pre-RoPE archive、episode sidecar、online head gate 和独立 readout bridge；
- 已准备区分 payload、episode selection、head routing 与 fusion 的实验矩阵和诊断日志。

暂时不能声称：

- HREM-v2 优于 Self-Forcing/Pyramid-Forcing/Echo-Forcing；
- head evidence 已保持 motion；
- dual evidence 已解决多 episode selection；
- 方法具有统计显著或跨 seed 稳健收益；
- 方法是相关领域首次提出。

## 16. Review 决策建议

当前应 review 的不是最终效果结论，而是以下四个设计决定是否合理：

1. 把 recall 分解为 episode admission 与 head admission；
2. 将 immediately previous episode 作为 return task 的 hard negative；
3. 使用 K/V persistence 与 query drift 作为 training-free head functional evidence；
4. 使用不写回 native cache 的 independent attention + fail-closed fusion。

如果这四点在方法逻辑上通过 review，下一步唯一优先级是服务器 Stage-1。若 oracle 上界失败，应停止 selector/head sweep 并检查 payload；若 oracle 成立但 automatic route 失败，聚焦 episode evidence；若 route 成立但 motion 下降，聚焦 head evidence；若三者均成立，再扩展多 seed 与通用 benchmark。
