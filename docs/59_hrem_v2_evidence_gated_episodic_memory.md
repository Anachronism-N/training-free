# HREM-v2: Head-Role Evidence-gated Episodic Memory

> 状态：代码完成，等待 GPU 服务器首轮实验。  
> 基座：Wan2.1-T2V-1.3B + Self-Forcing DMD。  
> 目标：training-free 长视频中的场景回访、主体身份和布局恢复，同时避免历史记忆冻结运动。
> 论文边界和服务器诊断见 `docs/60_hrem_v2_novelty_and_debug_protocol.md`。
> 完整 review、论文故事和运行手册见 `docs/61_hrem_v2_review_and_runbook.md`。

## 1. 当前决策

当前最高可行性的方向不是继续扩大 LifeCache token bank，也不是继续使用固定 head split，而是：

```text
HREM-v2 =
  episode-balanced K/V archive
  + dual-evidence non-recent episode admission
  + online continuous head-role routing
  + independent memory attention
  + uncertainty abstention and exact native fallback
```

选择它有四个直接依据：

1. CEMR oracle 实验已经说明正确的旧 K/V payload 能改善 scene return；主要问题是 episode selection，而不是 archive 中完全没有有效信息。
2. 32-prompt 结果中 CEMR 在 5/6 指标上有小幅正向信号，但 Dynamic Degree 下降，说明历史记忆需要按 head 控制，不能全 head 注入。
3. HREM-v1 的固定清理改变了 A2 的身份细节，但 return margin 下降，说明 head specialization 有作用，固定 head index 和原地 cache 操作不可靠。
4. Pyramid-Forcing 已证明 per-head 异构 cache 与 ragged-cache attention 在该 backbone 上可实现；本项目另外实现独立 side readout、episode sidecar、在线路由和 fail-closed fusion。

因此论文问题可表述为：

> 历史 K/V 的价值同时取决于“哪段历史与当前场景对应”和“哪些 attention head 适合读取长期状态”。现有训练自由方法通常只解决其中一个问题，或对所有 head 使用相同记忆策略。

## 2. 与相关代码的边界

| 参考实现 | 借鉴内容 | HREM-v2 的区别 |
|---|---|---|
| Self-Forcing | 原生 AR 生成和 rolling K/V | HREM-v2 是不修改 native K/V 的 sidecar branch |
| Pyramid-Forcing | head specialization、per-head 异构 cache、ragged-cache attention | PF 使用静态 head label；本方法在线计算连续 head-role evidence，并显式选择 episode；独立 side readout 不是 PF 的贡献 |
| Echo-Forcing | preserve/recall/forget 与 scene pool | Echo 的 scene memory 对 head 基本一致；本方法对每个 head 单独门控，且不依赖手工 recall id |
| Forcing-KV | static/dynamic head 分工 | 本方法不预设固定 head index，不训练 classifier |
| MemRoPE | pre-RoPE payload 与位置安全原则 | 首轮采用 position-free independent branch；不把旧 K 写回绝对位置 cache |
| LongLive-RAG | 临时 recall view | 召回只存在于当前 attention 调用，不成为永久 cache region |

HREM-v2 不应被描述为“Pyramid-Forcing 加一个 prompt prior”。核心新增量是两级可审计决策：episode admission 与 online head-role evidence。

## 3. 方法

### 3.1 Episode-balanced archive

每层只在 clean-context forward 后提交 pre-RoPE K 和 V。空间上做 stride-4 adaptive pooling，时间上保存完整代表帧。超过预算时执行 coverage selection，但先为每个 episode 保留至少一个代表帧；预算允许时再保留 episode 两端。

默认 Stage-1 配置：

```text
active layers: 15..20
archive frames per layer: 36
archive spatial stride: 4
readout frames: shared top-3
readout mode: noisy_only
```

这一设计修复两个已知问题：

- 全分辨率 30 层 archive 会占用过多显存；
- 全局 coverage 可能恰好删除被 gate 选中的旧 episode，造成 selected episode missing payload。

### 3.2 Dual-evidence episode admission

对当前 query 的每个 head 做 token mean，得到 `q_h`；对 archive frame 的 K 做 spatial mean，得到 `k_e,f,h`。视觉相似度为：

```text
v(e) = mean_top_rho_heads max_frame_in_e cos(q_h, k_e,f,h)
```

prompt descriptor 使用 tokenizer mask 后的 T5 token mean，避免 512-token padding 稀释。episode 语义分数为：

```text
s(e) = cos(prompt_current, mean_frame_prompt_in_e)
```

组合支持度使用几何均值：

```text
c(e) = sqrt(((s(e)+1)/2) * ((v(e)+1)/2))
```

候选集合强制满足：

```text
episode_id < current_episode_id
episode_id != previous_episode_id
```

这使首次 A->B transition 的 memory branch 必然 abstain，只执行所有主 cell 共享的 boundary reset；在 B->A 时，A1 才成为 non-recent candidate。默认还要求 semantic winner、visual winner 和 combined winner 一致。任一阈值、margin 或 cue agreement 不满足时直接 abstain。

### 3.3 Online head-role evidence

在已选 episode 的 top-k archive frames 上，保留池化后的完整空间 pattern，对每个 head 计算：

```text
K persistence = adjacent-frame cosine of flattened spatial K
V persistence = adjacent-frame cosine of flattened spatial V
Query stability = cosine(current query, preceding-block query EMA)
Motion risk = 0.5 * (1 - V persistence) + 0.5 * (1 - Query stability)
Persistent evidence = sqrt(K persistence * V persistence)
                      * Query stability * (1 - Motion risk)
Head gate = sigmoid(sharpness * (Persistent evidence - threshold))
```

诊断 role code 为 persistent/layout、motion-sensitive、refresh 三类，但实际融合只使用连续 gate，不使用硬编码 `0:4/4:8/8:12`。

关键假设是：布局和身份相关 head 在一个 episode 内具有更稳定的 K/V 方向；运动 head 的 V 或跨 block query 变化更大，应减少长期读出。

### 3.4 Independent attention and fail-closed fusion

native attention 先完整计算：

```text
O_native = Attention(Q_rope, K_native, V_native)
```

memory branch 使用 pre-RoPE archive，首轮 `position_mode=none`：

```text
O_memory = Attention(Q_raw, K_episode, V_episode)
```

融合采用 bounded convex mode，并乘 retrieval confidence、head gate 和正 alignment：

```text
w = alpha * confidence * head_gate * max(cos(O_native, O_memory), 0)
O = (1 - w) * O_native + w * rms_match(O_memory)
```

以下情况直接返回原始 `O_native` 对象：

- method gate 为 0；
- 当前还不是 return episode；
- 没有 non-recent candidate；
- dual evidence 不足或冲突；
- frame readout 全部拒绝；
- position mode 与 archive spatial layout 不兼容。

archive 不写回 `kv_cache["k"]` 或 `kv_cache["v"]`，因此不会出现旧 HREM-v2 原型中 disallowed head 被乘 0.7 的 cache 污染。

## 4. 已完成实现

| 文件 | 内容 |
|---|---|
| `src/lifecycle_kv/episodic_archive.py` | episode-balanced archive、spatial pooling、sidecar、trace |
| `src/lifecycle_kv/role_episodic.py` | masked prompt descriptor、dual evidence、online head-role gate |
| `third_party/Self-Forcing/utils/wan_wrapper.py` | prompt mask 和 structured-memory 参数桥接 |
| `third_party/Self-Forcing/wan/modules/causal_model.py` | pre-RoPE capture、independent readout、head gate、fusion |
| `third_party/Self-Forcing/pipeline/causal_inference.py` | episode 生命周期、clean commit、配置和 grid 校验 |
| `third_party/Self-Forcing/inference.py` | 完整 CLI/env 参数 |
| `tests/test_episodic_archive.py` | archive budget、去重、reset |
| `tests/test_role_episodic.py` | episode gate、cue conflict、head-role evidence |
| `prompts/hrem_v2_aba_complex_3.txt` | 3 个复杂且 B 场景明确的 A-B-A prompt |
| `scripts/run_hrem_v2_evidence.sh` | raw-native/reset-native/oracle/episode-only/full 五 cell 矩阵 |
| `scripts/evaluate_hrem_v2.py` | A1-A2、B-A2、return margin |
| `scripts/summarize_hrem_v2_trace.py` | episode route 因果审计 |
| `scripts/analyze_hrem_v2_debug.py` | archive/admission/head/fusion 根因诊断 |

## 5. Stage-1 因果矩阵

| Cell | Episode gate | Head routing | 目的 |
|---|---|---|---|
| native_raw | off | off | 原生 prompt schedule 诊断；检查场景惯性 |
| native_reset | off | off | 所有主 cell 共享的 fair boundary-reset 基线 |
| oracle_episode0 | oracle A1 | all heads | reset 后的历史 payload/fusion 上界 |
| dual_episode_only | dual evidence | all heads | 测试 episode selection，暴露运动污染 |
| hrem_v2 | dual evidence | role evidence | 完整方法 |

`native_reset`、oracle、episode-only 和 HREM-v2 都在 scene boundary 清空 native working-cache 的逻辑长度，但不清空 episodic archive。这个共享 control 用来保证 B 能形成；否则 HREM-v1 已经表明 raw SF 可能始终停留在 A。主 paired delta 必须以 `native_reset` 为 baseline，不能把 reset 带来的变化算作 HREM 增益。

服务器命令：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
PYTHONPATH=src pytest -q \
  tests/test_episodic_archive.py \
  tests/test_role_episodic.py \
  tests/test_structured_memory_readout.py \
  tests/test_hrem_debug_analyzer.py
bash scripts/run_hrem_v2_evidence.sh 0 1 2 3
python scripts/summarize_hrem_v2_trace.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl --strict
python scripts/analyze_hrem_v2_debug.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl \
  --strict --json-output runs/hrem_v2_evidence_s0/traces/hrem_v2_diagnosis.json
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hrem_v2.py \
  --run-root runs/hrem_v2_evidence_s0
```

运行前需要：

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
~/.cache/torch/hub/ 中的 DINOv2，或允许 evaluate 脚本联网下载
```

## 6. 晋级与停机标准

先检查机制，不先看单个好看的视频：

1. `--strict` trace audit 必须为 0 violation；所有 HREM readout 只能发生在 episode 2，且 `2 -> 0`。
2. 三个 prompt 的 B 段必须肉眼形成明确的新场景；B 未形成的样本不能用于声称 scene return 改善。
3. `native_reset` 的 B formation 必须明显优于 `native_raw`；主方法比较均以 `native_reset` 为基线。
4. oracle 的 mean return margin 必须优于 `native_reset`。若 oracle 不改善，暂停调 gate，优先检查 payload、readout timing 和融合位置。
5. dual_episode_only 若优于 `native_reset` 但 motion 明显下降，说明 episode gate 正确、head routing 必要。
6. 完整 HREM-v2 应在至少 2/3 prompt 上提高 return margin，且三 prompt mean 提高至少 `+0.02`，才进入多 seed。
7. 完整方法不能只靠 A2 变静态获得高 return；需要检查人物动作、雨/蒸汽/列车/人群等动态元素。

多 seed 阶段要求 seeds `0,1,2`，并报告 mean、std 和 paired bootstrap CI；不能以单 seed 或单 prompt 作为论文结论。

## 7. 后续实验顺序

### Stage 2: gate/layer 小矩阵

仅在 Stage 1 通过后运行：

- `alpha`: 0.05 / 0.10 / 0.15；
- layers: 12-18 / 15-21 / 18-24；
- role threshold: 0.35 / 0.45 / 0.55；
- seeds: 0 / 1 / 2。

每次只改变一类变量。首要输出为 return margin、B formation、motion smoothness、trace 中平均 active head 数。

### Stage 3: 必要 ablation

- A-B-C-A / A-B-C-B 多候选 stress suite，验证 dual cue 不是只靠“排除 previous 后只剩 A”；
- no episode gate；
- oracle episode；
- wrong previous episode；
- shuffled V；
- all heads；
- fixed PF labels；
- no query drift；
- no V persistence；
- gate=0 exactness；
- archive uniform vs episode-balanced coverage。

### Stage 4: 投稿规模

- 32-prompt long-video suite；
- Self-Forcing、Pyramid-Forcing、Echo-Forcing；
- VBench Subject/Background/Aesthetic/Imaging/Motion/Dynamic；
- 至少 3 seeds 的核心 A-B-A suite；
- memory、latency 和峰值显存；
- 失败案例：相似 episode、快速动作、无显式 return、四 episode 以上。

## 8. 当前不能声称的内容

在服务器结果回来前，只能声称：

- 已提出一个因果结构清晰、training-free、可 fail-closed 的实现；
- 已修通 Self-Forcing structured-memory bridge；
- 已准备 oracle、episode-only 和完整方法的区分性实验。

不能声称 HREM-v2 已优于 baseline、已保持 motion、或已经达到投稿结果。最终论文故事必须由 oracle 上界、episode route trace、多 seed return metric 和通用 VBench 共同支持。
