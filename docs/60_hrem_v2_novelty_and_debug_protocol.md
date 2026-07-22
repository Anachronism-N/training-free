# HREM-v2 论文边界与服务器诊断协议

> 状态：方法与诊断代码已实现，等待 GPU 服务器验证。
> 实现代号：HREM-v2。
> 论文核心：Factorized Evidence-Gated Episodic Recall，即先决定“哪段历史可用”，再决定“哪些 head 可用”。
> 论文/代码来源、许可证与安全 claim 以 `docs/64_related_work_code_provenance_and_claims.md` 为准。

## 1. 最终 idea 的边界

HREM-v2 不是把多个仓库的模块机械叠加。它将长期记忆读出分解成两个不可互换的因果决策：

```text
historical archive
  -> episode admission: semantic evidence AND visual-query evidence
  -> frame retrieval within the admitted episode
  -> head admission: K persistence AND V persistence AND query stability
  -> native-preserving bounded fusion
```

核心问题是：历史 K/V 即使本身有效，也只有在“历史 episode 正确”和“读取它的 head 功能合适”同时成立时才应注入。episode 选择错误会产生身份/场景污染；head 选择错误会冻结运动或覆盖当前动态。二者必须分别测量、分别消融。

### 1.1 明确借鉴、但不作为创新声明的组件

| 来源 | 借鉴内容 | 在 HREM-v2 中的处理 |
|---|---|---|
| Self-Forcing | AR 生成、rolling K/V、clean/noisy forward | 保持为 native branch 和主要 baseline |
| Pyramid-Forcing | head specialization、per-head 异构 cache、ragged-cache attention | 取消固定 head label，改为在线连续证据；本项目另行实现独立 side readout |
| Echo-Forcing | episode preserve/recall/forget | 不使用手工 recall id，增加可拒绝的 episode admission |
| Forcing-KV | static/dynamic head 差异 | 不预设 head index，也不训练分类器 |
| MemRoPE | pre-RoPE payload、位置安全原则 | 首轮采用 position-free side branch，绝不回写旧位置 K/V |
| LongLive-RAG | 临时 recall view | recall 只对当前 attention 调用生效 |
| LifeCache/CEMR 原型 | bounded archive、Q-K retrieval、实验 sidecar | 降级为支撑实现与 ablation，不作为论文主贡献 |

以下单项都不能单独声称为本方法创新：K/V archive、top-k retrieval、pre-RoPE 存储、memory attention、head-aware cache、scene memory、confidence gate。

### 1.2 可作为论文贡献验证的组合

1. **Non-recent dual-evidence episode admission**：当前 episode 与 immediately previous episode 都不可被读出；语义 winner、visual-query winner 和 combined winner 默认必须一致。首次 A->B 必然 abstain，只有 B->A 才可召回 A。
2. **Online functional head admission**：每次 readout 从被选 episode 的 K/V temporal persistence 和跨 block query drift 推断连续 head gate，不依赖训练、固定索引或离线 head 标签。
3. **Factorized causal controls**：oracle episode、episode-only/all-head、full HREM、wrong/shuffled controls 分别定位 payload、episode selection、head routing 和融合贡献。
4. **Fail-closed native preservation**：任何一级证据不足均返回原始 native output；memory 不修改 native cache。每个被拒绝 head 也严格保持 native，不只在全 head 拒绝时回退。

这里最有区分度的不是某个分数公式，而是可审计的两级准入结构。最终投稿前仍需做更广泛的论文检索；在检索和实验完成前，应写作“our proposed factorization”，不能写作“the first”。

## 2. 可证伪的论文故事

### 2.1 主假设

```text
H1: correct historical payload has a positive scene-return upper bound;
H2: dual evidence closes part of the oracle-vs-automatic episode gap;
H3: head evidence retains the return gain while recovering motion lost by all-head recall;
H4: abstention makes non-return prompts no worse than native-reset within noise.
```

对应证据链：

| 对比 | 回答的问题 | 失败后的结论 |
|---|---|---|
| oracle vs native_reset | 旧 payload 是否有用 | 若无增益，停止调 selector，检查 payload/readout |
| dual_episode_only vs oracle | 自动 episode admission 是否有效 | gap 大则检查 cue 阈值或 prompt schedule |
| HREM-v2 vs dual_episode_only | head routing 是否减少运动污染 | 无 motion 改善则 head evidence 不成立 |
| HREM-v2 vs native_reset | 完整方法是否有净收益 | return 与 motion 不能同时改善则不晋级 |
| shuffled-V / wrong-episode | 增益是否来自正确历史 | control 也改善则指标或融合存在混淆 |

### 2.2 投稿所需最低结果

- 3-prompt Stage-1 中 oracle 上界成立，full HREM 至少 2/3 prompt 提高 return margin；
- full HREM 相比 all-head recall 恢复 Dynamic Degree 或局部运动指标；
- A-B-C-A / A-B-C-B 多候选测试仍能选对 episode，排除“去掉 previous 后只剩唯一答案”；
- 32 prompts、3 seeds 报告 paired mean/std/bootstrap CI；
- gate=0 exactness、wrong episode、shuffled V 和 no-query-drift 等负对照完整；
- 报告 latency、peak memory、archive budget 和 abstention rate。

## 3. Debug 输出设计

默认 debug 关闭，不改变原生推理性能。Stage-1 脚本只打印 active range 的首尾层 `15,20`，每个生成 block 每个事件最多一次：

```bash
STRUCTURED_MEMORY_DEBUG=1
STRUCTURED_MEMORY_DEBUG_LAYERS=15,20
STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
```

stdout 前缀均可直接检索：

```text
[HREMv2][boundary] frame=... episode=1->2 working_cache_reset=1 archive_preserved=1
[HREMv2][archive][L15] ep=... added=... kept=... episodes=... k_rms=... v_rms=...
[HREMv2][episode][L15] block=... winner=... accepted=0 reason=cue_disagreement ...
[HREMv2][retrieval][L15] block=... accepted_heads=0 reason=... confidence_max=...
[HREMv2][fusion][L15] block=... allow=0 accepted_heads=... head_gate=... weight=... delta/native=...
```

JSONL `readout` 事件额外记录：

- 定位：`layer`, `block_index`, `current_frame`, `current_episode_id`；
- route：`allowed_episode_id`, `episode_decision`, `selected_indices`；
- retrieval：`confidence`, `retrieval_margin`, `retrieval_entropy`, `accepted_head_count`；
- role：`head_role`, `head_gate_mean`, `head_gate_active_count`；
- fusion：`effective_weight_mean/max`, `alignment_mean`, `alignment_positive_fraction`；
- effect：`native_rms`, `memory_rms`, `fused_rms`, `delta_rms`, `delta_to_native_rms`。

`commit` 事件记录每个 episode 的 survivor 数与 archive K/V RMS。`boundary` 事件记录 working cache 是否 reset、archive 是否 preserved。

## 4. 服务器运行与诊断

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only
PYTHONPATH=src pytest -q \
  tests/test_episodic_archive.py \
  tests/test_role_episodic.py \
  tests/test_structured_memory_readout.py \
  tests/test_hrem_debug_analyzer.py
bash scripts/run_hrem_v2_evidence.sh 0 1 2 3
grep -E '\[HREMv2\]' runs/hrem_v2_evidence_s0/logs/hrem_v2.log | tail -n 120
python scripts/summarize_hrem_v2_trace.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl --strict
python scripts/analyze_hrem_v2_debug.py \
  runs/hrem_v2_evidence_s0/traces/hrem_v2.jsonl \
  --strict \
  --json-output runs/hrem_v2_evidence_s0/traces/hrem_v2_diagnosis.json
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hrem_v2.py \
  --run-root runs/hrem_v2_evidence_s0
```

### 4.1 根因到下一步

| 诊断 | 高概率原因 | 下一步 |
|---|---|---|
| `no_archive_commits` | clean-pass bridge 或 layer mask 未接通 | 不调 gate，先检查 commit 调用 |
| `zero_archive_signal` | pre-RoPE capture 错误或空 tensor | 检查 K/V shape、RMS 和 clean/noisy 路径 |
| `no_accepted_readout` | episode 或 retrieval 阈值过严 | 先看 dominant abstain reason，再只放宽对应一级 |
| `role_gate_over_suppressed` | role threshold 高或 query EMA 不稳定 | 检查 per-head evidence，再下调 threshold |
| `role_gate_not_selective` | head evidence 没有区分度 | 上调 threshold；与 episode-only 比较 |
| `memory_native_conflict` | 位置约定、payload 或 episode 错误 | 禁止增大 global gate，先查 alignment |
| `fusion_effect_negligible` | 多级 gate 乘积接近零 | 分解 confidence/head/alignment，逐级定位 |
| `fusion_effect_too_large` | gate 过高或错误历史覆盖 native | 降 gate 并检查 wrong/shuffled controls |
| causal invariant violation | 读到了 current/previous episode | 结果作废，先修 episode mask |

## 5. 快速迭代纪律

1. 先跑单 prompt、单 seed 的机制 smoke test，确认 commit、boundary、route、head gate 和 delta 都存在。
2. 分析器有 `ERROR` 时不看视频挑样例，也不开始 sweep。
3. 只允许一次调整一个层级：episode thresholds、retrieval、role threshold、global gate。
4. 保存每轮 git commit、完整 env、stdout log、JSONL、diagnosis JSON 和视频指标。
5. 机制通过后再扩展 prompt/seed；不能用提高 gate 掩盖 route 或 positional bug。

该协议的目标不是让 log 证明方法有效，而是让首轮服务器结果能区分“代码未生效”“selector 未通过”“head gate 未生效”“融合过弱/过强”和“方法假设本身失败”。
