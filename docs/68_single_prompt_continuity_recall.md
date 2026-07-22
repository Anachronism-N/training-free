# HREM-v2.1: Single-Prompt Continuity Recall

> 日期: 2026-07-22
>
> 状态: 代码和静态检查阶段, 尚未在 GPU 服务器运行。
>
> 本文修正 `docs/67_post_sweep_optimization_and_server_protocol.md` 中“单 prompt 仅作 no-op 审计”的旧结论。旧结论准确描述 commit `95c88e2` 及以前的实现, 本文描述其后的显式 `intra_episode` 路径。

## 1. 为什么必须修正方向

此前 HREM-v2 只面向 A-B-A prompt 切换:

```text
STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
current episode is always excluded from readout
```

单 prompt 推理始终处于 episode 0, 同一 episode 的归档帧又被过滤, 所以 memory branch 即使完成 capture 也不会进入有效 fusion。这不是“收益小”, 而是结构性 no-op。

单 prompt 的 30s/60s 长视频是主任务而不是附属任务。此前 PF 路径的实验已经说明:

- full-frame Q-K retrieval 有小幅一致性信号, 但样本不足;
- all-head 历史注入可能造成多肢体、旧动作回放和重影;
- 32-prompt 上一致性的小幅提升可能伴随 Dynamic Degree 下降;
- 因此不能只优化 DINO/subject consistency, 必须同时约束 motion、dynamic 和人工伪影。

本轮增加一个独立、显式、可审计的同 episode 远期召回范围。

## 2. 当前 idea

暂定论文方向:

**Selective Historical Recall for Training-Free Long-Horizon Autoregressive Video Generation**

方法故事包含两个共享 archive、但准入规则不同的历史访问范围:

| Recall scope | 任务 | 候选历史 | 准入依据 | 主要风险 |
|---|---|---|---|---|
| continuity recall | 单 prompt 长视频 | 当前 episode 中足够旧的 clean K/V 帧 | frame age + Q-K confidence + optional head gate; margin/entropy 可配置并记录 | 动作回放、动态减弱、旧姿态污染 |
| return recall | A-B-A / A-B-C-A | 更早且非 previous 的 episode | prompt/visual dual evidence + Q-K retrieval + optional head gate | 错场景召回、边界伪影 |

二者共享:

```text
clean pre-RoPE K/V capture
+ bounded coverage archive
+ independent memory attention
+ confidence/alignment-controlled convex fusion
+ native fallback on abstention
```

二者不同的是候选集合和 admission evidence。当前 scope 由实验协议显式指定, 尚未实现自动任务识别, 不能声称为“自动双范围路由”。

### 2.1 Continuity recall 的因果规则

单 prompt 模式设置:

```text
STRUCTURED_MEMORY_EPISODE_GATE_MODE=intra_episode
STRUCTURED_MEMORY_MEMORY_START_FRAME=36
STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES=12
```

只有满足以下条件的 archive frame 才能参与 retrieval:

```text
archive_episode_id == current_episode_id
and archive_interval_end < current_frame - recent_exclude_frames
and current_frame >= memory_start_frame
```

默认 `allow_current_episode=False`, 所以旧的跨 episode 行为不变。只有 `intra_episode` 显式把该值设为 true。

`MEMORY_START_FRAME=36` 让前约 9 秒只积累 clean archive; `RECENT_EXCLUDE_FRAMES=12` 让独立 memory branch 不重复 native recent cache 的职责。首轮 gate 仅为 `0.05`, 并保留 confidence、alignment 和 convex fusion 抑制。

### 2.2 当前最小问题

首轮不同时加入 temporal RoPE、state segmentation、VLM entity registry、motion flow 或新的 eviction policy。先回答:

> 在 native recent cache 之外, 从同一 prompt 的非近期 clean 历史做小门控 Q-K recall, 是否能减缓长期漂移且不牺牲动态和动作正确性?

若答案是否定, 应停止包装该分支, 而不是继续叠加模块。

## 3. 代码修改

- `src/lifecycle_kv/attention_fusion.py`
  - 新增 `allow_current_episode`, 默认 false;
  - intra mode 只允许 `allowed_episode_id == current_episode_id`;
  - 缺失 episode sidecar 时继续 fail closed。
- `src/lifecycle_kv/episodic_archive.py`
  - `EpisodicArchiveConfig` 接受并校验 `intra_episode`。
- `third_party/Self-Forcing/pipeline/causal_inference.py`
  - 接入 `STRUCTURED_MEMORY_MEMORY_START_FRAME`;
  - config trace 和 stdout 打印 frame activation。
- `third_party/Self-Forcing/wan/modules/causal_model.py`
  - 单 prompt 按 frame 而不是 episode 激活;
  - 同 episode temporal admission;
  - interval sidecar 缺失时 fail closed;
  - trace 记录 selected interval、episode 和 frame age。
- `third_party/Self-Forcing/inference.py`
  - CLI 支持 `intra_episode` 和 `--structured_memory_memory_start_frame`。

### 3.1 Debug invariants

每条 accepted intra readout 记录:

```text
recall_scope
allow_current_episode
allowed_episode_id
memory_start_frame
recent_exclude_frames
eligible_frame_count
selected_intervals
selected_episode_ids
selected_frame_ages
selected_frame_age_min/mean/max
```

`scripts/analyze_hrem_v2_debug.py --strict` 会把以下情况判为硬错误:

1. intra readout 未显式允许 current episode;
2. `allowed_episode_id != current_episode_id`;
3. 选中的任一 frame 来自其他 episode;
4. 任一 selected age 小于等于 recent exclusion;
5. memory 在 `MEMORY_START_FRAME` 前生效;
6. accepted intra readout 缺少 interval/episode/age sidecar。

## 4. 首轮 30 秒实验

Prompt 文件:

```text
prompts/hrem_v2_single_long_complex_3.txt
```

三个 prompt 分别压力测试:

1. 人脸、服装、首饰、陶器几何和物体状态连续演化;
2. 高速 parkour、人体结构、动量和非重复动作;
3. 多人身份、食物状态、蒸汽、反射、拥挤背景和移动相机。

四个 cell:

| Cell | Archive | Readout | Head routing | 目的 |
|---|---:|---:|---|---|
| `native` | off | off | off | 原生 Self-Forcing baseline |
| `capture_only` | on | gate 0 | off | 检查 sidecar capture 的等价性和开销 |
| `intra_all_heads` | on | intra, gate 0.05 | off | continuity recall 上界/机制主测试 |
| `intra_role_hybrid` | on | intra, gate 0.05 | hybrid top 50% | head-selective 风险控制消融 |

首轮固定参数:

```text
120 latent frames, seed 0
archive budget 36, coverage eviction, spatial stride 4
top-k 3, shared Q-K selection
start frame 36, recent exclusion 12
noisy-only, layers [15, 21), convex fusion
retrieval temperature 0.20, confidence threshold 0.15
margin threshold 0.0, maximum entropy 1.0 (首轮只记录, 不作硬过滤)
```

不要在首轮同时搜索 gate、top-k、layer、position 和 archive budget。

## 5. 模型和目录

必须存在:

```text
third_party/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/
third_party/Self-Forcing/checkpoints/self_forcing_dmd.pt
```

默认输出:

```text
runs/hrem_v2_single_long_s0/
  native/
  capture_only/
  intra_all_heads/
  intra_role_hybrid/
  logs/
  traces/
  metrics_comprehensive.json
```

## 6. 服务器命令

拉取和测试:

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull --ff-only

PYTHONPATH=src pytest -q \
  tests/test_structured_memory_readout.py \
  tests/test_episodic_archive.py \
  tests/test_role_episodic.py \
  tests/test_hrem_debug_analyzer.py
```

四张 GPU 并行:

```bash
FORCE=0 RUN_EVAL=1 \
  bash scripts/run_hrem_v2_single_prompt.sh 0 1 2 3
```

单张 GPU 顺序运行:

```bash
PARALLEL=0 FORCE=0 RUN_EVAL=1 \
  bash scripts/run_hrem_v2_single_prompt.sh 0 0 0 0
```

显存或评估依赖不足时先只生成:

```bash
PARALLEL=0 FORCE=0 RUN_EVAL=0 \
  bash scripts/run_hrem_v2_single_prompt.sh 0 0 0 0
```

随后单独评估:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_comprehensive.py \
  --video_dirs \
    runs/hrem_v2_single_long_s0/native \
    runs/hrem_v2_single_long_s0/capture_only \
    runs/hrem_v2_single_long_s0/intra_all_heads \
    runs/hrem_v2_single_long_s0/intra_role_hybrid \
  --prompts prompts/hrem_v2_single_long_complex_3.txt \
  --output runs/hrem_v2_single_long_s0/metrics_comprehensive.json \
  --gpu 0
```

## 7. 必须观察的信息

结构检查:

```bash
python scripts/analyze_hrem_v2_debug.py \
  runs/hrem_v2_single_long_s0/traces/intra_all_heads.jsonl \
  --strict \
  --json-output runs/hrem_v2_single_long_s0/traces/intra_all_heads_diagnosis.json

grep -E '\[HREMv2\]' \
  runs/hrem_v2_single_long_s0/logs/intra_all_heads.log | tail -n 200
```

先确认:

```text
intra_episode_readouts > 0
intra_selected_frame_age_min > 12
all selected_episode_ids == current_episode_id == 0
delta_to_native_rms_median is nonzero and not excessive
no causal_invariant_violation
```

`capture_only` 的 analyzer 会按设计报告 `no_accepted_readout`, 因为 gate=0。该 cell 的正确检查是 archive commits 存在且最终视频与 native 一致, 不是要求 readout。

逐 prompt 人工检查:

| 维度 | 必须检查 |
|---|---|
| identity | 脸、头发、服装、首饰、配角身份是否随时间漂移 |
| object/state | 花瓶、食物、工具是否连续演化, 是否错误恢复旧状态 |
| motion | 多肢体、姿态回放、动作循环、冻结、速度突变 |
| camera | 移动是否连续, 是否被 memory 拉回旧视角 |
| background | 布局、文字、反射和光照是否稳定但仍有合理变化 |
| artifacts | 重影、闪回、双轮廓、局部粘连、突然变暗 |

VBench-Long 至少报告:

```text
subject_consistency
background_consistency
aesthetic_quality
imaging_quality
motion_smoothness
dynamic_degree
```

`DINO/subject` 上升但 `dynamic_degree` 明显下降不算成功。Parkour prompt 的人工 motion review 拥有否决权。

VBench-Long 四卡并行:

```bash
bash scripts/run_vbench_hrem_v2_single_prompt.sh 0 1 2 3
```

单卡顺序运行:

```bash
PARALLEL=0 bash scripts/run_vbench_hrem_v2_single_prompt.sh 0 0 0 0
```

## 8. Go/No-Go

### Stage 1: 机制有效

1. `intra_all_heads` 有 accepted readout;
2. selected age 和 episode invariants 全部通过;
3. median `delta/native RMS >= 1e-4` 且不超过保守上限 `0.10`;
4. `capture_only` 不改变生成结果, 或差异能被确定为运行非确定性;
5. 日志、commit、seed、frames 和 prompt SHA 完整。

### Stage 2: 三 prompt 筛选

1. 三条视频均无新增严重重影、多肢体或旧动作回放;
2. identity/object drift 至少两条稳定优于 native;
3. parkour 的 motion/dynamic 无明显退化;
4. 优势不是只出现在结尾单帧;
5. role cell 若保留, 必须有稳定非全开/非全关的 head gate。

不满足时先判定 continuity recall 负结果。只允许根据 trace 做一次单变量修正, 不做大规模无方向超参扫描。

### Stage 3: 扩展

仅将 Stage 2 winner 扩展到:

```text
seeds 1 and 2
32-prompt VBench-Long
60-second generation
correct-memory vs shuffled-V vs abstain specificity controls
```

60 秒收益必须比 30 秒更清楚, 否则“long-horizon”论点不成立。

## 9. 借鉴来源和学术边界

详细 provenance 以 `docs/64_related_work_code_provenance_and_claims.md` 为准。

| 工作 | 借鉴内容 | 当前区别和必须引用的边界 |
|---|---|---|
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | AR baseline、clean/noisy block 流程 | 我们是在其 attention 和 inference pipeline 上打补丁, 必须明确作为基础代码 |
| [Pyramid-Forcing](https://github.com/if-lab-pku/Pyramid-Forcing) | head-aware cache 的问题意识 | 当前 SF 分支使用在线 role evidence, 不能把 PF 的离线 head label 或其贡献表述为我们的发明 |
| [LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG) | 非近期历史、full-frame recall、recent exclusion | 不使用其训练 retrieval AE, 当前是 training-free Q-K 和独立 memory branch |
| [Echo-Forcing](https://github.com/mingqiangWu/Echo-Forcing) | preserve/recall/forget 和场景返回问题 | 必须引用其 episodic memory 先例, 不能声称首次做场景记忆 |
| [MemRoPE](https://github.com/YoungRaeKimm/MemRoPE) | pre-RoPE memory 和位置正确性原则 | 当前 continuity 首轮仍为 content readout, 尚未把其 positional method 包装为我们的贡献 |
| [SWIFT](https://github.com/ShanwenTan/SWIFT) | semantic injection cache、prompt-adaptive memory | HREM 的可守边界是显式 recall scope、fail-closed admission、独立 fusion 和审计协议, 仍需实验证明 |
| [IAMFlow](https://github.com/Eddie0521/IAMFlow) | identity/state memory 的相关任务定义 | 当前没有 entity registry、VLM/LLM 或 state graph, 不能声称 entity memory novelty |

当前可以诚实描述为:

> 我们研究一个 training-free、scope-conditioned historical recall hypothesis, 并实现了可失败关闭、可审计的 continuity/return 两类候选访问规则。

当前不能声称:

- 首次在长视频中使用 memory、retrieval、episode 或 head-aware cache;
- continuity recall 已经提升长视频质量;
- online head role 已经稳定或可解释;
- 已经优于任何第三方 baseline;
- 将公开工作的模块简单组合后就是新的论文贡献。

论文贡献只能在严格对照证明“scope-specific admission 比统一历史读取更有效, 且收益不是动态减弱造成”之后成立。

## 10. 结果回传清单

至少回传:

```text
runs/hrem_v2_single_long_s0/metrics_comprehensive.json
runs/hrem_v2_single_long_s0/traces/intra_all_heads_diagnosis.json
runs/hrem_v2_single_long_s0/traces/intra_role_hybrid_diagnosis.json
runs/hrem_v2_single_long_s0/logs/intra_all_heads.log
runs/hrem_v2_single_long_s0/logs/intra_role_hybrid.log
四个 cell 的 12 个视频或逐 prompt 对齐 comparison
```

人工 review 需要逐 prompt 给出 identity、state、motion、camera、background、artifact 和最终 preference, 不能只回复一个综合排序。
