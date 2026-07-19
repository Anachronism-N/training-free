# v4.2 30 秒对照、有效性审计与下一阶段设计

> 日期：2026-07-19
> 状态：三组 120 latent frames 与 VBench-Long 六维已完成；完整人工 review 待用户完成

## 1. 先纠正结论

`v41 clean-only 0.10` 不是已验证的最优方案。人工 review 已发现其开头存在重影和闪回，
而且 36 latent frames 只解码为 8.8 秒，不能代表长时间外推。当前没有任何 LifeCache
变体被证明在视觉质量上显著优于 PF，更不能据此声称优于 SF/CF。

需要区分三种证据：

| 结论 | 当前证据 | 状态 |
|---|---|---|
| memory 代码确实被执行 | 单测、不同配置输出 hash 不同、30 秒运行完成 | 已确认 |
| 压缩保持预算/区间并只提交 clean block | 单测覆盖 | 已确认 |
| memory 改善最终视频 | v41 人工发现伪影，v42 待完整 review | 未确认 |
| 能改善 SF/CF | 当前版本尚未移植做公平长视频对照 | 未确认 |
| scene switching/recall 有效 | 尚未开始正式 A-B/A-B-A 实验 | 未确认 |

## 2. v4.2 严格 30 秒对照

三组均为 120 latent frames，解码为 477 帧、16 FPS、29.8125 秒。checkpoint、prompt、
seed 和 PF 参数完全相同，只改变 memory 读取。

| 列 | 方法 | 路径 |
|---|---|---|
| 1 | PF | `runs/REVIEW_v42_memory_30s/pf/` |
| 2 | PF + all-step memory，gate 0.05 | `runs/REVIEW_v42_memory_30s/all005/` |
| 3 | PF + clean-only memory，gate 0.10 | `runs/REVIEW_v42_memory_30s/clean010/` |

人工 review 主入口：

```text
runs/REVIEW_v42_memory_30s/comparisons/
  0_threeway.mp4    # 秋日公园，三列完整 29.8 秒
  1_threeway.mp4    # 跑酷，三列完整 29.8 秒
  0_opening.png     # 前 2.5 秒密集抽帧
  1_opening.png
  0_timeline.png    # 0/6/12/18/24/29.8 秒
  1_timeline.png
```

初步静态检查只能得出两点。第一，跑酷开头约 0.5 秒的运动拖影在三组同时出现，主要是
PF/base 原生问题，不是 memory 单独引入；但当前方法也没有解决它。第二，三组在后半程
已明显分化，说明 memory 不是 no-op。是否存在液化、闪回、周期性背景边界和动作回放，
必须观看完整视频，不能由 contact sheet 代替。

两 prompt、30 秒的官方 VBench-Long 六维如下：

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.9355 | **0.9326** | 0.6046 | 0.6736 | 0.9763 | 0.9667 |
| All-step 0.05 | 0.9339 | 0.9315 | 0.6077 | 0.6828 | 0.9767 | 0.9667 |
| Clean-only 0.10 | **0.9371** | 0.9288 | **0.6108** | **0.6951** | **0.9782** | 0.9667 |

raw JSON 位于 `runs/vbench_long/v42_30s/`。clean-only 在四维略高，但 background 下降，
dynamic 完全相同；样本又只有两条。结合 v41 指标漏掉人工可见闪回的事实，这些数字不
构成晋级结论，只决定人工 review 时优先关注 clean-only 是否以伪影换取了局部稳定。

## 3. 当前实际融合了什么

| 来源 | 已采用内容 | 没有采用的内容 |
|---|---|---|
| Pyramid Forcing | 原生 per-head cache policy 作为强 baseline | PF 三类标签仍是原始离线标签，不属于我们的创新 |
| Flash-VAReason | 相邻信息融合、唯一性保留、固定 frame budget 的无训练压缩原则 | 没有照搬其任务特定模块 |
| Echo-Forcing | compatibility/confidence gating 的设计启发 | prompt schedule、scene pool、difference-aware decay、recall 尚未接入 |
| LongLive-RAG | clean-history descriptor、recent exclusion、固定总预算的设计审计 | 当前尚未实现完整帧检索；其 retrieval AE 需要训练，不能直接作为严格 training-free 组件 |
| IAMFlow | entity-organized memory 和覆盖选择的设计审计 | entity registry、LLM/VLM 选择尚未接入 |
| MemRoPE | 保存真实时空位置的必要性已纳入设计要求 | recalled frame 的 token-wise 3D position sidecar 尚未完成 |

所以当前代码不能写成“已经融合 Echo/LongLive/IAMFlow”。准确表述是：实现了 PF 上的
Flash-VAReason 风格结构化压缩和独立 query readout，其余工作目前只完成代码审计。

## 4. 为什么当前 readout 可能产生重影

当前 memory 把压缩历史 V 通过一个独立 attention 读出，再以小 gate 加到 native PF
attention output。它避免了与 native softmax 争抢概率，但仍有两个风险：

1. 历史主体和当前主体空间相位不一致时，直接叠加完整 V 可能形成双轮廓或闪回；
2. memory 是额外残差，不占用 native 历史预算，可能重复计入同一历史内容。

这只是与现象一致的故障假设，不是已证明根因。若 v4.2 完整人工 review 仍出现伪影，
不再继续扫 gate，而是停止 additive compressed-V 路线，改为完整帧检索和固定预算替换。

## 5. 下一项最有价值的融合

优先实现 **training-free full-frame archival retrieval**：

1. 从 clean generated latent 或现有 DiT 表示构造无训练 descriptor；不训练 LongLive-RAG AE；
2. archive 保存完整帧 K/V、绝对 frame index 和 spatial index，不把不同帧融合成伪帧；
3. 排除最近窗口，只从真正的远期历史检索，避免重复 local context；
4. memory token 从固定总 attention budget 中替换低价值历史 token，而不是额外加残差；
5. 用 MemRoPE 风格 token-wise 3D position 或有序 relative slot 保持时空关系；
6. 先做 oracle：手工指定正确历史帧若仍无提升，则停止自动 retrieval 开发。

这一组合吸收 LongLive-RAG 的检索和预算、MemRoPE 的位置、Echo 的冲突遗忘，但实现本身
保持 training-free。它比继续调整 `gate/temperature` 更可能直接解决重影和背景闪回。

## 6. PF 分类的可发表改造

不能只把 PF 的 Anchor/Wave/Veil 改名。建议改成 **Causal Functional Head Routing**：

- 对 anchor/history 做 drop 或 identity swap，测量每个 head 输出变化，定义 identity reliance；
- 打乱 recent frame 时间顺序，定义 motion/order reliance；
- 打乱历史 spatial token，定义 layout reliance；
- 在 A-B 和 A-B-A 中注入旧场景，定义 scene-conflict/recall response；
- 比较 clean 与 noisy pass 的响应，定义 denoising stability。

输出不是固定三分类，而是每个 `(layer, head, time/query)` 的连续角色向量：

```text
[identity, motion, layout, scene_conflict, clean_stability]
```

角色向量直接路由不同 bank：identity/scene/event archive、recent motion window、forget gate。
这与 PF 基于 sign-rate/FFT 的离线时间模式分类有实质区别，也能避免 AMA 中 SF/CF proxy
把 360 个 head 全分成 identity 的失败。深度维度不再多数投票，而是建模 role trajectory；
只有当该路由在同预算视频对照中带来收益，分类才升格为论文贡献，否则只保留为分析。

## 7. 场景切换路线

单 prompt 进入瓶颈后，直接复现 Echo 的三类协议：smooth A-B、hard-cut A-B、recall
A-B-A。第一轮比较 Echo native、Echo + training-free full-frame archive、Echo + causal
head routing，检查 transition latency、旧场景泄漏、A 的正确召回、B 污染和实体身份。

scene switching 目前尚未跑，不能报告为已有成果。它适合作为下一主任务，因为正确历史
有明确 ground truth，比单场景里“多一点历史是否有帮助”更容易验证 memory 是否真的会用。

## 8. 决策门

1. 先完整人工 review v4.2 两个三列视频；任一严重伪影即否决对应变体。
2. VBench-Long 六维只作辅助，不推翻人工伪影否决。
3. 若 v4.2 无一致优势，停止 additive readout 参数搜索，做 full-frame retrieval oracle。
4. oracle 通过后实现 fixed-budget + position-safe retrieval，再做 30/60 秒多 prompt。
5. PF 上通过后才移植到 native SF 和 Causal Forcing；最终论文必须在两者上成立。
6. 同时建立 Echo A-B/A-B-A benchmark，避免继续只在单 prompt 上盲目搜索。

## 9. 可复现性检查

- 30 秒生成：`bash scripts/run_v42_memory_30s.sh`
- 六维评估：`bash scripts/run_vbench_long_v42_30s.sh`
- 仓库测试：`PYTHONPATH=src pytest -q tests`，21 passed
- PF/memory 聚焦测试：46 passed
- 两个 shell 脚本均通过 `bash -n`，文档与脚本通过 `git diff --check`

这些检查确认实现链路可执行，不替代视觉有效性验证。
