# LifeCache v3.3 AMA Head 审计与双骨干实验结果

> 日期：2026-07-18  
> 状态：工程正确性通过；减少部分幻觉，但感知质量提升不显著，不能作为最终版本。

## 1. 评价原则

变暗、人物 ID 退化和背景幻觉本来就存在于原生 Self-Forcing/Causal-Forcing 长时外推中。LifeCache 的目标不是证明这些问题“不是我们引入的”，而是必须显著改善原生骨干。

本轮采用以下硬标准：

- SF 和原始 chunkwise CF 都必须验证；
- 人工 review 优先于 DINO/CLIP 等代理指标；
- 提高相似度但降低清晰度、自然运动或场景正确性不算成功；
- `longvideo.pt` 只作为训练上界，不作为 training-free 主对照；
- 正式结果必须保存在仓库工作目录，不能只放 `/tmp`。

## 2. AMA 历史结果复核

AMA 的早期 profiling 报告了大量 `identity_anchor` heads，并观察到 layer 10/20 强 anchor、layer 28 更混合。但后期审计推翻了直接移植这些标签的可靠性：

1. SF/CF FlashAttention 路径拿不到真实 attention weights，曾使用绝对 `|QK|` proxy。
2. 该 proxy 将 SF 和 CF 的 360 个 heads 全部判为 identity，导致 HRMR 对所有 heads 强化 anchor。
3. 结果是背景被锁、运动受限，旧的 SF/CF head-profile 实验不可作为可靠语义分类。
4. Pyramid-Forcing 的 `best_labels.csv` 中 `-1/1/2` 表示 oscillating/stable/stable-sparse 的 RoPE/cache 行为，不是 motion/identity 语义。

LifeCache v3.2 错把 PF oscillating heads 当成 motion heads，并禁止其读取 memory。layer 29 因此只有 5/12 heads 可访问历史帧。这是一个真实的分类语义错误。

处理方式：

- 新增 `memory_head_policy: all | pf_stable | explicit`；
- 默认 `all`，不再声称 PF label 是身份分类；
- `pf_stable` 仅保留为可复现实验对照；
- 后续 head 选择使用最终输出的因果消融，而不是失真的 proxy 分类。

## 3. 本轮首先修复的历史错误

- 恢复 LifeCache 未启用层的原生 attention fallback；此前 v3.2 灰噪声结果作废。
- gate=0 直接短路 memory branch。
- 修复 oracle clean block 捕获错帧。
- 修复 `compression: none` 丢失 frame/spatial RoPE metadata。
- strict oracle 不再静默回退 sparse bank。
- 每个 prompt 清理 bank、oracle 和 runtime state。
- QK score 改为 query chunk，消除 6.96 GiB 中间张量 OOM。
- clean-only 真实拒绝 denoising eviction。
- trace 每个进程启动时截断，避免旧实验混入。
- CF 使用真实 `chunkwise/causal_forcing.pt`，不再使用假 CF adapter。

## 4. Head-policy 审计

设置：单 prompt、120 帧、F0 clean full-frame memory、layer 29。

| Backbone/config | DINO | BG | Composite |
|---|---:|---:|---:|
| SF native | 0.8124 | 0.9418 | 0.4614 |
| SF gate 0.02 all | 0.8113 | 0.9561 | 0.4627 |
| SF gate 0.05 all | 0.8152 | 0.9425 | 0.4617 |
| SF gate 0.05 PF-stable | **0.7981** | 0.9457 | **0.4579** |
| CF native | 0.6828 | 0.8613 | 0.4211 |
| CF gate 0.02 all | **0.7022** | 0.8678 | **0.4259** |
| CF gate 0.05 all | 0.6957 | 0.8772 | 0.4241 |
| CF gate 0.05 PF-stable | 0.6990 | 0.8732 | 0.4228 |

结论：PF-stable 路由在 SF 上明确退化。解除错误 mask 是必要修复，但单层离散注入不足以显著改善 SF。

结果目录：`runs/v33_audits/head_audit/20260718_003150/`。

## 5. Layer 因果扫描

扫描 layer `0/5/10/15/20/25/29`，每次只启用一个 layer，gate=0.02。

主要结果：

- SF layer 15：DINO `0.8124 -> 0.8190`，drift slope `-0.01043 -> -0.01002`。
- SF layer 29：DINO基本持平，但 BG `0.9418 -> 0.9540`。
- CF layer 25：DINO `0.6828 -> 0.7157`，BG `0.8613 -> 0.8856`，drift slope `-0.01181 -> -0.00945`。

CF layer 25 是本轮最强的单 prompt 因果信号；视觉检查没有出现灰噪声或主体冻结。但其 RAFT acceleration 升高，必须依靠多 prompt 与人工 review 判断。

结果目录：`runs/v33_audits/layer_audit/20260718_005113/`。

## 6. 连续与离散 memory

SF 在原生 21 帧窗口淘汰 F0 后，从 frame 24 开始每 3 帧连续注入；CF 保持 frame 30/60/90 离散注入。

单 prompt 候选：

- SF layer15 continuous gate=0.02：DINO `0.8124 -> 0.8283`，BG `0.9418 -> 0.9502`。
- SF layer29 continuous gate=0.01：DINO `0.8124 -> 0.8249`，BG `0.9418 -> 0.9521`。
- CF continuous layer25 不如离散 layer25，说明 CF 对历史 memory dosage 更敏感。

结果目录：`runs/v33_audits/continuous_audit/20260718_105753/`。

## 7. 三 Prompt 泛化与人工结论

三类 prompt：

- p0：慢速人物与长时曝光；
- p1：快速跑酷与冻结/运动风险；
- p2：室内到雨夜场景切换与 stale memory。

### 7.1 固定门控

| Config | Composite | DINO | Drift slope | CLIP | BG | RAFT accel |
|---|---:|---:|---:|---:|---:|---:|
| SF native | 0.4450 | 0.6955 | -0.00727 | 0.2663 | 0.7935 | 70.56 |
| SF L15 continuous | 0.4510 | 0.7001 | -0.00666 | 0.2629 | 0.7985 | 82.77 |
| CF native | 0.4151 | 0.6487 | -0.00947 | 0.2539 | 0.8185 | 78.61 |
| CF L25 sparse | 0.4217 | 0.6608 | -0.00889 | 0.2550 | 0.8380 | 97.02 |

聚合指标有小幅提升，但逐 prompt 显示 SF p2 因持续读取 F0 而退化：DINO `0.6023 -> 0.5718`，BG `0.7661 -> 0.7398`。这是 stale recall，不是有效长期记忆。

人工 review 结论：相对 native 主要是部分幻觉减少，清晰度、曝光、纹理和整体质量没有显著提升；CF 同样没有明显质量跃升。因此该候选 **不满足项目目标**。

结果目录：`runs/v33_audits/candidate_3prompt/20260718_110948/`。

### 7.2 Output-alignment 自适应门控

新增逐 token/逐 head 门控：根据 recent branch 与 memory branch 输出余弦一致性抑制相反方向的 stale memory。

| Config | Composite | DINO | Drift slope | CLIP | BG | RAFT accel |
|---|---:|---:|---:|---:|---:|---:|
| SF adaptive threshold 0.00 | 0.4607 | 0.7151 | -0.00628 | 0.2591 | 0.8056 | 82.57 |
| SF adaptive threshold 0.25 | 0.4649 | 0.7158 | -0.00582 | 0.2664 | 0.8019 | 94.39 |
| CF adaptive threshold 0.00 | 0.4189 | 0.6593 | -0.00946 | 0.2562 | 0.8426 | 85.07 |
| CF adaptive threshold 0.25 | 0.4149 | 0.6533 | -0.00963 | 0.2519 | 0.8419 | 81.94 |

SF p2 DINO 恢复到 0.6098，说明自适应门控能抑制一部分 stale recall。但其完整帧额外 attention 开销很高，人工可见质量仍未形成显著跃升；CF 也不优于简单离散 memory。因此该机制保留为诊断工具，不作为最终方案。

结果目录：`runs/v33_audits/adaptive_3prompt/20260718_112425/`。

## 8. 当前研究判断

本轮证明：

1. clean historical KV 能对长时身份/背景漂移产生可测因果影响；
2. layer 与时间策略高度依赖 backbone；
3. 静态错误 head 分类会掩盖或反转收益；
4. full-frame memory 更擅长“限制幻觉”，不直接提升清晰度、纹理和曝光；
5. 相似度指标可能奖励保守生成，不能替代人工质量判断。

本轮没有证明：

- LifeCache 已显著优于原生 SF/CF；
- 变暗问题已经解决；
- 当前候选具备论文主结果强度。

## 9. 下一阶段优化方向

后续拆成两条正交路径，并做 2x2 消融：

### A. 原生质量稳定路径

- 记录每个 clean block 的 latent mean/std、亮度、频谱和 VAE decode 统计；
- 定位变暗首先发生在 latent、DiT output 还是 VAE decode；
- 移植并重新验证 AMA/RF 的 clean-stat anti-drift，但必须限制校正幅度；
- 对比 native、stat stabilization、memory、stat+memory。

### B. 低污染 memory 路径

- 从 full-frame attention 改为低预算 structured patch/identity memory；
- scene change 时停止旧背景 memory，人物 memory 与场景 memory 分离；
- 用 layer 因果 calibration 选择层，不使用 PF label 充当语义标签；
- 使用 A-B-A return prompt 验证 correct/wrong/shuffled memory，而不是只看连续 prompt。

Go 条件：两个骨干、至少 3 prompts × 3 seeds 上，人工 review 明显优于 native，同时 identity/background 指标提升，亮度与画质不退化，dynamic/motion 不通过冻结获得。

## 10. Review 路径

- `runs/REVIEW_v33_candidate_3prompt/`
- `runs/REVIEW_v33_adaptive_3prompt/`

这些目录仅用于人工 review；完整视频、日志、trace、contact sheet 和 JSON 均保存在 `runs/v33_audits/`。

原生曝光漂移的后续 latent 诊断见 `docs/38_native_latent_exposure_diagnosis.md`。
