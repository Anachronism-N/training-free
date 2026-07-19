# Memory readout 时机与 v3.9-v4.1 结果

> 日期: 2026-07-19
> 状态: 两 prompt / 36 latent frames 筛选完成；clean-only 0.10 待人工 review

## 1. 评估政策纠正

自定义 DINO、光流、loop、block-boundary 和 composite 只用于定位故障，不再用于选最终
方法，也不作为论文主结果。原因包括：

- DINO/背景一致性可能奖励冻结或重复；
- pixel flow 会混淆主体运动、镜头运动和背景漂移；
- 自定义 composite 无法与 PF、SF、LongLive-RAG 等工作横向比较；
- 单 prompt 的任意指标方差都很大。

当前筛选顺序改为：完整同步视频人工 review优先，VBench-Long 六维作为正式量化，
自定义指标仅解释具体伪影。最终主表必须使用 60s x 128 prompts 的 VBench-Long；当前
两 prompt 结果只用于快速淘汰。

## 2. v3.9: 独立 full-memory readout

结构化视觉 memory 使用 Flash-VAReason 启发的相邻融合、唯一性保留和固定预算，并通过
独立 query-conditioned attention 读取，不与 native PF self-attention 共用 softmax。

在单 prompt 上，all-step gate 0.05 提升自定义 DINO/BG，但 loop 明显增大。正式
VBench-Long 四个快速维度只有 subject 提升，其余三项下降：

| 方法 | Subject | Background | Aesthetic | Imaging |
|---|---:|---:|---:|---:|
| PF | 0.9838 | **0.9621** | **0.6098** | **0.6966** |
| Full memory 0.05 | **0.9864** | 0.9570 | 0.6083 | 0.6886 |

结论：独立 readout 确实影响生成并保主体，但持续注入完整历史 V 会损伤背景/成像，不能
直接作为 winner。

## 3. v4.0: spatial-detail memory

为去除可能导致旧背景回滚的低频场景分量，v4.0 从每个 memory frame 的 V 中减去空间
均值，只读空间细节残差。该方案降低部分 loop/debug 指标，但自定义 DINO 明显下降；
VBench-Long 也没有形成一致优势：

| 方法 | Subject | Background | Aesthetic | Imaging |
|---|---:|---:|---:|---:|
| PF | 0.9838 | **0.9621** | 0.6098 | **0.6966** |
| Detail memory 0.08 | **0.9840** | 0.9603 | **0.6125** | 0.6906 |

结论：身份信息并非纯空间高频残差，直接去均值会删除有用结构。v4.0 降级为负消融，
不扩到长视频。

## 4. v4.1: clean-only memory consolidation

PF 每块运行四次 noisy denoising forward，再运行一次输出不直接使用的 clean forward；
clean forward 的 K/V 会成为下一块历史。v3.9 在五次 forward 都读 memory，旧场景信号
被反复注入。

v4.1 只在 clean pass 读取历史：memory 不直接修改当前块的 denoised output，而是修正
将被下一块读取的 clean representation。这把方法从“每步历史残差注入”改为
“training-free memory consolidation”。

两 prompt VBench-Long 六维：

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.9270 | **0.9352** | **0.5844** | 0.6746 | 0.9710 | 0.8 |
| All-step 0.05 | **0.9321** | 0.9320 | 0.5821 | 0.6678 | 0.9710 | **0.9** |
| Clean-only 0.05 | 0.9293 | 0.9258 | 0.5796 | 0.6774 | **0.9724** | 0.8 |
| Clean-only 0.10 | 0.9295 | 0.9274 | 0.5751 | **0.6860** | 0.9723 | **0.9** |
| Clean-only 0.20 | 0.9220 | 0.9301 | 0.5698 | 0.6616 | 0.9717 | **0.9** |

相对 PF，clean-only 0.10 在 subject、imaging、motion、dynamic 四维上升，background 和
aesthetic 下降。clean-only 0.20 过强，明确淘汰。0.10 只能称为待人工确认候选：
VBench-Long 也可能奖励稳定/重复，且当前只有两条 8.8 秒视频。

## 5. 人工 review

首要入口：

```text
runs/REVIEW_v41_clean_memory/comparisons/
  0_fiveway.mp4  # 秋日公园
  1_fiveway.mp4  # 跑酷
  0_contact.png
  1_contact.png
```

五列顺序为 PF、all-step 0.05、clean-only 0.05、clean-only 0.10、clean-only 0.20。
人工检查必须观看完整视频，重点看：

- 公园背景是否仍在 block 边界突然换树/道路布局；
- 人脸、发型、衣物和身材是否持续一致；
- 跑酷腾空、落地和翻越时是否肢体液化、重影或障碍物穿透；
- 高 gate 是否出现动作回放、冻结或重复轨迹。

早期 v4.0 review：`runs/REVIEW_v40_memory/comparisons/`。

## 6. 官方结果路径

- v4.1 六维 VBench-Long：`runs/vbench_long/v41_full/`
- v3.9/v4.0 四维 VBench-Long：`runs/vbench_long/v39_v40_quick/`
- v4.1 raw videos：`runs/v35_pf_value_refresh/20260719_v41_*/`
- v3.9 debug metrics：`runs/v39_structured/`
- v4.0 debug metrics：`runs/v40_detail/`

## 7. 下一决策门

1. 人工 review 若 clean-only 0.10 在两条视频均优于 PF，才扩到 120 latent frames，并补
   cafe/scene transition prompt。
2. 若只改善跑酷或只改善公园，不继续扫单一 gate；改为 scene/identity 分支的动态读取
   时机，并保留 PF 作为控制。
3. 120 帧通过后移植到原生 SF 与 Causal Forcing；PF 结果不能替代双 backbone 验证。
4. 单场景长外推仍无稳定优势时，启动 Echo-Forcing prompt/scene switching 任务，直接
   检验 memory retrieval 是否选择正确历史而不是把旧场景拉回。
