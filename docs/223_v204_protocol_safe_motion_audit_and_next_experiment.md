# v204 协议安全的运动审计与下一步实验

> 日期：2026-09-07
>
> 状态：代码完成；复用已有视频，不生成新视频，不需要批量人工 review
>
> 主目标：判断现有 cache operator 是否相对 canonical SF 提升真实局部运动，同时不牺牲修正后的质量

## 1. 最新同步结果的正确解释

`main@dedbf4f5` 新增了四个连续 RAFT flow-magnitude 聚合值：

| 方法 | Raw flow magnitude |
|---|---:|
| `all_recent` | 535.8183 |
| `all_coverage_retrieval` | 584.5661 |
| `pf_native` | 545.8359 |
| `pf_hybrid_retrieval` | 588.1067 |

这四个数字不能放在同一张方法增益表中。协议审计发现：

- `all_recent` 使用 MovieGen source 128--255，prompt text SHA256 为
  `29b0390350dad3d14aa7acb10caacadded411cd94a39e9690b12c0b519bd4ac5`；
- Retrieval、PF 和 Hybrid 使用 source 0--127，prompt text SHA256 为
  `e05fc8e0f538aff5a810146b2c695ca23e5b5a9f0768ea996dfef6e4ccde2d9c`；
- 所以 `584.5661 - 535.8183` 混入了 prompt-suite 差异，既不能解释为
  Retrieval 的提升，也不能按相同索引做 paired bootstrap；
- 四份 raw-flow 结果中没有 `sf_native`，因此当前新增结果不能回答相对 SF 的运动增益。

唯一合法的新增配对是同属 source 0--127 的 Hybrid 对 PF：

| Window | Hybrid - PF | Relative | 95% paired bootstrap CI | Prompt wins |
|---|---:|---:|---|---:|
| Full | +42.2709 | +7.74% | [+13.2711, +71.8468] | 57.03% |
| Early | +48.7262 | +9.13% | [+23.6523, +72.8402] | 61.72% |
| Late | +36.7271 | +6.62% | [-2.3004, +74.8145] | 57.03% |

但 Hybrid 相对 PF 的修正后 Quality 为 `86.2431 - 86.6915 = -0.4485`，且七个
非 Dynamic Degree 质量维度全部下降。因此它只说明“替换长期历史 operator 可能提高
未补偿光流，同时造成质量损失”，不支持主方法晋级。

此外，生成 `video_mean_score` 的自定义 VBench 修改没有提交到当前仓库，raw flow 又会把
camera pan 计为主体运动，所以它只能是诊断信号，不能直接作为论文指标。

## 2. v204 新增内容

### 2.1 自动协议审计

`scripts/analyze_v204_continuous_flow_protocol.py` 在读取任何分数前绑定：

- 完整 prompt 文本哈希与 source-index 哈希；
- prompt 数量、seed、逐 prompt reseed 策略、latent 输出长度；
- decoded FPS、帧数和分辨率；
- 每个 prompt 的完整 15-clip 网格。

只有全部一致才计算 full、early 和 late 的 paired delta、bootstrap CI、prompt win
fraction、sign test 和 BH-corrected q-value。跨 suite 比较会明确输出
`invalid_cross_protocol_comparison`，不会产生一个看似可用的差值。

### 2.2 相机补偿的 SF 运动审计

v204 复用 v183 同 prompt、同 seed 的 512 个视频：

```text
sf_native
rccp_matched
all_recent
all_coverage
```

它调用仓库内已有且有 SHA 绑定的 `compute_v193_camera_motion.py`：

1. 计算 dense flow；
2. 对每个 transition 拟合稳健全局 affine camera field；
3. 从 raw flow 中移除全局 camera motion；
4. 测量局部 residual magnitude、active area、后段 motion ratio、低运动长区间和加速度异常；
5. 与 v202 中 Dynamic Degree 固定为 1.0 后的配对质量、身份/背景和时序证据联合判断。

主质量在报告中明确命名为 `quality_without_dynamic_degree`。这不是重新定义分数，而是
说明 v202 的所有方法和 prompt 已把 Dynamic Degree 固定为相同的 1.0，因此该维度不能
影响排序。

v204 只判断 operator 是否值得进入 v201，不重新证明旧 RCCP head membership，也不会
把探索性旧视频包装成论文确认结果。自动门控结束前不看片；即使通过，也最多输出四个
定点 review 样本。

## 3. 服务器执行

先拉取代码。在任意节点执行零 GPU 协议审计和质量修正：

```bash
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
git pull

bash scripts/run_v204_existing_video_motion_audit.sh protocol-audit
bash scripts/run_v204_existing_video_motion_audit.sh correct-quality
```

相机补偿分析使用 CPU。四节点分别设置 `NODE_RANK=0,1,2,3` 并行执行：

```bash
NUM_NODES=4 NODE_RANK=<0..3> V204_WORKERS=8 \
  bash scripts/run_v204_existing_video_motion_audit.sh motion-compute
```

完成后在 node 0 汇总：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v204_existing_video_motion_audit.sh motion-status
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v204_existing_video_motion_audit.sh motion-collect
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v204_existing_video_motion_audit.sh motion-analyze
NODE_RANK=0 bash scripts/run_v204_existing_video_motion_audit.sh show
NODE_RANK=0 bash scripts/run_v204_existing_video_motion_audit.sh package
```

该任务不占用视频生成 GPU，可以与 v189 profiling 并行。若 CPU 节点共享 GPU 作业的主机
资源，可将 `V204_WORKERS=4`，不会改变结果，只会延长运行时间。

需要推送的小文件：

```text
runs/v204_existing_video_motion_audit/analysis/
runs/v204_existing_video_motion_audit/motion/camera_compensated_motion.csv
runs/v204_existing_video_motion_audit/motion/camera_compensated_motion.contract.json
```

不要上传视频；这些视频已经存在于 v183。

## 4. GPU 主线不变

GitHub 仍没有 v189 或 v200 的最终服务器结果。因此下一项 GPU 工作不是再设计一个大规模
cache grid，而是完成已经冻结的证据链：

1. v189：128-prompt Landmark/Retrieval Head x Denoising-Phase shadow profiling；
2. v200：零 GPU 的 AR-horizon cross-fit 审计；
3. v201：只有 v200 通过后，运行 32-prompt classifier-holdout 生成。

v201 rev2 已经包含真正的 `sf_native`，主方法为：

```text
R(AR horizon, denoising call, layer, head) in {Recent, Coverage}

Recent   = sink1 + recent8
Coverage = sink1 + structured-middle4 + recent4
```

论文主门控只要求相对 SF 有可复现的显著正增益并通过非劣与运动安全检查，不要求超过
PF。static 与 horizon-shift controls 只决定能否进一步声称 AR-horizon 机制成立。

服务器当前应先执行：

```bash
# 先确认缺口
bash scripts/run_v196_campaign_frontier.sh show

# 若仍报告 v189 missing，按 docs/208 执行 v189；完成后立即运行
bash scripts/run_v200_head_phase_horizon_audit.sh analyze
bash scripts/run_v200_head_phase_horizon_audit.sh show
```

只有 v200 输出 `advance_head_phase_horizon_to_runtime_design`，才运行
`scripts/run_v201_head_phase_horizon_screen_32gpu.sh`。v204 的 operator 结果用于解释和
选择 Landmark/Retrieval 优先级，但不能绕过 v200 classifier gate。

## 5. 结果分支

| v204 | v200 | 下一步 |
|---|---|---|
| 存在质量非劣的强局部运动 operator | 通过 | 优先运行该 operator 的 v201，并保留另一个作机制对照 |
| 只有 motion-quality tradeoff | 通过 | 运行 v201，检验稀疏 Head x Phase x Horizon 路由能否恢复质量 |
| 无相对 SF 的局部运动信号 | 通过 | 仍可运行 v201，但主目标改为质量/身份/时序；不声称运动提升 |
| 任意 | 不通过 | 不运行 v201；停止当前 selective classifier，回到 operator/representation 设计 |

进入论文确认的最低条件是 v201 主候选在未参与 profiling 的 holdout 上相对 SF 有显著
正向主指标，其他轴非劣，并通过自动时序与局部运动安全检查。v204 本身不满足该条件。
