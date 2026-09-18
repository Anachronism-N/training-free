# v213：SF基线对齐与启动确认

日期：2026-09-18。分支：`codex/v178-v179-causal-validation`。
这是docs/231的启动补充，不改变已经冻结的v212或历史结果。

## 1. 决定：保留FIFO21，不切换到PF runtime

本轮仍使用同一EMA checkpoint、同一Wan权重、同一SF实现运行原生SF与LPHC。
主对照是显式`local_attn_size=21, sink_size=0`，不是故意削弱后的SF。

| 项目 | 本轮设置 | 对齐范围 |
|---|---|---|
| 模型 | Self-Forcing DMD EMA，Wan2.1 1.3B | SF与LPHC同权重；以manifest的SHA为准 |
| 去噪 | `[1000,750,500,250]`，warp开启，shift=5 | 与官方SF/PF plain-SF配置核对 |
| AR块 | 3 latent frames，无独立首帧，clean context timestep=0 | 同一生成流程 |
| 精度/尺寸 | bf16，480×832，latent 60×104 | 与所用SF路径一致 |
| 局部上下文 | FIFO21，sink0，含当前3帧块 | 不是21历史帧再加3当前帧 |
| 视频 | 120 latent → 477输出帧，16FPS，约29.8s | 30秒任务，不声称精确480帧 |
| prompt/seed | Qwen改写MovieGen的32条，`21300+source` | 开发复验；不是PF完整128条主表复现 |
| CFG | 不启用few-step CFG；不用训练配置中的guidance数值额外加CFG | 避免改变蒸馏模型的原生采样 |

PF默认`local_attn_size=-1`时可以保存增长的KV，但self-attention实际受
`max_attention_size=32760=21×1560`限制。**保存量不等于读取量。**
我们的显式FIFO21保持固定物理缓存，PF原路径的分配方式不同。
读取窗口相同不足以直接证明RoPE、写入以及kernel完全等价，因此不声称PF runtime数值复现。
没有必要为了使用PF的缓存分配方式，引入另一套已修改的推理实现。

Echo-Forcing论文附录将SF对照写为Recent21，而其自身使用不同的结构化21帧缓存。
这支持把Recent21作为对照，但不意味着我们的缓存或整个实验协议与EF完全相同。

参考原始来源：

- [官方SF固定版本pipeline](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/pipeline/causal_inference.py)
- [官方SF固定版本causal attention](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/wan/modules/causal_model.py)
- [Pyramid Forcing论文](https://arxiv.org/html/2605.13111v1)，以及仓库内`third_party/Pyramid-Forcing/configs/self-forcing.yaml`
- [Echo-Forcing论文](https://arxiv.org/html/2605.16003v1)，附录C.6

## 2. 新增的官方SF数值检查

原gate0只验证本地SF与LPHC alpha=0的数值轨迹一致，不能证明本地SF没有共同的历史修改。
新增`baseline`用公共、只观测的harness分别加载：

1. 固定commit、无源码修改的官方SF：`upstream_a`。
2. 仓库内SF，关闭所有干预：`local`。
3. 同一官方SF再次运行：`upstream_b`。

官方版本固定为`33593df3e81fa3ec10239271dd2c100facac6de1`，不自动跟踪main。
harness不改官方源码、不替换attention，只注册forward hooks记录张量。
使用本轮冻结的FIFO21配置与checkpoint，所以这是**官方实现下的显式FIFO21协议检查**，
不是宣称复现官方CLI的所有默认参数，也不是PF/EF的完整复现。

source 3、67各运行这三条，共6条短轨迹，同一物理GPU、同一seed。
每条30 latent frames，即10个AR块、约7.3秒；第8块开始发生21帧缓存淘汰。
这是独立的数值canary，不是缩短主实验。主实验仍全部30秒。

记录与比较：

- 初始噪声、prompt embedding完整比较；噪声与CUDA RNG精确一致。
- 每条共50次generator调用，即10块×(4次去噪+1次clean写入)。
- 每次完整保存输入latent、timestep、输出flow/x0；完整比较最终latent。
- 每层每次抽样2048个K、V、self-attention输出值；30层均覆盖。
- 检查每层global/local指针、窗口长度；日志中的frame IDs由FIFO指针推导，
  **不是独立追踪到的实际帧标签**。张量比对才是额外的数值证据。
- 输出视频仅保留抽样数值作为诊断，不生成MP4，不要求人工review。
- 公共harness绕过CLI数据加载/视频封装，不能据此声称整个CLI或MP4字节级一致。

非精确浮点字段同时要求`max_abs<=5e-4`、`relative_l2<=1e-5`。
这是预设的严格工程一致性容差，不是质量阈值或统计显著性阈值。
两次官方运行本身也必须通过同样容差；不根据官方重复误差自动放宽。
decoded sample只作诊断，不因其有限浮点差异判定DiT不一致；非有限值仍报错。

检查失败就停止，先定位差异，不能为了开始主实验直接提高容差或跳过检查。
报告包含首批差异的event/call/tensor以及分事件最大relative-L2位置。
`generator_output`中的`k_N/v_N/attn_N`直接对应层N。

注意：这是两条短轨迹与部分内部张量的检查，不能证明所有prompt、长时运行或全部KV元素正确。
后续仍须运行原有alpha0门禁及完整30秒smoke。

## 3. 启动命令

先按docs/231创建独立、干净的实验worktree，并设置所有节点相同的
`V213_OUT_ROOT/GPU_LIST`，各自对应的`NODE_RANK/V213_NODE_ADDRESS`。
不要在运行中的v212 checkout里pull，也不要占用已有任务的GPU。
从rank0冻结一次commit，其他节点使用同一个commit，而非各自随时取最新分支。

rank0额外准备一个独立的官方源码目录，不需要复制checkpoint：

```bash
export UPSTREAM_SF_ROOT=/apdcephfs_gy2/share_302533218/cedricnie/reference_code/Self-Forcing-33593df3
git clone https://github.com/guandeh17/Self-Forcing.git "$UPSTREAM_SF_ROOT"
git -C "$UPSTREAM_SF_ROOT" checkout --detach 33593df3e81fa3ec10239271dd2c100facac6de1

# Run in the new training-free experiment checkout, rank0 only.
bash scripts/run_v213_experiment.sh prepare
bash scripts/run_v213_experiment.sh baseline
bash scripts/run_v213_experiment.sh gate0
bash scripts/run_v213_experiment.sh smoke
```

如果上述官方目录已经存在，不要重复clone；检查其commit与工作区，确认是干净的固定版本即可。
模型通过每个job运行目录下的`wan_models/Wan2.1-T2V-1.3B`软链接访问；
不修改官方源码，也不重复下载模型。继续使用已经配好的longlive环境。
`baseline`使用冻结GPU列表的第一张卡，不另选机器或GPU。

可选服务器测试（无GPU的控制逻辑测试）：

```bash
python -m pytest -q tests/test_v213_sf_baseline.py tests/test_v213_lphc.py tests/test_v212_lphc.py
```

baseline/gate0/smoke均完成后，各节点执行：

```bash
bash scripts/run_v213_experiment.sh generate32
bash scripts/run_v213_experiment.sh status
```

之后沿用docs/231的`publish → split → preflight → eval → collect → analyze → package`。
启动脚本会检查baseline的输入绑定、6个完成记录和4个配对比较；缺少或失败时不能继续生成。
v212不新增此门禁。已有冻结v213也不要中途升级checkout；本次补充用新输出目录运行。

## 4. 本轮主实验不变

仍是7方法×32 prompts，共224条30秒视频：

1. SF FIFO21：主要效果对照。
2. SF FIFO25：单纯增加局部容量的参考。
3. FIFO21 + 正确历史、early1、alpha=.02：主候选的第二seed复验。
4. 相同预算的随机历史：判断内容选择是否必要。
5. 正确历史、early1、alpha=.10：增强修正。
6. 正确历史、early2、alpha=.02：延长早期介入。
7. 正确历史、full、alpha=.02：全部去噪步介入。

其中smoke的7条直接复用进224条，不重复生成。
不新增PF生成、不做ABA。官方baseline的6条与alpha0的4条都是额外短数值检查。
如果SF25获胜、随机历史同样有效或motion下降，应如实调整主张。
目标是确认对SF的稳定增益，不要求SOTA，但不能用较弱的SF设置包装收益。

## 5. 回传与检查边界

优先回传：

```text
baseline/contract.json
baseline/jobs/source_003/{upstream_a,local,upstream_b}/done.json
baseline/jobs/source_067/{upstream_a,local,upstream_b}/done.json
baseline/jobs/source_*/{upstream_a,local,upstream_b}/{stdout,stderr}.log
decisions/sf_upstream_gate.json
decisions/gate0.json
```

失败时即使没有done或gate报告，也可用`package`打包现有日志。
`.npz`张量保留服务器，不随小文件包推送GitHub；需要深挖时再读相关event。
源文件、权重、配置和receipt有hash绑定；中断重跑同一命令会验证后复用完整job。
失败的数值结果不自动覆盖，修复后使用新commit、新输出root保留前后证据。

本地只完成无Torch的配置/配对/完整性测试与Python、Bash语法检查。
相关v209/v210/v211/v212/v213协议与新增baseline测试合计**93 passed, 2 skipped**；
两项Torch测试因本地没有Torch而跳过。Linux专用的`test_v210_postprocessing.py`
在Windows因缺少`fcntl`无法收集，不计入通过项；共享runner的文件锁仍需Linux执行。
实际GPU、FlashAttention、官方与本地模型轨迹的一致性尚待服务器验证。
这次没有修改SF模型算子或LPHC公式，也没有验证SF25相对官方21帧的数值等价。
VBench DD饱和与完整Semantic/Total协议问题也不会因baseline通过而自动解决；
本轮core-9与DD固定Quality仍按docs/231的开发诊断边界解释。
