# 164: v156 精确上下文策略迁移实验与运行手册

日期：2026-08-02

## 1. 当前项目与实验进度

项目在研究长视频生成中的 training-free KV cache 分配：先通过 head
profiling 找出可能需要分散历史的 attention heads，再把分类结果转成有限
KV 预算下的生成策略，并用严格对照验证是否真的改善长程一致性。

截至当前版本：

- v155 生成完成：7 方法 x 16 prompts = 112 视频，audit、blind package 和
  diagnostics package 均通过；
- v155 冻结 VBench 合同实际是 16 dimensions x 7 methods = 112 tasks，当前
  53/112；科学上有效的 MovieBench core-9 是 53/63；
- 7 个 core dimensions 已全部完成，`subject_consistency` 与
  `background_consistency` 各完成 2/7，合计还缺 10 tasks；
- 旧 `collect` 要求全部 112 tasks，因此失败；现在已有 read-only status、
  missing-only resume 和 `collect-core`；
- DINO、DINOv2、CLIP、DreamSim 已改为共享离线缓存加载，已有 53 个结果的
  原冻结合同保持不变；
- v155 已完成维度上，top-reservoir 相对 bottom-reservoir 的 dynamic degree
  为 `-0.0375`，低于冻结 non-inferiority 下限 `-0.03`，所以原 metric
  promotion gate 已不可能通过；补齐 core-9 与人工盲评仍有诊断价值。

v155 的恢复细节和结果分别见 `docs/162_v155_run_results.md` 与
`docs/163_v155_recovery_head_profile_decision.md`。

## 2. 已修复的问题

### 2.1 VBench 网络与收集

- 不再把 63 误认为完整冻结任务数；完整数是 112，core-9 才是 63；
- `status` 不修改结果或合同；
- `resume-missing` 仅运行缺失 task；
- `collect-core` 允许独立收集合法 core-9，不等待缺少 benchmark auxiliary
  labels 的 7 个 semantic dimensions；
- 本地模型模式统一设置 torch hub 与 DreamSim runtime home，并校验离线模型；
- 旧结果合同只允许显式兼容的 wrapper 升级，不能宽泛跳过哈希检查。

### 2.2 v155 策略与 profiling 定义不完全一致

v155 的 `TemporalReservoirStrategy` 是确定性 Algorithm-R 样本，但不是
v152 的固定 `uniform8`：它包含 sink1、随机 reservoir4、recent4，读取上限
9 FFE；pending4 还会与 recent4 产生物理重复，选中 head 最多可占 13 FFE。

v156 新增 `TemporalProfileAnchorStrategy`，并加入完整配置链：

```text
CLI policy
  -> history_polarity_policy_overrides
  -> PyramidKVPipelineConfig
  -> CausalInferencePipeline
  -> build_compositions
  -> AdaptiveKVCache
  -> policy trace audit
```

新策略没有 pending queue、随机替换或 merge。显式 label sink 现允许为 0；
`profile_exact8` 会拒绝任何不是 `profile_anchor/recent8_exact` 的组合，避免
方法名和实际缓存行为不一致。

## 3. Head profiling 到底得到过什么分类

| 实验 | 可用结果 | 不能声称的内容 |
|---|---|---|
| v143 | 没有通过 gate 的静态 head axes | 不存在已验证的统一静态 taxonomy |
| v144 | camera 48、action 42、identity 32、scene 31、unresolved 207 | 57.5% unresolved，split agreement 仅 0.4556，不能当功能分类 |
| v145 | 16 个可复现 factor axes、51 个 state-specific candidates | 只是连续观测排序，不是硬类别 |
| v147-v148 | K 是最强 PF-independent axis | intervention specificity 失败，只能作 prior |
| v149-v151 | scalar tail / signed classes 未通过随机与校准确认 | 不可直接用于生成路由 |
| v152 | QK top4/layer 跨 seed 重叠 112/120 | oracle policy-choice gate 失败；其余 240 heads 只能叫 Default，不能叫 Recent-Critical |

因此没有已经验证、可部署的 head 功能分类。唯一值得做最后一次生成级反证的
是 v152 的 per-layer QK top4 candidate。正确应用方式是保留连续 score 与
bottom4/random4 对照，检验有限预算分配；不能把 top4 命名为已证实的语义
角色，也不能把 Default 解释为统一的 recent-preferring 类。

## 4. v156 精确上下文合同

v152 在 `history_frames=117`、`budget=8` 时的实现为：

```text
recent_count = 4
old_end = 117 - 4 = 113
old = round(linspace(0, 112, 4)) = [0, 37, 75, 112]
recent = [113, 114, 115, 116]
uniform8 = [0, 37, 75, 112, 113, 114, 115, 116]
recent8  = [109, 110, 111, 112, 113, 114, 115, 116]
```

v156 的 profile route 为 `sink0 + fixed anchors4 + recent4`，default route
为 `sink0 + recent8`。两者读预算和选中 head 的物理存储都不超过 8 FFE。
trace audit 强制检查 target ids、anchor 子集、唯一性、physical count 和读
token 上限。

边界必须明确：固定 anchors 在目标帧到达前会 underfill；它只在 v152 冻结的
frame-117 context 精确等价，不是每个生成时刻都重新均匀采样的 rolling
uniform8。要在有限 4-frame 存储下事后精确获得任意 rolling linspace 是做不到
的，因为未来才知道需要保留哪些已丢弃帧。v156 测试的是冻结 profiling
上下文能否迁移到生成，而不是宣称解决了通用 rolling uniform cache。

## 5. 冻结方法网格

| 方法 | label10 | label11 | 来源 |
|---|---|---|---|
| `sf_native` | SF | SF | 复用 v155 |
| `ours_qk_top4_profile_uniform4` | profile anchor4 + recent4 | recent8 | 新生成 |
| `ours_qk_bottom4_profile_uniform4_control` | profile anchor4 + recent4 | recent8 | 新生成 |
| `ours_qk_random4_profile_uniform4_control` | profile anchor4 + recent4 | recent8 | 新生成 |
| `ours_all_profile_uniform4_control` | profile anchor4 + recent4 | profile anchor4 + recent4 | 新生成 |
| `ours_all_recent8_exact_control` | recent8 | recent8 | 新生成 |
| `ours_qk_top4_reservoir4_reference` | v155 reservoir4 + recent4 | v155 recent8 | 复用 v155 |

总计 112 视频，其中 80 个新生成、32 个严格复用。复用路径校验 v155
experiment 名、prompt hash、contract hash、方法完整性与 16 个视频文件。

## 6. 代码入口

- `third_party/Pyramid-Forcing/pyramidkv/temporal_reservoir.py`：固定锚点策略；
- `third_party/Pyramid-Forcing/pyramidkv/policy_overrides.py`：精确预算路由；
- `scripts/run_v156_profile_exact_moviebench16.py`：冻结合同、复用、任务分片；
- `scripts/run_v156_profile_exact_moviebench16.sh`：生成、audit、blind、package；
- `scripts/prepare_v156_blind_review.py` / `analyze_v156_blind_review.py`：盲评；
- `scripts/prepare_v156_vbench_comparison.py`、`prepare_v156_vbench_splits.py`、
  `run_v156_vbench_long.py/.sh`、`analyze_v156_vbench.py`：离线可恢复 VBench。

## 7. 四节点运行方法

节点固定为：

```text
rank0  29.232.229.115
rank1  29.119.98.254
rank2  29.119.98.54
rank3  29.127.36.158
```

先在 rank0 执行 CPU preflight：

```bash
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate longlive
cd /apdcephfs_gy2/share_303214315/cedricnie/develop/training-free
PYTHON_BIN=python NODE_RANK=0 NUM_NODES=4 \
  bash scripts/run_v156_profile_exact_moviebench16.sh preflight
```

四节点分别设置 rank 后启动；每个节点 28 tasks，使用 8 张卡：

```bash
# 在对应节点将 NODE_RANK 设置为 0/1/2/3
export NODE_RANK=0 NUM_NODES=4 GPU_LIST=0,1,2,3,4,5,6,7
export V156_REUSE_V155_ROOT=$PWD/runs/v155_profile_aligned_moviebench16/full7
nohup bash scripts/run_v156_profile_exact_moviebench16.sh generate \
  > runs/v156_profile_exact_moviebench16/node${NODE_RANK}.log 2>&1 &
```

全部完成后在 rank0：

```bash
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v156_profile_exact_moviebench16.sh audit
bash scripts/run_v156_profile_exact_moviebench16.sh blind
bash scripts/run_v156_profile_exact_moviebench16.sh package
```

不得跳过 audit；`published_manifest.json` 的 `ok=true` 是后续 blind/VBench
输入的前置条件。

## 8. VBench core-9

v156 默认复用 v155 的共享离线模型缓存，不依赖网络代理：

```bash
NODE_RANK=0 bash scripts/run_v156_vbench_long.sh prepare

# 四节点分别运行 split，然后运行 eval；缺失任务用 resume-missing
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v156_vbench_long.sh split
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v156_vbench_long.sh eval
NODE_RANK=0 NUM_NODES=4 bash scripts/run_v156_vbench_long.sh resume-missing

# rank0 查看与收集合法 core-9
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v156_vbench_long.sh status
NODE_RANK=0 NUM_NODES=1 bash scripts/run_v156_vbench_long.sh collect-core
```

`eval` 与 `resume-missing` 默认 `V156_LOCAL_MODELS=1`。不要把缺少 auxiliary
labels 的 7 个 semantic dimensions 强行纳入 MovieBench 的有效结论。

## 9. 代理联通性

在 2026-08-02 当前执行节点实测：

```text
curl -x http://star-proxy.oa.com:3128 https://huggingface.co
curl: (5) Could not resolve proxy: star-proxy.oa.com

curl -x http://star-proxy.oa.com:3128 https://github.com
curl: (5) Could not resolve proxy: star-proxy.oa.com
```

因此以下环境变量在当前节点不能联网，失败发生在代理域名 DNS 解析阶段：

```bash
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
```

不同远程节点可能有不同 DNS，但必须逐节点用短超时验证；在验证前不能把代理
写入实验依赖。当前执行沙箱尝试 SSH 四节点时在 socket 层返回
`Operation not permitted`，所以没有得到四节点各自的代理结论。当前可靠方案
仍是共享离线缓存。

## 10. GPU 占卡约束

以 `/apdcephfs_gy2/share_303214315/cedricnie/GPU占卡.md` 为准。该文档把自动
占卡命令标记为“历史用法”，并明确：实验前可停止占卡，任务完成后除非获得
管理员明确批准，不要恢复占卡。这个现行规则与“实验结束后自动占卡”要求有
冲突，因此 v156 脚本不会自行启动占卡进程。

实验启动前应先确认 GPU 没有其他计算进程；需要释放旧占卡时在实例宿主终端
执行文档指定的停止命令：

```bash
bash /apdcephfs_gy4/share_302533218/cedricnie/stop_gpu_occupy.sh
```

实验期间由 4 节点 x 8 workers 保持 GPU 利用率。实验结束或失败后只做状态
报告；只有管理员明确批准时，才可按该文档恢复自动占卡。不要用 `pgrep` 判断
占卡状态，历史记录表明它会匹配 SSH 命令文本；应检查 `nvidia-smi` 的实际
进程、显存与利用率。

## 11. 晋级与停止条件

primary 是 `ours_qk_top4_profile_uniform4`。客观 gate 要求：

- 相对 bottom4 和 random4，history consistency 都严格提高；
- visual quality delta 均不低于 `-0.01`；
- temporal quality delta 均不低于 `-0.005`；
- dynamic degree delta 均不低于 `-0.03`；
- primary 的 history consistency 同时高于 all-profile 与 all-recent；
- 人工盲评对 bottom4 和 random4 的 required-control gate 通过。

仅当 objective gate 与 blind gate 同时通过，才讨论扩大 prompts。否则停止
静态 QK membership 路线；若 all-profile 或 reservoir 本身有收益，应把结果
归为 cache policy，而不是 head 分类。下一步转向 layer/timestep gating，把
K/QK propensity 当连续输入，而不是继续制造新的静态语义标签。

## 12. 本地验证结果

当前无可用 CUDA driver，所以没有在本节点生成视频。已完成的 CPU 验证：

```text
77 passed  # 新策略及 v97-v156 相关旧实验合同扩大回归
python -m py_compile: PASS
bash -n: PASS
git diff --check: PASS
v156 CPU preflight: PASS, rank0=28 tasks, reuse=true
```

新策略测试确认：固定 targets 为 `[0,37,75,112]`、无 pending storage、physical
frame count 为 4；7 方法共 112 tasks，四节点严格均分为 28/28/28/28。冻结
preflight contract SHA256 为
`3fcb89c774e9cf68b7b431eb5ceae4cb1741437e4b9adeb2b3503d34f2bca731`。
当前沙箱也禁止 SSH socket，因此没有在四个远程节点启动生成或占卡任务。
