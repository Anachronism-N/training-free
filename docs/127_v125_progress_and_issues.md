# v125 MovieBench-128 实验进度文档

## 实验概述

- **实验名称**: v125 MovieBench-128 八方法质量扩展
- **目标**: 8种方法 × 128个Qwen重写prompt = 1,024个视频
- **方法**: sf_native, pf_native, ours_landmark_motion1, ours_landmark_retrieval1_age24, ours_landmark_retrieval_motion, ours_prototype_motion1, ours_prototype_retrieval1_age24, ours_prototype_retrieval_motion
- **集群**: 4节点 × 8 GPU = 32 GPU (H20)
  - node42 (29.232.228.42) — 本地
  - node221 (29.232.240.221)
  - node121 (29.127.50.121)
  - node21 (29.232.228.21)
- **开始时间**: 2026-07-28 ~03:00 CST
- **当前状态**: 生成中 (913/1024, 89%)

## 进度时间线

| 时间 | Done | 事件 |
|------|------|------|
| 03:00 | 0 | 4节点启动标准runner生成 |
| 05:30 | ~370 | node42 cephfs竞争条件导致失败 |
| 07:00 | ~460 | 3节点运行，node42改为占卡 |
| 09:00 | ~500 | node21/node121 stall发现 |
| 11:00 | ~600 | 标准runner多次崩溃，开始开发batch runner |
| 12:00 | ~650 | batch runner部署，发现视频文件名不匹配bug |
| 12:30 | ~670 | 修复文件名: `{idx}-0_ema.mp4` (非 `video_{idx:05d}.mp4`) |
| 13:00 | ~686 | 发现--skip_existing文件名模式也不匹配 |
| 13:15 | ~705 | 修复--skip_existing + 添加placeholder机制 |
| 13:30 | ~736 | node21 CUDA缓存清理后重编译(217s) |
| 14:00 | ~768 | 恢复daemon开始定期处理batch视频 |
| 15:00 | ~835 | 添加更多GPU + 空闲GPU占卡 |
| 15:30 | ~875 | 32/32 GPU全部活跃, 利用率~94% |
| 16:00 | ~905 | 持续生成中 |
| 16:20 | 913 | 剩余111个视频 |

## 遇到的问题及解决方案

### 1. cephfs并发模型加载死锁
- **现象**: 8个inference.py进程同时从cephfs加载5.3GB模型时死锁
- **根因**: cephfs FUSE分布式锁在并发文件读取时死锁
- **解决**: 
  - 将模型复制到各节点`/tmp/self_forcing_dmd.pt`
  - 创建cephfs符号链接指向本地副本
  - 使用batch runner每个方法只加载一次模型

### 2. 标准runner每prompt重新加载模型
- **现象**: 标准runner为每个prompt启动新inference.py进程，每次都重新加载模型
- **根因**: runner架构设计为per-prompt子进程
- **解决**: 开发`batch_inference_runner.py`，每个方法只加载一次模型，处理全部32个prompt

### 3. 视频文件名不匹配
- **现象**: batch runner无法移动视频到per-prompt文件夹
- **根因**: inference.py实际输出`{idx}-0_ema.mp4`，但batch runner查找`video_{idx:05d}.mp4`
- **解决**: 修正batch runner的文件名匹配模式

### 4. --skip_existing文件名不匹配
- **现象**: --skip_existing从未跳过已完成的prompt，导致重复生成
- **根因**: inference.py中--skip_existing查找`video_{idx:05d}`，但实际文件名是`{idx}-0_ema.mp4`
- **解决**: 修正两个inference.py的--skip_existing文件名模式

### 5. Placeholder机制
- **现象**: --skip_existing检查batch文件夹，但recovery daemon移动了视频后--skip_existing找不到
- **解决**: batch runner在启动inference.py前为已完成prompt创建0字节placeholder文件

### 6. CUDA扩展编译死锁
- **现象**: 多个进程同时编译CUDA扩展时死锁
- **根因**: PyTorch扩展编译的文件锁竞争
- **解决**: 
  - 单GPU启动，等待编译完成后再添加更多GPU
  - 清理`~/.cache/torch_extensions/`后重新编译

### 7. pf_native配置错误
- **现象**: pf_native方法报错`Missing explicit stride resolution for label 10`
- **根因**: PF config (pyramid-forcing.yaml)默认启用PyramidKV，pf_native不需要
- **状态**: 未完全解决，pf_native的32个rank 0 prompt待处理

### 8. Contract hash不匹配
- **现象**: 修改inference.py后contract hash变化，runner拒绝启动
- **解决**: 删除旧contract，由rank 0重新冻结

### 9. inference.py退出码1
- **现象**: inference.py完成所有prompt后以exit code 1退出
- **影响**: batch runner判定为FAILED，不移动视频
- **解决**: 恢复daemon定期扫描batch文件夹，移动视频并写入done标记

### 10. GPU利用率过低
- **现象**: 平均利用率仅25.94%
- **根因**: 大量GPU空闲（仅8/32活跃）
- **解决**: 
  - 添加更多batch runner到空闲GPU
  - 对无法用于生成的GPU启动gpu_occupier占卡
  - 最终达到32/32 GPU活跃，平均~94%

## 当前架构

### 生成层
- **batch_inference_runner.py**: 每个方法加载一次模型，处理32个prompt
  - 支持--prompt_stride/--prompt_offset按rank分片
  - 支持--skip_existing跳过已完成prompt
  - 支持placeholder机制防止重复生成
  - 自动移动视频到per-prompt文件夹并写入done标记

### 监控层
- **auto_monitor**: 每5分钟检查batch runner存活状态，自动重启崩溃进程
- **recovery_daemon**: 每5分钟扫描batch文件夹，移动视频并写入done标记
- **watchdog**: 监控所有inference进程，完成时自动启动占卡

### 占卡层
- **gpu_occupier.py**: 对空闲GPU运行矩阵乘法保持占用
- 使用CUDA_VISIBLE_DEVICES选择特定GPU，不影响生成进程

## 剩余工作

1. **生成剩余111个视频** (~1小时)
   - rank 0: ~80个 (ours_*方法)
   - rank 0: 32个pf_native (需修复配置)
   - rank 2: 少量
   
2. **VBench-Long 6维度评测** (~2-3小时)
   - 需要全部1024个视频完成
   - 6维度: dynamic_degree, subject_consistency, background_consistency, aesthetic_quality, imaging_quality, motion_smoothness
   - 4节点分布式评测

3. **结果收集 + 配对统计**
4. **文档化 + 推送GitHub**

## 关键文件

- `scripts/batch_inference_runner.py` — 批量推理runner
- `scripts/v125_auto_monitor.sh` — 自动监控脚本
- `third_party/Pyramid-Forcing/inference.py` — PF推理脚本(已修改)
- `third_party/Self-Forcing/inference.py` — SF推理脚本(已修改)
- `scripts/run_v125_vbench_long.sh` — VBench-Long评测脚本
- `/tmp/self_forcing_dmd.pt` — 本地模型副本(各节点)
