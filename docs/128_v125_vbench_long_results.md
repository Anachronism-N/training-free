# v125 MovieBench-128 VBench-Long 评测结果

## 实验概述

- **实验**: v125 MovieBench-128 八方法质量扩展
- **视频数**: 8方法 × 128个Qwen重写prompt = 1,024个视频
- **评测**: VBench-Long 6维度 (slow-fast模式)
- **时间**: 2026-07-28 (生成~15h + 评测~3h)
- **集群**: 4节点 × 8 GPU (H20)

## VBench-Long 6维度评测结果 (×100)

| Method | subject_consistency | background_consistency | aesthetic_quality | imaging_quality | motion_smoothness | dynamic_degree |
|--------|---------------------|------------------------|-------------------|-----------------|-------------------|----------------|
| sf_native | 97.18 | 96.37 | 59.26 | 68.18 | 98.57 | 43.23 |
| pf_native | 97.63 | 96.65 | 61.37 | 69.24 | 98.55 | 51.72 |
| ours_landmark_motion1 | 97.38 | 96.41 | 60.99 | 69.49 | 98.38 | 59.58 |
| ours_landmark_retrieval1_age24 | 97.37 | 96.43 | 60.96 | 69.37 | 98.38 | 58.75 |
| ours_landmark_retrieval_motion | 97.35 | 96.37 | 60.78 | 69.06 | 98.36 | 59.69 |
| ours_prototype_motion1 | 97.29 | 96.34 | 60.61 | 69.23 | 98.35 | 60.99 |
| ours_prototype_retrieval1_age24 | 97.29 | 96.38 | 60.59 | 69.13 | 98.38 | 61.93 |
| ours_prototype_retrieval_motion | 97.23 | 96.34 | 60.81 | 69.04 | 98.31 | 60.16 |

## 关键发现

### 1. 动态自由度 (Dynamic Degree) — 最显著差异
- **sf_native**: 43.23 (最低)
- **pf_native**: 51.72
- **ours_prototype_retrieval1_age24**: 61.93 (最高)
- Ours方法比SF高+17.7%, 比PF高+10.2%
- Prototype策略在动态自由度上优于Landmark策略

### 2. 主体一致性 (Subject Consistency)
- 所有方法在0.972-0.976范围内，差异较小
- pf_native最高(97.63), ours_prototype_retrieval_motion最低(97.23)
- 差异仅0.4%, 说明所有方法在短期一致性上表现相近

### 3. 背景一致性 (Background Consistency)
- 所有方法在0.963-0.967范围内
- pf_native最高(96.65), ours_prototype_retrieval_motion最低(96.34)

### 4. 美学质量 (Aesthetic Quality)
- pf_native最高(61.37), sf_native最低(59.26)
- Ours方法在60.59-60.99之间，介于SF和PF之间

### 5. 成像质量 (Imaging Quality)
- ours_landmark_motion1最高(69.49), sf_native最低(68.18)
- Ours方法整体优于SF，与PF接近

### 6. 运动平滑度 (Motion Smoothness)
- 所有方法在0.983-0.986范围内，差异极小
- sf_native最高(98.57), ours_prototype_retrieval_motion最低(98.31)

## 方法策略分析

### Landmark vs Prototype
- **Dynamic Degree**: Prototype (60.16-61.93) > Landmark (58.75-59.69)
- **Subject/Background**: Landmark 略优于 Prototype (差异<0.1%)
- **Aesthetic/Imaging**: 两者接近

### Retrieval策略的影响
- retrieval1_age24 vs motion1: 两者在大多数维度上接近
- retrieval_motion组合: 在dynamic_degree上表现最好(Prototype策略)

## 与v120实验对比 (32 vs 128 prompts)

v120实验(32 prompts)的关键发现:
- Ours方法在长期一致性(DINO drift)上优于SF
- Ours DINO=0.905 vs SF=0.887 (+0.019)

v125实验(128 prompts)验证:
- Ours方法在dynamic_degree上显著优于SF和PF
- 在短期一致性上所有方法接近
- Prototype策略在动态自由度上优于Landmark策略

## 技术细节

### 生成配置
- 模型: Self-Forcing DMD (Pyramid-Forcing架构)
- 帧数: 120 (120 frames × 4x VAE = 480 video frames)
- 种子: 0 (reseed_per_prompt)
- Prompt: MovieGen_128_qwen.txt (Qwen重写版本)

### 评测配置
- VBench-Long: slow-fast模式 (inclip 2s + clip2clip 跨clip)
- 6维度: subject_consistency, background_consistency, aesthetic_quality, imaging_quality, motion_smoothness, dynamic_degree
- 每个视频分15个clips, 每维度处理1920 clips per method
- 4节点分布式评测, 每节点12 jobs (8 GPU并行)

### 遇到的问题及解决方案

1. **cephfs并发模型加载死锁**: 使用batch_inference_runner.py + /tmp本地模型副本
2. **视频文件名不匹配**: 修正inference.py输出格式 `{idx}-0_ema.mp4`
3. **--skip_existing不生效**: 修复文件名匹配模式 + placeholder机制
4. **CUDA扩展编译死锁**: 单GPU启动, 等待编译完成后再添加GPU
5. **pf_native配置错误**: 创建pyramid-forcing-native.yaml禁用PyramidKV
6. **published目录符号链接**: 创建comparison_quality8/published -> ours6_9434cf7084d6/published
7. **VBench模型缺失**: RAFT模型和AMT模型分发到共享缓存
8. **NODE_RANK未设置**: 修复远程节点eval启动脚本
