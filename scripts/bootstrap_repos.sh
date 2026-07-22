#!/usr/bin/env bash
set -euo pipefail
mkdir -p third_party
cd third_party

clone_if_missing () {
  local name="$1"
  local url="$2"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name is already a Git checkout"
  elif [ -d "$name" ] && [ -n "$(find "$name" -mindepth 1 -print -quit)" ]; then
    echo "[keep] $name contains vendored files; not overwriting with a Git clone"
  else
    echo "[clone] $name <- $url"
    git clone "$url" "$name" || echo "[warn] failed to clone $name; check URL or access"
  fi
}

# Core AR video-generation baselines.
clone_if_missing Self-Forcing https://github.com/guandeh17/Self-Forcing.git
clone_if_missing Causal-Forcing https://github.com/thu-ml/Causal-Forcing.git
clone_if_missing RollingForcing https://github.com/TencentARC/RollingForcing.git

# KV-cache / sink / head-aware / memory-cache baselines.
clone_if_missing DeepForcing https://github.com/cvlab-kaist/DeepForcing.git
clone_if_missing Pyramid-Forcing https://github.com/if-lab-pku/Pyramid-Forcing.git
clone_if_missing Forcing-KV https://github.com/zju-jiyicheng/Forcing-KV.git
clone_if_missing MemRoPE https://github.com/YoungRaeKimm/MemRoPE.git
clone_if_missing LongLive-RAG https://github.com/qixinhu11/LongLive-RAG.git
clone_if_missing Echo-Forcing https://github.com/mingqiangWu/Echo-Forcing.git

# Entity / scene / agentic memory references.
clone_if_missing IAMFlow https://github.com/Eddie0521/IAMFlow.git

# RoPE / positional extrapolation references.
clone_if_missing infinity-rope https://github.com/yesiltepe-hidir/infinity-rope.git

# Spectral / PCA-style direct extension references.
clone_if_missing FreePCA https://github.com/JosephTiTan/FreePCA.git

# Additional verified references. Existing vendored trees are intentionally kept.
clone_if_missing DiT-Extrapolation https://github.com/thu-ml/DiT-Extrapolation.git
clone_if_missing FlowCache https://github.com/mikeallen39/FlowCache.git
clone_if_missing FreeLOC https://github.com/Westlake-AGI-Lab/FreeLOC.git
clone_if_missing LongVideoSparseAttention https://github.com/JiusiServe/LongVideoSparseAttention.git
clone_if_missing MIGA https://github.com/XiaokunFeng/MIGA.git
clone_if_missing MotionCache https://github.com/MAC-AutoML/MotionCache.git
clone_if_missing SWIFT https://github.com/ShanwenTan/SWIFT.git
