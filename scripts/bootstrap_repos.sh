#!/usr/bin/env bash
set -euo pipefail
mkdir -p third_party
cd third_party

clone_if_missing () {
  local name="$1"
  local url="$2"
  if [ -d "$name/.git" ]; then
    echo "[skip] $name already exists"
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

# The following directories appear in our third_party inventory but their canonical
# public repositories still need manual verification. Keep their local clones if
# they already exist; do not overwrite them with guessed URLs.
for name in DiT-Extrapolation FlowCache FreeLOC LongVideoSparseAttention MIGA MotionCache SWIFT; do
  if [ -d "$name" ]; then
    echo "[keep] $name exists locally; canonical URL pending verification"
  else
    echo "[todo] $name canonical GitHub URL not verified; add clone_if_missing once confirmed"
  fi
done
