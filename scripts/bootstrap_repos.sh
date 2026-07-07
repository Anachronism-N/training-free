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

clone_if_missing Self-Forcing https://github.com/guandeh17/Self-Forcing.git
clone_if_missing Causal-Forcing https://github.com/thu-ml/Causal-Forcing.git
clone_if_missing RollingForcing https://github.com/TencentARC/RollingForcing.git
clone_if_missing Pyramid-Forcing https://github.com/if-lab-pku/Pyramid-Forcing.git
clone_if_missing Forcing-KV https://github.com/zju-jiyicheng/Forcing-KV.git
clone_if_missing MemRoPE https://github.com/YoungRaeKimm/MemRoPE.git
clone_if_missing LongLive-RAG https://github.com/qixinhu11/LongLive-RAG.git
clone_if_missing Echo-Forcing https://github.com/mingqiangWu/Echo-Forcing.git
