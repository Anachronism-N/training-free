#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-screen}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PF_REPO="${PF_REPO:-${REPO_ROOT}/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-${PF_REPO}/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-${PF_REPO}/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-${PF_REPO}/configs/head_configs/best_labels.csv}"
SINGLE_PROMPTS="${SINGLE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
SINGLE_PROMPT_INDEX="${SINGLE_PROMPT_INDEX:-0}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/v153_history_critical_transfer_1video}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
MODE="${V153_MODE:-all}"

runner=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_v153_history_critical_transfer_1video.py"
  "${MODE}"
  --repo-root "${REPO_ROOT}"
  --pf-repo "${PF_REPO}"
  --pf-config "${PF_CONFIG}"
  --pf-checkpoint "${PF_CHECKPOINT}"
  --pf-labels "${PF_LABELS}"
  --single-prompts "${SINGLE_PROMPTS}"
  --single-prompt-index "${SINGLE_PROMPT_INDEX}"
  --out-root "${OUT_ROOT}"
  --gpu-list "${GPU_LIST}"
)

check_maps() {
  "${PYTHON_BIN}" \
    "${REPO_ROOT}/scripts/analyze_v152_one_sided_history_critical.py" \
    --check
}

case "${ACTION}" in
  prepare)
    "${PYTHON_BIN}" \
      "${REPO_ROOT}/scripts/analyze_v152_one_sided_history_critical.py"
    ;;
  preflight)
    check_maps
    "${runner[@]}" --preflight-only
    ;;
  screen)
    check_maps
    "${runner[@]}"
    ;;
  status)
    if [[ -d "${OUT_ROOT}/status" ]]; then
      find "${OUT_ROOT}/status" -maxdepth 1 -type f -print -exec cat {} \;
    else
      echo "no status directory: ${OUT_ROOT}"
    fi
    ;;
  package)
    package_path="${OUT_ROOT}/v153_diagnostics.tar.gz"
    tar -C "${OUT_ROOT}" -czf "${package_path}" \
      contracts configs diagnostics logs status traces
    echo "${package_path}"
    ;;
  *)
    echo "usage: $0 {prepare|preflight|screen|status|package}" >&2
    exit 2
    ;;
esac
