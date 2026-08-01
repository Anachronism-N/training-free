#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-generate}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SF_REPO="${SF_REPO:-${ROOT}/third_party/Self-Forcing}"
PF_REPO="${PF_REPO:-${ROOT}/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-${SF_REPO}/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-${PF_REPO}/configs/pyramid-forcing.yaml}"
SHARED_CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
SF_CHECKPOINT="${SF_CHECKPOINT:-${SHARED_CHECKPOINT}}"
PF_CHECKPOINT="${PF_CHECKPOINT:-${SHARED_CHECKPOINT}}"
PROMPTS="${PROMPTS:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/v154_history_critical_moviebench16/full8}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export V153_TRANSFER_APPROVED="${V153_TRANSFER_APPROVED:-1}"
DEFAULT_REUSE_ROOT="${ROOT}/runs/v125_moviebench128_main"
if [[ -z "${V154_REUSE_V125_ROOT:-}" && \
      -s "${DEFAULT_REUSE_ROOT}/published_manifest.json" ]]; then
  export V154_REUSE_V125_ROOT="${DEFAULT_REUSE_ROOT}"
fi

runner=(
  "${PYTHON_BIN}" "${ROOT}/scripts/run_v154_history_critical_moviebench16.py"
  --repo-root "${ROOT}"
  --sf-repo "${SF_REPO}"
  --sf-config "${SF_CONFIG}"
  --sf-checkpoint "${SF_CHECKPOINT}"
  --pf-repo "${PF_REPO}"
  --pf-config "${PF_CONFIG}"
  --pf-checkpoint "${PF_CHECKPOINT}"
  --prompts "${PROMPTS}"
  --out-root "${OUT_ROOT}"
  --node-rank "${NODE_RANK}"
  --num-nodes "${NUM_NODES}"
  --gpu-list "${GPU_LIST}"
)

check_frozen_inputs() {
  "${PYTHON_BIN}" \
    "${ROOT}/scripts/analyze_v152_one_sided_history_critical.py" --check
  "${PYTHON_BIN}" \
    "${ROOT}/scripts/build_v154_history_critical_suite.py" --check
}

case "${ACTION}" in
  preflight)
    check_frozen_inputs
    "${runner[@]}" preflight
    ;;
  generate)
    check_frozen_inputs
    "${runner[@]}" generate
    ;;
  audit)
    if [[ "${NODE_RANK}" != "0" ]]; then
      echo "[error] audit must run on node rank 0" >&2
      exit 2
    fi
    check_frozen_inputs
    "${runner[@]}" audit
    ;;
  blind)
    "${PYTHON_BIN}" "${ROOT}/scripts/prepare_v154_blind_review.py" \
      --run-root "${OUT_ROOT}"
    ;;
  status)
    find "${OUT_ROOT}/status" -maxdepth 1 -name 'node*.summary.json' \
      -print -exec cat {} \;
    ;;
  package)
    package_path="${OUT_ROOT}/v154_diagnostics.tar.gz"
    tar -C "${OUT_ROOT}" -czf "${package_path}" \
      contracts diagnostics logs published_manifest.json status traces
    echo "${package_path}"
    ;;
  *)
    echo "usage: $0 {preflight|generate|audit|blind|status|package}" >&2
    exit 2
    ;;
esac
