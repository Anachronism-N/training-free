#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-generate}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SF_REPO="${SF_REPO:-${ROOT}/third_party/Self-Forcing}"
PF_REPO="${PF_REPO:-${ROOT}/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-${SF_REPO}/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-${PF_REPO}/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/v162_freshmotion4_moviebench16/full8}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

DEFAULT_REUSE_ROOT="${ROOT}/runs/v161_state_matched_motion_moviebench16/full8"
if [[ -z "${V162_REUSE_V161_ROOT:-}" && \
      -s "${DEFAULT_REUSE_ROOT}/published_manifest.json" ]]; then
  export V162_REUSE_V161_ROOT="${DEFAULT_REUSE_ROOT}"
fi
export V120_BASELINE_ONLY=0
export V120_OURS_ONLY=0

runner=(
  "${PYTHON_BIN}" "${ROOT}/scripts/run_v162_freshmotion4_moviebench16.py"
  --repo-root "${ROOT}"
  --sf-repo "${SF_REPO}"
  --sf-config "${SF_CONFIG}"
  --sf-checkpoint "${CHECKPOINT}"
  --pf-repo "${PF_REPO}"
  --pf-config "${PF_CONFIG}"
  --pf-checkpoint "${CHECKPOINT}"
  --prompts "${PROMPTS}"
  --out-root "${OUT_ROOT}"
  --node-rank "${NODE_RANK}"
  --num-nodes "${NUM_NODES}"
  --gpu-list "${GPU_LIST}"
  --candidates "middle10_reservoir2_freshmotion4,middle10_reservoir2_statemotionpair1_reference,middle10_reservoir2_motionpair1_reference,middle10_reservoir4_reference,all_recent8_reference"
)

case "${ACTION}" in
  preflight|generate|audit)
    "${PYTHON_BIN}" "${ROOT}/scripts/build_v157_layer_gate_maps.py" --check
    "${runner[@]}" "${ACTION}"
    ;;
  mechanism)
    "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v162_state_motion_trace.py" \
      --trace-dir "${OUT_ROOT}/traces" \
      --output "${OUT_ROOT}/automated_screen/state_motion_trace.json"
    ;;
  status)
    find "${OUT_ROOT}/status" -maxdepth 1 -name 'node*.summary.json' \
      -print -exec cat {} \;
    ;;
  package)
    package_path="${OUT_ROOT}/v162_diagnostics.tar.gz"
    tar -C "${OUT_ROOT}" -czf "${package_path}" \
      automated_screen contracts diagnostics logs published_manifest.json status traces
    echo "${package_path}"
    ;;
  *)
    echo "usage: $0 {preflight|generate|audit|mechanism|status|package}" >&2
    exit 2
    ;;
esac
