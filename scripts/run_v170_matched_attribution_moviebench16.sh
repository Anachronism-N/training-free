#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-preflight}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SF_REPO="${SF_REPO:-${ROOT}/third_party/Self-Forcing}"
PF_REPO="${PF_REPO:-${ROOT}/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-${SF_REPO}/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-${PF_REPO}/configs/pyramid-forcing.yaml}"
CHECKPOINT="${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-${ROOT}/prompts/moviegen_128_qwen_v154_diverse16.txt}"
RUN_NAME="v170_matched_attribution_moviebench16"
SMOKE_PROMPT_INDEX="${V170_SMOKE_PROMPT_INDEX:-3}"

if [[ "${ACTION}" == "smoke" ]]; then
  OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${RUN_NAME}/smoke_p$(printf '%03d' "${SMOKE_PROMPT_INDEX}")}"
  NODE_RANK="${NODE_RANK:-0}"
  NUM_NODES="${NUM_NODES:-1}"
  GPU_LIST="${GPU_LIST:-0,1}"
else
  OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${RUN_NAME}/full8}"
  NODE_RANK="${NODE_RANK:-0}"
  NUM_NODES="${NUM_NODES:-4}"
  GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
fi

export V120_BASELINE_ONLY=0
export V120_OURS_ONLY=1
export V119_PROMOTION_APPROVED=1
export PYRAMIDKV_POLICY_TRACE_HEADS=0

CANDIDATES="v170_v166_a,v170_queryweighted_a,v170_v166_b,v170_queryweighted_b"
runner=(
  "${PYTHON_BIN}" "${ROOT}/scripts/run_v170_matched_attribution_moviebench16.py"
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
  --candidates "${CANDIDATES}"
)

case "${ACTION}" in
  smoke)
    [[ "${NODE_RANK}" == "0" && "${NUM_NODES}" == "1" ]] || {
      echo "smoke requires NODE_RANK=0 NUM_NODES=1" >&2
      exit 2
    }
    "${PYTHON_BIN}" "${ROOT}/scripts/build_v157_layer_gate_maps.py" --check
    V170_SMOKE=1 V170_SMOKE_PROMPT_INDEX="${SMOKE_PROMPT_INDEX}" \
      "${runner[@]}" generate
    ;;
  preflight|generate)
    [[ "${NUM_NODES}" == "4" ]] || {
      echo "v170 full ${ACTION} requires NUM_NODES=4" >&2
      exit 2
    }
    "${PYTHON_BIN}" "${ROOT}/scripts/build_v157_layer_gate_maps.py" --check
    "${runner[@]}" "${ACTION}"
    ;;
  audit)
    [[ "${NODE_RANK}" == "0" ]] || {
      echo "audit requires NODE_RANK=0" >&2
      exit 2
    }
    "${runner[@]}" audit
    ;;
  mechanism)
    [[ "${NODE_RANK}" == "0" ]] || {
      echo "mechanism requires NODE_RANK=0" >&2
      exit 2
    }
    "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v170_full_layer_trace.py" \
      --trace-dir "${OUT_ROOT}/traces" \
      --output "${OUT_ROOT}/automated_screen/full_layer_trace.json"
    ;;
  replica-hash)
    [[ "${NODE_RANK}" == "0" ]] || {
      echo "replica-hash requires NODE_RANK=0" >&2
      exit 2
    }
    "${PYTHON_BIN}" "${ROOT}/scripts/analyze_v170_replica_hashes.py" \
      --run-root "${OUT_ROOT}" \
      --output "${OUT_ROOT}/automated_screen/replica_hashes.json"
    ;;
  status)
    find "${OUT_ROOT}/status" -maxdepth 1 -name '*.summary.json' \
      -print -exec cat {} \;
    ;;
  package)
    package_path="${OUT_ROOT}/v170_diagnostics.tar.gz"
    tar -C "${OUT_ROOT}" -czf "${package_path}" \
      automated_screen contracts diagnostics logs published_manifest.json \
      status traces
    echo "${package_path}"
    ;;
  *)
    echo "usage: $0 {smoke|preflight|generate|audit|mechanism|replica-hash|status|package}" >&2
    exit 2
    ;;
esac
