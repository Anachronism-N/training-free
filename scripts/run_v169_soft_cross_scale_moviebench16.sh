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
RUN_NAME="v169_soft_cross_scale_moviebench16"
SMOKE_PROMPT_INDEX="${V169_SMOKE_PROMPT_INDEX:-14}"
DEFAULT_OUT_ROOT="${ROOT}/runs/${RUN_NAME}/full8"
if [[ "${ACTION}" == "smoke" ]]; then
  DEFAULT_OUT_ROOT="${ROOT}/runs/${RUN_NAME}/smoke_p$(printf '%03d' "${SMOKE_PROMPT_INDEX}")"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
NODE_RANK="${NODE_RANK:-0}"
DEFAULT_NUM_NODES=4
if [[ "${ACTION}" == "smoke" ]]; then
  DEFAULT_NUM_NODES=1
fi
NUM_NODES="${NUM_NODES:-${DEFAULT_NUM_NODES}}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

DEFAULT_REUSE_ROOT="${ROOT}/runs/v168_cross_scale_consensus_moviebench16/full8"
export V169_REUSE_V168_ROOT="${V169_REUSE_V168_ROOT:-${DEFAULT_REUSE_ROOT}}"
export V120_BASELINE_ONLY=0
export V120_OURS_ONLY=0

runner=(
  "${PYTHON_BIN}" "${ROOT}/scripts/run_v169_soft_cross_scale_moviebench16.py"
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
  --candidates "middle10_reservoir2_directionmatch1,middle10_reservoir2_multiscalemotion1,middle10_reservoir2_multiscalepareto1,middle10_reservoir2_multiscalequeryweighted1,middle10_reservoir2_multiscalebottleneck1"
)

case "${ACTION}" in
  offline)
    [[ "${NODE_RANK}" == "0" ]] || {
      echo "offline requires NODE_RANK=0" >&2
      exit 2
    }
    "${PYTHON_BIN}" \
      "${ROOT}/scripts/analyze_v169_offline_counterfactual.py" \
      --output "${OUT_ROOT}/offline_counterfactual.json"
    ;;
  source-audit)
    [[ "${NODE_RANK}" == "0" ]] || {
      echo "source-audit requires NODE_RANK=0" >&2
      exit 2
    }
    "${PYTHON_BIN}" \
      "${ROOT}/scripts/analyze_v168_cross_scale_consensus_trace.py" \
      --trace-dir "${V169_REUSE_V168_ROOT}/traces" \
      --output "${OUT_ROOT}/source_v168_trace_corrected.json"
    ;;
  smoke)
    [[ "${NODE_RANK}" == "0" && "${NUM_NODES}" == "1" ]] || {
      echo "smoke requires NODE_RANK=0 NUM_NODES=1" >&2
      exit 2
    }
    "${PYTHON_BIN}" "${ROOT}/scripts/build_v157_layer_gate_maps.py" --check
    V169_SMOKE_PROMPT_INDEX="${SMOKE_PROMPT_INDEX}" "${runner[@]}" generate
    ;;
  preflight|generate|audit)
    "${PYTHON_BIN}" "${ROOT}/scripts/build_v157_layer_gate_maps.py" --check
    "${runner[@]}" "${ACTION}"
    ;;
  mechanism)
    "${PYTHON_BIN}" \
      "${ROOT}/scripts/analyze_v169_soft_cross_scale_trace.py" \
      --trace-dir "${OUT_ROOT}/traces" \
      --output "${OUT_ROOT}/automated_screen/soft_cross_scale_trace.json"
    ;;
  status)
    find "${OUT_ROOT}/status" -maxdepth 1 -name 'node*.summary.json' \
      -print -exec cat {} \;
    ;;
  package)
    package_path="${OUT_ROOT}/v169_diagnostics.tar.gz"
    tar -C "${OUT_ROOT}" -czf "${package_path}" \
      automated_screen contracts diagnostics logs offline_counterfactual.json \
      offline_counterfactual.md published_manifest.json status traces
    echo "${package_path}"
    ;;
  *)
    echo "usage: $0 {offline|source-audit|smoke|preflight|generate|audit|mechanism|status|package}" >&2
    exit 2
    ;;
esac
