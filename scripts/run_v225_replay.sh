#!/usr/bin/env bash
set -euo pipefail
ACTION=${1:?Use prepare, eval, status, collect, schedule, or package}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${V225_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v225_runs/replay32}
NODE_RANK=${NODE_RANK:-0}
if [[ "$ACTION" == package ]]; then
  [[ "$NODE_RANK" == 0 ]] || { echo "package requires rank 0"; exit 2; }
  [[ -f "$OUT/analysis/v225_analysis.json" ]] || { echo "collect must complete before package"; exit 2; }
  tar -C "$OUT" -czf "$OUT/v225_small_artifacts.tar.gz" \
    comparison_manifest.json v224_diagnosis.json schedule.json groups metrics analysis
  exit 0
fi
source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-longlive}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
python "$ROOT/scripts/v225_metric_replay.py" "$ACTION" --output-root "$OUT" --node-rank "$NODE_RANK" \
  --v219-root "${V219_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v219_runs/v219_38a06347_mechanism64}" \
  --v223-root "${V223_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v223_runs/v223_core32}" \
  --v224-root "${V224_OUT_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v224_runs/closure}" \
  --vbench-root "${VBENCH_ROOT:-/apdcephfs_gy2/share_302533218/cedricnie/v218_runtime/VBench_cf9b3b45}" \
  --vbench-cache "${VBENCH_CACHE_DIR:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/vbench}" \
  --torch-hub-dir "${TORCH_HUB_DIR:-/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/torch_hub}" \
  --runtime-home "${VBENCH_RUNTIME_HOME:-/tmp/training-free-v213-a39f503a8cb1/runs/_model_cache/dreamsim_home}" \
  --workers "${CPU_WORKERS:-8}" --ffmpeg "${FFMPEG:-ffmpeg}"
