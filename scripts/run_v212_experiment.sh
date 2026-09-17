#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?action required}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${V212_OUT_ROOT:?set one shared v212_ output root on all nodes}"
source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-longlive}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
NODE_RANK="${NODE_RANK:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
EVAL="$V212_OUT_ROOT/evaluation"
COMPARISON="$EVAL/vbench_comparison"

case "$ACTION" in
  prepare|gate0|smoke|publish|collect|analyze|package)
    [[ "$NODE_RANK" == 0 ]] || { echo "action requires rank0"; exit 2; } ;;
esac
case "$ACTION" in
  split)
    python - "$NODE_RANK" <<'PY'
import sys
from v212_lphc_protocol import validate_node
validate_node(int(sys.argv[1]))
PY
    ;;
esac
case "$ACTION" in
  prepare|gate0|smoke|generate32|status)
    python "$ROOT/scripts/run_v212_lphc.py" "$ACTION" --repo-root "$ROOT" \
      --output-root "$V212_OUT_ROOT" --node-rank "$NODE_RANK" --gpu-list "$GPU_LIST" \
      --source-prompts "${V212_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}" \
      --checkpoint "${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}" \
      --wan-model "${WAN_MODEL:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B}"
    ;;
  publish)
    python "$ROOT/scripts/prepare_v212_comparison.py" --run-root "$V212_OUT_ROOT" --vbench-root "$VBENCH_ROOT" ;;
  split)
    python "$ROOT/scripts/prepare_v174_vbench_splits.py" --comparison-root "$COMPARISON" \
      --vbench-root "$VBENCH_ROOT" --node-rank "$NODE_RANK" --num-nodes 6 --workers 4 ;;
  eval|eval-missing|collect|eval-status|preflight)
    CACHE="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
    HUB="${TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
    HOME_CACHE="${VBENCH_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
    if [[ "$ACTION" == eval || "$ACTION" == eval-missing ]]; then
      python "$ROOT/scripts/prepare_v155_vbench_local_cache.py" --vbench-cache "$CACHE" \
        --torch-hub-dir "$HUB" --runtime-home "$HOME_CACHE"
    fi
    MODE="$ACTION"
    [[ "$ACTION" != eval-status ]] || MODE=status
    python "$ROOT/scripts/run_v212_vbench.py" "$MODE" --comparison-root "$COMPARISON" \
      --vbench-root "$VBENCH_ROOT" --vbench-cache "$CACHE" --parts-root "$EVAL/metrics/vbench_long_parts" \
      --summary-root "$EVAL/metrics" --analysis-root "$EVAL/analysis" --node-rank "$NODE_RANK" \
      --num-nodes 6 --gpu-list "$GPU_LIST" --local-models --torch-hub-dir "$HUB" --runtime-home "$HOME_CACHE" \
      --summary-stem vbench_core9_summary --analysis-stem v212_aggregate --summary-title "v212 Matched History"
    ;;
  analyze)
    DIRS=()
    for method in sf_fifo21 sf_sink1_21 sf_fifo25 fifo_correct fifo_random sink_correct sink_random; do
      DIRS+=("$COMPARISON/published/$method")
    done
    python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" "${DIRS[@]}" \
      --output "$EVAL/metrics/temporal_diagnostics.csv" --expected-videos 32 --max-width 256 --frame-step 8 --workers 16
    python "$ROOT/scripts/bind_temporal_diagnostics.py" bind --comparison-manifest "$COMPARISON/comparison_manifest.json" \
      --temporal-csv "$EVAL/metrics/temporal_diagnostics.csv" --output "$EVAL/metrics/temporal_diagnostics.contract.json"
    python "$ROOT/scripts/analyze_v212_lphc.py" --run-root "$V212_OUT_ROOT"
    ;;
  package)
    python - "$V212_OUT_ROOT" <<'PY'
import sys, tarfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
with tarfile.open(root / "v212_small_artifacts.tar.gz", "w:gz") as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        parts = path.relative_to(root).parts
        if any(x in parts for x in ("runtime", "media", "published", "tensor_trace", "quarantine")):
            continue
        if path.suffix in {".json", ".jsonl", ".md", ".csv", ".yaml", ".txt", ".log"}:
            archive.add(path, arcname=str(path.relative_to(root)))
print(root / "v212_small_artifacts.tar.gz")
PY
    ;;
  *) echo "prepare gate0 smoke generate32 status publish split preflight eval eval-missing collect analyze package"; exit 2 ;;
esac
