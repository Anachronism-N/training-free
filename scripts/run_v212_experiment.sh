#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?action required}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN="${LPHC_CAMPAIGN:-v212}"
[[ "$CAMPAIGN" == v212 || "$CAMPAIGN" == v213 || "$CAMPAIGN" == v214 || "$CAMPAIGN" == v215 || "$CAMPAIGN" == v216 || "$CAMPAIGN" == v217 || "$CAMPAIGN" == v219 ]] || { echo "invalid campaign"; exit 2; }
OUT_VAR="${CAMPAIGN^^}_OUT_ROOT"
PROMPT_VAR="${CAMPAIGN^^}_SOURCE_PROMPTS"
OUT="${!OUT_VAR:?set the campaign shared output root on all nodes}"
source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-longlive}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
NODE_RANK="${NODE_RANK:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_NODES=6
if [[ "$CAMPAIGN" == v216 || "$CAMPAIGN" == v217 || "$CAMPAIGN" == v219 ]]; then NUM_NODES=8; fi
VBENCH_ROOT="${VBENCH_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/bench_baselines/VBench}"
EVAL="$OUT/evaluation"
COMPARISON="$EVAL/vbench_comparison"

case "$ACTION" in
  freeze|prepare|baseline|gate0|smoke|publish|collect|analyze|package)
    [[ "$NODE_RANK" == 0 ]] || { echo "action requires rank0"; exit 2; } ;;
esac
case "$ACTION" in
  split)
    python - "$NODE_RANK" "$CAMPAIGN" <<'PY'
import sys
from v212_lphc_protocol import load_protocol
load_protocol(sys.argv[2]).validate_node(int(sys.argv[1]))
PY
    ;;
esac
case "$ACTION" in
  freeze)
    if [[ "$CAMPAIGN" == v217 || "$CAMPAIGN" == v219 ]]; then
      PARENT_VAR="${CAMPAIGN^^}_V216_ROOT"
      FREEZE_SCRIPT=prepare_v217_replication.py
      [[ "$CAMPAIGN" != v219 ]] || FREEZE_SCRIPT=prepare_v219_mechanism.py
      python "$ROOT/scripts/$FREEZE_SCRIPT" --v216-root "${!PARENT_VAR:?set frozen v216 root}" --output-root "$OUT"
      exit 0
    fi
    [[ "$CAMPAIGN" == v216 ]] || { echo "freeze is only for v216"; exit 2; }
    python "$ROOT/scripts/prepare_v216_confirmation.py" --v215-root "${V216_V215_ROOT:?set completed v215 root}" \
      --output-root "$OUT" --nodes "${V216_NODES_FILE:?set JSON array of eight node interface IPs}" \
      --candidate "${V216_CANDIDATE:?select one candidate after v215 analysis}" \
      --primary-metric "${V216_PRIMARY_METRIC:-official_quality_score}" \
      --primary-window "${V216_PRIMARY_WINDOW:-full}" --rationale "${V216_RATIONALE:?record the choice and tradeoffs}"
    ;;
  baseline)
    [[ "$CAMPAIGN" == v213 || "$CAMPAIGN" == v214 || "$CAMPAIGN" == v215 || "$CAMPAIGN" == v216 || "$CAMPAIGN" == v217 || "$CAMPAIGN" == v219 ]] || { echo "baseline requires v213 or later"; exit 2; }
    python "$ROOT/scripts/run_v213_sf_baseline.py" --run-root "$OUT" --campaign "$CAMPAIGN" \
      --upstream-root "${UPSTREAM_SF_ROOT:?set a clean pinned official Self-Forcing checkout}" \
      --gpu "${GPU_LIST%%,*}"
    ;;
  prepare|gate0|smoke|generate32|generate48|generate64|generate80|generate96|status|schedule)
    python "$ROOT/scripts/run_v212_lphc.py" "$ACTION" --repo-root "$ROOT" \
      --campaign "$CAMPAIGN" --output-root "$OUT" --node-rank "$NODE_RANK" --gpu-list "$GPU_LIST" \
      --source-prompts "${!PROMPT_VAR:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}" \
      --checkpoint "${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}" \
      --wan-model "${WAN_MODEL:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B}"
    ;;
  publish)
    python "$ROOT/scripts/prepare_v212_comparison.py" --campaign "$CAMPAIGN" --run-root "$OUT" --vbench-root "$VBENCH_ROOT" ;;
  split)
    python "$ROOT/scripts/prepare_v174_vbench_splits.py" --comparison-root "$COMPARISON" \
      --vbench-root "$VBENCH_ROOT" --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" --workers 4 ;;
  eval|eval-missing|collect|eval-status|preflight)
    CACHE="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
    HUB="${TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
    HOME_CACHE="${VBENCH_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
    if [[ "$ACTION" == eval || "$ACTION" == eval-missing ]]; then
      python "$ROOT/scripts/prepare_v155_vbench_local_cache.py" --vbench-cache "$CACHE" \
        --torch-hub-dir "$HUB" --runtime-home "$HOME_CACHE" \
        --dino-repo "${VBENCH_DINO_REPO:-$CACHE/dino_model/facebookresearch_dino_main}" \
        --dreamsim-cache "${DREAMSIM_CACHE:-$HOME_CACHE/.cache}"
    fi
    MODE="$ACTION"
    [[ "$ACTION" != eval-status ]] || MODE=status
    [[ "$ACTION" != eval-missing ]] || MODE=eval
    python "$ROOT/scripts/run_v212_vbench.py" "$MODE" --comparison-root "$COMPARISON" \
      --campaign "$CAMPAIGN" \
      --vbench-root "$VBENCH_ROOT" --vbench-cache "$CACHE" --parts-root "$EVAL/metrics/vbench_long_parts" \
      --summary-root "$EVAL/metrics" --analysis-root "$EVAL/analysis" --node-rank "$NODE_RANK" \
      --num-nodes "$NUM_NODES" --gpu-list "$GPU_LIST" --local-models --torch-hub-dir "$HUB" --runtime-home "$HOME_CACHE" \
      --summary-stem vbench_core9_summary --analysis-stem "${CAMPAIGN}_aggregate" --summary-title "$CAMPAIGN LPHC"
    ;;
  analyze)
    DIRS=()
    mapfile -t METHODS < <(python -c "from v212_lphc_protocol import load_protocol; print('\n'.join(load_protocol('$CAMPAIGN').METHODS))")
    for method in "${METHODS[@]}"; do
      DIRS+=("$COMPARISON/published/$method")
    done
    PROMPT_COUNT="$(python -c "from v212_lphc_protocol import load_protocol; print(len(load_protocol('$CAMPAIGN').SOURCE_INDICES))")"
    python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" "${DIRS[@]}" \
      --output "$EVAL/metrics/temporal_diagnostics.csv" --expected-videos "$PROMPT_COUNT" --max-width 256 --frame-step 8 --workers 16
    python "$ROOT/scripts/bind_temporal_diagnostics.py" bind --comparison-manifest "$COMPARISON/comparison_manifest.json" \
      --temporal-csv "$EVAL/metrics/temporal_diagnostics.csv" --output "$EVAL/metrics/temporal_diagnostics.contract.json"
    python "$ROOT/scripts/analyze_${CAMPAIGN}_lphc.py" --run-root "$OUT"
    ;;
  package)
    python - "$OUT" "$CAMPAIGN" <<'PY'
import sys, tarfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
with tarfile.open(root / f"{sys.argv[2]}_small_artifacts.tar.gz", "w:gz") as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        parts = path.relative_to(root).parts
        if any(x in parts for x in ("runtime", "media", "published", "tensor_trace", "quarantine")):
            continue
        if path.suffix in {".json", ".jsonl", ".md", ".csv", ".yaml", ".txt", ".log"}:
            archive.add(path, arcname=str(path.relative_to(root)))
print(root / f"{sys.argv[2]}_small_artifacts.tar.gz")
PY
    ;;
  *) echo "freeze(v216/v217/v219) prepare baseline gate0 smoke generate32 generate48 generate64 generate80 generate96 schedule status publish split preflight eval eval-missing collect analyze package"; exit 2 ;;
esac
