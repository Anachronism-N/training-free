#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${V209_OUT_ROOT:-$ROOT/runs/v209_sf_protocol_budget}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
case "$ACTION" in
    prepare|canary|canary-native|canary-runtime|analyze-canary|generate32|status)
        python "$ROOT/scripts/run_v209_sf_protocol.py" "$ACTION" \
            --repo-root "$ROOT" --output-root "$OUT" \
            --source-prompts "${V209_SOURCE_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}" \
            --checkpoint "${SHARED_CHECKPOINT:-/apdcephfs_gy2/share_302533218/cedricnie/model_cache/self_forcing_dmd.pt}" \
            --backend "${V209_BACKEND:-production}"
        ;;
    vbench-prepare)
        [[ "${NODE_RANK:-0}" == "0" ]] || { echo "prepare requires node 0"; exit 2; }
        python "$ROOT/scripts/prepare_v209_comparison.py" --run-root "$OUT" --repo-root "$ROOT"
        ;;
    vbench-split)
        python "$ROOT/scripts/prepare_v174_vbench_splits.py" \
            --comparison-root "$OUT/screen32/vbench_comparison" \
            --vbench-root "${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}" \
            --workers 4 --node-rank "${NODE_RANK:-0}" --num-nodes "${NUM_NODES:-4}"
        ;;
    vbench-preflight|vbench-eval|vbench-status|vbench-collect|vbench-resume-missing)
        operation="${ACTION#vbench-}"
        [[ "$operation" != "resume-missing" ]] || operation=eval-missing
        if [[ "$operation" == "collect" && "${NODE_RANK:-0}" != "0" ]]; then
            echo "collect requires node 0"; exit 2
        fi
        CACHE="${VBENCH_CACHE_DIR:-$ROOT/runs/vbench_cache}"
        HUB="${V209_TORCH_HUB_DIR:-$ROOT/runs/_model_cache/torch_hub}"
        RUNTIME_HOME="${V209_RUNTIME_HOME:-$ROOT/runs/_model_cache/dreamsim_home}"
        if [[ "$operation" == "eval" || "$operation" == "eval-missing" ]]; then
            python "$ROOT/scripts/prepare_v155_vbench_local_cache.py" \
                --vbench-cache "$CACHE" --torch-hub-dir "$HUB" --runtime-home "$RUNTIME_HOME"
        fi
        python "$ROOT/scripts/run_v209_vbench.py" "$operation" \
            --comparison-root "$OUT/screen32/vbench_comparison" \
            --vbench-root "${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}" \
            --vbench-cache "$CACHE" --parts-root "$OUT/screen32/metrics/vbench_long_parts" \
            --summary-root "$OUT/screen32/metrics" --analysis-root "$OUT/screen32/analysis" \
            --node-rank "${NODE_RANK:-0}" --num-nodes "${NUM_NODES:-4}" --gpu-list "${GPU_LIST:-0,1,2,3,4,5,6,7}" \
            --summary-stem vbench_core9_summary --analysis-stem v209_aggregate \
            --summary-title "v209 Native SF Budget" --local-models --torch-hub-dir "$HUB" --runtime-home "$RUNTIME_HOME"
        if [[ "$operation" == "collect" ]]; then
            VIDEO_DIRS=()
            for method in sf_fifo21 sf_sink1_21 sf_sink1_13 sf_sink1_9 sf_sink3_9; do
                VIDEO_DIRS+=("$OUT/screen32/vbench_comparison/published/$method")
            done
            python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" "${VIDEO_DIRS[@]}" \
                --output "$OUT/screen32/metrics/temporal_diagnostics.csv" \
                --expected-videos 32 --max-width 256 --frame-step 8 --workers 16
            python "$ROOT/scripts/bind_temporal_diagnostics.py" bind \
                --comparison-manifest "$OUT/screen32/vbench_comparison/comparison_manifest.json" \
                --temporal-csv "$OUT/screen32/metrics/temporal_diagnostics.csv" \
                --output "$OUT/screen32/metrics/temporal_diagnostics.contract.json"
            python "$ROOT/scripts/analyze_v209_budget.py" --screen-root "$OUT/screen32"
        fi
        ;;
    package)
        python - "$OUT" <<'PY'
import sys, tarfile
from pathlib import Path
root = Path(sys.argv[1])
target = root / "v209_small_artifacts.tar.gz"
with tarfile.open(target, "w:gz") as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == target or "raw" in path.relative_to(root).parts or "published" in path.relative_to(root).parts:
            continue
        if path.suffix in {".json", ".jsonl", ".csv", ".md", ".yaml", ".log", ".txt"} and "vbench_long_parts" not in path.parts:
            archive.add(path, arcname=str(path.relative_to(root)))
print(target)
PY
        ;;
    *) echo "usage: $0 prepare|canary-native|canary-runtime|canary|analyze-canary|generate32|status|vbench-{prepare,split,eval,resume-missing,collect,status}|package"; exit 2 ;;
esac
