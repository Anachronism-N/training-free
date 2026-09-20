#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:?runtime probe prepare schedule split preflight eval eval-missing status collect analyze package}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${V218_OUT_ROOT:?set a new shared v215_..._v218_eval output directory}"
ORIGINAL="${V218_SOURCE_ROOT:?set the completed original v215 generation root}"
RUNTIME="${V218_VBENCH_ROOT:?set a separate corrected VBench clone path}"
RECEIPT="$OUT/v218_runtime.json"
NODE_RANK="${NODE_RANK:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CACHE="${VBENCH_CACHE_DIR:?set the complete VBench model cache}"
HUB="${TORCH_HUB_DIR:?set existing torch hub cache}"
HOME_CACHE="${VBENCH_RUNTIME_HOME:?set existing runtime home cache}"
source "${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-longlive}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts:$ROOT/src:$ROOT:${PYTHONPATH:-}"
EVAL="$OUT/evaluation"
COMPARISON="$EVAL/vbench_comparison"
case "$ACTION" in
  runtime|probe|prepare|collect|analyze|package)
    [[ "$NODE_RANK" == 0 ]] || { echo "action requires rank0"; exit 2; } ;;
esac
case "$ACTION" in
  schedule)
    python - <<'PY'
import json
from v218_evaluation_repair import evaluation_plan
print(json.dumps(evaluation_plan(), indent=2))
PY
    ;;
  runtime)
    python "$ROOT/scripts/prepare_v218_vbench_runtime.py" --source-vbench "${VBENCH_ROOT:?set original unchanged VBench checkout}" \
      --target-vbench "$RUNTIME" --comparison-manifest "$ORIGINAL/evaluation/vbench_comparison/comparison_manifest.json" --receipt "$RECEIPT" ;;
  probe)
    CUDA_VISIBLE_DEVICES="${GPU_LIST%%,*}" python "$ROOT/scripts/probe_v218_raft.py" \
      --runtime-receipt "$RECEIPT" --checkpoint "$CACHE/raft_model/models/raft-things.pth" --output "$OUT/raft_probe.json" ;;
  prepare)
    python "$ROOT/scripts/v218_evaluation_repair.py" --source-root "$ORIGINAL" --output-root "$OUT" \
      --nodes "${V218_NODES_FILE:?eight actual interface IPs}" --runtime-receipt "$RECEIPT" ;;
  split)
    python - "$OUT" "$NODE_RANK" <<'PY'
import sys
from v218_evaluation_repair import Protocol
Protocol(sys.argv[1]).validate_node(int(sys.argv[2]))
PY
    python "$ROOT/scripts/prepare_v174_vbench_splits.py" --comparison-root "$COMPARISON" \
      --vbench-root "$RUNTIME" --node-rank "$NODE_RANK" --num-nodes 8 --workers 4 ;;
  preflight|eval|eval-missing|status|collect)
    if [[ "$ACTION" == eval || "$ACTION" == eval-missing ]]; then
      python "$ROOT/scripts/prepare_v155_vbench_local_cache.py" --vbench-cache "$CACHE" --torch-hub-dir "$HUB" --runtime-home "$HOME_CACHE" \
        --dino-repo "${VBENCH_DINO_REPO:-$CACHE/dino_model/facebookresearch_dino_main}" \
        --dreamsim-cache "${DREAMSIM_CACHE:-$HOME_CACHE/.cache}"
    fi
    # The shared eval action already skips valid completed jobs. Its legacy
    # eval-missing mode is single-node-only, so resume via the same 8-node split.
    MODE="$ACTION"
    [[ "$MODE" != eval-missing ]] || MODE=eval
    python "$ROOT/scripts/run_v218_vbench.py" "$MODE" --run-root "$OUT" --comparison-root "$COMPARISON" \
      --vbench-root "$RUNTIME" --vbench-cache "$CACHE" --parts-root "$EVAL/metrics/vbench_long_parts" \
      --summary-root "$EVAL/metrics" --analysis-root "$EVAL/analysis" --node-rank "$NODE_RANK" --num-nodes 8 \
      --gpu-list "$GPU_LIST" --local-models --torch-hub-dir "$HUB" --runtime-home "$HOME_CACHE" \
      --summary-stem vbench_core9_summary --analysis-stem v218_corrected_aggregate --summary-title "v215 with corrected upstream RAFT" ;;
  analyze)
    python "$ROOT/scripts/analyze_v215_lphc.py" --run-root "$OUT" --evaluation-repair
    python "$ROOT/scripts/export_lphc_paper_evidence.py" --v215-root "$OUT" --output-root "$OUT/paper_evidence" ;;
  package)
    python - "$OUT" <<'PY'
import sys, tarfile
from pathlib import Path
root = Path(sys.argv[1]).resolve()
with tarfile.open(root / "v218_small_artifacts.tar.gz", "w:gz") as archive:
    for sub in ("inputs", "jobs", "baseline", "decisions", "evaluation", "paper_evidence"):
        for path in sorted((root/sub).rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            parts = path.relative_to(root).parts
            if any(x in parts for x in ("runtime", "media", "published", "tensor_trace", "quarantine")):
                continue
            if path.suffix in {".json", ".jsonl", ".md", ".csv", ".yaml", ".txt", ".log"}:
                archive.add(path, arcname=str(path.relative_to(root)))
    for name in ("evaluation_repair.json", "v218_runtime.json", "raft_probe.json"):
        archive.add(root/name, arcname=name)
print(root / "v218_small_artifacts.tar.gz")
PY
    ;;
  *) echo "No generation action: v218 reuses all original v215 media."; exit 2 ;;
esac
