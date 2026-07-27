#!/usr/bin/env bash
# Four-node entrypoint for v125 generation, assembly, and VBench-Long.
set -uo pipefail

ACTION="${1:-}"
if [[ "$ACTION" != "generate" && "$ACTION" != "audit" && \
      "$ACTION" != "assemble" && "$ACTION" != "vbench-preflight" && \
      "$ACTION" != "vbench-split" && "$ACTION" != "vbench-eval" && \
      "$ACTION" != "vbench-collect" ]]; then
    echo "usage: bash scripts/run_v125_moviebench128_10h.sh \\"
    echo "  generate|audit|assemble|vbench-split|vbench-preflight|vbench-eval|vbench-collect"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CANDIDATES="landmark_retrieval1_age24,landmark_retrieval_motion"
PROMPT_FILE="${V125_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
REWRITE_SCRIPT="${V125_REWRITE_SCRIPT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/RollingForcing/scripts/prompt_refine_qwen.py}"
METHOD_SET_ID="$(
    python - "$CANDIDATES" <<'PY'
import hashlib
import sys

keys = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(f"ours{len(keys)}_{hashlib.sha256(','.join(keys).encode()).hexdigest()[:12]}")
PY
)" || exit 2
RUN_ROOT="${V125_RUN_ROOT:-$ROOT/runs/v125_moviebench128_main/$METHOD_SET_ID}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$ROOT/runs/v125_moviebench128_main/comparison}"

export REPO_ROOT="$ROOT"
export NODE_RANK
export NUM_NODES
export GPU_LIST
export V125_PROMPTS="$PROMPT_FILE"
export V125_REWRITE_SCRIPT="$REWRITE_SCRIPT"
export COMPARISON_ROOT

if [[ "$ACTION" == "generate" || "$ACTION" == "audit" || \
      "$ACTION" == "assemble" ]]; then
    [[ -s "$PROMPT_FILE" ]] || {
        echo "[error] missing Qwen Rewrite prompt file: $PROMPT_FILE"
        exit 2
    }
    [[ -s "$REWRITE_SCRIPT" ]] || {
        echo "[error] missing Qwen rewrite script: $REWRITE_SCRIPT"
        exit 2
    }
fi

case "$ACTION" in
    generate)
        python "$ROOT/scripts/run_v125_moviebench128_main.py" generate \
            --promotion-approved \
            --candidates "$CANDIDATES" \
            --prompts "$PROMPT_FILE" \
            --out-root "$RUN_ROOT" \
            --node-rank "$NODE_RANK" \
            --num-nodes "$NUM_NODES" \
            --gpu-list "$GPU_LIST"
        ;;
    audit)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] audit must run only on NODE_RANK=0"
            exit 2
        }
        python "$ROOT/scripts/run_v125_moviebench128_main.py" audit \
            --promotion-approved \
            --candidates "$CANDIDATES" \
            --prompts "$PROMPT_FILE" \
            --out-root "$RUN_ROOT" \
            --node-rank 0 \
            --num-nodes "$NUM_NODES" \
            --gpu-list "$GPU_LIST"
        ;;
    assemble)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] assemble must run only on NODE_RANK=0"
            exit 2
        }
        python "$ROOT/scripts/prepare_v125_moviebench128_comparison.py" \
            --prompts "$PROMPT_FILE" \
            --rewrite-script "$REWRITE_SCRIPT" \
            --source-root "$RUN_ROOT" \
            --comparison-root "$COMPARISON_ROOT"
        ;;
    vbench-preflight)
        bash "$ROOT/scripts/run_v125_vbench_long.sh" preflight
        ;;
    vbench-split)
        bash "$ROOT/scripts/run_v125_vbench_long.sh" split
        ;;
    vbench-eval)
        bash "$ROOT/scripts/run_v125_vbench_long.sh" eval
        ;;
    vbench-collect)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] vbench-collect must run only on NODE_RANK=0"
            exit 2
        }
        bash "$ROOT/scripts/run_v125_vbench_long.sh" collect
        ;;
esac
