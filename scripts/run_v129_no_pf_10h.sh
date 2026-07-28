#!/usr/bin/env bash
# Four-node v129 generation and evaluation entrypoint. PF and ABA are excluded.
set -uo pipefail

ACTION="${1:-}"
case "$ACTION" in
    preflight|generate-internal|audit-internal|generate-external|\
audit-external|assemble|analyze-gates|vbench-split|vbench-preflight|\
vbench-eval|vbench-collect)
        ;;
    *)
        echo "usage: bash scripts/run_v129_no_pf_10h.sh ACTION"
        echo "actions: preflight generate-internal audit-internal"
        echo "         generate-external audit-external assemble analyze-gates"
        echo "         vbench-split vbench-preflight vbench-eval vbench-collect"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PROMPT_FILE="${V129_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
INTERNAL_CANDIDATES="prototype_retrieval_conf_recent,prototype_retrieval_conf_motion"
METHOD_SET_ID="$(
    python - "$INTERNAL_CANDIDATES" <<'PY'
import hashlib
import sys

keys = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
digest = hashlib.sha256(",".join(keys).encode()).hexdigest()[:12]
print(f"ours_only{len(keys)}_{digest}")
PY
)" || exit 2
INTERNAL_ROOT="${V129_INTERNAL_ROOT:-$ROOT/runs/v129_moviebench128_30s_internal/$METHOD_SET_ID}"
EXTERNAL_ROOT="${V129_EXTERNAL_ROOT:-$ROOT/runs/v129_moviebench128_30s_external}"
V125_ROOT="${V125_COMPARISON_ROOT:-$ROOT/runs/v125_moviebench128_main/comparison_quality8}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$ROOT/runs/v129_paper_comparison_30s}"

export REPO_ROOT="$ROOT"
export NODE_RANK
export NUM_NODES
export GPU_LIST
export V129_PROMPTS="$PROMPT_FILE"
export V129_DURATION_SECONDS=30
export COMPARISON_ROOT

[[ -s "$PROMPT_FILE" ]] || {
    echo "[error] missing Qwen Rewrite prompt file: $PROMPT_FILE"
    exit 2
}
python - "$PROMPT_FILE" <<'PY' || exit 2
import sys
from pathlib import Path

prompts = [
    line.strip()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(prompts) != 128:
    raise SystemExit(f"expected 128 prompts, found {len(prompts)}")
PY

run_internal() {
    local mode="$1"
    python "$ROOT/scripts/run_v129_ours128_main.py" "$mode" \
        --ours-only \
        --promotion-approved \
        --candidates "$INTERNAL_CANDIDATES" \
        --prompts "$PROMPT_FILE" \
        --out-root "$INTERNAL_ROOT" \
        --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES" \
        --gpu-list "$GPU_LIST"
}

run_external() {
    local mode="$1"
    python "$ROOT/scripts/run_v129_external_baselines.py" "$mode" \
        --duration 30 \
        --methods deep_forcing,rolling_forcing,longlive \
        --prompts "$PROMPT_FILE" \
        --out-root "$EXTERNAL_ROOT" \
        --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES" \
        --gpu-list "$GPU_LIST"
}

case "$ACTION" in
    preflight)
        run_internal preflight || exit $?
        run_external preflight
        ;;
    generate-internal)
        run_internal generate
        ;;
    audit-internal)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] audit-internal must run only on NODE_RANK=0"
            exit 2
        }
        run_internal audit
        ;;
    generate-external)
        run_external generate
        ;;
    audit-external)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] audit-external must run only on NODE_RANK=0"
            exit 2
        }
        run_external audit
        ;;
    assemble)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] assemble must run only on NODE_RANK=0"
            exit 2
        }
        python "$ROOT/scripts/prepare_v129_paper_comparison.py" \
            --repo-root "$ROOT" \
            --prompts "$PROMPT_FILE" \
            --v125-root "$V125_ROOT" \
            --internal-root "$INTERNAL_ROOT" \
            --external-root "$EXTERNAL_ROOT" \
            --comparison-root "$COMPARISON_ROOT"
        ;;
    analyze-gates)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] analyze-gates must run only on NODE_RANK=0"
            exit 2
        }
        python "$ROOT/scripts/analyze_v129_retrieval_gate.py" \
            --run-root "$INTERNAL_ROOT" \
            --output-root "$INTERNAL_ROOT/analysis/retrieval_gate"
        ;;
    vbench-split)
        bash "$ROOT/scripts/run_v129_vbench_long.sh" split
        ;;
    vbench-preflight)
        bash "$ROOT/scripts/run_v129_vbench_long.sh" preflight
        ;;
    vbench-eval)
        bash "$ROOT/scripts/run_v129_vbench_long.sh" eval
        ;;
    vbench-collect)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] vbench-collect must run only on NODE_RANK=0"
            exit 2
        }
        bash "$ROOT/scripts/run_v129_vbench_long.sh" collect
        ;;
esac
