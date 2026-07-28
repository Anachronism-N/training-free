#!/usr/bin/env bash
# Isolated v129 non-cache add-on screen and promotion runner.
set -uo pipefail

ACTION="${1:-}"
case "$ACTION" in
    screen-preflight|screen-generate|screen-audit|screen-analyze|\
confirm-preflight|confirm-generate|confirm-audit|confirm-analyze|\
full-preflight|full-generate|full-audit|full-analyze)
        ;;
    *)
        echo "usage: bash scripts/run_v129_noncache_addons.sh ACTION"
        echo "screen:  screen-preflight screen-generate screen-audit screen-analyze"
        echo "confirm: confirm-preflight confirm-generate confirm-audit confirm-analyze"
        echo "full:    full-preflight full-generate full-audit full-analyze"
        exit 2
        ;;
esac

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SOURCE_PROMPTS="${V129_PROMPTS:-/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"

SCREEN_CANDIDATES="value_control,value_var_s025,value_var_s050,value_var_s050_mid,value_var_s050_mid_t3"
SCREEN_INDEX="${V129_ADDON_SCREEN_INDEX:-0}"
CONFIRM_INDICES="${V129_ADDON_CONFIRM_INDICES:-0,7,15,23,31,39,47,55,63,71,79,87,95,103,111,127}"

case "$ACTION" in
    screen-*)
        STAGE="screen1"
        PROMPT_COUNT=1
        INDICES="$SCREEN_INDEX"
        CANDIDATES="${V129_ADDON_CANDIDATES:-$SCREEN_CANDIDATES}"
        ;;
    confirm-*)
        STAGE="confirm16"
        PROMPT_COUNT=16
        INDICES="$CONFIRM_INDICES"
        CANDIDATES="${V129_ADDON_CANDIDATES:-}"
        [[ -n "$CANDIDATES" ]] || {
            echo "[error] set V129_ADDON_CANDIDATES to one or two manually approved variants"
            exit 2
        }
        [[ "$CANDIDATES" != *,*,* ]] || {
            echo "[error] confirm16 accepts at most two promoted variants"
            exit 2
        }
        ;;
    full-*)
        STAGE="full128"
        PROMPT_COUNT=128
        INDICES=""
        CANDIDATES="${V129_ADDON_CANDIDATES:-}"
        [[ -n "$CANDIDATES" ]] || {
            echo "[error] set V129_ADDON_CANDIDATES to one approved variant"
            exit 2
        }
        [[ "$CANDIDATES" != *,* ]] || {
            echo "[error] full128 accepts exactly one promoted variant"
            exit 2
        }
        ;;
esac

METHOD_SET_ID="$(
    python - "$CANDIDATES" <<'PY'
import hashlib
import sys

keys = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(f"ours_only{len(keys)}_{hashlib.sha256(','.join(keys).encode()).hexdigest()[:12]}")
PY
)" || exit 2

STAGE_ROOT="${V129_ADDON_ROOT:-$ROOT/runs/v129_noncache_addons/$STAGE}"
RUN_ROOT="$STAGE_ROOT/$METHOD_SET_ID"
INPUT_ROOT="$STAGE_ROOT/inputs"
if [[ "$PROMPT_COUNT" == "128" ]]; then
    PROMPTS="$SOURCE_PROMPTS"
else
    PROMPTS="$INPUT_ROOT/moviegen_qwen_${PROMPT_COUNT}.txt"
    MANIFEST="$INPUT_ROOT/moviegen_qwen_${PROMPT_COUNT}.manifest.json"
    python "$ROOT/scripts/prepare_v129_addon_prompts.py" \
        --source "$SOURCE_PROMPTS" \
        --indices "$INDICES" \
        --output "$PROMPTS" \
        --manifest "$MANIFEST" || exit $?
fi

[[ -s "$PROMPTS" ]] || {
    echo "[error] missing add-on prompt file: $PROMPTS"
    exit 2
}

export REPO_ROOT="$ROOT"
export V129_ADDON_PROMPTS="$PROMPTS"
export V129_ADDON_PROMPT_COUNT="$PROMPT_COUNT"
export V129_ADDON_DURATION_SECONDS=30

run_generation() {
    local mode="$1"
    python "$ROOT/scripts/run_v129_noncache_addons.py" "$mode" \
        --ours-only \
        --promotion-approved \
        --candidates "$CANDIDATES" \
        --prompts "$PROMPTS" \
        --out-root "$RUN_ROOT" \
        --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES" \
        --gpu-list "$GPU_LIST"
}

case "$ACTION" in
    *-preflight)
        run_generation preflight
        ;;
    *-generate)
        run_generation generate
        ;;
    *-audit)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] audit must run only on NODE_RANK=0"
            exit 2
        }
        run_generation audit
        ;;
    *-analyze)
        [[ "$NODE_RANK" == "0" ]] || {
            echo "[error] analyze must run only on NODE_RANK=0"
            exit 2
        }
        python "$ROOT/scripts/analyze_v129_noncache_addons.py" \
            --run-root "$RUN_ROOT"
        ;;
esac
