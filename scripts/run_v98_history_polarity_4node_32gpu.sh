#!/usr/bin/env bash
# Four-node, 32-GPU screen for PF-independent history-polarity caching.
# Launch once per node with NODE_RANK=0,1,2,3 and eight local GPU ids.
set -uo pipefail

MODE="${1:-screen32}"
[[ "$MODE" == "screen32" || "$MODE" == "main128" ]] || {
    echo "usage: NODE_RANK=0..3 $0 screen32|main128"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
SCORE_ROOT="${SCORE_ROOT:-$ROOT/runs/v97_qk_head_scores}"
SCORES="${SCORES:-$SCORE_ROOT/scores/qk_head_scores.csv}"
SCORE_ARTIFACT="${SCORE_ARTIFACT:-$SCORE_ROOT/scores/qk_head_score_artifact.json}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
NODE_RANK="${NODE_RANK:-}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
PRELOAD_PYRAMIDKV="${PRELOAD_PYRAMIDKV:-1}"
MAP_WAIT_SECONDS="${MAP_WAIT_SECONDS:-600}"

if [[ "$MODE" == "screen32" ]]; then
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
    EXPECTED=32
    OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v98_history_polarity_screen32}"
else
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
    EXPECTED=128
    OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v98_history_polarity_main128}"
fi
MAP_DIR="${MAP_DIR:-$OUT_ROOT/maps}"

METHODS=(
    sf_native
    pf_native
    pf_explicit_parity
    pf_aw_hybrid_merge
    history_polarity_hybrid_merge
    history_polarity_stride_merge
    history_polarity_hybrid_merge_v78
    positive_rate_half_hybrid_merge
)

[[ "$NODE_RANK" =~ ^[0-3]$ ]] || {
    echo "[error] NODE_RANK must be one of 0,1,2,3"
    exit 2
}
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 8 ]] || {
    echo "[error] each v98 node requires exactly eight local GPU ids"
    exit 2
}
for path in \
    "$ROOT" "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" \
    "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$PF_LABELS" \
    "$SCORES" "$SCORE_ARTIFACT" "$PROMPTS" "$CONDA_SH"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq "$EXPECTED" ]] || {
    echo "[error] $MODE expects $EXPECTED prompts, found $PROMPT_COUNT"
    exit 2
}
(( PROMPT_COUNT % 4 == 0 )) || {
    echo "[error] prompt count must be divisible by four"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$SF:$ROOT/scripts:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1
export PYRAMIDKV_POLICY_TRACE_LAYERS="${PYRAMIDKV_POLICY_TRACE_LAYERS:-0,7,15,23,29}"
export PYRAMIDKV_POLICY_TRACE_STRIDE="${PYRAMIDKV_POLICY_TRACE_STRIDE:-3}"
export PYRAMIDKV_POLICY_TRACE_MAX_RECORDS="${PYRAMIDKV_POLICY_TRACE_MAX_RECORDS:-12000}"

mkdir -p \
    "$OUT_ROOT"/{logs,status,configs,traces,diagnostics,nodes} \
    "$MAP_DIR"

ensure_maps() {
    local manifest="$MAP_DIR/history_polarity_manifest.json"
    local lock="$OUT_ROOT/.map_build_lock"
    if [[ -s "$manifest" ]]; then
        return 0
    fi
    if mkdir "$lock" 2>/dev/null; then
        python "$ROOT/scripts/build_v98_history_polarity_maps.py" \
            --scores "$SCORES" \
            --score-artifact "$SCORE_ARTIFACT" \
            --pf-labels "$PF_LABELS" \
            --output-dir "$MAP_DIR"
        local status=$?
        rmdir "$lock" 2>/dev/null || true
        [[ "$status" -eq 0 && -s "$manifest" ]] || return 1
        return 0
    fi

    local waited=0
    while [[ ! -s "$manifest" && "$waited" -lt "$MAP_WAIT_SECONDS" ]]; do
        sleep 2
        waited=$((waited + 2))
    done
    [[ -s "$manifest" ]] || {
        echo "[error] timed out waiting for history-polarity maps"
        return 1
    }
}
ensure_maps || exit 1

POLARITY_ZERO="$MAP_DIR/history_polarity_zero.csv"
POSITIVE_HALF="$MAP_DIR/positive_rate_half.csv"
PF_AW="$MAP_DIR/pf_aw_binary_control.csv"
MAP_MANIFEST="$MAP_DIR/history_polarity_manifest.json"
for path in "$POLARITY_ZERO" "$POSITIVE_HALF" "$PF_AW" "$MAP_MANIFEST"; do
    [[ -s "$path" ]] || { echo "[error] missing generated map $path"; exit 2; }
done
python - "$MAP_MANIFEST" "$SCORES" "$PF_LABELS" <<'PY' || exit 2
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


manifest_path, scores, pf_labels = map(pathlib.Path, sys.argv[1:])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("score_csv_sha256") != digest(scores):
    raise SystemExit("[error] map manifest score hash mismatch")
if payload.get("pf_labels_sha256") != digest(pf_labels):
    raise SystemExit("[error] map manifest PF-label hash mismatch")
if payload.get("support_label") != 10 or payload.get("suppress_label") != 11:
    raise SystemExit("[error] map manifest does not use neutral labels 10/11")
for name, item in payload.get("maps", {}).items():
    path = pathlib.Path(item["path"])
    if not path.is_file() or digest(path) != item.get("sha256"):
        raise SystemExit(f"[error] stale or missing map {name}: {path}")
print("[HistoryPolarityMapAudit] hashes=ok labels=10/11", flush=True)
PY

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
SCORE_SHA256="$(sha256sum "$SCORES" | awk '{print $1}')"
MAP_MANIFEST_SHA256="$(sha256sum "$MAP_MANIFEST" | awk '{print $1}')"
SHARD_SIZE=$((PROMPT_COUNT / 4))
NODE_MANIFEST="$OUT_ROOT/nodes/node${NODE_RANK}.manifest.env"
if [[ -s "$NODE_MANIFEST" ]] && \
    find "$OUT_ROOT/status" -maxdepth 1 -type f \
        -name '*.done' -print -quit | grep -q .; then
    old_value() {
        local key="$1"
        awk -F= -v key="$key" \
            '$1==key {print substr($0,index($0,"=")+1)}' "$NODE_MANIFEST"
    }
    for pair in \
        "MODE=$MODE" \
        "RUN_COMMIT=$RUN_COMMIT" \
        "PROMPT_SHA256=$PROMPT_SHA256" \
        "PROMPT_COUNT=$PROMPT_COUNT" \
        "FRAMES=$FRAMES" \
        "SEED=$SEED" \
        "SCORE_SHA256=$SCORE_SHA256" \
        "MAP_MANIFEST_SHA256=$MAP_MANIFEST_SHA256"; do
        key="${pair%%=*}"
        expected_value="${pair#*=}"
        actual_value="$(old_value "$key")"
        [[ "$actual_value" == "$expected_value" ]] || {
            echo "[error] refusing mixed resume for $key: old=$actual_value new=$expected_value"
            echo "[error] use a new OUT_ROOT for a changed experiment"
            exit 2
        }
    done
fi
{
    printf 'EXPERIMENT=v98_history_polarity\n'
    printf 'MODE=%s\n' "$MODE"
    printf 'NODE_RANK=%s\n' "$NODE_RANK"
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'SCORE_SHA256=%s\n' "$SCORE_SHA256"
    printf 'MAP_MANIFEST_SHA256=%s\n' "$MAP_MANIFEST_SHA256"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'SHARDS=4\n'
    printf 'SHARD_SIZE=%s\n' "$SHARD_SIZE"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
} >"$NODE_MANIFEST"
rm -f "$OUT_ROOT/status/node${NODE_RANK}.done"

if [[ "$PRELOAD_PYRAMIDKV" == "1" ]]; then
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
        python -c "from pyramidkv import _ops; _ops._ensure_loaded(); print('[PyramidKVPreload] ok', flush=True)"
    ) >"$OUT_ROOT/logs/node${NODE_RANK}.pyramidkv_preload.log" 2>&1 || {
        echo "[error] PyramidKV extension preload failed on node $NODE_RANK"
        exit 2
    }
fi

write_config() {
    local name="$1" shard="$2" start="$3" end="$4" engine="$5"
    local labels="$6" route="$7" transition="$8" gpu="$9"
    local label_sha=""
    [[ -z "$labels" ]] || label_sha="$(sha256sum "$labels" | awk '{print $1}')"
    {
        printf 'name=%s\n' "$name"
        printf 'mode=%s\n' "$MODE"
        printf 'node_rank=%s\n' "$NODE_RANK"
        printf 'shard=%s\n' "$shard"
        printf 'start_idx=%s\n' "$start"
        printf 'end_idx=%s\n' "$end"
        printf 'gpu=%s\n' "$gpu"
        printf 'engine=%s\n' "$engine"
        printf 'labels=%s\n' "$labels"
        printf 'label_sha256=%s\n' "$label_sha"
        printf 'route=%s\n' "$route"
        printf 'transition=%s\n' "$transition"
        printf 'score_sha256=%s\n' "$SCORE_SHA256"
        printf 'map_manifest_sha256=%s\n' "$MAP_MANIFEST_SHA256"
        printf 'run_commit=%s\n' "$RUN_COMMIT"
        printf 'prompt_sha256=%s\n' "$PROMPT_SHA256"
        printf 'frames=%s\n' "$FRAMES"
        printf 'seed=%s\n' "$SEED"
        printf 'reseed_per_prompt=1\n'
    } >"$OUT_ROOT/configs/$name.shard$shard.env"
}

run_cell() {
    local name="$1" gpu="$2" shard="$3" start="$4" end="$5"
    local engine="$6" labels="$7" route="$8" transition="$9"
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.shard$shard.log"
    local marker="$OUT_ROOT/status/$name.shard$shard.done"
    local policy_trace="$OUT_ROOT/traces/$name.shard$shard.policy.jsonl"
    local transition_trace="$OUT_ROOT/traces/$name.shard$shard.transition.jsonl"
    local head_args=() route_args=() transition_args=()

    mkdir -p "$output"
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        python "$ROOT/scripts/audit_indexed_videos.py" \
            --video-dir "$output" --start-idx "$start" --end-idx "$end" \
            >/dev/null 2>&1; then
        echo "[skip] $name shard=$shard"
        return 0
    fi
    rm -f "$marker" "$policy_trace" "$transition_trace"
    write_config \
        "$name" "$shard" "$start" "$end" "$engine" \
        "$labels" "$route" "$transition" "$gpu"

    if [[ -n "$labels" ]]; then
        head_args=(--pyramidkv_head_config_path "$labels")
    fi
    case "$route" in
        none|native)
            ;;
        pf_explicit_parity)
            route_args=(
                --pyramidkv_binary_stable_policy stride
                --pyramidkv_binary_responsive_policy cyclic
            )
            ;;
        history_hybrid_merge)
            route_args=(
                --pyramidkv_history_polarity
                --pyramidkv_history_support_policy hybrid
                --pyramidkv_history_suppress_policy merge
            )
            ;;
        history_stride_merge)
            route_args=(
                --pyramidkv_history_polarity
                --pyramidkv_history_support_policy stride
                --pyramidkv_history_suppress_policy merge
            )
            ;;
        *)
            echo "[error] unknown route $route"
            return 2
            ;;
    esac
    if [[ "$transition" == "1" ]]; then
        transition_args=(
            --pyramidkv_cache_transition
            --pyramidkv_cache_transition_mode full
            --pyramidkv_cache_transition_min_reliability .55
            --pyramidkv_cache_transition_min_novelty .01
            --pyramidkv_cache_transition_max_commit_fraction .75
            --pyramidkv_cache_transition_stagger_period 1
            --pyramidkv_cache_transition_max_age_blocks 6
            --pyramidkv_cache_transition_branches both
            --pyramidkv_cache_transition_denoise_weight 2
            --pyramidkv_cache_transition_trace_path "$transition_trace"
            --pyramidkv_cache_transition_debug
        )
    fi

    if [[ "$engine" == "sf" ]]; then
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export COMMIT_FORCING_ENABLE=0
            cd "$SF" || exit 2
            python inference.py \
                --config_path "$SF_CONFIG" \
                --checkpoint_path "$SF_CHECKPOINT" \
                --data_path "$PROMPTS" \
                --output_folder "$output" \
                --num_output_frames "$FRAMES" \
                --seed "$SEED" --num_samples 1 --use_ema --save_with_index \
                --start_idx "$start" --end_idx "$end" --reseed_per_prompt
        ) >"$log" 2>&1
    else
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYRAMIDKV_POLICY_TRACE_PATH="$policy_trace"
            cd "$PF" || exit 2
            python inference.py \
                --config_path "$PF_CONFIG" \
                --checkpoint_path "$PF_CHECKPOINT" \
                --data_path "$PROMPTS" \
                --output_folder "$output" \
                --num_output_frames "$FRAMES" \
                --seed "$SEED" --num_samples 1 --use_ema --save_with_index \
                --start_idx "$start" --end_idx "$end" --reseed_per_prompt \
                "${head_args[@]}" "${route_args[@]}" "${transition_args[@]}"
        ) >"$log" 2>&1
    fi
    local status=$?
    [[ "$status" -eq 0 ]] || return "$status"

    if grep -Eqi \
        'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|PyramidKVPolicyTraceError' \
        "$log"; then
        echo "[error] failure signature in $log"
        return 1
    fi
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx "$start" --end-idx "$end" \
        --output-json "$OUT_ROOT/diagnostics/$name.shard$shard.video.json" \
        >"$OUT_ROOT/diagnostics/$name.shard$shard.video.log" 2>&1 || return 1
    if [[ "$engine" == "pf" ]]; then
        [[ -s "$policy_trace" ]] || {
            echo "[error] missing policy trace $policy_trace"
            return 1
        }
        grep -q '\[PyramidKVRuntimePolicy\]' "$log" || {
            echo "[error] missing runtime policy audit in $log"
            return 1
        }
    fi
    if [[ "$route" == history_* ]]; then
        grep -q '\[HistoryPolarityPolicy\].*legacy_pf_labels=false' "$log" || {
            echo "[error] missing neutral history-polarity marker in $log"
            return 1
        }
    fi
    if [[ "$route" == "pf_explicit_parity" ]]; then
        grep -q '\[BinaryPolicyOverride\].*stable=stride.*responsive=cyclic' "$log" || {
            echo "[error] missing PF parity marker in $log"
            return 1
        }
    fi
    if [[ "$transition" == "1" && ! -s "$transition_trace" ]]; then
        echo "[error] missing transition trace $transition_trace"
        return 1
    fi
    printf 'ok\n' >"$marker"
}

CELLS=(
    "sf_native|sf||none|0"
    "pf_native|pf|$PF_LABELS|native|0"
    "pf_explicit_parity|pf|$PF_LABELS|pf_explicit_parity|0"
    "pf_aw_hybrid_merge|pf|$PF_AW|history_hybrid_merge|0"
    "history_polarity_hybrid_merge|pf|$POLARITY_ZERO|history_hybrid_merge|0"
    "history_polarity_stride_merge|pf|$POLARITY_ZERO|history_stride_merge|0"
    "history_polarity_hybrid_merge_v78|pf|$POLARITY_ZERO|history_hybrid_merge|1"
    "positive_rate_half_hybrid_merge|pf|$POSITIVE_HALF|history_hybrid_merge|0"
)
[[ "${#CELLS[@]}" -eq "${#METHODS[@]}" ]] || exit 2
for index in "${!METHODS[@]}"; do
    IFS='|' read -r cell_name _ <<<"${CELLS[$index]}"
    [[ "$cell_name" == "${METHODS[$index]}" ]] || {
        echo "[error] method/cell mismatch at index $index"
        exit 2
    }
done

PIDS=()
STATUS=0
for local_slot in "${!GPUS[@]}"; do
    global_rank=$((NODE_RANK * 8 + local_slot))
    method_index=$((global_rank / 4))
    shard=$((global_rank % 4))
    start=$((shard * SHARD_SIZE))
    end=$((start + SHARD_SIZE))
    IFS='|' read -r name engine labels route transition \
        <<<"${CELLS[$method_index]}"
    echo "[launch] node=$NODE_RANK gpu=${GPUS[$local_slot]} global=$global_rank method=$name shard=$shard interval=[$start,$end)"
    run_cell \
        "$name" "${GPUS[$local_slot]}" "$shard" "$start" "$end" \
        "$engine" "$labels" "$route" "$transition" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

if [[ "$STATUS" -eq 0 ]]; then
    printf 'ok\n' >"$OUT_ROOT/status/node${NODE_RANK}.done"
fi
echo "[v98-generation] mode=$MODE node=$NODE_RANK status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
