#!/usr/bin/env bash
# LifeCache-v3 16-GPU generation matrix.
# Modes: baselines, screen, profile, refine, confirm, hybrid.
set -uo pipefail

MODE="${1:-screen}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="$ROOT/third_party/Self-Forcing"
PF="$ROOT/third_party/Pyramid-Forcing"
ECHO="$ROOT/third_party/Echo-Forcing"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-$ROOT/prompts/lifecache_v3_calibration_complex_12.txt}"
EVALUATION_PROMPTS="${EVALUATION_PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
if [[ -z "${PROMPTS:-}" ]]; then
    case "$MODE" in
        screen|profile|refine) PROMPTS="$CALIBRATION_PROMPTS" ;;
        *) PROMPTS="$EVALUATION_PROMPTS" ;;
    esac
fi
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
PROFILE_PATH="${PROFILE_PATH:-$ROOT/configs/lifecache_v3_intervention_profile.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v69_${MODE}_12p_30s}"
FORCE="${FORCE:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
REFINE_LAYER_START="${REFINE_LAYER_START:-15}"
REFINE_LAYER_END="${REFINE_LAYER_END:-22}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] GPU_LIST must contain exactly 16 device ids"
    exit 2
}
source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

for path in "$SF_CONFIG" "$SF_CHECKPOINT" "$PROMPTS"; do
    [[ -f "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -ge 12 ]] || {
    echo "[error] expected at least 12 prompts, found $PROMPT_COUNT"
    exit 2
}
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
{
    printf 'MODE=%s\n' "$MODE"
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
} >"$OUT_ROOT/run_manifest.env"
echo "[matrix] mode=$MODE commit=$RUN_COMMIT prompts=$PROMPT_COUNT prompt_sha256=$PROMPT_SHA256 frames=$FRAMES out=$OUT_ROOT"

BATCH_STATUS=0
PIDS=()
wait_batch() {
    local pid
    for pid in "${PIDS[@]}"; do
        wait "$pid" || BATCH_STATUS=1
    done
    PIDS=()
}

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

run_native() {
    local name="$1" gpu="$2" seed="$3"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        cd "$SF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0 STRUCTURED_MEMORY_TRACE_ENABLED=0
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_external() {
    local method="$1" name="$2" gpu="$3" seed="$4"
    local repo config checkpoint
    case "$method" in
        pf) repo="$PF"; config="$PF_CONFIG"; checkpoint="$PF_CHECKPOINT" ;;
        echo) repo="$ECHO"; config="$ECHO_CONFIG"; checkpoint="$ECHO_CHECKPOINT" ;;
        *) echo "[error] unsupported external method $method"; return 2 ;;
    esac
    for path in "$repo" "$config" "$checkpoint"; do
        [[ -e "$path" ]] || { echo "[error] missing $path"; return 2; }
    done
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    (
        cd "$repo" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0
        [[ "$method" != "echo" ]] || export ECHO_VERBOSE=1
        python inference.py \
            --config_path "$config" --checkpoint_path "$checkpoint" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

run_cell() {
    local name="$1" gpu="$2" seed="$3" policy="$4" routing="$5"
    local layer_start="$6" layer_end="$7" anchors="$8" summaries="$9"
    shift 9
    local budget="$1" max_delta="$2" motion_penalty="$3" top_k="$4"
    local head_start="$5" head_end="$6" target_call="$7"
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.jsonl"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$PROMPT_COUNT" && -s "$trace" ]]; then
        echo "[skip] $name"
        return
    fi
    mkdir -p "$output"
    rm -f "$trace"
    local archive_max=36
    if [[ "$policy" == "typed" ]]; then
        archive_max=$((anchors + summaries))
    fi
    (
        cd "$SF" || exit 2
        export CUDA_VISIBLE_DEVICES="$gpu"
        export HREM_RUN_COMMIT="$RUN_COMMIT" HREM_RUN_CELL="$name"
        export HREM_RUN_SEED="$seed" HREM_RUN_FRAMES="$FRAMES"
        export HREM_RUN_PROMPT_PATH="$PROMPTS" HREM_RUN_PROMPT_SHA256="$PROMPT_SHA256"
        export HREM_PROMPT_SHA256="$PROMPT_SHA256"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=1 STRUCTURED_MEMORY_GATE=0.05
        export STRUCTURED_MEMORY_ARCHIVE_POLICY="$policy"
        export STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES="$archive_max"
        export STRUCTURED_MEMORY_SPATIAL_STRIDE=4
        export STRUCTURED_MEMORY_TYPED_ANCHOR_FRAMES="$anchors"
        export STRUCTURED_MEMORY_TYPED_SUMMARY_SLOTS="$summaries"
        export STRUCTURED_MEMORY_TYPED_ANCHOR_MIN_GAP_FRAMES=6
        export STRUCTURED_MEMORY_TYPED_ANCHOR_MOTION_CEILING=0.35
        export STRUCTURED_MEMORY_TYPED_ANCHOR_REPLACE_MARGIN=0.05
        export STRUCTURED_MEMORY_TYPED_SUMMARY_MERGE_SIMILARITY=0.90
        export STRUCTURED_MEMORY_TYPED_SUMMARY_COUNT_CAP=8
        export STRUCTURED_MEMORY_TYPED_ANCHOR_BIAS=0.05
        export STRUCTURED_MEMORY_TYPED_SUMMARY_BIAS=0.0
        export STRUCTURED_MEMORY_TYPED_MOTION_PENALTY="$motion_penalty"
        export STRUCTURED_MEMORY_TOP_K_FRAMES="$top_k"
        export STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES=12
        export STRUCTURED_MEMORY_SELECTION_POLICY=query
        export STRUCTURED_MEMORY_SELECTION_SCOPE=per_head
        export STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE=0.20
        export STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD=0.15
        export STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
        export STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY=1.0
        export STRUCTURED_MEMORY_VALUE_MODE=full
        export STRUCTURED_MEMORY_READOUT_MODE=noisy_only
        export STRUCTURED_MEMORY_POSITION_MODE=none
        export STRUCTURED_MEMORY_FUSION_MODE=convex
        export STRUCTURED_MEMORY_HEAD_ROUTING="$routing"
        export STRUCTURED_MEMORY_INTERVENTION_HEAD_BUDGET_FRACTION="$budget"
        export STRUCTURED_MEMORY_INTERVENTION_EMA_DECAY=0.90
        export STRUCTURED_MEMORY_INTERVENTION_MIN_ALIGNMENT=0.0
        export STRUCTURED_MEMORY_INTERVENTION_MAX_DELTA_TO_NATIVE="$max_delta"
        export STRUCTURED_MEMORY_INTERVENTION_MIN_UTILITY_SPREAD=0.02
        export STRUCTURED_MEMORY_INTERVENTION_MIN_OBSERVATIONS=2
        export STRUCTURED_MEMORY_INTERVENTION_PROFILE_PATH=""
        if [[ "$routing" == "intervention_offline" || "$routing" == "intervention_hybrid" ]]; then
            export STRUCTURED_MEMORY_INTERVENTION_PROFILE_PATH="$PROFILE_PATH"
        fi
        export STRUCTURED_MEMORY_PROFILE_HEAD_START="$head_start"
        export STRUCTURED_MEMORY_PROFILE_HEAD_END="$head_end"
        export STRUCTURED_MEMORY_PROFILE_ATTENTION_CALL_INDEX="$target_call"
        export STRUCTURED_MEMORY_LAYER_START="$layer_start"
        export STRUCTURED_MEMORY_LAYER_END="$layer_end"
        export STRUCTURED_MEMORY_MEMORY_START_FRAME=36
        export STRUCTURED_MEMORY_EPISODE_GATE_MODE=intra_episode
        export STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=0
        export STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE=off
        export STRUCTURED_MEMORY_TRACE_ENABLED=1
        export STRUCTURED_MEMORY_TRACE_PATH="$trace"
        export STRUCTURED_MEMORY_DEBUG=1
        export STRUCTURED_MEMORY_DEBUG_LAYERS="$layer_start,$((layer_end - 1))"
        export STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
        env | grep -E '^(STRUCTURED_MEMORY|HREM_RUN|HREM_PROMPT)_' | sort >"$output/run_config.env"
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
}

launch_screen() {
    run_native sf_native "${GPUS[0]}" "$SEED" & PIDS+=("$!")
    run_cell coverage_all "${GPUS[1]}" "$SEED" coverage off 15 22 4 12 1.0 1.0 0.0 4 0 12 -1 & PIDS+=("$!")
    run_cell typed_all "${GPUS[2]}" "$SEED" typed off 15 22 4 12 1.0 1.0 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell anchor_only "${GPUS[3]}" "$SEED" typed off 15 22 8 0 1.0 1.0 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell summary_only "${GPUS[4]}" "$SEED" typed off 15 22 0 16 1.0 1.0 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_b25 "${GPUS[5]}" "$SEED" typed intervention_online 15 22 4 12 0.25 0.08 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_b50 "${GPUS[6]}" "$SEED" typed intervention_online 15 22 4 12 0.50 0.08 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_b75 "${GPUS[7]}" "$SEED" typed intervention_online 15 22 4 12 0.75 0.08 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_no_motion_penalty "${GPUS[8]}" "$SEED" typed intervention_online 15 22 4 12 0.50 0.08 0.0 4 0 12 -1 & PIDS+=("$!")
    run_cell online_strict_delta "${GPUS[9]}" "$SEED" typed intervention_online 15 22 4 12 0.50 0.03 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_loose_delta "${GPUS[10]}" "$SEED" typed intervention_online 15 22 4 12 0.50 0.15 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_l0_10 "${GPUS[11]}" "$SEED" typed intervention_online 0 10 4 12 0.50 0.08 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_l10_20 "${GPUS[12]}" "$SEED" typed intervention_online 10 20 4 12 0.50 0.08 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_l20_30 "${GPUS[13]}" "$SEED" typed intervention_online 20 30 4 12 0.50 0.08 0.10 4 0 12 -1 & PIDS+=("$!")
    run_cell online_top2 "${GPUS[14]}" "$SEED" typed intervention_online 15 22 4 12 0.50 0.08 0.10 2 0 12 -1 & PIDS+=("$!")
    run_cell online_top6 "${GPUS[15]}" "$SEED" typed intervention_online 15 22 4 12 0.50 0.08 0.10 6 0 12 -1 & PIDS+=("$!")
}

launch_baselines() {
    local index=0
    for seed in 0 1 2 3; do
        run_native "sf_native_s${seed}" "${GPUS[$index]}" "$seed" & PIDS+=("$!"); index=$((index + 1))
        run_external pf "sf_pf_s${seed}" "${GPUS[$index]}" "$seed" & PIDS+=("$!"); index=$((index + 1))
        run_external echo "sf_echo_s${seed}" "${GPUS[$index]}" "$seed" & PIDS+=("$!"); index=$((index + 1))
        run_cell "coverage_all_s${seed}" "${GPUS[$index]}" "$seed" coverage off 15 22 4 12 1.0 1.0 0.0 4 0 12 -1 & PIDS+=("$!"); index=$((index + 1))
    done
}

launch_profile() {
    local index=0 layer_start layer_end head_start head_end
    local layer_starts=(0 8 15 22)
    local layer_ends=(8 15 22 30)
    local head_starts=(0 3 6 9)
    local head_ends=(3 6 9 12)
    for layer_group in 0 1 2 3; do
        layer_start="${layer_starts[$layer_group]}"
        layer_end="${layer_ends[$layer_group]}"
        for head_group in 0 1 2 3; do
            head_start="${head_starts[$head_group]}"
            head_end="${head_ends[$head_group]}"
            run_cell "profile_l${layer_start}_${layer_end}_h${head_start}_${head_end}" \
                "${GPUS[$index]}" "$SEED" typed profile_group \
                "$layer_start" "$layer_end" 4 12 1.0 1.0 0.10 4 \
                "$head_start" "$head_end" -1 &
            PIDS+=("$!")
            index=$((index + 1))
        done
    done
}

launch_refine() {
    [[ "$REFINE_LAYER_START" =~ ^[0-9]+$ && "$REFINE_LAYER_END" =~ ^[0-9]+$ ]] || {
        echo "[error] REFINE_LAYER_START/END must be integers"
        exit 2
    }
    (( REFINE_LAYER_START >= 0 && REFINE_LAYER_START < REFINE_LAYER_END && REFINE_LAYER_END <= 30 )) || {
        echo "[error] refine layer range must be non-empty and within [0,30)"
        exit 2
    }
    local index=0 layer head gpu_index
    for ((layer=REFINE_LAYER_START; layer<REFINE_LAYER_END; layer++)); do
        for ((head=0; head<12; head++)); do
            gpu_index=$((index % 16))
            run_cell "refine_l${layer}_h${head}" "${GPUS[$gpu_index]}" "$SEED" \
                typed profile_group "$layer" "$((layer + 1))" 4 12 \
                1.0 1.0 0.10 4 "$head" "$((head + 1))" -1 &
            PIDS+=("$!")
            index=$((index + 1))
            if (( index % 16 == 0 )); then
                wait_batch
            fi
        done
    done
    wait_batch
}

launch_confirm() {
    local index=0
    for seed in 0 1 2 3; do
        run_native "sf_native_s${seed}" "${GPUS[$index]}" "$seed" & PIDS+=("$!"); index=$((index + 1))
        run_cell "typed_all_s${seed}" "${GPUS[$index]}" "$seed" typed off 15 22 4 12 1.0 1.0 0.10 4 0 12 -1 & PIDS+=("$!"); index=$((index + 1))
        run_cell "online_b25_s${seed}" "${GPUS[$index]}" "$seed" typed intervention_online 15 22 4 12 0.25 0.08 0.10 4 0 12 -1 & PIDS+=("$!"); index=$((index + 1))
        run_cell "online_b50_s${seed}" "${GPUS[$index]}" "$seed" typed intervention_online 15 22 4 12 0.50 0.08 0.10 4 0 12 -1 & PIDS+=("$!"); index=$((index + 1))
    done
}

launch_hybrid() {
    [[ -s "$PROFILE_PATH" ]] || { echo "[error] missing profile $PROFILE_PATH"; exit 2; }
    local index=0
    for seed in 0 1 2 3; do
        for routing in intervention_offline intervention_hybrid; do
            for budget in 0.25 0.50; do
                run_cell "${routing}_${budget}_s${seed}" "${GPUS[$index]}" "$seed" \
                    typed "$routing" 15 22 4 12 "$budget" 0.08 0.10 4 0 12 -1 &
                PIDS+=("$!")
                index=$((index + 1))
            done
        done
    done
}

case "$MODE" in
    baselines) launch_baselines ;;
    screen) launch_screen ;;
    profile) launch_profile ;;
    refine) launch_refine ;;
    confirm) launch_confirm ;;
    hybrid) launch_hybrid ;;
    *) echo "[error] mode must be baselines, screen, profile, refine, confirm, or hybrid"; exit 2 ;;
esac

wait_batch
status="$BATCH_STATUS"
for trace in "$OUT_ROOT"/traces/*.jsonl; do
    [[ -s "$trace" ]] || continue
    diagnosis="${trace%.jsonl}_diagnosis.json"
    python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$trace" \
        --json-output "$diagnosis" || status=1
done
echo "[done] mode=$MODE status=$status outputs=$OUT_ROOT"
exit "$status"
