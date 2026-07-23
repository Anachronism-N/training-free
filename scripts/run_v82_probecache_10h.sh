#!/usr/bin/env bash
# Deadline-aware ProbeCache follow-up queue for one 16-H20 node.
# Usage:
#   DEADLINE_EPOCH="$(date -d '+10 hours' +%s)" \
#     bash scripts/run_v82_probecache_10h.sh all
# Phases: profile-replica, labels, confirm, ultralong, switch, prepare, all.
set -euo pipefail

PHASE="${1:-all}"
ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PRIMARY_PROFILE_ROOT="${PRIMARY_PROFILE_ROOT:-$ROOT/runs/v81_probecache_profile}"
PRIMARY_LABEL="${PRIMARY_LABEL:-$PRIMARY_PROFILE_ROOT/labels/probecache_binary_labels.csv}"
PRIMARY_REPORT="${PRIMARY_REPORT:-$PRIMARY_PROFILE_ROOT/labels/probecache_profile_report.json}"
REPLICA_PROFILE_ROOT="${REPLICA_PROFILE_ROOT:-$ROOT/runs/v82_probecache_profile_replica}"
REPLICA_LABEL="$REPLICA_PROFILE_ROOT/labels/probecache_binary_labels.csv"
REPLICA_REPORT="$REPLICA_PROFILE_ROOT/labels/probecache_profile_report.json"
CONTROL_DIR="${CONTROL_DIR:-$ROOT/runs/v82_probecache_control_labels}"
PF_LABEL="${PF_LABEL:-$PF/configs/head_configs/best_labels.csv}"
DIAGNOSTIC_PROMPTS="${DIAGNOSTIC_PROMPTS:-$ROOT/prompts/probecache_v82_diagnostic_complex_3.txt}"
CONFIRM_PROMPTS="${CONFIRM_PROMPTS:-$ROOT/prompts/lifecache_v3_single_long_complex_12.txt}"
ULTRALONG_PROMPTS="${ULTRALONG_PROMPTS:-$ROOT/prompts/probecache_v82_ultralong_complex_6.txt}"
SWITCH_PROMPTS="${SWITCH_PROMPTS:-$ROOT/prompts/hrem_v2_aba_complex_3.txt}"
LABEL_ROOT="${LABEL_ROOT:-$ROOT/runs/v82_probecache_labels}"
CONFIRM_ROOT="${CONFIRM_ROOT:-$ROOT/runs/v82_probecache_confirm}"
ULTRALONG_ROOT="${ULTRALONG_ROOT:-$ROOT/runs/v82_probecache_ultralong}"
SWITCH_ROOT="${SWITCH_ROOT:-$ROOT/runs/v82_probecache_switch}"
V81_SINGLE_ROOT="${V81_SINGLE_ROOT:-$ROOT/runs/v81_probecache_single}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(( $(date +%s) + 10 * 3600 ))}"
SAFETY_MINUTES="${SAFETY_MINUTES:-20}"
FORCE="${FORCE:-0}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-0}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-2048}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 16 ]] || {
    echo "[error] v82 requires 16 GPU ids"
    exit 2
}
case "$PHASE" in
    profile-replica|labels|confirm|ultralong|switch|prepare|all) ;;
    *) echo "[error] unknown phase: $PHASE"; exit 2 ;;
esac

export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0

remaining_minutes() {
    local now
    now="$(date +%s)"
    printf '%s' "$(( (DEADLINE_EPOCH - now) / 60 ))"
}

has_budget() {
    local required="$1" remaining
    remaining="$(remaining_minutes)"
    if (( remaining < required + SAFETY_MINUTES )); then
        echo "[budget] skip required=${required}m remaining=${remaining}m safety=${SAFETY_MINUTES}m"
        return 1
    fi
    echo "[budget] start required=${required}m remaining=${remaining}m"
}

wait_for_idle_gpus() {
    [[ "$WAIT_FOR_IDLE" == "1" ]] || return 0
    command -v nvidia-smi >/dev/null || {
        echo "[error] WAIT_FOR_IDLE=1 but nvidia-smi is unavailable"
        exit 2
    }
    while true; do
        local busy=0
        for gpu in "${GPUS[@]:0:16}"; do
            local used
            used="$(
                nvidia-smi -i "$gpu" --query-gpu=memory.used \
                    --format=csv,noheader,nounits | tr -d ' '
            )"
            if (( used > IDLE_MEMORY_MIB )); then
                busy=$((busy + 1))
            fi
        done
        if (( busy == 0 )); then
            echo "[idle] all requested GPUs are below ${IDLE_MEMORY_MIB} MiB"
            return 0
        fi
        if (( $(remaining_minutes) <= SAFETY_MINUTES )); then
            echo "[error] deadline reached while waiting for GPUs"
            exit 1
        fi
        echo "[idle] waiting: busy_gpus=$busy remaining=$(remaining_minutes)m"
        sleep 60
    done
}

prompt_count() {
    grep -cve '^[[:space:]]*$' "$1"
}

video_count() {
    local path="$1"
    [[ -d "$path" ]] || { printf '0'; return; }
    find "$path" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

write_cell_config() {
    local run_root="$1" name="$2" method="$3" prompts="$4"
    local frames="$5" seed="$6" trace_required="$7" head_csv="${8:-}"
    mkdir -p "$run_root/configs"
    {
        printf 'name=%s\n' "$name"
        printf 'method=%s\n' "$method"
        printf 'prompt_file=%s\n' "$prompts"
        printf 'expected_videos=%s\n' "$(prompt_count "$prompts")"
        printf 'frames=%s\n' "$frames"
        printf 'seed=%s\n' "$seed"
        printf 'trace_required=%s\n' "$trace_required"
        printf 'head_csv=%s\n' "$head_csv"
    } >"$run_root/configs/$name.env"
}

verify_cell() {
    local run_root="$1" name="$2" expected="$3" trace_required="$4"
    local count trace
    count="$(video_count "$run_root/$name")"
    if (( count < expected )); then
        echo "[error] $run_root/$name has $count/$expected videos"
        return 1
    fi
    if [[ "$trace_required" == "1" ]]; then
        trace="$run_root/traces/$name.probecache.jsonl"
        [[ -s "$trace" ]] || {
            echo "[error] missing ProbeCache trace: $trace"
            return 1
        }
    fi
}

run_pf() {
    local run_root="$1" name="$2" gpu="$3" prompts="$4" frames="$5" seed="$6"
    local output="$run_root/$name" log="$run_root/logs/$name.log"
    local expected
    expected="$(prompt_count "$prompts")"
    write_cell_config "$run_root" "$name" pf "$prompts" "$frames" "$seed" 0
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$expected" ]]; then
        echo "[skip] $name"
        return 0
    fi
    mkdir -p "$output"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$prompts" --output_folder "$output" \
            --num_output_frames "$frames" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    verify_cell "$run_root" "$name" "$expected" 0
}

run_sf() {
    local run_root="$1" name="$2" gpu="$3" prompts="$4" frames="$5" seed="$6"
    local output="$run_root/$name" log="$run_root/logs/$name.log"
    local expected
    expected="$(prompt_count "$prompts")"
    write_cell_config "$run_root" "$name" sf "$prompts" "$frames" "$seed" 0
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$expected" ]]; then
        echo "[skip] $name"
        return 0
    fi
    mkdir -p "$output"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$prompts" --output_folder "$output" \
            --num_output_frames "$frames" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    verify_cell "$run_root" "$name" "$expected" 0
}

run_v78() {
    local run_root="$1" name="$2" gpu="$3" prompts="$4" frames="$5" seed="$6"
    local output="$run_root/$name" log="$run_root/logs/$name.log"
    local trace="$run_root/traces/$name.transition.jsonl" expected
    expected="$(prompt_count "$prompts")"
    write_cell_config "$run_root" "$name" v78 "$prompts" "$frames" "$seed" 0
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$expected" && -s "$trace" ]]; then
        echo "[skip] $name"
        return 0
    fi
    mkdir -p "$output"
    rm -f "$trace"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$prompts" --output_folder "$output" \
            --num_output_frames "$frames" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index \
            --pyramidkv_cache_transition \
            --pyramidkv_cache_transition_mode full \
            --pyramidkv_cache_transition_min_reliability .55 \
            --pyramidkv_cache_transition_min_novelty .01 \
            --pyramidkv_cache_transition_max_commit_fraction .75 \
            --pyramidkv_cache_transition_stagger_period 1 \
            --pyramidkv_cache_transition_max_age_blocks 6 \
            --pyramidkv_cache_transition_branches both \
            --pyramidkv_cache_transition_denoise_weight 2 \
            --pyramidkv_cache_transition_trace_path "$trace"
    ) >"$log" 2>&1
    verify_cell "$run_root" "$name" "$expected" 0
    [[ -s "$trace" ]] || {
        echo "[error] missing v78 transition trace: $trace"
        return 1
    }
}

# run_ours ROOT NAME GPU PROMPTS FRAMES SEED HEAD_CSV MODE LAYER_START LAYER_END [extra...]
run_ours() {
    local run_root="$1" name="$2" gpu="$3" prompts="$4" frames="$5" seed="$6"
    local head_csv="$7" mode="$8" layer_start="$9" layer_end="${10}"
    shift 10
    local output="$run_root/$name" log="$run_root/logs/$name.log"
    local trace="$run_root/traces/$name.probecache.jsonl"
    local transition_trace="$run_root/traces/$name.transition.jsonl" expected
    expected="$(prompt_count "$prompts")"
    write_cell_config \
        "$run_root" "$name" "probecache_$mode" "$prompts" "$frames" "$seed" 1 "$head_csv"
    if [[ "$FORCE" != "1" && "$(video_count "$output")" -ge "$expected" && -s "$trace" ]]; then
        echo "[skip] $name"
        return 0
    fi
    mkdir -p "$output"
    rm -f "$trace" "$transition_trace"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$prompts" --output_folder "$output" \
            --num_output_frames "$frames" --seed "$seed" --num_samples 1 \
            --use_ema --save_with_index \
            --pyramidkv_head_config_path "$head_csv" \
            --pyramidkv_cache_transition \
            --pyramidkv_cache_transition_mode full \
            --pyramidkv_cache_transition_min_reliability .55 \
            --pyramidkv_cache_transition_min_novelty .01 \
            --pyramidkv_cache_transition_max_commit_fraction .75 \
            --pyramidkv_cache_transition_stagger_period 1 \
            --pyramidkv_cache_transition_max_age_blocks 6 \
            --pyramidkv_cache_transition_branches both \
            --pyramidkv_cache_transition_denoise_weight 2 \
            --pyramidkv_cache_transition_trace_path "$transition_trace" \
            --pyramidkv_probecache --pyramidkv_probecache_mode "$mode" \
            --pyramidkv_probecache_layer_start "$layer_start" \
            --pyramidkv_probecache_layer_end "$layer_end" \
            --pyramidkv_probecache_trace_path "$trace" \
            --pyramidkv_probecache_trace_selection_stride 4 \
            --pyramidkv_probecache_debug \
            "$@"
    ) >"$log" 2>&1
    verify_cell "$run_root" "$name" "$expected" 1
}

wait_cells() {
    local status=0 pid
    for pid in "$@"; do
        wait "$pid" || status=1
    done
    if (( status != 0 )); then
        echo "[error] one or more cells failed"
        return 1
    fi
}

prepare_control_labels() {
    for path in "$PRIMARY_LABEL" "$PRIMARY_REPORT" "$PF_LABEL"; do
        [[ -s "$path" ]] || { echo "[error] missing $path"; return 1; }
    done
    python "$ROOT/scripts/build_probecache_control_labels.py" \
        --learned-csv "$PRIMARY_LABEL" \
        --profile-report "$PRIMARY_REPORT" \
        --pf-csv "$PF_LABEL" \
        --output-dir "$CONTROL_DIR" \
        --random-seeds 2026,2027,2028
}

replica_is_valid() {
    [[ -s "$REPLICA_LABEL" && -s "$REPLICA_REPORT" ]] || return 1
    python - "$REPLICA_REPORT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("acceptance_gates", {}).get("accepted") else 1)
PY
}

replication_label() {
    if replica_is_valid; then
        printf '%s' "$REPLICA_LABEL"
    else
        printf '%s' "$CONTROL_DIR/random_2028.csv"
    fi
}

run_profile_replica_phase() {
    echo "[phase] profile-replica"
    [[ -s "$PRIMARY_LABEL" && -s "$PRIMARY_REPORT" ]] || {
        echo "[error] primary v81 profile is incomplete"
        return 1
    }
    if ! SEEDS="2 3" OUT_ROOT="$REPLICA_PROFILE_ROOT" \
        bash "$ROOT/scripts/run_v81_probecache_profile_16gpu.sh"; then
        echo "[warning] independent profile failed; controls continue with primary labels"
        return 0
    fi
    python "$ROOT/scripts/compare_probecache_head_profiles.py" \
        --reference-csv "$PRIMARY_LABEL" \
        --candidate-csv "$REPLICA_LABEL" \
        --output-json "$REPLICA_PROFILE_ROOT/labels/profile_replication.json" \
        --output-md "$REPLICA_PROFILE_ROOT/labels/profile_replication.md" \
        --min-agreement .60
}

run_labels_phase() {
    echo "[phase] labels"
    prepare_control_labels
    local replica replica_name
    replica="$(replication_label)"
    replica_name="profile_replica"
    [[ "$replica" == "$REPLICA_LABEL" ]] || replica_name="random_2028_fallback"
    mkdir -p "$LABEL_ROOT"/{logs,traces,configs}
    local pids=()
    run_pf "$LABEL_ROOT" pf "${GPUS[0]}" "$DIAGNOSTIC_PROMPTS" 120 0 & pids+=("$!")
    run_v78 "$LABEL_ROOT" v78 "${GPUS[1]}" "$DIAGNOSTIC_PROMPTS" 120 0 & pids+=("$!")
    run_ours "$LABEL_ROOT" learned "${GPUS[2]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" "$replica_name" "${GPUS[3]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$replica" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" pf_binary "${GPUS[4]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/pf_binary.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" inverse "${GPUS[5]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/inverse.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" random_2026 "${GPUS[6]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/random_2026.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" random_2027 "${GPUS[7]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/random_2027.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" remote_only "${GPUS[8]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/remote_only.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" prompt_only "${GPUS[9]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/prompt_only.csv" full 0 -1 & pids+=("$!")
    run_ours "$LABEL_ROOT" layer_early "${GPUS[10]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" full 0 10 & pids+=("$!")
    run_ours "$LABEL_ROOT" layer_middle "${GPUS[11]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" full 10 20 & pids+=("$!")
    run_ours "$LABEL_ROOT" layer_late "${GPUS[12]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" full 20 30 & pids+=("$!")
    run_ours "$LABEL_ROOT" layer_first_half "${GPUS[13]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" full 0 15 & pids+=("$!")
    run_ours "$LABEL_ROOT" layer_second_half "${GPUS[14]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" full 15 30 & pids+=("$!")
    run_ours "$LABEL_ROOT" learned_audit "${GPUS[15]}" "$DIAGNOSTIC_PROMPTS" 120 0 \
        "$CONTROL_DIR/learned.csv" audit 0 -1 & pids+=("$!")
    wait_cells "${pids[@]}"
    python "$ROOT/scripts/audit_probecache_experiment_runs.py" \
        --run-root "$LABEL_ROOT" --strict
}

run_confirm_phase() {
    echo "[phase] confirm"
    prepare_control_labels
    mkdir -p "$CONFIRM_ROOT"/{logs,traces,configs}
    local pids=() slot=0 seed
    for seed in 1 2 3; do
        run_pf "$CONFIRM_ROOT" "pf_s$seed" "${GPUS[$slot]}" \
            "$CONFIRM_PROMPTS" 120 "$seed" & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 1 2 3; do
        run_v78 "$CONFIRM_ROOT" "v78_s$seed" "${GPUS[$slot]}" \
            "$CONFIRM_PROMPTS" 120 "$seed" & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 1 2 3; do
        run_ours "$CONFIRM_ROOT" "learned_s$seed" "${GPUS[$slot]}" \
            "$CONFIRM_PROMPTS" 120 "$seed" "$CONTROL_DIR/learned.csv" \
            full 0 -1 & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 1 2 3; do
        run_ours "$CONFIRM_ROOT" "pf_binary_s$seed" "${GPUS[$slot]}" \
            "$CONFIRM_PROMPTS" 120 "$seed" "$CONTROL_DIR/pf_binary.csv" \
            full 0 -1 & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 1 2; do
        run_ours "$CONFIRM_ROOT" "learned_open_s$seed" "${GPUS[$slot]}" \
            "$CONFIRM_PROMPTS" 120 "$seed" "$CONTROL_DIR/learned.csv" \
            full 0 -1 \
            --pyramidkv_probecache_min_similarity -1 \
            --pyramidkv_probecache_min_margin 0 \
            --pyramidkv_probecache_max_entropy 1 \
            & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 1 2; do
        run_ours "$CONFIRM_ROOT" "learned_conservative_s$seed" "${GPUS[$slot]}" \
            "$CONFIRM_PROMPTS" 120 "$seed" "$CONTROL_DIR/learned.csv" \
            full 0 -1 \
            --pyramidkv_probecache_min_similarity .20 \
            --pyramidkv_probecache_min_margin .05 \
            --pyramidkv_probecache_max_entropy .80 \
            & pids+=("$!"); slot=$((slot + 1))
    done
    wait_cells "${pids[@]}"
    python "$ROOT/scripts/audit_probecache_experiment_runs.py" \
        --run-root "$CONFIRM_ROOT" --strict
}

run_ultralong_phase() {
    echo "[phase] ultralong"
    prepare_control_labels
    local replica replica_name
    replica="$(replication_label)"
    replica_name="replica"
    [[ "$replica" == "$REPLICA_LABEL" ]] || replica_name="random_2028"
    mkdir -p "$ULTRALONG_ROOT"/{logs,traces,configs}
    local pids=() slot=0 seed
    for seed in 0 1; do
        run_sf "$ULTRALONG_ROOT" "sf_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_pf "$ULTRALONG_ROOT" "pf_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_v78 "$ULTRALONG_ROOT" "v78_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_ours "$ULTRALONG_ROOT" "learned_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" "$CONTROL_DIR/learned.csv" \
            full 0 -1 & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_ours "$ULTRALONG_ROOT" "pf_binary_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" "$CONTROL_DIR/pf_binary.csv" \
            full 0 -1 & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_ours "$ULTRALONG_ROOT" "${replica_name}_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" "$replica" \
            full 0 -1 & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_ours "$ULTRALONG_ROOT" "learned_open_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" "$CONTROL_DIR/learned.csv" \
            full 0 -1 \
            --pyramidkv_probecache_min_similarity -1 \
            --pyramidkv_probecache_min_margin 0 \
            --pyramidkv_probecache_max_entropy 1 \
            & pids+=("$!"); slot=$((slot + 1))
    done
    for seed in 0 1; do
        run_ours "$ULTRALONG_ROOT" "learned_conservative_s$seed" "${GPUS[$slot]}" \
            "$ULTRALONG_PROMPTS" 240 "$seed" "$CONTROL_DIR/learned.csv" \
            full 0 -1 \
            --pyramidkv_probecache_min_similarity .20 \
            --pyramidkv_probecache_min_margin .05 \
            --pyramidkv_probecache_max_entropy .80 \
            & pids+=("$!"); slot=$((slot + 1))
    done
    wait_cells "${pids[@]}"
    python "$ROOT/scripts/audit_probecache_experiment_runs.py" \
        --run-root "$ULTRALONG_ROOT" --strict
}

run_switch_phase() {
    echo "[phase] switch"
    OUT_ROOT="$SWITCH_ROOT" HEAD_CSV="$PRIMARY_LABEL" PROMPTS="$SWITCH_PROMPTS" \
        FRAMES=120 GPU_LIST="$GPU_LIST" FORCE="$FORCE" \
        bash "$ROOT/scripts/run_v81_probecache_16gpu.sh" switch
}

prepare_review_phase() {
    echo "[phase] prepare"
    local label_replica="profile_replica"
    [[ -d "$LABEL_ROOT/$label_replica" ]] || label_replica="random_2028_fallback"
    local ultra_replica="replica_s0"
    [[ -d "$ULTRALONG_ROOT/$ultra_replica" ]] || ultra_replica="random_2028_s0"
    local v81_methods=(
        sf_native pf_official echo_pc ours_persistent ours_reactive
        ours_full ours_no_trust ours_open_gate ours_conservative
    )
    local v81_ready=1 method
    for method in "${v81_methods[@]}"; do
        if [[ "$(video_count "$V81_SINGLE_ROOT/$method")" -lt 12 ]]; then
            v81_ready=0
        fi
    done
    if [[ "$v81_ready" == "1" && ! -f "$V81_SINGLE_ROOT/blind_review/scorecard.csv" ]]; then
        RUN_ROOT="$V81_SINGLE_ROOT" \
            bash "$ROOT/scripts/postprocess_v81_probecache.sh" prepare single
    elif [[ -f "$V81_SINGLE_ROOT/blind_review/scorecard.csv" ]]; then
        echo "[review] preserve existing v81 scorecard"
    else
        echo "[review] v81 single package skipped: core methods are incomplete"
    fi
    if [[ -d "$LABEL_ROOT/learned" && ! -f "$LABEL_ROOT/blind_review/scorecard.csv" ]]; then
        python "$ROOT/scripts/prepare_blind_review.py" \
            --run-root "$LABEL_ROOT" \
            --methods pf v78 learned "$label_replica" pf_binary inverse \
                random_2026 remote_only prompt_only \
            --prompts "$DIAGNOSTIC_PROMPTS" \
            --output "$LABEL_ROOT/blind_review" \
            --prompt-count 3 --seed 20260724 --force
    fi
    if [[ -d "$ULTRALONG_ROOT/learned_s0" && ! -f "$ULTRALONG_ROOT/blind_review/scorecard.csv" ]]; then
        python "$ROOT/scripts/prepare_blind_review.py" \
            --run-root "$ULTRALONG_ROOT" \
            --methods sf_s0 pf_s0 v78_s0 learned_s0 pf_binary_s0 \
                "$ultra_replica" learned_open_s0 learned_conservative_s0 \
            --prompts "$ULTRALONG_PROMPTS" \
            --output "$ULTRALONG_ROOT/blind_review" \
            --prompt-count 6 --seed 20260724 --force
    fi
    if [[ -d "$SWITCH_ROOT/ours_full" && ! -f "$SWITCH_ROOT/blind_review/scorecard.csv" ]]; then
        python "$ROOT/scripts/prepare_blind_review.py" \
            --run-root "$SWITCH_ROOT" \
            --methods sf_native pf_official echo_pc ours_persistent \
                ours_reactive ours_full ours_no_trust ours_open_gate \
                ours_conservative \
            --prompts "$SWITCH_PROMPTS" \
            --output "$SWITCH_ROOT/blind_review" \
            --prompt-count 3 --seed 20260724 --force
    fi
    echo "[review] scorecards are prepared; do not run quality metrics before freezing them"
}

run_requested_phase() {
    local requested="$1"
    case "$requested" in
        profile-replica) run_profile_replica_phase ;;
        labels) run_labels_phase ;;
        confirm) run_confirm_phase ;;
        ultralong) run_ultralong_phase ;;
        switch) run_switch_phase ;;
        prepare) prepare_review_phase ;;
    esac
}

for path in "$ROOT" "$PF" "$SF" "$PF_CONFIG" "$SF_CONFIG" \
    "$PF_CHECKPOINT" "$SF_CHECKPOINT"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
wait_for_idle_gpus

if [[ "$PHASE" != "all" ]]; then
    run_requested_phase "$PHASE"
    echo "[v82] phase=$PHASE complete remaining=$(remaining_minutes)m"
    exit 0
fi

# Conservative estimates include compile/startup variance and trace auditing.
if has_budget 45; then run_profile_replica_phase; fi
if has_budget 50; then run_labels_phase; fi
if has_budget 150; then run_confirm_phase; fi
if has_budget 180; then run_ultralong_phase; fi
if has_budget 70; then run_switch_phase; fi
prepare_review_phase
echo "[v82] queue complete remaining=$(remaining_minutes)m deadline=$DEADLINE_EPOCH"
