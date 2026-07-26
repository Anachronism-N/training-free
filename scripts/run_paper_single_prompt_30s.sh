#!/usr/bin/env bash
# Canonical 3-prompt x 30-second single-prompt generation matrix.
set -uo pipefail

GPU_NATIVE="${1:-0}"
GPU_PF="${2:-1}"
GPU_ECHO="${3:-2}"
GPU_OURS_ALL="${4:-3}"
GPU_OURS_ROLE="${5:-4}"

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="$ROOT/third_party/Self-Forcing"
PF="$ROOT/third_party/Pyramid-Forcing"
ECHO="$ROOT/third_party/Echo-Forcing"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_HEAD_LABELS="${PF_HEAD_LABELS:-$PF/configs/head_configs/best_labels.csv}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$ROOT/prompts/hrem_v2_single_long_complex_3.txt}"
ECHO_PROMPTS="${ECHO_PROMPTS:-$ROOT/prompts/paper_single_long_echo_3.txt}"
SEED="${SEED:-0}"
FRAMES="${FRAMES:-120}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/paper_single_30s_s${SEED}}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-3}"
FORCE="${FORCE:-0}"
PARALLEL="${PARALLEL:-1}"
PREPARE_REVIEW="${PREPARE_REVIEW:-1}"

source "$CONDA_SH" || { echo "[error] failed to source $CONDA_SH"; exit 2; }
conda activate "$CONDA_ENV" || { echo "[error] failed to activate $CONDA_ENV"; exit 2; }
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

for path in "$SF" "$PF" "$ECHO"; do
    [[ -d "$path" ]] || { echo "[error] missing repository: $path"; exit 2; }
done
for path in "$SF_CONFIG" "$PF_CONFIG" "$ECHO_CONFIG" \
            "$PF_HEAD_LABELS" "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$ECHO_CHECKPOINT" "$PROMPTS" \
            "$ECHO_PROMPTS"; do
    [[ -f "$path" ]] || { echo "[error] missing file: $path"; exit 2; }
done
for path in "$SF/wan_models/Wan2.1-T2V-1.3B" \
            "$PF/wan_models/Wan2.1-T2V-1.3B" \
            "$ECHO/wan_models/Wan2.1-T2V-1.3B"; do
    [[ -d "$path" ]] || { echo "[error] missing Wan model: $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq "$EXPECTED_VIDEOS" ]] || {
    echo "[error] expected $EXPECTED_VIDEOS prompts, found $PROMPT_COUNT in $PROMPTS"
    exit 2
}
python "$ROOT/scripts/validate_echo_prompts.py" "$ECHO_PROMPTS" \
    --expected-lines "$EXPECTED_VIDEOS" --plain-single --reference-sf "$PROMPTS" || exit 2
python -c "import torch; from lifecycle_kv.attention_fusion import query_conditioned_memory_readout; print('[preflight] torch', torch.__version__)" || exit 2

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if command -v sha256sum >/dev/null 2>&1; then
    PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
    ECHO_PROMPT_SHA256="$(sha256sum "$ECHO_PROMPTS" | awk '{print $1}')"
    SF_CONFIG_FINGERPRINT="$(sha256sum "$SF_CONFIG" | awk '{print $1}')"
    PF_CONFIG_FINGERPRINT="$(sha256sum "$PF_CONFIG" | awk '{print $1}'):$(sha256sum "$PF_HEAD_LABELS" | awk '{print $1}')"
    ECHO_CONFIG_FINGERPRINT="$(sha256sum "$ECHO_CONFIG" | awk '{print $1}')"
else
    PROMPT_SHA256="unavailable"
    ECHO_PROMPT_SHA256="unavailable"
    SF_CONFIG_FINGERPRINT="unavailable"
    PF_CONFIG_FINGERPRINT="unavailable"
    ECHO_CONFIG_FINGERPRINT="unavailable"
fi
export HREM_RUN_COMMIT="$RUN_COMMIT"
export HREM_RUN_SEED="$SEED"
export HREM_RUN_FRAMES="$FRAMES"
export HREM_PROMPT_SHA256="$PROMPT_SHA256"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
echo "[config] task=single_prompt commit=$RUN_COMMIT seed=$SEED frames=$FRAMES"
echo "[config] prompts=$PROMPTS sha256=$PROMPT_SHA256"
echo "[config] echo_prompts=$ECHO_PROMPTS sha256=$ECHO_PROMPT_SHA256 output=$OUT_ROOT"
echo "[protocol] generation only; complete blind review before running metrics"

COMMON_OURS=(
    STRUCTURED_MEMORY_ENABLE=1
    STRUCTURED_MEMORY_GATE=0.05
    STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES=36
    STRUCTURED_MEMORY_ARCHIVE_POLICY=coverage
    STRUCTURED_MEMORY_SPATIAL_STRIDE=4
    STRUCTURED_MEMORY_TOP_K_FRAMES=3
    STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES=12
    STRUCTURED_MEMORY_SELECTION_POLICY=query
    STRUCTURED_MEMORY_SELECTION_SCOPE=shared
    STRUCTURED_MEMORY_PROMPT_PRIOR_WEIGHT=0.0
    STRUCTURED_MEMORY_READOUT_MODE=noisy_only
    STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE=0.20
    STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD=0.15
    STRUCTURED_MEMORY_VALUE_MODE=full
    STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN=0.0
    STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY=1.0
    STRUCTURED_MEMORY_CONTROL_MODE=normal
    STRUCTURED_MEMORY_POSITION_MODE=none
    STRUCTURED_MEMORY_FUSION_MODE=convex
    STRUCTURED_MEMORY_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_EPISODE_WARMUP_BLOCKS=0
    STRUCTURED_MEMORY_LAYER_START=15
    STRUCTURED_MEMORY_LAYER_END=21
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=0
    STRUCTURED_MEMORY_MEMORY_START_FRAME=36
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=intra_episode
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=0
    STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE=off
    STRUCTURED_MEMORY_TRACE_ENABLED=1
    STRUCTURED_MEMORY_DEBUG=1
    STRUCTURED_MEMORY_DEBUG_LAYERS=15,18,20
    STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
    SCENE_TRANSITION_RESET=0
)

video_count() {
    local output="$1"
    if [[ ! -d "$output" ]]; then
        printf '0'
        return
    fi
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

write_metadata() {
    local name="$1" prompt_sha="$2" config_fingerprint="$3"
    printf '%s\n' \
        "task=single_prompt" \
        "method=$name" \
        "run_commit=$RUN_COMMIT" \
        "prompt_sha256=$prompt_sha" \
        "config_fingerprint=$config_fingerprint" \
        "seed=$SEED" \
        "frames=$FRAMES" \
        >"$OUT_ROOT/$name/run_metadata.txt"
}

should_skip() {
    local name="$1" trace_required="$2" prompt_sha="$3"
    local config_fingerprint="$4" require_commit="$5"
    local output="$OUT_ROOT/$name"
    local metadata="$output/run_metadata.txt"
    local count
    count="$(video_count "$output")"
    if [[ "$FORCE" == "1" || "$count" -lt "$EXPECTED_VIDEOS" ]]; then
        return 1
    fi
    if [[ ! -s "$metadata" ]] \
        || ! grep -Fxq "prompt_sha256=$prompt_sha" "$metadata" \
        || ! grep -Fxq "config_fingerprint=$config_fingerprint" "$metadata" \
        || ! grep -Fxq "seed=$SEED" "$metadata" \
        || ! grep -Fxq "frames=$FRAMES" "$metadata"; then
        echo "[rerun] $name outputs do not match the current run metadata"
        return 1
    fi
    if [[ "$require_commit" == "1" ]] \
        && ! grep -Fxq "run_commit=$RUN_COMMIT" "$metadata"; then
        echo "[rerun] $name was produced by a different method commit"
        return 1
    fi
    if [[ "$trace_required" == "1" && ! -s "$OUT_ROOT/traces/$name.jsonl" ]]; then
        echo "[rerun] $name has videos but no trace"
        return 1
    fi
    echo "[skip] $name already has $count videos"
    return 0
}

run_sf_native() {
    local name="sf_native" output="$OUT_ROOT/sf_native" log="$OUT_ROOT/logs/sf_native.log"
    should_skip "$name" 0 "$PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT" 0 && return 0
    mkdir -p "$output"
    echo "[run] $name gpu=$GPU_NATIVE"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$GPU_NATIVE"
        unset SF_FULL_ATTN_MAX_FRAMES AR_LATENT_TRACE_PATH
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0 STRUCTURED_MEMORY_TRACE_ENABLED=0
        export SCENE_TRANSITION_RESET=0
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    [[ "$rc" -eq 0 ]] && write_metadata "$name" "$PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT"
    return "$rc"
}

run_pf() {
    local name="sf_pyramid_forcing" output="$OUT_ROOT/sf_pyramid_forcing" log="$OUT_ROOT/logs/sf_pyramid_forcing.log"
    should_skip "$name" 0 "$PROMPT_SHA256" "$PF_CONFIG_FINGERPRINT" 0 && return 0
    mkdir -p "$output"
    echo "[run] $name gpu=$GPU_PF official_config=$PF_CONFIG"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$GPU_PF"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0 PYRAMIDKV_LAYOUT_DEBUG=0
        python inference.py \
            --config_path "$PF_CONFIG" --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    [[ "$rc" -eq 0 ]] && write_metadata "$name" "$PROMPT_SHA256" "$PF_CONFIG_FINGERPRINT"
    return "$rc"
}

run_echo() {
    local name="sf_echo_forcing" output="$OUT_ROOT/sf_echo_forcing" log="$OUT_ROOT/logs/sf_echo_forcing.log"
    should_skip "$name" 0 "$ECHO_PROMPT_SHA256" "$ECHO_CONFIG_FINGERPRINT" 0 && return 0
    mkdir -p "$output"
    echo "[run] $name gpu=$GPU_ECHO official_long_video_path"
    (
        cd "$ECHO"
        export CUDA_VISIBLE_DEVICES="$GPU_ECHO"
        export ECHO_VERBOSE=1
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export HEAD_ROLE_STATISTICAL=0 STRUCTURED_MEMORY_ENABLE=0
        python inference.py \
            --config_path "$ECHO_CONFIG" --checkpoint_path "$ECHO_CHECKPOINT" \
            --data_path "$ECHO_PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    [[ "$rc" -eq 0 ]] && write_metadata "$name" "$ECHO_PROMPT_SHA256" "$ECHO_CONFIG_FINGERPRINT"
    return "$rc"
}

run_ours() {
    local name="$1" gpu="$2" routing="$3"
    shift 3
    local output="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
    local trace="$OUT_ROOT/traces/$name.jsonl"
    should_skip "$name" 1 "$PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT" 1 && return 0
    mkdir -p "$output"
    rm -f "$trace" "$OUT_ROOT/traces/${name}_diagnosis.json"
    echo "[run] $name gpu=$gpu routing=$routing trace=$trace"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        unset SF_FULL_ATTN_MAX_FRAMES AR_LATENT_TRACE_PATH
        export HREM_RUN_CELL="$name"
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        for assignment in "${COMMON_OURS[@]}"; do export "$assignment"; done
        export STRUCTURED_MEMORY_HEAD_ROUTING="$routing"
        export STRUCTURED_MEMORY_TRACE_PATH="$trace"
        for assignment in "$@"; do export "$assignment"; done
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    [[ "$rc" -eq 0 ]] && write_metadata "$name" "$PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT"
    return "$rc"
}

status=0
if [[ "$PARALLEL" == "1" ]]; then
    run_sf_native & p0=$!
    run_pf & p1=$!
    run_echo & p2=$!
    run_ours ours_all_heads "$GPU_OURS_ALL" off & p3=$!
    run_ours ours_role "$GPU_OURS_ROLE" role_evidence \
        STRUCTURED_MEMORY_ROLE_CALIBRATION=hybrid \
        STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45 \
        STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.50 \
        STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.01 \
        STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 & p4=$!
    for pid in "$p0" "$p1" "$p2" "$p3" "$p4"; do
        wait "$pid" || status=1
    done
else
    run_sf_native || status=1
    run_pf || status=1
    run_echo || status=1
    run_ours ours_all_heads "$GPU_OURS_ALL" off || status=1
    run_ours ours_role "$GPU_OURS_ROLE" role_evidence \
        STRUCTURED_MEMORY_ROLE_CALIBRATION=hybrid \
        STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45 \
        STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.50 \
        STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.01 \
        STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 || status=1
fi

for name in ours_all_heads ours_role; do
    trace="$OUT_ROOT/traces/$name.jsonl"
    diagnosis="$OUT_ROOT/traces/${name}_diagnosis.json"
    if [[ ! -s "$trace" ]]; then
        echo "[error] missing trace: $trace"
        status=1
        continue
    fi
    python "$ROOT/scripts/analyze_hrem_v2_debug.py" "$trace" \
        --strict --json-output "$diagnosis" || status=1
done

if [[ "$PREPARE_REVIEW" == "1" && "$status" -eq 0 ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$OUT_ROOT" \
        --methods sf_native sf_pyramid_forcing sf_echo_forcing ours_all_heads ours_role \
        --prompts "$PROMPTS" --output "$OUT_ROOT/blind_review" \
        --prompt-count "$EXPECTED_VIDEOS" --seed 20260722 --force || status=1
fi

echo "[done] outputs=$OUT_ROOT status=$status"
echo "[next] freeze $OUT_ROOT/blind_review/scorecard.csv before revealing $OUT_ROOT/blind_review_private/key_private.json"
echo "[logs] inspect $OUT_ROOT/traces/*_diagnosis.json and $OUT_ROOT/logs/*.log"
exit "$status"
