#!/usr/bin/env bash
# Canonical 3-prompt x 30-second A-B-A scene-switch generation matrix.
set -uo pipefail

GPU_SF="${1:-0}"
GPU_ECHO="${2:-1}"
GPU_OURS_ALL="${3:-2}"
GPU_OURS_ROLE="${4:-3}"

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
SF="$ROOT/third_party/Self-Forcing"
ECHO="$ROOT/third_party/Echo-Forcing"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
ECHO_CONFIG="${ECHO_CONFIG:-$ECHO/configs/self_forcing_dmd.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
ECHO_CHECKPOINT="${ECHO_CHECKPOINT:-$ECHO/checkpoints/self_forcing_dmd.pt}"
SF_PROMPTS="${SF_PROMPTS:-$ROOT/prompts/paper_scene_switch_sf_3.txt}"
ECHO_PROMPTS="${ECHO_PROMPTS:-$ROOT/prompts/paper_scene_switch_echo_3.txt}"
SEED="${SEED:-0}"
FRAMES="${FRAMES:-120}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/paper_scene_30s_s${SEED}}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-3}"
FORCE="${FORCE:-0}"
PARALLEL="${PARALLEL:-1}"
PREPARE_REVIEW="${PREPARE_REVIEW:-1}"

source "$CONDA_SH" || { echo "[error] failed to source $CONDA_SH"; exit 2; }
conda activate "$CONDA_ENV" || { echo "[error] failed to activate $CONDA_ENV"; exit 2; }
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:$SF/scripts:${PYTHONPATH:-}"

for path in "$SF" "$ECHO"; do
    [[ -d "$path" ]] || { echo "[error] missing repository: $path"; exit 2; }
done
for path in "$SF_CONFIG" "$ECHO_CONFIG" "$SF_CHECKPOINT" "$ECHO_CHECKPOINT" \
            "$SF_PROMPTS" "$ECHO_PROMPTS"; do
    [[ -f "$path" ]] || { echo "[error] missing file: $path"; exit 2; }
done
for path in "$SF/wan_models/Wan2.1-T2V-1.3B" \
            "$ECHO/wan_models/Wan2.1-T2V-1.3B"; do
    [[ -d "$path" ]] || { echo "[error] missing Wan model: $path"; exit 2; }
done
for prompts in "$SF_PROMPTS" "$ECHO_PROMPTS"; do
    count="$(grep -cve '^[[:space:]]*$' "$prompts")"
    [[ "$count" -eq "$EXPECTED_VIDEOS" ]] || {
        echo "[error] expected $EXPECTED_VIDEOS prompts, found $count in $prompts"
        exit 2
    }
done
python "$ROOT/scripts/validate_echo_prompts.py" "$ECHO_PROMPTS" \
    --expected-lines "$EXPECTED_VIDEOS" --expected-segments 3 --expected-duration 30 \
    --reference-sf "$SF_PROMPTS" || exit 2
python -c "import torch; from lifecycle_kv.role_episodic import compute_head_role_evidence; print('[preflight] torch', torch.__version__)" || exit 2

RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if command -v sha256sum >/dev/null 2>&1; then
    SF_PROMPT_SHA256="$(sha256sum "$SF_PROMPTS" | awk '{print $1}')"
    ECHO_PROMPT_SHA256="$(sha256sum "$ECHO_PROMPTS" | awk '{print $1}')"
    SF_CONFIG_FINGERPRINT="$(sha256sum "$SF_CONFIG" | awk '{print $1}')"
    ECHO_CONFIG_FINGERPRINT="$(sha256sum "$ECHO_CONFIG" | awk '{print $1}')"
else
    SF_PROMPT_SHA256="unavailable"
    ECHO_PROMPT_SHA256="unavailable"
    SF_CONFIG_FINGERPRINT="unavailable"
    ECHO_CONFIG_FINGERPRINT="unavailable"
fi
export HREM_RUN_COMMIT="$RUN_COMMIT"
export HREM_RUN_SEED="$SEED"
export HREM_RUN_FRAMES="$FRAMES"
export HREM_PROMPT_SHA256="$SF_PROMPT_SHA256"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces"
echo "[config] task=scene_switch commit=$RUN_COMMIT seed=$SEED frames=$FRAMES"
echo "[config] sf_prompts=$SF_PROMPTS sha256=$SF_PROMPT_SHA256"
echo "[config] echo_prompts=$ECHO_PROMPTS sha256=$ECHO_PROMPT_SHA256 output=$OUT_ROOT"
echo "[protocol] generation only; complete blind review before running metrics"

COMMON_OURS=(
    STRUCTURED_MEMORY_ENABLE=1
    STRUCTURED_MEMORY_GATE=0.10
    STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES=36
    STRUCTURED_MEMORY_ARCHIVE_POLICY=coverage
    STRUCTURED_MEMORY_SPATIAL_STRIDE=4
    STRUCTURED_MEMORY_TOP_K_FRAMES=3
    STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES=3
    STRUCTURED_MEMORY_SELECTION_POLICY=query
    STRUCTURED_MEMORY_SELECTION_SCOPE=shared
    STRUCTURED_MEMORY_PROMPT_PRIOR_WEIGHT=0.0
    STRUCTURED_MEMORY_READOUT_MODE=noisy_only
    STRUCTURED_MEMORY_ROUTING_SHARPNESS=5.0
    STRUCTURED_MEMORY_MARGIN_THRESHOLD=0.10
    STRUCTURED_MEMORY_QUERY_EMA_DECAY=0.9
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
    STRUCTURED_MEMORY_MEMORY_START_EPISODE=2
    STRUCTURED_MEMORY_EPISODE_GATE_MODE=dual_evidence
    STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE=2
    STRUCTURED_MEMORY_ORACLE_EPISODE_ID=-1
    STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE=auto
    STRUCTURED_MEMORY_DUAL_MIN_SEMANTIC_SIMILARITY=0.20
    STRUCTURED_MEMORY_DUAL_MIN_VISUAL_SIMILARITY=0.00
    STRUCTURED_MEMORY_DUAL_MIN_COMBINED_SCORE=0.55
    STRUCTURED_MEMORY_DUAL_MIN_EPISODE_MARGIN=0.05
    STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT=1
    STRUCTURED_MEMORY_DUAL_VISUAL_HEAD_FRACTION=0.25
    STRUCTURED_MEMORY_TRACE_ENABLED=1
    STRUCTURED_MEMORY_DEBUG=1
    STRUCTURED_MEMORY_DEBUG_LAYERS=15,18,20
    STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS=1
    SCENE_TRANSITION_RESET=1
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
        "task=scene_switch" \
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
    local metadata="$OUT_ROOT/$name/run_metadata.txt"
    local count
    count="$(video_count "$OUT_ROOT/$name")"
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

run_sf_control() {
    local name="sf_segmented_reset" output="$OUT_ROOT/sf_segmented_reset"
    local log="$OUT_ROOT/logs/sf_segmented_reset.log"
    should_skip "$name" 0 "$SF_PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT" 0 && return 0
    mkdir -p "$output"
    echo "[run] $name gpu=$GPU_SF local_equal_thirds_scheduler=1"
    (
        cd "$SF"
        export CUDA_VISIBLE_DEVICES="$GPU_SF"
        unset SF_FULL_ATTN_MAX_FRAMES AR_LATENT_TRACE_PATH
        export LIFECACHE_ENABLE=0 HEAD_ROLE_ENABLE=0 HEAD_ROLE_POOL_ENABLE=0
        export STRUCTURED_MEMORY_ENABLE=0 STRUCTURED_MEMORY_TRACE_ENABLED=0
        export SCENE_TRANSITION_RESET=1
        python inference.py \
            --config_path "$SF_CONFIG" --checkpoint_path "$SF_CHECKPOINT" \
            --data_path "$SF_PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    [[ "$rc" -eq 0 ]] && write_metadata "$name" "$SF_PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT"
    return "$rc"
}

run_echo() {
    local name="sf_echo_forcing" output="$OUT_ROOT/sf_echo_forcing"
    local log="$OUT_ROOT/logs/sf_echo_forcing.log"
    should_skip "$name" 0 "$ECHO_PROMPT_SHA256" "$ECHO_CONFIG_FINGERPRINT" 0 && return 0
    mkdir -p "$output"
    echo "[run] $name gpu=$GPU_ECHO official_recall_syntax=1"
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
    should_skip "$name" 1 "$SF_PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT" 1 && return 0
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
            --data_path "$SF_PROMPTS" --output_folder "$output" \
            --num_output_frames "$FRAMES" --seed "$SEED" --num_samples 1 \
            --use_ema --save_with_index
    ) >"$log" 2>&1
    local rc=$?
    [[ "$rc" -eq 0 ]] && write_metadata "$name" "$SF_PROMPT_SHA256" "$SF_CONFIG_FINGERPRINT"
    return "$rc"
}

status=0
if [[ "$PARALLEL" == "1" ]]; then
    run_sf_control & p0=$!
    run_echo & p1=$!
    run_ours ours_all_heads "$GPU_OURS_ALL" off & p2=$!
    run_ours ours_role "$GPU_OURS_ROLE" role_evidence \
        STRUCTURED_MEMORY_ROLE_CALIBRATION=hybrid \
        STRUCTURED_MEMORY_ROLE_THRESHOLD=0.45 \
        STRUCTURED_MEMORY_ROLE_KEEP_FRACTION=0.50 \
        STRUCTURED_MEMORY_ROLE_MIN_EVIDENCE_SPREAD=0.01 \
        STRUCTURED_MEMORY_ROLE_SHARPNESS=8.0 & p3=$!
    for pid in "$p0" "$p1" "$p2" "$p3"; do
        wait "$pid" || status=1
    done
else
    run_sf_control || status=1
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
        --methods sf_segmented_reset sf_echo_forcing ours_all_heads ours_role \
        --prompts "$SF_PROMPTS" --output "$OUT_ROOT/blind_review" \
        --prompt-count "$EXPECTED_VIDEOS" --seed 20260723 --force || status=1
fi

echo "[done] outputs=$OUT_ROOT status=$status"
echo "[next] freeze $OUT_ROOT/blind_review/scorecard.csv before revealing $OUT_ROOT/blind_review_private/key_private.json"
echo "[logs] inspect Echo recall decisions and our per-layer/per-call diagnosis"
exit "$status"
