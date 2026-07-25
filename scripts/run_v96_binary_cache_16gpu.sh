#!/usr/bin/env bash
# Binary head membership x responsive cache policy, 16 cells x 32 prompts x 30 s.
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/runs/v96_qk_head_profile}"
LABEL_DIR="${LABEL_DIR:-$PROFILE_ROOT/labels}"
PROFILE_REPORT="$LABEL_DIR/qk_head_threshold_report.json"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v96_binary_cache32}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

METHODS=(
    pf
    pf_binary_cyclic pf_binary_merge pf_binary_recent
    cfg_cyclic cfg_merge semantic_cyclic semantic_merge
    consensus_cyclic consensus_merge consensus_recent
    consensus_merge_v78 consensus_cyclic_v78
    random_merge inverse_merge pf_binary_merge_v78
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 16 ]] || {
    echo "[error] v96 cache screen requires exactly 16 GPU ids"
    exit 2
}
CFG_MAP="$LABEL_DIR/prompt_cfg_threshold.csv"
SEMANTIC_MAP="$LABEL_DIR/prompt_semantic_threshold.csv"
CONSENSUS_MAP="$LABEL_DIR/prompt_consensus_threshold.csv"
RANDOM_MAP="$LABEL_DIR/prompt_consensus_random.csv"
INVERSE_MAP="$LABEL_DIR/prompt_consensus_inverse.csv"
PF_BINARY="$LABEL_DIR/pf_binary.csv"
for path in \
    "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PF_LABELS" "$PROMPTS" \
    "$PROFILE_REPORT" \
    "$CFG_MAP" "$SEMANTIC_MAP" "$CONSENSUS_MAP" \
    "$RANDOM_MAP" "$INVERSE_MAP" "$PF_BINARY"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done
PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq 32 ]] || {
    echo "[error] expected 32 prompts, found $PROMPT_COUNT"
    exit 2
}

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1

mkdir -p "$OUT_ROOT"/{logs,status,configs,traces,diagnostics}
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
PROFILE_SHA256="$(sha256sum "$PROFILE_REPORT" | awk '{print $1}')"
{
    printf 'EXPERIMENT=v96_binary_cache32\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'PROFILE_REPORT_SHA256=%s\n' "$PROFILE_SHA256"
} >"$OUT_ROOT/run_manifest.env"

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

write_config() {
    local name="$1" labels="$2" policy="$3" transition="$4"
    {
        printf 'name=%s\n' "$name"
        printf 'labels=%s\n' "$labels"
        printf 'label_sha256=%s\n' "$(sha256sum "$labels" | awk '{print $1}')"
        printf 'responsive_policy=%s\n' "$policy"
        printf 'transition=%s\n' "$transition"
        printf 'stable_policy=sink3+stride4+recent4\n'
        case "$policy" in
            cyclic) printf 'responsive_composition=sink1+cyclic4+recent4\n' ;;
            merge) printf 'responsive_composition=sink3+merge_patch2_cap4+recent4\n' ;;
            recent) printf 'responsive_composition=sink3+recent4\n' ;;
            pf) printf 'responsive_composition=pf_three_class_native\n' ;;
        esac
    } >"$OUT_ROOT/configs/$name.env"
}

run_cell() {
    local name="$1" gpu="$2" labels="$3" policy="$4" transition="$5"
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local marker="$OUT_ROOT/status/$name.done"
    local trace="$OUT_ROOT/traces/$name.transition.jsonl"
    local policy_args=() transition_args=()
    if [[ "$policy" != "pf" ]]; then
        policy_args=(
            --pyramidkv_binary_responsive_policy "$policy"
        )
    fi
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
            --pyramidkv_cache_transition_trace_path "$trace"
            --pyramidkv_cache_transition_debug
        )
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        [[ "$(video_count "$output")" -eq "$PROMPT_COUNT" ]]; then
        echo "[skip] $name"
        return
    fi
    if [[ "$(video_count "$output")" -ne 0 ]]; then
        echo "[error] partial output exists for $name; use a clean OUT_ROOT"
        return 2
    fi
    rm -f "$marker" "$trace"
    mkdir -p "$output"
    write_config "$name" "$labels" "$policy" "$transition"
    (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        python inference.py \
            --config_path "$PF_CONFIG" \
            --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$output" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" --num_samples 1 --use_ema --save_with_index \
            --start_idx 0 --end_idx "$PROMPT_COUNT" --reseed_per_prompt \
            --pyramidkv_head_config_path "$labels" \
            "${policy_args[@]}" "${transition_args[@]}"
    ) >"$log" 2>&1
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx 0 --end-idx "$PROMPT_COUNT" \
        --output-json "$OUT_ROOT/diagnostics/$name.audit.json" \
        >"$OUT_ROOT/diagnostics/$name.audit.log" 2>&1 || return 1
    grep -q '\[PyramidKVRuntimePolicy\]' "$log" || {
        echo "[error] missing runtime policy audit in $log"
        return 1
    }
    if [[ "$policy" != "pf" ]]; then
        grep -q "\[BinaryPolicyOverride\].*responsive=$policy" "$log" || {
            echo "[error] missing binary policy marker in $log"
            return 1
        }
    fi
    if [[ "$transition" == "1" && ! -s "$trace" ]]; then
        echo "[error] missing transition trace $trace"
        return 1
    fi
    printf 'ok\n' >"$marker"
}

PIDS=()
STATUS=0
launch() {
    run_cell "$@" &
    PIDS+=("$!")
}

launch pf                     "${GPUS[0]}"  "$PF_LABELS"     pf     0
launch pf_binary_cyclic       "${GPUS[1]}"  "$PF_BINARY"     cyclic 0
launch pf_binary_merge        "${GPUS[2]}"  "$PF_BINARY"     merge  0
launch pf_binary_recent       "${GPUS[3]}"  "$PF_BINARY"     recent 0
launch cfg_cyclic             "${GPUS[4]}"  "$CFG_MAP"       cyclic 0
launch cfg_merge              "${GPUS[5]}"  "$CFG_MAP"       merge  0
launch semantic_cyclic        "${GPUS[6]}"  "$SEMANTIC_MAP"  cyclic 0
launch semantic_merge         "${GPUS[7]}"  "$SEMANTIC_MAP"  merge  0
launch consensus_cyclic       "${GPUS[8]}"  "$CONSENSUS_MAP" cyclic 0
launch consensus_merge        "${GPUS[9]}"  "$CONSENSUS_MAP" merge  0
launch consensus_recent       "${GPUS[10]}" "$CONSENSUS_MAP" recent 0
launch consensus_merge_v78    "${GPUS[11]}" "$CONSENSUS_MAP" merge  1
launch consensus_cyclic_v78   "${GPUS[12]}" "$CONSENSUS_MAP" cyclic 1
launch random_merge           "${GPUS[13]}" "$RANDOM_MAP"    merge  0
launch inverse_merge          "${GPUS[14]}" "$INVERSE_MAP"   merge  0
launch pf_binary_merge_v78    "${GPUS[15]}" "$PF_BINARY"     merge  1

echo "[v96-cache] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES"
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done
if [[ "$STATUS" -eq 0 ]]; then
    mapfile -t TRACES < <(
        find "$OUT_ROOT/traces" -maxdepth 1 -type f -name '*.transition.jsonl' | sort
    )
    [[ "${#TRACES[@]}" -eq 3 ]] || STATUS=1
    if [[ "${#TRACES[@]}" -eq 3 ]]; then
        python "$ROOT/scripts/summarize_cache_transition_trace.py" \
            "${TRACES[@]}" --strict \
            --output-json "$OUT_ROOT/diagnostics/cache_transition_summary.json" \
            --output-md "$OUT_ROOT/diagnostics/cache_transition_summary.md" \
            >"$OUT_ROOT/diagnostics/cache_transition_summary.log" 2>&1 || STATUS=1
    fi
fi
echo "[v96-cache] completed status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
