#!/usr/bin/env bash
# Manual prompt thresholds, sign split, PF merges, and PF class contributions.
set -uo pipefail

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/runs/v97_qk_head_scores}"
SCORE_ARTIFACT="$PROFILE_ROOT/scores/qk_head_score_artifact.json"
MAP_DIR="${MAP_DIR:-$PROFILE_ROOT/maps}"
MAP_MANIFEST="$MAP_DIR/head_map_manifest.json"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v97_threshold_pf_merge32}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FRAMES="${FRAMES:-120}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

METHODS=(
    prompt_tau_0p0_merge
    prompt_tau_0p5_merge
    prompt_tau_1p0_merge
    prompt_tau_1p5_merge
    prompt_tau_2p0_merge
    prompt_tau_1p0_cyclic
    prompt_tau_1p0_recent
    prompt_tau_1p0_random_merge
    prompt_tau_1p0_reversed_merge
    sign_rpos_0p5_stride_merge
    pf_ar_stride_merge
    pf_aw_stride_merge
    pf_native
    pf_anchor_extended_recent
    pf_wave_extended_recent
    pf_veil_extended_recent
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 1 ]] || {
    echo "[error] v97 cache screen requires at least one GPU id"
    exit 2
}
for path in \
    "$PF" "$PF_CONFIG" "$PF_CHECKPOINT" "$PROMPTS" \
    "$SCORE_ARTIFACT" "$MAP_MANIFEST"; do
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
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1
export PYRAMIDKV_POLICY_TRACE_LAYERS="${PYRAMIDKV_POLICY_TRACE_LAYERS:-0,7,15,23,29}"
export PYRAMIDKV_POLICY_TRACE_STRIDE="${PYRAMIDKV_POLICY_TRACE_STRIDE:-3}"
export PYRAMIDKV_POLICY_TRACE_MAX_RECORDS="${PYRAMIDKV_POLICY_TRACE_MAX_RECORDS:-20000}"

mkdir -p "$OUT_ROOT"/{logs,status,configs,traces,diagnostics}
RUN_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
PROMPT_SHA256="$(sha256sum "$PROMPTS" | awk '{print $1}')"
SCORE_SHA256="$(python - "$SCORE_ARTIFACT" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["files"]["score_csv_sha256"])
PY
)"
MAP_MANIFEST_SHA256="$(sha256sum "$MAP_MANIFEST" | awk '{print $1}')"
RUN_MANIFEST="$OUT_ROOT/run_manifest.env"
if [[ -s "$RUN_MANIFEST" ]] && \
    find "$OUT_ROOT/status" -maxdepth 1 -type f -name '*.done' -print -quit |
        grep -q .; then
    old_commit="$(awk -F= '$1=="RUN_COMMIT"{print substr($0,index($0,"=")+1)}' "$RUN_MANIFEST")"
    old_prompt="$(awk -F= '$1=="PROMPT_SHA256"{print $2}' "$RUN_MANIFEST")"
    old_frames="$(awk -F= '$1=="FRAMES"{print $2}' "$RUN_MANIFEST")"
    old_seed="$(awk -F= '$1=="SEED"{print $2}' "$RUN_MANIFEST")"
    old_score="$(awk -F= '$1=="SCORE_SHA256"{print $2}' "$RUN_MANIFEST")"
    old_maps="$(awk -F= '$1=="MAP_MANIFEST_SHA256"{print $2}' "$RUN_MANIFEST")"
    if [[ "$old_commit" != "$RUN_COMMIT" || \
          "$old_prompt" != "$PROMPT_SHA256" || \
          "$old_frames" != "$FRAMES" || \
          "$old_seed" != "$SEED" || \
          "$old_score" != "$SCORE_SHA256" || \
          "$old_maps" != "$MAP_MANIFEST_SHA256" ]]; then
        echo "[error] existing v97 videos have different provenance; use a clean OUT_ROOT"
        exit 2
    fi
fi
{
    printf 'EXPERIMENT=v97_threshold_pf_merge32\n'
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'SEED=%s\n' "$SEED"
    printf 'SCORE_SHA256=%s\n' "$SCORE_SHA256"
    printf 'MAP_MANIFEST_SHA256=%s\n' "$MAP_MANIFEST_SHA256"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'POLICY_TRACE_LAYERS=%s\n' "$PYRAMIDKV_POLICY_TRACE_LAYERS"
    printf 'POLICY_TRACE_STRIDE=%s\n' "$PYRAMIDKV_POLICY_TRACE_STRIDE"
} >"$RUN_MANIFEST"

video_count() {
    local output="$1"
    [[ -d "$output" ]] || { printf '0'; return; }
    find "$output" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

audit_map() {
    local name="$1" path="$2"
    python - "$name" "$path" "$MAP_MANIFEST" \
        "$OUT_ROOT/diagnostics/$name.map_audit.json" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

name, map_path, manifest_path, output_path = sys.argv[1:]
path = pathlib.Path(map_path)
rows = [
    [int(value.strip()) for value in row]
    for row in csv.reader(path.open(encoding="utf-8"))
    if row
]
if len(rows) != 30 or any(len(row) != 12 for row in rows):
    raise SystemExit(f"{name}: invalid map shape")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
expected = manifest["maps"][path.stem]["sha256"]
if digest != expected:
    raise SystemExit(
        f"{name}: map hash mismatch expected={expected} actual={digest}"
    )
counts = {}
for value in (item for row in rows for item in row):
    counts[str(value)] = counts.get(str(value), 0) + 1
payload = {
    "method": name,
    "map": str(path.resolve()),
    "sha256": digest,
    "shape": [30, 12],
    "label_counts": counts,
    "per_layer_counts": [
        {str(value): row.count(value) for value in sorted(set(row))}
        for row in rows
    ],
}
pathlib.Path(output_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"[HeadMapAudit] method={name} sha256={digest} counts={counts}")
PY
}

write_config() {
    local name="$1" labels="$2" stable="$3" responsive="$4" ablation="$5"
    local mode="binary"
    if [[ "$ablation" != "none" ]]; then
        mode="pf_class_ablation"
    elif [[ "$responsive" == "pf" ]]; then
        mode="pf_native"
    fi
    {
        printf 'name=%s\n' "$name"
        printf 'mode=%s\n' "$mode"
        printf 'run_commit=%s\n' "$RUN_COMMIT"
        printf 'prompt_sha256=%s\n' "$PROMPT_SHA256"
        printf 'frames=%s\n' "$FRAMES"
        printf 'seed=%s\n' "$SEED"
        printf 'map_manifest_sha256=%s\n' "$MAP_MANIFEST_SHA256"
        printf 'labels=%s\n' "$labels"
        printf 'label_sha256=%s\n' "$(sha256sum "$labels" | awk '{print $1}')"
        printf 'score_sha256=%s\n' "$SCORE_SHA256"
        printf 'stable_policy=%s\n' "$stable"
        printf 'responsive_policy=%s\n' "$responsive"
        printf 'pf_extended_recent_ablation=%s\n' "$ablation"
        if [[ "$stable" == "hybrid" ]]; then
            printf 'stable_composition=sink3+stride_cap2+cyclic_cap2+recent4\n'
        elif [[ "$stable" == "stride" ]]; then
            printf 'stable_composition=sink3+stride_cap4+recent4\n'
        fi
        case "$responsive" in
            cyclic) printf 'responsive_composition=sink1+cyclic_cap4+recent4\n' ;;
            merge) printf 'responsive_composition=sink3+merge_patch2_cap4+recent4\n' ;;
            recent) printf 'responsive_composition=sink3+recent4\n' ;;
            pf) printf 'responsive_composition=pf_native_or_ablation\n' ;;
        esac
    } >"$OUT_ROOT/configs/$name.env"
}

run_cell() {
    local name="$1" gpu="$2" labels="$3" stable="$4" responsive="$5" ablation="$6"
    local output="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local marker="$OUT_ROOT/status/$name.done"
    local trace="$OUT_ROOT/traces/$name.policy.jsonl"
    local args=()
    if [[ "$responsive" != "pf" ]]; then
        args+=(
            --pyramidkv_binary_stable_policy "$stable"
            --pyramidkv_binary_responsive_policy "$responsive"
        )
    fi
    if [[ "$ablation" != "none" ]]; then
        args+=(
            --pyramidkv_pf_extended_recent_ablation "$ablation"
        )
    fi
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        [[ "$(video_count "$output")" -eq "$PROMPT_COUNT" ]]; then
        local frozen_config="$OUT_ROOT/configs/$name.env"
        [[ -s "$frozen_config" && -s "$trace" ]] || {
            echo "[error] incomplete provenance for completed cell $name"
            return 2
        }
        grep -Fqx "run_commit=$RUN_COMMIT" "$frozen_config" &&
        grep -Fqx "prompt_sha256=$PROMPT_SHA256" "$frozen_config" &&
        grep -Fqx "frames=$FRAMES" "$frozen_config" &&
        grep -Fqx "seed=$SEED" "$frozen_config" &&
        grep -Fqx "score_sha256=$SCORE_SHA256" "$frozen_config" &&
        grep -Fqx "map_manifest_sha256=$MAP_MANIFEST_SHA256" "$frozen_config" || {
            echo "[error] frozen config mismatch for completed cell $name"
            return 2
        }
        echo "[skip] $name"
        return
    fi
    if [[ "$(video_count "$output")" -ne 0 ]]; then
        echo "[error] partial output exists for $name; use a clean OUT_ROOT"
        return 2
    fi
    rm -f "$marker" "$trace"
    mkdir -p "$output"
    audit_map "$name" "$labels" || return 1
    write_config "$name" "$labels" "$stable" "$responsive" "$ablation"
    if ! (
        cd "$PF"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYRAMIDKV_POLICY_TRACE_PATH="$trace"
        python inference.py \
            --config_path "$PF_CONFIG" \
            --checkpoint_path "$PF_CHECKPOINT" \
            --data_path "$PROMPTS" \
            --output_folder "$output" \
            --num_output_frames "$FRAMES" \
            --seed "$SEED" --num_samples 1 --use_ema --save_with_index \
            --start_idx 0 --end_idx "$PROMPT_COUNT" --reseed_per_prompt \
            --pyramidkv_head_config_path "$labels" \
            "${args[@]}"
    ) >"$log" 2>&1; then
        echo "[error] inference failed for $name; inspect $log"
        return 1
    fi
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx 0 --end-idx "$PROMPT_COUNT" \
        --output-json "$OUT_ROOT/diagnostics/$name.video_audit.json" \
        >"$OUT_ROOT/diagnostics/$name.video_audit.log" 2>&1 || return 1
    grep -q '\[PyramidKVRuntimePolicy\]' "$log" || {
        echo "[error] missing runtime policy audit in $log"
        return 1
    }
    if [[ "$responsive" != "pf" ]]; then
        grep -q "\[BinaryPolicyOverride\].*stable=$stable.*responsive=$responsive" "$log" || {
            echo "[error] missing binary policy marker in $log"
            return 1
        }
    fi
    if [[ "$ablation" != "none" ]]; then
        grep -q "\[PFClassExtendedRecentAblation\].*target=$ablation" "$log" || {
            echo "[error] missing PF ablation marker in $log"
            return 1
        }
    fi
    [[ -s "$trace" ]] || {
        echo "[error] missing policy trace $trace"
        return 1
    }
    if grep -q '\[PyramidKVPolicyTraceError\]' "$log"; then
        echo "[error] policy trace failure in $log"
        return 1
    fi
    printf 'ok\n' >"$marker"
}

M="$MAP_DIR"
ALL_CELLS=(
    "prompt_tau_0p0_merge|$M/prompt_tau_0.csv|stride|merge|none"
    "prompt_tau_0p5_merge|$M/prompt_tau_0p5.csv|stride|merge|none"
    "prompt_tau_1p0_merge|$M/prompt_tau_1.csv|stride|merge|none"
    "prompt_tau_1p5_merge|$M/prompt_tau_1p5.csv|stride|merge|none"
    "prompt_tau_2p0_merge|$M/prompt_tau_2.csv|stride|merge|none"
    "prompt_tau_1p0_cyclic|$M/prompt_tau_1.csv|stride|cyclic|none"
    "prompt_tau_1p0_recent|$M/prompt_tau_1.csv|stride|recent|none"
    "prompt_tau_1p0_random_merge|$M/prompt_tau_1_random.csv|stride|merge|none"
    "prompt_tau_1p0_reversed_merge|$M/prompt_tau_1_reversed.csv|stride|merge|none"
    "sign_rpos_0p5_stride_merge|$M/sign_rpos_0p5.csv|stride|merge|none"
    "pf_ar_stride_merge|$M/pf_anchor_vs_rest.csv|stride|merge|none"
    "pf_aw_stride_merge|$M/pf_anchor_wave_vs_veil.csv|stride|merge|none"
    "pf_native|$M/pf_native.csv|pf|pf|none"
    "pf_anchor_extended_recent|$M/pf_anchor_extended_recent.csv|pf|pf|anchor"
    "pf_wave_extended_recent|$M/pf_wave_extended_recent.csv|pf|pf|wave"
    "pf_veil_extended_recent|$M/pf_veil_extended_recent.csv|pf|pf|veil"
)
for cell in "${ALL_CELLS[@]}"; do
    IFS='|' read -r name labels _stable _responsive _ablation <<<"$cell"
    [[ -s "$labels" ]] || { echo "[error] missing map for $name: $labels"; exit 2; }
done
[[ "${#METHODS[@]}" -eq "${#ALL_CELLS[@]}" ]] || {
    echo "[error] METHODS and ALL_CELLS have different lengths"
    exit 2
}
for index in "${!METHODS[@]}"; do
    IFS='|' read -r cell_name _rest <<<"${ALL_CELLS[$index]}"
    [[ "${METHODS[$index]}" == "$cell_name" ]] || {
        echo "[error] method/cell mismatch at $index"
        exit 2
    }
done

CELL_START="${CELL_START:-0}"
CELL_END="${CELL_END:-${#ALL_CELLS[@]}}"
[[ "$CELL_START" =~ ^[0-9]+$ && "$CELL_END" =~ ^[0-9]+$ ]] || {
    echo "[error] CELL_START and CELL_END must be non-negative integers"
    exit 2
}
(( CELL_START <= CELL_END && CELL_END <= ${#ALL_CELLS[@]} )) || {
    echo "[error] invalid cell range $CELL_START..$CELL_END"
    exit 2
}
CELLS=("${ALL_CELLS[@]:$CELL_START:$((CELL_END-CELL_START))}")
STATUS=0
echo "[v97-cache] commit=$RUN_COMMIT prompts=$PROMPT_COUNT frames=$FRAMES cells=$CELL_START..$CELL_END (${#CELLS[@]} cells) gpus=${#GPUS[@]}"
wave=0
for ((base=0; base<${#CELLS[@]}; base+=${#GPUS[@]})); do
    pids=()
    echo "[v97-cache] wave=$wave cells=$((CELL_START+base))..$((CELL_START + base + ${#GPUS[@]} - 1))"
    for ((slot=0; slot<${#GPUS[@]} && base+slot<${#CELLS[@]}; slot++)); do
        IFS='|' read -r name labels stable responsive ablation <<<"${CELLS[$((base+slot))]}"
        run_cell "$name" "${GPUS[$slot]}" "$labels" "$stable" "$responsive" "$ablation" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || STATUS=1
    done
    wave=$((wave + 1))
done
echo "[v97-cache] completed status=$STATUS out=$OUT_ROOT"
exit "$STATUS"
