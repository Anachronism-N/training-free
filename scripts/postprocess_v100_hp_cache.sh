#!/usr/bin/env bash
# Audit/stage v100 videos, prepare blind review, then run fixed metrics.
# Usage:
#   bash scripts/postprocess_v100_hp_cache.sh candidate32 prepare
#   HUMAN_REVIEW_DONE=1 bash scripts/postprocess_v100_hp_cache.sh candidate32 metrics
set -uo pipefail

MODE="${1:-}"
PHASE="${2:-prepare}"
[[ "$MODE" == "candidate32" || "$MODE" == "main128" ]] || {
    echo "usage: $0 candidate32|main128 prepare|metrics"
    exit 2
}
[[ "$PHASE" == "prepare" || "$PHASE" == "metrics" ]] || {
    echo "usage: $0 candidate32|main128 prepare|metrics"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
REUSE_PF_DIR="${REUSE_PF_DIR:-}"
REUSE_PF_BINARY_DIR="${REUSE_PF_BINARY_DIR:-}"
REUSE_SF_DIR="${REUSE_SF_DIR:-}"
HUMAN_REVIEW_DONE="${HUMAN_REVIEW_DONE:-0}"
FORCE_METRICS="${FORCE_METRICS:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_COMPREHENSIVE="${RUN_COMPREHENSIVE:-1}"
RUN_TEMPORAL="${RUN_TEMPORAL:-1}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
TEMPORAL_FRAME_STEP="${TEMPORAL_FRAME_STEP:-2}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree}"

for pair in \
    "HUMAN_REVIEW_DONE=$HUMAN_REVIEW_DONE" \
    "FORCE_METRICS=$FORCE_METRICS" \
    "RUN_VBENCH=$RUN_VBENCH" \
    "RUN_COMPREHENSIVE=$RUN_COMPREHENSIVE" \
    "RUN_TEMPORAL=$RUN_TEMPORAL"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    [[ "$value" == "0" || "$value" == "1" ]] || {
        echo "[error] $key must be 0 or 1"
        exit 2
    }
done

if [[ "$MODE" == "candidate32" ]]; then
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v100_hp_cache_candidate32}"
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
    EXPECTED=32
    GENERATED=(
        pf_ar_neutral_stride_cyclic
        pf_aw_neutral_stride_cyclic
        history_polarity_stride_cyclic
        history_polarity_random_stride_cyclic
        history_polarity_inverted_stride_cyclic
        history_polarity_tau_m0p1_stride_cyclic
        history_polarity_tau_p0p1_stride_cyclic
        history_polarity_stride_cyclic_v78
    )
else
    RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v100_hp_cache_main128}"
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
    EXPECTED=128
    GENERATED=(
        pf_ar_neutral_stride_cyclic
        history_polarity_stride_cyclic
        history_polarity_stride_cyclic_v78
    )
fi

[[ -n "$REUSE_PF_DIR" && -n "$REUSE_PF_BINARY_DIR" ]] || {
    echo "[error] set REUSE_PF_DIR and REUSE_PF_BINARY_DIR"
    exit 2
}
for path in \
    "$ROOT" "$PF" "$RUN_ROOT" "$PROMPTS" \
    "$REUSE_PF_DIR" "$REUSE_PF_BINARY_DIR" "$CONDA_SH"; do
    [[ -e "$path" ]] || {
        echo "[error] missing required path: $path"
        exit 2
    }
done
if [[ -n "$REUSE_SF_DIR" && ! -d "$REUSE_SF_DIR" ]]; then
    echo "[error] missing REUSE_SF_DIR: $REUSE_SF_DIR"
    exit 2
fi

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq "$EXPECTED" ]] || {
    echo "[error] expected $EXPECTED prompts, found $PROMPT_COUNT"
    exit 2
}

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -ge 1 ]] || {
    echo "[error] GPU_LIST is empty"
    exit 2
}
[[ "${#GPUS[@]}" -eq "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" ]] || {
    echo "[error] GPU_LIST contains duplicates"
    exit 2
}
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || {
        echo "[error] invalid GPU id: $gpu"
        exit 2
    }
done

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"

METRICS="$RUN_ROOT/metrics"
INPUTS="$METRICS/eval_inputs"
mkdir -p \
    "$METRICS/logs" "$METRICS/status" "$METRICS/video_audits" \
    "$METRICS/comprehensive_parts" "$METRICS/vbench_long" "$INPUTS"

python - \
    "$RUN_ROOT/experiment_contract.json" "$RUN_ROOT" "$PROMPTS" \
    "$MODE" "$EXPECTED" "$METRICS/workflow_contract_audit.json" \
    "${GENERATED[@]}" \
    >"$METRICS/logs/workflow_contract_audit.log" 2>&1 <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    contract_path,
    run_root,
    prompts,
    mode,
    raw_expected,
    output,
    *generated,
) = sys.argv[1:]
contract_path = Path(contract_path).resolve()
run_root = Path(run_root).resolve()
prompts = Path(prompts).resolve()
expected = int(raw_expected)
contract = json.loads(contract_path.read_text(encoding="utf-8"))

if contract.get("version") != 2:
    raise ValueError(f"expected v100 contract version 2: {contract_path}")
if contract.get("experiment") != "v99_binary_cache_recovery":
    raise ValueError("unexpected generation experiment")
if contract.get("mode") != mode:
    raise ValueError(
        f"mode mismatch: expected={mode} actual={contract.get('mode')}"
    )
prompt = contract.get("prompt") or {}
if (
    Path(str(prompt.get("path"))).resolve() != prompts
    or prompt.get("sha256") != sha256(prompts)
    or int(prompt.get("count", -1)) != expected
):
    raise ValueError("prompt binding differs from generation contract")
contract_cells = [str(item.get("name")) for item in contract.get("cells", [])]
if contract_cells != generated:
    raise ValueError(
        f"cell order mismatch: expected={generated} actual={contract_cells}"
    )
video = contract.get("video_contract") or {}
required_video = {
    "decoded_frames": 477,
    "fps": 16.0,
    "width": 832,
    "height": 480,
}
if any(video.get(key) != value for key, value in required_video.items()):
    raise ValueError(f"video contract mismatch: {video}")

map_audit = {}
for name, item in (contract.get("maps") or {}).items():
    if name == "pf_labels":
        continue
    counts = item.get("label_counts") or {}
    cross = item.get("pf_cross_tab") or {}
    if sum(int(value) for value in counts.values()) != 360:
        raise ValueError(f"{name}: label counts do not sum to 360")
    if sum(int(row.get("heads", 0)) for row in cross.values()) != 360:
        raise ValueError(f"{name}: PF cross-tab does not sum to 360")
    map_audit[name] = {
        "sha256": item.get("sha256"),
        "label_counts": counts,
        "pf_cross_tab": cross,
    }

files = []
for method in generated:
    if not (run_root / method).is_dir():
        raise ValueError(f"missing generated video directory: {method}")
    for rank in range(4):
        required = [
            run_root / "status" / f"{method}.shard{rank}.done.json",
            run_root / "configs" / f"{method}.shard{rank}.json",
            run_root / "diagnostics" / f"{method}.shard{rank}.video.json",
            run_root / "diagnostics" / f"{method}.shard{rank}.trace.json",
            run_root / "traces" / f"{method}.shard{rank}.policy.jsonl",
            run_root / "logs" / f"{method}.shard{rank}.log",
        ]
        if method.endswith("_v78"):
            required.append(
                run_root
                / "traces"
                / f"{method}.shard{rank}.transition.jsonl"
            )
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing/empty generation evidence: {path}")
            files.append({"path": str(path), "sha256": sha256(path)})

payload = {
    "version": 1,
    "mode": mode,
    "expected_videos": expected,
    "contract_path": str(contract_path),
    "contract_sha256": sha256(contract_path),
    "cells": generated,
    "maps": map_audit,
    "evidence_files": files,
    "ok": True,
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
[[ "$?" -eq 0 ]] || {
    echo "[error] generation contract audit failed"
    exit 1
}

METHODS=(pf_native pf_binary_read_reference "${GENERATED[@]}")
declare -A SOURCES
SOURCES[pf_native]="$REUSE_PF_DIR"
SOURCES[pf_binary_read_reference]="$REUSE_PF_BINARY_DIR"
for method in "${GENERATED[@]}"; do
    SOURCES["$method"]="$RUN_ROOT/$method"
done
if [[ -n "$REUSE_SF_DIR" ]]; then
    METHODS=(sf_native "${METHODS[@]}")
    SOURCES[sf_native]="$REUSE_SF_DIR"
fi

AUDIT_STATUS=0
VIDEO_DIRS=()
for method in "${METHODS[@]}"; do
    source_dir="${SOURCES[$method]}"
    stage_dir="$INPUTS/$method"
    VIDEO_DIRS+=("$stage_dir")
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$source_dir" \
        --start-idx 0 --end-idx "$EXPECTED" \
        --expected-frames 477 --expected-fps 16 \
        --expected-width 832 --expected-height 480 \
        --fps-tolerance 0.05 --allow-outside-interval \
        --output-json "$METRICS/video_audits/$method.json" \
        --stage-dir "$stage_dir" --replace-stage \
        >"$METRICS/logs/video_audit.$method.log" 2>&1 || AUDIT_STATUS=1
done
[[ "$AUDIT_STATUS" -eq 0 ]] || {
    echo "[error] at least one video audit/stage failed"
    exit 1
}

BLIND_REVIEW="$RUN_ROOT/blind_review"
BLIND_PRIVATE="$RUN_ROOT/blind_review_private"
if [[ ! -f "$BLIND_REVIEW/.complete.json" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$INPUTS" --methods "${METHODS[@]}" \
        --prompts "$PROMPTS" --prompt-count "$EXPECTED" \
        --output "$BLIND_REVIEW" --private-output "$BLIND_PRIVATE" \
        >"$METRICS/logs/prepare_blind_review.log" 2>&1 || exit 1
fi

{
    printf 'MODE=%s\n' "$MODE"
    printf 'PHASE=%s\n' "$PHASE"
    printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'EXPECTED=%s\n' "$EXPECTED"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'SAMPLE_FRAMES=%s\n' "$SAMPLE_FRAMES"
    printf 'TEMPORAL_FRAME_STEP=%s\n' "$TEMPORAL_FRAME_STEP"
    printf 'VBENCH_DIMS=%s\n' "$VBENCH_DIMS"
    printf 'BLIND_REVIEW=%s\n' "$BLIND_REVIEW"
} >"$METRICS/metric_manifest.env"

if [[ "$PHASE" == "prepare" ]]; then
    echo "[v100-postprocess] audits passed; review $BLIND_REVIEW"
    echo "[v100-postprocess] do not open $BLIND_PRIVATE before freezing review"
    exit 0
fi
[[ "$HUMAN_REVIEW_DONE" == "1" ]] || {
    echo "[error] set HUMAN_REVIEW_DONE=1 only after blind review is frozen"
    exit 2
}
python "$ROOT/scripts/prepare_blind_review.py" \
    --run-root "$INPUTS" --methods "${METHODS[@]}" \
    --prompts "$PROMPTS" --prompt-count "$EXPECTED" \
    --output "$BLIND_REVIEW" --private-output "$BLIND_PRIVATE" \
    --verify-frozen \
    >"$METRICS/logs/verify_frozen_blind_review.log" 2>&1 || {
        echo "[error] blind review is not frozen or no longer verifies"
        exit 2
    }

wait_batch() {
    local status=0
    local pid
    for pid in "${BATCH_PIDS[@]}"; do
        wait "$pid" || status=1
    done
    BATCH_PIDS=()
    BATCH_COUNT=0
    return "$status"
}

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" && -f "$INFO" ]] || {
        echo "[error] VBench-Long missing under $VBENCH_ROOT"
        exit 2
    }
    read -r -a DIMS <<<"$VBENCH_DIMS"
    BATCH_PIDS=()
    BATCH_COUNT=0
    VBENCH_STATUS=0
    for method in "${METHODS[@]}"; do
        output="$METRICS/vbench_long/$method"
        marker="$METRICS/status/vbench.$method.done"
        mkdir -p "$output"
        if [[ "$FORCE_METRICS" != "1" && -s "$marker" && -s "$output/results.json" ]]; then
            echo "[skip] VBench-Long $method"
            continue
        fi
        gpu="${GPUS[$BATCH_COUNT]}"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            cd "$VBENCH_ROOT" || exit 2
            python "$EVAL" \
                --videos_path "$INPUTS/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO" &&
                test -s "$output/results.json" &&
                printf 'ok\n' >"$marker"
        ) >"$output/run.log" 2>&1 &
        BATCH_PIDS+=("$!")
        BATCH_COUNT=$((BATCH_COUNT + 1))
        if [[ "$BATCH_COUNT" -eq "${#GPUS[@]}" ]]; then
            wait_batch || VBENCH_STATUS=1
        fi
    done
    [[ "$BATCH_COUNT" -eq 0 ]] || wait_batch || VBENCH_STATUS=1
    [[ "$VBENCH_STATUS" -eq 0 ]] || {
        echo "[error] at least one VBench-Long job failed"
        exit 1
    }
    python "$ROOT/scripts/collect_vbench_long_results.py" \
        --root "$METRICS/vbench_long" \
        --methods "${METHODS[@]}" --dimensions "${DIMS[@]}" \
        --output-json "$METRICS/vbench_long_summary.json" \
        --output-csv "$METRICS/vbench_long_summary.csv" \
        --output-md "$METRICS/vbench_long_summary.md" \
        >"$METRICS/logs/collect_vbench.log" 2>&1 || exit 1
fi

if [[ "$RUN_COMPREHENSIVE" == "1" ]]; then
    BATCH_PIDS=()
    BATCH_COUNT=0
    COMP_STATUS=0
    for method in "${METHODS[@]}"; do
        output="$METRICS/comprehensive_parts/$method.json"
        marker="$METRICS/status/comprehensive.$method.done"
        log="$METRICS/logs/comprehensive.$method.log"
        if [[ "$FORCE_METRICS" != "1" && -s "$marker" && -s "$output" ]]; then
            echo "[skip] comprehensive $method"
            continue
        fi
        gpu="${GPUS[$BATCH_COUNT]}"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            python "$ROOT/scripts/evaluate_comprehensive.py" \
                --video_dirs "$INPUTS/$method" \
                --prompts "$PROMPTS" --output "$output" \
                --gpu 0 --sample_frames "$SAMPLE_FRAMES" \
                --batch_size 8 --skip_m3 &&
                test -s "$output" &&
                printf 'ok\n' >"$marker"
        ) >"$log" 2>&1 &
        BATCH_PIDS+=("$!")
        BATCH_COUNT=$((BATCH_COUNT + 1))
        if [[ "$BATCH_COUNT" -eq "${#GPUS[@]}" ]]; then
            wait_batch || COMP_STATUS=1
        fi
    done
    [[ "$BATCH_COUNT" -eq 0 ]] || wait_batch || COMP_STATUS=1
    [[ "$COMP_STATUS" -eq 0 ]] || {
        echo "[error] at least one comprehensive metric job failed"
        exit 1
    }
    PARTS=()
    for method in "${METHODS[@]}"; do
        PARTS+=("$METRICS/comprehensive_parts/$method.json")
    done
    python "$ROOT/scripts/merge_comprehensive_results.py" \
        "${PARTS[@]}" --output "$METRICS/comprehensive.json" \
        --expected-methods "${METHODS[@]}" --expected-videos "$EXPECTED" \
        >"$METRICS/logs/merge_comprehensive.log" 2>&1 || exit 1
fi

if [[ "$RUN_TEMPORAL" == "1" ]]; then
    python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
        "${VIDEO_DIRS[@]}" --frame-step "$TEMPORAL_FRAME_STEP" \
        --expected-videos "$EXPECTED" \
        --output "$METRICS/temporal_jump.csv" \
        >"$METRICS/logs/temporal_jump.log" 2>&1 || exit 1
fi

for pair in \
    "$RUN_VBENCH:$METRICS/vbench_long_summary.json" \
    "$RUN_COMPREHENSIVE:$METRICS/comprehensive.json" \
    "$RUN_TEMPORAL:$METRICS/temporal_jump.csv"; do
    enabled="${pair%%:*}"
    path="${pair#*:}"
    if [[ "$enabled" == "1" && ! -s "$path" ]]; then
        echo "[error] expected metric output missing: $path"
        exit 2
    fi
done

echo "[v100-postprocess] metrics complete: $METRICS"
