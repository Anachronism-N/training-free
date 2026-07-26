#!/usr/bin/env bash
# Audit v101, prepare blind review, and run broad metrics.
#
# Usage:
#   bash scripts/postprocess_v101_paper_ablation.sh prepare
#   HUMAN_REVIEW_DONE=1 bash scripts/postprocess_v101_paper_ablation.sh metrics
set -uo pipefail

PHASE="${1:-prepare}"
[[ "$PHASE" == "prepare" || "$PHASE" == "metrics" ]] || {
    echo "usage: $0 prepare|metrics"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v101_paper_ablation_128}"
CONTRACT="${CONTRACT:-$RUN_ROOT/contracts/experiment.json}"
PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
HUMAN_REVIEW_DONE="${HUMAN_REVIEW_DONE:-0}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_COMPREHENSIVE="${RUN_COMPREHENSIVE:-1}"
RUN_TEMPORAL="${RUN_TEMPORAL:-1}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-64}"
TEMPORAL_FRAME_STEP="${TEMPORAL_FRAME_STEP:-2}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree}"

for path in "$ROOT" "$PF" "$RUN_ROOT" "$CONTRACT" "$PROMPTS"; do
    [[ -e "$path" ]] || {
        echo "[error] missing required path: $path"
        exit 2
    }
done
for pair in \
    "HUMAN_REVIEW_DONE=$HUMAN_REVIEW_DONE" \
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

mapfile -t METHODS < <(
    python - "$CONTRACT" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("version") != 1:
    raise SystemExit("unexpected v101 contract version")
if payload.get("experiment") != "v101_paper_ablation_128":
    raise SystemExit("unexpected v101 experiment name")
methods = payload.get("methods") or []
if len(methods) != 8:
    raise SystemExit(f"expected 8 methods, found {len(methods)}")
for method in methods:
    print(method["name"])
PY
)
[[ "${#METHODS[@]}" -eq 8 ]] || {
    echo "[error] failed to load eight methods from contract"
    exit 2
}

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 8 ]] || {
    echo "[error] v101 postprocess requires exactly 8 GPU ids"
    exit 2
}
[[ "${#GPUS[@]}" -eq "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" ]] || {
    echo "[error] GPU_LIST contains duplicate ids"
    exit 2
}

METRICS="$RUN_ROOT/metrics"
mkdir -p \
    "$METRICS/logs" "$METRICS/video_audits" \
    "$METRICS/vbench_long" "$METRICS/comprehensive" \
    "$METRICS/temporal"

python - "$CONTRACT" "$RUN_ROOT" "$PROMPTS" \
    "$METRICS/workflow_contract_audit.json" <<'PY'
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


contract_path, run_root, prompts, output = map(Path, sys.argv[1:])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract.get("experiment") != "v101_paper_ablation_128":
    raise ValueError("unexpected experiment")
if contract.get("prompts", {}).get("sha256") != sha256(prompts):
    raise ValueError("prompt hash differs from generation contract")
if int(contract.get("prompts", {}).get("count", -1)) != 128:
    raise ValueError("generation contract is not MovieGenVideoBench-128")
if contract.get("shards") != [[0, 32], [32, 64], [64, 96], [96, 128]]:
    raise ValueError("unexpected prompt shard contract")

method_items = {
    item["name"]: item for item in contract.get("methods", [])
}
methods = list(method_items)
evidence = []
for method in methods:
    if not (run_root / "videos" / method).is_dir():
        raise ValueError(f"missing video directory for {method}")
    for shard in range(4):
        required = [
            f"status/{method}.shard{shard}.done.json",
            f"configs/{method}.shard{shard}.json",
            f"diagnostics/{method}.shard{shard}.video.json",
            f"diagnostics/{method}.shard{shard}.policy.json",
            f"traces/{method}.shard{shard}.policy.jsonl",
            f"logs/{method}.shard{shard}.log",
        ]
        item = method_items[method]
        if item.get("suppress_policy") in {"motion", "motion_cyclic"}:
            required.extend(
                [
                    f"diagnostics/{method}.shard{shard}.motion.json",
                    f"traces/{method}.shard{shard}.motion.jsonl",
                ]
            )
        if bool(item.get("transition")):
            required.append(
                f"traces/{method}.shard{shard}.transition.jsonl"
            )
        for relative in required:
            path = run_root / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing/empty evidence: {path}")
            evidence.append({"path": str(path), "sha256": sha256(path)})

payload = {
    "version": 1,
    "experiment": contract["experiment"],
    "contract_sha256": sha256(contract_path),
    "prompt_sha256": sha256(prompts),
    "methods": methods,
    "evidence": evidence,
    "ok": True,
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
[[ "$?" -eq 0 ]] || {
    echo "[error] workflow contract audit failed"
    exit 1
}

for method in "${METHODS[@]}"; do
    report="$METRICS/video_audits/$method.json"
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$RUN_ROOT/videos/$method" \
        --start-idx 0 --end-idx 128 \
        --expected-frames 477 --expected-fps 16 \
        --expected-width 832 --expected-height 480 \
        --fps-tolerance .05 --output-json "$report" \
        >"$METRICS/logs/audit_${method}.log" 2>&1 || {
            echo "[error] full video audit failed for $method"
            exit 1
        }
done

BLIND_REVIEW="$RUN_ROOT/blind_review"
BLIND_PRIVATE="$RUN_ROOT/blind_review_private"
if [[ ! -f "$BLIND_REVIEW/.complete.json" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        --run-root "$RUN_ROOT/videos" --methods "${METHODS[@]}" \
        --prompts "$PROMPTS" --prompt-count 128 \
        --output "$BLIND_REVIEW" --private-output "$BLIND_PRIVATE" \
        >"$METRICS/logs/prepare_blind_review.log" 2>&1 || exit 1
fi

{
    printf 'PHASE=%s\n' "$PHASE"
    printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'VBENCH_DIMS=%s\n' "$VBENCH_DIMS"
    printf 'SAMPLE_FRAMES=%s\n' "$SAMPLE_FRAMES"
    printf 'TEMPORAL_FRAME_STEP=%s\n' "$TEMPORAL_FRAME_STEP"
} >"$METRICS/metric_manifest.env"

if [[ "$PHASE" == "prepare" ]]; then
    echo "[complete] blind review prepared at $BLIND_REVIEW"
    echo "[next] freeze the blind review, review it, then run metrics with HUMAN_REVIEW_DONE=1"
    exit 0
fi

[[ "$HUMAN_REVIEW_DONE" == "1" ]] || {
    echo "[error] set HUMAN_REVIEW_DONE=1 only after blind review is frozen"
    exit 2
}
python "$ROOT/scripts/prepare_blind_review.py" \
    --run-root "$RUN_ROOT/videos" --methods "${METHODS[@]}" \
    --prompts "$PROMPTS" --prompt-count 128 \
    --output "$BLIND_REVIEW" --private-output "$BLIND_PRIVATE" \
    --verify-frozen \
    >"$METRICS/logs/verify_frozen_blind_review.log" 2>&1 || {
        echo "[error] blind review is not frozen or no longer verifies"
        exit 2
    }

[[ -f "$CONDA_SH" ]] || {
    echo "[error] missing conda activation script: $CONDA_SH"
    exit 2
}
source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"

wait_batch() {
    local status=0
    local pid
    for pid in "${BATCH_PIDS[@]}"; do
        wait "$pid" || status=1
    done
    BATCH_PIDS=()
    return "$status"
}

if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    FULL_JSON="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    [[ -f "$EVAL" && -f "$FULL_JSON" ]] || {
        echo "[error] VBench-Long files are missing under $VBENCH_ROOT"
        exit 2
    }
    read -r -a DIMS <<<"$VBENCH_DIMS"
    BATCH_PIDS=()
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        gpu="${GPUS[$index]}"
        output="$METRICS/vbench_long/$method"
        mkdir -p "$output"
        (
            cd "$VBENCH_ROOT" || exit 2
            CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL" \
                --videos_path "$RUN_ROOT/videos/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" \
                --full_json_dir "$FULL_JSON"
        ) >"$METRICS/logs/vbench_${method}.log" 2>&1 &
        BATCH_PIDS+=("$!")
    done
    wait_batch || {
        echo "[error] one or more VBench-Long jobs failed"
        exit 1
    }
    python "$ROOT/scripts/collect_vbench_long_results.py" \
        --root "$METRICS/vbench_long" \
        --methods "${METHODS[@]}" \
        --dimensions "${DIMS[@]}" \
        --output-json "$METRICS/vbench_long_summary.json" \
        --output-csv "$METRICS/vbench_long_summary.csv" \
        --output-md "$METRICS/vbench_long_summary.md" || exit 1
fi

if [[ "$RUN_COMPREHENSIVE" == "1" ]]; then
    BATCH_PIDS=()
    COMPREHENSIVE_PARTS=()
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        gpu="${GPUS[$index]}"
        output="$METRICS/comprehensive/$method.json"
        COMPREHENSIVE_PARTS+=("$output")
        CUDA_VISIBLE_DEVICES="$gpu" python \
            "$ROOT/scripts/evaluate_comprehensive.py" \
            --video_dirs "$RUN_ROOT/videos/$method" \
            --prompts "$PROMPTS" --output "$output" \
            --gpu 0 --sample_frames "$SAMPLE_FRAMES" --skip_m4 \
            >"$METRICS/logs/comprehensive_${method}.log" 2>&1 &
        BATCH_PIDS+=("$!")
    done
    wait_batch || {
        echo "[error] one or more comprehensive metric jobs failed"
        exit 1
    }
    python "$ROOT/scripts/merge_comprehensive_results.py" \
        "${COMPREHENSIVE_PARTS[@]}" \
        --output "$METRICS/comprehensive_summary.json" \
        --expected-methods "${METHODS[@]}" \
        --expected-videos 128 || exit 1
fi

if [[ "$RUN_TEMPORAL" == "1" ]]; then
    BATCH_PIDS=()
    for method in "${METHODS[@]}"; do
        python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
            "$RUN_ROOT/videos/$method" \
            --output "$METRICS/temporal/$method.json" \
            --expected-videos 128 \
            --frame-step "$TEMPORAL_FRAME_STEP" \
            >"$METRICS/logs/temporal_${method}.log" 2>&1 &
        BATCH_PIDS+=("$!")
    done
    wait_batch || {
        echo "[error] one or more temporal diagnostics failed"
        exit 1
    }
fi

echo "[complete] v101 metrics written under $METRICS"
