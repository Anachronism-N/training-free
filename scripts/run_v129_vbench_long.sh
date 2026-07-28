#!/usr/bin/env bash
# Prompt-correct, dimension-sharded VBench-Long evaluation for v129.
set -uo pipefail

ACTION="${1:-}"
if [[ "$ACTION" != "split" && "$ACTION" != "preflight" && \
      "$ACTION" != "eval" && "$ACTION" != "collect" ]]; then
    echo "usage: bash scripts/run_v129_vbench_long.sh split|preflight|eval|collect"
    exit 2
fi

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$ROOT/runs/v129_paper_comparison_30s}"
MANIFEST="$COMPARISON_ROOT/comparison_manifest.json"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}"
PARTS_ROOT="${PARTS_ROOT:-$COMPARISON_ROOT/metrics/vbench_long_parts}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PROFILE="${V129_METRIC_PROFILE:-core}"

if [[ "$PROFILE" != "core" && "$PROFILE" != "semantic_extension" && \
      "$PROFILE" != "full" ]]; then
    echo "[error] V129_METRIC_PROFILE must be core, semantic_extension, or full"
    exit 2
fi
if (( NUM_NODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "[error] require 0 <= NODE_RANK < NUM_NODES"
    exit 2
fi
[[ -s "$MANIFEST" ]] || {
    echo "[error] missing comparison manifest: $MANIFEST"
    exit 2
}

mapfile -t METHODS < <(
    python - "$MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["experiment"] == "v129_no_pf_paper_comparison_30s"
assert payload["prompt_count"] == 128
assert payload["num_output_frames"] == 120
assert payload["pf_required"] is False
methods = [row["key"] for row in payload["methods"]]
assert len(methods) == len(set(methods)) == 8
assert not any("pf" in key.lower() for key in methods)
print("\n".join(methods))
PY
)
[[ "${#METHODS[@]}" -eq 8 ]] || {
    echo "[error] comparison manifest must contain eight unique no-PF methods"
    exit 2
}

mapfile -t DIMS < <(
    python - "$MANIFEST" "$PROFILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
profiles = payload["metric_profiles"]
profile = sys.argv[2]
dimensions = profiles[profile]
assert len(dimensions) == len(set(dimensions))
if profile == "core":
    assert dimensions == [
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "overall_consistency",
    ]
elif profile == "semantic_extension":
    assert dimensions == [
        "object_class",
        "multiple_objects",
        "human_action",
        "color",
        "spatial_relationship",
        "scene",
        "appearance_style",
        "temporal_style",
    ]
else:
    assert len(dimensions) == 16
print("\n".join(dimensions))
PY
)
[[ "${#DIMS[@]}" -gt 0 ]] || {
    echo "[error] selected metric profile is empty"
    exit 2
}

python - "$MANIFEST" <<'PY' || exit 2
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for row in payload["methods"]:
    video_dir = Path(row["video_dir"])
    expected = {f"{index:06d}-0.mp4" for index in range(128)}
    actual = {path.name for path in video_dir.glob("*.mp4")}
    if actual != expected:
        raise SystemExit(
            f"{row['key']}: invalid video set, "
            f"missing={len(expected-actual)} extra={len(actual-expected)}"
        )
PY

MANIFEST_SHA256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
VBENCH_COMMIT="$(git -C "$VBENCH_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"

if [[ "$ACTION" == "split" ]]; then
    source "$CONDA_SH" || exit 2
    conda activate "$CONDA_ENV" || exit 2
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    python "$ROOT/scripts/prepare_v129_vbench_splits.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --vbench-root "$VBENCH_ROOT" \
        --workers "${V129_SPLIT_WORKERS:-2}" \
        --node-rank "$NODE_RANK" \
        --num-nodes "$NUM_NODES"
    exit $?
fi

python - \
    "$ROOT/scripts" "$MANIFEST" "$MANIFEST_SHA256" "$VBENCH_COMMIT" <<'PY' || exit 2
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from prepare_v129_vbench_splits import validate_split

manifest = json.load(open(sys.argv[2], encoding="utf-8"))
for row in manifest["methods"]:
    result = validate_split(
        Path(row["video_dir"]) / "split_clip",
        comparison_manifest_sha256=sys.argv[3],
        vbench_commit=sys.argv[4],
        prompt_count=128,
        clips_per_video=15,
    )
    if result is None:
        raise SystemExit(
            f"{row['key']}: missing or stale split cache; "
            "run split on all four nodes"
        )
PY

WRAPPER="$ROOT/scripts/eval_vbench_long_prompt_aware.py"
INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
OFFICIAL_MODULE="$VBENCH_ROOT/vbench2_beta_long/__init__.py"
[[ -s "$WRAPPER" ]] || { echo "[error] missing wrapper: $WRAPPER"; exit 2; }
[[ -s "$INFO" ]] || { echo "[error] missing metadata: $INFO"; exit 2; }
[[ -s "$OFFICIAL_MODULE" ]] || {
    echo "[error] missing VBench-Long module: $OFFICIAL_MODULE"
    exit 2
}

RAFT_MODEL="$VBENCH_CACHE_DIR/raft_model/models/raft-things.pth"
AMT_MODEL="$VBENCH_CACHE_DIR/amt_model/amt-s.pth"
MISSING_MODELS=()
[[ -s "$RAFT_MODEL" ]] || MISSING_MODELS+=("$RAFT_MODEL")
[[ -s "$AMT_MODEL" ]] || MISSING_MODELS+=("$AMT_MODEL")
if [[ "${#MISSING_MODELS[@]}" -gt 0 ]]; then
    echo "[error] missing required VBench model files:"
    printf '  %s\n' "${MISSING_MODELS[@]}"
    exit 2
fi

RAFT_SHA256="$(sha256sum "$RAFT_MODEL" | awk '{print $1}')"
AMT_SHA256="$(sha256sum "$AMT_MODEL" | awk '{print $1}')"
WRAPPER_SHA256="$(sha256sum "$WRAPPER" | awk '{print $1}')"
OFFICIAL_MODULE_SHA256="$(sha256sum "$OFFICIAL_MODULE" | awk '{print $1}')"

if [[ "$ACTION" == "collect" ]]; then
    if [[ "$NODE_RANK" != "0" ]]; then
        echo "[error] collect must run only on NODE_RANK=0"
        exit 2
    fi
    python "$ROOT/scripts/merge_v129_vbench_long_parts.py" \
        --comparison-root "$COMPARISON_ROOT" \
        --profile "$PROFILE" \
        --vbench-root "$VBENCH_ROOT"
    exit $?
fi

if [[ "$ACTION" == "preflight" ]]; then
    echo "[v129-vbench-preflight] profile=$PROFILE"
    echo "[v129-vbench-preflight] methods=${#METHODS[@]} dimensions=${#DIMS[@]}"
    echo "[v129-vbench-preflight] jobs=$((${#METHODS[@]} * ${#DIMS[@]}))"
    echo "[v129-vbench-preflight] manifest_sha256=$MANIFEST_SHA256"
    echo "[v129-vbench-preflight] vbench_commit=$VBENCH_COMMIT"
    echo "[v129-vbench-preflight] prompt_mapping=comparison_manifest_exact"
    echo "[v129-vbench-preflight] wrapper_sha256=$WRAPPER_SHA256"
    echo "[v129-vbench-preflight] raft_sha256=$RAFT_SHA256"
    echo "[v129-vbench-preflight] amt_sha256=$AMT_SHA256"
    exit 0
fi

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VBENCH_CACHE_DIR
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -gt 0 ]] || {
    echo "[error] GPU_LIST must contain at least one GPU id"
    exit 2
}
for gpu in "${GPUS[@]}"; do
    [[ -n "${gpu//[[:space:]]/}" ]] || {
        echo "[error] GPU_LIST contains an empty GPU id"
        exit 2
    }
done
mkdir -p "$PARTS_ROOT"

ALL_JOBS=()
for dimension in "${DIMS[@]}"; do
    for method in "${METHODS[@]}"; do
        ALL_JOBS+=("$method|$dimension")
    done
done
LOCAL_JOBS=()
for index in "${!ALL_JOBS[@]}"; do
    if (( index % NUM_NODES == NODE_RANK )); then
        LOCAL_JOBS+=("${ALL_JOBS[$index]}")
    fi
done

write_contract() {
    local contract="$1" method="$2" dimension="$3" video_dir="$4"
    local split_contract="$video_dir/split_clip/.v129_split_manifest.json"
    local split_sha256
    split_sha256="$(sha256sum "$split_contract" | awk '{print $1}')" || return 1
    python - \
        "$contract" "$MANIFEST_SHA256" "$method" "$dimension" \
        "$video_dir" "$VBENCH_COMMIT" "$VBENCH_CACHE_DIR" \
        "$RAFT_SHA256" "$AMT_SHA256" "$WRAPPER_SHA256" \
        "$OFFICIAL_MODULE_SHA256" "$split_sha256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "version": 1,
    "comparison_manifest_sha256": sys.argv[2],
    "method": sys.argv[3],
    "dimension": sys.argv[4],
    "video_dir": str(Path(sys.argv[5]).resolve()),
    "vbench_commit": sys.argv[6],
    "vbench_cache_dir": str(Path(sys.argv[7]).resolve()),
    "raft_sha256": sys.argv[8],
    "amt_sha256": sys.argv[9],
    "wrapper_sha256": sys.argv[10],
    "official_module_sha256": sys.argv[11],
    "split_manifest_sha256": sys.argv[12],
    "prompt_mapping": "comparison_manifest_exact",
    "mode": "long_custom_input",
    "dev_flag": True,
    "num_of_samples_per_prompt": 1,
}
encoded = (
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
).encode()
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    if path.read_bytes() != encoded:
        raise SystemExit(f"frozen job contract differs: {path}")
else:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
PY
}

normalize_result() {
    local output="$1" dimension="$2"
    python - "$output" "$dimension" <<'PY'
import json
import math
import os
import re
import sys
from pathlib import Path

output = Path(sys.argv[1])
dimension = sys.argv[2]
target = output / "results.json"

def score(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        for key in ("score", "overall", "mean", "average", "total_score"):
            if key in value:
                found = score(value[key])
                if found is not None:
                    return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = score(item)
            if found is not None:
                return found
    return None

pattern = re.compile(r"^(\d+)-(\d+)(?:_|$)")

def prompt_indices(value):
    indices = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"video_path", "video", "path"} and isinstance(
                item, str
            ):
                for part in reversed(Path(item).parts):
                    match = pattern.match(Path(part).stem)
                    if match:
                        if int(match.group(2)) != 0:
                            raise ValueError(
                                f"unexpected sample index in {item}"
                            )
                        indices.add(int(match.group(1)))
                        break
            else:
                indices.update(prompt_indices(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            indices.update(prompt_indices(item))
    return indices

def valid(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or dimension not in payload:
        return None
    if score(payload[dimension]) is None:
        return None
    if prompt_indices(payload[dimension]) != set(range(128)):
        return None
    return payload

payload = valid(target) if target.is_file() else None
if payload is None:
    candidates = sorted(
        output.glob("*_eval_results.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        payload = valid(candidate)
        if payload is not None:
            encoded = (
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            )
            temporary = target.with_name(
                f".{target.name}.tmp.{os.getpid()}"
            )
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, target)
            break
if payload is None:
    raise SystemExit(
        f"no finite prompt-complete {dimension} result under {output}"
    )
PY
}

validate_prompt_mapping() {
    local mapping="$1"
    python - "$mapping" "$MANIFEST_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
full_info = Path(payload["full_info"])
digest = hashlib.sha256(full_info.read_bytes()).hexdigest()
checks = {
    "comparison_manifest_sha256": sys.argv[2],
    "prompt_mapping": "comparison_manifest_exact",
    "prompt_count": 128,
    "mapped_count": 128,
    "indices": list(range(128)),
    "full_info_sha256": digest,
}
failures = {
    key: {"actual": payload.get(key), "expected": value}
    for key, value in checks.items()
    if payload.get(key) != value
}
if failures:
    raise SystemExit(f"invalid prompt mapping artifact: {failures}")
PY
}

write_done_marker() {
    local marker="$1" result="$2" contract="$3" mapping="$4"
    local method="$5" dimension="$6"
    local result_sha256 contract_sha256 mapping_sha256
    result_sha256="$(sha256sum "$result" | awk '{print $1}')"
    contract_sha256="$(sha256sum "$contract" | awk '{print $1}')"
    mapping_sha256="$(sha256sum "$mapping" | awk '{print $1}')"
    python - \
        "$marker" "$MANIFEST_SHA256" "$method" "$dimension" \
        "$result_sha256" "$contract_sha256" "$mapping_sha256" \
        "$VBENCH_COMMIT" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "version": 1,
    "comparison_manifest_sha256": sys.argv[2],
    "method": sys.argv[3],
    "dimension": sys.argv[4],
    "result_sha256": sys.argv[5],
    "job_contract_sha256": sys.argv[6],
    "prompt_mapping_sha256": sys.argv[7],
    "vbench_commit": sys.argv[8],
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

done_marker_valid() {
    local marker="$1" result="$2" contract="$3" mapping="$4"
    local method="$5" dimension="$6"
    python - \
        "$marker" "$result" "$contract" "$mapping" "$MANIFEST_SHA256" \
        "$method" "$dimension" "$VBENCH_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
contract_path = Path(sys.argv[3])
mapping_path = Path(sys.argv[4])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {
    "comparison_manifest_sha256": sys.argv[5],
    "method": sys.argv[6],
    "dimension": sys.argv[7],
    "result_sha256": sha256(result_path),
    "job_contract_sha256": sha256(contract_path),
    "prompt_mapping_sha256": sha256(mapping_path),
    "vbench_commit": sys.argv[8],
}
raise SystemExit(
    0 if all(marker.get(key) == value for key, value in expected.items())
    else 1
)
PY
}

prepare_cat_directory() {
    local video_dir="$1" dimension="$2"
    local dirname=""
    if [[ "$dimension" == "subject_consistency" ]]; then
        dirname="subject_consistency_cat_firstframes_videos"
    elif [[ "$dimension" == "background_consistency" ]]; then
        dirname="background_consistency_cat_firstframes_videos"
    else
        return 0
    fi
    python - "$video_dir" "$dirname" <<'PY'
import shutil
import sys
from pathlib import Path

video_dir = Path(sys.argv[1]).resolve()
target = (video_dir / sys.argv[2]).resolve()
if target.parent != video_dir:
    raise SystemExit(f"unsafe cat-video directory: {target}")
if not target.exists():
    raise SystemExit(0)
expected = {f"{index:06d}-0.mp4" for index in range(128)}
observed = {path.name for path in target.glob("*.mp4")}
valid = observed == expected and all(
    path.stat().st_size > 0 for path in target.glob("*.mp4")
)
if valid:
    raise SystemExit(0)
print(f"[repair] removing incomplete VBench cat-video directory: {target}")
shutil.rmtree(target)
PY
}

run_job() {
    local gpu="$1" job="$2"
    local method="${job%%|*}"
    local dimension="${job#*|}"
    local video_dir="$COMPARISON_ROOT/published/$method"
    local output="$PARTS_ROOT/$method/$dimension"
    local result="$output/results.json"
    local marker="$output/done.json"
    local contract="$output/job_contract.json"
    local mapping="$output/prompt_mapping.json"
    local log="$output/run.log"
    mkdir -p "$output"
    write_contract "$contract" "$method" "$dimension" "$video_dir" || return 1

    if [[ -s "$result" && -s "$marker" && -s "$mapping" ]]; then
        if normalize_result "$output" "$dimension" && \
           validate_prompt_mapping "$mapping" && \
           done_marker_valid \
               "$marker" "$result" "$contract" "$mapping" \
               "$method" "$dimension"; then
            echo "[skip] gpu=$gpu method=$method dimension=$dimension"
            return 0
        fi
    fi

    rm -f "$result" "$marker" "$mapping"
    prepare_cat_directory "$video_dir" "$dimension" || return 1
    echo "[launch] gpu=$gpu method=$method dimension=$dimension"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python "$WRAPPER" \
            --vbench-root "$VBENCH_ROOT" \
            --comparison-manifest "$MANIFEST" \
            --videos_path "$video_dir" \
            --dimension "$dimension" \
            --output_path "$output" \
            --full_json_dir "$INFO" \
            --num_of_samples_per_prompt 1 \
            --dev_flag
    ) >"$log" 2>&1 || return 1
    if grep -Eqi \
        'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|FileNotFoundError' \
        "$log"; then
        echo "[error] failure signature in $log"
        return 1
    fi
    normalize_result "$output" "$dimension" || return 1
    validate_prompt_mapping "$mapping" || return 1
    write_done_marker \
        "$marker" "$result" "$contract" "$mapping" \
        "$method" "$dimension" || return 1
    echo "[done] gpu=$gpu method=$method dimension=$dimension"
}

run_worker() {
    local slot="$1" gpu="$2"
    local status=0
    local index
    for ((index=slot; index<${#LOCAL_JOBS[@]}; index+=${#GPUS[@]})); do
        run_job "$gpu" "${LOCAL_JOBS[$index]}" || {
            echo "[failed] gpu=$gpu job=${LOCAL_JOBS[$index]}"
            status=1
        }
    done
    return "$status"
}

echo "[v129-vbench] profile=$PROFILE node=$NODE_RANK/$NUM_NODES jobs=${#LOCAL_JOBS[@]} gpus=${#GPUS[@]}"
PIDS=()
for slot in "${!GPUS[@]}"; do
    if (( slot >= ${#LOCAL_JOBS[@]} )); then
        break
    fi
    gpu="${GPUS[$slot]//[[:space:]]/}"
    run_worker "$slot" "$gpu" &
    PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done
echo "[complete] node=$NODE_RANK profile=$PROFILE jobs=${#LOCAL_JOBS[@]} status=$STATUS"
exit "$STATUS"
