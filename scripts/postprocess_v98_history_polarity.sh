#!/usr/bin/env bash
# Verify generation, freeze blind review, then run fingerprinted metrics.
set -uo pipefail

MODE="${1:-screen32}"
[[ "$MODE" == "screen32" || "$MODE" == "main128" ]] || {
    echo "usage: $0 screen32|main128"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/../research_sprint/bench_baselines/VBench}"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_COMPREHENSIVE="${RUN_COMPREHENSIVE:-1}"
RUN_TEMPORAL="${RUN_TEMPORAL:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-}"
FORCE_METRICS="${FORCE_METRICS:-0}"
FORCE_BLIND="${FORCE_BLIND:-0}"
FOLLOWUP_V78="${FOLLOWUP_V78:-0}"
SAMPLE_FRAMES=64
TEMPORAL_FRAME_STEP=2
VBENCH_DIMS="subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree"
VBENCH_EXPECTED_COMMIT="${VBENCH_EXPECTED_COMMIT:-}"
VBENCH_ALLOW_DIRTY="${VBENCH_ALLOW_DIRTY:-0}"

for pair in \
    "RUN_VBENCH=$RUN_VBENCH" \
    "RUN_COMPREHENSIVE=$RUN_COMPREHENSIVE" \
    "RUN_TEMPORAL=$RUN_TEMPORAL" \
    "FORCE_METRICS=$FORCE_METRICS" \
    "FORCE_BLIND=$FORCE_BLIND" \
    "FOLLOWUP_V78=$FOLLOWUP_V78" \
    "VBENCH_ALLOW_DIRTY=$VBENCH_ALLOW_DIRTY"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    [[ "$value" == "0" || "$value" == "1" ]] || {
        echo "[error] $key must be 0 or 1"
        exit 2
    }
done

if [[ "$MODE" == "screen32" ]]; then
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
    EXPECTED=32
    PRIMARY_RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v98_history_polarity_screen32_corrected}"
else
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
    EXPECTED=128
    PRIMARY_RUN_ROOT="${RUN_ROOT:-$ROOT/runs/v98_history_polarity_main128_corrected}"
fi

METHODS=(
    sf_native
    pf_native
    pf_explicit_parity
    pf_aw_hybrid_merge
    history_polarity_hybrid_merge
    history_polarity_stride_merge
    history_polarity_zero_random_hybrid_merge
    positive_rate_half_hybrid_merge
)
PHASE=primary
RUN_ROOT="$PRIMARY_RUN_ROOT"
if [[ "$FOLLOWUP_V78" == "1" ]]; then
    PHASE=followup_v78
    RUN_ROOT="$PRIMARY_RUN_ROOT/followup_v78"
    METHODS=(
        followup_history_polarity_hybrid_merge_base
        followup_history_polarity_hybrid_merge_v78
    )
fi
if [[ -z "$RUN_ANALYSIS" ]]; then
    if [[ "$PHASE" == "primary" ]]; then RUN_ANALYSIS=1; else RUN_ANALYSIS=0; fi
fi
[[ "$RUN_ANALYSIS" == "0" || "$RUN_ANALYSIS" == "1" ]] || {
    echo "[error] RUN_ANALYSIS must be 0 or 1"
    exit 2
}
if [[ "$PHASE" != "primary" && "$RUN_ANALYSIS" == "1" ]]; then
    echo "[error] v78 follow-up cannot feed the primary go/no-go analyzer"
    exit 2
fi
if [[ "$RUN_ANALYSIS" == "1" ]] && {
    [[ "$RUN_VBENCH" != "1" ]] ||
    [[ "$RUN_COMPREHENSIVE" != "1" ]] ||
    [[ "$RUN_TEMPORAL" != "1" ]]
}; then
    echo "[error] RUN_ANALYSIS=1 requires all three metric stages"
    exit 2
fi

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 8 ]] || {
    echo "[error] v98 postprocess requires exactly eight local GPU ids"
    exit 2
}
[[ "${#GPUS[@]}" -eq "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" ]] || {
    echo "[error] GPU_LIST contains duplicate local ids"
    exit 2
}
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || {
        echo "[error] invalid GPU id $gpu"
        exit 2
    }
done
for path in "$CONDA_SH" "$PROMPTS" "$RUN_ROOT"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

POSTPROCESS_RUN_LOCK="$RUN_ROOT/.postprocess_run_lock"
if ! mkdir "$POSTPROCESS_RUN_LOCK" 2>/dev/null; then
    echo "[error] postprocess is already running, or a stale lock exists: $POSTPROCESS_RUN_LOCK"
    echo "[error] remove a stale lock only after confirming no postprocess owns this run root"
    exit 2
fi
printf 'pid=%s\nhost=%s\nstarted_utc=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$POSTPROCESS_RUN_LOCK/owner.env"
cleanup_postprocess_lock() {
    rm -f "$POSTPROCESS_RUN_LOCK/owner.env"
    rmdir "$POSTPROCESS_RUN_LOCK" 2>/dev/null || true
}
trap cleanup_postprocess_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$ROOT/scripts:${PYTHONPATH:-}"

GLOBAL_MANIFEST="$RUN_ROOT/experiment_manifest.env"
EXPERIMENT_CONTRACT="$RUN_ROOT/experiment_contract.json"
for path in "$GLOBAL_MANIFEST" "$EXPERIMENT_CONTRACT"; do
    [[ -s "$path" ]] || { echo "[error] missing generation contract $path"; exit 2; }
done

METRICS="$RUN_ROOT/metrics"
mkdir -p \
    "$METRICS/logs" "$METRICS/comprehensive_parts" \
    "$METRICS/vbench_long" "$METRICS/status" \
    "$METRICS/video_audits" "$METRICS/eval_inputs"

# Validate the global JSON/env pair, every node manifest, every shard config,
# every completion marker, every frozen input hash, and the exact label map
# selected by each method.  This is independent of the runtime trace audit.
python - \
    "$RUN_ROOT" "$GLOBAL_MANIFEST" "$EXPERIMENT_CONTRACT" \
    "$PROMPTS" "$MODE" "$PHASE" "$EXPECTED" \
    "$METRICS/workflow_contract_audit.json" "${METHODS[@]}" \
    >"$METRICS/logs/workflow_contract_audit.log" 2>&1 <<'PY'
import csv
import hashlib
import json
import pathlib
import sys


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_env(path):
    result = {}
    for line_number, raw in enumerate(
        pathlib.Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = line.split("=", 1)
        if key in result:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        result[key] = value
    return result


def require(actual, expected, message):
    if actual != expected:
        raise ValueError(f"{message}: expected={expected!r} actual={actual!r}")


def canonical(path):
    return str(pathlib.Path(path).resolve())


(
    run_root,
    global_manifest_path,
    contract_path,
    prompts,
    mode,
    phase,
    raw_expected,
    output_path,
    *expected_methods,
) = sys.argv[1:]
run_root = pathlib.Path(run_root).resolve()
global_manifest_path = pathlib.Path(global_manifest_path).resolve()
contract_path = pathlib.Path(contract_path).resolve()
prompts = pathlib.Path(prompts).resolve()
expected = int(raw_expected)
global_manifest = load_env(global_manifest_path)
contract = json.loads(contract_path.read_text(encoding="utf-8"))
global_sha = digest(global_manifest_path)
contract_sha = digest(contract_path)
fingerprint_payload = dict(contract)
run_fingerprint = fingerprint_payload.pop("run_fingerprint", None)
computed_run_fingerprint = hashlib.sha256(
    json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

require(contract.get("version"), 2, "experiment contract version")
require(contract.get("experiment"), "v98_history_polarity", "experiment")
require(contract.get("mode"), mode, "contract mode")
require(contract.get("phase"), phase, "contract phase")
require(contract.get("tracked_worktree_dirty"), False, "tracked worktree")
require(run_fingerprint, computed_run_fingerprint, "run fingerprint")
require(contract.get("frames"), contract["video"]["latent_frames"], "latent frames")
require(contract.get("seed"), contract["prompt"]["seed"], "seed")
require(contract.get("frames"), 120, "v98 latent-frame protocol")
require(contract.get("seed"), 0, "v98 seed protocol")
require(contract["video"]["decoded_frames"], 477, "decoded-frame protocol")
require(contract["video"]["fps"], 16.0, "fps protocol")
require(contract["video"]["width"], 832, "width protocol")
require(contract["video"]["height"], 480, "height protocol")
require(contract["video"]["fps_tolerance"], 0.01, "fps tolerance protocol")
require(contract.get("shards"), 4, "shard count")
require(contract.get("few_step_cfg_enabled"), False, "few-step CFG")
require(contract["sharding"]["method_index_expression"], "local_slot", "method mapping")
require(contract["sharding"]["shard_expression"], "NODE_RANK", "shard mapping")
expected_gpu_mapping = (
    "((local_slot + [0,2,5,7][NODE_RANK]) % 8)"
    if phase == "primary"
    else "2*floor(NODE_RANK/2)+((local_slot+NODE_RANK)%2)"
)
require(
    contract["sharding"].get("gpu_slot_expression"),
    expected_gpu_mapping,
    "GPU-slot mapping",
)
require(contract["prompt"]["path"], canonical(prompts), "prompt path")
require(contract["prompt"]["sha256"], digest(prompts), "prompt hash")
require(contract["prompt"]["count"], expected, "prompt count")

require(global_manifest.get("CONTRACT_VERSION"), "3", "env contract version")
require(global_manifest.get("MODE"), mode, "env mode")
require(global_manifest.get("PHASE"), phase, "env phase")
require(global_manifest.get("EXPERIMENT_CONTRACT_SHA256"), contract_sha, "JSON hash")
require(
    canonical(global_manifest.get("EXPERIMENT_CONTRACT_JSON", "")),
    canonical(contract_path),
    "JSON path",
)
require(global_manifest.get("METHODS", "").split(), expected_methods, "method order")
require(global_manifest.get("PROMPT_SHA256"), digest(prompts), "env prompt hash")
require(global_manifest.get("PROMPT_COUNT"), str(expected), "env prompt count")
require(global_manifest.get("MAPPING"), "method_index=local_slot;shard=NODE_RANK", "mapping")
require(
    global_manifest.get("GPU_SLOT_MAPPING"),
    expected_gpu_mapping,
    "env GPU-slot mapping",
)
require(global_manifest.get("FEW_STEP_CFG_ENABLED"), "0", "env CFG")
require(
    global_manifest.get("PRELOAD_PYRAMIDKV"),
    "1" if contract["runtime"]["preload_pyramidkv_extension"] else "0",
    "extension preload",
)
trace_contract = contract["runtime"]["policy_trace"]
require(
    global_manifest.get("POLICY_TRACE_LAYERS"),
    ",".join(str(value) for value in trace_contract["layers"]),
    "policy trace layers",
)
require(
    global_manifest.get("POLICY_TRACE_STRIDE"),
    str(trace_contract["stride"]),
    "policy trace stride",
)
require(
    global_manifest.get("POLICY_TRACE_MAX_RECORDS"),
    str(trace_contract["max_records"]),
    "policy trace max records",
)

for item in contract["inputs"].values():
    path = pathlib.Path(item["path"])
    if not path.is_file():
        raise ValueError(f"frozen generation input is missing: {path}")
    require(digest(path), item["sha256"], f"frozen input hash {path}")
score = contract["score"]
for path_key, hash_key in (
    ("artifact_path", "artifact_sha256"),
    ("csv_path", "csv_sha256"),
    ("map_manifest_path", "map_manifest_sha256"),
):
    path = pathlib.Path(score[path_key])
    require(digest(path), score[hash_key], f"score input hash {path}")
score_artifact = json.loads(
    pathlib.Path(score["artifact_path"]).read_text(encoding="utf-8")
)
require(
    score_artifact.get("version"),
    score["artifact_version"],
    "score artifact version",
)
require(
    score_artifact.get("method"),
    score["artifact_method"],
    "score artifact method",
)
require(score_artifact.get("accepted"), True, "score artifact acceptance")
score_definition = score_artifact.get("score_definition", {})
require(
    score_definition.get("primary_field"),
    score["primary_field"],
    "score primary field",
)
require(
    score_definition.get("probe_policy_balanced"),
    True,
    "score probe balance",
)
require(
    score_definition.get("probe_policies"),
    ["uniform_stride", "uniform_merge"],
    "score probe policies",
)
require(
    score_definition.get("bootstrap_unit"),
    "counterfactual_prompt_pair",
    "score bootstrap unit",
)

contract_methods = contract.get("methods")
if not isinstance(contract_methods, list):
    raise ValueError("contract methods must be a list")
require([item["name"] for item in contract_methods], expected_methods, "contract methods")
map_manifest_path = pathlib.Path(score["map_manifest_path"])
map_manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
require(map_manifest.get("version"), 2, "map manifest version")
require(
    map_manifest.get("method"),
    "v98_middle_relative_history_map_builder",
    "map manifest method",
)
require(
    map_manifest.get("claims", {}).get("probe_policy_balanced"),
    True,
    "map manifest probe balance",
)
map_entries = map_manifest.get("maps", {})
method_by_name = {}
for method_index, item in enumerate(contract_methods):
    require(item.get("method_index"), method_index, "method index")
    require(item.get("few_step_cfg_enabled"), False, f"{item['name']} CFG")
    expected_branches = [] if item["engine"] == "sf" else ["cond"]
    require(
        item.get("policy_trace_branches"),
        expected_branches,
        f"{item['name']} trace branches",
    )
    transition = item.get("transition")
    if not isinstance(transition, dict):
        raise ValueError(f"{item['name']} has no transition contract")
    require(
        transition.get("branches"),
        ["cond"] if transition.get("enabled") else [],
        f"{item['name']} transition branches",
    )
    map_key = item.get("map_key")
    if map_key is None:
        require(item.get("map_path"), None, f"{item['name']} map path")
        require(item.get("map_sha256"), None, f"{item['name']} map hash")
    elif map_key == "pf_labels":
        expected_path = pathlib.Path(global_manifest["PF_LABELS"]).resolve()
        expected_hash = global_manifest["PF_LABELS_SHA256"]
        require(canonical(item["map_path"]), canonical(expected_path), "PF label path")
        require(item["map_sha256"], expected_hash, "PF label hash")
    else:
        if map_key not in map_entries:
            raise ValueError(f"{item['name']}: unknown map key {map_key!r}")
        entry = map_entries[map_key]
        expected_path = pathlib.Path(entry["path"])
        if not expected_path.is_absolute():
            expected_path = map_manifest_path.parent / expected_path
        expected_path = expected_path.resolve()
        require(canonical(item["map_path"]), canonical(expected_path), "map path")
        require(item["map_sha256"], entry["sha256"], "map hash")
    if item.get("map_path"):
        map_path = pathlib.Path(item["map_path"])
        require(digest(map_path), item["map_sha256"], f"{item['name']} map file")
        with map_path.open(encoding="utf-8", newline="") as handle:
            labels = {
                int(value)
                for row in csv.reader(handle)
                for value in row
            }
        require(sorted(labels), item["expected_labels"], f"{item['name']} labels")
        policy_labels = sorted(int(value) for value in item["policies"])
        require(policy_labels, item["expected_labels"], f"{item['name']} policies")
    method_by_name[item["name"]] = item

shard_size = contract["sharding"]["shard_size"]
if shard_size * 4 != expected:
    raise ValueError("contract shard size does not cover the prompt set")
configs = 0
for shard in range(4):
    node_manifest_path = run_root / "nodes" / f"node{shard}.manifest.env"
    node_done_path = run_root / "status" / f"node{shard}.done"
    node_manifest = load_env(node_manifest_path)
    node_done = load_env(node_done_path)
    require(node_manifest.get("NODE_RANK"), str(shard), "node rank")
    require(node_manifest.get("ASSIGNED_SHARD"), str(shard), "assigned shard")
    require(node_manifest.get("GLOBAL_MANIFEST_SHA256"), global_sha, "node global hash")
    require(node_manifest.get("METHODS", "").split(), expected_methods, "node methods")
    require(
        node_manifest.get("LOCAL_METHOD_COUNT"),
        str(len(expected_methods)),
        "local method count",
    )
    require(node_done.get("GLOBAL_MANIFEST_SHA256"), global_sha, "done global hash")
    require(
        node_done.get("NODE_MANIFEST_SHA256"),
        digest(node_manifest_path),
        "done node hash",
    )
    gpus = node_manifest.get("GPU_LIST", "").split(",")
    if len(gpus) != 8 or len(gpus) != len(set(gpus)):
        raise ValueError(f"node {shard}: invalid GPU list")
    for method_index, method in enumerate(expected_methods):
        item = method_by_name[method]
        config_path = run_root / "configs" / f"{method}.shard{shard}.env"
        marker_path = run_root / "status" / f"{method}.shard{shard}.done"
        config = load_env(config_path)
        marker = load_env(marker_path)
        start = shard * shard_size
        end = start + shard_size
        if phase == "primary":
            slot_offset = (0, 2, 5, 7)[shard]
            gpu_slot = (method_index + slot_offset) % 8
        else:
            pair_base = 2 * (shard // 2)
            gpu_slot = pair_base + ((method_index + shard) % 2)
        expected_fields = {
            "contract_version": "3",
            "name": method,
            "phase": phase,
            "mode": mode,
            "node_rank": str(shard),
            "shard": str(shard),
            "start_idx": str(start),
            "end_idx": str(end),
            "gpu": gpus[gpu_slot],
            "engine": item["engine"],
            "labels": item.get("map_path") or "",
            "label_sha256": item.get("map_sha256") or "",
            "route": item["route"],
            "transition": "1" if item["transition"]["enabled"] else "0",
            "global_manifest_sha256": global_sha,
            "experiment_contract_sha256": contract_sha,
            "method_contract_sha256": contract["method_contract_sha256"],
            "run_commit": contract["run_commit"],
            "run_dirty": global_manifest["RUN_DIRTY"],
            "prompt_sha256": contract["prompt"]["sha256"],
            "prompt_count": str(expected),
            "score_sha256": score["csv_sha256"],
            "score_artifact_sha256": score["artifact_sha256"],
            "map_manifest_sha256": score["map_manifest_sha256"],
            "sf_config_sha256": contract["inputs"]["sf_config"]["sha256"],
            "pf_config_sha256": contract["inputs"]["pf_config"]["sha256"],
            "sf_checkpoint_sha256": contract["inputs"]["sf_checkpoint"]["sha256"],
            "pf_checkpoint_sha256": contract["inputs"]["pf_checkpoint"]["sha256"],
            "frames": str(contract["frames"]),
            "expected_video_frames": str(contract["video"]["decoded_frames"]),
            "expected_video_fps": global_manifest["EXPECTED_VIDEO_FPS"],
            "expected_video_width": str(contract["video"]["width"]),
            "expected_video_height": str(contract["video"]["height"]),
            "video_fps_tolerance": global_manifest["VIDEO_FPS_TOLERANCE"],
            "seed": str(contract["seed"]),
            "reseed_per_prompt": "1",
            "few_step_cfg_enabled": "0",
            "policy_trace_layers": global_manifest["POLICY_TRACE_LAYERS"],
            "policy_trace_stride": global_manifest["POLICY_TRACE_STRIDE"],
            "policy_trace_max_records": global_manifest[
                "POLICY_TRACE_MAX_RECORDS"
            ],
            "python_reference_path": "1",
        }
        for key, value in expected_fields.items():
            require(config.get(key), value, f"{config_path}:{key}")
        require(
            marker.get("CELL_CONFIG_SHA256"),
            digest(config_path),
            f"{marker_path}:config hash",
        )
        if not marker.get("VIDEO_INPUT_FINGERPRINT"):
            raise ValueError(f"{marker_path}: missing video fingerprint")
        configs += 1

audit = {
    "ok": True,
    "phase": phase,
    "mode": mode,
    "global_manifest_sha256": global_sha,
    "experiment_contract_sha256": contract_sha,
    "run_fingerprint": contract["run_fingerprint"],
    "methods": expected_methods,
    "configs": configs,
    "shards": 4,
}
output = pathlib.Path(output_path)
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
print(
    f"[WorkflowContractAudit] phase={phase} methods={len(expected_methods)} "
    f"configs={configs} ok=true"
)
PY
if [[ "$?" -ne 0 ]]; then
    cat "$METRICS/logs/workflow_contract_audit.log"
    exit 2
fi

EXPECTED_VIDEO_FRAMES="$(
    python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["video"]["decoded_frames"])' \
        "$EXPERIMENT_CONTRACT"
)" || exit 2
EXPECTED_VIDEO_FPS="$(
    python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["video"]["fps"])' \
        "$EXPERIMENT_CONTRACT"
)" || exit 2
EXPECTED_VIDEO_WIDTH="$(
    python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["video"]["width"])' \
        "$EXPERIMENT_CONTRACT"
)" || exit 2
EXPECTED_VIDEO_HEIGHT="$(
    python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["video"]["height"])' \
        "$EXPERIMENT_CONTRACT"
)" || exit 2
VIDEO_FPS_TOLERANCE="$(
    python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["video"]["fps_tolerance"])' \
        "$EXPERIMENT_CONTRACT"
)" || exit 2
POLICY_TRACE_LAYERS="$(
    python -c \
        'import json,sys; print(",".join(str(v) for v in json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["policy_trace"]["layers"]))' \
        "$EXPERIMENT_CONTRACT"
)" || exit 2

STATUS=0
VIDEO_DIRS=()
VIDEO_AUDITS=()
for method in "${METHODS[@]}"; do
    source_dir="$RUN_ROOT/$method"
    stage_dir="$METRICS/eval_inputs/$method"
    audit_json="$METRICS/video_audits/$method.json"
    VIDEO_DIRS+=("$stage_dir")
    VIDEO_AUDITS+=("$audit_json")
    shard_size=$((EXPECTED / 4))
    for shard in 0 1 2 3; do
        start=$((shard * shard_size))
        end=$((start + shard_size))
        generation_audit="$RUN_ROOT/diagnostics/$method.shard$shard.video.json"
        generation_marker="$RUN_ROOT/status/$method.shard$shard.done"
        python "$ROOT/scripts/audit_indexed_videos.py" \
            --video-dir "$source_dir" --start-idx "$start" --end-idx "$end" \
            --expected-frames "$EXPECTED_VIDEO_FRAMES" \
            --expected-fps "$EXPECTED_VIDEO_FPS" \
            --expected-width "$EXPECTED_VIDEO_WIDTH" \
            --expected-height "$EXPECTED_VIDEO_HEIGHT" \
            --fps-tolerance "$VIDEO_FPS_TOLERANCE" \
            --allow-outside-interval --output-json "$generation_audit" \
            --reuse-valid-report \
            >"$METRICS/logs/$method.shard$shard.video_recheck.log" 2>&1 || {
            echo "[error] generation-time video audit is stale for $method shard $shard"
            STATUS=1
            continue
        }
        current_fingerprint="$(
            python -c \
                'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["input_fingerprint"])' \
                "$generation_audit"
        )" || {
            STATUS=1
            continue
        }
        frozen_fingerprint="$(
            awk -F= '$1=="VIDEO_INPUT_FINGERPRINT"{print $2}' \
                "$generation_marker"
        )"
        if [[ -z "$frozen_fingerprint" || \
              "$current_fingerprint" != "$frozen_fingerprint" ]]; then
            echo "[error] videos changed after generation for $method shard $shard"
            STATUS=1
        fi
    done
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$source_dir" --start-idx 0 --end-idx "$EXPECTED" \
        --expected-frames "$EXPECTED_VIDEO_FRAMES" \
        --expected-fps "$EXPECTED_VIDEO_FPS" \
        --expected-width "$EXPECTED_VIDEO_WIDTH" \
        --expected-height "$EXPECTED_VIDEO_HEIGHT" \
        --fps-tolerance "$VIDEO_FPS_TOLERANCE" \
        --output-json "$audit_json" --reuse-valid-report \
        --stage-dir "$stage_dir" --replace-stage \
        >"$METRICS/logs/$method.video_audit.log" 2>&1 || STATUS=1
    for shard in 0 1 2 3; do
        log="$RUN_ROOT/logs/$method.shard$shard.log"
        [[ -s "$log" ]] || {
            echo "[error] missing generation log $log"
            STATUS=1
            continue
        }
        if grep -Eqi \
            'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|PyramidKVPolicyTraceError' \
            "$log"; then
            echo "[error] failure signature in $log"
            STATUS=1
        fi
    done
done
[[ "$STATUS" -eq 0 ]] || exit "$STATUS"

python "$ROOT/scripts/audit_v98_policy_traces.py" \
    --run-root "$RUN_ROOT" \
    --experiment-contract "$EXPERIMENT_CONTRACT" \
    --expected-layers "$POLICY_TRACE_LAYERS" \
    --output-json "$METRICS/policy_trace_audit.json" \
    --output-md "$METRICS/policy_trace_audit.md" \
    --strict >"$METRICS/logs/policy_trace_audit.log" 2>&1 || {
    cat "$METRICS/logs/policy_trace_audit.log"
    exit 1
}

BLIND_REVIEW="$RUN_ROOT/blind_review"
BLIND_PRIVATE="$RUN_ROOT/blind_review_private"
BLIND_ARGS=(
    --run-root "$RUN_ROOT"
    --methods "${METHODS[@]}"
    --prompts "$PROMPTS"
    --prompt-count "$EXPECTED"
    --output "$BLIND_REVIEW"
    --private-output "$BLIND_PRIVATE"
)
BLIND_CREATED=0
if [[ ! -d "$BLIND_REVIEW" || ! -d "$BLIND_PRIVATE" ]]; then
    if [[ -e "$BLIND_REVIEW" || -e "$BLIND_PRIVATE" ]]; then
        if [[ "$FORCE_BLIND" != "1" ]]; then
            echo "[error] blind public/private package is partial; inspect it or set FORCE_BLIND=1"
            exit 2
        fi
        python "$ROOT/scripts/prepare_blind_review.py" \
            "${BLIND_ARGS[@]}" --force || exit 1
    else
        python "$ROOT/scripts/prepare_blind_review.py" \
            "${BLIND_ARGS[@]}" || exit 1
    fi
    BLIND_CREATED=1
elif ! python "$ROOT/scripts/prepare_blind_review.py" \
    "${BLIND_ARGS[@]}" --verify \
    --output-json "$METRICS/blind_package_verification.json" \
    >"$METRICS/logs/blind_package_verify.log" 2>&1; then
    if [[ "$FORCE_BLIND" != "1" ]]; then
        cat "$METRICS/logs/blind_package_verify.log"
        echo "[error] blind package is stale or partial; inspect it or set FORCE_BLIND=1"
        exit 2
    fi
    python "$ROOT/scripts/prepare_blind_review.py" \
        "${BLIND_ARGS[@]}" --force || exit 1
    BLIND_CREATED=1
fi
if [[ "$BLIND_CREATED" == "1" ]]; then
    python "$ROOT/scripts/prepare_blind_review.py" \
        "${BLIND_ARGS[@]}" --verify \
        --output-json "$METRICS/blind_package_verification.json" \
        >"$METRICS/logs/blind_package_verify.log" 2>&1 || exit 1
fi

METRICS_REQUESTED=0
if [[ "$RUN_VBENCH" == "1" || "$RUN_COMPREHENSIVE" == "1" || "$RUN_TEMPORAL" == "1" ]]; then
    METRICS_REQUESTED=1
fi
if [[ "$METRICS_REQUESTED" == "0" ]]; then
    echo "[v98-postprocess] generation and blind package verified; metrics disabled"
    exit 0
fi

if ! python "$ROOT/scripts/prepare_blind_review.py" \
    "${BLIND_ARGS[@]}" --verify-frozen \
    --output-json "$METRICS/blind_frozen_verification.json" \
    >"$METRICS/logs/blind_frozen_verify.log" 2>&1; then
    cat "$METRICS/logs/blind_frozen_verify.log"
    echo "[gate] automated quality metrics have NOT been run"
    echo "[gate] fill $BLIND_REVIEW/scorecard.csv, then freeze with:"
    printf 'python %q' "$ROOT/scripts/prepare_blind_review.py"
    printf ' %q' "${BLIND_ARGS[@]}"
    printf ' --freeze\n'
    exit 3
fi

if [[ "$PHASE" == "followup_v78" ]]; then
    mapfile -t TRANSITION_TRACES < <(
        find "$RUN_ROOT/traces" -maxdepth 1 -type f \
            -name 'followup_history_polarity_hybrid_merge_v78.shard*.transition.jsonl' |
            sort
    )
    [[ "${#TRANSITION_TRACES[@]}" -eq 4 ]] || {
        echo "[error] expected four follow-up v78 transition traces"
        exit 2
    }
    python "$ROOT/scripts/summarize_cache_transition_trace.py" \
        "${TRANSITION_TRACES[@]}" --strict \
        --output-json "$METRICS/cache_transition_summary.json" \
        --output-md "$METRICS/cache_transition_summary.md" \
        >"$METRICS/logs/cache_transition_summary.log" 2>&1 || exit 1
fi

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

marker_value() {
    local marker="$1" key="$2"
    awk -F= -v key="$key" \
        '$1==key {print substr($0,index($0,"=")+1)}' "$marker"
}

VBENCH_COMMIT=""
VBENCH_DIRTY=0
VBENCH_DIFF_SHA256=""
VBENCH_STATUS_SHA256=""
VBENCH_EVAL_SHA256=""
VBENCH_INFO_SHA256=""
if [[ "$RUN_VBENCH" == "1" ]]; then
    EVAL="$VBENCH_ROOT/vbench2_beta_long/eval_long.py"
    INFO="$VBENCH_ROOT/vbench2_beta_long/VBench_full_info.json"
    for path in "$VBENCH_ROOT" "$EVAL" "$INFO"; do
        [[ -e "$path" ]] || {
            echo "[error] VBench-Long input missing: $path"
            exit 2
        }
    done
    VBENCH_COMMIT="$(git -C "$VBENCH_ROOT" rev-parse --verify HEAD 2>/dev/null)" || {
        echo "[error] VBench repository commit cannot be resolved"
        exit 2
    }
    if [[ -n "$VBENCH_EXPECTED_COMMIT" && "$VBENCH_COMMIT" != "$VBENCH_EXPECTED_COMMIT" ]]; then
        echo "[error] VBench commit mismatch: expected=$VBENCH_EXPECTED_COMMIT actual=$VBENCH_COMMIT"
        exit 2
    fi
    VBENCH_STATUS_TEXT="$(
        git -C "$VBENCH_ROOT" status --porcelain=v1 --untracked-files=no
    )"
    if [[ -n "$VBENCH_STATUS_TEXT" ]]; then VBENCH_DIRTY=1; fi
    if [[ "$VBENCH_DIRTY" == "1" && "$VBENCH_ALLOW_DIRTY" != "1" ]]; then
        echo "[error] VBench worktree is dirty; pin a clean revision or set VBENCH_ALLOW_DIRTY=1"
        exit 2
    fi
    VBENCH_DIFF_SHA256="$(
        git -C "$VBENCH_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}'
    )"
    VBENCH_STATUS_SHA256="$(
        printf '%s' "$VBENCH_STATUS_TEXT" | sha256sum | awk '{print $1}'
    )"
    VBENCH_EVAL_SHA256="$(sha256_file "$EVAL")"
    VBENCH_INFO_SHA256="$(sha256_file "$INFO")"
    VBENCH_LOCK="$METRICS/vbench_version.lock.env"
    VBENCH_LOCK_TMP="$METRICS/.vbench_version.lock.$$.tmp"
    {
        printf 'VBENCH_ROOT=%s\n' "$VBENCH_ROOT"
        printf 'VBENCH_COMMIT=%s\n' "$VBENCH_COMMIT"
        printf 'VBENCH_DIRTY=%s\n' "$VBENCH_DIRTY"
        printf 'VBENCH_DIFF_SHA256=%s\n' "$VBENCH_DIFF_SHA256"
        printf 'VBENCH_STATUS_SHA256=%s\n' "$VBENCH_STATUS_SHA256"
        printf 'VBENCH_EVAL_SHA256=%s\n' "$VBENCH_EVAL_SHA256"
        printf 'VBENCH_INFO_SHA256=%s\n' "$VBENCH_INFO_SHA256"
    } >"$VBENCH_LOCK_TMP"
    if [[ -e "$VBENCH_LOCK" ]] && ! cmp -s "$VBENCH_LOCK_TMP" "$VBENCH_LOCK"; then
        echo "[error] VBench version differs from the frozen metrics lock"
        diff -u "$VBENCH_LOCK" "$VBENCH_LOCK_TMP" || true
        rm -f "$VBENCH_LOCK_TMP"
        exit 2
    fi
    mv "$VBENCH_LOCK_TMP" "$VBENCH_LOCK"
fi

EVALUATOR_FILES=(
    "$ROOT/scripts/postprocess_v98_history_polarity.sh"
    "$ROOT/scripts/audit_indexed_videos.py"
    "$ROOT/scripts/prepare_blind_review.py"
    "$ROOT/scripts/audit_v98_policy_traces.py"
    "$ROOT/scripts/summarize_cache_transition_trace.py"
    "$ROOT/scripts/evaluate_comprehensive.py"
    "$ROOT/scripts/merge_comprehensive_results.py"
    "$ROOT/scripts/compute_temporal_jump_diagnostic.py"
    "$ROOT/scripts/collect_vbench_long_results.py"
    "$ROOT/scripts/analyze_v98_history_polarity.py"
)
for path in "${EVALUATOR_FILES[@]}"; do
    [[ -s "$path" ]] || { echo "[error] missing evaluator $path"; exit 2; }
done

METRIC_MANIFEST="$METRICS/metric_manifest.json"
METRIC_MANIFEST_TMP="$METRICS/.metric_manifest.$$.tmp"
python - \
    "$METRIC_MANIFEST_TMP" "$GLOBAL_MANIFEST" "$EXPERIMENT_CONTRACT" \
    "$METRICS/blind_frozen_verification.json" \
    "$RUN_VBENCH" "$RUN_COMPREHENSIVE" "$RUN_TEMPORAL" "$RUN_ANALYSIS" \
    "$SAMPLE_FRAMES" "$TEMPORAL_FRAME_STEP" "$VBENCH_DIMS" \
    "$VBENCH_COMMIT" "$VBENCH_DIRTY" "$VBENCH_DIFF_SHA256" \
    "$VBENCH_STATUS_SHA256" "$VBENCH_EVAL_SHA256" "$VBENCH_INFO_SHA256" \
    -- "${VIDEO_AUDITS[@]}" -- "${EVALUATOR_FILES[@]}" <<'PY' || exit 2
import hashlib
import json
import pathlib
import sys


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


args = sys.argv[1:]
first_sep = args.index("--")
second_sep = args.index("--", first_sep + 1)
fixed = args[:first_sep]
video_paths = args[first_sep + 1 : second_sep]
evaluator_paths = args[second_sep + 1 :]
(
    output,
    global_manifest,
    experiment_contract,
    blind_verification,
    run_vbench,
    run_comprehensive,
    run_temporal,
    run_analysis,
    sample_frames,
    temporal_step,
    vbench_dims,
    vbench_commit,
    vbench_dirty,
    vbench_diff,
    vbench_status,
    vbench_eval,
    vbench_info,
) = fixed
contract = json.loads(pathlib.Path(experiment_contract).read_text(encoding="utf-8"))
blind = json.loads(pathlib.Path(blind_verification).read_text(encoding="utf-8"))
video_audits = [
    json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for path in video_paths
]
if not all(item.get("ok") for item in video_audits):
    raise SystemExit("cannot fingerprint failed video audits")
payload = {
    "version": 2,
    "generation": {
        "global_manifest_sha256": digest(global_manifest),
        "experiment_contract_sha256": digest(experiment_contract),
        "run_fingerprint": contract["run_fingerprint"],
    },
    "blind": blind,
    "video_inputs": [
        {
            "method": pathlib.Path(item["video_dir"]).name,
            "count": item["found"],
            "input_fingerprint": item["input_fingerprint"],
        }
        for item in video_audits
    ],
    "stages": {
        "vbench": run_vbench == "1",
        "comprehensive": run_comprehensive == "1",
        "temporal": run_temporal == "1",
        "analysis": run_analysis == "1",
    },
    "parameters": {
        "sample_frames": int(sample_frames),
        "temporal_frame_step": int(temporal_step),
        "vbench_dimensions": vbench_dims.split(),
    },
    "vbench": {
        "commit": vbench_commit or None,
        "dirty": vbench_dirty == "1",
        "diff_sha256": vbench_diff or None,
        "status_sha256": vbench_status or None,
        "evaluator_sha256": vbench_eval or None,
        "info_sha256": vbench_info or None,
    },
    "evaluators": {
        str(pathlib.Path(path).resolve()): digest(path)
        for path in evaluator_paths
    },
}
encoded = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
payload["metric_input_fingerprint"] = hashlib.sha256(encoded).hexdigest()
path = pathlib.Path(output)
path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(payload["metric_input_fingerprint"])
PY
METRIC_INPUT_FINGERPRINT="$(
    python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["metric_input_fingerprint"])' \
        "$METRIC_MANIFEST_TMP"
)" || exit 2

RESUME_VALID=0
if [[ "$FORCE_METRICS" != "1" && -s "$METRIC_MANIFEST" ]]; then
    OLD_METRIC_INPUT_FINGERPRINT="$(
        python -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("metric_input_fingerprint",""))' \
            "$METRIC_MANIFEST"
    )" || OLD_METRIC_INPUT_FINGERPRINT=""
    if [[ "$OLD_METRIC_INPUT_FINGERPRINT" == "$METRIC_INPUT_FINGERPRINT" ]]; then
        RESUME_VALID=1
    fi
fi
mv "$METRIC_MANIFEST_TMP" "$METRIC_MANIFEST"
if [[ "$RESUME_VALID" != "1" ]]; then
    rm -f "$METRICS/status/"*.done
fi

if [[ "$RUN_VBENCH" == "1" ]]; then
    read -r -a DIMS <<<"$VBENCH_DIMS"
    PIDS=()
    STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        output="$METRICS/vbench_long/$method"
        marker="$METRICS/status/vbench.$method.done"
        mkdir -p "$output"
        if [[ "$RESUME_VALID" == "1" && -s "$marker" && \
              -s "$output/results.json" && -s "$output/result_contract.json" ]] && \
            grep -qx "METRIC_INPUT_FINGERPRINT=$METRIC_INPUT_FINGERPRINT" "$marker" && \
            [[ "$(marker_value "$marker" RESULTS_SHA256)" == \
               "$(sha256_file "$output/results.json")" ]] && \
            [[ "$(marker_value "$marker" RESULT_CONTRACT_SHA256)" == \
               "$(sha256_file "$output/result_contract.json")" ]]; then
            continue
        fi
        rm -f "$marker" "$output/results.json" "$output/result_contract.json"
        find "$output" -type f \
            \( -name '*_eval_results.json' -o -name '*_full_info.json' \) \
            -delete
        (
            export CUDA_VISIBLE_DEVICES="${GPUS[$index]}"
            cd "$VBENCH_ROOT" || exit 2
            python "$EVAL" \
                --videos_path "$METRICS/eval_inputs/$method" \
                --dimension "${DIMS[@]}" \
                --mode long_custom_input --dev_flag \
                --num_of_samples_per_prompt 1 \
                --output_path "$output" --full_json_dir "$INFO"
        ) >"$output/run.log" 2>&1 &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || STATUS=1
    done
    [[ "$STATUS" -eq 0 ]] || exit 1

    for method in "${METHODS[@]}"; do
        output="$METRICS/vbench_long/$method"
        marker="$METRICS/status/vbench.$method.done"
        if [[ "$RESUME_VALID" == "1" && -s "$marker" && \
              -s "$output/results.json" && -s "$output/result_contract.json" ]] && \
            grep -qx "METRIC_INPUT_FINGERPRINT=$METRIC_INPUT_FINGERPRINT" "$marker" && \
            [[ "$(marker_value "$marker" RESULTS_SHA256)" == \
               "$(sha256_file "$output/results.json")" ]] && \
            [[ "$(marker_value "$marker" RESULT_CONTRACT_SHA256)" == \
               "$(sha256_file "$output/result_contract.json")" ]]; then
            continue
        fi
        python - \
            "$output" "$METRICS/eval_inputs/$method" "$EXPECTED" \
            "$METRIC_INPUT_FINGERPRINT" "${DIMS[@]}" <<'PY' || exit 1
import hashlib
import json
import os
import pathlib
import shutil
import sys


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


output = pathlib.Path(sys.argv[1])
stage = pathlib.Path(sys.argv[2])
expected = int(sys.argv[3])
fingerprint = sys.argv[4]
dims = sys.argv[5:]
stage_manifest_path = stage / ".video_input.json"
stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
videos = sorted(stage.glob("*.mp4"))
if len(videos) != expected or len(stage_manifest.get("videos", [])) != expected:
    raise SystemExit(
        f"VBench input count mismatch: files={len(videos)} "
        f"manifest={len(stage_manifest.get('videos', []))} expected={expected}"
    )
for item in stage_manifest["videos"]:
    staged = stage / (
        f"{int(item['prompt_idx']):06d}-{int(item['sample_idx'])}_"
        f"{item['suffix']}.mp4"
    )
    if not staged.is_file() or staged.stat().st_size != item["size"]:
        raise SystemExit(f"VBench staged input is missing or resized: {staged}")
    if digest(staged) != item["sha256"]:
        raise SystemExit(f"VBench staged input hash mismatch: {staged}")

canonical = output / "results.json"
candidates = sorted(output.rglob("*_eval_results.json"))
if canonical.is_file():
    candidates.append(canonical)
candidates = list(dict.fromkeys(path.resolve() for path in candidates))
if len(candidates) != 1:
    raise SystemExit(
        f"expected exactly one VBench aggregate output, found {candidates}"
    )
source = candidates[0]
payload = json.loads(source.read_text(encoding="utf-8"))
scores = payload.get("results", payload) if isinstance(payload, dict) else {}
missing = [dimension for dimension in dims if dimension not in scores]
if missing:
    raise SystemExit(f"VBench output is missing dimensions: {missing}")
normalized = {dimension: scores[dimension] for dimension in dims}
temporary = canonical.with_name(f".{canonical.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, canonical)
auxiliary = [
    {
        "path": str(path.relative_to(output)),
        "sha256": digest(path),
    }
    for path in sorted(output.rglob("*_full_info.json"))
]
contract = {
    "version": 1,
    "metric_input_fingerprint": fingerprint,
    "input_count": expected,
    "input_fingerprint": stage_manifest["input_fingerprint"],
    "source_result": str(source),
    "source_result_sha256": digest(source),
    "canonical_result": str(canonical),
    "canonical_result_sha256": digest(canonical),
    "dimensions": dims,
    "auxiliary_full_info": auxiliary,
}
(output / "result_contract.json").write_text(
    json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
        {
            printf 'METRIC_INPUT_FINGERPRINT=%s\n' "$METRIC_INPUT_FINGERPRINT"
            printf 'INPUT_COUNT=%s\n' "$EXPECTED"
            printf 'RESULTS_SHA256=%s\n' \
                "$(sha256_file "$output/results.json")"
            printf 'RESULT_CONTRACT_SHA256=%s\n' \
                "$(sha256_file "$output/result_contract.json")"
        } >"$marker"
    done
    python "$ROOT/scripts/collect_vbench_long_results.py" \
        --root "$METRICS/vbench_long" \
        --methods "${METHODS[@]}" --dimensions "${DIMS[@]}" \
        --output-json "$METRICS/vbench_long_summary.json" \
        --output-csv "$METRICS/vbench_long_summary.csv" \
        --output-md "$METRICS/vbench_long_summary.md" \
        >"$METRICS/logs/collect_vbench.log" 2>&1 || exit 1
fi

if [[ "$RUN_COMPREHENSIVE" == "1" ]]; then
    PIDS=()
    STATUS=0
    for index in "${!METHODS[@]}"; do
        method="${METHODS[$index]}"
        output="$METRICS/comprehensive_parts/$method.json"
        marker="$METRICS/status/comprehensive.$method.done"
        if [[ "$RESUME_VALID" == "1" && -s "$marker" && -s "$output" ]] && \
            grep -qx "METRIC_INPUT_FINGERPRINT=$METRIC_INPUT_FINGERPRINT" "$marker" && \
            [[ "$(marker_value "$marker" OUTPUT_SHA256)" == \
               "$(sha256_file "$output")" ]]; then
            continue
        fi
        rm -f "$marker" "$output"
        (
            export CUDA_VISIBLE_DEVICES="${GPUS[$index]}"
            python "$ROOT/scripts/evaluate_comprehensive.py" \
                --video_dirs "$METRICS/eval_inputs/$method" \
                --prompts "$PROMPTS" --output "$output" \
                --gpu 0 --sample_frames "$SAMPLE_FRAMES" \
                --batch_size 8 --skip_m3 && {
                printf 'METRIC_INPUT_FINGERPRINT=%s\n' \
                    "$METRIC_INPUT_FINGERPRINT"
                printf 'OUTPUT_SHA256=%s\n' "$(sha256_file "$output")"
            } >"$marker"
        ) >"$METRICS/logs/comprehensive.$method.log" 2>&1 &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || STATUS=1
    done
    [[ "$STATUS" -eq 0 ]] || exit 1
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
    marker="$METRICS/status/temporal.done"
    if ! {
        [[ "$RESUME_VALID" == "1" && -s "$marker" && -s "$METRICS/temporal_jump.csv" ]] &&
        grep -qx "METRIC_INPUT_FINGERPRINT=$METRIC_INPUT_FINGERPRINT" "$marker" &&
        [[ "$(marker_value "$marker" OUTPUT_SHA256)" == \
           "$(sha256_file "$METRICS/temporal_jump.csv")" ]]
    }; then
        rm -f "$marker" "$METRICS/temporal_jump.csv"
        VIDEO_FILES=()
        for method in "${METHODS[@]}"; do
            mapfile -t METHOD_FILES < <(
                find "$METRICS/eval_inputs/$method" -maxdepth 1 -type f \
                    -name '*.mp4' | sort
            )
            [[ "${#METHOD_FILES[@]}" -eq "$EXPECTED" ]] || {
                echo "[error] temporal input count mismatch for $method"
                exit 2
            }
            VIDEO_FILES+=("${METHOD_FILES[@]}")
        done
        python "$ROOT/scripts/compute_temporal_jump_diagnostic.py" \
            "${VIDEO_FILES[@]}" --frame-step "$TEMPORAL_FRAME_STEP" \
            --expected-videos "$((EXPECTED * ${#METHODS[@]}))" \
            --output "$METRICS/temporal_jump.csv" \
            >"$METRICS/logs/temporal_jump.log" 2>&1 || exit 1
        {
            printf 'METRIC_INPUT_FINGERPRINT=%s\n' "$METRIC_INPUT_FINGERPRINT"
            printf 'OUTPUT_SHA256=%s\n' \
                "$(sha256_file "$METRICS/temporal_jump.csv")"
        } >"$marker"
    fi
fi

if [[ "$RUN_VBENCH" == "1" && ! -s "$METRICS/vbench_long_summary.json" ]]; then
    echo "[error] VBench summary is missing"
    exit 2
fi
if [[ "$RUN_COMPREHENSIVE" == "1" && ! -s "$METRICS/comprehensive.json" ]]; then
    echo "[error] comprehensive result is missing"
    exit 2
fi
if [[ "$RUN_TEMPORAL" == "1" && ! -s "$METRICS/temporal_jump.csv" ]]; then
    echo "[error] temporal result is missing"
    exit 2
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
    python "$ROOT/scripts/analyze_v98_history_polarity.py" \
        --experiment-contract "$EXPERIMENT_CONTRACT" \
        --comprehensive "$METRICS/comprehensive.json" \
        --vbench "$METRICS/vbench_long_summary.json" \
        --temporal-jump "$METRICS/temporal_jump.csv" \
        --map-manifest "$PRIMARY_RUN_ROOT/maps/history_polarity_manifest.json" \
        --policy-audit "$METRICS/policy_trace_audit.json" \
        --metric-manifest "$METRICS/metric_manifest.json" \
        --blind-scorecard "$BLIND_REVIEW/scorecard.csv" \
        --blind-key "$BLIND_PRIVATE/key_private.json" \
        --blind-verification "$METRICS/blind_frozen_verification.json" \
        --output-json "$METRICS/v98_analysis.json" \
        --output-md "$METRICS/v98_analysis.md" \
        >"$METRICS/logs/v98_analysis.log" 2>&1 || exit 1
fi

echo "[v98-postprocess] phase=$PHASE mode=$MODE metrics=$METRICS"
echo "[v98-postprocess] metric_input_fingerprint=$METRIC_INPUT_FINGERPRINT"
