#!/usr/bin/env bash
# Four-node v98 generation with a cross-node immutable experiment contract.
# Primary: each node runs all eight methods for shard=NODE_RANK.
# Follow-up: FOLLOWUP_V78=1 runs a separate matched base/v78 pair only.
set -uo pipefail

MODE="${1:-screen32}"
[[ "$MODE" == "screen32" || "$MODE" == "main128" ]] || {
    echo "usage: NODE_RANK=0..3 $0 screen32|main128"
    exit 2
}

ROOT="${REPO_ROOT:-/apdcephfs_gy2/share_303214315/cedricnie/develop/training-free}"
SF="${SF_REPO:-$ROOT/third_party/Self-Forcing}"
PF="${PF_REPO:-$ROOT/third_party/Pyramid-Forcing}"
SF_CONFIG="${SF_CONFIG:-$SF/configs/self_forcing_dmd.yaml}"
PF_CONFIG="${PF_CONFIG:-$PF/configs/pyramid-forcing.yaml}"
SF_CHECKPOINT="${SF_CHECKPOINT:-$SF/checkpoints/self_forcing_dmd.pt}"
PF_CHECKPOINT="${PF_CHECKPOINT:-$PF/checkpoints/self_forcing_dmd.pt}"
PF_LABELS="${PF_LABELS:-$PF/configs/head_configs/best_labels.csv}"
SCORE_ROOT="${SCORE_ROOT:-$ROOT/runs/v98_middle_relative_scores}"
SCORES="${SCORES:-$SCORE_ROOT/scores/qk_head_scores.csv}"
SCORE_ARTIFACT="${SCORE_ARTIFACT:-$SCORE_ROOT/scores/qk_head_score_artifact.json}"
EXPECTED_SCORE_ARTIFACT_VERSION=2
EXPECTED_SCORE_ARTIFACT_METHOD="v98_middle_relative_qk_head_scores"
EXPECTED_SCORE_PRIMARY_FIELD="middle_relative_logit_margin"
CONDA_SH="${CONDA_SH:-/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-longlive}"
NODE_RANK="${NODE_RANK:-}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FRAMES="${FRAMES:-120}"
EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES:-$((FRAMES * 4 - 3))}"
EXPECTED_VIDEO_FPS="${EXPECTED_VIDEO_FPS:-16}"
EXPECTED_VIDEO_WIDTH="${EXPECTED_VIDEO_WIDTH:-832}"
EXPECTED_VIDEO_HEIGHT="${EXPECTED_VIDEO_HEIGHT:-480}"
VIDEO_FPS_TOLERANCE="${VIDEO_FPS_TOLERANCE:-0.01}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"
PRELOAD_PYRAMIDKV="${PRELOAD_PYRAMIDKV:-0}"
MAP_WAIT_SECONDS="${MAP_WAIT_SECONDS:-600}"
CONTRACT_WAIT_SECONDS="${CONTRACT_WAIT_SECONDS:-120}"
FOLLOWUP_V78="${FOLLOWUP_V78:-0}"
[[ "$FOLLOWUP_V78" == "0" || "$FOLLOWUP_V78" == "1" ]] || {
    echo "[error] FOLLOWUP_V78 must be 0 or 1"
    exit 2
}

if [[ "$MODE" == "screen32" ]]; then
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num32.txt}"
    EXPECTED=32
    OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v98_history_polarity_screen32_corrected}"
else
    PROMPTS="${PROMPTS:-$PF/prompts/MovieGenVideoBench_num128.txt}"
    EXPECTED=128
    OUT_ROOT="${OUT_ROOT:-$ROOT/runs/v98_history_polarity_main128_corrected}"
fi
MAP_DIR="${MAP_DIR:-$OUT_ROOT/maps}"

[[ "$FRAMES" == "120" ]] || {
    echo "[error] v98 protocol requires FRAMES=120"
    exit 2
}
[[ "$SEED" == "0" ]] || {
    echo "[error] v98 protocol requires SEED=0"
    exit 2
}
[[ "$EXPECTED_VIDEO_FRAMES" == "477" ]] || {
    echo "[error] v98 protocol requires EXPECTED_VIDEO_FRAMES=477"
    exit 2
}
[[ "$EXPECTED_VIDEO_WIDTH" == "832" && "$EXPECTED_VIDEO_HEIGHT" == "480" ]] || {
    echo "[error] v98 protocol requires decoded resolution 832x480"
    exit 2
}
python - "$EXPECTED_VIDEO_FPS" "$VIDEO_FPS_TOLERANCE" <<'PY' || exit 2
import math
import sys

fps = float(sys.argv[1])
tolerance = float(sys.argv[2])
if not math.isclose(fps, 16.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("[error] v98 protocol requires EXPECTED_VIDEO_FPS=16")
if not math.isclose(tolerance, 0.01, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("[error] v98 protocol requires VIDEO_FPS_TOLERANCE=0.01")
PY

[[ "$NODE_RANK" =~ ^[0-3]$ ]] || {
    echo "[error] NODE_RANK must be one of 0,1,2,3"
    exit 2
}
IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ "${#GPUS[@]}" -eq 8 ]] || {
    echo "[error] each v98 node requires exactly eight local GPU ids"
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
for path in \
    "$ROOT" "$SF" "$PF" "$SF_CONFIG" "$PF_CONFIG" \
    "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$PF_LABELS" \
    "$SCORES" "$SCORE_ARTIFACT" "$PROMPTS" "$CONDA_SH"; do
    [[ -e "$path" ]] || { echo "[error] missing $path"; exit 2; }
done

PROMPT_COUNT="$(grep -cve '^[[:space:]]*$' "$PROMPTS")"
[[ "$PROMPT_COUNT" -eq "$EXPECTED" ]] || {
    echo "[error] $MODE expects $EXPECTED prompts, found $PROMPT_COUNT"
    exit 2
}
CANONICAL_PROMPTS="$PF/prompts/MovieGenVideoBench_num${EXPECTED}.txt"
[[ -s "$CANONICAL_PROMPTS" ]] || {
    echo "[error] missing canonical MovieBench prompt file $CANONICAL_PROMPTS"
    exit 2
}
[[ "$(sha256sum "$PROMPTS" | awk '{print $1}')" == \
   "$(sha256sum "$CANONICAL_PROMPTS" | awk '{print $1}')" ]] || {
    echo "[error] v98 $MODE prompts must match the canonical MovieBench-$EXPECTED file"
    exit 2
}
(( PROMPT_COUNT % 4 == 0 )) || {
    echo "[error] prompt count must be divisible by four"
    exit 2
}
SHARD_SIZE=$((PROMPT_COUNT / 4))

RUN_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null)" || {
    echo "[error] repository commit cannot be resolved"
    exit 2
}
RUN_STATUS_TEXT="$(
    git -C "$ROOT" status --porcelain=v1 --untracked-files=all
)"
if [[ -n "$RUN_STATUS_TEXT" ]]; then
    echo "[error] worktree, including non-ignored untracked files, must be clean"
    printf '%s\n' "$RUN_STATUS_TEXT"
    exit 2
fi
RUN_DIRTY=0
RUN_DIFF_SHA256="$(git -C "$ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')"
RUN_STATUS_SHA256="$(
    printf '%s' "$RUN_STATUS_TEXT" | sha256sum | awk '{print $1}'
)"
SCREEN_GATE_ANALYSIS_PATH=""
SCREEN_GATE_ANALYSIS_SHA256=""
SCREEN_GATE_CONTRACT_PATH=""
SCREEN_GATE_CONTRACT_SHA256=""
if [[ "$MODE" == "main128" ]]; then
    SCREEN32_RUN_ROOT="${SCREEN32_RUN_ROOT:-$ROOT/runs/v98_history_polarity_screen32_corrected}"
    SCREEN_GATE_ANALYSIS_PATH="$SCREEN32_RUN_ROOT/metrics/v98_analysis.json"
    SCREEN_GATE_CONTRACT_PATH="$SCREEN32_RUN_ROOT/experiment_contract.json"
    for path in "$SCREEN_GATE_ANALYSIS_PATH" "$SCREEN_GATE_CONTRACT_PATH"; do
        [[ -s "$path" ]] || {
            echo "[error] main128 requires completed screen32 evidence: $path"
            exit 2
        }
    done
    python - \
        "$SCREEN_GATE_ANALYSIS_PATH" "$SCREEN_GATE_CONTRACT_PATH" \
        "$RUN_COMMIT" "$PF/prompts/MovieGenVideoBench_num32.txt" \
        "$SCORE_ARTIFACT" "$SCORES" "$SF_CONFIG" "$PF_CONFIG" \
        "$SF_CHECKPOINT" "$PF_CHECKPOINT" "$PF_LABELS" <<'PY' || exit 2
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


analysis_path = pathlib.Path(sys.argv[1]).resolve()
contract_path = pathlib.Path(sys.argv[2]).resolve()
current_commit = sys.argv[3]
screen_prompts = pathlib.Path(sys.argv[4]).resolve()
current_inputs = {
    "score_artifact": pathlib.Path(sys.argv[5]).resolve(),
    "scores": pathlib.Path(sys.argv[6]).resolve(),
    "sf_config": pathlib.Path(sys.argv[7]).resolve(),
    "pf_config": pathlib.Path(sys.argv[8]).resolve(),
    "sf_checkpoint": pathlib.Path(sys.argv[9]).resolve(),
    "pf_checkpoint": pathlib.Path(sys.argv[10]).resolve(),
}
pf_labels = pathlib.Path(sys.argv[11]).resolve()
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if (
    analysis.get("version") != 2
    or analysis.get("method")
    != "v98_history_polarity_paired_decision_analysis"
    or analysis.get("hard_gate_pass") is not True
):
    raise SystemExit("[error] screen32 analysis did not pass every hard gate")
bound = analysis.get("experiment_contract", {})
if (
    contract.get("experiment") != "v98_history_polarity"
    or contract.get("phase") != "primary"
    or contract.get("mode") != "screen32"
    or contract.get("run_commit") != current_commit
    or contract.get("prompt", {}).get("count") != 32
    or contract.get("prompt", {}).get("sha256") != digest(screen_prompts)
    or pathlib.Path(str(bound.get("path", ""))).resolve() != contract_path
    or bound.get("sha256") != digest(contract_path)
):
    raise SystemExit("[error] screen32 analysis/contract binding is invalid")
screen_inputs = contract.get("inputs", {})
for name, path in current_inputs.items():
    if (
        not path.is_file()
        or not isinstance(screen_inputs.get(name), dict)
        or screen_inputs[name].get("sha256") != digest(path)
    ):
        raise SystemExit(
            f"[error] screen32 evidence uses a different current input: {name}"
        )
screen_methods = {
    item.get("name"): item
    for item in contract.get("methods", [])
    if isinstance(item, dict)
}
if (
    set(screen_methods) != {
        "sf_native",
        "pf_native",
        "pf_explicit_parity",
        "pf_aw_hybrid_merge",
        "history_polarity_hybrid_merge",
        "history_polarity_stride_merge",
        "history_polarity_zero_random_hybrid_merge",
        "positive_rate_half_hybrid_merge",
    }
    or screen_methods["pf_native"].get("map_sha256") != digest(pf_labels)
):
    raise SystemExit("[error] screen32 method matrix or PF labels are stale")
required_usable = set(
    analysis.get("gates", {})
    .get("blind_scorecard", {})
    .get("required_usable_methods", [])
)
if required_usable != {
    "pf_native",
    "pf_explicit_parity",
    "history_polarity_hybrid_merge",
}:
    raise SystemExit("[error] screen32 blind usability gate is incomplete")
inputs = analysis.get("input_artifacts")
if not isinstance(inputs, dict) or "metric_manifest" not in inputs:
    raise SystemExit("[error] screen32 analysis input evidence is incomplete")
for name, item in inputs.items():
    if not isinstance(item, dict):
        raise SystemExit(f"[error] malformed screen32 input evidence {name}")
    path = pathlib.Path(str(item.get("path", ""))).resolve()
    if not path.is_file() or digest(path) != item.get("sha256"):
        raise SystemExit(f"[error] stale screen32 input evidence {name}: {path}")
print("[v98-main128-gate] frozen screen32 hard gates passed")
PY
    SCREEN_GATE_ANALYSIS_SHA256="$(
        sha256sum "$SCREEN_GATE_ANALYSIS_PATH" | awk '{print $1}'
    )"
    SCREEN_GATE_CONTRACT_SHA256="$(
        sha256sum "$SCREEN_GATE_CONTRACT_PATH" | awk '{print $1}'
    )"
fi

source "$CONDA_SH" || exit 2
conda activate "$CONDA_ENV" || exit 2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$PF:$SF:$ROOT/scripts:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Freeze the pure Python reference path.  PYRAMIDKV_USE_CPP_STRATEGY was a
# historical misspelling; set both it and the real switch defensively.
export PYRAMIDKV_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_STRATEGY=0
export PYRAMIDKV_USE_CPP_PACK=0
export PYRAMIDKV_USE_CPP_PACK_OUTPUT=0
export PYRAMIDKV_USE_MEGA_CACHE=0
export PYRAMIDKV_USE_MEGA_ATTN=0
export PYRAMIDKV_CONTIG_ANCHOR_STORE=0
export PYRAMIDKV_HEAD_MAP_DEBUG=1
export PYRAMIDKV_POLICY_TRACE_LAYERS="${PYRAMIDKV_POLICY_TRACE_LAYERS:-0,7,15,23,29}"
export PYRAMIDKV_POLICY_TRACE_STRIDE="${PYRAMIDKV_POLICY_TRACE_STRIDE:-3}"
export PYRAMIDKV_POLICY_TRACE_MAX_RECORDS="${PYRAMIDKV_POLICY_TRACE_MAX_RECORDS:-60000}"
[[ "$PYRAMIDKV_POLICY_TRACE_LAYERS" == "0,7,15,23,29" ]] || {
    echo "[error] v98 policy trace layers are frozen at 0,7,15,23,29"
    exit 2
}
[[ "$PYRAMIDKV_POLICY_TRACE_STRIDE" == "3" ]] || {
    echo "[error] v98 policy trace stride is frozen at 3"
    exit 2
}
[[ "$PYRAMIDKV_POLICY_TRACE_MAX_RECORDS" == "60000" ]] || {
    echo "[error] v98 policy trace max-record budget is frozen at 60000"
    exit 2
}

mkdir -p "$OUT_ROOT" "$MAP_DIR"

python - \
    "$SCORE_ARTIFACT" "$SCORES" \
    "$EXPECTED_SCORE_ARTIFACT_VERSION" \
    "$EXPECTED_SCORE_ARTIFACT_METHOD" \
    "$EXPECTED_SCORE_PRIMARY_FIELD" \
    "$RUN_COMMIT" "$PF_CONFIG" "$PF_CHECKPOINT" <<'PY' || exit 2
import csv
import hashlib
import json
import math
import pathlib
import sys


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


artifact_path = pathlib.Path(sys.argv[1])
scores_path = pathlib.Path(sys.argv[2])
expected_version = int(sys.argv[3])
expected_method = sys.argv[4]
expected_primary = sys.argv[5]
current_commit = sys.argv[6]
current_config = pathlib.Path(sys.argv[7]).resolve()
current_checkpoint = pathlib.Path(sys.argv[8]).resolve()
payload = json.loads(artifact_path.read_text(encoding="utf-8"))
if payload.get("version") != expected_version:
    raise SystemExit(
        f"[error] score artifact version must be {expected_version}, "
        f"found {payload.get('version')!r}"
    )
if payload.get("method") != expected_method:
    raise SystemExit(
        f"[error] score artifact method must be {expected_method!r}, "
        f"found {payload.get('method')!r}"
    )
if payload.get("accepted") is not True:
    raise SystemExit("[error] score artifact is not explicitly accepted=true")
definition = payload.get("score_definition", {})
primary = definition.get("primary_field")
if primary != expected_primary:
    raise SystemExit(
        f"[error] score artifact primary_field must be {expected_primary!r}, "
        f"found {primary!r}"
    )
if definition.get("probe_policy_balanced") is not True:
    raise SystemExit("[error] score artifact is not probe-policy balanced")
if definition.get("probe_policies") != ["uniform_stride", "uniform_merge"]:
    raise SystemExit(
        "[error] score artifact probe_policies must be exactly "
        "['uniform_stride', 'uniform_merge']"
    )
if definition.get("bootstrap_unit") != "counterfactual_prompt_pair":
    raise SystemExit(
        "[error] score artifact bootstrap_unit must be "
        "'counterfactual_prompt_pair'"
    )
expected_bootstrap = {
    "rounds": 500,
    "seed": 20260726,
    "zero_effect_is_stable": False,
}
if payload.get("bootstrap_protocol") != expected_bootstrap:
    raise SystemExit(
        "[error] score artifact bootstrap protocol is not the frozen v98 "
        f"protocol: {payload.get('bootstrap_protocol')!r}"
    )
expected_acceptance = {
    "min_profiles_per_policy_head": 32,
    "min_stable_head_fraction": 0.80,
    "min_head_bootstrap_agreement": 0.75,
    "min_topology_sign_agreement_fraction": 0.80,
    "min_minority_fraction": 0.05,
}
if payload.get("acceptance_protocol") != expected_acceptance:
    raise SystemExit(
        "[error] score artifact acceptance protocol is not the frozen v98 "
        f"protocol: {payload.get('acceptance_protocol')!r}"
    )
gates = payload.get("acceptance_gates", {})
if set(gates) != {
    "complete_head_grid",
    "bootstrap_stable_head_fraction",
    "topology_sign_agreement_fraction",
    "minority_role_fraction",
} or any(item.get("passed") is not True for item in gates.values()):
    raise SystemExit("[error] score artifact acceptance gates are incomplete")
if (
    gates["complete_head_grid"].get("observed") != 360
    or gates["complete_head_grid"].get("required") != 360
    or gates["bootstrap_stable_head_fraction"].get("required") != 0.80
    or gates["bootstrap_stable_head_fraction"].get("per_head_threshold")
    != 0.75
    or gates["topology_sign_agreement_fraction"].get("required") != 0.80
    or gates["minority_role_fraction"].get("required") != 0.05
):
    raise SystemExit("[error] score artifact gate thresholds are not frozen")
with scores_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 360:
    raise SystemExit(
        f"[error] score CSV must contain exactly 360 heads, found {len(rows)}"
    )
stable_fraction = (
    sum(float(row["bootstrap_sign_agreement"]) >= 0.75 for row in rows)
    / len(rows)
)
topology_fraction = (
    sum(
        float(row["uniform_stride_margin"]) != 0.0
        and float(row["uniform_merge_margin"]) != 0.0
        and (
            (float(row["uniform_stride_margin"]) > 0.0)
            == (float(row["uniform_merge_margin"]) > 0.0)
        )
        for row in rows
    )
    / len(rows)
)
for row in rows:
    expected_topology = int(
        float(row["uniform_stride_margin"]) != 0.0
        and float(row["uniform_merge_margin"]) != 0.0
        and (
            (float(row["uniform_stride_margin"]) > 0.0)
            == (float(row["uniform_merge_margin"]) > 0.0)
        )
    )
    if int(float(row["topology_sign_agreement"])) != expected_topology:
        raise SystemExit(
            "[error] score CSV topology agreement is inconsistent with "
            "the two policy margins"
        )
support_count = sum(
    float(row[expected_primary]) >= 0.0 for row in rows
)
minority_fraction = min(support_count, len(rows) - support_count) / len(rows)
for gate_name, observed in (
    ("bootstrap_stable_head_fraction", stable_fraction),
    ("topology_sign_agreement_fraction", topology_fraction),
    ("minority_role_fraction", minority_fraction),
):
    declared = gates[gate_name].get("observed")
    if not isinstance(declared, (int, float)) or not math.isclose(
        float(declared), observed, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SystemExit(
            f"[error] score artifact gate {gate_name} does not match CSV"
        )
    required = float(gates[gate_name]["required"])
    if observed < required or gates[gate_name].get("passed") is not True:
        raise SystemExit(
            f"[error] score artifact gate {gate_name} is below its frozen "
            f"threshold: observed={observed} required={required}"
        )
if payload.get("label_counts_at_zero") != {
    "10": support_count,
    "11": len(rows) - support_count,
}:
    raise SystemExit("[error] score artifact label counts do not match CSV")
expected_hash = payload.get("files", {}).get("score_csv_sha256")
actual_hash = digest(scores_path)
if expected_hash != actual_hash:
    raise SystemExit(
        "[error] score artifact/CSV hash mismatch: "
        f"expected={expected_hash} actual={actual_hash}"
    )
profile_protocol = payload.get("profile_protocol")
if not isinstance(profile_protocol, dict):
    raise SystemExit("[error] score artifact profile protocol is missing")
for key, expected in {
    "RUN_COMMIT": current_commit,
    "TRACKED_WORKTREE_DIRTY": "0",
    "SINK_FRAMES": "3",
    "RECENT_FRAMES": "4",
    "PROFILE_FRAMES": "120",
    "PROFILE_BRANCHES": "cond",
    "PROFILE_UPDATE_MODES": "noisy",
}.items():
    if profile_protocol.get(key) != expected:
        raise SystemExit(
            f"[error] calibration/deployment mismatch for {key}: "
            f"expected={expected!r} actual={profile_protocol.get(key)!r}"
        )
for name, current_path, path_key, hash_key in (
    ("config", current_config, "CONFIG", "CONFIG_SHA256"),
    (
        "checkpoint",
        current_checkpoint,
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
    ),
):
    calibrated_path = pathlib.Path(
        str(profile_protocol.get(path_key, ""))
    ).resolve()
    if calibrated_path != current_path:
        raise SystemExit(
            f"[error] calibration/deployment {name} path mismatch: "
            f"calibrated={calibrated_path} current={current_path}"
        )
    calibrated_hash = profile_protocol.get(hash_key)
    current_hash = digest(current_path)
    if calibrated_hash != current_hash:
        raise SystemExit(
            f"[error] calibration/deployment {name} hash mismatch: "
            f"calibrated={calibrated_hash} current={current_hash}"
        )
print(
    f"[ScoreArtifactAudit] version={expected_version} method={expected_method} "
    f"primary={expected_primary} accepted=true sha256={actual_hash}",
    flush=True,
)
PY

ensure_maps() {
    local manifest="$MAP_DIR/history_polarity_manifest.json"
    local lock="$OUT_ROOT/.map_build_lock"
    if [[ -s "$manifest" ]]; then
        return 0
    fi
    if [[ "$FOLLOWUP_V78" == "1" ]]; then
        echo "[error] follow-up requires primary maps; missing $manifest"
        return 1
    fi
    if mkdir "$lock" 2>/dev/null; then
        python "$ROOT/scripts/build_v98_history_polarity_maps.py" \
            --scores "$SCORES" \
            --score-artifact "$SCORE_ARTIFACT" \
            --pf-labels "$PF_LABELS" \
            --output-dir "$MAP_DIR"
        local status=$?
        rmdir "$lock" 2>/dev/null || true
        [[ "$status" -eq 0 && -s "$manifest" ]] || return 1
        return 0
    fi

    local waited=0
    while [[ ! -s "$manifest" && "$waited" -lt "$MAP_WAIT_SECONDS" ]]; do
        sleep 2
        waited=$((waited + 2))
    done
    [[ -s "$manifest" ]] || {
        echo "[error] timed out waiting for history-polarity maps"
        return 1
    }
}
ensure_maps || exit 1
python "$ROOT/scripts/build_v98_history_polarity_maps.py" \
    --scores "$SCORES" \
    --score-artifact "$SCORE_ARTIFACT" \
    --pf-labels "$PF_LABELS" \
    --output-dir "$MAP_DIR" \
    --validate-only || exit 1

POLARITY_ZERO="$MAP_DIR/history_polarity_zero.csv"
POLARITY_ZERO_RANDOM="$MAP_DIR/history_polarity_zero_random.csv"
POSITIVE_HALF="$MAP_DIR/positive_rate_half.csv"
PF_AW="$MAP_DIR/pf_aw_binary_control.csv"
MAP_MANIFEST="$MAP_DIR/history_polarity_manifest.json"
for path in \
    "$POLARITY_ZERO" "$POLARITY_ZERO_RANDOM" "$POSITIVE_HALF" \
    "$PF_AW" "$MAP_MANIFEST"; do
    [[ -s "$path" ]] || { echo "[error] missing generated map $path"; exit 2; }
done
python - \
    "$MAP_MANIFEST" "$SCORES" "$SCORE_ARTIFACT" "$PF_LABELS" \
    "$POLARITY_ZERO" "$POLARITY_ZERO_RANDOM" "$POSITIVE_HALF" "$PF_AW" \
    <<'PY' || exit 2
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


(
    manifest_path,
    scores,
    score_artifact,
    pf_labels,
    polarity_zero,
    polarity_zero_random,
    positive_half,
    pf_aw,
) = map(pathlib.Path, sys.argv[1:])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("version") != 2:
    raise SystemExit("[error] map manifest version must be exactly 2")
if payload.get("method") != "v98_middle_relative_history_map_builder":
    raise SystemExit("[error] map manifest method mismatch")
if payload.get("claims", {}).get("probe_policy_balanced") is not True:
    raise SystemExit("[error] map manifest is not probe-policy balanced")
if payload.get("score_csv_sha256") != digest(scores):
    raise SystemExit("[error] map manifest score hash mismatch")
if payload.get("score_artifact_sha256") != digest(score_artifact):
    raise SystemExit("[error] map manifest score-artifact hash mismatch")
if payload.get("pf_labels_sha256") != digest(pf_labels):
    raise SystemExit("[error] map manifest PF-label hash mismatch")
if payload.get("support_label") != 10 or payload.get("suppress_label") != 11:
    raise SystemExit("[error] map manifest does not use neutral labels 10/11")
if payload.get("thresholds") != [-0.1, 0.0, 0.1]:
    raise SystemExit("[error] map manifest thresholds are not frozen at -0.1/0/0.1")
required = {
    "history_polarity_zero",
    "history_polarity_zero_random",
    "positive_rate_half",
    "pf_aw_binary_control",
}
if not required.issubset(payload.get("maps", {})):
    missing = sorted(required - set(payload.get("maps", {})))
    raise SystemExit(f"[error] map manifest is missing required maps: {missing}")
if (
    payload["maps"]["history_polarity_zero_random"].get("seed") != 2026
    or payload["maps"]["history_polarity_zero_random"].get("reference")
    != "history_polarity_zero"
):
    raise SystemExit("[error] count-matched random control is not frozen")
expected_paths = {
    "history_polarity_zero": polarity_zero,
    "history_polarity_zero_random": polarity_zero_random,
    "positive_rate_half": positive_half,
    "pf_aw_binary_control": pf_aw,
}
for name, item in payload.get("maps", {}).items():
    path = pathlib.Path(item["path"])
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    path = path.resolve()
    if not path.is_file() or digest(path) != item.get("sha256"):
        raise SystemExit(f"[error] stale or missing map {name}: {path}")
    if name in expected_paths and path != expected_paths[name].resolve():
        raise SystemExit(
            f"[error] runner label path for {name} is not the manifest path: "
            f"runner={expected_paths[name].resolve()} manifest={path}"
        )
print("[HistoryPolarityMapAudit] hashes=ok labels=10/11", flush=True)
PY

# The primary screen is an eight-cell go/no-go matrix.  v78 is deliberately
# excluded and can only be run as a separate, freshly matched two-cell phase.
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
CELLS=(
    "sf_native|sf||none|0"
    "pf_native|pf|$PF_LABELS|native|0"
    "pf_explicit_parity|pf|$PF_LABELS|pf_explicit_parity|0"
    "pf_aw_hybrid_merge|pf|$PF_AW|history_hybrid_merge|0"
    "history_polarity_hybrid_merge|pf|$POLARITY_ZERO|history_hybrid_merge|0"
    "history_polarity_stride_merge|pf|$POLARITY_ZERO|history_stride_merge|0"
    "history_polarity_zero_random_hybrid_merge|pf|$POLARITY_ZERO_RANDOM|history_hybrid_merge|0"
    "positive_rate_half_hybrid_merge|pf|$POSITIVE_HALF|history_hybrid_merge|0"
)
PHASE=primary
RUN_ROOT="$OUT_ROOT"
PRIMARY_MANIFEST_PATH=""
PRIMARY_EXPERIMENT_CONTRACT_PATH=""
PRIMARY_ANALYSIS_PATH=""
PRIMARY_BLIND_FROZEN_PATH=""
PRIMARY_BLIND_VERIFICATION_PATH=""
PRIMARY_BLIND_COMPLETION_PATH=""
PRIMARY_BLIND_SCORECARD_PATH=""
PRIMARY_BLIND_KEY_PATH=""
if [[ "$FOLLOWUP_V78" == "1" ]]; then
    PHASE=followup_v78
    RUN_ROOT="$OUT_ROOT/followup_v78"
    METHODS=(
        followup_history_polarity_hybrid_merge_base
        followup_history_polarity_hybrid_merge_v78
    )
    CELLS=(
        "followup_history_polarity_hybrid_merge_base|pf|$POLARITY_ZERO|history_hybrid_merge|0"
        "followup_history_polarity_hybrid_merge_v78|pf|$POLARITY_ZERO|history_hybrid_merge|1"
    )
    PRIMARY_MANIFEST_PATH="$OUT_ROOT/experiment_manifest.env"
    PRIMARY_EXPERIMENT_CONTRACT_PATH="$OUT_ROOT/experiment_contract.json"
    PRIMARY_ANALYSIS_PATH="$OUT_ROOT/metrics/v98_analysis.json"
    PRIMARY_BLIND_FROZEN_PATH="$OUT_ROOT/blind_review_private/FROZEN.json"
    PRIMARY_BLIND_VERIFICATION_PATH="$OUT_ROOT/metrics/blind_frozen_verification.json"
    PRIMARY_BLIND_COMPLETION_PATH="$OUT_ROOT/blind_review_private/.complete.json"
    PRIMARY_BLIND_SCORECARD_PATH="$OUT_ROOT/blind_review/scorecard.csv"
    PRIMARY_BLIND_KEY_PATH="$OUT_ROOT/blind_review_private/key_private.json"
    [[ -s "$PRIMARY_MANIFEST_PATH" ]] || {
        echo "[error] follow-up requires a completed primary experiment manifest"
        exit 2
    }
    for node in 0 1 2 3; do
        [[ -s "$OUT_ROOT/status/node${node}.done" ]] || {
            echo "[error] follow-up requires completed primary node $node"
            exit 2
        }
    done
    for path in \
        "$PRIMARY_EXPERIMENT_CONTRACT_PATH" \
        "$PRIMARY_ANALYSIS_PATH" \
        "$PRIMARY_BLIND_FROZEN_PATH" \
        "$PRIMARY_BLIND_VERIFICATION_PATH" \
        "$PRIMARY_BLIND_COMPLETION_PATH" \
        "$PRIMARY_BLIND_SCORECARD_PATH" \
        "$PRIMARY_BLIND_KEY_PATH"; do
        [[ -s "$path" ]] || {
            echo "[error] follow-up requires completed primary review/analysis: $path"
            exit 2
        }
    done
    python - \
        "$PRIMARY_ANALYSIS_PATH" \
        "$PRIMARY_EXPERIMENT_CONTRACT_PATH" \
        "$PRIMARY_BLIND_FROZEN_PATH" \
        "$PRIMARY_BLIND_VERIFICATION_PATH" \
        "$PRIMARY_BLIND_COMPLETION_PATH" \
        "$PRIMARY_BLIND_SCORECARD_PATH" \
        "$PRIMARY_BLIND_KEY_PATH" \
        "$MODE" "$RUN_COMMIT" "$PROMPTS" "$PROMPT_COUNT" \
        "$MAP_MANIFEST" "$SCORE_ARTIFACT" "$SCORES" <<'PY' || exit 2
import hashlib
import json
import pathlib
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


path_args = list(map(pathlib.Path, sys.argv[1:8]))
(
    analysis_path,
    contract_path,
    frozen_path,
    verification_path,
    completion_path,
    scorecard_path,
    key_path,
) = path_args
(
    expected_mode,
    expected_commit,
    raw_prompts_path,
    raw_prompt_count,
    raw_map_manifest_path,
    raw_score_artifact_path,
    raw_scores_path,
) = sys.argv[8:]
prompts_path = pathlib.Path(raw_prompts_path).resolve()
map_manifest_path = pathlib.Path(raw_map_manifest_path).resolve()
score_artifact_path = pathlib.Path(raw_score_artifact_path).resolve()
scores_path = pathlib.Path(raw_scores_path).resolve()
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if (
    analysis.get("version") != 2
    or analysis.get("method")
    != "v98_history_polarity_paired_decision_analysis"
    or analysis.get("hard_gate_pass") is not True
):
    raise SystemExit(
        "[error] v78 follow-up requires a valid primary v98_analysis with "
        "hard_gate_pass=true"
    )
binding = analysis.get("experiment_contract", {})
if (
    pathlib.Path(str(binding.get("path", ""))).resolve()
    != contract_path.resolve()
    or binding.get("sha256") != digest(contract_path)
):
    raise SystemExit(
        "[error] primary analysis is not bound to the current experiment contract"
    )
prompt_contract = contract.get("prompt", {})
score_contract = contract.get("score", {})
if (
    contract.get("experiment") != "v98_history_polarity"
    or contract.get("phase") != "primary"
    or contract.get("mode") != expected_mode
    or contract.get("run_commit") != expected_commit
    or pathlib.Path(str(prompt_contract.get("path", ""))).resolve()
    != prompts_path
    or prompt_contract.get("sha256") != digest(prompts_path)
    or prompt_contract.get("count") != int(raw_prompt_count)
    or pathlib.Path(str(score_contract.get("map_manifest_path", ""))).resolve()
    != map_manifest_path
    or score_contract.get("map_manifest_sha256")
    != digest(map_manifest_path)
    or pathlib.Path(str(score_contract.get("artifact_path", ""))).resolve()
    != score_artifact_path
    or score_contract.get("artifact_sha256") != digest(score_artifact_path)
    or pathlib.Path(str(score_contract.get("csv_path", ""))).resolve()
    != scores_path
    or score_contract.get("csv_sha256") != digest(scores_path)
):
    raise SystemExit(
        "[error] primary experiment semantics do not match this follow-up "
        "mode/commit/prompt/map/score contract"
    )
blind_gate = analysis.get("gates", {}).get("blind_scorecard", {})
if (
    blind_gate.get("pass") is not True
    or blind_gate.get("frozen_verified") is not True
):
    raise SystemExit(
        "[error] primary analysis does not contain a passing frozen blind gate"
    )
required_inputs = {
    "comprehensive",
    "vbench",
    "temporal_jump",
    "map_manifest",
    "policy_audit",
    "metric_manifest",
    "blind_scorecard",
    "blind_key",
    "blind_verification",
}
input_artifacts = analysis.get("input_artifacts", {})
if not required_inputs.issubset(input_artifacts):
    raise SystemExit(
        "[error] primary analysis is missing immutable input-artifact bindings"
    )
for name, item in input_artifacts.items():
    path = pathlib.Path(str(item.get("path", "")))
    if not path.is_file() or digest(path) != item.get("sha256"):
        raise SystemExit(
            f"[error] primary analysis input is stale: {name} -> {path}"
        )

completion = json.loads(completion_path.read_text(encoding="utf-8"))
frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
verification = json.loads(verification_path.read_text(encoding="utf-8"))
completion_sha = digest(completion_path)
frozen_sha = digest(frozen_path)
scorecard_sha = digest(scorecard_path)
key_sha = digest(key_path)
if (
    frozen.get("completion_sha256") != completion_sha
    or verification.get("completion_sha256") != completion_sha
    or frozen.get("scorecard_sha256") != scorecard_sha
    or verification.get("scorecard_sha256") != scorecard_sha
    or verification.get("freeze_marker_sha256") != frozen_sha
    or completion.get("key_private_sha256") != key_sha
):
    raise SystemExit(
        "[error] primary blind completion/freeze/scorecard/key hash chain is stale"
    )
source_inventory = completion.get("source_inventory")
candidate_inventory = completion.get("candidate_inventory")
if not isinstance(source_inventory, list) or not isinstance(
    candidate_inventory, list
):
    raise SystemExit("[error] primary blind inventory is malformed")
for item in source_inventory:
    path = pathlib.Path(str(item.get("file", "")))
    if (
        not path.is_file()
        or path.stat().st_size != item.get("size")
        or digest(path) != item.get("sha256")
    ):
        raise SystemExit(f"[error] primary blind source is stale: {path}")
source_fingerprint = canonical_digest(source_inventory)
if (
    completion.get("source_fingerprint") != source_fingerprint
    or frozen.get("source_fingerprint") != source_fingerprint
    or verification.get("source_fingerprint") != source_fingerprint
):
    raise SystemExit("[error] primary blind source fingerprint is stale")
public_output = pathlib.Path(str(completion.get("public_output", ""))).resolve()
private_output = pathlib.Path(str(completion.get("private_output", ""))).resolve()
if (
    private_output != completion_path.parent.resolve()
    or public_output != scorecard_path.parent.resolve()
):
    raise SystemExit("[error] primary blind public/private path binding is stale")
for item in candidate_inventory:
    path = public_output / str(item.get("video", ""))
    if (
        not path.is_file()
        or path.stat().st_size != item.get("size")
        or digest(path) != item.get("sha256")
    ):
        raise SystemExit(f"[error] primary blinded candidate is stale: {path}")
public_path = public_output / "manifest_public.json"
if (
    not public_path.is_file()
    or digest(public_path) != completion.get("manifest_public_sha256")
):
    raise SystemExit("[error] primary blind public manifest is stale")
print(
    "[v98-followup-gate] primary hard gates, analysis inputs, and frozen "
    "blind evidence passed"
)
PY
fi
[[ "${#CELLS[@]}" -eq "${#METHODS[@]}" ]] || exit 2
for index in "${!METHODS[@]}"; do
    IFS='|' read -r cell_name _ <<<"${CELLS[$index]}"
    [[ "$cell_name" == "${METHODS[$index]}" ]] || {
        echo "[error] method/cell mismatch at index $index"
        exit 2
    }
done

mkdir -p \
    "$RUN_ROOT"/{logs,status,configs,traces,diagnostics,nodes}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

PROMPT_SHA256="$(file_sha256 "$PROMPTS")"
SCORE_SHA256="$(file_sha256 "$SCORES")"
SCORE_ARTIFACT_SHA256="$(file_sha256 "$SCORE_ARTIFACT")"
MAP_MANIFEST_SHA256="$(file_sha256 "$MAP_MANIFEST")"
POLARITY_ZERO_SHA256="$(file_sha256 "$POLARITY_ZERO")"
POLARITY_ZERO_RANDOM_SHA256="$(file_sha256 "$POLARITY_ZERO_RANDOM")"
POSITIVE_HALF_SHA256="$(file_sha256 "$POSITIVE_HALF")"
PF_AW_SHA256="$(file_sha256 "$PF_AW")"
PF_LABELS_SHA256="$(file_sha256 "$PF_LABELS")"
SF_CONFIG_SHA256="$(file_sha256 "$SF_CONFIG")"
PF_CONFIG_SHA256="$(file_sha256 "$PF_CONFIG")"
SF_CHECKPOINT_SHA256="$(file_sha256 "$SF_CHECKPOINT")"
PF_CHECKPOINT_SHA256="$(file_sha256 "$PF_CHECKPOINT")"
SF_INFERENCE_SHA256="$(file_sha256 "$SF/inference.py")"
PF_INFERENCE_SHA256="$(file_sha256 "$PF/inference.py")"
RUNNER_SHA256="$(file_sha256 "$ROOT/scripts/run_v98_history_polarity_4node_32gpu.sh")"
VIDEO_AUDITOR_SHA256="$(file_sha256 "$ROOT/scripts/audit_indexed_videos.py")"
METHOD_CONTRACT_SHA256="$(
    printf '%s\n' "${CELLS[@]}" | sha256sum | awk '{print $1}'
)"
PRIMARY_MANIFEST_SHA256=""
PRIMARY_EXPERIMENT_CONTRACT_SHA256=""
PRIMARY_ANALYSIS_SHA256=""
PRIMARY_BLIND_FROZEN_SHA256=""
PRIMARY_BLIND_VERIFICATION_SHA256=""
PRIMARY_BLIND_COMPLETION_SHA256=""
PRIMARY_BLIND_SCORECARD_SHA256=""
PRIMARY_BLIND_KEY_SHA256=""
if [[ "$PHASE" == "followup_v78" ]]; then
    PRIMARY_MANIFEST_SHA256="$(file_sha256 "$PRIMARY_MANIFEST_PATH")"
    PRIMARY_EXPERIMENT_CONTRACT_SHA256="$(
        file_sha256 "$PRIMARY_EXPERIMENT_CONTRACT_PATH"
    )"
    PRIMARY_ANALYSIS_SHA256="$(file_sha256 "$PRIMARY_ANALYSIS_PATH")"
    PRIMARY_BLIND_FROZEN_SHA256="$(
        file_sha256 "$PRIMARY_BLIND_FROZEN_PATH"
    )"
    PRIMARY_BLIND_VERIFICATION_SHA256="$(
        file_sha256 "$PRIMARY_BLIND_VERIFICATION_PATH"
    )"
    PRIMARY_BLIND_COMPLETION_SHA256="$(
        file_sha256 "$PRIMARY_BLIND_COMPLETION_PATH"
    )"
    PRIMARY_BLIND_SCORECARD_SHA256="$(
        file_sha256 "$PRIMARY_BLIND_SCORECARD_PATH"
    )"
    PRIMARY_BLIND_KEY_SHA256="$(file_sha256 "$PRIMARY_BLIND_KEY_PATH")"
fi

EXPERIMENT_CONTRACT_JSON="$RUN_ROOT/experiment_contract.json"
CONTRACT_JSON_TMP="$RUN_ROOT/nodes/.experiment_contract.node${NODE_RANK}.$$.tmp"
PHASE_VALUE="$PHASE" \
MODE_VALUE="$MODE" \
RUN_COMMIT_VALUE="$RUN_COMMIT" \
RUN_DIRTY_VALUE="$RUN_DIRTY" \
TRACKED_WORKTREE_DIRTY_VALUE=0 \
RUN_DIFF_SHA256_VALUE="$RUN_DIFF_SHA256" \
RUN_STATUS_SHA256_VALUE="$RUN_STATUS_SHA256" \
RUNNER_SHA256_VALUE="$RUNNER_SHA256" \
SF_INFERENCE_SHA256_VALUE="$SF_INFERENCE_SHA256" \
PF_INFERENCE_SHA256_VALUE="$PF_INFERENCE_SHA256" \
VIDEO_AUDITOR_SHA256_VALUE="$VIDEO_AUDITOR_SHA256" \
PRELOAD_PYRAMIDKV_VALUE="$PRELOAD_PYRAMIDKV" \
POLICY_TRACE_LAYERS_VALUE="$PYRAMIDKV_POLICY_TRACE_LAYERS" \
POLICY_TRACE_STRIDE_VALUE="$PYRAMIDKV_POLICY_TRACE_STRIDE" \
POLICY_TRACE_MAX_RECORDS_VALUE="$PYRAMIDKV_POLICY_TRACE_MAX_RECORDS" \
PROMPTS_VALUE="$PROMPTS" \
PROMPT_SHA256_VALUE="$PROMPT_SHA256" \
PROMPT_COUNT_VALUE="$PROMPT_COUNT" \
FRAMES_VALUE="$FRAMES" \
EXPECTED_VIDEO_FRAMES_VALUE="$EXPECTED_VIDEO_FRAMES" \
EXPECTED_VIDEO_FPS_VALUE="$EXPECTED_VIDEO_FPS" \
EXPECTED_VIDEO_WIDTH_VALUE="$EXPECTED_VIDEO_WIDTH" \
EXPECTED_VIDEO_HEIGHT_VALUE="$EXPECTED_VIDEO_HEIGHT" \
VIDEO_FPS_TOLERANCE_VALUE="$VIDEO_FPS_TOLERANCE" \
SEED_VALUE="$SEED" \
SHARD_SIZE_VALUE="$SHARD_SIZE" \
METHOD_CONTRACT_SHA256_VALUE="$METHOD_CONTRACT_SHA256" \
SF_CONFIG_VALUE="$SF_CONFIG" \
SF_CONFIG_SHA256_VALUE="$SF_CONFIG_SHA256" \
PF_CONFIG_VALUE="$PF_CONFIG" \
PF_CONFIG_SHA256_VALUE="$PF_CONFIG_SHA256" \
SF_CHECKPOINT_VALUE="$SF_CHECKPOINT" \
SF_CHECKPOINT_SHA256_VALUE="$SF_CHECKPOINT_SHA256" \
PF_CHECKPOINT_VALUE="$PF_CHECKPOINT" \
PF_CHECKPOINT_SHA256_VALUE="$PF_CHECKPOINT_SHA256" \
PF_LABELS_VALUE="$PF_LABELS" \
PF_LABELS_SHA256_VALUE="$PF_LABELS_SHA256" \
POLARITY_ZERO_VALUE="$POLARITY_ZERO" \
POLARITY_ZERO_SHA256_VALUE="$POLARITY_ZERO_SHA256" \
POLARITY_ZERO_RANDOM_VALUE="$POLARITY_ZERO_RANDOM" \
POLARITY_ZERO_RANDOM_SHA256_VALUE="$POLARITY_ZERO_RANDOM_SHA256" \
POSITIVE_HALF_VALUE="$POSITIVE_HALF" \
POSITIVE_HALF_SHA256_VALUE="$POSITIVE_HALF_SHA256" \
PF_AW_VALUE="$PF_AW" \
PF_AW_SHA256_VALUE="$PF_AW_SHA256" \
SCORES_VALUE="$SCORES" \
SCORE_SHA256_VALUE="$SCORE_SHA256" \
SCORE_ARTIFACT_VALUE="$SCORE_ARTIFACT" \
SCORE_ARTIFACT_SHA256_VALUE="$SCORE_ARTIFACT_SHA256" \
EXPECTED_SCORE_ARTIFACT_VERSION_VALUE="$EXPECTED_SCORE_ARTIFACT_VERSION" \
EXPECTED_SCORE_ARTIFACT_METHOD_VALUE="$EXPECTED_SCORE_ARTIFACT_METHOD" \
EXPECTED_SCORE_PRIMARY_FIELD_VALUE="$EXPECTED_SCORE_PRIMARY_FIELD" \
MAP_MANIFEST_VALUE="$MAP_MANIFEST" \
MAP_MANIFEST_SHA256_VALUE="$MAP_MANIFEST_SHA256" \
PRIMARY_MANIFEST_SHA256_VALUE="$PRIMARY_MANIFEST_SHA256" \
PRIMARY_MANIFEST_PATH_VALUE="$PRIMARY_MANIFEST_PATH" \
PRIMARY_EXPERIMENT_CONTRACT_PATH_VALUE="$PRIMARY_EXPERIMENT_CONTRACT_PATH" \
PRIMARY_EXPERIMENT_CONTRACT_SHA256_VALUE="$PRIMARY_EXPERIMENT_CONTRACT_SHA256" \
PRIMARY_ANALYSIS_PATH_VALUE="$PRIMARY_ANALYSIS_PATH" \
PRIMARY_ANALYSIS_SHA256_VALUE="$PRIMARY_ANALYSIS_SHA256" \
PRIMARY_BLIND_FROZEN_PATH_VALUE="$PRIMARY_BLIND_FROZEN_PATH" \
PRIMARY_BLIND_FROZEN_SHA256_VALUE="$PRIMARY_BLIND_FROZEN_SHA256" \
PRIMARY_BLIND_VERIFICATION_PATH_VALUE="$PRIMARY_BLIND_VERIFICATION_PATH" \
PRIMARY_BLIND_VERIFICATION_SHA256_VALUE="$PRIMARY_BLIND_VERIFICATION_SHA256" \
PRIMARY_BLIND_COMPLETION_PATH_VALUE="$PRIMARY_BLIND_COMPLETION_PATH" \
PRIMARY_BLIND_COMPLETION_SHA256_VALUE="$PRIMARY_BLIND_COMPLETION_SHA256" \
PRIMARY_BLIND_SCORECARD_PATH_VALUE="$PRIMARY_BLIND_SCORECARD_PATH" \
PRIMARY_BLIND_SCORECARD_SHA256_VALUE="$PRIMARY_BLIND_SCORECARD_SHA256" \
PRIMARY_BLIND_KEY_PATH_VALUE="$PRIMARY_BLIND_KEY_PATH" \
PRIMARY_BLIND_KEY_SHA256_VALUE="$PRIMARY_BLIND_KEY_SHA256" \
SCREEN_GATE_ANALYSIS_PATH_VALUE="$SCREEN_GATE_ANALYSIS_PATH" \
SCREEN_GATE_ANALYSIS_SHA256_VALUE="$SCREEN_GATE_ANALYSIS_SHA256" \
SCREEN_GATE_CONTRACT_PATH_VALUE="$SCREEN_GATE_CONTRACT_PATH" \
SCREEN_GATE_CONTRACT_SHA256_VALUE="$SCREEN_GATE_CONTRACT_SHA256" \
python - "$CONTRACT_JSON_TMP" "${CELLS[@]}" <<'PY' || exit 2
import csv
import hashlib
import json
import os
import pathlib
import sys


def env(name):
    return os.environ[name]


def canonical(path):
    return str(pathlib.Path(path).resolve())


def strategy(name, **params):
    return {"name": name, "params": params}


pf_native_policies = {
    "-1": {
        "sink_frames": 1,
        "recent_frames": 4,
        "policy_type": "osc",
        "strategies": [
            strategy(
                "CyclicStrategy",
                period=6,
                bucket_cap=4,
                dynamic_rope=True,
            )
        ],
        "max_union_frames": 4,
    },
    "1": {
        "sink_frames": 3,
        "recent_frames": 4,
        "policy_type": "stride",
        "strategies": [
            strategy(
                "StrideStrategy",
                interval=6,
                capacity=4,
                dynamic_rope=True,
            )
        ],
        "max_union_frames": 4,
    },
    "2": {
        "sink_frames": 3,
        "recent_frames": 4,
        "policy_type": "merge",
        "strategies": [
            strategy(
                "MergeStrategy",
                patch_size=2,
                block_frames=4,
                capacity=4,
                dynamic_rope=True,
            )
        ],
        "max_union_frames": 4,
    },
}
history_hybrid_policies = {
    "10": {
        "sink_frames": 3,
        "recent_frames": 4,
        "policy_type": "stride",
        "strategies": [
            strategy(
                "CyclicStrategy",
                period=6,
                bucket_cap=2,
                dynamic_rope=True,
            ),
            strategy(
                "StrideStrategy",
                interval=6,
                capacity=2,
                dynamic_rope=True,
            ),
        ],
        "max_union_frames": 4,
    },
    "11": {
        "sink_frames": 3,
        "recent_frames": 4,
        "policy_type": "merge",
        "strategies": [
            strategy(
                "MergeStrategy",
                patch_size=2,
                block_frames=4,
                capacity=4,
                dynamic_rope=True,
            )
        ],
        "max_union_frames": 4,
    },
}
history_stride_policies = {
    "10": {
        "sink_frames": 3,
        "recent_frames": 4,
        "policy_type": "stride",
        "strategies": [
            strategy(
                "StrideStrategy",
                interval=6,
                capacity=4,
                dynamic_rope=True,
            )
        ],
        "max_union_frames": 4,
    },
    "11": history_hybrid_policies["11"],
}


label_contracts = {
    canonical(env("PF_LABELS_VALUE")): (
        "pf_labels",
        env("PF_LABELS_SHA256_VALUE"),
    ),
    canonical(env("POLARITY_ZERO_VALUE")): (
        "history_polarity_zero",
        env("POLARITY_ZERO_SHA256_VALUE"),
    ),
    canonical(env("POLARITY_ZERO_RANDOM_VALUE")): (
        "history_polarity_zero_random",
        env("POLARITY_ZERO_RANDOM_SHA256_VALUE"),
    ),
    canonical(env("POSITIVE_HALF_VALUE")): (
        "positive_rate_half",
        env("POSITIVE_HALF_SHA256_VALUE"),
    ),
    canonical(env("PF_AW_VALUE")): (
        "pf_aw_binary_control",
        env("PF_AW_SHA256_VALUE"),
    ),
}
route_parameters = {
    "none": {},
    "native": {},
    "pf_explicit_parity": {
        "binary_stable_policy": "stride",
        "binary_responsive_policy": "cyclic",
    },
    "history_hybrid_merge": {
        "history_polarity": True,
        "history_support_policy": "hybrid",
        "history_suppress_policy": "merge",
    },
    "history_stride_merge": {
        "history_polarity": True,
        "history_support_policy": "stride",
        "history_suppress_policy": "merge",
    },
}
transition_parameters = {
    "mode": "full",
    "min_reliability": 0.55,
    "min_novelty": 0.01,
    "max_commit_fraction": 0.75,
    "stagger_period": 1,
    "max_age_blocks": 6,
    "branches": "cond",
    "denoise_weight": 2.0,
}
methods = []
for method_index, cell in enumerate(sys.argv[2:]):
    name, engine, raw_labels, route, raw_transition = cell.split("|")
    if route not in route_parameters:
        raise SystemExit(f"unknown route in cell contract: {route}")
    if raw_labels:
        label_path = canonical(raw_labels)
        if label_path not in label_contracts:
            raise SystemExit(f"unregistered label path in cell contract: {label_path}")
        map_key, label_sha256 = label_contracts[label_path]
    else:
        label_path = ""
        label_sha256 = ""
        map_key = None
    transition = raw_transition == "1"
    if engine == "sf":
        expected_labels = None
        policies = None
    elif route in {"native", "pf_explicit_parity"}:
        expected_labels = [-1, 1, 2]
        policies = pf_native_policies
    elif route == "history_hybrid_merge":
        with pathlib.Path(label_path).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            expected_labels = sorted(
                {
                    int(value)
                    for row in csv.reader(handle)
                    for value in row
                }
            )
        if not expected_labels or not set(expected_labels).issubset({10, 11}):
            raise SystemExit(
                f"invalid history labels for {name}: {expected_labels}"
            )
        policies = {
            str(label): history_hybrid_policies[str(label)]
            for label in expected_labels
        }
    elif route == "history_stride_merge":
        with pathlib.Path(label_path).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            expected_labels = sorted(
                {
                    int(value)
                    for row in csv.reader(handle)
                    for value in row
                }
            )
        if not expected_labels or not set(expected_labels).issubset({10, 11}):
            raise SystemExit(
                f"invalid history labels for {name}: {expected_labels}"
            )
        policies = {
            str(label): history_stride_policies[str(label)]
            for label in expected_labels
        }
    else:
        raise SystemExit(f"missing policy contract for route {route}")
    methods.append(
        {
            "method_index": method_index,
            "name": name,
            "engine": engine,
            "route": route,
            "route_parameters": route_parameters[route],
            "map_key": map_key,
            "map_path": label_path or None,
            "map_sha256": label_sha256 or None,
            "transition": {
                "enabled": transition,
                "branches": ["cond"] if transition else [],
                "parameters": transition_parameters if transition else None,
            },
            "expected_labels": expected_labels,
            "policies": policies,
            "few_step_cfg_enabled": False,
            "policy_trace_branches": ["cond"] if engine == "pf" else [],
        }
    )

payload = {
    "version": 2,
    "experiment": "v98_history_polarity",
    "phase": env("PHASE_VALUE"),
    "mode": env("MODE_VALUE"),
    "run_commit": env("RUN_COMMIT_VALUE"),
    "tracked_worktree_dirty": (
        env("TRACKED_WORKTREE_DIRTY_VALUE") == "1"
    ),
    "worktree": {
        "any_dirty": env("RUN_DIRTY_VALUE") == "1",
        "diff_sha256": env("RUN_DIFF_SHA256_VALUE"),
        "status_sha256": env("RUN_STATUS_SHA256_VALUE"),
    },
    "implementation": {
        "runner_sha256": env("RUNNER_SHA256_VALUE"),
        "sf_inference_sha256": env("SF_INFERENCE_SHA256_VALUE"),
        "pf_inference_sha256": env("PF_INFERENCE_SHA256_VALUE"),
        "video_auditor_sha256": env("VIDEO_AUDITOR_SHA256_VALUE"),
    },
    "prompt": {
        "path": canonical(env("PROMPTS_VALUE")),
        "sha256": env("PROMPT_SHA256_VALUE"),
        "count": int(env("PROMPT_COUNT_VALUE")),
        "seed": int(env("SEED_VALUE")),
        "reseed_per_prompt": True,
    },
    "frames": int(env("FRAMES_VALUE")),
    "seed": int(env("SEED_VALUE")),
    "shards": 4,
    "few_step_cfg_enabled": False,
    "video": {
        "latent_frames": int(env("FRAMES_VALUE")),
        "decoded_frames": int(env("EXPECTED_VIDEO_FRAMES_VALUE")),
        "fps": float(env("EXPECTED_VIDEO_FPS_VALUE")),
        "width": int(env("EXPECTED_VIDEO_WIDTH_VALUE")),
        "height": int(env("EXPECTED_VIDEO_HEIGHT_VALUE")),
        "fps_tolerance": float(env("VIDEO_FPS_TOLERANCE_VALUE")),
        "sample_index": 0,
    },
    "sharding": {
        "shards": 4,
        "shard_size": int(env("SHARD_SIZE_VALUE")),
        "method_index_expression": "local_slot",
        "shard_expression": "NODE_RANK",
        "interval_expression": (
            "[NODE_RANK * shard_size, (NODE_RANK + 1) * shard_size)"
        ),
        "gpu_slot_expression": (
            "((local_slot + [0,2,5,7][NODE_RANK]) % 8)"
            if env("PHASE_VALUE") == "primary"
            else (
                "2*floor(NODE_RANK/2)"
                "+((local_slot+NODE_RANK)%2)"
            )
        ),
    },
    "runtime": {
        "few_step_cfg_enabled": False,
        "python_reference_path": True,
        "preload_pyramidkv_extension": env("PRELOAD_PYRAMIDKV_VALUE") == "1",
        "policy_trace": {
            "layers": [
                int(value)
                for value in env("POLICY_TRACE_LAYERS_VALUE").split(",")
                if value
            ],
            "stride": int(env("POLICY_TRACE_STRIDE_VALUE")),
            "max_records": int(env("POLICY_TRACE_MAX_RECORDS_VALUE")),
        },
        "environment": {
            "PYRAMIDKV_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_PACK": "0",
            "PYRAMIDKV_USE_CPP_PACK_OUTPUT": "0",
            "PYRAMIDKV_USE_MEGA_CACHE": "0",
            "PYRAMIDKV_USE_MEGA_ATTN": "0",
            "PYRAMIDKV_CONTIG_ANCHOR_STORE": "0",
        },
    },
    "score": {
        "artifact_path": canonical(env("SCORE_ARTIFACT_VALUE")),
        "artifact_sha256": env("SCORE_ARTIFACT_SHA256_VALUE"),
        "csv_path": canonical(env("SCORES_VALUE")),
        "csv_sha256": env("SCORE_SHA256_VALUE"),
        "map_manifest_path": canonical(env("MAP_MANIFEST_VALUE")),
        "map_manifest_sha256": env("MAP_MANIFEST_SHA256_VALUE"),
        "artifact_version": int(env("EXPECTED_SCORE_ARTIFACT_VERSION_VALUE")),
        "artifact_method": env("EXPECTED_SCORE_ARTIFACT_METHOD_VALUE"),
        "artifact_accepted": True,
        "primary_field": env("EXPECTED_SCORE_PRIMARY_FIELD_VALUE"),
        "probe_policy_balanced": True,
        "probe_policies": ["uniform_stride", "uniform_merge"],
        "bootstrap_unit": "counterfactual_prompt_pair",
    },
    "inputs": {
        "scores": {
            "path": canonical(env("SCORES_VALUE")),
            "sha256": env("SCORE_SHA256_VALUE"),
        },
        "score_artifact": {
            "path": canonical(env("SCORE_ARTIFACT_VALUE")),
            "sha256": env("SCORE_ARTIFACT_SHA256_VALUE"),
            "version": int(env("EXPECTED_SCORE_ARTIFACT_VERSION_VALUE")),
            "method": env("EXPECTED_SCORE_ARTIFACT_METHOD_VALUE"),
            "accepted": True,
            "primary_field": env("EXPECTED_SCORE_PRIMARY_FIELD_VALUE"),
            "probe_policy_balanced": True,
            "probe_policies": ["uniform_stride", "uniform_merge"],
            "bootstrap_unit": "counterfactual_prompt_pair",
        },
        "map_manifest": {
            "path": canonical(env("MAP_MANIFEST_VALUE")),
            "sha256": env("MAP_MANIFEST_SHA256_VALUE"),
        },
        "sf_config": {
            "path": canonical(env("SF_CONFIG_VALUE")),
            "sha256": env("SF_CONFIG_SHA256_VALUE"),
        },
        "pf_config": {
            "path": canonical(env("PF_CONFIG_VALUE")),
            "sha256": env("PF_CONFIG_SHA256_VALUE"),
        },
        "sf_checkpoint": {
            "path": canonical(env("SF_CHECKPOINT_VALUE")),
            "sha256": env("SF_CHECKPOINT_SHA256_VALUE"),
        },
        "pf_checkpoint": {
            "path": canonical(env("PF_CHECKPOINT_VALUE")),
            "sha256": env("PF_CHECKPOINT_SHA256_VALUE"),
        },
    },
    "method_contract_sha256": env("METHOD_CONTRACT_SHA256_VALUE"),
    "primary_manifest_sha256": env("PRIMARY_MANIFEST_SHA256_VALUE") or None,
    "methods": methods,
}
if env("PHASE_VALUE") == "followup_v78":
    primary_gate_evidence = {
        "primary_manifest": {
            "path": canonical(env("PRIMARY_MANIFEST_PATH_VALUE")),
            "sha256": env("PRIMARY_MANIFEST_SHA256_VALUE"),
        },
        "primary_experiment_contract": {
            "path": canonical(
                env("PRIMARY_EXPERIMENT_CONTRACT_PATH_VALUE")
            ),
            "sha256": env(
                "PRIMARY_EXPERIMENT_CONTRACT_SHA256_VALUE"
            ),
        },
        "primary_analysis": {
            "path": canonical(env("PRIMARY_ANALYSIS_PATH_VALUE")),
            "sha256": env("PRIMARY_ANALYSIS_SHA256_VALUE"),
        },
        "primary_blind_frozen": {
            "path": canonical(env("PRIMARY_BLIND_FROZEN_PATH_VALUE")),
            "sha256": env("PRIMARY_BLIND_FROZEN_SHA256_VALUE"),
        },
        "primary_blind_verification": {
            "path": canonical(
                env("PRIMARY_BLIND_VERIFICATION_PATH_VALUE")
            ),
            "sha256": env(
                "PRIMARY_BLIND_VERIFICATION_SHA256_VALUE"
            ),
        },
        "primary_blind_completion": {
            "path": canonical(
                env("PRIMARY_BLIND_COMPLETION_PATH_VALUE")
            ),
            "sha256": env(
                "PRIMARY_BLIND_COMPLETION_SHA256_VALUE"
            ),
        },
        "primary_blind_scorecard": {
            "path": canonical(
                env("PRIMARY_BLIND_SCORECARD_PATH_VALUE")
            ),
            "sha256": env(
                "PRIMARY_BLIND_SCORECARD_SHA256_VALUE"
            ),
        },
        "primary_blind_key": {
            "path": canonical(env("PRIMARY_BLIND_KEY_PATH_VALUE")),
            "sha256": env("PRIMARY_BLIND_KEY_SHA256_VALUE"),
        },
    }
    payload["primary_gate_evidence"] = primary_gate_evidence
    payload["inputs"].update(primary_gate_evidence)
else:
    payload["primary_gate_evidence"] = None
if env("MODE_VALUE") == "main128":
    screen_gate_evidence = {
        "screen32_analysis": {
            "path": canonical(env("SCREEN_GATE_ANALYSIS_PATH_VALUE")),
            "sha256": env("SCREEN_GATE_ANALYSIS_SHA256_VALUE"),
        },
        "screen32_experiment_contract": {
            "path": canonical(env("SCREEN_GATE_CONTRACT_PATH_VALUE")),
            "sha256": env("SCREEN_GATE_CONTRACT_SHA256_VALUE"),
        },
    }
    screen_contract = json.loads(
        pathlib.Path(
            env("SCREEN_GATE_CONTRACT_PATH_VALUE")
        ).read_text(encoding="utf-8")
    )
    if screen_contract.get("run_commit") != payload["run_commit"]:
        raise SystemExit("[error] screen32 and main128 commits differ")
    for key in ("implementation", "runtime", "video", "frames", "seed"):
        if screen_contract.get(key) != payload.get(key):
            raise SystemExit(
                f"[error] screen32/main128 protocol differs for {key}"
            )
    for key in (
        "artifact_sha256",
        "csv_sha256",
        "artifact_version",
        "artifact_method",
        "artifact_accepted",
        "primary_field",
        "probe_policy_balanced",
        "probe_policies",
        "bootstrap_unit",
    ):
        if screen_contract.get("score", {}).get(key) != payload["score"].get(key):
            raise SystemExit(
                f"[error] screen32/main128 score protocol differs for {key}"
            )

    def normalized_method(item):
        return {
            key: item.get(key)
            for key in (
                "method_index",
                "name",
                "engine",
                "route",
                "route_parameters",
                "map_key",
                "map_sha256",
                "transition",
                "expected_labels",
                "policies",
                "few_step_cfg_enabled",
                "policy_trace_branches",
            )
        }

    if (
        [normalized_method(item) for item in screen_contract.get("methods", [])]
        != [normalized_method(item) for item in payload["methods"]]
    ):
        raise SystemExit(
            "[error] screen32/main128 method policies or maps differ"
        )
    payload["screen32_gate_evidence"] = screen_gate_evidence
    payload["inputs"].update(screen_gate_evidence)
else:
    payload["screen32_gate_evidence"] = None
fingerprint_payload = dict(payload)
payload["run_fingerprint"] = hashlib.sha256(
    json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

publish_json_contract() {
    local lock="$RUN_ROOT/.experiment_json_contract_lock"
    local waited=0
    if mkdir "$lock" 2>/dev/null; then
        if [[ -e "$EXPERIMENT_CONTRACT_JSON" ]]; then
            if ! cmp -s "$CONTRACT_JSON_TMP" "$EXPERIMENT_CONTRACT_JSON"; then
                echo "[error] refusing mixed experiment_contract.json"
                diff -u "$EXPERIMENT_CONTRACT_JSON" "$CONTRACT_JSON_TMP" || true
                rmdir "$lock" 2>/dev/null || true
                return 1
            fi
            rm -f "$CONTRACT_JSON_TMP"
        else
            mv "$CONTRACT_JSON_TMP" "$EXPERIMENT_CONTRACT_JSON"
        fi
        rmdir "$lock" 2>/dev/null || true
    else
        while [[ ! -s "$EXPERIMENT_CONTRACT_JSON" && "$waited" -lt "$CONTRACT_WAIT_SECONDS" ]]; do
            sleep 1
            waited=$((waited + 1))
        done
        [[ -s "$EXPERIMENT_CONTRACT_JSON" ]] || {
            echo "[error] timed out waiting for experiment_contract.json"
            return 1
        }
        if ! cmp -s "$CONTRACT_JSON_TMP" "$EXPERIMENT_CONTRACT_JSON"; then
            echo "[error] node $NODE_RANK has a different experiment_contract.json"
            diff -u "$EXPERIMENT_CONTRACT_JSON" "$CONTRACT_JSON_TMP" || true
            return 1
        fi
        rm -f "$CONTRACT_JSON_TMP"
    fi
}
publish_json_contract || {
    rm -f "$CONTRACT_JSON_TMP"
    exit 2
}
EXPERIMENT_CONTRACT_SHA256="$(file_sha256 "$EXPERIMENT_CONTRACT_JSON")"

GLOBAL_MANIFEST="$RUN_ROOT/experiment_manifest.env"
CONTRACT_TMP="$RUN_ROOT/nodes/.experiment.node${NODE_RANK}.$$.tmp"
{
    printf 'CONTRACT_VERSION=3\n'
    printf 'EXPERIMENT=v98_history_polarity\n'
    printf 'PHASE=%s\n' "$PHASE"
    printf 'MODE=%s\n' "$MODE"
    printf 'RUN_COMMIT=%s\n' "$RUN_COMMIT"
    printf 'RUN_DIRTY=%s\n' "$RUN_DIRTY"
    printf 'RUN_DIFF_SHA256=%s\n' "$RUN_DIFF_SHA256"
    printf 'RUN_STATUS_SHA256=%s\n' "$RUN_STATUS_SHA256"
    printf 'RUNNER_SHA256=%s\n' "$RUNNER_SHA256"
    printf 'SF_INFERENCE_SHA256=%s\n' "$SF_INFERENCE_SHA256"
    printf 'PF_INFERENCE_SHA256=%s\n' "$PF_INFERENCE_SHA256"
    printf 'VIDEO_AUDITOR_SHA256=%s\n' "$VIDEO_AUDITOR_SHA256"
    printf 'EXPERIMENT_CONTRACT_JSON=%s\n' "$EXPERIMENT_CONTRACT_JSON"
    printf 'EXPERIMENT_CONTRACT_SHA256=%s\n' "$EXPERIMENT_CONTRACT_SHA256"
    printf 'PROMPTS=%s\n' "$PROMPTS"
    printf 'PROMPT_SHA256=%s\n' "$PROMPT_SHA256"
    printf 'PROMPT_COUNT=%s\n' "$PROMPT_COUNT"
    printf 'FRAMES=%s\n' "$FRAMES"
    printf 'EXPECTED_VIDEO_FRAMES=%s\n' "$EXPECTED_VIDEO_FRAMES"
    printf 'EXPECTED_VIDEO_FPS=%s\n' "$EXPECTED_VIDEO_FPS"
    printf 'EXPECTED_VIDEO_WIDTH=%s\n' "$EXPECTED_VIDEO_WIDTH"
    printf 'EXPECTED_VIDEO_HEIGHT=%s\n' "$EXPECTED_VIDEO_HEIGHT"
    printf 'VIDEO_FPS_TOLERANCE=%s\n' "$VIDEO_FPS_TOLERANCE"
    printf 'SEED=%s\n' "$SEED"
    printf 'SCORES=%s\n' "$SCORES"
    printf 'SCORE_SHA256=%s\n' "$SCORE_SHA256"
    printf 'SCORE_ARTIFACT=%s\n' "$SCORE_ARTIFACT"
    printf 'SCORE_ARTIFACT_SHA256=%s\n' "$SCORE_ARTIFACT_SHA256"
    printf 'SCORE_ARTIFACT_VERSION=%s\n' "$EXPECTED_SCORE_ARTIFACT_VERSION"
    printf 'SCORE_ARTIFACT_METHOD=%s\n' "$EXPECTED_SCORE_ARTIFACT_METHOD"
    printf 'SCORE_ARTIFACT_ACCEPTED=true\n'
    printf 'SCORE_PRIMARY_FIELD=%s\n' "$EXPECTED_SCORE_PRIMARY_FIELD"
    printf 'SF_CONFIG=%s\n' "$SF_CONFIG"
    printf 'SF_CONFIG_SHA256=%s\n' "$SF_CONFIG_SHA256"
    printf 'PF_CONFIG=%s\n' "$PF_CONFIG"
    printf 'PF_CONFIG_SHA256=%s\n' "$PF_CONFIG_SHA256"
    printf 'SF_CHECKPOINT=%s\n' "$SF_CHECKPOINT"
    printf 'SF_CHECKPOINT_SHA256=%s\n' "$SF_CHECKPOINT_SHA256"
    printf 'PF_CHECKPOINT=%s\n' "$PF_CHECKPOINT"
    printf 'PF_CHECKPOINT_SHA256=%s\n' "$PF_CHECKPOINT_SHA256"
    printf 'PF_LABELS=%s\n' "$PF_LABELS"
    printf 'PF_LABELS_SHA256=%s\n' "$PF_LABELS_SHA256"
    printf 'MAP_MANIFEST=%s\n' "$MAP_MANIFEST"
    printf 'MAP_MANIFEST_SHA256=%s\n' "$MAP_MANIFEST_SHA256"
    printf 'POLARITY_ZERO_SHA256=%s\n' "$POLARITY_ZERO_SHA256"
    printf 'POLARITY_ZERO_RANDOM_SHA256=%s\n' "$POLARITY_ZERO_RANDOM_SHA256"
    printf 'POSITIVE_HALF_SHA256=%s\n' "$POSITIVE_HALF_SHA256"
    printf 'PF_AW_SHA256=%s\n' "$PF_AW_SHA256"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'METHOD_CONTRACT_SHA256=%s\n' "$METHOD_CONTRACT_SHA256"
    printf 'SHARDS=4\n'
    printf 'SHARD_SIZE=%s\n' "$SHARD_SIZE"
    printf 'MAPPING=method_index=local_slot;shard=NODE_RANK\n'
    if [[ "$PHASE" == "primary" ]]; then
        printf 'GPU_SLOT_MAPPING=((local_slot + [0,2,5,7][NODE_RANK]) %% 8)\n'
    else
        printf 'GPU_SLOT_MAPPING=2*floor(NODE_RANK/2)+((local_slot+NODE_RANK)%%2)\n'
    fi
    printf 'FEW_STEP_CFG_ENABLED=0\n'
    printf 'PRELOAD_PYRAMIDKV=%s\n' "$PRELOAD_PYRAMIDKV"
    printf 'POLICY_TRACE_LAYERS=%s\n' "$PYRAMIDKV_POLICY_TRACE_LAYERS"
    printf 'POLICY_TRACE_STRIDE=%s\n' "$PYRAMIDKV_POLICY_TRACE_STRIDE"
    printf 'POLICY_TRACE_MAX_RECORDS=%s\n' "$PYRAMIDKV_POLICY_TRACE_MAX_RECORDS"
    printf 'PYRAMIDKV_CPP_STRATEGY=0\n'
    printf 'PYRAMIDKV_USE_CPP_PACK=0\n'
    printf 'PYRAMIDKV_USE_CPP_PACK_OUTPUT=0\n'
    printf 'PYRAMIDKV_USE_MEGA_CACHE=0\n'
    printf 'PYRAMIDKV_USE_MEGA_ATTN=0\n'
    printf 'PYRAMIDKV_CONTIG_ANCHOR_STORE=0\n'
    printf 'PRIMARY_MANIFEST_SHA256=%s\n' "$PRIMARY_MANIFEST_SHA256"
    printf 'PRIMARY_EXPERIMENT_CONTRACT_SHA256=%s\n' \
        "$PRIMARY_EXPERIMENT_CONTRACT_SHA256"
    printf 'PRIMARY_ANALYSIS_SHA256=%s\n' "$PRIMARY_ANALYSIS_SHA256"
    printf 'PRIMARY_BLIND_FROZEN_SHA256=%s\n' \
        "$PRIMARY_BLIND_FROZEN_SHA256"
    printf 'PRIMARY_BLIND_VERIFICATION_SHA256=%s\n' \
        "$PRIMARY_BLIND_VERIFICATION_SHA256"
    printf 'PRIMARY_BLIND_COMPLETION_SHA256=%s\n' \
        "$PRIMARY_BLIND_COMPLETION_SHA256"
    printf 'PRIMARY_BLIND_SCORECARD_SHA256=%s\n' \
        "$PRIMARY_BLIND_SCORECARD_SHA256"
    printf 'PRIMARY_BLIND_KEY_SHA256=%s\n' "$PRIMARY_BLIND_KEY_SHA256"
    printf 'SCREEN_GATE_ANALYSIS_SHA256=%s\n' \
        "$SCREEN_GATE_ANALYSIS_SHA256"
    printf 'SCREEN_GATE_CONTRACT_SHA256=%s\n' \
        "$SCREEN_GATE_CONTRACT_SHA256"
} >"$CONTRACT_TMP"

publish_global_contract() {
    local lock="$RUN_ROOT/.experiment_contract_lock"
    local waited=0
    if mkdir "$lock" 2>/dev/null; then
        if [[ -e "$GLOBAL_MANIFEST" ]]; then
            if ! cmp -s "$CONTRACT_TMP" "$GLOBAL_MANIFEST"; then
                echo "[error] refusing mixed cross-node experiment contract"
                diff -u "$GLOBAL_MANIFEST" "$CONTRACT_TMP" || true
                rmdir "$lock" 2>/dev/null || true
                return 1
            fi
            rm -f "$CONTRACT_TMP"
        else
            mv "$CONTRACT_TMP" "$GLOBAL_MANIFEST"
        fi
        rmdir "$lock" 2>/dev/null || true
    else
        while [[ ! -s "$GLOBAL_MANIFEST" && "$waited" -lt "$CONTRACT_WAIT_SECONDS" ]]; do
            sleep 1
            waited=$((waited + 1))
        done
        [[ -s "$GLOBAL_MANIFEST" ]] || {
            echo "[error] timed out waiting for global experiment contract"
            return 1
        }
        if ! cmp -s "$CONTRACT_TMP" "$GLOBAL_MANIFEST"; then
            echo "[error] node $NODE_RANK does not match the global experiment contract"
            diff -u "$GLOBAL_MANIFEST" "$CONTRACT_TMP" || true
            return 1
        fi
        rm -f "$CONTRACT_TMP"
    fi
}
publish_global_contract || {
    rm -f "$CONTRACT_TMP"
    exit 2
}
GLOBAL_MANIFEST_SHA256="$(file_sha256 "$GLOBAL_MANIFEST")"

NODE_RUN_LOCK="$RUN_ROOT/.node${NODE_RANK}.run_lock"
if ! mkdir "$NODE_RUN_LOCK" 2>/dev/null; then
    echo "[error] another process is already running node $NODE_RANK: $NODE_RUN_LOCK"
    exit 2
fi
cleanup_node_lock() {
    rmdir "$NODE_RUN_LOCK" 2>/dev/null || true
}
trap cleanup_node_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

NODE_MANIFEST="$RUN_ROOT/nodes/node${NODE_RANK}.manifest.env"
NODE_TMP="$RUN_ROOT/nodes/.node${NODE_RANK}.$$.tmp"
{
    printf 'CONTRACT_VERSION=3\n'
    printf 'PHASE=%s\n' "$PHASE"
    printf 'MODE=%s\n' "$MODE"
    printf 'NODE_RANK=%s\n' "$NODE_RANK"
    printf 'ASSIGNED_SHARD=%s\n' "$NODE_RANK"
    printf 'GLOBAL_MANIFEST_SHA256=%s\n' "$GLOBAL_MANIFEST_SHA256"
    printf 'METHODS=%s\n' "${METHODS[*]}"
    printf 'LOCAL_METHOD_COUNT=%s\n' "${#METHODS[@]}"
    printf 'GPU_LIST=%s\n' "$GPU_LIST"
} >"$NODE_TMP"
if [[ -e "$NODE_MANIFEST" ]] && ! cmp -s "$NODE_TMP" "$NODE_MANIFEST"; then
    echo "[error] refusing mixed node manifest for node $NODE_RANK"
    diff -u "$NODE_MANIFEST" "$NODE_TMP" || true
    rm -f "$NODE_TMP"
    exit 2
fi
mv "$NODE_TMP" "$NODE_MANIFEST"
rm -f "$RUN_ROOT/status/node${NODE_RANK}.done"

if [[ "$PRELOAD_PYRAMIDKV" == "1" ]]; then
    (
        cd "$PF" || exit 2
        export CUDA_VISIBLE_DEVICES="${GPUS[0]}"
        python -c "from pyramidkv import _ops; _ops._ensure_loaded(); print('[PyramidKVPreload] ok', flush=True)"
    ) >"$RUN_ROOT/logs/node${NODE_RANK}.pyramidkv_preload.log" 2>&1 || {
        echo "[error] PyramidKV extension preload failed on node $NODE_RANK"
        exit 2
    }
fi

write_config() {
    local name="$1" shard="$2" start="$3" end="$4" engine="$5"
    local labels="$6" route="$7" transition="$8" gpu="$9"
    local label_sha="" config="$RUN_ROOT/configs/$name.shard$shard.env"
    local temporary="$RUN_ROOT/configs/.$name.shard$shard.$$.tmp"
    if [[ -n "$labels" ]]; then
        labels="$(readlink -f "$labels")"
        label_sha="$(file_sha256 "$labels")"
    fi
    {
        printf 'contract_version=3\n'
        printf 'name=%s\n' "$name"
        printf 'phase=%s\n' "$PHASE"
        printf 'mode=%s\n' "$MODE"
        printf 'node_rank=%s\n' "$NODE_RANK"
        printf 'shard=%s\n' "$shard"
        printf 'start_idx=%s\n' "$start"
        printf 'end_idx=%s\n' "$end"
        printf 'gpu=%s\n' "$gpu"
        printf 'engine=%s\n' "$engine"
        printf 'labels=%s\n' "$labels"
        printf 'label_sha256=%s\n' "$label_sha"
        printf 'route=%s\n' "$route"
        printf 'transition=%s\n' "$transition"
        printf 'global_manifest_sha256=%s\n' "$GLOBAL_MANIFEST_SHA256"
        printf 'experiment_contract_sha256=%s\n' "$EXPERIMENT_CONTRACT_SHA256"
        printf 'method_contract_sha256=%s\n' "$METHOD_CONTRACT_SHA256"
        printf 'run_commit=%s\n' "$RUN_COMMIT"
        printf 'run_dirty=%s\n' "$RUN_DIRTY"
        printf 'prompt_sha256=%s\n' "$PROMPT_SHA256"
        printf 'prompt_count=%s\n' "$PROMPT_COUNT"
        printf 'score_sha256=%s\n' "$SCORE_SHA256"
        printf 'score_artifact_sha256=%s\n' "$SCORE_ARTIFACT_SHA256"
        printf 'map_manifest_sha256=%s\n' "$MAP_MANIFEST_SHA256"
        printf 'sf_config_sha256=%s\n' "$SF_CONFIG_SHA256"
        printf 'pf_config_sha256=%s\n' "$PF_CONFIG_SHA256"
        printf 'sf_checkpoint_sha256=%s\n' "$SF_CHECKPOINT_SHA256"
        printf 'pf_checkpoint_sha256=%s\n' "$PF_CHECKPOINT_SHA256"
        printf 'frames=%s\n' "$FRAMES"
        printf 'expected_video_frames=%s\n' "$EXPECTED_VIDEO_FRAMES"
        printf 'expected_video_fps=%s\n' "$EXPECTED_VIDEO_FPS"
        printf 'expected_video_width=%s\n' "$EXPECTED_VIDEO_WIDTH"
        printf 'expected_video_height=%s\n' "$EXPECTED_VIDEO_HEIGHT"
        printf 'video_fps_tolerance=%s\n' "$VIDEO_FPS_TOLERANCE"
        printf 'seed=%s\n' "$SEED"
        printf 'reseed_per_prompt=1\n'
        printf 'few_step_cfg_enabled=0\n'
        printf 'policy_trace_layers=%s\n' "$PYRAMIDKV_POLICY_TRACE_LAYERS"
        printf 'policy_trace_stride=%s\n' "$PYRAMIDKV_POLICY_TRACE_STRIDE"
        printf 'policy_trace_max_records=%s\n' "$PYRAMIDKV_POLICY_TRACE_MAX_RECORDS"
        printf 'python_reference_path=1\n'
    } >"$temporary"
    if [[ -e "$config" ]] && ! cmp -s "$temporary" "$config"; then
        echo "[error] refusing mixed cell config: $config"
        diff -u "$config" "$temporary" || true
        rm -f "$temporary"
        return 2
    fi
    mv "$temporary" "$config"
}

run_cell() {
    local name="$1" gpu="$2" shard="$3" start="$4" end="$5"
    local engine="$6" labels="$7" route="$8" transition="$9"
    local output="$RUN_ROOT/$name"
    local log="$RUN_ROOT/logs/$name.shard$shard.log"
    local marker="$RUN_ROOT/status/$name.shard$shard.done"
    local policy_trace="$RUN_ROOT/traces/$name.shard$shard.policy.jsonl"
    local transition_trace="$RUN_ROOT/traces/$name.shard$shard.transition.jsonl"
    local video_json="$RUN_ROOT/diagnostics/$name.shard$shard.video.json"
    local video_log="$RUN_ROOT/diagnostics/$name.shard$shard.video.log"
    local config="$RUN_ROOT/configs/$name.shard$shard.env"
    local head_args=() route_args=() transition_args=()

    mkdir -p "$output"
    write_config \
        "$name" "$shard" "$start" "$end" "$engine" \
        "$labels" "$route" "$transition" "$gpu" || return $?
    local config_sha
    config_sha="$(file_sha256 "$config")"
    local resume_artifacts_ok=1
    if [[ "$FORCE" != "1" && -s "$marker" ]] && \
        grep -qx "CELL_CONFIG_SHA256=$config_sha" "$marker"; then
        [[ -s "$log" ]] || resume_artifacts_ok=0
        if [[ -s "$log" ]] && grep -Eqi \
            'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|PyramidKVPolicyTraceError' \
            "$log"; then
            resume_artifacts_ok=0
        fi
        if [[ "$engine" == "pf" ]]; then
            [[ -s "$policy_trace" ]] || resume_artifacts_ok=0
            if [[ -s "$policy_trace" ]] && \
                ! grep -q '"branch": "cond"' "$policy_trace"; then
                resume_artifacts_ok=0
            fi
            if [[ -s "$log" ]] && \
                ! grep -q '\[PyramidKVRuntimePolicy\]' "$log"; then
                resume_artifacts_ok=0
            fi
        fi
        if [[ "$route" == history_* && -s "$log" ]] && \
            ! grep -q '\[HistoryPolarityPolicy\].*legacy_pf_labels=false' "$log"; then
            resume_artifacts_ok=0
        fi
        if [[ "$route" == "pf_explicit_parity" && -s "$log" ]] && \
            ! grep -q '\[BinaryPolicyOverride\].*stable=stride.*responsive=cyclic' "$log"; then
            resume_artifacts_ok=0
        fi
        if [[ "$transition" == "1" && ! -s "$transition_trace" ]]; then
            resume_artifacts_ok=0
        fi
        if [[ "$resume_artifacts_ok" == "1" ]] && \
            python "$ROOT/scripts/audit_indexed_videos.py" \
                --video-dir "$output" --start-idx "$start" --end-idx "$end" \
                --expected-frames "$EXPECTED_VIDEO_FRAMES" \
                --expected-fps "$EXPECTED_VIDEO_FPS" \
                --expected-width "$EXPECTED_VIDEO_WIDTH" \
                --expected-height "$EXPECTED_VIDEO_HEIGHT" \
                --fps-tolerance "$VIDEO_FPS_TOLERANCE" \
                --allow-outside-interval --output-json "$video_json" \
                >"$video_log" 2>&1; then
            local resumed_video_fingerprint
            resumed_video_fingerprint="$(
                python -c \
                    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["input_fingerprint"])' \
                    "$video_json"
            )" || resumed_video_fingerprint=""
            if [[ -n "$resumed_video_fingerprint" ]] && \
                grep -qx \
                    "VIDEO_INPUT_FINGERPRINT=$resumed_video_fingerprint" \
                    "$marker"; then
                echo "[skip] phase=$PHASE $name shard=$shard"
                return 0
            fi
        fi
    fi
    rm -f "$marker" "$policy_trace" "$transition_trace" "$video_json"

    if [[ -n "$labels" ]]; then
        head_args=(--pyramidkv_head_config_path "$labels")
    fi
    case "$route" in
        none|native)
            ;;
        pf_explicit_parity)
            route_args=(
                --pyramidkv_binary_stable_policy stride
                --pyramidkv_binary_responsive_policy cyclic
            )
            ;;
        history_hybrid_merge)
            route_args=(
                --pyramidkv_history_polarity
                --pyramidkv_history_support_policy hybrid
                --pyramidkv_history_suppress_policy merge
            )
            ;;
        history_stride_merge)
            route_args=(
                --pyramidkv_history_polarity
                --pyramidkv_history_support_policy stride
                --pyramidkv_history_suppress_policy merge
            )
            ;;
        *)
            echo "[error] unknown route $route"
            return 2
            ;;
    esac
    if [[ "$transition" == "1" ]]; then
        transition_args=(
            --pyramidkv_cache_transition
            --pyramidkv_cache_transition_mode full
            --pyramidkv_cache_transition_min_reliability .55
            --pyramidkv_cache_transition_min_novelty .01
            --pyramidkv_cache_transition_max_commit_fraction .75
            --pyramidkv_cache_transition_stagger_period 1
            --pyramidkv_cache_transition_max_age_blocks 6
            --pyramidkv_cache_transition_branches cond
            --pyramidkv_cache_transition_denoise_weight 2
            --pyramidkv_cache_transition_trace_path "$transition_trace"
            --pyramidkv_cache_transition_debug
        )
    fi

    if [[ "$engine" == "sf" ]]; then
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export COMMIT_FORCING_ENABLE=0
            cd "$SF" || exit 2
            python inference.py \
                --config_path "$SF_CONFIG" \
                --checkpoint_path "$SF_CHECKPOINT" \
                --data_path "$PROMPTS" \
                --output_folder "$output" \
                --num_output_frames "$FRAMES" \
                --seed "$SEED" --num_samples 1 --use_ema --save_with_index \
                --start_idx "$start" --end_idx "$end" --reseed_per_prompt
        ) >"$log" 2>&1
    else
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYRAMIDKV_POLICY_TRACE_PATH="$policy_trace"
            cd "$PF" || exit 2
            python inference.py \
                --config_path "$PF_CONFIG" \
                --checkpoint_path "$PF_CHECKPOINT" \
                --data_path "$PROMPTS" \
                --output_folder "$output" \
                --num_output_frames "$FRAMES" \
                --seed "$SEED" --num_samples 1 --use_ema --save_with_index \
                --start_idx "$start" --end_idx "$end" --reseed_per_prompt \
                "${head_args[@]}" "${route_args[@]}" "${transition_args[@]}"
        ) >"$log" 2>&1
    fi
    local status=$?
    [[ "$status" -eq 0 ]] || return "$status"

    if grep -Eqi \
        'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|PyramidKVPolicyTraceError' \
        "$log"; then
        echo "[error] failure signature in $log"
        return 1
    fi
    python "$ROOT/scripts/audit_indexed_videos.py" \
        --video-dir "$output" --start-idx "$start" --end-idx "$end" \
        --expected-frames "$EXPECTED_VIDEO_FRAMES" \
        --expected-fps "$EXPECTED_VIDEO_FPS" \
        --expected-width "$EXPECTED_VIDEO_WIDTH" \
        --expected-height "$EXPECTED_VIDEO_HEIGHT" \
        --fps-tolerance "$VIDEO_FPS_TOLERANCE" \
        --allow-outside-interval --output-json "$video_json" \
        >"$video_log" 2>&1 || return 1
    if [[ "$engine" == "pf" ]]; then
        [[ -s "$policy_trace" ]] || {
            echo "[error] missing policy trace $policy_trace"
            return 1
        }
        grep -q '\[PyramidKVRuntimePolicy\]' "$log" || {
            echo "[error] missing runtime policy audit in $log"
            return 1
        }
    fi
    if [[ "$route" == history_* ]]; then
        grep -q '\[HistoryPolarityPolicy\].*legacy_pf_labels=false' "$log" || {
            echo "[error] missing neutral history-polarity marker in $log"
            return 1
        }
    fi
    if [[ "$route" == "pf_explicit_parity" ]]; then
        grep -q '\[BinaryPolicyOverride\].*stable=stride.*responsive=cyclic' "$log" || {
            echo "[error] missing PF parity marker in $log"
            return 1
        }
    fi
    if [[ "$transition" == "1" && ! -s "$transition_trace" ]]; then
        echo "[error] missing transition trace $transition_trace"
        return 1
    fi
    local video_fingerprint
    video_fingerprint="$(
        python -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["input_fingerprint"])' \
            "$video_json"
    )" || return 1
    local marker_tmp="$RUN_ROOT/status/.$name.shard$shard.$$.done.tmp"
    {
        printf 'CELL_CONFIG_SHA256=%s\n' "$config_sha"
        printf 'VIDEO_INPUT_FINGERPRINT=%s\n' "$video_fingerprint"
    } >"$marker_tmp"
    mv "$marker_tmp" "$marker"
}

PIDS=()
STATUS=0
for local_slot in "${!METHODS[@]}"; do
    method_index=$local_slot
    shard=$NODE_RANK
    start=$((shard * SHARD_SIZE))
    end=$((start + SHARD_SIZE))
    IFS='|' read -r name engine labels route transition \
        <<<"${CELLS[$method_index]}"
    if [[ "$PHASE" == "primary" ]]; then
        GPU_SLOT_OFFSETS=(0 2 5 7)
        gpu_slot=$(((local_slot + GPU_SLOT_OFFSETS[NODE_RANK]) % 8))
    else
        pair_base=$((2 * (NODE_RANK / 2)))
        gpu_slot=$((pair_base + (local_slot + NODE_RANK) % 2))
    fi
    assigned_gpu="${GPUS[$gpu_slot]}"
    echo "[launch] phase=$PHASE node=$NODE_RANK gpu=$assigned_gpu gpu_slot=$gpu_slot method_index=$method_index method=$name shard=$shard interval=[$start,$end)"
    run_cell \
        "$name" "$assigned_gpu" "$shard" "$start" "$end" \
        "$engine" "$labels" "$route" "$transition" &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
done

if [[ "$STATUS" -eq 0 ]]; then
    NODE_DONE="$RUN_ROOT/status/.node${NODE_RANK}.$$.done.tmp"
    {
        printf 'GLOBAL_MANIFEST_SHA256=%s\n' "$GLOBAL_MANIFEST_SHA256"
        printf 'NODE_MANIFEST_SHA256=%s\n' "$(file_sha256 "$NODE_MANIFEST")"
    } >"$NODE_DONE"
    mv "$NODE_DONE" "$RUN_ROOT/status/node${NODE_RANK}.done"
fi
echo "[v98-generation] phase=$PHASE mode=$MODE node=$NODE_RANK status=$STATUS out=$RUN_ROOT"
exit "$STATUS"
