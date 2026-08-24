#!/usr/bin/env python3
"""Freeze a no-refit Causal-Forcing checkpoint transfer after passing v192."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v190_head_phase_causal_screen import validate_map
from prepare_v192_head_phase_robustness import verify as verify_v192

METHODS = (
    "cf_native_21",
    "cf_all_recent_9ffe",
    "cf_head_phase_transfer",
)
CANDIDATE = "cf_head_phase_transfer"
LOCAL_CONTROL = "cf_all_recent_9ffe"
NATIVE_CONTROL = "cf_native_21"
PROMPT_POSITIONS = tuple(range(1, 128, 2))
PROMPT_COUNT = len(PROMPT_POSITIONS)
SEED = 10000
NUM_OUTPUT_FRAMES = 120
DECODED_VIDEO_CONTRACT = {
    "frames": 477,
    "fps": 16.0,
    "duration_seconds": 29.8125,
    "width": 832,
    "height": 480,
}
REQUIRED_V192_RECOMMENDATION = (
    "freeze_within_model_head_phase_method_for_cross_model_transfer"
)
RUNTIME_FILES = (
    "inference.py",
    "configs/default_config.yaml",
    "configs/pyramid-forcing-native.yaml",
    "configs/pyramid-forcing.yaml",
    "pipeline/causal_inference.py",
    "pipeline/pyramidkv_config.py",
    "wan/modules/causal_model.py",
    "pyramidkv/adaptive_cache.py",
    "pyramidkv/denoise_schedule.py",
    "pyramidkv/policy_overrides.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen v194 artifact differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_json_frozen(path: Path, payload: dict) -> str:
    return write_frozen(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _verify_bound_path(row: dict, key: str, *, label: str) -> Path:
    path = Path(str(row.get(key, "")))
    if not path.is_file() or sha256(path) != row.get(f"{key}_sha256"):
        raise ValueError(f"v194 prerequisite drifted: {label}/{key}")
    return path


def validate_v192(
    decision_path: Path,
    input_manifest_path: Path,
) -> tuple[dict, dict]:
    decision = load_json(decision_path)
    frozen = verify_v192(input_manifest_path)
    gates = decision.get("combined_gates") or {}
    if (
        decision.get("experiment") != "v192_head_phase_seed_length_robustness"
        or decision.get("confirmatory") is not True
        or decision.get("within_model_seed_length_robustness_confirmed") is not True
        or decision.get("recommendation") != REQUIRED_V192_RECOMMENDATION
        or set(gates)
        != {
            "new_seed_scope_pass",
            "two_seed_pooled_positive_effect",
            "long60_scope_pass",
        }
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("v194 requires every frozen v192 robustness gate to pass")
    source = decision.get("source") or {}
    bound_input = _verify_bound_path(source, "input_manifest", label="v192-decision")
    for key in ("seed_report", "long_report"):
        _verify_bound_path(source, key, label="v192-decision")
    if not bound_input.samefile(input_manifest_path):
        raise ValueError("v192 decision is bound to a different input manifest")
    if (
        tuple(decision.get("methods") or ())
        != ("sf_native", "all_recent", "head_phase_joint")
        or tuple(frozen.get("method_order") or ())
        != ("sf_native", "all_recent", "head_phase_joint")
        or decision.get("selected_v190_method") != frozen.get("selected_v190_method")
        or decision.get("selected_operator") != frozen.get("selected_operator")
        or not frozen.get("v191_positive_metrics_to_replicate")
    ):
        raise ValueError("v192 decision and frozen route disagree")
    for method in ("all_recent", "head_phase_joint"):
        row = frozen["methods"][method]
        map_path = _verify_bound_path(row, "head_phase_map", label=method)
        _verify_bound_path(row, "head_bank_map", label=method)
        map_payload = load_json(map_path)
        validate_map(map_payload, operator=str(row["operator"]))
        if (
            row.get("schedule") != "head_phase"
            or row.get("clean_policy") != "recent"
            or int(row.get("read_frame_equivalents", -1)) != 9
            or map_payload.get("map_id") != row.get("phase_map_id")
        ):
            raise ValueError(f"v192 method contract drifted: {method}")
    return decision, frozen


def _copy(source: Path, target: Path) -> str:
    return write_frozen(target, source.read_bytes())


def _runtime_contract(pf_root: Path) -> dict:
    files = []
    for relative in RUNTIME_FILES:
        path = pf_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing v194 runtime source: {path}")
        files.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    return {
        "runtime_root": str(pf_root.resolve()),
        "files": files,
        "common_model_local_attn_size": 21,
        "checkpoint_state_key": "generator",
        "use_ema": False,
        "strict_checkpoint_load": True,
    }


def _cache_method(
    upstream: dict,
    *,
    role: str,
    map_path: Path,
    map_sha: str,
    bank_path: Path,
    bank_sha: str,
) -> dict:
    row = dict(upstream)
    row.update(
        {
            "runtime": "common_pf_runtime_causal_checkpoint",
            "role": role,
            "head_phase_map": str(map_path.resolve()),
            "head_phase_map_sha256": map_sha,
            "head_bank_map": str(bank_path.resolve()),
            "head_bank_map_sha256": bank_sha,
            "model_local_attn_size": 21,
            "checkpoint_state_key": "generator",
            "use_ema": False,
        }
    )
    return row


def prepare(
    v192_decision: Path,
    v192_input_manifest: Path,
    pf_runtime_root: Path,
    causal_checkpoint: Path,
    output_root: Path,
) -> dict:
    decision, v192 = validate_v192(v192_decision, v192_input_manifest)
    if not causal_checkpoint.is_file():
        raise FileNotFoundError(
            f"missing Causal-Forcing checkpoint: {causal_checkpoint}"
        )
    runtime = _runtime_contract(pf_runtime_root)

    bank_source = Path(v192["methods"]["head_phase_joint"]["head_bank_map"])
    bank_path = output_root / "maps" / "all_profile_banks.csv"
    bank_sha = _copy(bank_source, bank_path)
    copied = {}
    for source_method, target_method in (
        ("all_recent", LOCAL_CONTROL),
        ("head_phase_joint", CANDIDATE),
    ):
        source_path = Path(v192["methods"][source_method]["head_phase_map"])
        target_path = output_root / "maps" / f"{target_method}.json"
        copied[target_method] = (target_path, _copy(source_path, target_path))

    source_scope = next(
        row for row in v192["scopes"] if row["key"] == "seed2026_30s_128"
    )
    source_items = list(source_scope["prompt_items"])
    selected = [source_items[position] for position in PROMPT_POSITIONS]
    prompt_items = [
        {
            "index": index,
            "v192_prompt_index": position,
            "v191_prompt_index": int(item["v191_prompt_index"]),
            "source_index": int(item["source_index"]),
            "text": str(item["text"]),
        }
        for index, (position, item) in enumerate(zip(PROMPT_POSITIONS, selected))
    ]
    prompt_path = output_root / "prompts" / "moviegen_transfer64.txt"
    prompt_sha = write_frozen(
        prompt_path,
        ("\n".join(row["text"] for row in prompt_items) + "\n").encode("utf-8"),
    )

    all_recent_path, all_recent_sha = copied[LOCAL_CONTROL]
    transfer_path, transfer_sha = copied[CANDIDATE]
    methods = {
        NATIVE_CONTROL: {
            "runtime": "common_pf_runtime_causal_checkpoint",
            "role": "causal_checkpoint_native_21_frame_rolling_baseline",
            "config": str(
                (pf_runtime_root / "configs/pyramid-forcing-native.yaml").resolve()
            ),
            "config_sha256": sha256(
                pf_runtime_root / "configs/pyramid-forcing-native.yaml"
            ),
            "model_local_attn_size": 21,
            "checkpoint_state_key": "generator",
            "use_ema": False,
            "cache_enabled": False,
            "effective_history_frames": 21,
        },
        LOCAL_CONTROL: _cache_method(
            v192["methods"]["all_recent"],
            role="causal_checkpoint_equal_budget_all_recent_control",
            map_path=all_recent_path,
            map_sha=all_recent_sha,
            bank_path=bank_path,
            bank_sha=bank_sha,
        ),
        CANDIDATE: _cache_method(
            v192["methods"]["head_phase_joint"],
            role="no_refit_frozen_sf_head_phase_route_on_causal_checkpoint",
            map_path=transfer_path,
            map_sha=transfer_sha,
            bank_path=bank_path,
            bank_sha=bank_sha,
        ),
    }
    cache_config = pf_runtime_root / "configs/pyramid-forcing.yaml"
    for method in (LOCAL_CONTROL, CANDIDATE):
        methods[method]["config"] = str(cache_config.resolve())
        methods[method]["config_sha256"] = sha256(cache_config)

    v191_decision = Path(v192["v191_provenance"]["decision"])
    payload = {
        "version": 1,
        "experiment": "v194_causal_checkpoint_transfer_inputs",
        "confirmatory": True,
        "prerequisite": "passing_v192_seed_length_robustness",
        "transfer_axis": "generator_checkpoint_within_shared_wan_architecture",
        "method_order": list(METHODS),
        "methods": methods,
        "candidate": CANDIDATE,
        "local_control": LOCAL_CONTROL,
        "native_control": NATIVE_CONTROL,
        "prompt_count": PROMPT_COUNT,
        "prompt_positions_in_v192": list(PROMPT_POSITIONS),
        "prompt_items": prompt_items,
        "prompt_file": str(prompt_path.resolve()),
        "prompt_file_sha256": prompt_sha,
        "selection_rule": (
            "odd positions 1,3,...,127 from the frozen v192 128-prompt scope; "
            "fixed without consulting v192 per-prompt outcomes"
        ),
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": DECODED_VIDEO_CONTRACT,
        "seed": SEED,
        "selected_v190_method": v192["selected_v190_method"],
        "selected_operator": v192["selected_operator"],
        "positive_metrics_to_transfer": v192["v191_positive_metrics_to_replicate"],
        "cache_contract": v192["cache_contract"],
        "checkpoint": {
            "path": str(causal_checkpoint.resolve()),
            "sha256": sha256(causal_checkpoint),
            "state_key": "generator",
            "use_ema": False,
            "training_family": "Causal-Forcing chunkwise",
        },
        "runtime_contract": runtime,
        "v192_provenance": {
            "decision": str(v192_decision.resolve()),
            "decision_sha256": sha256(v192_decision),
            "input_manifest": str(v192_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v192_input_manifest),
            "recommendation": decision["recommendation"],
        },
        "same_prompt_sf_reference": {
            "v191_decision": str(v191_decision.resolve()),
            "v191_decision_sha256": sha256(v191_decision),
            "seed": SEED,
            "v191_prompt_positions": [row["v191_prompt_index"] for row in prompt_items],
        },
        "claim_boundary": (
            "v194 tests zero-refit transfer of one frozen Head x Denoising-Phase "
            "route from the Self-Forcing checkpoint to the Causal-Forcing chunkwise "
            "checkpoint under a common audited Wan/PyramidKV code host. It is a "
            "cross-checkpoint and training-objective test, not a cross-architecture "
            "claim. The native 21-frame control is not equal-budget; only all-Recent "
            "and Head x Phase share the 9 FFE read budget."
        ),
    }
    write_json_frozen(output_root / "manifest.json", payload)
    return payload


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    if (
        payload.get("experiment") != "v194_causal_checkpoint_transfer_inputs"
        or payload.get("confirmatory") is not True
        or payload.get("prerequisite") != "passing_v192_seed_length_robustness"
        or payload.get("transfer_axis")
        != "generator_checkpoint_within_shared_wan_architecture"
        or tuple(payload.get("method_order") or ()) != METHODS
        or set(payload.get("methods") or {}) != set(METHODS)
        or payload.get("candidate") != CANDIDATE
        or payload.get("local_control") != LOCAL_CONTROL
        or payload.get("native_control") != NATIVE_CONTROL
        or int(payload.get("prompt_count", -1)) != PROMPT_COUNT
        or int(payload.get("seed", -1)) != SEED
        or int(payload.get("num_output_frames", -1)) != NUM_OUTPUT_FRAMES
        or payload.get("prompt_positions_in_v192") != list(PROMPT_POSITIONS)
        or not payload.get("positive_metrics_to_transfer")
    ):
        raise ValueError("invalid v194 input manifest")

    provenance = payload.get("v192_provenance") or {}
    decision_path = _verify_bound_path(provenance, "decision", label="v192")
    input_path = _verify_bound_path(provenance, "input_manifest", label="v192")
    decision, v192 = validate_v192(decision_path, input_path)
    if (
        provenance.get("recommendation") != decision.get("recommendation")
        or payload.get("selected_v190_method") != v192.get("selected_v190_method")
        or payload.get("selected_operator") != v192.get("selected_operator")
    ):
        raise ValueError("v194 upstream selection drifted")

    prompt_path = Path(payload["prompt_file"])
    prompts = prompt_path.read_text(encoding="utf-8").splitlines()
    items = payload["prompt_items"]
    source_scope = next(
        row for row in v192["scopes"] if row["key"] == "seed2026_30s_128"
    )
    expected = [source_scope["prompt_items"][position] for position in PROMPT_POSITIONS]
    if (
        sha256(prompt_path) != payload["prompt_file_sha256"]
        or len(prompts) != PROMPT_COUNT
        or prompts != [str(row["text"]) for row in expected]
        or [int(row["v192_prompt_index"]) for row in items] != list(PROMPT_POSITIONS)
        or [int(row["source_index"]) for row in items]
        != [int(row["source_index"]) for row in expected]
    ):
        raise ValueError("v194 prompt contract drifted")

    checkpoint = payload.get("checkpoint") or {}
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if (
        not checkpoint_path.is_file()
        or sha256(checkpoint_path) != checkpoint.get("sha256")
        or checkpoint.get("state_key") != "generator"
        or checkpoint.get("use_ema") is not False
    ):
        raise ValueError("v194 checkpoint contract drifted")
    runtime = payload.get("runtime_contract") or {}
    if (
        int(runtime.get("common_model_local_attn_size", -1)) != 21
        or runtime.get("checkpoint_state_key") != "generator"
        or runtime.get("use_ema") is not False
        or runtime.get("strict_checkpoint_load") is not True
        or tuple(row.get("relative_path") for row in runtime.get("files") or ())
        != RUNTIME_FILES
    ):
        raise ValueError("v194 runtime contract is incomplete")
    for row in runtime["files"]:
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"v194 runtime source drifted: {row['relative_path']}")

    for method in METHODS:
        row = payload["methods"][method]
        config = Path(row["config"])
        if (
            not config.is_file()
            or sha256(config) != row["config_sha256"]
            or int(row.get("model_local_attn_size", -1)) != 21
            or row.get("checkpoint_state_key") != "generator"
            or row.get("use_ema") is not False
        ):
            raise ValueError(f"v194 method runtime drifted: {method}")
        if method == NATIVE_CONTROL:
            if row.get("cache_enabled") is not False:
                raise ValueError(
                    "v194 native control unexpectedly enables cache runtime"
                )
            continue
        map_path = _verify_bound_path(row, "head_phase_map", label=method)
        _verify_bound_path(row, "head_bank_map", label=method)
        map_payload = load_json(map_path)
        validate_map(map_payload, operator=str(row["operator"]))
        if (
            row.get("schedule") != "head_phase"
            or row.get("clean_policy") != "recent"
            or int(row.get("read_frame_equivalents", -1)) != 9
            or map_payload.get("map_id") != row.get("phase_map_id")
        ):
            raise ValueError(f"v194 cache method drifted: {method}")

    reference = payload.get("same_prompt_sf_reference") or {}
    _verify_bound_path(reference, "v191_decision", label="sf-reference")
    if int(reference.get("seed", -1)) != SEED or reference.get(
        "v191_prompt_positions"
    ) != [int(row["v191_prompt_index"]) for row in items]:
        raise ValueError("v194 same-prompt SF reference drifted")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v192-decision", type=Path, required=True)
    prepare_parser.add_argument("--v192-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--pf-runtime-root", type=Path, required=True)
    prepare_parser.add_argument("--causal-checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(
            args.v192_decision,
            args.v192_input_manifest,
            args.pf_runtime_root,
            args.causal_checkpoint,
            args.output_root,
        )
        print(
            "[v194-prepare] PASS "
            f"checkpoint_sha256={payload['checkpoint']['sha256']} "
            f"operator={payload['selected_operator']} prompts={payload['prompt_count']}"
        )
    else:
        payload = verify(args.manifest)
        print(
            "[v194-verify] PASS "
            f"methods={len(payload['method_order'])} prompts={payload['prompt_count']}"
        )


if __name__ == "__main__":
    main()
