#!/usr/bin/env python3
"""Strictly audit sharded v98 runtime cache-policy traces.

The audit deliberately treats frozen configs, map artifacts, and runtime
selection traces as one contract.  A route name alone is not evidence that the
intended method ran.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METHODS = (
    "sf_native",
    "pf_native",
    "pf_explicit_parity",
    "pf_aw_hybrid_merge",
    "history_polarity_hybrid_merge",
    "history_polarity_stride_merge",
    "history_polarity_zero_random_hybrid_merge",
    "positive_rate_half_hybrid_merge",
)
FOLLOWUP_METHODS = (
    "followup_history_polarity_hybrid_merge_base",
    "followup_history_polarity_hybrid_merge_v78",
)


@dataclass(frozen=True)
class MethodContract:
    engine: str
    route: str
    map_role: str | None
    transition: int = 0


METHOD_CONTRACTS = {
    "sf_native": MethodContract("sf", "none", None),
    "pf_native": MethodContract("pf", "native", "pf_labels"),
    "pf_explicit_parity": MethodContract(
        "pf", "pf_explicit_parity", "pf_labels"
    ),
    "pf_aw_hybrid_merge": MethodContract(
        "pf", "history_hybrid_merge", "pf_aw_binary_control"
    ),
    "history_polarity_hybrid_merge": MethodContract(
        "pf", "history_hybrid_merge", "primary"
    ),
    "history_polarity_stride_merge": MethodContract(
        "pf", "history_stride_merge", "primary"
    ),
    "history_polarity_zero_random_hybrid_merge": MethodContract(
        "pf", "history_hybrid_merge", "history_polarity_zero_random"
    ),
    "positive_rate_half_hybrid_merge": MethodContract(
        "pf", "history_hybrid_merge", "positive_rate_half"
    ),
}
FOLLOWUP_METHOD_CONTRACTS = {
    "followup_history_polarity_hybrid_merge_base": MethodContract(
        "pf", "history_hybrid_merge", "primary", 0
    ),
    "followup_history_polarity_hybrid_merge_v78": MethodContract(
        "pf", "history_hybrid_merge", "primary", 1
    ),
}


def _strategy(name: str, **parameters: Any) -> dict[str, Any]:
    return {"name": name, **parameters}


PF_NATIVE = {
    -1: {
        "strategies": (
            _strategy(
                "CyclicStrategy",
                period=6,
                bucket_cap=4,
                dynamic_rope=True,
            ),
        ),
        "sink": 1,
        "recent": 4,
        "policy_type": "osc",
    },
    1: {
        "strategies": (
            _strategy(
                "StrideStrategy",
                interval=6,
                capacity=4,
                dynamic_rope=True,
            ),
        ),
        "sink": 3,
        "recent": 4,
        "policy_type": "stride",
    },
    2: {
        "strategies": (
            _strategy(
                "MergeStrategy",
                patch_size=2,
                block_frames=4,
                capacity=4,
                dynamic_rope=True,
            ),
        ),
        "sink": 3,
        "recent": 4,
        "policy_type": "merge",
    },
}
HISTORY_HYBRID = {
    10: {
        "strategies": (
            _strategy(
                "CyclicStrategy",
                period=6,
                bucket_cap=2,
                dynamic_rope=True,
            ),
            _strategy(
                "StrideStrategy",
                interval=6,
                capacity=2,
                dynamic_rope=True,
            ),
        ),
        "sink": 3,
        "recent": 4,
        "policy_type": "stride",
    },
    11: {
        "strategies": (
            _strategy(
                "MergeStrategy",
                patch_size=2,
                block_frames=4,
                capacity=4,
                dynamic_rope=True,
            ),
        ),
        "sink": 3,
        "recent": 4,
        "policy_type": "merge",
    },
}
HISTORY_STRIDE = {
    10: {
        "strategies": (
            _strategy(
                "StrideStrategy",
                interval=6,
                capacity=4,
                dynamic_rope=True,
            ),
        ),
        "sink": 3,
        "recent": 4,
        "policy_type": "stride",
    },
    11: HISTORY_HYBRID[11],
}

REQUIRED_CONFIG_FIELDS = {
    "contract_version",
    "name",
    "phase",
    "mode",
    "node_rank",
    "shard",
    "start_idx",
    "end_idx",
    "gpu",
    "engine",
    "labels",
    "label_sha256",
    "route",
    "transition",
    "score_sha256",
    "map_manifest_sha256",
    "run_commit",
    "prompt_sha256",
    "prompt_count",
    "score_artifact_sha256",
    "method_contract_sha256",
    "frames",
    "expected_video_frames",
    "seed",
    "reseed_per_prompt",
    "few_step_cfg_enabled",
    "policy_trace_layers",
    "policy_trace_stride",
    "policy_trace_max_records",
    "experiment_contract_sha256",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--experiment-contract", type=Path)
    parser.add_argument("--expected-layers", default="0,7,15,23,29")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
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


def load_labels(
    path: Path,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != num_layers:
        raise ValueError(
            f"{path}: expected {num_layers} layers, found {len(rows)}"
        )
    for layer, row in enumerate(rows):
        if len(row) != num_heads:
            raise ValueError(
                f"{path}: layer {layer} expected {num_heads} heads, "
                f"found {len(row)}"
            )
    return rows


def expected_routes(route: str) -> dict[int, dict[str, Any]]:
    if route in {"native", "pf_explicit_parity"}:
        return PF_NATIVE
    if route == "history_hybrid_merge":
        return HISTORY_HYBRID
    if route == "history_stride_merge":
        return HISTORY_STRIDE
    raise ValueError(f"unsupported PF route {route!r}")


def _manifest_primary_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    """Expose the declared score without assuming a particular statistic."""
    descriptor_keys = (
        "score_column",
        "primary_score_column",
        "statistic",
        "score",
        "support_rule",
        "threshold",
        "threshold_provenance",
    )
    descriptor = {
        key: item[key]
        for key in descriptor_keys
        if key in item
    }
    if not descriptor:
        raise ValueError(
            "primary map does not declare a score/statistic descriptor"
        )
    return descriptor


def _load_manifest_contract(
    run_root: Path,
    *,
    num_layers: int,
    num_heads: int,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    path = (
        run_root / "maps" / "history_polarity_manifest.json"
        if manifest_path is None
        else manifest_path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims")
    maps = payload.get("maps")
    if not isinstance(claims, dict) or not isinstance(maps, dict):
        raise ValueError("map manifest must contain claims and maps objects")
    primary = claims.get("primary_classifier")
    if not isinstance(primary, str) or primary not in maps:
        raise ValueError(
            f"invalid primary classifier declaration: {primary!r}"
        )
    if claims.get("pf_labels_used_for_primary_classifier") is not False:
        raise ValueError("primary classifier is not declared PF-independent")
    if (
        int(payload.get("support_label", -999)) != 10
        or int(payload.get("suppress_label", -999)) != 11
        or set(payload.get("reserved_pf_labels", [])) != {-1, 1, 2}
    ):
        raise ValueError("map manifest neutral/reserved labels are invalid")

    required_maps = {
        primary,
        "pf_aw_binary_control",
        "history_polarity_zero_random",
        "positive_rate_half",
    }
    missing_maps = sorted(required_maps - set(maps))
    if missing_maps:
        raise ValueError(f"map manifest missing maps: {missing_maps}")
    random_item = maps["history_polarity_zero_random"]
    if random_item.get("reference") != primary:
        raise ValueError(
            "random control is not count-matched to the declared primary map"
        )

    resolved_maps: dict[str, dict[str, Any]] = {}
    for name in required_maps:
        item = maps[name]
        if not isinstance(item, dict):
            raise ValueError(f"map entry {name!r} is not an object")
        map_path = Path(str(item.get("path", "")))
        if not map_path.is_absolute():
            map_path = path.resolve().parent / map_path
        map_path = map_path.resolve()
        expected_hash = str(item.get("sha256", ""))
        if not map_path.is_file():
            raise FileNotFoundError(f"map does not exist: {map_path}")
        actual_hash = sha256(map_path)
        if not HEX_SHA256.fullmatch(expected_hash):
            raise ValueError(f"map {name!r} has invalid SHA256")
        if actual_hash != expected_hash:
            raise ValueError(f"map {name!r} SHA256 mismatch")
        labels = load_labels(map_path, num_layers, num_heads)
        flat = [value for row in labels for value in row]
        observed_labels = set(flat)
        if not observed_labels or not observed_labels.issubset({10, 11}):
            raise ValueError(
                f"map {name!r} contains invalid labels "
                f"{sorted(observed_labels)}"
            )
        if name == primary and observed_labels != {10, 11}:
            raise ValueError(
                f"primary map {name!r} must contain both labels 10/11"
            )
        declared_counts = {
            int(key): int(value)
            for key, value in dict(item.get("label_counts", {})).items()
        }
        actual_counts = dict(Counter(flat))
        if declared_counts and declared_counts != actual_counts:
            raise ValueError(
                f"map {name!r} label counts do not match manifest"
            )
        resolved_maps[name] = {
            "name": name,
            "path": map_path,
            "sha256": actual_hash,
            "labels": labels,
        }

    pf_path = Path(str(payload.get("pf_labels", ""))).resolve()
    pf_hash = str(payload.get("pf_labels_sha256", ""))
    if not pf_path.is_file() or sha256(pf_path) != pf_hash:
        raise ValueError("canonical PF label path/hash is invalid")
    pf_matrix = load_labels(pf_path, num_layers, num_heads)
    if {value for row in pf_matrix for value in row} != {-1, 1, 2}:
        raise ValueError("canonical PF map must contain labels -1/1/2")
    resolved_maps["pf_labels"] = {
        "name": "pf_labels",
        "path": pf_path,
        "sha256": pf_hash,
        "labels": pf_matrix,
    }

    score_hash = str(payload.get("score_csv_sha256", ""))
    if not HEX_SHA256.fullmatch(score_hash):
        raise ValueError("map manifest score_csv_sha256 is invalid")
    return {
        "path": path.resolve(),
        "sha256": sha256(path),
        "score_sha256": score_hash,
        "primary_classifier": primary,
        "primary_score": _manifest_primary_descriptor(maps[primary]),
        "maps": resolved_maps,
    }


def _method_map(
    contract: MethodContract,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    role = contract.map_role
    if role is None:
        return None
    if role == "primary":
        role = manifest["primary_classifier"]
    return manifest["maps"][role]


def _contract_routes(method: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw_policies = method.get("policies")
    if not isinstance(raw_policies, dict):
        raise ValueError(f"{method.get('name')}: policies must be an object")
    routes: dict[int, dict[str, Any]] = {}
    for raw_label, raw_policy in raw_policies.items():
        label = int(raw_label)
        if not isinstance(raw_policy, dict):
            raise ValueError(
                f"{method.get('name')}: policy {raw_label} is not an object"
            )
        strategies = []
        for raw_strategy in raw_policy.get("strategies", []):
            if not isinstance(raw_strategy, dict):
                raise ValueError(
                    f"{method.get('name')}: invalid strategy contract"
                )
            parameters = raw_strategy.get("params")
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"{method.get('name')}: strategy params are missing"
                )
            strategies.append(
                {"name": str(raw_strategy.get("name", "")), **parameters}
            )
        routes[label] = {
            "strategies": tuple(strategies),
            "sink": int(raw_policy["sink_frames"]),
            "recent": int(raw_policy["recent_frames"]),
            "policy_type": str(raw_policy["policy_type"]),
            "max_union_frames": int(raw_policy["max_union_frames"]),
        }
    return routes


def _route_without_budget(
    routes: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        label: {
            key: value
            for key, value in policy.items()
            if key != "max_union_frames"
        }
        for label, policy in routes.items()
    }


def _load_experiment_contract(
    path: Path,
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if int(payload.get("version", -1)) != 2:
        failures.append("version must be 2")
    if payload.get("experiment") != "v98_history_polarity":
        failures.append("experiment name mismatch")
    phase = payload.get("phase")
    if phase not in {"primary", "followup_v78"}:
        failures.append(f"unsupported experiment phase {phase!r}")
    if payload.get("tracked_worktree_dirty") is not False:
        failures.append("tracked worktree was dirty when the run was frozen")
    if not isinstance(payload.get("few_step_cfg_enabled"), bool):
        failures.append("few_step_cfg_enabled must be explicit")
    if int(payload.get("frames", -1)) != 120:
        failures.append("primary screen must use 120 latent frames")
    if int(payload.get("shards", -1)) != 4:
        failures.append("primary screen must declare four shards")

    fingerprint = payload.get("run_fingerprint")
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("run_fingerprint", None)
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        failures.append("run_fingerprint does not match contract contents")
    if not payload.get("run_commit"):
        failures.append("run_commit is missing")
    if not HEX_SHA256.fullmatch(
        str(payload.get("method_contract_sha256", ""))
    ):
        failures.append("method_contract_sha256 is invalid")
    if phase == "followup_v78" and not HEX_SHA256.fullmatch(
        str(payload.get("primary_manifest_sha256", ""))
    ):
        failures.append(
            "follow-up contract is not bound to the primary manifest"
        )
    expected_primary_evidence = {
        "primary_manifest",
        "primary_experiment_contract",
        "primary_analysis",
        "primary_blind_frozen",
        "primary_blind_verification",
        "primary_blind_completion",
        "primary_blind_scorecard",
        "primary_blind_key",
    }
    primary_evidence = payload.get("primary_gate_evidence")
    if phase == "followup_v78":
        if (
            not isinstance(primary_evidence, dict)
            or set(primary_evidence) != expected_primary_evidence
        ):
            failures.append(
                "follow-up contract has incomplete primary gate evidence"
            )
        else:
            input_contracts = payload.get("inputs")
            if not isinstance(input_contracts, dict):
                input_contracts = {}
            for name, binding in primary_evidence.items():
                if not isinstance(binding, dict):
                    failures.append(
                        f"follow-up primary evidence {name} is malformed"
                    )
                    continue
                evidence_path = Path(str(binding.get("path", "")))
                evidence_hash = str(binding.get("sha256", ""))
                if (
                    not evidence_path.is_file()
                    or not HEX_SHA256.fullmatch(evidence_hash)
                    or sha256(evidence_path) != evidence_hash
                ):
                    failures.append(
                        f"follow-up primary evidence {name} path/hash is invalid"
                    )
                if input_contracts.get(name) != binding:
                    failures.append(
                        f"follow-up primary evidence {name} is not frozen "
                        "as a generation input"
                    )
            manifest_binding = primary_evidence.get("primary_manifest")
            if (
                not isinstance(manifest_binding, dict)
                or manifest_binding.get("sha256")
                != payload.get("primary_manifest_sha256")
            ):
                failures.append(
                    "follow-up primary manifest evidence/hash mismatch"
                )
    elif primary_evidence is not None:
        failures.append("primary phase must not declare follow-up gate evidence")

    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        prompt = {}
        failures.append("prompt contract is missing")
    expected_prompt_count = {"screen32": 32, "main128": 128}.get(
        payload.get("mode")
    )
    if (
        expected_prompt_count is None
        or int(prompt.get("count", -1)) != expected_prompt_count
    ):
        failures.append("mode/prompt-count contract mismatch")
    prompt_path = Path(str(prompt.get("path", "")))
    if (
        not prompt_path.is_file()
        or sha256(prompt_path) != prompt.get("sha256")
    ):
        failures.append("prompt path/hash is invalid")
    if (
        prompt.get("reseed_per_prompt") is not True
        or int(prompt.get("seed", -1)) != int(payload.get("seed", -2))
    ):
        failures.append("prompt seed/reseed contract is invalid")

    video = payload.get("video")
    if not isinstance(video, dict):
        video = {}
        failures.append("video contract is missing")
    if (
        int(video.get("latent_frames", -1)) != int(payload.get("frames", -2))
        or int(video.get("decoded_frames", -1)) <= 0
        or int(video.get("sample_index", -1)) != 0
    ):
        failures.append("video frame/sample contract is invalid")
    sharding = payload.get("sharding")
    if not isinstance(sharding, dict):
        sharding = {}
        failures.append("sharding contract is missing")
    if (
        int(sharding.get("shards", -1)) != int(payload.get("shards", -2))
        or int(sharding.get("shard_size", -1))
        * int(payload.get("shards", -1))
        != int(prompt.get("count", -2))
    ):
        failures.append("sharding contract does not cover all prompts")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        failures.append("runtime contract is missing")
    policy_trace = runtime.get("policy_trace")
    if not isinstance(policy_trace, dict):
        policy_trace = {}
        failures.append("runtime policy_trace contract is missing")
    try:
        trace_layers = [int(value) for value in policy_trace["layers"]]
        if (
            not trace_layers
            or trace_layers != sorted(set(trace_layers))
            or min(trace_layers) < 0
            or int(policy_trace["stride"]) <= 0
            or int(policy_trace["max_records"]) <= 0
        ):
            raise ValueError("invalid trace layer/stride/record values")
        if (
            int(policy_trace["stride"]) != 3
            or int(policy_trace["max_records"]) != 60000
        ):
            raise ValueError(
                "v98 policy trace contract must use stride 3 and "
                "max_records 60000"
            )
    except (KeyError, TypeError, ValueError) as error:
        trace_layers = []
        failures.append(f"invalid runtime policy_trace contract: {error}")

    score = payload.get("score")
    if not isinstance(score, dict):
        score = {}
        failures.append("score contract is missing")
    if score.get("map_manifest_sha256") != manifest["sha256"]:
        failures.append("contract map-manifest hash mismatch")
    if score.get("csv_sha256") != manifest["score_sha256"]:
        failures.append("contract score CSV hash mismatch")
    for path_field, hash_field in (
        ("artifact_path", "artifact_sha256"),
        ("csv_path", "csv_sha256"),
        ("map_manifest_path", "map_manifest_sha256"),
    ):
        artifact_path = Path(str(score.get(path_field, ""))).resolve()
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != score.get(hash_field)
        ):
            failures.append(
                f"score {path_field}/{hash_field} binding is invalid"
            )
    if score.get("artifact_accepted") is not True:
        failures.append("score artifact acceptance is false")
    try:
        artifact_payload = json.loads(
            Path(str(score["artifact_path"])).read_text(encoding="utf-8")
        )
        if (
            int(artifact_payload.get("version", -1))
            != int(score.get("artifact_version", -2))
            or artifact_payload.get("method") != score.get("artifact_method")
            or artifact_payload.get("accepted") is not True
        ):
            failures.append("score artifact identity/acceptance mismatch")
        definition = artifact_payload.get("score_definition", {})
        if (
            not isinstance(definition, dict)
            or definition.get("primary_field") != score.get("primary_field")
        ):
            failures.append("score artifact primary-field mismatch")
        if (
            definition.get("bootstrap_unit")
            != "counterfactual_prompt_pair"
            or score.get("bootstrap_unit")
            != "counterfactual_prompt_pair"
        ):
            failures.append("score artifact bootstrap unit mismatch")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"cannot validate score artifact: {error}")
    primary_field = str(score.get("primary_field", ""))
    descriptor_values = {
        str(value)
        for value in manifest["primary_score"].values()
        if isinstance(value, str)
    }
    if not primary_field or primary_field not in descriptor_values:
        failures.append(
            "contract primary_field is not declared by the primary map"
        )

    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, list):
        raw_methods = []
        failures.append("methods must be a list")
    methods: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_methods):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            failures.append(f"method {index} is malformed")
            continue
        name = item["name"]
        if name in methods:
            failures.append(f"duplicate method {name!r}")
            continue
        if int(item.get("method_index", -1)) != index:
            failures.append(f"{name}: method_index mismatch")
        methods[name] = item
    expected_method_names = (
        METHODS if phase == "primary" else FOLLOWUP_METHODS
    )
    expected_method_contracts = (
        METHOD_CONTRACTS
        if phase == "primary"
        else FOLLOWUP_METHOD_CONTRACTS
    )
    if tuple(methods) != expected_method_names:
        failures.append(
            f"{phase} methods differ from the predeclared phase: "
            f"{list(methods)}"
        )

    routes: dict[str, dict[int, dict[str, Any]]] = {}
    for name, item in methods.items():
        if name not in expected_method_contracts:
            failures.append(f"unexpected method {name!r}")
            continue
        expected = expected_method_contracts[name]
        if (
            item.get("engine") != expected.engine
            or item.get("route") != expected.route
        ):
            failures.append(f"{name}: engine/route contract mismatch")
        expected_map = _method_map(expected, manifest)
        expected_key = None if expected_map is None else expected_map["name"]
        expected_path = (
            None if expected_map is None else str(expected_map["path"])
        )
        expected_hash = (
            None if expected_map is None else expected_map["sha256"]
        )
        if (
            item.get("map_key") != expected_key
            or item.get("map_path") != expected_path
            or item.get("map_sha256") != expected_hash
        ):
            failures.append(f"{name}: map key/path/hash contract mismatch")
        transition = item.get("transition")
        expected_transition_branches = (
            ["cond"] if expected.transition else []
        )
        if (
            not isinstance(transition, dict)
            or bool(transition.get("enabled")) != bool(expected.transition)
            or transition.get("branches") != expected_transition_branches
        ):
            failures.append(f"{name}: transition contract mismatch")
        if expected.transition:
            expected_parameters = {
                "mode": "full",
                "min_reliability": 0.55,
                "min_novelty": 0.01,
                "max_commit_fraction": 0.75,
                "stagger_period": 1,
                "max_age_blocks": 6,
                "branches": "cond",
                "denoise_weight": 2.0,
            }
            if transition.get("parameters") != expected_parameters:
                failures.append(
                    f"{name}: transition parameters differ from "
                    "the predeclared v78 follow-up"
                )
        elif transition.get("parameters") is not None:
            failures.append(
                f"{name}: disabled transition must not declare parameters"
            )
        if item.get("few_step_cfg_enabled") != payload.get(
            "few_step_cfg_enabled"
        ):
            failures.append(f"{name}: few-step CFG contract mismatch")
        expected_branches = [] if expected.engine == "sf" else ["cond"]
        if item.get("policy_trace_branches") != expected_branches:
            failures.append(f"{name}: policy trace branch contract mismatch")
        if expected.engine == "sf":
            if item.get("expected_labels") is not None or item.get(
                "policies"
            ) is not None:
                failures.append("sf_native must not declare PF policies")
            continue
        try:
            method_routes = _contract_routes(item)
            routes[name] = method_routes
            full_expected_routes = expected_routes(expected.route)
            applicable_expected_routes = {
                label: full_expected_routes[label]
                for label in method_routes
                if label in full_expected_routes
            }
            if (
                set(applicable_expected_routes) != set(method_routes)
                or _route_without_budget(method_routes)
                != applicable_expected_routes
            ):
                failures.append(f"{name}: declared policy semantics mismatch")
            declared_labels = {int(value) for value in item["expected_labels"]}
            if declared_labels != set(method_routes):
                failures.append(f"{name}: expected_labels/policies mismatch")
            if expected_map is not None:
                actual_labels = {
                    value
                    for row in expected_map["labels"]
                    for value in row
                }
                if declared_labels != actual_labels:
                    failures.append(f"{name}: labels differ from bound map")
            for label, policy in method_routes.items():
                if policy["max_union_frames"] != 4:
                    failures.append(
                        f"{name}/{label}: middle budget must be four frames"
                    )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{name}: invalid policy contract: {error}")
    return {
        "path": path.resolve(),
        "sha256": sha256(path),
        "payload": payload,
        "methods": methods,
        "routes": routes,
        "prompt_count": int(prompt.get("count", -1)),
        "frames": int(payload.get("frames", -1)),
        "phase": phase,
        "expected_methods": expected_method_names,
        "trace_layers": trace_layers,
        "trace_stride": int(policy_trace.get("stride", -1)),
        "trace_max_records": int(policy_trace.get("max_records", -1)),
        "failures": failures,
        "pass": not failures,
    }


def _config_failure(
    method: str,
    shard: int,
    path: Path,
    failures: list[str],
) -> dict[str, Any]:
    return {
        "method": method,
        "shard": shard,
        "status": "failed",
        "config": str(path.resolve()),
        "events": 0,
        "failures": failures,
    }


def _audit_config(
    *,
    method: str,
    shard: int,
    config_path: Path,
    method_contract: dict[str, Any],
    experiment_contract: dict[str, Any],
    manifest: dict[str, Any],
    num_layers: int,
    num_heads: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    failures: list[str] = []
    try:
        config = load_env(config_path)
    except (OSError, ValueError) as error:
        return _config_failure(
            method, shard, config_path, [f"cannot load config: {error}"]
        ), None

    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        failures.append(f"missing required config fields: {missing}")
    if config.get("name") != method:
        failures.append(
            f"config method {config.get('name')!r} != {method!r}"
        )
    if config.get("contract_version") != "3":
        failures.append("cell config contract_version must be 3")
    if config.get("phase") != experiment_contract["payload"]["phase"]:
        failures.append("phase differs from experiment contract")
    try:
        if int(config.get("shard", -1)) != shard:
            failures.append(
                f"config shard {config.get('shard')!r} != {shard}"
            )
        start = int(config["start_idx"])
        end = int(config["end_idx"])
        if start < 0 or end <= start:
            failures.append(f"invalid prompt interval [{start},{end})")
    except (KeyError, ValueError):
        start, end = -1, -1
        failures.append("start_idx/end_idx are not valid integers")
    expected_engine = str(method_contract["engine"])
    expected_route = str(method_contract["route"])
    if config.get("engine") != expected_engine:
        failures.append(
            f"engine {config.get('engine')!r} != {expected_engine!r}"
        )
    if config.get("route") != expected_route:
        failures.append(
            f"route {config.get('route')!r} != {expected_route!r}"
        )
    try:
        transition = int(config.get("transition", -1))
    except ValueError:
        transition = -1
    expected_transition = int(
        bool(method_contract["transition"]["enabled"])
    )
    if transition != expected_transition:
        failures.append(
            f"transition {config.get('transition')!r} != "
            f"{expected_transition}"
        )
    if config.get("experiment_contract_sha256") != experiment_contract[
        "sha256"
    ]:
        failures.append("experiment-contract SHA256 does not match config")
    if config.get("map_manifest_sha256") != manifest["sha256"]:
        failures.append("map-manifest SHA256 does not match frozen config")
    if config.get("score_sha256") != manifest["score_sha256"]:
        failures.append("score SHA256 does not match map manifest")
    if config.get("score_artifact_sha256") != experiment_contract[
        "payload"
    ]["score"]["artifact_sha256"]:
        failures.append("score-artifact SHA256 differs from contract")
    if config.get("method_contract_sha256") != experiment_contract[
        "payload"
    ].get("method_contract_sha256"):
        failures.append("method-contract SHA256 differs from run contract")
    if config.get("mode") != experiment_contract["payload"]["mode"]:
        failures.append("mode differs from experiment contract")
    if config.get("prompt_sha256") != experiment_contract["payload"][
        "prompt"
    ]["sha256"]:
        failures.append("prompt SHA256 differs from experiment contract")
    if config.get("prompt_count") != str(
        experiment_contract["payload"]["prompt"]["count"]
    ):
        failures.append("prompt_count differs from experiment contract")
    if config.get("run_commit") != experiment_contract["payload"][
        "run_commit"
    ]:
        failures.append("run_commit differs from experiment contract")
    for key in ("map_manifest_sha256", "score_sha256", "prompt_sha256"):
        if not HEX_SHA256.fullmatch(config.get(key, "")):
            failures.append(f"{key} is not a valid SHA256")
    if not config.get("run_commit") or config.get("run_commit") == "unknown":
        failures.append("run_commit is missing or unknown")
    try:
        expected_frames = int(experiment_contract["payload"]["frames"])
        expected_decoded = int(
            experiment_contract["payload"]["video"]["decoded_frames"]
        )
        if int(config.get("frames", -1)) != expected_frames:
            failures.append(
                f"frames must be {expected_frames}, "
                f"got {config.get('frames')!r}"
            )
        if int(config.get("expected_video_frames", -1)) != expected_decoded:
            failures.append(
                f"expected_video_frames must be {expected_decoded}"
            )
        if int(config["seed"]) != int(
            experiment_contract["payload"]["seed"]
        ):
            failures.append("seed differs from experiment contract")
        if int(config.get("reseed_per_prompt", 0)) != 1:
            failures.append("reseed_per_prompt must be 1")
        if bool(int(config.get("few_step_cfg_enabled", -1))) != bool(
            experiment_contract["payload"]["few_step_cfg_enabled"]
        ):
            failures.append("few_step_cfg_enabled differs from contract")
        policy_trace = experiment_contract["payload"]["runtime"][
            "policy_trace"
        ]
        expected_layers_text = ",".join(
            str(value) for value in policy_trace["layers"]
        )
        if config.get("policy_trace_layers") != expected_layers_text:
            failures.append("policy_trace_layers differs from contract")
        if int(config.get("policy_trace_stride", -1)) != int(
            policy_trace["stride"]
        ):
            failures.append("policy_trace_stride differs from contract")
        if int(config.get("policy_trace_max_records", -1)) != int(
            policy_trace["max_records"]
        ):
            failures.append(
                "policy_trace_max_records differs from contract"
            )
    except (KeyError, ValueError):
        failures.append("frames/seed/reseed_per_prompt are malformed")

    map_path_value = method_contract.get("map_path")
    expected_map = (
        None
        if map_path_value is None
        else {
            "path": Path(str(map_path_value)).resolve(),
            "sha256": method_contract.get("map_sha256"),
        }
    )
    labels: list[list[int]] | None = None
    actual_label_hash = ""
    label_path: Path | None = None
    if expected_map is None:
        if config.get("labels") or config.get("label_sha256"):
            failures.append("SF native must not bind a PF head map")
    else:
        try:
            label_path = Path(config["labels"]).resolve()
            if label_path != expected_map["path"]:
                failures.append(
                    f"head-map path {label_path} != "
                    f"{expected_map['path']}"
                )
            actual_label_hash = sha256(label_path)
            if actual_label_hash != expected_map["sha256"]:
                failures.append("head-map SHA256 does not match manifest")
            if actual_label_hash != config.get("label_sha256"):
                failures.append("head-map SHA256 does not match frozen config")
            labels = load_labels(label_path, num_layers, num_heads)
            expected_labels = {
                int(value) for value in method_contract["expected_labels"]
            }
            actual_labels = {
                value for row in labels for value in row
            }
            if actual_labels != expected_labels:
                failures.append(
                    f"head-map labels {sorted(actual_labels)} != "
                    f"{sorted(expected_labels)}"
                )
        except (KeyError, OSError, ValueError) as error:
            failures.append(f"invalid head map: {error}")

    result = {
        "method": method,
        "shard": shard,
        "status": "failed" if failures else "config_nominal",
        "config": str(config_path.resolve()),
        "engine": config.get("engine"),
        "route": config.get("route"),
        "transition": transition,
        "prompt_interval": [start, end],
        "head_map": None if label_path is None else str(label_path),
        "head_map_sha256": actual_label_hash or None,
        "events": 0,
        "failures": failures,
    }
    context = {
        "config": config,
        "labels": labels,
        "start": start,
        "end": end,
        "routes": (
            None
            if expected_engine == "sf"
            else experiment_contract["routes"][method]
        ),
        "trace_branches": set(method_contract["policy_trace_branches"]),
    }
    return result, context


def _read_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], [f"cannot read trace: {error}"]
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as error:
            failures.append(f"line {line_number}: invalid JSON: {error}")
            continue
        if not isinstance(event, dict):
            failures.append(f"line {line_number}: event is not an object")
            continue
        events.append(event)
    if not events:
        failures.append("trace is empty")
    return events, failures


def _frame_budget(spec: dict[str, Any]) -> int:
    if spec["name"] == "CyclicStrategy":
        return int(spec["bucket_cap"])
    return int(spec["capacity"])


def _normalize_strategy(
    item: dict[str, Any],
    spec: dict[str, Any],
    *,
    event_index: int,
    sync_t: int | None,
    sink: int,
    recent: int,
    strict_schema: bool,
    failures: list[str],
) -> tuple[dict[str, Any], list[int], int]:
    name = str(item.get("name", ""))
    if name != spec["name"]:
        failures.append(
            f"event {event_index}: strategy {name!r} != {spec['name']!r}"
        )
    normalized: dict[str, Any] = {"name": name}
    for key, expected in spec.items():
        if key == "name":
            continue
        if key not in item:
            if strict_schema:
                failures.append(
                    f"event {event_index}: {name} missing parameter {key}"
                )
            continue
        actual = item[key]
        if isinstance(expected, bool):
            valid = isinstance(actual, bool) and actual is expected
        else:
            try:
                valid = int(actual) == expected
                actual = int(actual)
            except (TypeError, ValueError):
                valid = False
        if not valid:
            failures.append(
                f"event {event_index}: {name}.{key}={item[key]!r} "
                f"!= {expected!r}"
            )
        normalized[key] = actual

    if "frame_ids" not in item and not strict_schema:
        return normalized, [], int(item.get("token_count", 0) or 0)
    try:
        frame_ids = [int(value) for value in item["frame_ids"]]
        token_count = int(item["token_count"])
    except (KeyError, TypeError, ValueError) as error:
        failures.append(
            f"event {event_index}: malformed {name} selection: {error}"
        )
        return normalized, [], -1
    normalized["frame_ids"] = frame_ids
    normalized["token_count"] = token_count
    if frame_ids != sorted(set(frame_ids)):
        failures.append(
            f"event {event_index}: {name} frame ids are not sorted unique"
        )
    if len(frame_ids) > _frame_budget(spec):
        failures.append(
            f"event {event_index}: {name} budget exceeded "
            f"({len(frame_ids)} > {_frame_budget(spec)})"
        )
    if token_count < 0:
        failures.append(
            f"event {event_index}: {name} has negative token count"
        )
    if (not frame_ids and token_count != 0) or (
        frame_ids and token_count <= 0
    ):
        failures.append(
            f"event {event_index}: {name} frame/token presence is inconsistent"
        )
    if sync_t is not None:
        tail_min_t = sync_t - recent + 1
        sink_max_t = sink - 1 if sink > 0 else -1
        invalid = [
            frame_id
            for frame_id in frame_ids
            if frame_id <= sink_max_t or frame_id >= tail_min_t
        ]
        if invalid:
            failures.append(
                f"event {event_index}: {name} selected sink/recent/future "
                f"frames {invalid} for sync_t={sync_t}"
            )
        if name == "StrideStrategy":
            interval = int(spec["interval"])
            misaligned = [
                frame_id
                for frame_id in frame_ids
                if frame_id % interval != 0
            ]
            if misaligned:
                failures.append(
                    f"event {event_index}: StrideStrategy selected "
                    f"non-interval frames {misaligned}"
                )
        elif name == "CyclicStrategy":
            period = int(spec["period"])
            phase = sync_t % period
            misaligned = [
                frame_id
                for frame_id in frame_ids
                if frame_id % period != phase
            ]
            if misaligned:
                failures.append(
                    f"event {event_index}: CyclicStrategy selected "
                    f"wrong-phase frames {misaligned} for phase={phase}"
                )
        elif name == "MergeStrategy":
            block_frames = int(spec["block_frames"])
            median_offset = (block_frames - 1) // 2
            invalid_blocks = []
            for frame_id in frame_ids:
                block_start = frame_id - median_offset
                block_end = block_start + block_frames - 1
                if (
                    block_start < 0
                    or block_start % block_frames != 0
                    or block_start <= sink_max_t
                    or block_end >= tail_min_t
                    or block_end >= sync_t
                ):
                    invalid_blocks.append(frame_id)
            if invalid_blocks:
                failures.append(
                    f"event {event_index}: MergeStrategy selected invalid "
                    f"block medians {invalid_blocks}"
                )
    return normalized, frame_ids, token_count


def _audit_events(
    *,
    method: str,
    shard: int,
    trace_path: Path,
    labels: list[list[int]],
    routes: dict[int, dict[str, Any]],
    expected_branches: set[str],
    expected_layers: set[int],
    num_heads: int,
    expected_prompt_count: int | None,
    strict_schema: bool,
    expected_sync_times: set[int] | None = None,
) -> dict[str, Any]:
    events, failures = _read_events(trace_path)
    labels_seen: Counter[int] = Counter()
    strategies_seen: Counter[str] = Counter()
    observed_cells: set[tuple[int, int, int, str]] = set()
    observed_pairs: set[tuple[int, int]] = set()
    observed_branches: set[str] = set()
    prompt_ids: set[int] = set()
    normalized_events: dict[tuple[int, int, int, int, str, int], Any] = {}
    max_union = 0
    overlap_count = 0
    cache_contract_failures = 0
    require_exclusive_owner = set(routes).issubset({10, 11})

    for index, event in enumerate(events):
        if event.get("event") != "middle_selection":
            failures.append(
                f"event {index}: unexpected type {event.get('event')!r}"
            )
            continue
        try:
            layer = int(event["layer"])
            head = int(event["head"])
            label = int(event["label"])
            branch = str(event["branch"])
            names = [
                str(item["name"]) for item in list(event["strategies"])
            ]
            strategy_items = list(event["strategies"])
            sink = int(event["sink_frames"])
            recent = int(event["recent_frames"])
            union_ids = [int(value) for value in event["union_frame_ids"]]
            union_count = int(event["union_frame_count"])
            union_tokens = int(event["union_token_count"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"event {index}: malformed: {error}")
            continue
        try:
            if strict_schema and require_exclusive_owner:
                sink_ids = [
                    int(value) for value in event["sink_frame_ids"]
                ]
                sink_count = int(event["sink_frame_count"])
                sink_tokens = int(event["sink_token_count"])
                recent_ids = [
                    int(value) for value in event["recent_frame_ids"]
                ]
                recent_count = int(event["recent_frame_count"])
                recent_tokens = int(event["recent_token_count"])
                middle_sink_overlap = [
                    int(value) for value in event["middle_sink_overlap"]
                ]
                middle_recent_overlap = [
                    int(value) for value in event["middle_recent_overlap"]
                ]
                explicit_owns_dynamic = bool(
                    event["explicit_composition_owns_dynamic"]
                )
                composition_present = bool(event["composition_present"])
                dynamic_owner = str(event["dynamic_policy_owner"])
                cache_violations = [
                    str(value)
                    for value in event["cache_contract_violations"]
                ]
                cache_contract_pass = bool(event["cache_contract_pass"])
            else:
                sink_ids = [
                    int(value)
                    for value in event.get("sink_frame_ids", [])
                ]
                sink_count = int(
                    event.get("sink_frame_count", len(sink_ids))
                )
                sink_tokens = int(event.get("sink_token_count", 0))
                recent_ids = [
                    int(value)
                    for value in event.get("recent_frame_ids", [])
                ]
                recent_count = int(
                    event.get("recent_frame_count", len(recent_ids))
                )
                recent_tokens = int(event.get("recent_token_count", 0))
                middle_sink_overlap = [
                    int(value)
                    for value in event.get("middle_sink_overlap", [])
                ]
                middle_recent_overlap = [
                    int(value)
                    for value in event.get("middle_recent_overlap", [])
                ]
                explicit_owns_dynamic = bool(
                    event.get("explicit_composition_owns_dynamic", True)
                )
                composition_present = bool(
                    event.get("composition_present", True)
                )
                dynamic_owner = str(
                    event.get("dynamic_policy_owner", "composition_recent")
                )
                cache_violations = [
                    str(value)
                    for value in event.get(
                        "cache_contract_violations",
                        [],
                    )
                ]
                cache_contract_pass = bool(
                    event.get("cache_contract_pass", True)
                )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(
                f"event {index}: malformed cache contract: {error}"
            )
            continue

        prompt_id: int | None = None
        seq: int | None = None
        sync_t: int | None = None
        if strict_schema and require_exclusive_owner:
            try:
                prompt_id = int(event["prompt_id"])
                seq = int(event["seq"])
                sync_t = int(event["sync_t"])
                if prompt_id < 0 or seq < 0 or sync_t < 0:
                    raise ValueError("negative prompt_id/seq/sync_t")
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    f"event {index}: missing runtime identity: {error}"
                )
                continue
        else:
            if "prompt_id" in event:
                prompt_id = int(event["prompt_id"])
            if "seq" in event:
                seq = int(event["seq"])
            if "sync_t" in event:
                sync_t = int(event["sync_t"])

        if layer not in expected_layers:
            failures.append(f"event {index}: unexpected traced layer {layer}")
            continue
        if not 0 <= head < num_heads:
            failures.append(f"event {index}: invalid head {head}")
            continue
        if branch not in expected_branches:
            failures.append(f"event {index}: invalid branch {branch!r}")
            continue
        expected_label = labels[layer][head]
        if label != expected_label:
            failures.append(
                f"event {index}: label {label} != map {expected_label}"
            )
            continue
        if label not in routes:
            failures.append(
                f"event {index}: label {label} is invalid for this route"
            )
            continue
        policy = routes[label]
        expected_specs = list(policy["strategies"])
        expected_names = [spec["name"] for spec in expected_specs]
        if names != expected_names:
            failures.append(
                f"event {index}: strategies {names} != {expected_names}"
            )
        if len(strategy_items) != len(expected_specs):
            continue
        if (sink, recent) != (policy["sink"], policy["recent"]):
            failures.append(
                f"event {index}: sink/recent {sink}/{recent} != "
                f"{policy['sink']}/{policy['recent']}"
            )
        policy_type = str(event.get("policy_type", ""))
        if strict_schema and policy_type != policy["policy_type"]:
            failures.append(
                f"event {index}: policy_type {policy_type!r} != "
                f"{policy['policy_type']!r}"
            )

        normalized_strategies = []
        individual_ids: list[int] = []
        individual_tokens = 0
        for item, spec in zip(strategy_items, expected_specs):
            if not isinstance(item, dict):
                failures.append(
                    f"event {index}: strategy item is not an object"
                )
                continue
            normalized, frame_ids, token_count = _normalize_strategy(
                item,
                spec,
                event_index=index,
                sync_t=sync_t,
                sink=sink,
                recent=recent,
                strict_schema=strict_schema,
                failures=failures,
            )
            normalized_strategies.append(normalized)
            individual_ids.extend(frame_ids)
            individual_tokens += max(0, token_count)
            strategies_seen[str(item.get("name", ""))] += 1

        if union_count != len(union_ids):
            failures.append(f"event {index}: union count mismatch")
        if union_ids != sorted(set(union_ids)):
            failures.append(
                f"event {index}: union frame ids are not sorted unique"
            )
        if strict_schema:
            expected_union = sorted(set(individual_ids))
            if union_ids != expected_union:
                failures.append(
                    f"event {index}: union ids {union_ids} do not equal "
                    f"strategy union {expected_union}"
                )
            if union_tokens < 0 or union_tokens > individual_tokens:
                failures.append(
                    f"event {index}: union token count {union_tokens} is "
                    f"outside [0,{individual_tokens}]"
                )
            if len(individual_ids) == len(expected_union):
                if union_tokens != individual_tokens:
                    failures.append(
                        f"event {index}: disjoint strategy token total "
                        "does not match union"
                    )
            overlap_count += len(individual_ids) - len(expected_union)
        elif union_tokens < 0:
            failures.append(f"event {index}: negative union token count")
        if union_count > int(policy.get("max_union_frames", 4)):
            failures.append(
                f"event {index}: middle read budget exceeded ({union_count})"
            )
        if strict_schema:
            if (
                sink_count != len(sink_ids)
                or sink_ids != sorted(set(sink_ids))
            ):
                failures.append(
                    f"event {index}: sink frame ids/count are inconsistent"
                )
            if recent_count != len(recent_ids) or recent_ids != sorted(
                set(recent_ids)
            ):
                failures.append(
                    f"event {index}: recent frame ids/count are inconsistent"
                )
            if sink_tokens < 0 or recent_tokens < 0:
                failures.append(
                    f"event {index}: negative sink/recent token count"
                )
            if sink_count > sink:
                failures.append(
                    f"event {index}: actual sink frame budget exceeded "
                    f"({sink_count}>{sink})"
                )
            if recent_count > recent:
                failures.append(
                    f"event {index}: dynamic cache leaked non-recent frames "
                    f"({recent_count}>{recent})"
                )
            expected_sink_overlap = sorted(set(union_ids) & set(sink_ids))
            expected_recent_overlap = sorted(
                set(union_ids) & set(recent_ids)
            )
            if middle_sink_overlap != expected_sink_overlap:
                failures.append(
                    f"event {index}: middle/sink overlap report is "
                    "inconsistent"
                )
            if middle_recent_overlap != expected_recent_overlap:
                failures.append(
                    f"event {index}: middle/recent overlap report is "
                    "inconsistent"
                )
            if middle_sink_overlap or middle_recent_overlap:
                failures.append(
                    f"event {index}: cache segments overlap "
                    f"sink={middle_sink_overlap} "
                    f"recent={middle_recent_overlap}"
                )
            if not explicit_owns_dynamic:
                failures.append(
                    f"event {index}: explicit composition does not own "
                    "dynamic cache"
                )
            if not composition_present or dynamic_owner != "composition_recent":
                failures.append(
                    f"event {index}: invalid cache owner metadata "
                    f"composition_present={composition_present} "
                    f"dynamic_policy_owner={dynamic_owner!r}"
                )
            expected_cache_pass = not cache_violations
            if cache_contract_pass != expected_cache_pass:
                failures.append(
                    f"event {index}: cache contract pass/violations disagree"
                )
            if not cache_contract_pass:
                cache_contract_failures += 1
                failures.append(
                    f"event {index}: runtime cache contract failed: "
                    f"{cache_violations}"
                )

        observed_pairs.add((layer, head))
        observed_branches.add(branch)
        labels_seen[label] += 1
        max_union = max(max_union, union_count)
        if prompt_id is not None:
            prompt_ids.add(prompt_id)
            observed_cells.add((prompt_id, layer, head, branch))
        if (
            strict_schema
            and prompt_id is not None
            and seq is not None
            and sync_t is not None
        ):
            if seq != head:
                failures.append(
                    f"event {index}: seq={seq} must equal head={head} "
                    "for frozen batch_size=1"
                )
            if (
                expected_sync_times is not None
                and sync_t not in expected_sync_times
            ):
                failures.append(
                    f"event {index}: sync_t={sync_t} is outside the frozen "
                    f"trace grid {sorted(expected_sync_times)}"
                )
            event_key = (prompt_id, layer, head, seq, branch, sync_t)
            normalized_value = {
                "label": label,
                "sink_frames": sink,
                "recent_frames": recent,
                "policy_type": policy_type,
                "strategies": normalized_strategies,
                "union_frame_ids": union_ids,
                "union_frame_count": union_count,
                "union_token_count": union_tokens,
                "sink_frame_ids": sink_ids,
                "sink_frame_count": sink_count,
                "sink_token_count": sink_tokens,
                "recent_frame_ids": recent_ids,
                "recent_frame_count": recent_count,
                "recent_token_count": recent_tokens,
                "middle_sink_overlap": middle_sink_overlap,
                "middle_recent_overlap": middle_recent_overlap,
                "composition_present": composition_present,
                "dynamic_policy_owner": dynamic_owner,
                "explicit_composition_owns_dynamic": explicit_owns_dynamic,
                "cache_contract_violations": cache_violations,
                "cache_contract_pass": cache_contract_pass,
            }
            if event_key in normalized_events:
                failures.append(
                    f"event {index}: duplicate runtime key {event_key}"
                )
            else:
                normalized_events[event_key] = normalized_value

    expected_pairs = {
        (layer, head)
        for layer in expected_layers
        for head in range(num_heads)
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        failures.append(
            f"missing traced layer/head pairs: {missing_pairs[:16]} "
            f"(total={len(missing_pairs)})"
        )
    if observed_branches != expected_branches:
        failures.append(
            f"expected branches {sorted(expected_branches)}, "
            f"found {sorted(observed_branches)}"
        )
    if strict_schema and expected_prompt_count is not None:
        sorted_prompts = sorted(prompt_ids)
        contiguous = (
            bool(sorted_prompts)
            and sorted_prompts
            == list(
                range(
                    sorted_prompts[0],
                    sorted_prompts[0] + len(sorted_prompts),
                )
            )
        )
        if len(prompt_ids) != expected_prompt_count or not contiguous:
            failures.append(
                f"prompt trace coverage is not one contiguous epoch per "
                f"sample: observed={sorted_prompts} "
                f"expected_count={expected_prompt_count}"
            )
        expected_cells = {
            (prompt_id, layer, head, branch)
            for prompt_id in prompt_ids
            for layer in expected_layers
            for head in range(num_heads)
            for branch in expected_branches
        }
        missing_cells = sorted(expected_cells - observed_cells)
        if missing_cells:
            failures.append(
                f"missing prompt/layer/head/branch cells: "
                f"{missing_cells[:16]} (total={len(missing_cells)})"
            )
        if expected_sync_times is not None:
            expected_event_keys = {
                (prompt_id, layer, head, head, branch, sync_t)
                for prompt_id in prompt_ids
                for layer in expected_layers
                for head in range(num_heads)
                for branch in expected_branches
                for sync_t in expected_sync_times
            }
            actual_event_keys = set(normalized_events)
            missing_event_keys = sorted(expected_event_keys - actual_event_keys)
            extra_event_keys = sorted(actual_event_keys - expected_event_keys)
            if missing_event_keys or extra_event_keys:
                failures.append(
                    "trace event grid mismatch: "
                    f"missing={missing_event_keys[:16]} "
                    f"(total={len(missing_event_keys)}) "
                    f"extra={extra_event_keys[:16]} "
                    f"(total={len(extra_event_keys)})"
                )

    is_history = set(routes) == {10, 11}
    if is_history and not set(labels_seen).issubset({10, 11}):
        failures.append(
            f"history route leaked PF labels: {sorted(labels_seen)}"
        )
    return {
        "status": "failed" if failures else "nominal",
        "trace": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path) if trace_path.is_file() else None,
        "events": len(events),
        "observed_pairs": len(observed_pairs),
        "observed_cartesian_cells": len(observed_cells),
        "observed_runtime_keys": len(normalized_events),
        "expected_sync_times": (
            sorted(expected_sync_times)
            if expected_sync_times is not None
            else None
        ),
        "prompt_ids": sorted(prompt_ids),
        "branches": sorted(observed_branches),
        "label_events": dict(sorted(labels_seen.items())),
        "strategy_events": dict(sorted(strategies_seen.items())),
        "max_union_frames": max_union,
        "strategy_overlap_frames": overlap_count,
        "cache_contract_failures": cache_contract_failures,
        "failures": failures,
        "_normalized_events": normalized_events,
    }


def audit_trace(
    *,
    method: str,
    shard: int,
    config_path: Path,
    trace_path: Path,
    expected_layers: set[int],
    num_layers: int,
    num_heads: int,
) -> dict[str, Any]:
    """Compatibility entry point used by focused unit tests.

    Full experiment auditing goes through :func:`main`, which additionally
    enforces manifest, engine, method, interval, and provenance contracts.
    """
    failures: list[str] = []
    try:
        config = load_env(config_path)
        if config.get("name") != method:
            failures.append(
                f"config method {config.get('name')!r} != {method!r}"
            )
        if int(config.get("shard", -1)) != shard:
            failures.append(
                f"config shard {config.get('shard')!r} != {shard}"
            )
        label_path = Path(config["labels"])
        labels = load_labels(label_path, num_layers, num_heads)
        actual_hash = sha256(label_path)
        if actual_hash != config.get("label_sha256"):
            failures.append("head-map SHA256 does not match frozen config")
        routes = expected_routes(config["route"])
    except (KeyError, OSError, ValueError) as error:
        return {
            "method": method,
            "shard": shard,
            "status": "failed",
            "events": 0,
            "failures": [f"invalid config: {error}"],
        }
    event_result = _audit_events(
        method=method,
        shard=shard,
        trace_path=trace_path,
        labels=labels,
        routes=routes,
        expected_branches={"cond", "uncond"},
        expected_layers=expected_layers,
        num_heads=num_heads,
        expected_prompt_count=None,
        strict_schema=False,
    )
    failures.extend(event_result.pop("failures"))
    event_result.pop("_normalized_events", None)
    result = {
        "method": method,
        "shard": shard,
        "config": str(config_path.resolve()),
        "route": config["route"],
        "head_map": str(label_path.resolve()),
        "head_map_sha256": actual_hash,
        **event_result,
        "failures": failures,
    }
    result["status"] = "failed" if failures else "nominal"
    return result


def _global_config_contract(
    contexts: dict[tuple[str, int], dict[str, Any]],
    *,
    methods: tuple[str, ...],
    shards: int,
    experiment_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_keys = {
        (method, shard)
        for method in methods
        for shard in range(shards)
    }
    if set(contexts) != expected_keys:
        failures.append(
            f"config matrix mismatch: missing={sorted(expected_keys-set(contexts))} "
            f"extra={sorted(set(contexts)-expected_keys)}"
        )
    if not contexts:
        return {"pass": False, "failures": failures or ["no valid configs"]}

    intervals: list[tuple[int, int]] = []
    for shard in range(shards):
        shard_intervals = {
            (context["start"], context["end"])
            for (method, current_shard), context in contexts.items()
            if current_shard == shard
        }
        if len(shard_intervals) != 1:
            failures.append(
                f"shard {shard} intervals disagree: "
                f"{sorted(shard_intervals)}"
            )
            continue
        intervals.append(next(iter(shard_intervals)))
    if len(intervals) == shards:
        cursor = 0
        for shard, (start, end) in enumerate(intervals):
            if start != cursor or end <= start:
                failures.append(
                    f"shard {shard} interval [{start},{end}) is not "
                    f"contiguous after {cursor}"
                )
            cursor = end
    else:
        cursor = -1

    invariant_fields = (
        "mode",
        "score_sha256",
        "score_artifact_sha256",
        "map_manifest_sha256",
        "method_contract_sha256",
        "run_commit",
        "prompt_sha256",
        "prompt_count",
        "frames",
        "expected_video_frames",
        "seed",
        "reseed_per_prompt",
        "few_step_cfg_enabled",
        "policy_trace_layers",
        "policy_trace_stride",
        "policy_trace_max_records",
        "experiment_contract_sha256",
    )
    invariant_values = {}
    for field in invariant_fields:
        values = {
            context["config"].get(field)
            for context in contexts.values()
        }
        invariant_values[field] = sorted(
            "" if value is None else value for value in values
        )
        if len(values) != 1:
            failures.append(
                f"frozen config field {field} disagrees: "
                f"{invariant_values[field]}"
            )
    mode = next(iter(contexts.values()))["config"].get("mode")
    expected_total = (
        int(experiment_contract["prompt_count"])
        if experiment_contract is not None
        else {"screen32": 32, "main128": 128}.get(mode)
    )
    if expected_total is None:
        failures.append(f"unsupported mode {mode!r}")
    elif cursor != expected_total:
        failures.append(
            f"shard union covers [0,{cursor}) but {mode} requires "
            f"[0,{expected_total})"
        )
    return {
        "pass": not failures,
        "mode": mode,
        "prompt_count": cursor,
        "shard_intervals": [list(value) for value in intervals],
        "invariants": invariant_values,
        "failures": failures,
    }


def _parity_contract(
    event_records: dict[tuple[str, int], dict[Any, Any]],
    *,
    shards: int,
) -> dict[str, Any]:
    failures: list[str] = []
    compared = 0
    mismatch_count = 0
    shard_rows = []
    for shard in range(shards):
        native = event_records.get(("pf_native", shard), {})
        explicit = event_records.get(("pf_explicit_parity", shard), {})
        native_keys = set(native)
        explicit_keys = set(explicit)
        missing = sorted(native_keys - explicit_keys)
        extra = sorted(explicit_keys - native_keys)
        mismatches = [
            key
            for key in sorted(native_keys & explicit_keys)
            if native[key] != explicit[key]
        ]
        compared += len(native_keys & explicit_keys)
        mismatch_count += len(mismatches)
        if missing or extra or mismatches or not native_keys:
            failures.append(
                f"shard {shard}: missing={len(missing)} "
                f"extra={len(extra)} mismatched={len(mismatches)} "
                f"compared={len(native_keys & explicit_keys)}"
            )
        shard_rows.append(
            {
                "shard": shard,
                "native_events": len(native_keys),
                "explicit_events": len(explicit_keys),
                "compared_events": len(native_keys & explicit_keys),
                "missing_keys": len(missing),
                "extra_keys": len(extra),
                "mismatched_events": len(mismatches),
                "examples": {
                    "missing": [list(value) for value in missing[:3]],
                    "extra": [list(value) for value in extra[:3]],
                    "mismatched": [list(value) for value in mismatches[:3]],
                },
            }
        )
    return {
        "pass": not failures and compared > 0,
        "compared_events": compared,
        "mismatched_events": mismatch_count,
        "shards": shard_rows,
        "failures": failures,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    parity = payload["pf_parity_observed_contract"]
    config_contract = payload["config_matrix_contract"]
    lines = [
        "# v98 Policy Trace Audit",
        "",
        f"- strict pass: `{payload['strict_pass']}`",
        f"- config matrix: `{config_contract['pass']}`",
        f"- experiment contract: "
        f"`{payload['experiment_contract']['pass']}`",
        f"- audited configs: `{len(payload['shards'])}`",
        f"- total PF events: `{payload['event_count']}`",
        f"- observed PF parity: `{parity['pass']}` "
        f"({parity['compared_events']} events compared)",
        f"- primary classifier: `{payload['map_contract']['primary_classifier']}`",
        f"- primary score declaration: "
        f"`{json.dumps(payload['map_contract']['primary_score'], sort_keys=True)}`",
        "",
        "| Method | Shard | Engine | Route | Interval | Status | Events | Failures |",
        "|---|---:|---|---|---|---|---:|---:|",
    ]
    for item in payload["shards"]:
        lines.append(
            f"| {item['method']} | {item['shard']} | "
            f"{item.get('engine', 'unknown')} | "
            f"{item.get('route', 'unknown')} | "
            f"{item.get('prompt_interval', 'unknown')} | "
            f"{item['status']} | {item.get('events', 0)} | "
            f"{len(item.get('failures', []))} |"
        )
    failures = [
        (item["method"], item["shard"], failure)
        for item in payload["shards"]
        for failure in item.get("failures", [])
    ]
    failures.extend(
        ("config_matrix", "-", failure)
        for failure in config_contract.get("failures", [])
    )
    failures.extend(
        ("pf_parity", "-", failure)
        for failure in parity.get("failures", [])
    )
    failures.extend(
        ("experiment_contract", "-", failure)
        for failure in payload["experiment_contract"].get("failures", [])
    )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{method}.shard{shard}`: {failure}"
            for method, shard, failure in failures
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    expected_layers = {
        int(value.strip())
        for value in args.expected_layers.split(",")
        if value.strip()
    }
    if (
        not expected_layers
        or min(expected_layers) < 0
        or max(expected_layers) >= args.num_layers
    ):
        raise ValueError("expected layers are empty or out of range")
    if args.num_layers <= 0 or args.num_heads <= 0 or args.shards <= 0:
        raise ValueError("layer/head/shard counts must be positive")

    experiment_contract_path = (
        args.experiment_contract
        if args.experiment_contract is not None
        else args.run_root / "experiment_contract.json"
    )
    contracted_manifest_path: Path | None = None
    try:
        raw_experiment_contract = json.loads(
            experiment_contract_path.read_text(encoding="utf-8")
        )
        raw_manifest_path = raw_experiment_contract.get("score", {}).get(
            "map_manifest_path"
        )
        if raw_manifest_path:
            contracted_manifest_path = Path(str(raw_manifest_path))
    except (OSError, AttributeError, json.JSONDecodeError):
        pass

    try:
        manifest = _load_manifest_contract(
            args.run_root,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            manifest_path=contracted_manifest_path,
        )
        manifest_failure = None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        manifest = {
            "path": (args.run_root / "maps/history_polarity_manifest.json"),
            "sha256": "",
            "score_sha256": "",
            "primary_classifier": None,
            "primary_score": {},
            "maps": {},
        }
        manifest_failure = str(error)

    try:
        if manifest_failure is not None:
            raise ValueError(
                "cannot validate experiment contract without a valid map "
                "manifest"
            )
        experiment_contract = _load_experiment_contract(
            experiment_contract_path,
            manifest=manifest,
        )
        if args.shards != int(experiment_contract["payload"]["shards"]):
            experiment_contract["failures"].append(
                f"CLI shards={args.shards} differs from experiment contract"
            )
            experiment_contract["pass"] = False
        if set(experiment_contract["trace_layers"]) != expected_layers:
            experiment_contract["failures"].append(
                "CLI expected-layers differs from experiment contract"
            )
            experiment_contract["pass"] = False
        experiment_contract_failure = None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        experiment_contract = {
            "path": experiment_contract_path.resolve(),
            "sha256": "",
            "payload": {"shards": args.shards},
            "methods": {},
            "routes": {},
            "prompt_count": -1,
            "phase": None,
            "expected_methods": METHODS,
            "failures": [str(error)],
            "pass": False,
        }
        experiment_contract_failure = str(error)

    audit_methods = tuple(experiment_contract.get("expected_methods", METHODS))
    phase = experiment_contract.get("phase")
    config_dir = args.run_root / "configs"
    trace_dir = args.run_root / "traces"
    results: list[dict[str, Any]] = []
    contexts: dict[tuple[str, int], dict[str, Any]] = {}
    event_records: dict[tuple[str, int], dict[Any, Any]] = {}
    for method in audit_methods:
        method_contract = experiment_contract["methods"].get(method)
        for shard in range(args.shards):
            config_path = config_dir / f"{method}.shard{shard}.env"
            if manifest_failure is not None or method_contract is None:
                reason = (
                    f"invalid map manifest: {manifest_failure}"
                    if manifest_failure is not None
                    else "method missing from experiment contract"
                )
                result = _config_failure(
                    method,
                    shard,
                    config_path,
                    [reason],
                )
                results.append(result)
                continue
            result, context = _audit_config(
                method=method,
                shard=shard,
                config_path=config_path,
                method_contract=method_contract,
                experiment_contract=experiment_contract,
                manifest=manifest,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
            )
            if context is not None and result["status"] == "config_nominal":
                contexts[(method, shard)] = context
            if method_contract["engine"] == "sf":
                trace_path = trace_dir / f"{method}.shard{shard}.policy.jsonl"
                if trace_path.exists() and trace_path.stat().st_size:
                    result["failures"].append(
                        "SF native unexpectedly produced a PF policy trace"
                    )
                result["status"] = (
                    "failed" if result["failures"] else "nominal"
                )
                results.append(result)
                continue
            if context is None or context["labels"] is None:
                result["status"] = "failed"
                results.append(result)
                continue
            event_result = _audit_events(
                method=method,
                shard=shard,
                trace_path=(
                    trace_dir / f"{method}.shard{shard}.policy.jsonl"
                ),
                labels=context["labels"],
                routes=context["routes"],
                expected_branches=context["trace_branches"],
                expected_layers=expected_layers,
                num_heads=args.num_heads,
                expected_prompt_count=context["end"] - context["start"],
                strict_schema=True,
                expected_sync_times={
                    sync_t
                    for sync_t in range(
                        0,
                        int(experiment_contract["frames"]),
                        3,
                    )
                    if sync_t
                    % int(experiment_contract["trace_stride"])
                    == 0
                },
            )
            event_records[(method, shard)] = event_result.pop(
                "_normalized_events"
            )
            result["failures"].extend(event_result.pop("failures"))
            result.update(event_result)
            result["status"] = (
                "failed" if result["failures"] else "nominal"
            )
            results.append(result)

    config_contract = _global_config_contract(
        contexts,
        methods=audit_methods,
        shards=args.shards,
        experiment_contract=experiment_contract,
    )
    if phase == "primary":
        parity_contract = _parity_contract(
            event_records,
            shards=args.shards,
        )
    else:
        parity_contract = {
            "pass": True,
            "status": "not_applicable_followup_v78",
            "compared_events": 0,
            "mismatched_events": 0,
            "shards": [],
            "failures": [],
        }
    payload = {
        "version": 2,
        "method": "v98_sharded_policy_trace_audit",
        "phase": phase,
        "expected_methods": list(audit_methods),
        "experiment_contract": {
            "path": str(experiment_contract["path"]),
            "sha256": experiment_contract["sha256"] or None,
            "pass": experiment_contract["pass"],
            "failures": experiment_contract["failures"],
        },
        "experiment_contract_sha256": (
            experiment_contract["sha256"] or None
        ),
        "method_contracts": experiment_contract["methods"],
        "expected_layers": sorted(expected_layers),
        "expected_shards": args.shards,
        "map_contract": {
            "manifest": str(manifest["path"]),
            "manifest_sha256": manifest["sha256"] or None,
            "primary_classifier": manifest["primary_classifier"],
            "primary_score": manifest["primary_score"],
            "failure": manifest_failure,
        },
        "config_matrix_contract": config_contract,
        "pf_parity_route_contract": (
            (
                METHOD_CONTRACTS["pf_native"].map_role
                == METHOD_CONTRACTS["pf_explicit_parity"].map_role
                and METHOD_CONTRACTS["pf_native"].engine
                == METHOD_CONTRACTS["pf_explicit_parity"].engine
            )
            if phase == "primary"
            else None
        ),
        "pf_parity_observed_contract": parity_contract,
        "event_count": sum(int(item.get("events", 0)) for item in results),
        "strict_pass": (
            manifest_failure is None
            and experiment_contract_failure is None
            and experiment_contract["pass"]
            and len(results) == len(audit_methods) * args.shards
            and config_contract["pass"]
            and parity_contract["pass"]
            and all(item["status"] == "nominal" for item in results)
        ),
        "shards": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[V98PolicyTraceAudit] "
        f"configs={len(results)} events={payload['event_count']} "
        f"parity_events={parity_contract['compared_events']} "
        f"strict_pass={payload['strict_pass']}",
        flush=True,
    )
    if args.strict and not payload["strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
