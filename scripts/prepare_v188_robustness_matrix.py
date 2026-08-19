#!/usr/bin/env python3
"""Freeze post-v187 seed, duration, and denoising-phase robustness scopes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_v187_unseen128_confirmation import (
    METHODS as V187_METHODS,
    SELECTED_OPERATORS,
)


BASE_METHODS = V187_METHODS
MECHANISM_METHODS = BASE_METHODS + (
    "opposite_phase_deterministic",
    "all_noisy_deterministic",
)
CACHE_METHODS = tuple(method for method in MECHANISM_METHODS if method != "sf_native")
SCHEDULE_CALLS = {
    "recent": (),
    "coverage": (0, 1, 2, 3),
    "early1": (0,),
    "early2": (0, 1),
    "late1": (3,),
    "late2": (2, 3),
}
OPPOSITE_SCHEDULE = {
    "early1": "late1",
    "late1": "early1",
    "early2": "late2",
    "late2": "early2",
}
SCOPE_SPECS = (
    {
        "key": "replica64_seed20000",
        "prompt_count": 64,
        "num_output_frames": 120,
        "seed": 20000,
        "methods": BASE_METHODS,
        "generated_methods": BASE_METHODS,
        "reused_methods": (),
        "priority": 1,
        "purpose": "independent_seed_replication",
    },
    {
        "key": "long60_seed10000_32",
        "prompt_count": 32,
        "num_output_frames": 240,
        "seed": 10000,
        "methods": BASE_METHODS,
        "generated_methods": BASE_METHODS,
        "reused_methods": (),
        "priority": 2,
        "purpose": "late_horizon_persistence",
    },
    {
        "key": "mechanism32_seed10000",
        "prompt_count": 32,
        "num_output_frames": 120,
        "seed": 10000,
        "methods": MECHANISM_METHODS,
        "generated_methods": (
            "phase_deterministic",
            "opposite_phase_deterministic",
            "all_noisy_deterministic",
        ),
        "reused_methods": ("sf_native", "all_recent", "phase_reservoir"),
        "priority": 3,
        "purpose": "equal_budget_denoising_phase_counterfactual",
    },
)
SCOPE_BY_KEY = {str(row["key"]): row for row in SCOPE_SPECS}
PARTITION_SALT = "v188-unseen128-disjoint-partition-v1"


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
        raise RuntimeError(f"frozen v188 artifact differs: {path}")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _rank_key(item: dict) -> tuple[str, int]:
    source_index = int(item["source_index"])
    normalized = " ".join(str(item["text"]).split()).casefold()
    digest = hashlib.sha256(
        f"{PARTITION_SALT}\0{source_index}\0{normalized}".encode("utf-8")
    ).hexdigest()
    return digest, source_index


def _validate_v187(
    decision_path: Path,
    input_manifest_path: Path,
    published_path: Path,
) -> tuple[dict, dict, dict, dict[str, dict]]:
    decision = load_json(decision_path)
    frozen = load_json(input_manifest_path)
    published = load_json(published_path)
    contract_path = Path(published.get("experiment_contract", ""))
    if (
        decision.get("experiment")
        != "v187_unseen128_phase_operator_vbench"
        or decision.get("confirmatory") is not True
        or int(decision.get("prompt_count", -1)) != 128
        or int(decision.get("seed", -1)) != 10000
        or decision.get("prompt_source_index_range") != [128, 255]
        or tuple(decision.get("methods") or ()) != BASE_METHODS
        or decision.get("recommendation")
        != "freeze_method_for_replication_and_cross_model"
        or decision.get("benchmark_advantage_confirmed") is not True
        or decision.get("operator_attribution_confirmed") is not True
    ):
        raise ValueError("v188 requires the successful frozen v187 decision")
    if (
        frozen.get("experiment")
        != "v187_unseen128_phase_operator_confirmation"
        or frozen.get("scope") != "confirmatory_unseen128"
        or frozen.get("confirmatory") is not True
        or int(frozen.get("prompt_count", -1)) != 128
        or tuple(frozen.get("method_order") or ()) != BASE_METHODS
        or frozen.get("selected_schedule") not in OPPOSITE_SCHEDULE
        or frozen.get("selected_v186_method") not in SELECTED_OPERATORS
        or decision.get("selected_schedule") != frozen.get("selected_schedule")
        or decision.get("selected_operator") != frozen.get("selected_operator")
    ):
        raise ValueError("v187 decision and frozen method contract disagree")
    if not contract_path.is_file():
        raise ValueError("v187 generation contract is missing")
    contract = load_json(contract_path)
    rows = {str(row.get("key")): row for row in published.get("methods") or ()}
    if (
        published.get("ok") is not True
        or published.get("experiment")
        != "v187_unseen128_phase_operator_generation"
        or published.get("scope") != "confirm128"
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("scope") != "confirm128"
        or contract.get("confirmatory") is not True
        or int(contract.get("prompt_count", -1)) != 128
        or tuple(contract.get("methods") or ()) != BASE_METHODS
        or contract.get("input_manifest_sha256") != sha256(input_manifest_path)
        or tuple(rows) != BASE_METHODS
        or not all(row.get("ok") is True for row in rows.values())
    ):
        raise ValueError("v187 published generation evidence is incomplete or mixed")
    expected_names = {f"{index:06d}.mp4" for index in range(128)}
    for method in BASE_METHODS:
        row = rows[method]
        audit = Path(row.get("audit", ""))
        video_dir = Path(row.get("video_dir", ""))
        if (
            not audit.is_file()
            or sha256(audit) != row.get("audit_sha256")
            or not video_dir.is_dir()
            or {path.name for path in video_dir.glob("*.mp4")} != expected_names
        ):
            raise ValueError(f"v187 source evidence drifted: {method}")
    return decision, frozen, published, rows


def _method_templates(v187: dict) -> dict[str, dict]:
    selected = str(v187["selected_v186_method"])
    selected_contract = SELECTED_OPERATORS[selected]
    schedule = str(v187["selected_schedule"])
    opposite = OPPOSITE_SCHEDULE[schedule]
    map_row = v187["methods"]["phase_deterministic"]
    map_path = Path(map_row["head_map"])
    if not map_path.is_file() or sha256(map_path) != map_row["head_map_sha256"]:
        raise ValueError("v187 all-profile head map drifted")

    common = {
        "runtime": "phase_cache_runtime",
        "head_map": str(map_path.resolve()),
        "head_map_sha256": sha256(map_path),
        "head_route_counts": {"10": 360, "11": 0},
        "read_frame_equivalents": 9,
        "clean_policy": "recent",
    }

    def cache_row(
        *,
        role: str,
        schedule_name: str,
        operator: str,
        history_policy: str,
        source_kind: str,
        read_capacity: int,
        storage_capacity: int,
    ) -> dict:
        return {
            **common,
            "role": role,
            "schedule": schedule_name,
            "coverage_noisy_calls": list(SCHEDULE_CALLS[schedule_name]),
            "operator": operator,
            "history_policy": history_policy,
            "expected_middle_source_kind": source_kind,
            "middle_read_capacity": read_capacity,
            "middle_storage_capacity": storage_capacity,
        }

    deterministic = {
        "operator": selected_contract["operator"],
        "history_policy": selected_contract["history_policy"],
        "source_kind": selected_contract["source_kind"],
        "storage_capacity": selected_contract["storage_ffe"],
    }
    return {
        "sf_native": {
            "runtime": "self_forcing_native",
            "role": "native_self_forcing_baseline",
        },
        "all_recent": cache_row(
            role="equal_budget_local_control",
            schedule_name="recent",
            operator="reservoir",
            history_policy="reservoir4_multiscalemotion1",
            source_kind="temporal_reservoir",
            read_capacity=0,
            storage_capacity=4,
        ),
        "phase_reservoir": cache_row(
            role="random_coverage_reference",
            schedule_name=schedule,
            operator="reservoir",
            history_policy="reservoir4_multiscalemotion1",
            source_kind="temporal_reservoir",
            read_capacity=4,
            storage_capacity=4,
        ),
        "phase_deterministic": cache_row(
            role="frozen_v187_candidate",
            schedule_name=schedule,
            read_capacity=4,
            **deterministic,
        ),
        "opposite_phase_deterministic": cache_row(
            role="equal_dose_opposite_phase_counterfactual",
            schedule_name=opposite,
            read_capacity=4,
            **deterministic,
        ),
        "all_noisy_deterministic": cache_row(
            role="all_noisy_calls_counterfactual",
            schedule_name="coverage",
            read_capacity=4,
            **deterministic,
        ),
    }


def prepare(
    v187_decision: Path,
    v187_input_manifest: Path,
    v187_published: Path,
    output_root: Path,
) -> dict:
    decision, v187, published, source_rows = _validate_v187(
        v187_decision,
        v187_input_manifest,
        v187_published,
    )
    prompt_items = list(v187.get("prompt_items") or ())
    if (
        len(prompt_items) != 128
        or [int(row.get("index", -1)) for row in prompt_items] != list(range(128))
        or len({int(row.get("source_index", -1)) for row in prompt_items}) != 128
    ):
        raise ValueError("v187 prompt items are incomplete")
    ranked = sorted(prompt_items, key=_rank_key)
    templates = _method_templates(v187)

    scopes = []
    cursor = 0
    for spec in SCOPE_SPECS:
        count = int(spec["prompt_count"])
        selected_items = ranked[cursor : cursor + count]
        cursor += count
        local_items = [
            {
                "index": local_index,
                "v187_index": int(item["index"]),
                "source_index": int(item["source_index"]),
                "partition_rank": cursor - count + local_index,
                "text": str(item["text"]),
            }
            for local_index, item in enumerate(selected_items)
        ]
        prompt_path = output_root / "prompts" / f"{spec['key']}.txt"
        prompt_sha = write_frozen(
            prompt_path,
            ("\n".join(row["text"] for row in local_items) + "\n").encode(
                "utf-8"
            ),
        )
        decoded = {
            "frames": 957 if int(spec["num_output_frames"]) == 240 else 477,
            "fps": 16.0,
            "duration_seconds": 59.8125
            if int(spec["num_output_frames"]) == 240
            else 29.8125,
            "width": 832,
            "height": 480,
        }
        scopes.append(
            {
                **spec,
                "methods": list(spec["methods"]),
                "generated_methods": list(spec["generated_methods"]),
                "reused_methods": list(spec["reused_methods"]),
                "prompt_file": str(prompt_path.resolve()),
                "prompt_file_sha256": prompt_sha,
                "prompt_items": local_items,
                "decoded_video_contract": decoded,
            }
        )
    if cursor != 128:
        raise AssertionError("v188 scope partition must consume all 128 prompts")
    partitions = [
        {int(row["v187_index"]) for row in scope["prompt_items"]}
        for scope in scopes
    ]
    if set.union(*partitions) != set(range(128)) or any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise AssertionError("v188 prompt partitions must be disjoint and complete")

    contract_path = Path(published["experiment_contract"])
    payload = {
        "version": 1,
        "experiment": "v188_post_confirmation_robustness_matrix",
        "confirmatory_extension": True,
        "partition_rule": {
            "algorithm": "sha256_rank_then_contiguous_partition",
            "salt": PARTITION_SALT,
            "scope_order": [row["key"] for row in SCOPE_SPECS],
            "outcome_blind": True,
            "disjoint": True,
            "complete_v187_unseen128": True,
        },
        "selected_schedule": v187["selected_schedule"],
        "opposite_schedule": OPPOSITE_SCHEDULE[v187["selected_schedule"]],
        "selected_operator": v187["selected_operator"],
        "selected_v186_method": v187["selected_v186_method"],
        "method_templates": templates,
        "scopes": scopes,
        "cache_contract": {
            "recent_read": "sink1 + recent8",
            "coverage_read": "sink1 + exactly one middle4 operator + recent4",
            "clean_read": "Recent for every cache method",
            "read_budget_frame_equivalents": 9,
            "shared_clean_updates": True,
            "dynamic_rope": True,
            "storage_disclosure": {
                method: int(row.get("middle_storage_capacity", 0))
                for method, row in templates.items()
                if method in CACHE_METHODS
            },
        },
        "v187_provenance": {
            "decision": str(v187_decision.resolve()),
            "decision_sha256": sha256(v187_decision),
            "input_manifest": str(v187_input_manifest.resolve()),
            "input_manifest_sha256": sha256(v187_input_manifest),
            "published_manifest": str(v187_published.resolve()),
            "published_manifest_sha256": sha256(v187_published),
            "generation_contract": str(contract_path.resolve()),
            "generation_contract_sha256": sha256(contract_path),
            "source_methods": {
                method: {
                    "video_dir": source_rows[method]["video_dir"],
                    "audit": source_rows[method]["audit"],
                    "audit_sha256": source_rows[method]["audit_sha256"],
                }
                for method in BASE_METHODS
            },
            "automated_decision": decision["recommendation"],
        },
        "analysis_contract": {
            "primary_candidate": "phase_deterministic",
            "local_control": "all_recent",
            "random_operator_control": "phase_reservoir",
            "native_reference": "sf_native",
            "mechanism_counterfactuals": [
                "opposite_phase_deterministic",
                "all_noisy_deterministic",
            ],
            "long60_windows": {"full": [0, 30], "early_half": [0, 15], "late_half": [15, 30]},
            "manual_review_cap": 6,
        },
        "claim_boundary": (
            "v188 tests seed replication, 60-second persistence, and denoising-"
            "phase mechanism on the same Self-Forcing model. It does not establish "
            "cross-model transfer or scene-switch behavior."
        ),
    }
    write_frozen(
        output_root / "manifest.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def scope_config(manifest: dict, key: str) -> dict:
    rows = {str(row.get("key")): row for row in manifest.get("scopes") or ()}
    if key not in rows:
        raise ValueError(f"unsupported v188 scope: {key}")
    return rows[key]


def verify(manifest_path: Path) -> dict:
    payload = load_json(manifest_path)
    if (
        payload.get("experiment") != "v188_post_confirmation_robustness_matrix"
        or payload.get("confirmatory_extension") is not True
        or tuple(row.get("key") for row in payload.get("scopes") or ())
        != tuple(row["key"] for row in SCOPE_SPECS)
        or payload.get("selected_schedule") not in OPPOSITE_SCHEDULE
        or payload.get("opposite_schedule")
        != OPPOSITE_SCHEDULE[payload["selected_schedule"]]
        or set(payload.get("method_templates") or {}) != set(MECHANISM_METHODS)
    ):
        raise ValueError("invalid v188 manifest")
    provenance = payload["v187_provenance"]
    for key in (
        "decision",
        "input_manifest",
        "published_manifest",
        "generation_contract",
    ):
        path = Path(provenance[key])
        if not path.is_file() or sha256(path) != provenance[f"{key}_sha256"]:
            raise ValueError(f"v188 upstream provenance drifted: {key}")
    for method in BASE_METHODS:
        row = provenance["source_methods"][method]
        audit = Path(row["audit"])
        video_dir = Path(row["video_dir"])
        if (
            not audit.is_file()
            or sha256(audit) != row["audit_sha256"]
            or not video_dir.is_dir()
        ):
            raise ValueError(f"v188 reusable source drifted: {method}")
    templates = payload["method_templates"]
    for method in CACHE_METHODS:
        row = templates[method]
        map_path = Path(row["head_map"])
        if (
            row.get("head_route_counts") != {"10": 360, "11": 0}
            or int(row.get("read_frame_equivalents", -1)) != 9
            or row.get("clean_policy") != "recent"
            or row.get("schedule") not in SCHEDULE_CALLS
            or row.get("coverage_noisy_calls")
            != list(SCHEDULE_CALLS[row["schedule"]])
            or not map_path.is_file()
            or sha256(map_path) != row["head_map_sha256"]
        ):
            raise ValueError(f"v188 cache method drifted: {method}")
    observed: set[int] = set()
    for frozen, expected in zip(payload["scopes"], SCOPE_SPECS):
        for key, value in expected.items():
            expected_value = list(value) if isinstance(value, tuple) else value
            if frozen.get(key) != expected_value:
                raise ValueError(f"v188 scope contract drifted: {expected['key']}:{key}")
        prompt_path = Path(frozen["prompt_file"])
        items = frozen.get("prompt_items") or ()
        indices = {int(row.get("v187_index", -1)) for row in items}
        if (
            len(items) != int(expected["prompt_count"])
            or [int(row.get("index", -1)) for row in items]
            != list(range(int(expected["prompt_count"])))
            or len(indices) != len(items)
            or observed & indices
            or not prompt_path.is_file()
            or sha256(prompt_path) != frozen["prompt_file_sha256"]
            or prompt_path.read_text(encoding="utf-8").splitlines()
            != [str(row["text"]) for row in items]
        ):
            raise ValueError(f"v188 scope prompt contract drifted: {expected['key']}")
        observed.update(indices)
    if observed != set(range(128)):
        raise ValueError("v188 prompt partition is not complete")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--v187-decision", type=Path, required=True)
    prepare_parser.add_argument("--v187-input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--v187-published", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        prepare(
            args.v187_decision,
            args.v187_input_manifest,
            args.v187_published,
            args.output_root,
        )
        if args.action == "prepare"
        else verify(args.manifest)
    )
    videos = sum(
        int(scope["prompt_count"]) * len(scope["generated_methods"])
        for scope in payload["scopes"]
    )
    print(
        "[v188-inputs] PASS "
        f"schedule={payload['selected_schedule']} "
        f"opposite={payload['opposite_schedule']} "
        f"operator={payload['selected_operator']} "
        f"scopes={len(payload['scopes'])} new_videos={videos}"
    )


if __name__ == "__main__":
    main()
