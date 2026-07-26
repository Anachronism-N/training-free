#!/usr/bin/env python3
"""Build PF-independent history-polarity maps from frozen QK statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


SUPPORT_LABEL = 10
SUPPRESS_LABEL = 11
PF_NAMES = {-1: "wave", 1: "anchor", 2: "veil"}
SCORE_ARTIFACT_METHOD = "v98_middle_relative_qk_head_scores"
PRIMARY_SCORE_FIELD = "middle_relative_logit_margin"
FROZEN_BOOTSTRAP_PROTOCOL = {
    "rounds": 500,
    "seed": 20260726,
    "zero_effect_is_stable": False,
}
FROZEN_ACCEPTANCE_PROTOCOL = {
    "min_profiles_per_policy_head": 32,
    "min_stable_head_fraction": 0.80,
    "min_head_bootstrap_agreement": 0.75,
    "min_topology_sign_agreement_fraction": 0.80,
    "min_minority_fraction": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--score-artifact", required=True, type=Path)
    parser.add_argument("--pf-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--polarity-thresholds", default="-0.1,0,0.1")
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted(
        {
            float(item.strip())
            for item in str(value).split(",")
            if item.strip()
        }
    )
    if not thresholds or any(not math.isfinite(value) for value in thresholds):
        raise ValueError("polarity thresholds must be finite")
    if 0.0 not in thresholds:
        raise ValueError("polarity thresholds must include the natural zero split")
    slugs = [threshold_slug(item) for item in thresholds]
    if len(slugs) != len(set(slugs)):
        raise ValueError(
            "polarity thresholds collide after filename normalization; "
            "use values with distinct six-significant-digit representations"
        )
    return thresholds


def threshold_slug(value: float) -> str:
    if value == 0:
        return "zero"
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def read_matrix(
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


def read_scores(
    path: Path,
    num_layers: int,
    num_heads: int,
) -> dict[tuple[int, int], dict[str, object]]:
    numeric_fields = {
        PRIMARY_SCORE_FIELD,
        "uniform_stride_margin",
        "uniform_merge_margin",
        "topology_sign_agreement",
        "profile_observation_count",
        "record_observation_count",
        "profile_positive_fraction",
        "bootstrap_sign_agreement",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = num_layers * num_heads
    if len(rows) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} heads, found {len(rows)}"
        )

    result: dict[tuple[int, int], dict[str, object]] = {}
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        if key in result:
            raise ValueError(f"{path}: duplicate head {key}")
        parsed: dict[str, object] = dict(row)
        for field in numeric_fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite {field} for head {key}")
            parsed[field] = value
        positive_fraction = float(parsed["profile_positive_fraction"])
        bootstrap_agreement = float(parsed["bootstrap_sign_agreement"])
        topology_agreement = float(parsed["topology_sign_agreement"])
        profile_count = float(parsed["profile_observation_count"])
        record_count = float(parsed["record_observation_count"])
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError(
                f"{path}: profile_positive_fraction outside [0, 1] "
                f"for head {key}: {positive_fraction}"
            )
        if not 0.0 <= bootstrap_agreement <= 1.0:
            raise ValueError(
                f"{path}: bootstrap_sign_agreement outside [0, 1] "
                f"for head {key}: {bootstrap_agreement}"
            )
        if topology_agreement not in {0.0, 1.0}:
            raise ValueError(
                f"{path}: topology_sign_agreement must be 0 or 1 "
                f"for head {key}: {topology_agreement}"
            )
        if (
            not profile_count.is_integer()
            or profile_count <= 0
            or not record_count.is_integer()
            or record_count <= 0
        ):
            raise ValueError(
                f"{path}: observation counts must be positive integers "
                f"for head {key}"
            )
        result[key] = parsed

    expected = {
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    }
    if set(result) != expected:
        raise ValueError(
            f"{path}: incomplete layer/head grid, "
            f"missing={sorted(expected - set(result))[:8]}"
        )
    return result


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
            raise ValueError(f"{path}:{line_number}: duplicate key {key}")
        result[key] = value
    return result


def _paired_cluster_bootstrap_sign_agreement(
    values_by_policy: dict[str, dict[str, float]],
    pair_by_sample: dict[str, str],
    *,
    rounds: int,
    seed: int,
) -> float:
    pair_ids = sorted(set(pair_by_sample.values()))
    if not pair_ids:
        raise ValueError("paired bootstrap requires counterfactual pairs")
    grouped = {
        policy: {
            pair_id: [
                value
                for sample, value in sorted(values.items())
                if pair_by_sample[sample] == pair_id
            ]
            for pair_id in pair_ids
        }
        for policy, values in sorted(values_by_policy.items())
    }

    def estimate(selected_pairs: list[str]) -> float:
        policy_medians = []
        for pair_values in grouped.values():
            selected = [
                value
                for pair_id in selected_pairs
                for value in pair_values[pair_id]
            ]
            policy_medians.append(float(statistics.median(selected)))
        return float(statistics.median(policy_medians))

    reference = estimate(pair_ids)
    if reference == 0.0:
        return 0.0
    reference_positive = reference > 0.0
    rng = random.Random(seed)
    agreements = 0
    for _ in range(rounds):
        selected_pairs = [
            pair_ids[rng.randrange(len(pair_ids))] for _ in pair_ids
        ]
        bootstrap_estimate = estimate(selected_pairs)
        agreements += (
            bootstrap_estimate != 0.0
            and (bootstrap_estimate > 0.0) == reference_positive
        )
    return agreements / rounds


def validate_observations(
    observations_path: Path,
    *,
    scores: dict[tuple[int, int], dict[str, object]],
    samples_by_policy: dict[str, set[tuple[str, str, int]]],
    num_layers: int,
    num_heads: int,
) -> None:
    payload = json.loads(observations_path.read_text(encoding="utf-8"))
    expected_header = {
        "version": 3,
        "method": SCORE_ARTIFACT_METHOD,
        "primary_field": PRIMARY_SCORE_FIELD,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"observation artifact {key} mismatch: "
                f"expected={expected!r} actual={payload.get(key)!r}"
            )
    raw_heads = payload.get("per_head_policy_profile_margins")
    if not isinstance(raw_heads, dict):
        raise ValueError("observation artifact has no per-head margins")
    expected_head_names = {
        f"L{layer}H{head}"
        for layer in range(num_layers)
        for head in range(num_heads)
    }
    if set(raw_heads) != expected_head_names:
        raise ValueError(
            "observation artifact head grid mismatch: "
            f"missing={sorted(expected_head_names - set(raw_heads))[:8]} "
            f"extra={sorted(set(raw_heads) - expected_head_names)[:8]}"
        )

    expected_sample_names = {
        policy: {
            f"{pair_id}|{side}|{seed}"
            for pair_id, side, seed in samples
        }
        for policy, samples in samples_by_policy.items()
    }
    pair_by_sample = {
        f"{pair_id}|{side}|{seed}": pair_id
        for samples in samples_by_policy.values()
        for pair_id, side, seed in samples
    }
    expected_policies = {"uniform_stride", "uniform_merge"}
    for layer in range(num_layers):
        for head in range(num_heads):
            head_name = f"L{layer}H{head}"
            raw_policies = raw_heads[head_name]
            if not isinstance(raw_policies, dict) or set(raw_policies) != (
                expected_policies
            ):
                raise ValueError(
                    f"{head_name}: observation probe-policy set is invalid"
                )
            values_by_policy: dict[str, dict[str, float]] = {}
            for policy in sorted(expected_policies):
                raw_values = raw_policies[policy]
                if (
                    not isinstance(raw_values, dict)
                    or set(raw_values) != expected_sample_names[policy]
                ):
                    raise ValueError(
                        f"{head_name}/{policy}: observation sample set "
                        "does not match the 32 frozen profiles"
                    )
                values: dict[str, float] = {}
                for sample, raw_value in raw_values.items():
                    if isinstance(raw_value, bool):
                        raise ValueError(
                            f"{head_name}/{policy}/{sample}: boolean margin"
                        )
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError(
                            f"{head_name}/{policy}/{sample}: non-finite margin"
                        )
                    values[sample] = value
                values_by_policy[policy] = values

            policy_scores = {
                policy: float(statistics.median(values.values()))
                for policy, values in sorted(values_by_policy.items())
            }
            stride = policy_scores["uniform_stride"]
            merge = policy_scores["uniform_merge"]
            recomputed = {
                PRIMARY_SCORE_FIELD: float(
                    statistics.median(policy_scores.values())
                ),
                "uniform_stride_margin": stride,
                "uniform_merge_margin": merge,
                "topology_sign_agreement": float(
                    stride != 0.0
                    and merge != 0.0
                    and (stride > 0.0) == (merge > 0.0)
                ),
                "profile_observation_count": float(
                    sum(len(values) for values in values_by_policy.values())
                ),
                "profile_positive_fraction": sum(
                    sum(value > 0.0 for value in values.values()) / len(values)
                    for values in values_by_policy.values()
                )
                / len(values_by_policy),
                "bootstrap_sign_agreement": (
                    _paired_cluster_bootstrap_sign_agreement(
                        values_by_policy,
                        pair_by_sample,
                        rounds=FROZEN_BOOTSTRAP_PROTOCOL["rounds"],
                        seed=(
                            FROZEN_BOOTSTRAP_PROTOCOL["seed"]
                            + layer * num_heads
                            + head
                        ),
                    )
                ),
            }
            score = scores[(layer, head)]
            for field, expected in recomputed.items():
                actual = float(score[field])
                if not math.isclose(
                    actual, expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"{head_name}: score CSV {field} is not derived "
                        f"from frozen observations: "
                        f"expected={expected} actual={actual}"
                    )


def validate_score_artifact(
    artifact: dict[str, object],
    *,
    artifact_path: Path,
    scores_path: Path,
    scores: dict[tuple[int, int], dict[str, object]],
    num_layers: int,
    num_heads: int,
) -> str:
    if int(artifact.get("version", -1)) != 2:
        raise ValueError("score artifact version must be exactly 2")
    if artifact.get("method") != SCORE_ARTIFACT_METHOD:
        raise ValueError(
            "score artifact method mismatch: "
            f"expected={SCORE_ARTIFACT_METHOD!r} "
            f"actual={artifact.get('method')!r}"
        )
    if artifact.get("accepted") is not True:
        raise ValueError("score artifact did not pass its frozen acceptance gates")
    if int(artifact.get("num_layers", -1)) != num_layers:
        raise ValueError("score artifact layer count mismatch")
    if int(artifact.get("num_heads", -1)) != num_heads:
        raise ValueError("score artifact head count mismatch")
    if int(artifact.get("head_count", -1)) != num_layers * num_heads:
        raise ValueError("score artifact total head count mismatch")

    definition = dict(artifact.get("score_definition") or {})
    expected_definition = {
        "primary_field": PRIMARY_SCORE_FIELD,
        "branch": "cond",
        "update_mode": "noisy",
        "sink_frames_excluded": 3,
        "recent_distinct_key_frames": 4,
        "common_logit_shift_invariant": True,
        "pf_labels_used": False,
        "probe_policy_balanced": True,
        "probe_policies": ["uniform_stride", "uniform_merge"],
        "bootstrap_unit": "counterfactual_prompt_pair",
    }
    mismatches = {
        key: (expected, definition.get(key))
        for key, expected in expected_definition.items()
        if definition.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"score definition mismatch: {mismatches}")
    bootstrap_protocol = dict(artifact.get("bootstrap_protocol") or {})
    if bootstrap_protocol != FROZEN_BOOTSTRAP_PROTOCOL:
        raise ValueError(
            "score artifact bootstrap protocol mismatch: "
            f"expected={FROZEN_BOOTSTRAP_PROTOCOL} actual={bootstrap_protocol}"
        )
    acceptance_protocol = dict(artifact.get("acceptance_protocol") or {})
    if acceptance_protocol != FROZEN_ACCEPTANCE_PROTOCOL:
        raise ValueError(
            "score artifact acceptance protocol mismatch: "
            f"expected={FROZEN_ACCEPTANCE_PROTOCOL} actual={acceptance_protocol}"
        )
    gates = dict(artifact.get("acceptance_gates") or {})
    expected_gate_requirements = {
        "bootstrap_stable_head_fraction": {
            "required": FROZEN_ACCEPTANCE_PROTOCOL[
                "min_stable_head_fraction"
            ],
            "per_head_threshold": FROZEN_ACCEPTANCE_PROTOCOL[
                "min_head_bootstrap_agreement"
            ],
        },
        "topology_sign_agreement_fraction": {
            "required": FROZEN_ACCEPTANCE_PROTOCOL[
                "min_topology_sign_agreement_fraction"
            ],
        },
        "minority_role_fraction": {
            "required": FROZEN_ACCEPTANCE_PROTOCOL[
                "min_minority_fraction"
            ],
        },
    }
    if set(gates) != {
        "complete_head_grid",
        *expected_gate_requirements,
    }:
        raise ValueError("score artifact acceptance gate set is incomplete")
    for name, required_fields in expected_gate_requirements.items():
        gate = dict(gates.get(name) or {})
        if gate.get("passed") is not True:
            raise ValueError(f"score artifact gate {name} did not pass")
        for field, expected in required_fields.items():
            if gate.get(field) != expected:
                raise ValueError(
                    f"score artifact gate {name}/{field} mismatch: "
                    f"expected={expected} actual={gate.get(field)}"
                )
    complete_gate = dict(gates.get("complete_head_grid") or {})
    if (
        complete_gate.get("passed") is not True
        or int(complete_gate.get("required", -1)) != num_layers * num_heads
        or int(complete_gate.get("observed", -1)) != num_layers * num_heads
    ):
        raise ValueError("score artifact complete-head-grid gate is invalid")
    head_count = num_layers * num_heads
    stable_fraction = (
        sum(
            float(row["bootstrap_sign_agreement"])
            >= FROZEN_ACCEPTANCE_PROTOCOL[
                "min_head_bootstrap_agreement"
            ]
            for row in scores.values()
        )
        / head_count
    )
    topology_values = {
        key: int(
            float(row["uniform_stride_margin"]) != 0.0
            and float(row["uniform_merge_margin"]) != 0.0
            and (
                (float(row["uniform_stride_margin"]) > 0.0)
                == (float(row["uniform_merge_margin"]) > 0.0)
            )
        )
        for key, row in scores.items()
    }
    for key, expected in topology_values.items():
        actual = int(float(scores[key]["topology_sign_agreement"]))
        if actual != expected:
            raise ValueError(
                f"score CSV topology_sign_agreement is inconsistent "
                f"for head {key}: expected={expected} actual={actual}"
            )
    topology_fraction = sum(topology_values.values()) / head_count
    support_count = sum(
        float(row[PRIMARY_SCORE_FIELD]) >= 0.0
        for row in scores.values()
    )
    suppress_count = head_count - support_count
    minority_fraction = min(support_count, suppress_count) / head_count
    recomputed_observations = {
        "bootstrap_stable_head_fraction": stable_fraction,
        "topology_sign_agreement_fraction": topology_fraction,
        "minority_role_fraction": minority_fraction,
    }
    for name, observed in recomputed_observations.items():
        declared = gates[name].get("observed")
        if (
            not isinstance(declared, (int, float))
            or not math.isclose(
                float(declared),
                observed,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"score artifact gate {name}/observed does not match "
                f"the score CSV: declared={declared} recomputed={observed}"
            )
        required = float(expected_gate_requirements[name]["required"])
        expected_pass = observed >= required
        if bool(gates[name].get("passed")) != expected_pass:
            raise ValueError(
                f"score artifact gate {name}/passed is inconsistent: "
                f"observed={observed} required={required}"
            )
        if not expected_pass:
            raise ValueError(
                f"score artifact gate {name} is below the frozen threshold: "
                f"observed={observed} required={required}"
            )
    expected_label_counts = {
        "10": support_count,
        "11": suppress_count,
    }
    if artifact.get("label_counts_at_zero") != expected_label_counts:
        raise ValueError(
            "score artifact zero-threshold label counts do not match "
            f"the score CSV: expected={expected_label_counts} "
            f"actual={artifact.get('label_counts_at_zero')}"
        )
    if any(
        int(float(row["profile_observation_count"])) != 64
        for row in scores.values()
    ):
        raise ValueError(
            "score CSV must contain exactly 32 observations per probe "
            "policy and head (64 balanced observations total)"
        )

    files = dict(artifact.get("files") or {})
    expected_score_hash = files.get("score_csv_sha256")
    actual_score_hash = sha256(scores_path)
    if expected_score_hash != actual_score_hash:
        raise ValueError(
            "score CSV does not match immutable artifact: "
            f"expected={expected_score_hash} actual={actual_score_hash}"
        )
    artifact_dir = artifact_path.resolve().parent
    dependencies: dict[str, Path] = {}
    for path_key, hash_key in (
        ("observations", "observations_sha256"),
        ("run_manifest", "run_manifest_sha256"),
    ):
        raw_path = files.get(path_key)
        expected_hash = files.get(hash_key)
        if not raw_path or not expected_hash:
            raise ValueError(f"score artifact is missing {path_key}/{hash_key}")
        dependency = Path(str(raw_path))
        if not dependency.is_absolute():
            dependency = artifact_dir / dependency
        if not dependency.is_file():
            raise ValueError(f"score artifact dependency is missing: {dependency}")
        actual_hash = sha256(dependency)
        if actual_hash != expected_hash:
            raise ValueError(
                f"score artifact dependency hash mismatch for {path_key}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        dependencies[path_key] = dependency.resolve()
    profile_protocol = dict(artifact.get("profile_protocol") or {})
    run_manifest = load_env(dependencies["run_manifest"])
    if profile_protocol != run_manifest:
        raise ValueError(
            "score artifact profile_protocol is not an exact copy of the "
            "hash-bound run manifest"
        )
    expected_protocol = {
        "EXPERIMENT": "v98_middle_relative_scores",
        "PROFILE_FRAMES": "120",
        "PROFILE_BRANCHES": "cond",
        "PROFILE_UPDATE_MODES": "noisy",
        "FEW_STEP_CFG_ENABLED": "0",
        "PROBE_POLICIES": "uniform_stride,uniform_merge",
        "PAIR_COUNT": "8",
        "PROFILE_COUNT_PER_POLICY": "32",
        "PROFILE_COUNT": "64",
        "SEEDS": "0 1",
        "SINK_FRAMES": "3",
        "RECENT_FRAMES": "4",
        "TRACKED_WORKTREE_DIRTY": "0",
    }
    protocol_mismatches = {
        key: (expected, profile_protocol.get(key))
        for key, expected in expected_protocol.items()
        if profile_protocol.get(key) != expected
    }
    if protocol_mismatches:
        raise ValueError(
            f"score artifact profile protocol mismatch: {protocol_mismatches}"
        )
    if not run_manifest.get("RUN_COMMIT"):
        raise ValueError("score artifact run manifest has no source commit")
    run_dependencies: dict[str, Path] = {}
    for path_key, hash_key in (
        ("CONFIG", "CONFIG_SHA256"),
        ("CHECKPOINT", "CHECKPOINT_SHA256"),
        ("PAIR_JSON", "PAIR_SHA256"),
        ("PROBE_MAP_STRIDE", "PROBE_MAP_STRIDE_SHA256"),
        ("PROBE_MAP_MERGE", "PROBE_MAP_MERGE_SHA256"),
    ):
        dependency = Path(str(run_manifest.get(path_key) or ""))
        expected_hash = str(run_manifest.get(hash_key) or "")
        if not dependency.is_file() or not expected_hash:
            raise ValueError(
                f"score run manifest dependency is missing: {path_key}"
            )
        actual_hash = sha256(dependency)
        if actual_hash != expected_hash:
            raise ValueError(
                f"score run manifest dependency hash mismatch for {path_key}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        run_dependencies[path_key] = dependency.resolve()
    pair_payload = json.loads(
        run_dependencies["PAIR_JSON"].read_text(encoding="utf-8")
    )
    pair_rows = list(pair_payload.get("prompt_pairs") or [])
    expected_pair_ids = {str(row.get("id") or "") for row in pair_rows}
    if (
        len(pair_rows) != 8
        or len(expected_pair_ids) != 8
        or "" in expected_pair_ids
    ):
        raise ValueError(
            "score run manifest pair file must contain eight unique pairs"
        )
    expected_prompts = {
        (str(row["id"]), side): str(row[side])
        for row in pair_rows
        for side in ("a", "b")
    }
    for path_key, expected_label in (
        ("PROBE_MAP_STRIDE", 1),
        ("PROBE_MAP_MERGE", 2),
    ):
        probe = read_matrix(
            run_dependencies[path_key], num_layers, num_heads
        )
        if set(flatten(probe)) != {expected_label}:
            raise ValueError(
                f"{path_key} is not a uniform label-{expected_label} probe"
            )

    profile_audit = list(artifact.get("profile_audit") or [])
    if len(profile_audit) != 64:
        raise ValueError(
            "score artifact must retain exactly 64 independently generated "
            f"profile records, found {len(profile_audit)}"
        )
    seen_profiles: set[str] = set()
    samples_by_policy: dict[str, set[tuple[str, str, int]]] = {}
    for item in profile_audit:
        if not isinstance(item, dict):
            raise ValueError("score artifact profile_audit entries must be objects")
        profile_path = Path(str(item.get("path") or ""))
        expected_hash = item.get("sha256")
        if not profile_path.is_file() or not expected_hash:
            raise ValueError(
                f"score artifact profile dependency is missing: {profile_path}"
            )
        canonical = str(profile_path.resolve())
        if canonical in seen_profiles:
            raise ValueError(f"duplicate profile dependency {profile_path}")
        seen_profiles.add(canonical)
        actual_hash = sha256(profile_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"profile dependency hash mismatch for {profile_path}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        prompt_path = Path(str(item.get("prompt_path") or ""))
        expected_prompt_hash = item.get("prompt_sha256")
        if not prompt_path.is_file() or not expected_prompt_hash:
            raise ValueError(
                f"score artifact prompt dependency is missing: {prompt_path}"
            )
        if sha256(prompt_path) != expected_prompt_hash:
            raise ValueError(
                f"score artifact prompt dependency hash mismatch: {prompt_path}"
            )
        policy = str(item.get("probe_policy") or "")
        pair_id = str(item.get("pair_id") or "")
        side = str(item.get("side") or "")
        seed = int(item.get("seed", -1))
        if policy not in {"uniform_stride", "uniform_merge"}:
            raise ValueError(f"invalid profile probe policy {policy!r}")
        if not pair_id or side not in {"a", "b"} or seed not in {0, 1}:
            raise ValueError(f"invalid frozen profile sample metadata: {item}")
        prompt_lines = [
            line.strip()
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            pair_id not in expected_pair_ids
            or prompt_lines != [expected_prompts[(pair_id, side)]]
        ):
            raise ValueError(
                f"profile prompt does not match frozen pair {pair_id}/{side}"
            )
        sample = (pair_id, side, seed)
        policy_samples = samples_by_policy.setdefault(policy, set())
        if sample in policy_samples:
            raise ValueError(f"duplicate profile sample for {policy}: {sample}")
        policy_samples.add(sample)
    if set(samples_by_policy) != {"uniform_stride", "uniform_merge"}:
        raise ValueError("score artifact is missing a frozen probe policy")
    if (
        len(samples_by_policy["uniform_stride"]) != 32
        or samples_by_policy["uniform_stride"]
        != samples_by_policy["uniform_merge"]
    ):
        raise ValueError("score artifact probe policies are not sample-balanced")
    reference_samples = samples_by_policy["uniform_stride"]
    if {pair_id for pair_id, _, _ in reference_samples} != expected_pair_ids:
        raise ValueError("score artifact profile pair ids do not match pair file")
    expected_pair_cells = {
        (side, seed) for side in ("a", "b") for seed in (0, 1)
    }
    for pair_id in sorted(expected_pair_ids):
        observed_cells = {
            (side, seed)
            for observed_pair, side, seed in reference_samples
            if observed_pair == pair_id
        }
        if observed_cells != expected_pair_cells:
            raise ValueError(
                f"score artifact pair {pair_id} does not cover a/b x seeds 0/1"
            )
    validate_observations(
        dependencies["observations"],
        scores=scores,
        samples_by_policy=samples_by_policy,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    return actual_score_hash


def score_map(
    scores: dict[tuple[int, int], dict[str, object]],
    *,
    column: str,
    threshold: float,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    return [
        [
            (
                SUPPORT_LABEL
                if float(scores[(layer, head)][column]) >= threshold
                else SUPPRESS_LABEL
            )
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def random_count_control(
    reference: list[list[int]],
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    result: list[list[int]] = []
    for row in reference:
        support_count = row.count(SUPPORT_LABEL)
        heads = list(range(len(row)))
        rng.shuffle(heads)
        support = set(heads[:support_count])
        result.append(
            [
                SUPPORT_LABEL if head in support else SUPPRESS_LABEL
                for head in range(len(row))
            ]
        )
    return result


def pf_cross_tab(
    matrix: list[list[int]],
    pf_labels: list[list[int]],
) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for pf_label, name in PF_NAMES.items():
        roles = [
            matrix[layer][head]
            for layer, row in enumerate(pf_labels)
            for head, value in enumerate(row)
            if value == pf_label
        ]
        table[name] = {
            "pf_label": pf_label,
            "heads": len(roles),
            "history_supportive": roles.count(SUPPORT_LABEL),
            "history_suppressive": roles.count(SUPPRESS_LABEL),
        }
    return table


def binary_agreement(
    matrix: list[list[int]],
    reference: list[list[int]],
    *,
    positive_label: int,
) -> dict[str, float | int]:
    predicted = flatten(matrix)
    truth = flatten(reference)
    positive = {
        index for index, value in enumerate(predicted) if value == SUPPRESS_LABEL
    }
    reference_positive = {
        index for index, value in enumerate(truth) if value == positive_label
    }
    tp = len(positive & reference_positive)
    fp = len(positive - reference_positive)
    fn = len(reference_positive - positive)
    tn = len(predicted) - tp - fp - fn
    tpr = tp / (tp + fn) if tp + fn else 1.0
    tnr = tn / (tn + fp) if tn + fp else 1.0
    union = positive | reference_positive
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "agreement": (tp + tn) / len(predicted),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "suppressive_jaccard": (
            len(positive & reference_positive) / len(union) if union else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    artifact = json.loads(args.score_artifact.read_text(encoding="utf-8"))
    scores = read_scores(args.scores, args.num_layers, args.num_heads)
    actual_score_hash = validate_score_artifact(
        artifact,
        artifact_path=args.score_artifact,
        scores_path=args.scores,
        scores=scores,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    pf_labels = read_matrix(args.pf_labels, args.num_layers, args.num_heads)
    pf_values = set(flatten(pf_labels))
    if pf_values != set(PF_NAMES):
        raise ValueError(
            f"{args.pf_labels}: expected PF labels {sorted(PF_NAMES)}, "
            f"found {sorted(pf_values)}"
        )
    thresholds = parse_thresholds(args.polarity_thresholds)
    if thresholds != [-0.1, 0.0, 0.1]:
        raise ValueError(
            "v98 polarity thresholds are frozen at -0.1,0,0.1"
        )
    if args.random_seed != 2026:
        raise ValueError("v98 random-control seed is frozen at 2026")
    if args.validate_only:
        print(
            "[V98HistoryPolarityMaps] frozen score evidence validated",
            flush=True,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    maps: dict[str, list[list[int]]] = {}
    sources: dict[str, dict[str, object]] = {}
    for threshold in thresholds:
        name = f"history_polarity_{threshold_slug(threshold)}"
        maps[name] = score_map(
            scores,
            column=PRIMARY_SCORE_FIELD,
            threshold=threshold,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
        sources[name] = {
            "family": "middle_relative_history_preference",
            "score": (
                "equal_probe_policy_median(median_profiles(median_records("
                "standardized(mean_middle_logits-mean_recent_logits))))"
            ),
            "score_column": PRIMARY_SCORE_FIELD,
            "support_rule": f"{PRIMARY_SCORE_FIELD} >= {threshold}",
            "threshold": threshold,
            "threshold_provenance": (
                "shift_invariant_equal_preference_zero_no_pf_labels"
                if threshold == 0
                else "symmetric_robustness_ablation"
            ),
        }

    maps["positive_rate_half"] = score_map(
        scores,
        column="profile_positive_fraction",
        threshold=0.5,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    sources["positive_rate_half"] = {
        "family": "middle_relative_margin_sign_fraction",
        "score_column": "profile_positive_fraction",
        "support_rule": "profile_positive_fraction >= 0.5",
        "threshold": 0.5,
        "threshold_provenance": "majority_profile_middle_preference",
    }
    maps["pf_aw_binary_control"] = [
        [
            SUPPORT_LABEL if value in {-1, 1} else SUPPRESS_LABEL
            for value in row
        ]
        for row in pf_labels
    ]
    sources["pf_aw_binary_control"] = {
        "family": "pf_oracle_control_not_proposed_classifier",
        "support": ["anchor", "wave"],
        "suppress": ["veil"],
    }
    maps["pf_ar_binary_control"] = [
        [
            SUPPORT_LABEL if value == 1 else SUPPRESS_LABEL
            for value in row
        ]
        for row in pf_labels
    ]
    sources["pf_ar_binary_control"] = {
        "family": "pf_oracle_control_not_proposed_classifier",
        "support": ["anchor"],
        "suppress": ["wave", "veil"],
    }
    maps["history_polarity_zero_random"] = random_count_control(
        maps["history_polarity_zero"], args.random_seed
    )
    sources["history_polarity_zero_random"] = {
        "family": "layerwise_count_matched_random_control",
        "reference": "history_polarity_zero",
        "seed": args.random_seed,
    }
    maps["history_polarity_zero_inverted"] = [
        [
            SUPPRESS_LABEL if value == SUPPORT_LABEL else SUPPORT_LABEL
            for value in row
        ]
        for row in maps["history_polarity_zero"]
    ]
    sources["history_polarity_zero_inverted"] = {
        "family": "causal_role_assignment_inversion_control",
        "reference": "history_polarity_zero",
        "mapping": {
            str(SUPPORT_LABEL): SUPPRESS_LABEL,
            str(SUPPRESS_LABEL): SUPPORT_LABEL,
        },
    }

    manifest_maps: dict[str, dict[str, object]] = {}
    for name, matrix in maps.items():
        values = flatten(matrix)
        observed_labels = set(values)
        if not observed_labels or not observed_labels.issubset(
            {SUPPORT_LABEL, SUPPRESS_LABEL}
        ):
            raise ValueError(
                f"{name}: map contains invalid neutral role labels "
                f"{sorted(observed_labels)}"
            )
        if (
            name == "history_polarity_zero"
            and observed_labels != {SUPPORT_LABEL, SUPPRESS_LABEL}
        ):
            raise ValueError(
                "history_polarity_zero must contain both accepted role labels"
            )
        path = args.output_dir / f"{name}.csv"
        write_matrix(path, matrix)
        manifest_maps[name] = {
            **sources[name],
            "path": path.name,
            "sha256": sha256(path),
            "label_counts": dict(sorted(Counter(values).items())),
            "pf_cross_tab": pf_cross_tab(matrix, pf_labels),
            "agreement_pf_aw": binary_agreement(
                matrix, pf_labels, positive_label=2
            ),
            "agreement_pf_ar": binary_agreement(
                matrix,
                [
                    [0 if value == 1 else 3 for value in row]
                    for row in pf_labels
                ],
                positive_label=3,
            ),
        }

    assignments_path = args.output_dir / "head_assignments.csv"
    assignment_fields = [
        "layer",
        "head",
        "pf_label",
        "pf_name",
        PRIMARY_SCORE_FIELD,
        "uniform_stride_margin",
        "uniform_merge_margin",
        "topology_sign_agreement",
        "profile_positive_fraction",
        "bootstrap_sign_agreement",
        "profile_observation_count",
        "record_observation_count",
        *maps,
    ]
    with assignments_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fields)
        writer.writeheader()
        for layer in range(args.num_layers):
            for head in range(args.num_heads):
                score = scores[(layer, head)]
                pf_label = pf_labels[layer][head]
                writer.writerow(
                    {
                        "layer": layer,
                        "head": head,
                        "pf_label": pf_label,
                        "pf_name": PF_NAMES.get(pf_label, "unknown"),
                        PRIMARY_SCORE_FIELD: score[PRIMARY_SCORE_FIELD],
                        "uniform_stride_margin": score[
                            "uniform_stride_margin"
                        ],
                        "uniform_merge_margin": score[
                            "uniform_merge_margin"
                        ],
                        "topology_sign_agreement": score[
                            "topology_sign_agreement"
                        ],
                        "profile_positive_fraction": score[
                            "profile_positive_fraction"
                        ],
                        "bootstrap_sign_agreement": score[
                            "bootstrap_sign_agreement"
                        ],
                        "profile_observation_count": score[
                            "profile_observation_count"
                        ],
                        "record_observation_count": score[
                            "record_observation_count"
                        ],
                        **{
                            name: matrix[layer][head]
                            for name, matrix in maps.items()
                        },
                    }
                )

    manifest = {
        "version": 2,
        "method": "v98_middle_relative_history_map_builder",
        "support_label": SUPPORT_LABEL,
        "suppress_label": SUPPRESS_LABEL,
        "reserved_pf_labels": [-1, 1, 2],
        "score_field": PRIMARY_SCORE_FIELD,
        "score_csv": str(args.scores.resolve()),
        "score_csv_sha256": actual_score_hash,
        "score_artifact": str(args.score_artifact.resolve()),
        "score_artifact_sha256": sha256(args.score_artifact),
        "pf_labels": str(args.pf_labels.resolve()),
        "pf_labels_sha256": sha256(args.pf_labels),
        "thresholds": thresholds,
        "maps": manifest_maps,
        "assignments": {
            "path": str(assignments_path.resolve()),
            "sha256": sha256(assignments_path),
        },
        "claims": {
            "primary_classifier": "history_polarity_zero",
            "pf_labels_used_for_primary_classifier": False,
            "pf_labels_used_for_controls_and_posthoc_analysis": True,
            "prompt_sensitivity_used_as_static_classifier": False,
            "absolute_logit_sign_used_as_primary_classifier": False,
            "common_logit_shift_invariant": True,
            "sink_recent_excluded_from_middle_score": True,
            "probe_policy_balanced": True,
        },
    }
    manifest_path = args.output_dir / "history_polarity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[HistoryPolarityMaps] "
        f"scores={actual_score_hash} maps={len(maps)} "
        f"primary_counts={manifest_maps['history_polarity_zero']['label_counts']} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
