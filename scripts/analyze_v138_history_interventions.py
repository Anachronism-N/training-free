#!/usr/bin/env python3
"""Analyze v138 causal history interventions and cross-video specificity."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


COMMON_PATH = Path(__file__).with_name(
    "analyze_v136_multi_axis_head_discovery.py"
)
COMMON_SPEC = importlib.util.spec_from_file_location(
    "v136_analysis_common", COMMON_PATH
)
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

EPS = 1e-4
EXPECTED_LAYERS = 30
EXPECTED_HEADS = 12
REQUIRED_INTERVENTIONS = (
    "reverse",
    "phase_shift",
    "freeze_latest",
    "value_mismatch",
)


def _load_profiles(directory: Path) -> list[dict]:
    profiles = []
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", 0)) != 3:
            raise ValueError(f"v138 requires profile version 3: {path}")
        if "job" not in payload or "records" not in payload:
            raise ValueError(f"malformed profile: {path}")
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def _state_key(record: dict) -> tuple:
    return (
        str(record["mode"]),
        int(record["current_frame"]),
        int(record["nominal_timestep"]),
        int(record["layer"]),
    )


def _audit_profiles(
    profiles: list[dict],
    *,
    expected_count: int,
    expected_states: int,
) -> list[dict]:
    if len(profiles) != expected_count:
        raise ValueError(
            f"profile count mismatch: {len(profiles)} != {expected_count}"
        )
    expected_layers = set(range(EXPECTED_LAYERS))
    output = []
    failures = []
    seen_indices = set()
    run_commits = set()
    for profile in profiles:
        job = profile["job"]
        job_id = str(job["job_id"])
        index = int(job["dataset_index"])
        if index in seen_indices:
            failures.append(f"{job_id}: duplicate dataset_index={index}")
        seen_indices.add(index)
        metadata = profile.get("metadata", {})
        run_commit = str(metadata.get("run_commit") or "")
        if run_commit:
            run_commits.add(run_commit)
        groups = defaultdict(set)
        tensor_failures = []
        max_rope_error = 0.0
        for record in profile["records"]:
            if str(record["branch"]) != "base":
                tensor_failures.append("non_base_branch")
                continue
            state = _state_key(record)[:3]
            groups[state].add(int(record["layer"]))
            required_signatures = (
                "full_history_signature",
                "recent_history_signature",
            ) + tuple(
                f"history_{name}_signature"
                for name in REQUIRED_INTERVENTIONS
            )
            for name in required_signatures:
                value = record.get(name)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 2
                    or value.shape[0] != EXPECTED_HEADS
                ):
                    tensor_failures.append(
                        f"{name}:{getattr(value, 'shape', None)}"
                    )
            query = record.get("query_projection")
            history = record.get("history_key_projection")
            if (
                not isinstance(query, torch.Tensor)
                or query.ndim != 3
                or query.shape[0] != EXPECTED_HEADS
            ):
                tensor_failures.append(
                    f"query_projection:{getattr(query, 'shape', None)}"
                )
            if (
                not isinstance(history, torch.Tensor)
                or history.ndim != 4
                or history.shape[0] != EXPECTED_HEADS
            ):
                tensor_failures.append(
                    f"history_key_projection:{getattr(history, 'shape', None)}"
                )
            if (
                isinstance(query, torch.Tensor)
                and isinstance(history, torch.Tensor)
                and (
                    query.shape[1] != history.shape[2]
                    or query.shape[2] != history.shape[3]
                    or query.shape[2]
                    != int(metadata.get("projection_dim", -1))
                )
            ):
                tensor_failures.append("projection_shape_mismatch")
            error = float(
                record.get(
                    "history_intervention_rope_reconstruction_relative_max",
                    float("inf"),
                )
            )
            if not math.isfinite(error):
                tensor_failures.append("non_finite_rope_error")
            else:
                max_rope_error = max(max_rope_error, error)
        bad_layer_groups = [
            state for state, layers in groups.items() if layers != expected_layers
        ]
        passed = (
            str(job.get("kind")) == "history_intervention"
            and bool(metadata.get("history_interventions", False))
            and int(metadata.get("projection_seed", -1)) == 20260729
            and int(metadata.get("projection_dim", -1)) == 16
            and bool(run_commit)
            and int(job.get("seed", -1)) == 0
            and len(groups) == expected_states
            and not bad_layer_groups
            and not tensor_failures
            and max_rope_error <= 5e-3
        )
        row = {
            "dataset_index": index,
            "job_id": job_id,
            "kind": str(job.get("kind")),
            "records": len(profile["records"]),
            "states": len(groups),
            "bad_layer_groups": len(bad_layer_groups),
            "tensor_failures": len(tensor_failures),
            "max_rope_reconstruction_error": max_rope_error,
            "run_commit": run_commit,
            "passed": int(passed),
        }
        output.append(row)
        if not passed:
            failures.append(
                f"{job_id}: {row}; "
                f"layer_examples={bad_layer_groups[:2]}; "
                f"tensor_examples={tensor_failures[:3]}"
            )
    if seen_indices != set(range(expected_count)):
        failures.append("dataset indices are not exactly 0..expected_count-1")
    if len(run_commits) != 1:
        failures.append(
            f"profiles do not share exactly one run commit: {run_commits}"
        )
    if failures:
        raise ValueError(
            "v138 profile audit failed:\n" + "\n".join(failures[:8])
        )
    return output


def _relative(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return COMMON._relative_per_head(left, right)


def _local_intervention_rows(
    profiles: list[dict],
    recent_frames: int,
) -> list[dict]:
    rows = []
    for profile in profiles:
        job = profile["job"]
        for record in profile["records"]:
            history_frames = int(record["history_frames"])
            if history_frames <= recent_frames:
                continue
            full = record["full_history_signature"]
            recent = record["recent_history_signature"]
            old_effect = _relative(full, recent)
            distances = {
                name: _relative(
                    record[f"history_{name}_signature"], full
                )
                for name in REQUIRED_INTERVENTIONS
            }
            for head in range(EXPECTED_HEADS):
                denominator = float(old_effect[head]) + EPS
                row = {
                    "dataset_index": int(job["dataset_index"]),
                    "job_id": str(job["job_id"]),
                    "mode": str(record["mode"]),
                    "current_frame": int(record["current_frame"]),
                    "nominal_timestep": int(record["nominal_timestep"]),
                    "layer": int(record["layer"]),
                    "head": head,
                    "old_history_effect": float(old_effect[head]),
                }
                for name in REQUIRED_INTERVENTIONS:
                    value = float(distances[name][head])
                    row[f"{name}_effect"] = value
                    row[f"{name}_relative_log"] = math.log(
                        (value + EPS) / denominator
                    )
                row["reverse_vs_phase_log"] = math.log(
                    (float(distances["reverse"][head]) + EPS)
                    / (float(distances["phase_shift"][head]) + EPS)
                )
                rows.append(row)
    return rows


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "camera",
    "cinematic",
    "during",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "scene",
    "the",
    "through",
    "to",
    "video",
    "while",
    "with",
}


def _prompt_tokens(prompt: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", prompt.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _build_donor_map(profiles: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    ordered = sorted(
        profiles, key=lambda profile: int(profile["job"]["dataset_index"])
    )
    job_ids = [str(profile["job"]["job_id"]) for profile in ordered]
    tokens = {
        str(profile["job"]["job_id"]): _prompt_tokens(
            str(profile["job"]["base_prompt"])
        )
        for profile in ordered
    }
    donor_map = {}
    audit = []
    count = len(ordered)
    offsets = (1, 37, 73)
    for position, profile in enumerate(ordered):
        target = str(profile["job"]["job_id"])
        candidates = [job for job in job_ids if job != target]
        hard = max(
            candidates,
            key=lambda job: (_jaccard(tokens[target], tokens[job]), job),
        )
        random_donors = []
        for offset in offsets:
            donor = job_ids[(position + offset) % count]
            if donor == target:
                donor = job_ids[(position + offset + 1) % count]
            if donor not in random_donors and donor != hard:
                random_donors.append(donor)
        if not random_donors:
            random_donors = [hard]
        donor_map[target] = {
            "hard": hard,
            "random": random_donors,
        }
        audit.append(
            {
                "target_job": target,
                "hard_donor_job": hard,
                "hard_lexical_jaccard": _jaccard(
                    tokens[target], tokens[hard]
                ),
                "random_donor_jobs": ",".join(random_donors),
            }
        )
    return donor_map, audit


def _descriptor_similarity(
    query: torch.Tensor,
    history_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        query.ndim != 3
        or history_key.ndim != 4
        or query.shape[0] != history_key.shape[0]
        or query.shape[1] != history_key.shape[2]
        or query.shape[2] != history_key.shape[3]
    ):
        raise ValueError("projected Q/K descriptor shapes do not align")
    query = torch.nn.functional.normalize(query.float(), dim=-1)
    history_key = torch.nn.functional.normalize(
        history_key.float(), dim=-1
    )
    frame_similarity = torch.einsum(
        "hsp,hfsp->hfs", query, history_key
    ).mean(dim=-1)
    top_values = frame_similarity.topk(
        k=min(2, frame_similarity.shape[-1]), dim=-1
    ).values
    top1 = top_values[:, 0]
    margin = (
        top_values[:, 0] - top_values[:, 1]
        if top_values.shape[-1] > 1
        else top_values[:, 0]
    )
    peak = frame_similarity.argmax(dim=-1)
    return top1, margin, peak


def _cross_video_rows(
    profiles: list[dict],
    donor_map: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    records = {}
    profile_by_job = {}
    for profile in profiles:
        job_id = str(profile["job"]["job_id"])
        profile_by_job[job_id] = profile
        for record in profile["records"]:
            records[(job_id, _state_key(record))] = record

    rows = []
    missing = []
    for target_job, target_profile in profile_by_job.items():
        donors = donor_map[target_job]
        donor_jobs = [donors["hard"]] + list(donors["random"])
        for target_record in target_profile["records"]:
            state = _state_key(target_record)
            donor_records = []
            for donor_job in donor_jobs:
                donor = records.get((donor_job, state))
                if donor is None:
                    missing.append(
                        {
                            "target_job": target_job,
                            "donor_job": donor_job,
                            "state": repr(state),
                        }
                    )
                else:
                    donor_records.append((donor_job, donor))
            if len(donor_records) != len(donor_jobs):
                continue
            query = target_record["query_projection"]
            correct_top1, correct_margin, correct_peak = (
                _descriptor_similarity(
                    query, target_record["history_key_projection"]
                )
            )
            donor_scores = {}
            for donor_job, donor_record in donor_records:
                score, _, _ = _descriptor_similarity(
                    query, donor_record["history_key_projection"]
                )
                donor_scores[donor_job] = score
            hard_score = donor_scores[donors["hard"]]
            random_score = torch.stack(
                [donor_scores[job] for job in donors["random"]], dim=0
            ).max(dim=0).values
            wrong_score = torch.maximum(hard_score, random_score)
            frame_ids = target_record["history_frame_ids"].long()
            ages = int(target_record["current_frame"]) - frame_ids
            for head in range(EXPECTED_HEADS):
                rows.append(
                    {
                        "dataset_index": int(
                            target_profile["job"]["dataset_index"]
                        ),
                        "job_id": target_job,
                        "mode": str(target_record["mode"]),
                        "current_frame": int(
                            target_record["current_frame"]
                        ),
                        "nominal_timestep": int(
                            target_record["nominal_timestep"]
                        ),
                        "layer": int(target_record["layer"]),
                        "head": head,
                        "correct_top1": float(correct_top1[head]),
                        "correct_margin": float(correct_margin[head]),
                        "hard_wrong_top1": float(hard_score[head]),
                        "random_wrong_top1": float(random_score[head]),
                        "wrong_top1": float(wrong_score[head]),
                        "history_specificity": float(
                            correct_top1[head] - wrong_score[head]
                        ),
                        "hard_history_specificity": float(
                            correct_top1[head] - hard_score[head]
                        ),
                        "correct_peak_age": int(
                            ages[int(correct_peak[head])].item()
                        ),
                    }
                )
    return rows, missing


LOCAL_FIELDS = (
    "old_history_effect",
    "reverse_effect",
    "reverse_relative_log",
    "phase_shift_effect",
    "phase_shift_relative_log",
    "freeze_latest_effect",
    "freeze_latest_relative_log",
    "value_mismatch_effect",
    "value_mismatch_relative_log",
    "reverse_vs_phase_log",
)

CROSS_FIELDS = (
    "correct_top1",
    "correct_margin",
    "hard_wrong_top1",
    "random_wrong_top1",
    "wrong_top1",
    "history_specificity",
    "hard_history_specificity",
    "correct_peak_age",
)


def _merge_observation_rows(
    local_rows: list[dict],
    cross_rows: list[dict],
) -> list[dict]:
    key_fields = (
        "job_id",
        "mode",
        "current_frame",
        "nominal_timestep",
        "layer",
        "head",
    )
    local_map = {
        tuple(row[field] for field in key_fields): row for row in local_rows
    }
    cross_map = {
        tuple(row[field] for field in key_fields): row for row in cross_rows
    }
    if set(local_map) != set(cross_map):
        missing_local = sorted(set(cross_map) - set(local_map))
        missing_cross = sorted(set(local_map) - set(cross_map))
        raise ValueError(
            "local/cross observation grids differ: "
            f"missing_local={missing_local[:2]} "
            f"missing_cross={missing_cross[:2]}"
        )
    output = []
    for key in sorted(local_map):
        row = dict(local_map[key])
        for field, value in cross_map[key].items():
            if field not in row:
                row[field] = value
        output.append(row)
    return output


def _load_v136_axes(path: Path | None) -> dict[tuple[int, int], dict]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        result[key] = {
            field: float(row[field])
            for field in (
                "cphi_score",
                "age_js_score",
                "middle_recent_margin",
                "old_mass_excess",
                "temporal_reach_ratio",
            )
            if row.get(field, "") != ""
        }
    return result


def analyze(
    profiles: list[dict],
    *,
    output_dir: Path,
    expected_count: int,
    expected_states: int,
    recent_frames: int,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    v136_head_axes: Path | None,
) -> dict:
    contract = _audit_profiles(
        profiles,
        expected_count=expected_count,
        expected_states=expected_states,
    )
    donor_map, donor_audit = _build_donor_map(profiles)
    local_rows = _local_intervention_rows(profiles, recent_frames)
    cross_rows, missing_donors = _cross_video_rows(profiles, donor_map)
    if missing_donors:
        raise ValueError(
            "cross-video donor state mismatch: "
            + json.dumps(missing_donors[:3], sort_keys=True)
        )
    combined_rows = _merge_observation_rows(local_rows, cross_rows)
    local_head, local_job = COMMON._two_stage_aggregate(
        local_rows,
        keys=("layer", "head"),
        cluster_key="job_id",
        fields=LOCAL_FIELDS,
    )
    cross_head, cross_job = COMMON._two_stage_aggregate(
        cross_rows,
        keys=("layer", "head"),
        cluster_key="job_id",
        fields=CROSS_FIELDS,
    )
    local_map = {
        (int(row["layer"]), int(row["head"])): row for row in local_head
    }
    cross_map = {
        (int(row["layer"]), int(row["head"])): row for row in cross_head
    }
    bootstrap = COMMON._bootstrap_sign(
        combined_rows,
        cluster_key="job_id",
        fields=(
            "reverse_relative_log",
            "phase_shift_relative_log",
            "freeze_latest_relative_log",
            "value_mismatch_relative_log",
            "history_specificity",
        ),
        rounds=bootstrap_rounds,
        seed=bootstrap_seed,
    )
    v136 = _load_v136_axes(v136_head_axes)
    head_axes = []
    for layer in range(EXPECTED_LAYERS):
        for head in range(EXPECTED_HEADS):
            key = (layer, head)
            if key not in local_map or key not in cross_map:
                raise ValueError(f"incomplete v138 head grid at {key}")
            row = {"layer": layer, "head": head}
            for source in (local_map[key], cross_map[key]):
                for field, value in source.items():
                    if field not in {"layer", "head"}:
                        row[field] = value
            row.update(bootstrap.get(key, {}))
            for field, value in v136.get(key, {}).items():
                row[f"v136_{field}"] = value
            head_axes.append(row)

    diagnostic_fields = (
        "reverse_relative_log",
        "phase_shift_relative_log",
        "freeze_latest_relative_log",
        "value_mismatch_relative_log",
        "history_specificity",
        "hard_history_specificity",
        "correct_margin",
    )
    diagnostics, models = COMMON._axis_diagnostics(
        head_axes, diagnostic_fields
    )
    diagnostic_map = {row["axis"]: row for row in diagnostics}
    order_diag = diagnostic_map["reverse_relative_log"]
    order_threshold = float(order_diag["gmm2_threshold"])
    order_minority = min(
        int(order_diag["gmm2_high"]), int(order_diag["gmm2_low"])
    ) / len(head_axes)
    order_cluster_gate = (
        float(order_diag["bic1_minus_bic2"]) >= 10
        and float(order_diag["bic3_minus_bic2"]) >= 0
        and order_minority >= 0.10
    )
    for row in head_axes:
        row["history_specificity_label"] = (
            "self_history_specific"
            if float(row["history_specificity"]) > 0
            else "no_self_history_preference"
        )
        row["order_gmm_side"] = (
            "high_order_response"
            if float(row["reverse_relative_log"]) > order_threshold
            else "low_order_response"
        )
        row["order_label_admissible"] = int(order_cluster_gate)

    correlations = COMMON._axis_correlations(
        head_axes,
        diagnostic_fields
        + tuple(
            field
            for field in (
                "v136_cphi_score",
                "v136_age_js_score",
                "v136_middle_recent_margin",
                "v136_old_mass_excess",
                "v136_temporal_reach_ratio",
            )
            if any(field in row for row in head_axes)
        ),
    )
    local_split = COMMON._split_half_reproducibility(
        local_rows,
        cluster_key="job_id",
        fields=(
            "reverse_relative_log",
            "phase_shift_relative_log",
            "freeze_latest_relative_log",
            "value_mismatch_relative_log",
        ),
    )
    specificity_split = COMMON._split_half_reproducibility(
        cross_rows,
        cluster_key="job_id",
        fields=("history_specificity", "hard_history_specificity"),
    )
    timestep_axes, _ = COMMON._two_stage_aggregate(
        combined_rows,
        keys=("mode", "nominal_timestep", "layer", "head"),
        cluster_key="job_id",
        fields=LOCAL_FIELDS + CROSS_FIELDS,
    )
    ar_axes, _ = COMMON._two_stage_aggregate(
        combined_rows,
        keys=("mode", "current_frame", "layer", "head"),
        cluster_key="job_id",
        fields=LOCAL_FIELDS + CROSS_FIELDS,
    )
    specialization = COMMON._axis_specialization(
        timestep_axes,
        category_fields=("mode", "nominal_timestep"),
        axes=(
            "reverse_relative_log",
            "freeze_latest_relative_log",
            "history_specificity",
        ),
    )

    specificity_counts = {
        label: sum(
            row["history_specificity_label"] == label for row in head_axes
        )
        for label in (
            "self_history_specific",
            "no_self_history_preference",
        )
    }
    specificity_minority = min(specificity_counts.values()) / len(head_axes)
    specificity_reliable = COMMON._mean(
        row.get("history_specificity_bootstrap_confidence", 0.0) >= 0.80
        for row in head_axes
    )
    specificity_gate = (
        COMMON._median(row["history_specificity"] for row in head_axes) > 0
        and specificity_split["history_specificity"]["spearman"] >= 0.30
        and specificity_reliable >= 0.70
        and specificity_minority >= 0.10
    )
    order_reproducible = (
        local_split["reverse_relative_log"]["spearman"] >= 0.30
    )
    order_gate = order_cluster_gate and order_reproducible
    if specificity_gate and order_gate:
        recommendation = "history_specificity_plus_order_axis"
    elif specificity_gate:
        recommendation = "history_specificity_only"
    elif order_gate:
        recommendation = "order_axis_only"
    else:
        recommendation = "continuous_intervention_scores_only"

    report = {
        "method": "v138_history_intervention_and_cross_video_specificity",
        "profile_count": len(profiles),
        "head_count": len(head_axes),
        "profile_contract_passed": all(
            bool(row["passed"]) for row in contract
        ),
        "maximum_rope_reconstruction_error": max(
            row["max_rope_reconstruction_error"] for row in contract
        ),
        "observation_counts": {
            "local": len(local_rows),
            "cross_video": len(cross_rows),
        },
        "donor_policy": {
            "hard": "maximum lexical Jaccard excluding self",
            "random_offsets": [1, 37, 73],
            "same_seed_required": True,
        },
        "specificity_counts": specificity_counts,
        "specificity_reliable_fraction": specificity_reliable,
        "split_half": {
            "local": local_split,
            "specificity": specificity_split,
        },
        "gates": {
            "history_specificity": specificity_gate,
            "order_axis": order_gate,
            "order_gmm_structure": order_cluster_gate,
            "order_reproducibility": order_reproducible,
        },
        "recommendation": recommendation,
        "gmm_thresholds_are_diagnostic_until_gates_pass": True,
        "mixture_models": models,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    COMMON._write_csv(output_dir / "head_axes.csv", head_axes)
    COMMON._write_csv(output_dir / "head_local_job_axes.csv", local_job)
    COMMON._write_csv(output_dir / "head_cross_job_axes.csv", cross_job)
    COMMON._write_csv(output_dir / "head_timestep_axes.csv", timestep_axes)
    COMMON._write_csv(output_dir / "head_ar_axes.csv", ar_axes)
    COMMON._write_csv(
        output_dir / "head_timestep_specialization.csv", specialization
    )
    COMMON._write_csv(output_dir / "axis_diagnostics.csv", diagnostics)
    COMMON._write_csv(output_dir / "axis_correlations.csv", correlations)
    COMMON._write_csv(output_dir / "profile_contract_audit.csv", contract)
    COMMON._write_csv(output_dir / "donor_audit.csv", donor_audit)
    COMMON._write_json(output_dir / "analysis_report.json", report)

    summary = [
        "# v138 History Intervention Analysis",
        "",
        f"- Recommendation: `{recommendation}`",
        (
            "- Maximum RoPE reconstruction error: "
            f"`{report['maximum_rope_reconstruction_error']:.6g}`"
        ),
        f"- History-specificity gate: `{specificity_gate}`",
        f"- Order-axis gate: `{order_gate}`",
        "",
        "## Specificity",
        "",
        *[
            f"- `{label}`: {count}"
            for label, count in specificity_counts.items()
        ],
        (
            "- Split-half Spearman: "
            f"`{specificity_split['history_specificity']['spearman']:.4f}`"
        ),
        f"- Bootstrap-reliable fraction: `{specificity_reliable:.4f}`",
        "",
        "## Order Intervention",
        "",
        (
            "- Reverse split-half Spearman: "
            f"`{local_split['reverse_relative_log']['spearman']:.4f}`"
        ),
        (
            "- GMM BIC1-BIC2 / BIC3-BIC2: "
            f"`{order_diag['bic1_minus_bic2']:.4f}` / "
            f"`{order_diag['bic3_minus_bic2']:.4f}`"
        ),
        f"- GMM threshold: `{order_threshold:.6f}`",
        "",
        "## Evidence Boundary",
        "",
        "- Cross-video specificity compares self history with unrelated and "
        "lexically similar wrong trajectories; it is not yet an "
        "identity-versus-scene decomposition.",
        "- Reverse/phase/freeze interventions reposition cached layer "
        "features with corrected RoPE. They measure attention-level history "
        "sensitivity, not final-prediction causal utility.",
        "- A generation map must not be constructed unless its gate passes "
        "and grouped top/bottom/random/reversed causal controls are run.",
        "",
    ]
    (output_dir / "analysis_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=128)
    parser.add_argument("--expected-states", type=int, default=9)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--v136-head-axes", type=Path)
    args = parser.parse_args()
    report = analyze(
        _load_profiles(args.profile_dir),
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        expected_states=args.expected_states,
        recent_frames=args.recent_frames,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed,
        v136_head_axes=args.v136_head_axes,
    )
    print(
        "[v138] "
        f"heads={report['head_count']} "
        f"recommendation={report['recommendation']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
