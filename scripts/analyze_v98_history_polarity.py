#!/usr/bin/env python3
"""Create a paired, decision-oriented report for the v98 primary screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import string
from pathlib import Path
from typing import Any, Iterable

from prepare_blind_review import (
    SCORE_FIELDS as PREPARED_BLIND_SCORE_FIELDS,
    verify_frozen_package,
    verify_package,
)


PRIMARY_METHODS = (
    "sf_native",
    "pf_native",
    "pf_explicit_parity",
    "pf_aw_hybrid_merge",
    "history_polarity_hybrid_merge",
    "history_polarity_stride_merge",
    "history_polarity_zero_random_hybrid_merge",
    "positive_rate_half_hybrid_merge",
)
OPTIONAL_FOLLOWUP_METHODS = (
    "followup_history_polarity_hybrid_merge_base",
    "followup_history_polarity_hybrid_merge_v78",
)
COMPARISONS = (
    (
        "implementation_parity",
        "pf_native",
        "pf_explicit_parity",
        "The explicit PF route must reproduce native PF before interpretation.",
    ),
    (
        "classifier_gap_to_pf_oracle",
        "pf_aw_hybrid_merge",
        "history_polarity_hybrid_merge",
        "Isolate the cost of PF-independent head discovery.",
    ),
    (
        "hybrid_support_memory",
        "history_polarity_stride_merge",
        "history_polarity_hybrid_merge",
        "Test whether periodic plus sparse supportive history is useful.",
    ),
    (
        "polarity_vs_count_matched_random",
        "history_polarity_zero_random_hybrid_merge",
        "history_polarity_hybrid_merge",
        "Test whether the frozen head assignment carries signal beyond counts.",
    ),
    (
        "polarity_vs_sign_fraction",
        "positive_rate_half_hybrid_merge",
        "history_polarity_hybrid_merge",
        "Compare the primary statistic with majority-sign classification.",
    ),
    (
        "proposed_vs_pf",
        "pf_native",
        "history_polarity_hybrid_merge",
        "Measure the primary quality gap to the borrowed PF baseline.",
    ),
    (
        "proposed_vs_sf",
        "sf_native",
        "history_polarity_hybrid_merge",
        "Measure the primary gain over native Self-Forcing.",
    ),
)
COMPREHENSIVE_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "m5_temporal_flickering",
    "m8_max_long_range_sim",
    "composite",
)
VBENCH_METRICS = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
)
PAIRED_METRICS = (*COMPREHENSIVE_METRICS, "temporal_jump")
PARITY_TOLERANCES = {
    "m1_dino_consistency": 0.002,
    "m1_min_stability": 0.005,
    "m2_drift_slope": 0.0005,
    "m7_background_consistency": 0.002,
    "m5_temporal_flickering": 0.002,
    "m8_max_long_range_sim": 0.002,
    "composite": 0.002,
    "temporal_jump": 0.02,
    "vbench_subject_consistency": 0.005,
    "vbench_background_consistency": 0.005,
    "vbench_aesthetic_quality": 0.005,
    "vbench_imaging_quality": 0.005,
    "vbench_motion_smoothness": 0.005,
    "vbench_dynamic_degree": 0.005,
}
LOWER_IS_BETTER = {
    "m5_temporal_flickering",
    "m8_max_long_range_sim",
    "temporal_jump",
}
DESCRIPTIVE_METRICS = {"vbench_dynamic_degree"}
INDEXED_VIDEO_PATTERN = re.compile(r"^(\d+)-(\d+)_[^.]+\.mp4$")
BLIND_RATING_FIELDS = (
    "identity_1_to_5",
    "background_1_to_5",
    "motion_1_to_5",
    "camera_1_to_5",
    "artifact_1_to_5",
    "prompt_alignment_1_to_5",
    "long_range_drift_1_to_5",
    "repetition_looping_1_to_5",
)
BLIND_FLAG_FIELDS = (
    "startup_flashback_0_or_1",
    "abrupt_jump_0_or_1",
    "polygon_noise_0_or_1",
)
BLIND_SCORE_FIELDS = tuple(PREPARED_BLIND_SCORE_FIELDS)
REQUIRED_BLIND_USABLE_METHODS = (
    "pf_native",
    "pf_explicit_parity",
    "history_polarity_hybrid_merge",
)
FROZEN_BOOTSTRAP_SAMPLES = 2000
FROZEN_BOOTSTRAP_SEED = 20260727
FROZEN_METRIC_PARAMETERS = {
    "sample_frames": 64,
    "temporal_frame_step": 2,
    "vbench_dimensions": list(VBENCH_METRICS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--vbench", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--map-manifest", required=True, type=Path)
    parser.add_argument("--policy-audit", required=True, type=Path)
    parser.add_argument("--metric-manifest", required=True, type=Path)
    parser.add_argument("--experiment-contract", type=Path)
    parser.add_argument("--transition-summary", type=Path)
    parser.add_argument("--blind-scorecard", type=Path)
    parser.add_argument("--blind-key", type=Path)
    parser.add_argument("--blind-verification", type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=FROZEN_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=FROZEN_BOOTSTRAP_SEED,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _method_contracts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("experiment contract methods must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in methods:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("invalid method in experiment contract")
        name = item["name"]
        if name in result:
            raise ValueError(f"duplicate contract method {name!r}")
        result[name] = item
    return result


def load_experiment_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    methods = _method_contracts(payload)
    if int(payload.get("version", -1)) != 2:
        failures.append("contract version must be 2")
    if payload.get("experiment") != "v98_history_polarity":
        failures.append("unexpected experiment name")
    if payload.get("phase") != "primary":
        failures.append("analysis requires a primary-phase contract")
    if tuple(methods) != PRIMARY_METHODS:
        failures.append(
            f"primary method order/set mismatch: {list(methods)}"
        )
    if any(name in methods for name in OPTIONAL_FOLLOWUP_METHODS):
        failures.append("v78 follow-up leaked into the primary eight-cell screen")
    if int(payload.get("frames", -1)) != 120:
        failures.append("primary contract must use 120 frames")
    if int(payload.get("shards", -1)) != 4:
        failures.append("primary contract must use four shards")
    if not isinstance(payload.get("few_step_cfg_enabled"), bool):
        failures.append("few-step CFG state must be explicit")
    if payload.get("tracked_worktree_dirty") is not False:
        failures.append("contract was not frozen from a clean tracked worktree")
    if not payload.get("run_fingerprint") or not payload.get("run_commit"):
        failures.append("run fingerprint/commit is missing")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(payload.get("method_contract_sha256", ""))
    ):
        failures.append("method_contract_sha256 is invalid")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        failures.append("experiment contract input inventory is missing")
    else:
        for name, item in inputs.items():
            if not isinstance(item, dict):
                failures.append(f"contract input {name} is malformed")
                continue
            input_path = Path(str(item.get("path", "")))
            expected_hash = str(item.get("sha256", ""))
            if (
                not input_path.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or sha256(input_path) != expected_hash
            ):
                failures.append(
                    f"contract input {name} is missing or stale: {input_path}"
                )
    fingerprint_payload = dict(payload)
    declared_fingerprint = fingerprint_payload.pop(
        "run_fingerprint", None
    )
    actual_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if declared_fingerprint != actual_fingerprint:
        failures.append("run fingerprint does not match contract contents")
    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        failures.append("prompt contract is missing")
        prompt = {}
    try:
        prompt_count = int(prompt.get("count", -1))
    except (TypeError, ValueError):
        prompt_count = -1
    expected_count = {"screen32": 32, "main128": 128}.get(
        payload.get("mode")
    )
    if expected_count is None or prompt_count != expected_count:
        failures.append(
            f"mode/prompt count mismatch: mode={payload.get('mode')!r} "
            f"count={prompt_count}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(prompt.get("sha256", ""))):
        failures.append("prompt SHA256 is invalid")
    prompt_path = Path(str(prompt.get("path", "")))
    if (
        not prompt_path.is_file()
        or sha256(prompt_path) != prompt.get("sha256")
    ):
        failures.append("prompt path/hash is invalid")
    score = payload.get("score")
    if not isinstance(score, dict):
        failures.append("score contract is missing")
        score = {}
    for field in (
        "artifact_sha256",
        "csv_sha256",
        "map_manifest_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(score.get(field, ""))):
            failures.append(f"score.{field} is invalid")
    score_csv = Path(str(score.get("csv_path", "")))
    if (
        not score_csv.is_file()
        or sha256(score_csv) != score.get("csv_sha256")
    ):
        failures.append("score CSV path/hash is invalid")
    if score.get("artifact_accepted") is not True:
        failures.append("score artifact acceptance is false")
    if score.get("bootstrap_unit") != "counterfactual_prompt_pair":
        failures.append("score bootstrap unit is not prompt-paired")
    runtime = payload.get("runtime")
    policy_trace = (
        runtime.get("policy_trace")
        if isinstance(runtime, dict)
        else None
    )
    if not isinstance(policy_trace, dict):
        failures.append("runtime policy_trace contract is missing")
    else:
        try:
            layers = [int(value) for value in policy_trace["layers"]]
            if (
                not layers
                or layers != sorted(set(layers))
                or int(policy_trace["stride"]) <= 0
                or int(policy_trace["max_records"]) <= 0
            ):
                raise ValueError("invalid trace settings")
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"runtime policy_trace contract is invalid: {error}")

    random_cell = methods.get(
        "history_polarity_zero_random_hybrid_merge", {}
    )
    if random_cell.get("map_key") != "history_polarity_zero_random":
        failures.append("random-control method is not bound to its random map")
    for name, item in methods.items():
        for field in (
            "engine",
            "route",
            "map_key",
            "transition",
            "expected_labels",
            "policies",
        ):
            if field not in item:
                failures.append(f"{name}: missing contract field {field}")
        transition = item.get("transition")
        if (
            not isinstance(transition, dict)
            or transition.get("enabled") is not False
        ):
            failures.append(
                f"{name}: primary methods must disable transition writes"
            )
        if item.get("few_step_cfg_enabled") != payload.get(
            "few_step_cfg_enabled"
        ):
            failures.append(f"{name}: few-step CFG state differs from run")
        if name == "sf_native":
            if (
                item.get("engine") != "sf"
                or item.get("route") != "none"
                or item.get("map_key") is not None
            ):
                failures.append("sf_native engine/route/map binding is invalid")
        elif item.get("engine") != "pf":
            failures.append(f"{name}: expected PF engine")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "payload": payload,
        "methods": methods,
        "prompt_count": prompt_count,
        "failures": failures,
        "pass": not failures,
    }


def validate_artifact_provenance(
    contract: dict[str, Any],
    map_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    payload = contract["payload"]
    score = payload["score"]
    manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(map_manifest_path)
    if manifest_hash != score.get("map_manifest_sha256"):
        failures.append("map manifest hash differs from experiment contract")
    if manifest.get("score_csv_sha256") != score.get("csv_sha256"):
        failures.append("score CSV hash differs between map and run contracts")
    artifact_path = Path(str(score.get("artifact_path", "")))
    if (
        not artifact_path.is_file()
        or sha256(artifact_path) != score.get("artifact_sha256")
    ):
        failures.append("primary score artifact path/hash is invalid")
        score_artifact = {}
    else:
        score_artifact = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
        if (
            int(score_artifact.get("version", -1))
            != int(score.get("artifact_version", -2))
            or score_artifact.get("method") != score.get("artifact_method")
            or score_artifact.get("accepted") is not True
        ):
            failures.append("score artifact identity/acceptance mismatch")
        definition = score_artifact.get("score_definition", {})
        if (
            not isinstance(definition, dict)
            or definition.get("primary_field") != score.get("primary_field")
        ):
            failures.append("score artifact primary field mismatch")
        if (
            isinstance(definition, dict)
            and definition.get("bootstrap_unit")
            != score.get("bootstrap_unit")
        ):
            failures.append("score artifact bootstrap unit mismatch")
    if (
        manifest.get("score_artifact_sha256")
        != score.get("artifact_sha256")
    ):
        failures.append("map manifest uses a different score artifact")
    claims = manifest.get("claims", {})
    maps = manifest.get("maps", {})
    primary = claims.get("primary_classifier")
    if not isinstance(primary, str) or primary not in maps:
        failures.append("map manifest primary classifier is invalid")
    if claims.get("pf_labels_used_for_primary_classifier") is not False:
        failures.append("primary classifier is not declared PF-independent")
    if (
        int(manifest.get("support_label", -1)) != 10
        or int(manifest.get("suppress_label", -1)) != 11
        or set(manifest.get("reserved_pf_labels", [])) != {-1, 1, 2}
    ):
        failures.append("neutral/reserved label contract is invalid")
    natural = contract["methods"].get(
        "history_polarity_hybrid_merge", {}
    )
    if natural.get("map_key") != primary:
        failures.append(
            "primary run method is not bound to the manifest primary map"
        )
    random_item = maps.get("history_polarity_zero_random", {})
    if random_item.get("reference") != primary:
        failures.append("random control does not reference the primary map")

    descriptors = {
        key: maps.get(primary, {}).get(key)
        for key in (
            "score_column",
            "primary_score_column",
            "statistic",
            "score",
            "support_rule",
            "threshold",
            "threshold_provenance",
        )
        if key in maps.get(primary, {})
    }
    if not descriptors:
        failures.append("primary map has no generic score declaration")
    elif score.get("primary_field") not in {
        str(value)
        for value in descriptors.values()
        if isinstance(value, str)
    }:
        failures.append(
            "experiment primary field is not declared by the primary map"
        )
    return manifest, {
        "pass": not failures,
        "manifest_sha256": manifest_hash,
        "primary_classifier": primary,
        "primary_score": descriptors,
        "failures": failures,
    }


def _video_index(row: dict[str, Any]) -> tuple[int, int] | None:
    name = row.get("video_name")
    if not isinstance(name, str):
        path = row.get("video_path")
        name = Path(path).name if isinstance(path, str) else ""
    match = INDEXED_VIDEO_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def load_comprehensive(
    path: Path,
    *,
    required_methods: tuple[str, ...],
    prompt_count: int,
    allow_followup: bool,
) -> tuple[
    dict[str, dict[int, dict[str, float]]],
    dict[int, str],
    list[str],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_video = payload.get("per_video")
    per_method = payload.get("per_method")
    if not isinstance(per_video, dict) or not per_video:
        raise ValueError("comprehensive output has no per_video observations")
    if not isinstance(per_method, dict):
        raise ValueError("comprehensive output has no per_method object")
    allowed = set(required_methods)
    if allow_followup:
        allowed.update(OPTIONAL_FOLLOWUP_METHODS)
    extra = set(per_method) - allowed
    missing = set(required_methods) - set(per_method)
    if missing or extra:
        raise ValueError(
            f"comprehensive method mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )

    rows: dict[str, dict[int, dict[str, float]]] = {
        method: {} for method in per_method
    }
    prompts: dict[int, str] = {}
    failures: list[str] = []
    for key, item in per_video.items():
        if not isinstance(item, dict):
            failures.append(f"{key}: row is not an object")
            continue
        method = item.get("method")
        prompt_index = item.get("prompt_index")
        prompt = item.get("prompt")
        if method not in rows:
            failures.append(f"{key}: unknown method {method!r}")
            continue
        if (
            isinstance(prompt_index, bool)
            or not isinstance(prompt_index, int)
            or not 0 <= prompt_index < prompt_count
        ):
            failures.append(f"{key}: invalid prompt index {prompt_index!r}")
            continue
        if prompt_index in rows[method]:
            failures.append(
                f"{method}: duplicate prompt index {prompt_index}"
            )
            continue
        if not isinstance(prompt, str) or not prompt:
            failures.append(f"{key}: prompt text is missing")
            continue
        previous = prompts.get(prompt_index)
        if previous is not None and previous != prompt:
            failures.append(
                f"prompt {prompt_index}: text differs across methods"
            )
        prompts[prompt_index] = prompt
        filename_index = _video_index(item)
        if filename_index != (prompt_index, 0):
            failures.append(
                f"{key}: filename index {filename_index} does not bind "
                f"prompt {prompt_index}, sample 0"
            )
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            failures.append(f"{key}: metrics object is missing")
            continue
        metric_row: dict[str, float] = {}
        for metric in COMPREHENSIVE_METRICS:
            value = finite(metrics.get(metric))
            if value is None:
                failures.append(f"{key}: required metric {metric} is missing")
            else:
                metric_row[metric] = value
        rows[method][prompt_index] = metric_row

    expected_indices = set(range(prompt_count))
    for method, method_rows in rows.items():
        indices = set(method_rows)
        if indices != expected_indices:
            failures.append(
                f"{method}: prompt coverage missing="
                f"{sorted(expected_indices-indices)} "
                f"extra={sorted(indices-expected_indices)}"
            )
        summary = per_method[method]
        if int(summary.get("num_videos", -1)) != prompt_count:
            failures.append(
                f"{method}: aggregate num_videos is not {prompt_count}"
            )
        declared_indices = summary.get("prompt_indices")
        if declared_indices is not None and list(declared_indices) != list(
            range(prompt_count)
        ):
            failures.append(f"{method}: aggregate prompt_indices mismatch")
        for metric in COMPREHENSIVE_METRICS:
            values = [
                row[metric]
                for row in method_rows.values()
                if metric in row
            ]
            declared = finite(summary.get(metric))
            if len(values) == prompt_count and (
                declared is None
                or not math.isclose(
                    declared,
                    statistics.fmean(values),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                failures.append(
                    f"{method}: aggregate {metric} does not equal "
                    "per-video mean"
                )
    if set(prompts) != expected_indices:
        failures.append("bound prompt text does not cover the full prompt set")
    return rows, prompts, failures


def load_temporal(
    path: Path,
    *,
    methods: Iterable[str],
    prompt_count: int,
) -> tuple[dict[str, dict[int, float]], list[str]]:
    expected_methods = set(methods)
    rows = {method: {} for method in expected_methods}
    failures: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            method = row.get("method")
            try:
                prompt_index = int(row.get("prompt_index", ""))
            except ValueError:
                failures.append(
                    f"temporal line {line_number}: invalid prompt index"
                )
                continue
            if method not in rows:
                failures.append(
                    f"temporal line {line_number}: unknown method {method!r}"
                )
                continue
            if prompt_index in rows[method]:
                failures.append(
                    f"temporal {method}: duplicate prompt {prompt_index}"
                )
                continue
            value = finite(row.get("temporal_jump"))
            if value is None:
                failures.append(
                    f"temporal {method}/{prompt_index}: missing score"
                )
                continue
            match = INDEXED_VIDEO_PATTERN.fullmatch(
                Path(row.get("video", "")).name
            )
            if (
                match is None
                or int(match.group(1)) != prompt_index
                or int(match.group(2)) != 0
            ):
                failures.append(
                    f"temporal {method}/{prompt_index}: filename mismatch"
                )
            rows[method][prompt_index] = value
    expected_indices = set(range(prompt_count))
    for method, values in rows.items():
        if set(values) != expected_indices:
            failures.append(
                f"temporal {method}: prompt coverage missing="
                f"{sorted(expected_indices-set(values))} "
                f"extra={sorted(set(values)-expected_indices)}"
            )
    return rows, failures


def load_vbench(
    path: Path,
    *,
    required_methods: tuple[str, ...],
    allow_followup: bool,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, dict):
        raise ValueError("VBench summary has no methods object")
    allowed = set(required_methods)
    if allow_followup:
        allowed.update(OPTIONAL_FOLLOWUP_METHODS)
    missing = set(required_methods) - set(raw_methods)
    extra = set(raw_methods) - allowed
    failures = []
    if missing or extra:
        failures.append(
            f"VBench method mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    declared_dimensions = payload.get("dimensions")
    if not isinstance(declared_dimensions, list) or not set(
        VBENCH_METRICS
    ).issubset(declared_dimensions):
        failures.append("VBench required dimensions are not declared")
    if payload.get("missing"):
        failures.append(f"VBench reports missing values: {payload['missing']}")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not set(required_methods).issubset(
        sources
    ):
        failures.append("VBench source coverage is incomplete")
    result: dict[str, dict[str, float]] = {}
    for method, raw in raw_methods.items():
        if method not in allowed or not isinstance(raw, dict):
            continue
        row = {}
        for metric in VBENCH_METRICS:
            value = finite(raw.get(metric))
            if value is None:
                failures.append(f"VBench {method}/{metric} is missing")
            else:
                row[f"vbench_{metric}"] = value
        result[method] = row
    return result, failures


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of no values")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


def _direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    if metric in DESCRIPTIVE_METRICS:
        return "descriptive"
    return "higher_is_better"


def _paired_summary(
    values: list[float],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    mean = statistics.fmean(values)
    direction = _direction(metric)
    favorable = None
    if direction == "higher_is_better":
        favorable = mean > 0
    elif direction == "lower_is_better":
        favorable = mean < 0
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean_abs": statistics.fmean(abs(value) for value in values),
        "max_abs": max(abs(value) for value in values),
        "bootstrap_mean_ci95": _bootstrap_ci(
            values,
            samples=samples,
            seed=seed,
        ),
        "direction": direction,
        "candidate_mean_favorable": favorable,
    }


def build_comparison(
    *,
    name: str,
    baseline: str,
    candidate: str,
    purpose: str,
    comprehensive: dict[str, dict[int, dict[str, float]]],
    temporal: dict[str, dict[int, float]],
    vbench: dict[str, dict[str, float]],
    prompt_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for ordinal, metric in enumerate(PAIRED_METRICS):
        if metric == "temporal_jump":
            deltas = [
                temporal[candidate][index] - temporal[baseline][index]
                for index in range(prompt_count)
            ]
        else:
            deltas = [
                comprehensive[candidate][index][metric]
                - comprehensive[baseline][index][metric]
                for index in range(prompt_count)
            ]
        stable_seed = int.from_bytes(
            hashlib.sha256(
                f"{bootstrap_seed}:{name}:{metric}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        paired[metric] = _paired_summary(
            deltas,
            metric=metric,
            samples=bootstrap_samples,
            seed=stable_seed + ordinal,
        )
    aggregate = {
        metric: {
            "delta": vbench[candidate][metric] - vbench[baseline][metric],
            "direction": _direction(metric),
        }
        for metric in sorted(set(vbench[baseline]) & set(vbench[candidate]))
    }
    return {
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "purpose": purpose,
        "paired_deltas": paired,
        "aggregate_vbench_deltas": aggregate,
    }


def parity_metric_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for metric, tolerance in PARITY_TOLERANCES.items():
        if metric.startswith("vbench_"):
            row = comparison["aggregate_vbench_deltas"].get(metric)
            delta = None if row is None else row["delta"]
            metrics[metric] = {
                "kind": "aggregate",
                "tolerance": tolerance,
                "delta": delta,
                "pass": delta is not None and abs(delta) <= tolerance,
            }
        else:
            row = comparison["paired_deltas"].get(metric)
            maximum = None if row is None else row["max_abs"]
            metrics[metric] = {
                "kind": "max_abs_paired_delta",
                "tolerance": tolerance,
                "max_abs_delta": maximum,
                "pass": maximum is not None and maximum <= tolerance,
            }
    return {
        "pass": bool(metrics) and all(row["pass"] for row in metrics.values()),
        "metrics": metrics,
    }


def validate_policy_audit(
    payload: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    if payload.get("strict_pass") is not True:
        failures.append("policy audit strict_pass is false")
    if payload.get("experiment_contract_sha256") != contract["sha256"]:
        failures.append("policy audit used a different experiment contract")
    if tuple(payload.get("expected_methods", [])) != tuple(
        contract["methods"]
    ):
        failures.append("policy audit method coverage differs from contract")
    observed_parity = payload.get("pf_parity_observed_contract", {})
    if observed_parity.get("pass") is not True:
        failures.append("runtime native/explicit parity trace comparison failed")
    shards = payload.get("shards")
    if not isinstance(shards, list):
        failures.append("policy audit shard results are missing")
    else:
        for index, item in enumerate(shards):
            if not isinstance(item, dict):
                failures.append(f"policy audit shard {index} is malformed")
                continue
            trace_value = item.get("trace")
            expected_hash = item.get("trace_sha256")
            if trace_value is None and expected_hash is None:
                continue
            trace_path = Path(str(trace_value))
            if (
                not trace_path.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash))
                or sha256(trace_path) != expected_hash
            ):
                failures.append(
                    f"policy trace changed after audit: {trace_path}"
                )
    return {"pass": not failures, "failures": failures}


def validate_metric_manifest(
    path: Path,
    *,
    contract: dict[str, Any],
    blind_verification: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 2:
        failures.append("metric manifest version must be 2")
    fingerprint_payload = dict(payload)
    declared_fingerprint = fingerprint_payload.pop(
        "metric_input_fingerprint", None
    )
    actual_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if declared_fingerprint != actual_fingerprint:
        failures.append("metric manifest fingerprint is invalid")
    generation = payload.get("generation")
    if not isinstance(generation, dict):
        failures.append("metric manifest generation contract is missing")
        generation = {}
    if generation.get("experiment_contract_sha256") != contract["sha256"]:
        failures.append("metric manifest used a different experiment contract")
    if generation.get("run_fingerprint") != contract["payload"].get(
        "run_fingerprint"
    ):
        failures.append("metric manifest run fingerprint mismatch")
    if payload.get("parameters") != FROZEN_METRIC_PARAMETERS:
        failures.append(
            "metric parameters differ from the frozen v98 protocol: "
            f"{payload.get('parameters')!r}"
        )
    if payload.get("stages") != {
        "vbench": True,
        "comprehensive": True,
        "temporal": True,
        "analysis": True,
    }:
        failures.append("metric manifest does not enable all primary stages")

    video_inputs = payload.get("video_inputs")
    if not isinstance(video_inputs, list):
        failures.append("metric manifest video inputs are missing")
        video_inputs = []
    video_by_method = {
        str(item.get("method")): item
        for item in video_inputs
        if isinstance(item, dict)
    }
    if set(video_by_method) != set(contract["methods"]):
        failures.append("metric manifest method/video coverage mismatch")
    for method, item in video_by_method.items():
        if int(item.get("count", -1)) != contract["prompt_count"]:
            failures.append(f"{method}: metric video count mismatch")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(item.get("input_fingerprint", ""))
        ):
            failures.append(f"{method}: invalid metric video fingerprint")

    if blind_verification is not None and payload.get("blind") != (
        blind_verification
    ):
        failures.append("metric manifest blind verification is stale")
    vbench = payload.get("vbench")
    if not isinstance(vbench, dict):
        failures.append("metric manifest VBench provenance is missing")
        vbench = {}
    if not vbench.get("commit") or vbench.get("dirty") is not False:
        failures.append("VBench evaluator commit is missing or dirty")
    for key in (
        "status_sha256",
        "evaluator_sha256",
        "info_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(vbench.get(key, ""))):
            failures.append(f"VBench {key} is missing or invalid")
    evaluators = payload.get("evaluators")
    if not isinstance(evaluators, dict) or not evaluators:
        failures.append("metric evaluator hash inventory is missing")
    else:
        for raw_path, expected_hash in evaluators.items():
            evaluator = Path(str(raw_path))
            if (
                not evaluator.is_file()
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash))
                or sha256(evaluator) != expected_hash
            ):
                failures.append(
                    f"metric evaluator changed after freeze: {evaluator}"
                )
    return payload, {
        "pass": not failures,
        "metric_input_fingerprint": declared_fingerprint,
        "failures": failures,
    }


def load_transition_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = payload.get("summaries")
    failures = []
    if not isinstance(summaries, list) or not summaries:
        return {
            "pass": False,
            "failures": ["transition summary has no shard summaries"],
        }
    total = accepted = events = 0
    branches: set[str] = set()
    for index, item in enumerate(summaries):
        if not isinstance(item, dict):
            failures.append(f"transition summary {index} is invalid")
            continue
        if item.get("status") != "nominal" or item.get("failures"):
            failures.append(f"transition summary {index} is not nominal")
        events += int(item.get("events", 0))
        total += int(item.get("total", 0))
        accepted += int(item.get("accepted", 0))
        branches.update(dict(item.get("branches", {})))
    if events <= 0 or total <= 0:
        failures.append("transition trace has no decisions")
    if not 0 < accepted < total:
        failures.append(
            "transition write gate did not show both acceptance and rejection"
        )
    if not branches or not branches.issubset({"cond", "uncond"}):
        failures.append(
            f"transition trace branches are invalid: {sorted(branches)}"
        )
    return {
        "pass": not failures,
        "events": events,
        "accepted": accepted,
        "total": total,
        "acceptance_rate": accepted / total if total else None,
        "branches": sorted(branches),
        "failures": failures,
    }


def _bounded_number(
    value: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    number = finite(value)
    if number is None or not minimum <= number <= maximum:
        return None
    return number


def load_blind_review(
    scorecard_path: Path,
    key_path: Path,
    *,
    run_root: Path,
    prompts_path: Path,
    verification_path: Path,
    methods: tuple[str, ...],
    prompt_count: int,
) -> dict[str, Any]:
    failures: list[str] = []
    freeze_failures: list[str] = []
    public_output = scorecard_path.resolve().parent
    private_output = key_path.resolve().parent
    completion_path = private_output / ".complete.json"
    frozen_path = private_output / "FROZEN.json"
    live_verification: dict[str, Any] | None = None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        package_verification = verify_package(
            run_root=run_root,
            methods=list(methods),
            prompts=prompts_path,
            output=public_output,
            private_output=private_output,
            prompt_count=prompt_count,
            seed=None,
        )
        live_verification = verify_frozen_package(
            output=public_output,
            private_output=private_output,
            prompt_count=prompt_count,
            method_count=len(methods),
            verification=package_verification,
        )
        persisted_verification = json.loads(
            verification_path.read_text(encoding="utf-8")
        )
        if persisted_verification != live_verification:
            freeze_failures.append(
                "persisted blind verification differs from live package"
            )
        freeze_contract = {
            "completion version": (
                2,
                completion.get("version"),
            ),
            "completion methods": (
                list(methods),
                completion.get("methods"),
            ),
            "completion prompt_count": (
                prompt_count,
                completion.get("prompt_count"),
            ),
            "completion candidate_count": (
                prompt_count * len(methods),
                completion.get("candidate_count"),
            ),
            "completion scorecard schema": (
                list(BLIND_SCORE_FIELDS),
                completion.get("scorecard_fields"),
            ),
            "completion key hash": (
                completion.get("key_private_sha256"),
                sha256(key_path),
            ),
            "completion public output": (
                str(public_output),
                completion.get("public_output"),
            ),
            "completion private output": (
                str(private_output),
                completion.get("private_output"),
            ),
            "freeze version": (1, frozen.get("version")),
            "freeze completion hash": (
                sha256(completion_path),
                frozen.get("completion_sha256"),
            ),
            "freeze source fingerprint": (
                completion.get("source_fingerprint"),
                frozen.get("source_fingerprint"),
            ),
            "freeze candidate_count": (
                prompt_count * len(methods),
                frozen.get("candidate_count"),
            ),
            "freeze scorecard hash": (
                sha256(scorecard_path),
                frozen.get("scorecard_sha256"),
            ),
            "freeze scorecard rows": (
                prompt_count * len(methods),
                frozen.get("rows"),
            ),
        }
        freeze_failures.extend(
            f"{name} mismatch: expected={expected!r} actual={actual!r}"
            for name, (expected, actual) in freeze_contract.items()
            if expected != actual
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        freeze_failures.append(f"blind freeze validation failed: {error}")
    failures.extend(freeze_failures)

    key_payload = json.loads(key_path.read_text(encoding="utf-8"))
    mapping: dict[tuple[int, str], str] = {}
    for item in key_payload.get("items", []):
        prompt_index = int(item.get("prompt_index", -1))
        for candidate in item.get("candidates", []):
            pair = (prompt_index, str(candidate.get("label", "")))
            method = candidate.get("method")
            if pair in mapping or method not in methods:
                failures.append(f"invalid blind key entry {pair}/{method!r}")
            else:
                mapping[pair] = method

    for prompt_index in range(prompt_count):
        prompt_entries = [
            (label, method)
            for (candidate_prompt, label), method in mapping.items()
            if candidate_prompt == prompt_index
        ]
        prompt_methods = [method for _, method in prompt_entries]
        if (
            len(prompt_methods) != len(methods)
            or set(prompt_methods) != set(methods)
        ):
            failures.append(
                f"prompt {prompt_index}: blind key methods are not a "
                f"permutation of {list(methods)}"
            )
        expected_labels = set(string.ascii_uppercase[: len(methods)])
        if {label for label, _ in prompt_entries} != expected_labels:
            failures.append(
                f"prompt {prompt_index}: blind labels are not the exact "
                f"set {sorted(expected_labels)}"
            )

    rating_fields = BLIND_RATING_FIELDS
    flag_fields = BLIND_FLAG_FIELDS
    method_rows: dict[str, list[dict[str, float]]] = {
        method: [] for method in methods
    }
    seen: set[tuple[int, str]] = set()
    ranks: dict[int, set[int]] = {}
    with scorecard_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BLIND_SCORE_FIELDS:
            failures.append("scorecard schema does not match frozen v98 fields")
        for line_number, row in enumerate(reader, start=2):
            try:
                prompt_index = int(row.get("prompt_index", ""))
            except ValueError:
                failures.append(f"scorecard line {line_number}: bad prompt")
                continue
            pair = (prompt_index, str(row.get("label", "")))
            method = mapping.get(pair)
            if method is None or pair in seen:
                failures.append(
                    f"scorecard line {line_number}: unmapped/duplicate {pair}"
                )
                continue
            seen.add(pair)
            parsed: dict[str, float] = {}
            for field in rating_fields:
                value = _bounded_number(
                    str(row.get(field, "")), minimum=1, maximum=5
                )
                if value is None or not value.is_integer():
                    failures.append(
                        f"scorecard line {line_number}: incomplete {field}"
                    )
                else:
                    parsed[field] = value
            for field in flag_fields:
                value = _bounded_number(
                    str(row.get(field, "")), minimum=0, maximum=1
                )
                if value is None or value not in {0.0, 1.0}:
                    failures.append(
                        f"scorecard line {line_number}: incomplete {field}"
                    )
                else:
                    parsed[field] = value
            rank = _bounded_number(
                str(row.get("overall_rank", "")),
                minimum=1,
                maximum=len(methods),
            )
            if rank is None or not rank.is_integer():
                failures.append(
                    f"scorecard line {line_number}: invalid overall_rank"
                )
            else:
                rank_int = int(rank)
                parsed["overall_rank"] = rank
                if rank_int in ranks.setdefault(prompt_index, set()):
                    failures.append(
                        f"prompt {prompt_index}: duplicate rank {rank_int}"
                    )
                ranks[prompt_index].add(rank_int)
            failure_time = str(row.get("failure_time_seconds", "")).strip()
            if failure_time:
                parsed_time = finite(failure_time)
                if parsed_time is None or parsed_time < 0:
                    failures.append(
                        f"scorecard line {line_number}: invalid failure time"
                    )
                else:
                    parsed["failure_time_seconds"] = parsed_time
            method_rows[method].append(parsed)

    expected_pairs = {
        (prompt_index, label)
        for prompt_index in range(prompt_count)
        for label in {
            pair_label
            for pair_prompt, pair_label in mapping
            if pair_prompt == prompt_index
        }
    }
    if len(mapping) != prompt_count * len(methods):
        failures.append("blind key does not cover every prompt/method")
    if seen != expected_pairs or len(seen) != prompt_count * len(methods):
        failures.append("scorecard does not cover every blind candidate")
    for prompt_index in range(prompt_count):
        if ranks.get(prompt_index) != set(range(1, len(methods) + 1)):
            failures.append(
                f"prompt {prompt_index}: ranks are not a complete permutation"
            )

    summaries = {}
    for method, rows in method_rows.items():
        complete_rows = [
            row
            for row in rows
            if all(
                field in row
                for field in (*rating_fields, *flag_fields, "overall_rank")
            )
        ]
        catastrophic = sum(
            row["startup_flashback_0_or_1"] > 0
            or row["abrupt_jump_0_or_1"] > 0
            or row["polygon_noise_0_or_1"] > 0
            or row["identity_1_to_5"] <= 2
            or row["artifact_1_to_5"] <= 2
            or row["long_range_drift_1_to_5"] <= 2
            or row["repetition_looping_1_to_5"] <= 2
            for row in complete_rows
        )
        summaries[method] = {
            "n": len(complete_rows),
            "means": {
                field: (
                    statistics.fmean(row[field] for row in complete_rows)
                    if complete_rows
                    else None
                )
                for field in (*rating_fields, "overall_rank")
            },
            "catastrophic_artifact_rows": catastrophic,
            "catastrophic_rows": catastrophic,
            "usable": len(complete_rows) == prompt_count and catastrophic == 0,
        }
    return {
        "pass": not failures,
        "scorecard": str(scorecard_path.resolve()),
        "key": str(key_path.resolve()),
        "verification": str(verification_path.resolve()),
        "live_verification": live_verification,
        "frozen_verified": not freeze_failures,
        "methods": summaries,
        "failures": sorted(set(failures)),
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    gates = payload["gates"]
    lines = [
        "# v98 History-Polarity Paired Decision Report",
        "",
        "## Hard integrity gates",
        "",
    ]
    for name, value in gates.items():
        if isinstance(value, dict) and "pass" in value:
            lines.append(f"- {name}: `{value['pass']}`")
    lines.extend(
        [
            "",
            f"- overall hard-gate pass: `{payload['hard_gate_pass']}`",
            f"- paired prompts per primary method: "
            f"`{payload['sample_contract']['prompt_count']}`",
            "",
            "## Method means",
            "",
        ]
    )
    metric_names = sorted(
        {
            metric
            for row in payload["metrics"].values()
            for metric in row
        }
    )
    lines.extend(
        [
            "| Method | " + " | ".join(metric_names) + " |",
            "|---|" + "|".join("---:" for _ in metric_names) + "|",
        ]
    )
    for method in payload["method_order"]:
        row = payload["metrics"][method]
        values = [
            "n/a" if metric not in row else f"{row[metric]:.6f}"
            for metric in metric_names
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")

    lines.extend(["", "## Controlled paired comparisons", ""])
    for comparison in payload["comparisons"]:
        lines.extend(
            [
                f"### {comparison['name']}",
                "",
                comparison["purpose"],
                "",
                f"- Baseline: `{comparison['baseline']}`",
                f"- Candidate: `{comparison['candidate']}`",
                "",
                "| Metric | n | Mean delta | 95% bootstrap CI | Direction |",
                "|---|---:|---:|---|---|",
            ]
        )
        for metric, row in comparison["paired_deltas"].items():
            lines.append(
                f"| {metric} | {row['n']} | {row['mean']:.6f} | "
                f"[{row['bootstrap_mean_ci95'][0]:.6f}, "
                f"{row['bootstrap_mean_ci95'][1]:.6f}] | "
                f"{row['direction']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "No automatic winner is selected. Integrity gates must pass first; "
            "then catastrophic visual failures exclude individual cells, and "
            "the paired deltas plus blind review determine whether the "
            "PF-independent primary method is competitive.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    comprehensive_path: Path,
    vbench_path: Path,
    temporal_path: Path,
    map_manifest_path: Path,
    policy_audit_path: Path,
    metric_manifest_path: Path,
    experiment_contract_path: Path,
    transition_summary_path: Path | None,
    blind_scorecard_path: Path | None,
    blind_key_path: Path | None,
    blind_verification_path: Path | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    contract = load_experiment_contract(experiment_contract_path)
    manifest, artifact_gate = validate_artifact_provenance(
        contract,
        map_manifest_path,
    )
    allow_followup = transition_summary_path is not None
    comprehensive, prompts, comprehensive_failures = load_comprehensive(
        comprehensive_path,
        required_methods=tuple(contract["methods"]),
        prompt_count=contract["prompt_count"],
        allow_followup=allow_followup,
    )
    temporal_methods = tuple(comprehensive)
    temporal, temporal_failures = load_temporal(
        temporal_path,
        methods=temporal_methods,
        prompt_count=contract["prompt_count"],
    )
    vbench, vbench_failures = load_vbench(
        vbench_path,
        required_methods=tuple(contract["methods"]),
        allow_followup=allow_followup,
    )
    optional_complete = all(
        method in comprehensive
        and method in temporal
        and method in vbench
        for method in OPTIONAL_FOLLOWUP_METHODS
    )
    optional_partial = any(
        method in comprehensive
        or method in temporal
        or method in vbench
        for method in OPTIONAL_FOLLOWUP_METHODS
    ) and not optional_complete
    sample_failures = [
        *comprehensive_failures,
        *temporal_failures,
        *vbench_failures,
    ]
    prompt_path = Path(
        str(contract["payload"].get("prompt", {}).get("path", ""))
    )
    frozen_prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(frozen_prompts) != contract["prompt_count"]:
        sample_failures.append("frozen prompt file count changed")
    else:
        for prompt_index, expected_prompt in enumerate(frozen_prompts):
            if prompts.get(prompt_index) != expected_prompt:
                sample_failures.append(
                    f"prompt {prompt_index}: comprehensive text differs "
                    "from the frozen prompt file"
                )
    if optional_partial:
        sample_failures.append("optional v78 follow-up metrics are incomplete")

    method_order = list(contract["methods"])
    if optional_complete:
        method_order.extend(OPTIONAL_FOLLOWUP_METHODS)
    metrics: dict[str, dict[str, float]] = {}
    for method in method_order:
        row = {
            metric: statistics.fmean(
                comprehensive[method][index][metric]
                for index in range(contract["prompt_count"])
            )
            for metric in COMPREHENSIVE_METRICS
        }
        row.update(vbench[method])
        row["temporal_jump"] = statistics.fmean(
            temporal[method].values()
        )
        metrics[method] = row

    comparisons = [
        build_comparison(
            name=name,
            baseline=baseline,
            candidate=candidate,
            purpose=purpose,
            comprehensive=comprehensive,
            temporal=temporal,
            vbench=vbench,
            prompt_count=contract["prompt_count"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for name, baseline, candidate, purpose in COMPARISONS
    ]
    if optional_complete:
        comparisons.append(
            build_comparison(
                name="trusted_write_admission_followup",
                baseline=OPTIONAL_FOLLOWUP_METHODS[0],
                candidate=OPTIONAL_FOLLOWUP_METHODS[1],
                purpose=(
                    "Optional paired follow-up; it is not part of the "
                    "primary eight-cell screen."
                ),
                comprehensive=comprehensive,
                temporal=temporal,
                vbench=vbench,
                prompt_count=contract["prompt_count"],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    parity_gate = parity_metric_gate(comparisons[0])
    policy_payload = json.loads(policy_audit_path.read_text(encoding="utf-8"))
    policy_gate = validate_policy_audit(policy_payload, contract=contract)

    transition_gate: dict[str, Any]
    if transition_summary_path is None:
        transition_gate = {
            "pass": True,
            "status": "not_supplied_optional_followup",
        }
    else:
        transition_gate = load_transition_summary(transition_summary_path)
        transition_gate["status"] = "validated_optional_followup"

    blind_gate: dict[str, Any]
    if blind_scorecard_path is None:
        blind_gate = {
            "pass": False,
            "status": "missing_required_frozen_human_review",
        }
    else:
        if blind_verification_path is None:
            raise ValueError(
                "blind verification is required with a blind scorecard"
            )
        key_path = (
            blind_key_path
            if blind_key_path is not None
            else blind_scorecard_path.parent.with_name(
                f"{blind_scorecard_path.parent.name}_private"
            )
            / "key_private.json"
        )
        blind_gate = load_blind_review(
            blind_scorecard_path,
            key_path,
            run_root=experiment_contract_path.resolve().parent,
            prompts_path=prompt_path,
            verification_path=blind_verification_path,
            methods=tuple(contract["methods"]),
            prompt_count=contract["prompt_count"],
        )
        unusable_required = [
            method
            for method in REQUIRED_BLIND_USABLE_METHODS
            if not bool(
                blind_gate.get("methods", {})
                .get(method, {})
                .get("usable", False)
            )
        ]
        blind_gate["required_usable_methods"] = list(
            REQUIRED_BLIND_USABLE_METHODS
        )
        if unusable_required:
            blind_gate["pass"] = False
            blind_gate.setdefault("failures", []).append(
                "required proposed methods failed blind usability: "
                f"{unusable_required}"
            )
        blind_gate["status"] = (
            "validated_frozen_scorecard"
            if blind_gate.get("frozen_verified") is True
            else "invalid_or_stale_blind_package"
        )

    metric_payload, metric_gate = validate_metric_manifest(
        metric_manifest_path,
        contract=contract,
        blind_verification=(
            blind_gate.get("live_verification")
            if isinstance(blind_gate, dict)
            else None
        ),
    )

    sample_gate = {
        "pass": not sample_failures,
        "prompt_count": contract["prompt_count"],
        "methods": method_order,
        "paired_observations": {
            method: len(comprehensive[method])
            for method in method_order
        },
        "failures": sorted(set(sample_failures)),
    }
    required_metric_gate = {
        "pass": all(
            all(metric in metrics[method] for metric in PARITY_TOLERANCES)
            for method in contract["methods"]
        ),
        "required_metrics": sorted(PARITY_TOLERANCES),
    }
    gates = {
        "experiment_contract": {
            "pass": contract["pass"],
            "sha256": contract["sha256"],
            "failures": contract["failures"],
        },
        "artifact_provenance": artifact_gate,
        "sample_completeness": sample_gate,
        "required_metrics": required_metric_gate,
        "metric_manifest": metric_gate,
        "policy_trace_audit": policy_gate,
        "pf_parity_metrics": parity_gate,
        "transition_summary": transition_gate,
        "blind_scorecard": blind_gate,
    }
    hard_gate_pass = all(
        value.get("pass") is True
        for value in gates.values()
    )
    input_artifacts = {
        "comprehensive": {
            "path": str(comprehensive_path.resolve()),
            "sha256": sha256(comprehensive_path),
        },
        "vbench": {
            "path": str(vbench_path.resolve()),
            "sha256": sha256(vbench_path),
        },
        "temporal_jump": {
            "path": str(temporal_path.resolve()),
            "sha256": sha256(temporal_path),
        },
        "map_manifest": {
            "path": str(map_manifest_path.resolve()),
            "sha256": sha256(map_manifest_path),
        },
        "policy_audit": {
            "path": str(policy_audit_path.resolve()),
            "sha256": sha256(policy_audit_path),
        },
        "metric_manifest": {
            "path": str(metric_manifest_path.resolve()),
            "sha256": sha256(metric_manifest_path),
        },
    }
    trace_index = 0
    for item in policy_payload.get("shards", []):
        if not isinstance(item, dict) or not item.get("trace"):
            continue
        trace_path = Path(str(item["trace"]))
        expected_hash = str(item.get("trace_sha256", ""))
        if not trace_path.is_file() or sha256(trace_path) != expected_hash:
            raise ValueError(
                f"policy trace changed while assembling analysis: {trace_path}"
            )
        input_artifacts[f"policy_trace_{trace_index:03d}"] = {
            "path": str(trace_path.resolve()),
            "sha256": expected_hash,
        }
        trace_index += 1
    for name, input_path in (
        ("transition_summary", transition_summary_path),
        ("blind_scorecard", blind_scorecard_path),
        ("blind_verification", blind_verification_path),
    ):
        if input_path is not None:
            input_artifacts[name] = {
                "path": str(input_path.resolve()),
                "sha256": sha256(input_path),
            }
    if blind_scorecard_path is not None:
        input_artifacts["blind_key"] = {
            "path": str(key_path.resolve()),
            "sha256": sha256(key_path),
        }
    return {
        "version": 2,
        "method": "v98_history_polarity_paired_decision_analysis",
        "experiment_contract": {
            "path": contract["path"],
            "sha256": contract["sha256"],
            "mode": contract["payload"].get("mode"),
            "run_fingerprint": contract["payload"].get("run_fingerprint"),
        },
        "input_artifacts": input_artifacts,
        "map_contract": {
            "path": str(map_manifest_path.resolve()),
            "primary_classifier": artifact_gate["primary_classifier"],
            "primary_score": artifact_gate["primary_score"],
        },
        "method_order": method_order,
        "sample_contract": {
            "prompt_count": contract["prompt_count"],
            "prompt_indices": sorted(prompts),
            "paired_primary_methods": list(contract["methods"]),
        },
        "metrics": metrics,
        "bootstrap_protocol": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "paired_prompt",
        },
        "metric_protocol": metric_payload.get("parameters"),
        "comparisons": comparisons,
        "gates": gates,
        "hard_gate_pass": hard_gate_pass,
        "selection": {
            "automatic_winner": None,
            "reason": (
                "blind artifact review remains a hard method-usability gate"
                if blind_scorecard_path is None
                else "select only among blind-review-usable methods"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if (
        args.bootstrap_samples != FROZEN_BOOTSTRAP_SAMPLES
        or args.bootstrap_seed != FROZEN_BOOTSTRAP_SEED
    ):
        raise ValueError(
            "v98 analysis bootstrap is frozen at "
            f"samples={FROZEN_BOOTSTRAP_SAMPLES}, "
            f"seed={FROZEN_BOOTSTRAP_SEED}"
        )
    experiment_contract = (
        args.experiment_contract
        if args.experiment_contract is not None
        else args.map_manifest.parent.parent / "experiment_contract.json"
    )
    payload = analyze(
        comprehensive_path=args.comprehensive,
        vbench_path=args.vbench,
        temporal_path=args.temporal_jump,
        map_manifest_path=args.map_manifest,
        policy_audit_path=args.policy_audit,
        metric_manifest_path=args.metric_manifest,
        experiment_contract_path=experiment_contract,
        transition_summary_path=args.transition_summary,
        blind_scorecard_path=args.blind_scorecard,
        blind_key_path=args.blind_key,
        blind_verification_path=args.blind_verification,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(
        "[V98Analysis] "
        f"hard_gate_pass={payload['hard_gate_pass']} "
        f"paired_prompts={payload['sample_contract']['prompt_count']} "
        f"bootstrap_samples={args.bootstrap_samples}",
        flush=True,
    )
    if not payload["hard_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
