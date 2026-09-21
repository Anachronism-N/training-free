#!/usr/bin/env python3
"""Paired development-only decision for the frozen v210 LPHC screen8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import io
import math
import os
from pathlib import Path

import numpy as np

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as paired
import analyze_v190_head_phase_causal_screen as temporal
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v210_lphc import METHODS, SCREEN_FRAMES, SOURCE_INDICES, effective_seed
from prepare_v210_vbench_comparison import DIMENSIONS, EXPERIMENT
from vbench_quality_contract import quality_score_with_fixed_dynamic


PROMPT_COUNT = len(SOURCE_INDICES)
CLIPS_PER_VIDEO = SCREEN_FRAMES // 8
WINDOWS = {
    "full": (0, CLIPS_PER_VIDEO),
    "early_half": (0, CLIPS_PER_VIDEO // 2),
    "late_half": (CLIPS_PER_VIDEO // 2, CLIPS_PER_VIDEO),
}
PRIMARY_CONTROLS = ("sf_fifo21", "sf_sink1_21")
CAPACITY_CONTEXT = "sf_fifo25"
CANDIDATES = tuple(method for method in METHODS if method.startswith("lphc_"))
PRIMARY_METRICS = (
    "quality_without_dynamic_degree",
    "identity_background",
    "temporal_mechanics",
    "semantic_alignment",
    "visual_quality",
)
ANALYSIS_METRICS = (
    "quality_without_dynamic_degree",
    *paired.METRICS,
    "subject_consistency",
)
NONINFERIORITY_MARGINS = {
    "quality_without_dynamic_degree": -0.15,
    "identity_background": -0.0015,
    "temporal_mechanics": -0.0030,
    "semantic_alignment": -0.0030,
    "visual_quality": -0.0040,
}
POSITIVE_MEAN_THRESHOLDS = {
    "quality_without_dynamic_degree": 0.05,
    "identity_background": 0.0003,
    "temporal_mechanics": 0.0005,
    "semantic_alignment": 0.0010,
    "visual_quality": 0.0010,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def generation_seconds(method_row: dict) -> float:
    """Return the frozen aggregate generation time used as the third tie-breaker."""
    for key in ("generation_seconds", "elapsed_seconds", "total_elapsed_seconds"):
        if key in method_row:
            value = _finite(method_row[key], name=f"{method_row.get('key')}:{key}")
            if value < 0.0:
                raise ValueError(f"{method_row.get('key')}:{key} is negative")
            return value
    generation = method_row.get("generation")
    if isinstance(generation, dict):
        for key in ("seconds", "elapsed_seconds", "total_elapsed_seconds"):
            if key in generation:
                value = _finite(
                    generation[key], name=f"{method_row.get('key')}:generation:{key}"
                )
                if value < 0.0:
                    raise ValueError(
                        f"{method_row.get('key')}:generation:{key} is negative"
                    )
                return value
    jobs = method_row.get("jobs")
    if isinstance(jobs, list) and len(jobs) == PROMPT_COUNT:
        values = [
            _finite(row.get("elapsed_seconds"), name=f"{method_row.get('key')}:job")
            for row in jobs
        ]
        if any(value < 0.0 for value in values):
            raise ValueError(f"{method_row.get('key')}: negative job duration")
        return float(sum(values))
    raise ValueError(f"{method_row.get('key')}: generation time is absent")


def verify_manifest_provenance(manifest: dict) -> bool:
    """Recheck frozen source artifacts so selection fails closed after materialization."""
    source = manifest.get("source") or {}
    for stem in ("generation_manifest", "gate0_decision", "smoke_decision"):
        path = Path(str(source.get(stem, "")))
        digest = str(source.get(f"{stem}_sha256", ""))
        if not path.is_file() or len(digest) != 64 or sha256(path) != digest:
            return False
    try:
        generation = json.loads(Path(source["generation_manifest"]).read_text(encoding="utf-8"))
        gate0 = json.loads(Path(source["gate0_decision"]).read_text(encoding="utf-8"))
        smoke = json.loads(Path(source["smoke_decision"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        generation.get("source_commit") != manifest.get("source_generation_commit")
        or gate0.get("pass") is not True
        or smoke.get("pass") is not True
    ):
        return False

    jobs = manifest.get("jobs") or ()
    expected = {
        (method, prompt, source_index)
        for method in METHODS
        for prompt, source_index in enumerate(SOURCE_INDICES)
    }
    observed = set()
    for row in jobs:
        try:
            key = (
                str(row["method"]),
                int(row["prompt_index"]),
                int(row["source_index"]),
            )
            observed.add(key)
            if int(row["effective_seed"]) != effective_seed(key[2]):
                return False
            for path_key in ("done", "media"):
                path = Path(str(row[path_key]))
                if not path.is_file() or sha256(path) != row[f"{path_key}_sha256"]:
                    return False
            if key[0] in CANDIDATES:
                trace_path = Path(str(row["trace"]))
                trace_audit = row.get("trace_audit") or {}
                if (
                    not trace_path.is_file()
                    or sha256(trace_path) != row.get("trace_sha256")
                    or trace_audit.get("pass") is not True
                    or int((trace_audit.get("totals") or {}).get("random", -1)) != 0
                ):
                    return False
            elif row.get("trace") is not None or row.get("trace_sha256") is not None:
                return False
        except (KeyError, TypeError, ValueError, OSError):
            return False
    return observed == expected and len(jobs) == len(expected)


def manifest_generation_seconds(manifest: dict, method: str) -> float:
    method_row = next(
        (row for row in manifest.get("methods") or () if row.get("key") == method),
        None,
    )
    if method_row is None:
        raise ValueError(f"{method}: method provenance is absent")
    try:
        return generation_seconds(method_row)
    except ValueError:
        job_rows = [
            row for row in manifest.get("jobs") or () if row.get("method") == method
        ]
        if len(job_rows) != PROMPT_COUNT:
            raise ValueError(f"{method}: generation job timing is incomplete")
        values = []
        for row in job_rows:
            done_path = Path(str(row.get("done", "")))
            try:
                done = json.loads(done_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"{method}: cannot read generation timing") from error
            value = _finite(done.get("elapsed_seconds"), name=f"{method}:elapsed_seconds")
            if value < 0.0:
                raise ValueError(f"{method}: generation time is negative")
            values.append(value)
        return float(sum(values))


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _split_provenance_valid(manifest: dict, provenance: object) -> bool:
    if not isinstance(provenance, dict):
        return False
    methods = provenance.get("methods")
    if (
        provenance.get("version") != 1
        or provenance.get("method_count") != len(METHODS)
        or provenance.get("source_count") != len(METHODS) * PROMPT_COUNT
        or provenance.get("clip_count") != len(METHODS) * PROMPT_COUNT * CLIPS_PER_VIDEO
        or not _valid_sha256(provenance.get("aggregate_digest"))
        or not isinstance(methods, dict)
        or tuple(methods) != tuple(str(row["key"]) for row in manifest["methods"])
    ):
        return False
    for method in METHODS:
        row = methods.get(method)
        if not isinstance(row, dict):
            return False
        path = Path(str(row.get("provenance", "")))
        if (
            not path.is_file()
            or not _valid_sha256(row.get("provenance_sha256"))
            or sha256(path) != row["provenance_sha256"]
            or not _valid_sha256(row.get("source_digest"))
            or not _valid_sha256(row.get("clip_digest"))
        ):
            return False
    campaign_path_value = provenance.get("campaign_provenance")
    campaign_sha = provenance.get("campaign_provenance_sha256")
    if campaign_path_value is None and campaign_sha is None:
        return True
    campaign_path = Path(str(campaign_path_value or ""))
    return bool(
        campaign_path.is_file()
        and _valid_sha256(campaign_sha)
        and sha256(campaign_path) == campaign_sha
    )


def runtime_provenance_checks(manifest: dict, summary: dict) -> dict[str, bool]:
    provenance = summary.get("runtime_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    evaluation_source = provenance.get("evaluation_source")
    if not isinstance(evaluation_source, dict):
        evaluation_source = {}
    source_media_audit = provenance.get("source_media_audit")
    if not isinstance(source_media_audit, dict):
        source_media_audit = {}
    return {
        "evaluation_runtime_provenance": bool(
            evaluation_source.get("evaluation_commit")
            == manifest.get("evaluation_commit")
            and evaluation_source.get("runtime_file_sha256")
            == manifest.get("evaluation_runtime_sha256")
        ),
        "vbench_runtime_provenance": bool(
            provenance.get("vbench_checkout_fingerprint")
            == manifest.get("vbench_checkout_fingerprint")
        ),
        "source_media_audit": bool(source_media_audit.get("video_count") == 64),
        "split_provenance": _split_provenance_valid(
            manifest, provenance.get("split_provenance")
        ),
    }


def validate_manifest_summary(manifest: dict, summary: dict) -> dict:
    methods = tuple(str(row.get("key", "")) for row in manifest.get("methods") or ())
    prompt_items = manifest.get("prompt_items") or ()
    checks = {
        "experiment": manifest.get("experiment") == EXPERIMENT,
        "development_only": manifest.get("development_only") is True,
        "method_order": methods == METHODS,
        "primary_controls": (
            manifest.get("primary_baseline") == PRIMARY_CONTROLS[0]
            and manifest.get("equal_budget_control") == PRIMARY_CONTROLS[1]
            and manifest.get("capacity_context") == CAPACITY_CONTEXT
        ),
        "prompt_count": int(manifest.get("prompt_count", -1)) == PROMPT_COUNT,
        "frame_count": int(manifest.get("num_output_frames", -1)) == SCREEN_FRAMES,
        "dense_prompt_indices": [int(row.get("index", -1)) for row in prompt_items]
        == list(range(PROMPT_COUNT)),
        "source_indices": [int(row.get("source_index", -1)) for row in prompt_items]
        == list(SOURCE_INDICES),
        "effective_seeds": [int(row.get("effective_seed", -1)) for row in prompt_items]
        == [effective_seed(index) for index in SOURCE_INDICES],
        "dimensions": tuple(manifest.get("vbench_long_dimensions") or ()) == DIMENSIONS,
        "summary_experiment": summary.get("experiment") == EXPERIMENT,
        "summary_methods": tuple(summary.get("methods") or ()) == METHODS,
        "summary_dimensions": tuple(summary.get("dimensions") or ()) == DIMENSIONS,
        "summary_complete": not bool(summary.get("missing")),
        "source_provenance": verify_manifest_provenance(manifest),
        "read_only_materialization": manifest.get("materialization")
        == {"mode": "read_only_symlink", "generation_root_modified": False},
        "claim_boundary": bool(manifest.get("claim_boundary")),
        **runtime_provenance_checks(manifest, summary),
    }
    method_rows = manifest.get("methods") or ()
    try:
        checks["generation_times"] = (
            len(method_rows) == len(METHODS)
            and all(manifest_generation_seconds(manifest, method) >= 0.0 for method in METHODS)
        )
    except (TypeError, ValueError):
        checks["generation_times"] = False
    checks["pass"] = all(checks.values())
    if not checks["pass"]:
        failed = sorted(key for key, value in checks.items() if key != "pass" and not value)
        raise ValueError(f"incomplete/mixed v210 analysis inputs: {', '.join(failed)}")
    return checks


def verify_vbench_provenance(
    manifest_path: Path, parts_root: Path, summary: dict
) -> bool:
    try:
        manifest_digest = sha256(manifest_path)
    except OSError:
        return False
    if (
        Path(str(summary.get("comparison_manifest", ""))).resolve()
        != manifest_path.resolve()
        or summary.get("comparison_manifest_sha256") != manifest_digest
    ):
        return False
    sources = summary.get("sources") or {}
    for method in METHODS:
        if set(sources.get(method) or {}) != set(DIMENSIONS):
            return False
        for dimension in DIMENSIONS:
            result = parts_root / method / dimension / "results.json"
            marker_path = result.with_name("done.json")
            contract_path = result.with_name("job_contract.json")
            mapping_path = result.with_name("prompt_mapping.json")
            if Path(str(sources[method][dimension])).resolve() != result.resolve():
                return False
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if (
                not result.is_file()
                or not mapping_path.is_file()
                or marker.get("method") != method
                or marker.get("dimension") != dimension
                or marker.get("result_sha256") != sha256(result)
                or marker.get("job_contract_sha256") != sha256(contract_path)
                or marker.get("prompt_mapping_sha256") != sha256(mapping_path)
                or contract.get("comparison_manifest_sha256") != manifest_digest
            ):
                return False
    return True


def load_window_rows(
    parts_root: Path,
    summary: dict,
    methods: tuple[str, ...] = METHODS,
    *,
    prompt_count: int = PROMPT_COUNT,
    include_raw: bool = False,
    clips_per_video: int = CLIPS_PER_VIDEO,
) -> dict[str, dict]:
    if not isinstance(clips_per_video, int) or clips_per_video < 2:
        raise ValueError("at least two clips per video required for early/late analysis")
    windows = {"full": (0, clips_per_video), "early_half": (0, clips_per_video // 2),
               "late_half": (clips_per_video // 2, clips_per_video)}
    raw_by_window = {
        window: {
            (method, prompt): {} for method in methods for prompt in range(prompt_count)
        }
        for window in windows
    }
    for method in methods:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=prompt_count,
                clips_per_video=clips_per_video,
            )
            flattened = [value for prompt in range(prompt_count) for value in clips[prompt]]
            summary_value = detail.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = detail.scale_factor(
                float(np.mean(flattened)), summary_value, name=f"{method}:{dimension}"
            )
            for window, (start, end) in windows.items():
                for prompt in range(prompt_count):
                    raw_by_window[window][(method, prompt)][dimension] = factor * float(
                        np.mean(clips[prompt][start:end])
                    )
    result = {}
    for window, raw_rows in raw_by_window.items():
        derived = paired.derived_rows(raw_rows, methods, prompt_count)
        for key, row in derived.items():
            if include_raw:
                row.update(raw_rows[key])
            row["quality_without_dynamic_degree"] = quality_score_with_fixed_dynamic(
                raw_rows[key], dynamic_value=1.0
            )
            row["subject_consistency"] = float(raw_rows[key]["subject_consistency"])
        result[window] = derived
    return result


def contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    metric: str,
    window: str,
    prompt_count: int,
    seed: int,
) -> dict:
    deltas = np.asarray(
        [
            rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
            for prompt in range(prompt_count)
        ],
        dtype=np.float64,
    )
    return {
        "comparison": f"{candidate}_minus_{control}",
        "candidate": candidate,
        "control": control,
        "window": window,
        "metric": metric,
        "prompt_count": prompt_count,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "win_fraction": float(np.mean(deltas > 0.0)),
        "tie_fraction": float(np.mean(deltas == 0.0)),
        "bootstrap_ci95": paired.bootstrap_ci(deltas.tolist(), seed=seed),
        "p_value": paired.sign_p(deltas.tolist()),
        "per_prompt_delta": deltas.tolist(),
    }


def comparison(
    rows: list[dict], candidate: str, control: str, metric: str, window: str
) -> dict:
    matches = [
        row
        for row in rows
        if row["candidate"] == candidate
        and row["control"] == control
        and row["metric"] == metric
        and row["window"] == window
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing v210 comparison {candidate}/{control}/{metric}/{window}"
        )
    return matches[0]


def noninferiority(rows: list[dict], candidate: str, control: str) -> dict:
    windows = {}
    for window in ("full", "late_half"):
        windows[window] = {}
        for metric, margin in NONINFERIORITY_MARGINS.items():
            row = comparison(rows, candidate, control, metric, window)
            lower = float(row["bootstrap_ci95"][0])
            windows[window][metric] = {
                "margin": margin,
                "ci95_lower": lower,
                "pass": lower >= margin,
            }
    return {
        "candidate": candidate,
        "control": control,
        "windows": windows,
        "pass": all(item["pass"] for values in windows.values() for item in values.values()),
        "development_tolerance_only": True,
    }


def positive_support(rows: list[dict], candidate: str, control: str) -> dict:
    axes = []
    for window in ("full", "late_half"):
        for metric, threshold in POSITIVE_MEAN_THRESHOLDS.items():
            row = comparison(rows, candidate, control, metric, window)
            if float(row["mean_delta"]) >= threshold:
                axes.append(
                    {
                        "window": window,
                        "metric": metric,
                        "mean_delta": row["mean_delta"],
                        "threshold": threshold,
                    }
                )
    return {
        "candidate": candidate,
        "control": control,
        "directional_axes": axes,
        "pass": bool(axes),
        "interval_or_significance_required": False,
    }


def method_means(
    rows_by_window: dict[str, dict], methods: tuple[str, ...], prompt_count: int
) -> dict:
    return {
        window: {
            method: {
                metric: float(
                    np.mean([rows[(method, prompt)][metric] for prompt in range(prompt_count)])
                )
                for metric in ANALYSIS_METRICS
            }
            for method in methods
        }
        for window, rows in rows_by_window.items()
    }


def analyze_from_rows(
    manifest: dict,
    rows_by_window: dict[str, dict],
    temporal_rows: dict,
    *,
    provenance_verified: bool = True,
    provenance_checks: dict | None = None,
    split_provenance: dict | None = None,
) -> dict:
    methods = tuple(str(row.get("key", "")) for row in manifest.get("methods") or ())
    if methods != METHODS:
        raise ValueError("v210 analysis requires the frozen eight-method order")
    prompt_count = int(manifest.get("prompt_count", PROMPT_COUNT))
    if prompt_count != PROMPT_COUNT or tuple(rows_by_window) != tuple(WINDOWS):
        raise ValueError("v210 analysis requires the frozen screen8 window grid")
    expected_keys = {(method, prompt) for method in methods for prompt in range(prompt_count)}
    for window, rows in rows_by_window.items():
        if set(rows) != expected_keys:
            raise ValueError(f"v210 {window} prompt grid is incomplete or mixed")
        for key in expected_keys:
            missing = set(ANALYSIS_METRICS) - set(rows[key])
            if missing:
                raise ValueError(f"v210 {window}/{key} lacks metrics: {sorted(missing)}")
    if set(temporal_rows) != expected_keys:
        raise ValueError("v210 temporal prompt grid is incomplete or mixed")

    controls = (*PRIMARY_CONTROLS, CAPACITY_CONTEXT)
    comparisons = []
    for window_index, (window, rows) in enumerate(rows_by_window.items()):
        for candidate_index, candidate in enumerate(CANDIDATES):
            for control_index, control in enumerate(controls):
                for metric_index, metric in enumerate(ANALYSIS_METRICS):
                    comparisons.append(
                        contrast(
                            rows,
                            candidate=candidate,
                            control=control,
                            metric=metric,
                            window=window,
                            prompt_count=prompt_count,
                            seed=(
                                2100000
                                + window_index * 10000
                                + candidate_index * 1000
                                + control_index * 100
                                + metric_index
                            ),
                        )
                    )
    development_primary = [
        row
        for row in comparisons
        if row["control"] in PRIMARY_CONTROLS
        and row["window"] in {"full", "late_half"}
        and row["metric"] in PRIMARY_METRICS
    ]
    paired.bh(development_primary)
    primary_ids = {id(row) for row in development_primary}
    for row in comparisons:
        if id(row) in primary_ids:
            row["inferential_role"] = "development_screen_descriptive"
        else:
            row["q_value"] = None
            row["inferential_role"] = (
                "capacity_context"
                if row["control"] == CAPACITY_CONTEXT
                else "development_window_context"
            )

    times = {method: manifest_generation_seconds(manifest, method) for method in methods}
    means = method_means(rows_by_window, methods, prompt_count)
    dynamic = temporal.dynamic_metric_validity(
        rows_by_window["full"], methods=methods, prompt_count=prompt_count
    )
    statuses = {}
    eligible = []
    for candidate in CANDIDATES:
        noninferiority_by_control = {
            control: noninferiority(comparisons, candidate, control)
            for control in PRIMARY_CONTROLS
        }
        positive_by_control = {
            control: positive_support(comparisons, candidate, control)
            for control in PRIMARY_CONTROLS
        }
        temporal_guards = {
            control: temporal.temporal_guard(
                temporal_rows,
                candidate=candidate,
                control=control,
                prompt_count=prompt_count,
            )
            for control in PRIMARY_CONTROLS
        }
        provenance_pass = bool(provenance_verified)
        dual_noninferiority_pass = all(
            row["pass"] for row in noninferiority_by_control.values()
        )
        dual_positive_pass = all(row["pass"] for row in positive_by_control.values())
        temporal_safety_pass = all(
            row["automatic_safety_pass"] for row in temporal_guards.values()
        )
        candidate_eligible = bool(
            provenance_pass
            and dual_noninferiority_pass
            and dual_positive_pass
            and temporal_safety_pass
        )
        if candidate_eligible:
            eligible.append(candidate)
        statuses[candidate] = {
            "provenance_pass": provenance_pass,
            "noninferiority_by_control": noninferiority_by_control,
            "dual_sf21_noninferiority_pass": dual_noninferiority_pass,
            "positive_support_by_control": positive_by_control,
            "dual_sf21_directional_positive_pass": dual_positive_pass,
            "temporal_guards": temporal_guards,
            "temporal_safety_pass": temporal_safety_pass,
            "eligible": candidate_eligible,
        }

    ranked = sorted(
        eligible,
        key=lambda method: (
            -means["full"][method]["quality_without_dynamic_degree"],
            -means["full"][method]["subject_consistency"],
            times[method],
            method,
        ),
    )
    selected = ranked[0] if ranked else None
    ranking = [
        {
            "rank": index,
            "method": method,
            "quality_without_dynamic_degree": means["full"][method][
                "quality_without_dynamic_degree"
            ],
            "subject_consistency": means["full"][method]["subject_consistency"],
            "generation_seconds": times[method],
        }
        for index, method in enumerate(ranked, start=1)
    ]
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "development_only": True,
        "paper_claim_ready": False,
        "prompt_count": prompt_count,
        "windows": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "methods": list(methods),
        "primary_controls": list(PRIMARY_CONTROLS),
        "capacity_context": CAPACITY_CONTEXT,
        "candidates": list(CANDIDATES),
        "method_means": means,
        "generation_seconds": times,
        "comparisons": comparisons,
        "candidate_status": statuses,
        "eligible_candidates": eligible,
        "selection_ranking": ranking,
        "selected_for_matched_random_extension": selected,
        "selection_count": int(selected is not None),
        "recommendation": (
            "advance_selected_lphc_to_matched_random_extension"
            if selected is not None
            else "stop_v210_no_eligible_lphc_candidate"
        ),
        "provenance_gate": {
            "pass": bool(provenance_verified),
            "checks": provenance_checks or {},
            "split_provenance": split_provenance,
        },
        "metric_validity": {
            "dynamic_degree": dynamic,
            "dynamic_degree_used_for_selection": False,
            "dynamic_degree_used_for_noninferiority": False,
            "dynamic_degree_used_for_positive_support": False,
            "primary_quality_metric": "quality_without_dynamic_degree",
        },
        "development_statistics_only": True,
        "significance_used_for_selection": False,
        "manual_review_required_for_decision": False,
        "claim_boundary": (
            "The eight-prompt screen is a development prefilter only. Bootstrap "
            "intervals, sign-test p-values, and BH q-values are descriptive and do "
            "not support a paper claim. Dynamic Degree and the sf_fifo25 capacity "
            "context do not affect selection. At most one method may advance to a "
            "separate matched-random extension."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v210 LPHC Development Screen8",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Selected for matched-random extension: `{report['selected_for_matched_random_extension']}`",
        f"- Provenance gate: `{report['provenance_gate']['pass']}`",
        "- Dynamic Degree used for selection: `False`",
        "- Significance used for selection: `False`",
        "- Paper claim ready: `False`",
        "",
        "## Provenance checks",
        "",
    ]
    checks = report["provenance_gate"].get("checks") or {}
    lines.extend(f"- `{name}`: `{bool(value)}`" for name, value in sorted(checks.items()))
    lines.extend(
        [
            "",
            "| Candidate | Provenance | Dual SF21 NI | Dual SF21 positive | Temporal safe | Eligible |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in report["candidates"]:
        row = report["candidate_status"][candidate]
        lines.append(
            f"| {candidate} | {row['provenance_pass']} | "
            f"{row['dual_sf21_noninferiority_pass']} | "
            f"{row['dual_sf21_directional_positive_pass']} | "
            f"{row['temporal_safety_pass']} | {row['eligible']} |"
        )
    lines.extend(
        [
            "",
            "| Rank | Candidate | Full quality w/o DD | Full subject consistency | Generation seconds |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in report["selection_ranking"]:
        lines.append(
            f"| {row['rank']} | {row['method']} | "
            f"{row['quality_without_dynamic_degree']:.6f} | "
            f"{row['subject_consistency']:.6f} | {row['generation_seconds']:.3f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def write_frozen(path: Path, payload: bytes) -> None:
    """Create an immutable artifact, accepting only a byte-identical rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen v210 analysis artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_comparisons(path: Path, rows: list[dict]) -> None:
    fields = (
        "candidate",
        "control",
        "window",
        "prompt_count",
        "metric",
        "mean_delta",
        "median_delta",
        "win_fraction",
        "tie_fraction",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "p_value",
        "q_value",
        "inferential_role",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **{field: row.get(field) for field in fields},
                "bootstrap_ci_lower": row["bootstrap_ci95"][0],
                "bootstrap_ci_upper": row["bootstrap_ci95"][1],
            }
        )
    write_frozen(path, handle.getvalue().encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.screen_root.resolve()
    manifest_path = root / "vbench_comparison" / "comparison_manifest.json"
    summary_path = root / "metrics" / "vbench_core9_summary.json"
    parts_root = root / "metrics" / "vbench_long_parts"
    temporal_csv = root / "metrics" / "temporal_diagnostics.csv"
    temporal_contract = root / "metrics" / "temporal_diagnostics.contract.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = validate_manifest_summary(manifest, summary)
    checks["vbench_part_provenance"] = verify_vbench_provenance(
        manifest_path, parts_root, summary
    )
    checks["pass"] = bool(checks["pass"] and checks["vbench_part_provenance"])
    if not checks["pass"]:
        raise ValueError("v210 VBench part provenance drifted")
    verify_temporal_contract(temporal_contract, manifest_path, temporal_csv)
    checks["temporal_contract"] = True
    rows_by_window = load_window_rows(parts_root, summary)
    temporal_rows = temporal.load_temporal_rows(
        temporal_csv, methods=METHODS, prompt_count=PROMPT_COUNT
    )
    report = analyze_from_rows(
        manifest,
        rows_by_window,
        temporal_rows,
        provenance_verified=True,
        provenance_checks=checks,
        split_provenance=summary["runtime_provenance"]["split_provenance"],
    )
    report["source"] = {
        "comparison_manifest": str(manifest_path),
        "comparison_manifest_sha256": sha256(manifest_path),
        "vbench_summary": str(summary_path),
        "vbench_summary_sha256": sha256(summary_path),
        "temporal_diagnostics": str(temporal_csv),
        "temporal_diagnostics_sha256": sha256(temporal_csv),
        "temporal_contract": str(temporal_contract),
        "temporal_contract_sha256": sha256(temporal_contract),
    }
    output_dir = root / "analysis"
    output = output_dir / "v210_lphc_screen.json"
    write_frozen(
        output,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    write_frozen(output.with_suffix(".md"), render(report).encode("utf-8"))
    write_comparisons(output_dir / "v210_lphc_screen_comparisons.csv", report["comparisons"])
    print(
        "[v210-analysis] "
        f"recommendation={report['recommendation']} "
        f"selected={report['selected_for_matched_random_extension']}"
    )


if __name__ == "__main__":
    main()
