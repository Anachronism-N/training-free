#!/usr/bin/env python3
"""Analyze no-refit transfer of the frozen Head x Phase route to Causal-Forcing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import analyze_v174_paired_metrics as base
import numpy as np
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from analyze_v190_head_phase_causal_screen import (
    dynamic_metric_validity,
    load_temporal_rows,
    temporal_guard,
)
from bind_temporal_diagnostics import verify_contract as verify_temporal_contract
from prepare_v191_vbench_comparison import DIMENSIONS
from prepare_v194_cf_checkpoint_transfer import (
    CANDIDATE,
    LOCAL_CONTROL,
    METHODS,
    NATIVE_CONTROL,
    PROMPT_COUNT,
)
from prepare_v194_cf_checkpoint_transfer import (
    verify as verify_input,
)
from prepare_v194_vbench_comparison import EXPERIMENT

PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
    "temporal_mechanics",
)
NONINFERIORITY_MARGINS = {
    LOCAL_CONTROL: {
        "official_quality_score": -0.15,
        "identity_background": -0.001,
        "dynamic_degree": -0.02,
        "temporal_mechanics": -0.002,
    },
    NATIVE_CONTROL: {
        "official_quality_score": -0.20,
        "identity_background": -0.0015,
        "dynamic_degree": -0.03,
        "temporal_mechanics": -0.003,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    role: str,
    metric: str,
    seed: int,
) -> dict:
    deltas = [
        rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
        for prompt in range(PROMPT_COUNT)
    ]
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "comparison": f"{candidate}_minus_{control}",
        "comparison_role": role,
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "window": "full",
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "tie_fraction": float(np.mean(values == 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
    }


def by_metric(comparisons: list[dict], control: str) -> dict[str, dict]:
    return {
        row["metric"]: row
        for row in comparisons
        if row["candidate"] == CANDIDATE and row["control"] == control
    }


def method_means(rows: dict) -> dict:
    return {
        method: {
            metric: float(
                np.mean(
                    [rows[(method, prompt)][metric] for prompt in range(PROMPT_COUNT)]
                )
            )
            for metric in base.METRICS
        }
        for method in METHODS
    }


def noninferiority_gate(
    comparisons: dict[str, dict],
    *,
    control: str,
    dynamic_validity: dict,
    means: dict,
) -> dict:
    margins = NONINFERIORITY_MARGINS[control]
    passes = {
        metric: float(comparisons[metric]["bootstrap_ci95"][0]) >= margin
        for metric, margin in margins.items()
        if metric != "dynamic_degree"
    }
    if dynamic_validity["informative"]:
        dynamic_pass = (
            float(comparisons["dynamic_degree"]["bootstrap_ci95"][0])
            >= margins["dynamic_degree"]
        )
        dynamic_rule = "paired_ci_lower_ge_margin"
    elif dynamic_validity["ceiling_nonregression_only"]:
        dynamic_pass = bool(
            means[CANDIDATE]["dynamic_degree"] >= 1.0 - 1e-12
            and means[control]["dynamic_degree"] >= 1.0 - 1e-12
        )
        dynamic_rule = "all_one_ceiling_nonregression"
    else:
        dynamic_pass = False
        dynamic_rule = "constant_non_ceiling_metric_rejected"
    passes["dynamic_degree"] = dynamic_pass
    return {
        "control": control,
        "margins": margins,
        "metric_pass": passes,
        "dynamic_rule": dynamic_rule,
        "pass": all(passes.values()),
    }


def _same_prompt_transfer(
    frozen: dict,
    comparisons: list[dict],
    *,
    eligible_targets: list[str],
) -> dict:
    reference = frozen["same_prompt_sf_reference"]
    path = Path(reference["v191_decision"])
    if not path.is_file() or sha256(path) != reference["v191_decision_sha256"]:
        raise ValueError("v194 same-prompt Self-Forcing reference drifted")
    v191 = json.loads(path.read_text(encoding="utf-8"))
    positions = [int(row["v191_prompt_index"]) for row in frozen["prompt_items"]]
    rows = {}
    for metric_index, metric in enumerate(eligible_targets):
        matches = [
            row
            for row in v191.get("comparisons") or ()
            if row.get("candidate") == "head_phase_joint"
            and row.get("control") == "all_recent"
            and row.get("metric") == metric
            and row.get("window", "full") == "full"
        ]
        if len(matches) != 1:
            raise ValueError(f"v191 same-prompt reference is missing metric {metric}")
        full_sf = np.asarray(matches[0].get("per_prompt_delta") or (), dtype=np.float64)
        if full_sf.shape != (128,):
            raise ValueError(f"v191 per-prompt effect is incomplete: {metric}")
        sf = full_sf[positions]
        cf_row = by_metric(comparisons, LOCAL_CONTROL)[metric]
        cf = np.asarray(cf_row["per_prompt_delta"], dtype=np.float64)
        pooled = 0.5 * (sf + cf)
        interval = base.bootstrap_ci(
            pooled.tolist(),
            seed=1949000 + metric_index,
        )
        sign_agreement = float(np.mean(np.sign(sf) == np.sign(cf)))
        rows[metric] = {
            "sf_checkpoint_mean_delta": float(sf.mean()),
            "cf_checkpoint_mean_delta": float(cf.mean()),
            "two_checkpoint_prompt_average_delta": float(pooled.mean()),
            "bootstrap_ci95_over_prompts": interval,
            "per_prompt_effect_sign_agreement": sign_agreement,
            "both_checkpoint_means_positive": bool(sf.mean() > 0 and cf.mean() > 0),
            "pooled_ci_lower_gt_zero": float(interval[0]) > 0.0,
            "transfer_supported": bool(
                sf.mean() > 0 and cf.mean() > 0 and float(interval[0]) > 0.0
            ),
        }
    return {
        "prompt_pairing": "same prompt text, source index, and seed across checkpoints",
        "metrics": rows,
        "pass": any(row["transfer_supported"] for row in rows.values()),
    }


def _camera_context(path: Path | None, comparison_manifest: Path) -> dict:
    if path is None:
        return {
            "available": False,
            "reason": "v193 camera-compensated motion report not supplied",
            "motion_improvement_claim_supported": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source") or {}
    if (
        payload.get("experiment") != "v193_camera_compensated_motion_calibration"
        or payload.get("candidate") != CANDIDATE
        or tuple(payload.get("controls") or ()) != (LOCAL_CONTROL, NATIVE_CONTROL)
        or source.get("comparison_manifest_sha256") != sha256(comparison_manifest)
    ):
        raise ValueError("v194 received a mismatched v193 camera-motion report")
    supported = bool(
        payload.get("measurement_calibration_pass") is True
        and payload.get("directional_local_motion_signal_against_all_controls") is True
        and (payload.get("quality_context") or {}).get("all_controls_noninferior")
        is True
    )
    return {
        "available": True,
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "measurement_calibration_pass": payload["measurement_calibration_pass"],
        "directional_against_all_controls": payload[
            "directional_local_motion_signal_against_all_controls"
        ],
        "strong_against_all_controls": payload[
            "strong_local_motion_signal_against_all_controls"
        ],
        "motion_improvement_claim_supported": supported,
        "recommendation": payload["recommendation"],
    }


def _review_queue(
    manifest: dict,
    rows: dict,
    temporal_rows: dict,
    guards: tuple[dict, dict],
    *,
    limit: int = 4,
) -> list[dict]:
    flags = {}
    for guard in guards:
        for row in guard.get("flagged_prompts", ()):
            flags.setdefault(int(row["prompt_index"]), set()).update(row["flags"])
    video_dirs = {
        str(row["key"]): Path(row["video_dir"]) for row in manifest["methods"]
    }
    ranked = []
    for prompt in range(PROMPT_COUNT):
        deltas = {
            metric: rows[(CANDIDATE, prompt)][metric]
            - rows[(LOCAL_CONTROL, prompt)][metric]
            for metric in PRIMARY_METRICS
        }
        current = temporal_rows[(CANDIDATE, prompt)]
        recent = temporal_rows[(LOCAL_CONTROL, prompt)]
        temporal_disagreement = abs(
            math.log(
                (current["flow_speed_median"] + 1e-8)
                / (recent["flow_speed_median"] + 1e-8)
            )
        )
        priority = (
            10.0 * bool(flags.get(prompt))
            + 100.0 * abs(deltas["identity_background"])
            + 20.0 * abs(deltas["temporal_mechanics"])
            + abs(deltas["dynamic_degree"])
            + 0.05 * abs(deltas["official_quality_score"])
            + temporal_disagreement
        )
        ranked.append((priority, prompt, deltas, temporal_disagreement))
    queue = []
    for priority, prompt, deltas, disagreement in sorted(ranked, reverse=True)[:limit]:
        item = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "v191_prompt_index": int(item["v191_prompt_index"]),
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "priority": float(priority),
                "deltas_vs_all_recent": deltas,
                "temporal_disagreement": float(disagreement),
                "automatic_flags": sorted(flags.get(prompt, ())),
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in METHODS
                },
            }
        )
    return queue


def analyze(
    manifest: dict,
    frozen: dict,
    summary: dict,
    parts_root: Path,
    *,
    temporal_rows: dict,
    camera_context: dict,
) -> dict:
    methods = tuple(str(row.get("key")) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("confirmatory") is not True
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or manifest.get("prompt_items") != frozen["prompt_items"]
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
        or tuple(summary.get("methods") or {}) != METHODS
        or tuple(summary.get("dimensions") or ()) != DIMENSIONS
        or summary.get("missing")
    ):
        raise ValueError("v194 analysis requires one complete frozen grid")
    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    comparisons = []
    pairs = (
        (CANDIDATE, LOCAL_CONTROL, "primary_equal_budget"),
        (CANDIDATE, NATIVE_CONTROL, "causal_checkpoint_native_reference"),
        (LOCAL_CONTROL, NATIVE_CONTROL, "budget_context"),
    )
    for pair_index, (candidate, control, role) in enumerate(pairs):
        for metric_index, metric in enumerate(base.METRICS):
            comparisons.append(
                contrast(
                    rows,
                    candidate=candidate,
                    control=control,
                    role=role,
                    metric=metric,
                    seed=1940000 + pair_index * 101 + metric_index,
                )
            )
    base.bh(comparisons)
    means = method_means(rows)
    dynamic = dynamic_metric_validity(
        rows,
        methods=METHODS,
        prompt_count=PROMPT_COUNT,
    )
    noninferiority = {
        control: noninferiority_gate(
            by_metric(comparisons, control),
            control=control,
            dynamic_validity=dynamic,
            means=means,
        )
        for control in (LOCAL_CONTROL, NATIVE_CONTROL)
    }
    eligible_targets = [
        metric
        for metric in manifest["positive_metrics_to_transfer"]
        if metric != "dynamic_degree" or dynamic["informative"]
    ]
    local = by_metric(comparisons, LOCAL_CONTROL)
    replicated = {
        metric: {
            "mean_delta": float(local[metric]["mean_delta"]),
            "bootstrap_ci95": local[metric]["bootstrap_ci95"],
            "replicated": float(local[metric]["bootstrap_ci95"][0]) > 0.0,
        }
        for metric in eligible_targets
    }
    positive = {
        "eligible_frozen_targets": eligible_targets,
        "metrics": replicated,
        "pass": bool(replicated)
        and any(row["replicated"] for row in replicated.values()),
    }
    transfer = _same_prompt_transfer(
        frozen,
        comparisons,
        eligible_targets=eligible_targets,
    )
    recent_guard = temporal_guard(
        temporal_rows,
        candidate=CANDIDATE,
        control=LOCAL_CONTROL,
        prompt_count=PROMPT_COUNT,
    )
    native_guard = temporal_guard(
        temporal_rows,
        candidate=CANDIDATE,
        control=NATIVE_CONTROL,
        prompt_count=PROMPT_COUNT,
    )
    gates = {
        "equal_budget_noninferiority": noninferiority[LOCAL_CONTROL]["pass"],
        "native21_noninferiority": noninferiority[NATIVE_CONTROL]["pass"],
        "frozen_positive_target_replicated": positive["pass"],
        "same_prompt_cross_checkpoint_effect": transfer["pass"],
        "temporal_safety_vs_equal_budget": recent_guard["automatic_safety_pass"],
        "temporal_safety_vs_native21": native_guard["automatic_safety_pass"],
    }
    passed = all(gates.values())
    recommendation = (
        "freeze_head_phase_route_as_cross_checkpoint_supported"
        if passed
        else "do_not_claim_v194_checkpoint_transfer"
    )
    review = (
        _review_queue(
            manifest,
            rows,
            temporal_rows,
            (recent_guard, native_guard),
        )
        if passed
        else []
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "confirmatory": True,
        "transfer_axis": frozen["transfer_axis"],
        "prompt_count": PROMPT_COUNT,
        "seed": int(frozen["seed"]),
        "methods": list(METHODS),
        "candidate": CANDIDATE,
        "local_control": LOCAL_CONTROL,
        "native_control": NATIVE_CONTROL,
        "method_means": means,
        "comparisons": comparisons,
        "metric_validity": {"dynamic_degree": dynamic},
        "noninferiority": noninferiority,
        "frozen_positive_effect": positive,
        "same_prompt_cross_checkpoint_effect": transfer,
        "automatic_temporal_guards": {
            LOCAL_CONTROL: recent_guard,
            NATIVE_CONTROL: native_guard,
        },
        "camera_compensated_motion": camera_context,
        "motion_improvement_claim_supported": bool(
            passed and camera_context["motion_improvement_claim_supported"]
        ),
        "confirmation_gates": gates,
        "cross_checkpoint_transfer_confirmed": passed,
        "recommendation": recommendation,
        "manual_review_required_for_recommendation": passed,
        "targeted_review_queue_cap": 4,
        "targeted_review_queue": review,
        "claim_boundary": frozen["claim_boundary"],
    }


def render(report: dict) -> str:
    lines = [
        "# v194 Causal Checkpoint Transfer Decision",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Transfer confirmed: `{report['cross_checkpoint_transfer_confirmed']}`",
        f"- Motion claim supported: `{report['motion_improvement_claim_supported']}`",
        f"- Manual review videos: `{len(report['targeted_review_queue'])}`",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for gate, passed in report["confirmation_gates"].items():
        lines.append(f"| {gate} | {passed} |")
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--temporal-contract", type=Path, required=True)
    parser.add_argument("--camera-motion-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source") or {}
    input_path = Path(str(source.get("input_manifest", "")))
    if not input_path.is_file() or sha256(input_path) != source.get(
        "input_manifest_sha256"
    ):
        raise ValueError("v194 comparison lost its frozen input manifest")
    frozen = verify_input(input_path)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verify_temporal_contract(args.temporal_contract, manifest_path, args.temporal_csv)
    temporal_rows = load_temporal_rows(
        args.temporal_csv,
        methods=METHODS,
        prompt_count=PROMPT_COUNT,
    )
    camera = _camera_context(args.camera_motion_report, manifest_path)
    report = analyze(
        manifest,
        frozen,
        summary,
        args.parts_root,
        temporal_rows=temporal_rows,
        camera_context=camera,
    )
    report["metric_runtime_fingerprint"] = metric_runtime_fingerprint(
        args.parts_root,
        METHODS,
        tuple(summary["dimensions"]),
    )
    report["source"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "input_manifest": str(input_path.resolve()),
        "input_manifest_sha256": sha256(input_path),
        "vbench_summary": str(args.summary.resolve()),
        "vbench_summary_sha256": sha256(args.summary),
        "temporal_diagnostics": str(args.temporal_csv.resolve()),
        "temporal_diagnostics_sha256": sha256(args.temporal_csv),
        "temporal_contract": str(args.temporal_contract.resolve()),
        "temporal_contract_sha256": sha256(args.temporal_contract),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v194-analysis] "
        f"recommendation={report['recommendation']} "
        f"confirmed={str(report['cross_checkpoint_transfer_confirmed']).lower()} "
        f"motion={str(report['motion_improvement_claim_supported']).lower()}"
    )


if __name__ == "__main__":
    main()
