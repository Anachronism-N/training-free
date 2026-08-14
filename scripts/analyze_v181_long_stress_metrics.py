#!/usr/bin/env python3
"""Paired full- and late-window analysis for one v181 60-second scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v165_final_decision as detail
import analyze_v174_paired_metrics as base
import numpy as np
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v178_rccp_holdout import sha256
from prepare_v181_rccp_long_stress import METHODS
from prepare_v181_vbench_comparison import EXPERIMENT

METHOD = "rccp_matched"
CONTROLS = ("sf_native", "all_recent")
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
)
WINDOWS = ("full", "late_half")
CLIPS_PER_VIDEO = 30


def _load_window_rows(
    parts_root: Path,
    summary: dict,
    prompt_count: int,
    start: int,
    end: int,
) -> dict:
    if not 0 <= start < end <= CLIPS_PER_VIDEO:
        raise ValueError(f"invalid v181 clip window [{start}, {end})")
    rows = {
        (method, prompt): {} for method in METHODS for prompt in range(prompt_count)
    }
    for method in METHODS:
        for dimension in DIMENSIONS:
            clips = detail.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
                prompt_count=prompt_count,
                clips_per_video=CLIPS_PER_VIDEO,
            )
            raw_values = [
                value for prompt in range(prompt_count) for value in clips[prompt]
            ]
            summary_value = detail.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = detail.scale_factor(
                float(np.mean(raw_values)),
                summary_value,
                name=f"{method}:{dimension}",
            )
            for prompt in range(prompt_count):
                rows[(method, prompt)][dimension] = factor * float(
                    np.mean(clips[prompt][start:end])
                )
    return base.derived_rows(rows, METHODS, prompt_count)


def _contrast(
    control: str,
    metric: str,
    window: str,
    values: list[float],
    seed_offset: int,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "comparison": f"{METHOD}_minus_{control}",
        "control": control,
        "metric": metric,
        "window": window,
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "win_fraction": float(np.mean(array > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(
            values,
            seed=1812026 + seed_offset,
        ),
        "p_value": base.sign_p(values),
        "per_prompt_delta": values,
    }


def _confirmed(row: dict) -> bool:
    return bool(
        row["mean_delta"] > 0.0
        and row["bootstrap_ci95"][0] > 0.0
        and row["q_value"] <= 0.10
        and row["win_fraction"] >= 0.55
    )


def _targeted_review(
    manifest: dict,
    windows: dict[str, dict],
    *,
    limit: int = 4,
) -> list[dict]:
    prompt_count = int(manifest["prompt_count"])
    prompt_items = manifest["prompt_items"]
    video_dirs = {row["key"]: row["video_dir"] for row in manifest["methods"]}
    late = windows["late_half"]
    candidates = []
    for prompt in range(prompt_count):
        identity = (
            late[(METHOD, prompt)]["identity_background"]
            - late[("sf_native", prompt)]["identity_background"]
        )
        dynamic = (
            late[(METHOD, prompt)]["dynamic_degree"]
            - late[("sf_native", prompt)]["dynamic_degree"]
        )
        quality = (
            late[(METHOD, prompt)]["official_quality_score"]
            - late[("sf_native", prompt)]["official_quality_score"]
        )
        sign_conflict = (identity > 0.0) != (dynamic > 0.0)
        priority = abs(identity) + abs(dynamic) + 0.25 * abs(quality)
        if sign_conflict:
            priority += 1.0
        candidates.append(
            {
                "prompt_index": prompt,
                "source_index": int(prompt_items[prompt]["source_index"]),
                "prompt": prompt_items[prompt]["text"],
                "late_identity_delta_vs_sf": float(identity),
                "late_dynamic_delta_vs_sf": float(dynamic),
                "late_quality_delta_vs_sf": float(quality),
                "identity_dynamic_sign_conflict": sign_conflict,
                "review_priority": float(priority),
                "videos": {
                    method: str(Path(video_dirs[method]) / f"{prompt:06d}.mp4")
                    for method in METHODS
                },
            }
        )
    return sorted(
        candidates,
        key=lambda row: (-row["review_priority"], row["prompt_index"]),
    )[:limit]


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row["key"] for row in manifest.get("methods") or ())
    prompt_items = manifest.get("prompt_items") or ()
    prompt_count = int(manifest.get("prompt_count", -1))
    scope = str(manifest.get("scope", ""))
    expected = {
        "long60_seed0": (128, 0, list(range(256, 384))),
        "long60_seed10000_64": (64, 10000, list(range(256, 320))),
    }
    if scope not in expected:
        raise ValueError(f"unsupported v181 analysis scope: {scope}")
    expected_count, expected_seed, expected_sources = expected[scope]
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("profile_contract") != "v177"
        or manifest.get("evaluation_prompts_used_for_membership") is not False
        or prompt_count != expected_count
        or int(manifest.get("num_output_frames", -1)) != 240
        or int(manifest.get("seed", -1)) != expected_seed
        or methods != METHODS
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
        or len(prompt_items) != prompt_count
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(prompt_count))
        or [int(row.get("source_index", -1)) for row in prompt_items]
        != expected_sources
    ):
        raise ValueError("invalid v181 comparison manifest")
    if tuple(summary.get("methods") or {}) != METHODS or summary.get("missing"):
        raise ValueError("v181 paired analysis received an incomplete summary")
    if tuple(summary.get("dimensions") or ()) != DIMENSIONS:
        raise ValueError("v181 paired analysis received a mixed metric profile")

    window_rows = {
        "full": _load_window_rows(
            parts_root,
            summary,
            prompt_count,
            0,
            CLIPS_PER_VIDEO,
        ),
        "early_half": _load_window_rows(
            parts_root,
            summary,
            prompt_count,
            0,
            CLIPS_PER_VIDEO // 2,
        ),
        "late_half": _load_window_rows(
            parts_root,
            summary,
            prompt_count,
            CLIPS_PER_VIDEO // 2,
            CLIPS_PER_VIDEO,
        ),
    }
    comparisons = []
    for window_index, window in enumerate(WINDOWS):
        rows = window_rows[window]
        for control_index, control in enumerate(CONTROLS):
            for metric_index, metric in enumerate(base.METRICS):
                values = [
                    rows[(METHOD, prompt)][metric] - rows[(control, prompt)][metric]
                    for prompt in range(prompt_count)
                ]
                comparisons.append(
                    _contrast(
                        control,
                        metric,
                        window,
                        values,
                        window_index * 1000 + control_index * 101 + metric_index,
                    )
                )

    primary = [
        row
        for row in comparisons
        if row["metric"] in PRIMARY_METRICS and row["window"] in WINDOWS
    ]
    base.bh(primary)
    for row in comparisons:
        if "q_value" in row:
            row["inferential_role"] = "preregistered_primary"
        else:
            row["q_value"] = None
            row["inferential_role"] = "descriptive"

    def get(control: str, metric: str, window: str) -> dict:
        return next(
            row
            for row in comparisons
            if row["control"] == control
            and row["metric"] == metric
            and row["window"] == window
        )

    quality_identity = [
        get(control, metric, window)
        for control in CONTROLS
        for metric in ("official_quality_score", "identity_background")
        for window in WINDOWS
    ]
    identity_motion = [
        get(control, metric, window)
        for control in CONTROLS
        for metric in ("identity_background", "dynamic_degree")
        for window in WINDOWS
    ]
    late_identity = [
        get(control, "identity_background", "late_half") for control in CONTROLS
    ]
    dynamic = [
        get(control, "dynamic_degree", window)
        for control in CONTROLS
        for window in WINDOWS
    ]
    quality_identity_gate = all(_confirmed(row) for row in quality_identity)
    identity_motion_gate = all(_confirmed(row) for row in identity_motion)
    late_identity_gate = all(_confirmed(row) for row in late_identity)
    dynamic_nonregression = all(row["mean_delta"] >= -0.02 for row in dynamic)
    directional = all(
        get(control, metric, window)["mean_delta"] > 0.0
        for control in CONTROLS
        for metric in PRIMARY_METRICS
        for window in WINDOWS
    )
    if quality_identity_gate and identity_motion_gate:
        decision = "long_horizon_quality_identity_motion_confirmed"
    elif quality_identity_gate and dynamic_nonregression:
        decision = "long_horizon_quality_identity_confirmed"
    elif late_identity_gate and identity_motion_gate:
        decision = "long_horizon_identity_motion_confirmed"
    elif directional:
        decision = "long_horizon_directional_only"
    else:
        decision = "long_horizon_rccp_not_confirmed"

    effect_decay = []
    for control in CONTROLS:
        for metric in PRIMARY_METRICS:
            early = window_rows["early_half"]
            late = window_rows["late_half"]
            values = [
                (late[(METHOD, prompt)][metric] - late[(control, prompt)][metric])
                - (early[(METHOD, prompt)][metric] - early[(control, prompt)][metric])
                for prompt in range(prompt_count)
            ]
            effect_decay.append(
                {
                    "control": control,
                    "metric": metric,
                    "late_minus_early_effect": float(np.mean(values)),
                    "bootstrap_ci95": base.bootstrap_ci(
                        values,
                        seed=1819026 + len(effect_decay),
                    ),
                    "per_prompt_delta": values,
                    "inferential_role": "descriptive_effect_persistence",
                }
            )

    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "profile_contract": "v177",
        "scope": scope,
        "prompt_count": prompt_count,
        "seed": expected_seed,
        "methods": list(METHODS),
        "clips_per_video": CLIPS_PER_VIDEO,
        "windows": {
            "full": [0, 30],
            "early_half": [0, 15],
            "late_half": [15, 30],
        },
        "comparisons": comparisons,
        "effect_persistence": effect_decay,
        "per_prompt_metrics": {
            window: {
                method: [
                    {"prompt_index": prompt, **window_rows[window][(method, prompt)]}
                    for prompt in range(prompt_count)
                ]
                for method in METHODS
            }
            for window in ("full", "late_half")
        },
        "quality_identity_gate": quality_identity_gate,
        "identity_motion_gate": identity_motion_gate,
        "late_identity_gate": late_identity_gate,
        "dynamic_nonregression_gate": dynamic_nonregression,
        "decision": decision,
        "targeted_review": _targeted_review(manifest, window_rows),
        "manual_review_required_for_decision": False,
        "claim_boundary": (
            "v181 tests the exact frozen v177 five-head RCCP map on unseen "
            "60-second prompts. Full and late-half endpoints are frozen; this "
            "scope does not establish cross-model or scene-switch transfer."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v181 Long-Stress Paired Analysis",
        "",
        f"Scope: `{report['scope']}`",
        f"Decision: `{report['decision']}`",
        f"Quality + identity gate: `{report['quality_identity_gate']}`",
        f"Identity + motion gate: `{report['identity_motion_gate']}`",
        f"Late identity gate: `{report['late_identity_gate']}`",
        f"Dynamic non-regression: `{report['dynamic_nonregression_gate']}`",
        "",
        "| Window | Control | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        lines.append(
            f"| {row['window']} | {row['control']} | {row['metric']} | "
            f"{row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {row['q_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The review queue is capped at four late-window metric-conflict cases.",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(manifest, summary, args.parts_root)
    report["metric_runtime_fingerprint"] = metric_runtime_fingerprint(
        args.parts_root,
        METHODS,
        tuple(summary["dimensions"]),
    )
    report["input_provenance"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "metric_summary": str(args.summary.resolve()),
        "metric_summary_sha256": sha256(args.summary),
        "parts_root": str(args.parts_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v181-paired] "
        f"scope={report['scope']} decision={report['decision']} output={args.output}"
    )


if __name__ == "__main__":
    main()
