#!/usr/bin/env python3
"""Exploratory paired full/early/late analysis for recovered v181 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as paired
import analyze_v181_long_stress_metrics as long_base
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from prepare_v174_vbench_comparison import DIMENSIONS
from prepare_v178_rccp_holdout import sha256
from prepare_v185_recovered_long60_comparison import (
    EXPERIMENT,
    METHODS,
    PROMPT_COUNT,
)


METHOD = "rccp_matched"
CONTROLS = ("sf_native", "all_recent")
WINDOWS = {
    "full": (0, 30),
    "early_half": (0, 15),
    "late_half": (15, 30),
}
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
)


def _load_window_rows(
    parts_root: Path,
    summary: dict,
    start: int,
    end: int,
) -> dict:
    return long_base._load_window_rows(
        parts_root,
        summary,
        PROMPT_COUNT,
        start,
        end,
    )


def _contrast(
    rows: dict,
    candidate: str,
    control: str,
    window: str,
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
        "candidate": candidate,
        "control": control,
        "window": window,
        "metric": metric,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "bootstrap_ci95": paired.bootstrap_ci(deltas, seed=seed),
        "p_value": paired.sign_p(deltas),
        "per_prompt_delta": deltas,
        "inferential_role": "exploratory_recovered",
    }


def _find(
    comparisons: list[dict],
    candidate: str,
    control: str,
    window: str,
    metric: str,
) -> dict:
    return next(
        row
        for row in comparisons
        if row["candidate"] == candidate
        and row["control"] == control
        and row["window"] == window
        and row["metric"] == metric
    )


def _targeted_review(manifest: dict, rows: dict, limit: int = 4) -> list[dict]:
    prompt_items = manifest["prompt_items"]
    video_dirs = {row["key"]: Path(row["video_dir"]) for row in manifest["methods"]}
    queue = []
    for prompt in range(PROMPT_COUNT):
        identity = {
            control: rows[(METHOD, prompt)]["identity_background"]
            - rows[(control, prompt)]["identity_background"]
            for control in CONTROLS
        }
        dynamic = {
            control: rows[(METHOD, prompt)]["dynamic_degree"]
            - rows[(control, prompt)]["dynamic_degree"]
            for control in CONTROLS
        }
        quality = {
            control: rows[(METHOD, prompt)]["official_quality_score"]
            - rows[(control, prompt)]["official_quality_score"]
            for control in CONTROLS
        }
        conflict = any(
            identity[control] * dynamic[control] < 0.0 for control in CONTROLS
        )
        priority = max(abs(value) for value in dynamic.values())
        priority += 20.0 * max(abs(value) for value in identity.values())
        priority += 0.1 * max(abs(value) for value in quality.values())
        if conflict:
            priority += 1.0
        item = prompt_items[prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(item["source_index"]),
                "prompt": item["text"],
                "late_identity_delta": identity,
                "late_dynamic_delta": dynamic,
                "late_quality_delta": quality,
                "identity_motion_sign_conflict": conflict,
                "review_priority": float(priority),
                "videos": {
                    method: str(video_dirs[method] / f"{prompt:06d}-0.mp4")
                    for method in METHODS
                },
            }
        )
    return sorted(
        queue,
        key=lambda row: (
            not row["identity_motion_sign_conflict"],
            -row["review_priority"],
            row["prompt_index"],
        ),
    )[:limit]


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row.get("key") for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("evidence_grade") != "exploratory_recovered"
        or manifest.get("formal_classifier_claim_eligible") is not False
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or int(manifest.get("num_output_frames", -1)) != 240
        or methods != METHODS
        or tuple(manifest.get("vbench_long_dimensions") or ()) != DIMENSIONS
        or len(manifest.get("prompt_items") or ()) != PROMPT_COUNT
        or tuple(summary.get("methods") or {}) != METHODS
        or tuple(summary.get("dimensions") or ()) != DIMENSIONS
        or summary.get("missing")
    ):
        raise ValueError("v185 paired analysis requires the complete recovered grid")

    window_rows = {
        window: _load_window_rows(parts_root, summary, start, end)
        for window, (start, end) in WINDOWS.items()
    }
    pairs = (
        (METHOD, "sf_native"),
        (METHOD, "all_recent"),
        ("all_recent", "sf_native"),
    )
    comparisons = []
    for window_index, (window, rows) in enumerate(window_rows.items()):
        for pair_index, (candidate, control) in enumerate(pairs):
            for metric_index, metric in enumerate(paired.METRICS):
                comparisons.append(
                    _contrast(
                        rows,
                        candidate,
                        control,
                        window,
                        metric,
                        seed=1852026
                        + window_index * 1000
                        + pair_index * 101
                        + metric_index,
                    )
                )
    paired.bh(comparisons)

    primary_windows = ("full", "late_half")
    quality_better = all(
        _find(comparisons, METHOD, control, window, "official_quality_score")[
            "mean_delta"
        ]
        > 0.0
        for control in CONTROLS
        for window in primary_windows
    )
    identity_better = all(
        _find(comparisons, METHOD, control, window, "identity_background")[
            "mean_delta"
        ]
        > 0.0
        for control in CONTROLS
        for window in primary_windows
    )
    dynamic_nonregression = all(
        _find(comparisons, METHOD, control, window, "dynamic_degree")["mean_delta"]
        >= -0.02
        for control in CONTROLS
        for window in primary_windows
    )
    dynamic_better = all(
        _find(comparisons, METHOD, control, window, "dynamic_degree")["mean_delta"]
        > 0.0
        for control in CONTROLS
        for window in primary_windows
    )
    if quality_better and identity_better and dynamic_nonregression:
        verdict = "static_five_long60_promising_exploratory"
    elif identity_better and not dynamic_nonregression:
        verdict = "static_five_identity_motion_tradeoff"
    elif dynamic_better and not identity_better:
        verdict = "static_five_motion_identity_tradeoff"
    else:
        verdict = "static_five_long60_not_supported"

    effect_persistence = []
    for control in CONTROLS:
        for metric in PRIMARY_METRICS:
            early = window_rows["early_half"]
            late = window_rows["late_half"]
            values = [
                (
                    late[(METHOD, prompt)][metric]
                    - late[(control, prompt)][metric]
                )
                - (
                    early[(METHOD, prompt)][metric]
                    - early[(control, prompt)][metric]
                )
                for prompt in range(PROMPT_COUNT)
            ]
            effect_persistence.append(
                {
                    "control": control,
                    "metric": metric,
                    "late_minus_early_effect": float(np.mean(values)),
                    "bootstrap_ci95": paired.bootstrap_ci(
                        values,
                        seed=1859026 + len(effect_persistence),
                    ),
                    "inferential_role": "exploratory_recovered",
                }
            )

    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "evidence_grade": "exploratory_recovered",
        "formal_classifier_claim_eligible": False,
        "prompt_count": PROMPT_COUNT,
        "methods": list(METHODS),
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "comparisons": comparisons,
        "effect_persistence": effect_persistence,
        "directional_summary": {
            "quality_better_than_both_controls": quality_better,
            "identity_better_than_both_controls": identity_better,
            "dynamic_nonregression_vs_both_controls": dynamic_nonregression,
            "dynamic_better_than_both_controls": dynamic_better,
        },
        "verdict": verdict,
        "manual_review_required_for_verdict": False,
        "targeted_review": _targeted_review(
            manifest,
            window_rows["late_half"],
        ),
        "claim_boundary": (
            "The verdict screens whether the recovered static-five videos "
            "deserve further study. It is not a classifier confirmation, a "
            "formal benchmark result, or a substitute for clean generation "
            "provenance."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v185 Recovered Long60 Paired Analysis",
        "",
        f"Verdict: `{report['verdict']}`",
        "Evidence grade: `exploratory_recovered`",
        "Formal classifier claim eligible: `false`",
        "",
        "| Window | Comparison | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        lines.append(
            f"| {row['window']} | {row['comparison']} | {row['metric']} | "
            f"{row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {row['q_value']:.4g} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
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
        f"[v185-recovered-decision] verdict={report['verdict']} "
        "formal_classifier_claim_eligible=false"
    )


if __name__ == "__main__":
    main()
