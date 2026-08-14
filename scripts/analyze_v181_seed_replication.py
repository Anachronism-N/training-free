#!/usr/bin/env python3
"""Combine the shared 64 prompts across the two frozen v181 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v174_paired_metrics as stats
import numpy as np
from prepare_v178_rccp_holdout import sha256
from prepare_v181_rccp_long_stress import METHODS

METHOD = "rccp_matched"
CONTROLS = ("sf_native", "all_recent")
METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
)
WINDOWS = ("full", "late_half")
SHARED_PROMPTS = 64


def _load_scope(root: Path, expected_scope: str) -> tuple[dict, dict, Path, Path]:
    comparison_path = root / "vbench_comparison" / "comparison_manifest.json"
    analysis_path = root / "analysis" / "v181_long_stress_metrics.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        comparison.get("scope") != expected_scope
        or analysis.get("scope") != expected_scope
        or analysis.get("input_provenance", {}).get("comparison_manifest_sha256")
        != sha256(comparison_path)
        or tuple(analysis.get("methods") or ()) != METHODS
    ):
        raise ValueError(f"invalid or mixed v181 scope: {expected_scope}")
    return comparison, analysis, comparison_path, analysis_path


def _metric_rows(analysis: dict, window: str, method: str) -> list[dict]:
    rows = analysis.get("per_prompt_metrics", {}).get(window, {}).get(method)
    if not isinstance(rows, list) or [row.get("prompt_index") for row in rows] != list(
        range(int(analysis["prompt_count"]))
    ):
        raise ValueError(f"incomplete {analysis['scope']}:{window}:{method} rows")
    return rows


def _correlation(left: list[float], right: list[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _confirmed(row: dict) -> bool:
    return bool(
        row["pooled_mean_delta"] > 0.0
        and row["pooled_bootstrap_ci95"][0] > 0.0
        and row["q_value"] <= 0.10
        and row["pooled_win_fraction"] >= 0.55
        and row["seed0_mean_delta"] > 0.0
        and row["seed10000_mean_delta"] > 0.0
    )


def analyze(
    main_comparison: dict,
    main: dict,
    replicate_comparison: dict,
    replicate: dict,
) -> dict:
    main_items = main_comparison.get("prompt_items") or ()
    replicate_items = replicate_comparison.get("prompt_items") or ()
    main_source = main_comparison.get("source") or {}
    replicate_source = replicate_comparison.get("source") or {}
    if (
        int(main.get("prompt_count", -1)) != 128
        or int(replicate.get("prompt_count", -1)) != SHARED_PROMPTS
        or int(main.get("seed", -1)) != 0
        or int(replicate.get("seed", -1)) != 10000
        or main_items[:SHARED_PROMPTS] != replicate_items
        or [row.get("source_index") for row in replicate_items] != list(range(256, 320))
        or not main_source.get("input_manifest_sha256")
        or main_source.get("input_manifest_sha256")
        != replicate_source.get("input_manifest_sha256")
        or not main_source.get("v178_paired_result_sha256")
        or main_source.get("v178_paired_result_sha256")
        != replicate_source.get("v178_paired_result_sha256")
        or main.get("metric_runtime_fingerprint", {}).get("sha256") is None
        or main.get("metric_runtime_fingerprint", {}).get("sha256")
        != replicate.get("metric_runtime_fingerprint", {}).get("sha256")
    ):
        raise ValueError(
            "v181 seed scopes do not share prompts, inputs, or metric runtime"
        )

    comparisons = []
    for window_index, window in enumerate(WINDOWS):
        for control_index, control in enumerate(CONTROLS):
            for metric_index, metric in enumerate(METRICS):
                main_method = _metric_rows(main, window, METHOD)
                main_control = _metric_rows(main, window, control)
                replicate_method = _metric_rows(replicate, window, METHOD)
                replicate_control = _metric_rows(replicate, window, control)
                seed0 = [
                    float(main_method[index][metric])
                    - float(main_control[index][metric])
                    for index in range(SHARED_PROMPTS)
                ]
                seed10000 = [
                    float(replicate_method[index][metric])
                    - float(replicate_control[index][metric])
                    for index in range(SHARED_PROMPTS)
                ]
                pooled = [
                    0.5 * (seed0[index] + seed10000[index])
                    for index in range(SHARED_PROMPTS)
                ]
                array = np.asarray(pooled, dtype=np.float64)
                comparisons.append(
                    {
                        "control": control,
                        "metric": metric,
                        "window": window,
                        "seed0_mean_delta": float(np.mean(seed0)),
                        "seed10000_mean_delta": float(np.mean(seed10000)),
                        "pooled_mean_delta": float(array.mean()),
                        "pooled_median_delta": float(np.median(array)),
                        "pooled_win_fraction": float(np.mean(array > 0.0)),
                        "pooled_bootstrap_ci95": stats.bootstrap_ci(
                            pooled,
                            seed=(
                                1815026
                                + window_index * 1000
                                + control_index * 101
                                + metric_index
                            ),
                        ),
                        "p_value": stats.sign_p(pooled),
                        "per_prompt_sign_agreement": float(
                            np.mean(
                                np.sign(np.asarray(seed0))
                                == np.sign(np.asarray(seed10000))
                            )
                        ),
                        "per_prompt_effect_correlation": _correlation(
                            seed0,
                            seed10000,
                        ),
                        "seed0_per_prompt_delta": seed0,
                        "seed10000_per_prompt_delta": seed10000,
                        "pooled_per_prompt_delta": pooled,
                    }
                )
    stats.bh(comparisons)
    for row in comparisons:
        row["confirmed"] = _confirmed(row)

    def select(metrics: tuple[str, ...]) -> list[dict]:
        return [row for row in comparisons if row["metric"] in metrics]

    quality_identity_gate = all(
        row["confirmed"]
        for row in select(("official_quality_score", "identity_background"))
    )
    identity_motion_gate = all(
        row["confirmed"] for row in select(("identity_background", "dynamic_degree"))
    )
    dynamic_nonregression = all(
        row["seed0_mean_delta"] >= -0.02 and row["seed10000_mean_delta"] >= -0.02
        for row in comparisons
        if row["metric"] == "dynamic_degree"
    )
    if quality_identity_gate and identity_motion_gate:
        decision = "two_seed_quality_identity_motion_confirmed"
    elif quality_identity_gate and dynamic_nonregression:
        decision = "two_seed_quality_identity_confirmed"
    elif identity_motion_gate:
        decision = "two_seed_identity_motion_confirmed"
    elif all(
        row["seed0_mean_delta"] > 0.0 and row["seed10000_mean_delta"] > 0.0
        for row in comparisons
    ):
        decision = "two_seed_directional_only"
    else:
        decision = "two_seed_rccp_not_confirmed"

    conflicts = []
    identity_row = next(
        row
        for row in comparisons
        if row["control"] == "sf_native"
        and row["metric"] == "identity_background"
        and row["window"] == "late_half"
    )
    dynamic_row = next(
        row
        for row in comparisons
        if row["control"] == "sf_native"
        and row["metric"] == "dynamic_degree"
        and row["window"] == "late_half"
    )
    video_dirs = {
        "seed0": {row["key"]: row["video_dir"] for row in main_comparison["methods"]},
        "seed10000": {
            row["key"]: row["video_dir"] for row in replicate_comparison["methods"]
        },
    }
    for index in range(SHARED_PROMPTS):
        identity_disagreement = abs(
            identity_row["seed0_per_prompt_delta"][index]
            - identity_row["seed10000_per_prompt_delta"][index]
        )
        motion_disagreement = abs(
            dynamic_row["seed0_per_prompt_delta"][index]
            - dynamic_row["seed10000_per_prompt_delta"][index]
        )
        conflicts.append(
            {
                "prompt_index": index,
                "source_index": int(replicate_items[index]["source_index"]),
                "prompt": replicate_items[index]["text"],
                "review_priority": float(identity_disagreement + motion_disagreement),
                "late_identity_seed_delta_disagreement": float(identity_disagreement),
                "late_dynamic_seed_delta_disagreement": float(motion_disagreement),
                "videos": {
                    seed: {
                        method: str(Path(methods[method]) / f"{index:06d}.mp4")
                        for method in METHODS
                    }
                    for seed, methods in video_dirs.items()
                },
            }
        )
    conflicts.sort(key=lambda row: (-row["review_priority"], row["prompt_index"]))

    return {
        "version": 1,
        "experiment": "v181_rccp_two_seed_replication",
        "profile_contract": "v177",
        "shared_prompt_count": SHARED_PROMPTS,
        "source_prompt_indices": list(range(256, 320)),
        "seeds": [0, 10000],
        "comparisons": comparisons,
        "quality_identity_gate": quality_identity_gate,
        "identity_motion_gate": identity_motion_gate,
        "dynamic_nonregression_gate": dynamic_nonregression,
        "decision": decision,
        "targeted_review": conflicts[:4],
        "manual_review_required_for_decision": False,
        "claim_boundary": (
            "The two-seed gate averages paired effects within each shared prompt "
            "before bootstrap inference. It supports seed robustness only for "
            "the frozen 64-prompt prefix and does not add independent prompts."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v181 Two-Seed Replication",
        "",
        f"Decision: `{report['decision']}`",
        f"Quality + identity gate: `{report['quality_identity_gate']}`",
        f"Identity + motion gate: `{report['identity_motion_gate']}`",
        f"Dynamic non-regression: `{report['dynamic_nonregression_gate']}`",
        "",
        "| Window | Control | Metric | Seed 0 | Seed 10000 | Pooled | CI95 | q |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"| {row['window']} | {row['control']} | {row['metric']} | "
            f"{row['seed0_mean_delta']:.6f} | {row['seed10000_mean_delta']:.6f} | "
            f"{row['pooled_mean_delta']:.6f} | "
            f"[{row['pooled_bootstrap_ci95'][0]:.6f}, "
            f"{row['pooled_bootstrap_ci95'][1]:.6f}] | {row['q_value']:.4g} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-scope-root", type=Path, required=True)
    parser.add_argument("--replicate-scope-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_comparison, main, main_manifest, main_analysis = _load_scope(
        args.main_scope_root,
        "long60_seed0",
    )
    replicate_comparison, replicate, replicate_manifest, replicate_analysis = (
        _load_scope(
            args.replicate_scope_root,
            "long60_seed10000_64",
        )
    )
    report = analyze(main_comparison, main, replicate_comparison, replicate)
    report["input_provenance"] = {
        "main_comparison_manifest": str(main_manifest.resolve()),
        "main_comparison_manifest_sha256": sha256(main_manifest),
        "main_analysis": str(main_analysis.resolve()),
        "main_analysis_sha256": sha256(main_analysis),
        "replicate_comparison_manifest": str(replicate_manifest.resolve()),
        "replicate_comparison_manifest_sha256": sha256(replicate_manifest),
        "replicate_analysis": str(replicate_analysis.resolve()),
        "replicate_analysis_sha256": sha256(replicate_analysis),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v181-seed] decision={report['decision']} output={args.output}")


if __name__ == "__main__":
    main()
