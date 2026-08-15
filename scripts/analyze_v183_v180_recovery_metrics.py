#!/usr/bin/env python3
"""Paired exploratory analysis for the recovered 128-prompt v180 grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from audit_v183_v180_recovery import METHODS, PROMPT_COUNT
from prepare_v178_rccp_holdout import sha256
from prepare_v183_v180_recovery_vbench import EXPERIMENT


PAIRS = (
    ("rccp_matched", "sf_native", "end_to_end"),
    ("rccp_matched", "all_recent", "strict5_increment"),
    ("all_coverage", "all_recent", "all_head_operator"),
    ("all_recent", "sf_native", "equal_budget_host_control"),
    ("rccp_matched", "all_coverage", "sparse_vs_dense_coverage"),
)
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
)


def _contrast(
    rows: dict,
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
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(deltas, seed=seed),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
        "inferential_role": "exploratory_descriptive",
    }


def _primary_rows(comparisons: list[dict], role: str) -> dict[str, dict]:
    return {
        row["metric"]: row
        for row in comparisons
        if row["comparison_role"] == role and row["metric"] in PRIMARY_METRICS
    }


def _directional_nonregression(rows: dict[str, dict]) -> bool:
    return bool(
        rows["official_quality_score"]["mean_delta"] >= 0.0
        and rows["identity_background"]["mean_delta"] >= 0.0
        and rows["dynamic_degree"]["mean_delta"] >= -0.02
    )


def _targeted_review(manifest: dict, rows: dict, limit: int = 6) -> list[dict]:
    video_dirs = {row["key"]: row["video_dir"] for row in manifest["methods"]}
    queue = []
    for prompt in range(PROMPT_COUNT):
        identity = (
            rows[("rccp_matched", prompt)]["identity_background"]
            - rows[("sf_native", prompt)]["identity_background"]
        )
        dynamic = (
            rows[("rccp_matched", prompt)]["dynamic_degree"]
            - rows[("sf_native", prompt)]["dynamic_degree"]
        )
        quality = (
            rows[("rccp_matched", prompt)]["official_quality_score"]
            - rows[("sf_native", prompt)]["official_quality_score"]
        )
        conflict = (identity > 0.0) != (dynamic > 0.0)
        priority = abs(identity) + abs(dynamic) + 0.25 * abs(quality)
        if conflict:
            priority += 1.0
        prompt_row = manifest["prompt_items"][prompt]
        queue.append(
            {
                "prompt_index": prompt,
                "source_index": int(prompt_row["source_index"]),
                "prompt": prompt_row["text"],
                "identity_delta_vs_sf": float(identity),
                "dynamic_delta_vs_sf": float(dynamic),
                "quality_delta_vs_sf": float(quality),
                "identity_dynamic_sign_conflict": conflict,
                "review_priority": float(priority),
                "videos": {
                    method: str(Path(video_dirs[method]) / f"{prompt:06d}-0.mp4")
                    for method in METHODS
                },
            }
        )
    return sorted(
        queue,
        key=lambda row: (-row["review_priority"], row["prompt_index"]),
    )[:limit]


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    methods = tuple(row.get("key") for row in manifest.get("methods") or ())
    prompt_items = manifest.get("prompt_items") or ()
    if (
        manifest.get("experiment") != EXPERIMENT
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or len(prompt_items) != PROMPT_COUNT
        or [int(row.get("index", -1)) for row in prompt_items] != list(range(PROMPT_COUNT))
        or [int(row.get("source_index", -1)) for row in prompt_items]
        != list(range(128, 256))
        or manifest.get("evaluation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid v183 recovery comparison manifest")
    if tuple(summary.get("methods") or {}) != METHODS or summary.get("missing"):
        raise ValueError("v183 paired analysis requires a complete core-9 summary")

    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    comparisons = []
    for pair_index, (candidate, control, role) in enumerate(PAIRS):
        for metric_index, metric in enumerate(base.METRICS):
            comparisons.append(
                _contrast(
                    rows,
                    candidate,
                    control,
                    role,
                    metric,
                    seed=1832026 + pair_index * 101 + metric_index,
                )
            )
    base.bh(comparisons)

    method_means = {
        method: {
            metric: float(
                np.mean([rows[(method, prompt)][metric] for prompt in range(PROMPT_COUNT)])
            )
            for metric in base.METRICS
        }
        for method in METHODS
    }
    end_to_end = _primary_rows(comparisons, "end_to_end")
    strict5_increment = _primary_rows(comparisons, "strict5_increment")
    all_head = _primary_rows(comparisons, "all_head_operator")
    end_to_end_gate = _directional_nonregression(end_to_end)
    strict5_increment_gate = _directional_nonregression(strict5_increment)
    all_head_gate = _directional_nonregression(all_head)
    strong_strict5_signal = bool(
        end_to_end["official_quality_score"]["bootstrap_ci95"][0] > 0.0
        and end_to_end["identity_background"]["bootstrap_ci95"][0] > 0.0
        and strict5_increment["official_quality_score"]["bootstrap_ci95"][0] > 0.0
        and strict5_increment["identity_background"]["bootstrap_ci95"][0] > 0.0
    )

    if end_to_end_gate and strict5_increment_gate:
        recommendation = "rerun_formal_membership_controls"
    elif end_to_end_gate:
        recommendation = "strict5_end_to_end_promising_membership_unresolved"
    elif all_head_gate:
        recommendation = "reprofile_coverage_operator_before_new_membership_test"
    else:
        recommendation = "stop_static_strict5_and_revisit_operator"

    formal = bool(manifest.get("formal_rccp_membership_claim_allowed"))
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "evidence_scope": manifest.get("evidence_scope"),
        "formal_rccp_membership_claim_allowed": formal,
        "prompt_count": PROMPT_COUNT,
        "methods": list(METHODS),
        "method_means": method_means,
        "comparisons": comparisons,
        "end_to_end_directional_nonregression": end_to_end_gate,
        "strict5_increment_directional_nonregression": strict5_increment_gate,
        "all_head_coverage_directional_nonregression": all_head_gate,
        "strong_strict5_exploratory_signal": strong_strict5_signal,
        "recommendation": recommendation,
        "targeted_review": _targeted_review(manifest, rows),
        "manual_review_required_for_recommendation": False,
        "claim_boundary": (
            "The 128-prompt result can compare generated videos, cache operators, and the "
            "frozen strict-five candidate against SF. It cannot establish that RCCP chose "
            "better heads than count/layer-matched alternatives because the recorded v178 "
            "gate is not a real paired metric artifact. All confidence intervals and q-values "
            "in this report are exploratory."
            if not formal
            else "Fresh-suite transfer is interpreted together with the separately valid v178 gate."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v183 Recovered v180 Paired Analysis",
        "",
        f"Evidence scope: `{report['evidence_scope']}`",
        f"Recommendation: `{report['recommendation']}`",
        f"Formal RCCP membership claim allowed: `{report['formal_rccp_membership_claim_allowed']}`",
        "",
        "| Comparison role | Metric | Mean delta | CI95 | Win | q (exploratory) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        lines.append(
            f"| {row['comparison_role']} | {row['metric']} | {row['mean_delta']:.6f} | "
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
        "[v183-v180-paired] "
        f"recommendation={report['recommendation']} "
        f"formal_membership={report['formal_rccp_membership_claim_allowed']}"
    )


if __name__ == "__main__":
    main()
