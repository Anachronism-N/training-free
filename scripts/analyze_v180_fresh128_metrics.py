#!/usr/bin/env python3
"""Paired fresh-suite quality and identity-motion analysis for v180."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from prepare_v178_rccp_holdout import sha256
from prepare_v180_rccp_fresh128 import METHODS, PROMPT_COUNT


EXPERIMENT = "v180_rccp_fresh128_vbench"
METHOD = "rccp_matched"
CONTROLS = ("sf_native", "all_recent", "all_coverage")
PRIMARY_CONTROLS = ("sf_native", "all_recent")
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "dynamic_degree",
)


def _contrast(
    control: str,
    metric: str,
    values: list[float],
    seed_offset: int,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "comparison": f"{METHOD}_minus_{control}",
        "control": control,
        "metric": metric,
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "win_fraction": float(np.mean(array > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(
            values,
            seed=1802026 + seed_offset,
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
    rows: dict,
    *,
    limit: int = 6,
) -> list[dict]:
    prompt_items = manifest["prompt_items"]
    video_dirs = {row["key"]: row["video_dir"] for row in manifest["methods"]}
    candidates = []
    for prompt in range(PROMPT_COUNT):
        identity = (
            rows[(METHOD, prompt)]["identity_background"]
            - rows[("sf_native", prompt)]["identity_background"]
        )
        dynamic = (
            rows[(METHOD, prompt)]["dynamic_degree"]
            - rows[("sf_native", prompt)]["dynamic_degree"]
        )
        quality = (
            rows[(METHOD, prompt)]["official_quality_score"]
            - rows[("sf_native", prompt)]["official_quality_score"]
        )
        sign_conflict = (identity > 0.0) != (dynamic > 0.0)
        score = abs(identity) + abs(dynamic) + 0.25 * abs(quality)
        if sign_conflict:
            score += 1.0
        candidates.append(
            {
                "prompt_index": prompt,
                "source_index": int(prompt_items[prompt]["source_index"]),
                "prompt": prompt_items[prompt]["text"],
                "identity_delta_vs_sf": float(identity),
                "dynamic_delta_vs_sf": float(dynamic),
                "quality_delta_vs_sf": float(quality),
                "identity_dynamic_sign_conflict": sign_conflict,
                "review_priority": float(score),
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
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("profile_contract") != "v177"
        or manifest.get("evaluation_prompts_used_for_membership") is not False
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or methods != METHODS
        or len(prompt_items) != PROMPT_COUNT
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(PROMPT_COUNT))
        or [int(row.get("source_index", -1)) for row in prompt_items]
        != list(range(128, 256))
    ):
        raise ValueError("invalid v180 comparison manifest")
    if tuple(summary.get("methods") or {}) != METHODS or summary.get("missing"):
        raise ValueError("v180 paired analysis received an incomplete summary")

    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    rows = base.derived_rows(raw, METHODS, PROMPT_COUNT)
    comparisons = []
    for control_index, control in enumerate(CONTROLS):
        for metric_index, metric in enumerate(base.METRICS):
            deltas = [
                rows[(METHOD, prompt)][metric] - rows[(control, prompt)][metric]
                for prompt in range(PROMPT_COUNT)
            ]
            comparisons.append(
                _contrast(
                    control,
                    metric,
                    deltas,
                    control_index * 101 + metric_index,
                )
            )

    primary = [
        row
        for row in comparisons
        if row["control"] in PRIMARY_CONTROLS
        and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(primary)
    for row in comparisons:
        if "q_value" in row:
            row["inferential_role"] = "preregistered_primary"
        else:
            row["q_value"] = None
            row["inferential_role"] = "descriptive"

    def get(control: str, metric: str) -> dict:
        return next(
            row
            for row in comparisons
            if row["control"] == control and row["metric"] == metric
        )

    quality_identity = [
        get(control, metric)
        for control in PRIMARY_CONTROLS
        for metric in ("official_quality_score", "identity_background")
    ]
    identity = [get(control, "identity_background") for control in PRIMARY_CONTROLS]
    dynamic = [get(control, "dynamic_degree") for control in PRIMARY_CONTROLS]
    quality_identity_gate = all(_confirmed(row) for row in quality_identity)
    identity_motion_gate = all(_confirmed(row) for row in (*identity, *dynamic))
    dynamic_nonregression = all(row["mean_delta"] >= -0.02 for row in dynamic)
    directional = all(
        get(control, metric)["mean_delta"] > 0.0
        for control in PRIMARY_CONTROLS
        for metric in PRIMARY_METRICS
    )
    if quality_identity_gate and identity_motion_gate:
        decision = "fresh128_quality_identity_motion_confirmed"
    elif quality_identity_gate and dynamic_nonregression:
        decision = "fresh128_quality_identity_confirmed"
    elif identity_motion_gate:
        decision = "fresh128_identity_motion_confirmed"
    elif directional:
        decision = "fresh128_directional_only"
    else:
        decision = "fresh128_rccp_not_confirmed"

    per_prompt_metrics = {
        method: [
            {"prompt_index": prompt, **rows[(method, prompt)]}
            for prompt in range(PROMPT_COUNT)
        ]
        for method in METHODS
    }
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "profile_contract": "v177",
        "prompt_count": PROMPT_COUNT,
        "methods": list(METHODS),
        "comparisons": comparisons,
        "per_prompt_metrics": per_prompt_metrics,
        "quality_identity_gate": quality_identity_gate,
        "identity_motion_gate": identity_motion_gate,
        "dynamic_nonregression_gate": dynamic_nonregression,
        "decision": decision,
        "targeted_review": _targeted_review(manifest, rows),
        "manual_review_required_for_decision": False,
        "claim_boundary": (
            "v180 tests fresh-prompt transfer of the exact v177 five-head map "
            "after a passing v178 causal membership gate. It does not establish "
            "cross-model transfer or scene-switching behavior."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v180 Fresh128 Paired Analysis",
        "",
        f"Decision: `{report['decision']}`",
        f"Quality + identity gate: `{report['quality_identity_gate']}`",
        f"Identity + motion gate: `{report['identity_motion_gate']}`",
        f"Dynamic non-regression: `{report['dynamic_nonregression_gate']}`",
        "",
        "| Control | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["control"] not in PRIMARY_CONTROLS or row["metric"] not in PRIMARY_METRICS:
            continue
        lines.append(
            f"| {row['control']} | {row['metric']} | {row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {row['q_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The review queue is capped at six metric-conflict cases and is diagnostic only.",
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
    print(f"[v180-paired] decision={report['decision']} output={args.output}")


if __name__ == "__main__":
    main()
