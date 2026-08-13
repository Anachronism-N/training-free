#!/usr/bin/env python3
"""Paired 2x2 and Shapley attribution for the v177-selected head set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base
from analyze_v178_paired_metrics import metric_runtime_fingerprint
from prepare_v178_rccp_holdout import sha256
from prepare_v179_head_attribution import GENERATED_METHODS, METHODS


PRIMARY_METRICS = ("official_quality_score", "identity_background")


def _load_reused(path: Path, expected_sha: str, prompt_count: int) -> dict:
    if not path.is_file() or sha256(path) != expected_sha:
        raise ValueError("v178 reused prompt metrics are absent or hash-drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("membership_hypothesis_gate") is not True
        or payload.get("decision") != "advance_rccp_membership_to_broader_generation"
        or int(payload.get("prompt_count", -1)) != prompt_count
    ):
        raise ValueError("v179 cannot reuse a non-passing v178 result")
    result = {}
    rows = payload.get("per_prompt_metrics") or {}
    for method in ("all_recent", "matched"):
        method_rows = rows.get(method) or ()
        if len(method_rows) != prompt_count:
            raise ValueError(f"v178 reused prompt rows incomplete: {method}")
        for prompt, row in enumerate(method_rows):
            if int(row.get("prompt_index", -1)) != prompt:
                raise ValueError(f"v178 reused prompt order drift: {method}")
            result[(method, prompt)] = {
                metric: float(row[metric]) for metric in base.METRICS
            }
    return result


def _contrast_row(
    name: str,
    metric: str,
    deltas: list[float],
    seed_offset: int,
) -> dict:
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "contrast": name,
        "metric": metric,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(
            deltas, seed=1792026 + seed_offset
        ),
        "p_value": base.sign_p(deltas),
        "per_prompt_delta": deltas,
    }


def analyze(
    manifest: dict,
    summary: dict,
    parts_root: Path,
    v178_paired_path: Path,
    v178_paired_sha256: str,
) -> dict:
    if (
        manifest.get("experiment")
        != "v179_rccp_head_attribution_vbench_incremental"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("generation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid v179 incremental comparison manifest")
    methods = tuple(row["key"] for row in manifest.get("methods") or ())
    prompt_count = int(manifest.get("prompt_count", -1))
    if methods != GENERATED_METHODS or prompt_count != 32:
        raise ValueError("v179 incremental method or prompt grid drift")
    if tuple(summary.get("methods") or {}) != methods or summary.get("missing"):
        raise ValueError("v179 incremental metric summary is incomplete")

    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)
    rows.update(
        _load_reused(v178_paired_path, v178_paired_sha256, prompt_count)
    )

    contrasts = []
    names = (
        "matched_total",
        "top1_alone",
        "remainder_alone",
        "top1_given_remainder",
        "remainder_given_top1",
        "factor_interaction",
        "top1_shapley",
        "remainder_shapley",
    )
    for metric_index, metric in enumerate(base.METRICS):
        values_by_name = {name: [] for name in names}
        for prompt in range(prompt_count):
            y00 = rows[("all_recent", prompt)][metric]
            y10 = rows[("profile_top1_only", prompt)][metric]
            y01 = rows[("profile_remainder", prompt)][metric]
            y11 = rows[("matched", prompt)][metric]
            values_by_name["matched_total"].append(y11 - y00)
            values_by_name["top1_alone"].append(y10 - y00)
            values_by_name["remainder_alone"].append(y01 - y00)
            values_by_name["top1_given_remainder"].append(y11 - y01)
            values_by_name["remainder_given_top1"].append(y11 - y10)
            values_by_name["factor_interaction"].append(y11 - y10 - y01 + y00)
            values_by_name["top1_shapley"].append(
                0.5 * ((y10 - y00) + (y11 - y01))
            )
            values_by_name["remainder_shapley"].append(
                0.5 * ((y01 - y00) + (y11 - y10))
            )
        for contrast_index, name in enumerate(names):
            contrasts.append(
                _contrast_row(
                    name,
                    metric,
                    values_by_name[name],
                    metric_index * 101 + contrast_index,
                )
            )
    primary_family = [
        row
        for row in contrasts
        if row["contrast"] in {"top1_shapley", "remainder_shapley"}
        and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(primary_family)
    for row in contrasts:
        if "q_value" not in row:
            row["q_value"] = None
            row["inferential_role"] = "descriptive"
        else:
            row["inferential_role"] = "preregistered_primary"

    def rows_for(name: str) -> list[dict]:
        return [
            row
            for row in contrasts
            if row["contrast"] == name and row["metric"] in PRIMARY_METRICS
        ]

    top_primary = rows_for("top1_shapley")
    remainder_primary = rows_for("remainder_shapley")
    top_directional = all(row["mean_delta"] > 0.0 for row in top_primary)
    remainder_directional = all(
        row["mean_delta"] > 0.0 for row in remainder_primary
    )
    top_confirmatory = all(
        row["bootstrap_ci95"][0] > 0.0 and row["q_value"] <= 0.10
        for row in top_primary
    )
    remainder_confirmatory = all(
        row["bootstrap_ci95"][0] > 0.0 and row["q_value"] <= 0.10
        for row in remainder_primary
    )
    if top_confirmatory and remainder_confirmatory:
        decision = "distributed_selected_set_confirmed"
    elif top_directional and remainder_directional:
        decision = "distributed_selected_set_directional_only"
    elif top_directional and not remainder_directional:
        decision = "profile_top1_dominated"
    elif remainder_directional and not top_directional:
        decision = "profile_remainder_dominated"
    else:
        decision = "head_attribution_inconclusive"

    contribution_share = {}
    for metric in PRIMARY_METRICS:
        total = next(
            row["mean_delta"]
            for row in contrasts
            if row["contrast"] == "matched_total" and row["metric"] == metric
        )
        top = next(
            row["mean_delta"]
            for row in contrasts
            if row["contrast"] == "top1_shapley" and row["metric"] == metric
        )
        remainder = next(
            row["mean_delta"]
            for row in contrasts
            if row["contrast"] == "remainder_shapley" and row["metric"] == metric
        )
        contribution_share[metric] = {
            "matched_total_mean_delta": total,
            "top1_shapley_mean_delta": top,
            "remainder_shapley_mean_delta": remainder,
            "top1_fraction_of_total": None if abs(total) < 1e-12 else top / total,
            "remainder_fraction_of_total": (
                None if abs(total) < 1e-12 else remainder / total
            ),
            "additivity_error": (top + remainder) - total,
        }
    return {
        "version": 1,
        "experiment": "v179_rccp_head_attribution",
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "methods": list(METHODS),
        "profile_top1_head": manifest["profile_top1_head"],
        "factorial_design": manifest["factorial_design"],
        "contrasts": contrasts,
        "primary_metrics": list(PRIMARY_METRICS),
        "top1_directional_positive": top_directional,
        "remainder_directional_positive": remainder_directional,
        "top1_confirmatory_positive": top_confirmatory,
        "remainder_confirmatory_positive": remainder_confirmatory,
        "contribution_share": contribution_share,
        "decision": decision,
        "claim_boundary": (
            "v179 attributes the already validated v178 matched-vs-recent "
            "effect between the profile-strongest head and the remaining "
            "selected set. It does not independently validate individual "
            "heads inside the remainder or cross-model transfer."
        ),
    }


def render(report: dict) -> str:
    top = report["profile_top1_head"]
    lines = [
        "# v179 RCCP Head Attribution",
        "",
        f"Top profile head: L{top['layer']}H{top['head']}",
        f"Decision: {report['decision']}",
        "",
        "| Contrast | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    displayed = {
        "matched_total",
        "top1_shapley",
        "remainder_shapley",
        "factor_interaction",
    }
    for row in report["contrasts"]:
        if row["contrast"] not in displayed:
            continue
        q_value = (
            f"{row['q_value']:.4g}"
            if row["q_value"] is not None
            else "descriptive"
        )
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {q_value} |"
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
    reused = manifest["reused_metric_cells"]["matched"]
    report = analyze(
        manifest,
        summary,
        args.parts_root,
        Path(reused["v178_paired_result"]),
        reused["v178_paired_result_sha256"],
    )
    observed_runtime = metric_runtime_fingerprint(
        args.parts_root,
        GENERATED_METHODS,
        tuple(summary["dimensions"]),
    )
    required_runtime = manifest.get("required_metric_runtime_fingerprint") or {}
    if observed_runtime["sha256"] != required_runtime.get("sha256"):
        raise ValueError(
            "v179 metric runtime differs from reused v178 prompt metrics"
        )
    report["metric_runtime_fingerprint"] = observed_runtime
    report["input_provenance"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "incremental_metric_summary": str(args.summary.resolve()),
        "incremental_metric_summary_sha256": sha256(args.summary),
        "v178_paired_result": reused["v178_paired_result"],
        "v178_paired_result_sha256": reused["v178_paired_result_sha256"],
        "parts_root": str(args.parts_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(f"[v179-attribution] decision={report['decision']} output={args.output}")


if __name__ == "__main__":
    main()
