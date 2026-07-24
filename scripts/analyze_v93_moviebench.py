#!/usr/bin/env python3
"""Analyze v93 MovieBench main-table and head-classification experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


CORE_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m2_drift_slope",
    "m7_background_consistency",
    "composite",
)
QUALITY_METRICS = (
    "m1_dino_consistency",
    "m1_min_stability",
    "m7_background_consistency",
    "composite",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
MAIN_METHODS = (
    "sf_native",
    "pf",
    "echo_pc",
    "v78",
    "pf_binary_read_v78",
    "prompt_pfcount_read_v78",
    "prompt_kmeans_read_v78",
    "veil_priority_b005",
)
HEAD_METHODS = (
    "pf",
    "pf_binary_read",
    "prompt_pfcount_read",
    "prompt_kmeans_read",
    "v78",
    "pf_binary_read_v78",
    "prompt_pfcount_read_v78",
    "prompt_kmeans_read_v78",
    "prompt_replica_read_v78",
    "prompt_consensus_read_v78",
    "prompt_inverse_read_v78",
    "prompt_random_read_v78",
    "remote_read_v78",
    "role_score_read_v78",
    "pf_read_prompt_priority",
    "prompt_read_prompt_priority",
)
MAIN_COMPARISONS = (
    ("sf_to_pf", "sf_native", "pf"),
    ("pf_to_echo", "pf", "echo_pc"),
    ("pf_to_v78", "pf", "v78"),
    ("pf_to_pf_binary_transition", "pf", "pf_binary_read_v78"),
    ("pf_binary_to_prompt_membership", "pf_binary_read_v78", "prompt_pfcount_read_v78"),
    ("prompt_membership_to_kmeans", "prompt_pfcount_read_v78", "prompt_kmeans_read_v78"),
    ("v78_to_veil_priority", "v78", "veil_priority_b005"),
)
HEAD_COMPARISONS = (
    ("pf_tri_to_binary_read", "pf", "pf_binary_read"),
    ("binary_to_prompt_read", "pf_binary_read", "prompt_pfcount_read"),
    ("prompt_read_add_transition", "prompt_pfcount_read", "prompt_pfcount_read_v78"),
    ("pf_binary_transition_to_prompt", "pf_binary_read_v78", "prompt_pfcount_read_v78"),
    ("prompt_membership_to_kmeans", "prompt_pfcount_read_v78", "prompt_kmeans_read_v78"),
    ("inverse_to_prompt", "prompt_inverse_read_v78", "prompt_pfcount_read_v78"),
    ("random_to_prompt", "prompt_random_read_v78", "prompt_pfcount_read_v78"),
    ("replica_to_primary", "prompt_replica_read_v78", "prompt_pfcount_read_v78"),
    ("consensus_to_primary", "prompt_consensus_read_v78", "prompt_pfcount_read_v78"),
    ("remote_to_prompt", "remote_read_v78", "prompt_pfcount_read_v78"),
    ("role_score_to_prompt", "role_score_read_v78", "prompt_pfcount_read_v78"),
    ("prompt_priority_on_pf_read", "v78", "pf_read_prompt_priority"),
    (
        "prompt_priority_on_prompt_read",
        "prompt_pfcount_read_v78",
        "prompt_read_prompt_priority",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "head32"), required=True)
    parser.add_argument("--comprehensive", required=True, type=Path)
    parser.add_argument("--temporal-jump", required=True, type=Path)
    parser.add_argument("--vbench", required=True, type=Path)
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument("--label-manifest", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_temporal(path: Path) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = Path(row["video"]).parent.name
            score = _finite(row.get("temporal_jump"))
            if score is not None:
                values[method].append(score)
    return {
        method: {
            "count": len(scores),
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
        }
        for method, scores in sorted(values.items())
    }


def _load_traces(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in payload.get("summaries", []):
        name = Path(str(summary.get("trace", ""))).name
        name = re.sub(r"\.transition\.jsonl$", "", name)
        name = re.sub(r"\.shard\d+$", "", name)
        grouped[name].append(summary)
    output = {}
    for method, rows in grouped.items():
        total = sum(int(row.get("total") or 0) for row in rows)
        accepted = sum(int(row.get("accepted") or 0) for row in rows)
        output[method] = {
            "trace_count": len(rows),
            "status": (
                "nominal"
                if all(row.get("status") == "nominal" for row in rows)
                else "failed"
            ),
            "accepted": accepted,
            "total": total,
            "acceptance_rate": accepted / total if total else None,
            "traces": [row.get("trace") for row in rows],
        }
    return output


def _metric_row(
    method: str,
    comprehensive: dict[str, Any],
    temporal: dict[str, dict[str, float | int]],
    vbench: dict[str, Any],
) -> dict[str, Any]:
    row = {"method": method}
    for metric in CORE_METRICS:
        row[metric] = _finite(comprehensive[method].get(metric))
    row["temporal_jump"] = _finite(temporal.get(method, {}).get("mean"))
    for dimension, value in (vbench.get(method) or {}).items():
        row[f"vbench_{dimension}"] = _finite(value)
    return row


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    metrics = sorted((set(left) | set(right)) - {"method"})
    output = {}
    for metric in metrics:
        left_value = _finite(left.get(metric))
        right_value = _finite(right.get(metric))
        output[metric] = (
            right_value - left_value
            if left_value is not None and right_value is not None
            else None
        )
    return output


def _wins(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    deltas = _delta(baseline, candidate)
    available = [metric for metric in QUALITY_METRICS if deltas.get(metric) is not None]
    won = [metric for metric in available if float(deltas[metric]) > 0]
    return {
        "available": len(available),
        "wins": len(won),
        "won_metrics": won,
        "majority": bool(available) and len(won) > len(available) / 2,
    }


def analyze(
    *,
    mode: str,
    comprehensive_payload: dict[str, Any],
    temporal: dict[str, dict[str, float | int]],
    vbench_payload: dict[str, Any],
    traces: dict[str, dict[str, Any]],
    label_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = MAIN_METHODS if mode == "main" else HEAD_METHODS
    comparisons = MAIN_COMPARISONS if mode == "main" else HEAD_COMPARISONS
    comprehensive = comprehensive_payload.get("per_method") or {}
    vbench = vbench_payload.get("methods") or {}
    missing = {
        "comprehensive": sorted(set(expected) - set(comprehensive)),
        "temporal": sorted(set(expected) - set(temporal)),
        "vbench": sorted(set(expected) - set(vbench)),
    }
    if any(missing.values()):
        raise ValueError(f"incomplete v93 metrics: {missing}")

    rows = {
        method: _metric_row(method, comprehensive, temporal, vbench)
        for method in expected
    }
    factors = {
        name: {
            "baseline": baseline,
            "candidate": candidate,
            "delta": _delta(rows[baseline], rows[candidate]),
        }
        for name, baseline, candidate in comparisons
    }
    ranking = sorted(
        rows.values(),
        key=lambda row: (
            row.get("m1_dino_consistency") is not None,
            row.get("m1_dino_consistency") or -math.inf,
        ),
        reverse=True,
    )

    screen = None
    if mode == "head32":
        candidate = rows["prompt_pfcount_read_v78"]
        versus_inverse = _wins(rows["prompt_inverse_read_v78"], candidate)
        versus_random = _wins(rows["prompt_random_read_v78"], candidate)
        dino = candidate["m1_dino_consistency"]
        pf_dino = rows["pf_binary_read_v78"]["m1_dino_consistency"]
        replica_dino = rows["prompt_replica_read_v78"]["m1_dino_consistency"]
        competitive = (
            dino is not None and pf_dino is not None and dino - pf_dino >= -0.005
        )
        reproducible = (
            dino is not None
            and replica_dino is not None
            and abs(dino - replica_dino) <= 0.01
        )
        screen = {
            "candidate": "prompt_pfcount_read_v78",
            "versus_inverse": versus_inverse,
            "versus_random": versus_random,
            "within_0p005_pf_binary_dino": competitive,
            "replica_within_0p01_dino": reproducible,
            "automated_screen_passed": (
                versus_inverse["majority"]
                and versus_random["majority"]
                and competitive
                and reproducible
            ),
        }

    return {
        "mode": mode,
        "methods": rows,
        "ranking_by_dino": ranking,
        "factor_comparisons": factors,
        "classification_screen": screen,
        "transition_diagnostics": traces,
        "label_manifest": label_manifest,
        "interpretation_guard": (
            "The automated screen is exploratory. Freeze prompt-wise blind human "
            "review before using these metrics for a paper claim; dynamic_degree "
            "is reported as behavior and is not treated as a quality win."
        ),
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.5f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# v93 MovieBench {payload['mode']} analysis",
        "",
        "## Ranking by DINO",
        "",
        "| Method | DINO | Min DINO | BG | Composite | Jump | VBench subject | VBench BG | Aesthetic | Imaging | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ranking_by_dino"]:
        lines.append(
            f"| {row['method']} | {_fmt(row.get('m1_dino_consistency'))} | "
            f"{_fmt(row.get('m1_min_stability'))} | "
            f"{_fmt(row.get('m7_background_consistency'))} | "
            f"{_fmt(row.get('composite'))} | "
            f"{_fmt(row.get('temporal_jump'))} | "
            f"{_fmt(row.get('vbench_subject_consistency'))} | "
            f"{_fmt(row.get('vbench_background_consistency'))} | "
            f"{_fmt(row.get('vbench_aesthetic_quality'))} | "
            f"{_fmt(row.get('vbench_imaging_quality'))} | "
            f"{_fmt(row.get('vbench_dynamic_degree'))} |"
        )
    lines.extend(
        [
            "",
            "## Factor comparisons",
            "",
            "| Factor | Baseline | Candidate | Delta DINO | Delta min DINO | Delta BG | Delta jump | Delta VBench subject |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, factor in payload["factor_comparisons"].items():
        delta = factor["delta"]
        lines.append(
            f"| {name} | {factor['baseline']} | {factor['candidate']} | "
            f"{_fmt(delta.get('m1_dino_consistency'))} | "
            f"{_fmt(delta.get('m1_min_stability'))} | "
            f"{_fmt(delta.get('m7_background_consistency'))} | "
            f"{_fmt(delta.get('temporal_jump'))} | "
            f"{_fmt(delta.get('vbench_subject_consistency'))} |"
        )
    if payload["classification_screen"] is not None:
        screen = payload["classification_screen"]
        lines.extend(
            [
                "",
                "## Classification screen",
                "",
                f"- Candidate: `{screen['candidate']}`",
                f"- Majority over inverse: `{screen['versus_inverse']['majority']}` "
                f"({screen['versus_inverse']['wins']}/{screen['versus_inverse']['available']})",
                f"- Majority over random: `{screen['versus_random']['majority']}` "
                f"({screen['versus_random']['wins']}/{screen['versus_random']['available']})",
                f"- Within 0.005 DINO of PF-binary: `{screen['within_0p005_pf_binary_dino']}`",
                f"- Replica within 0.01 DINO: `{screen['replica_within_0p01_dino']}`",
                f"- Automated screen passed: `{screen['automated_screen_passed']}`",
            ]
        )
    lines.extend(["", "## Guard", "", payload["interpretation_guard"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comprehensive = json.loads(args.comprehensive.read_text(encoding="utf-8"))
    temporal = _load_temporal(args.temporal_jump)
    vbench = json.loads(args.vbench.read_text(encoding="utf-8"))
    traces = _load_traces(args.trace_summary)
    label_manifest = (
        json.loads(args.label_manifest.read_text(encoding="utf-8"))
        if args.label_manifest is not None and args.label_manifest.is_file()
        else None
    )
    payload = analyze(
        mode=args.mode,
        comprehensive_payload=comprehensive,
        temporal=temporal,
        vbench_payload=vbench,
        traces=traces,
        label_manifest=label_manifest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_md)
    print(f"[v93-analysis] mode={args.mode} methods={len(payload['methods'])}")


if __name__ == "__main__":
    main()
