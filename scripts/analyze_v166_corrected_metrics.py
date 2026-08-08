#!/usr/bin/env python3
"""Analyze v166 VBench results without duplicate metric weighting."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import analyze_v165_final_decision as base
from prepare_v166_vbench_comparison import DIMENSIONS, METHODS, PROMPT_COUNT
from vbench_quality_contract import (
    EXCLUSIVE_GROUPS,
    OFFICIAL_CONSTANTS_SOURCE,
    QUALITY_DIMENSIONS,
    exclusive_scores,
    official_quality_score,
)


SF = "sf_native"
DIRECTION_MATCH = "ours_middle10_reservoir2_directionmatch1"
PRIMARY = "ours_middle10_reservoir2_multiscalemotion1"
REFERENCES = tuple(method for method in METHODS if method != PRIMARY)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "v166_multiscale_motion_moviebench16" / "full8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench-parts-root",
        type=Path,
        default=run_root / "metrics" / "vbench_long_parts",
    )
    parser.add_argument(
        "--vbench-summary",
        type=Path,
        default=run_root / "metrics" / "vbench_core9_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "analysis" / "v166_corrected_metrics.json",
    )
    return parser.parse_args()


def validate_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("methods") or {}
    dimensions = tuple(payload.get("dimensions") or ())
    if (
        tuple(rows) != METHODS
        or dimensions != DIMENSIONS
        or payload.get("missing")
    ):
        raise ValueError("v166 VBench summary violates the frozen grid")
    for method, row in rows.items():
        if set(row) != set(DIMENSIONS):
            raise ValueError(f"{method}: incomplete VBench summary")
        for dimension in DIMENSIONS:
            base.finite(row[dimension], name=f"summary:{method}:{dimension}")
    return payload


def load_prompt_rows(parts_root: Path, summary: dict) -> tuple[dict, dict]:
    rows = {
        (method, prompt): {}
        for method in METHODS
        for prompt in range(PROMPT_COUNT)
    }
    scales = {}
    for method in METHODS:
        scales[method] = {}
        for dimension in DIMENSIONS:
            clips = base.load_dimension(
                parts_root / method / dimension / "results.json",
                dimension,
            )
            raw_values = [
                value
                for prompt in range(PROMPT_COUNT)
                for value in clips[prompt]
            ]
            summary_value = base.finite(
                summary["methods"][method][dimension],
                name=f"summary:{method}:{dimension}",
            )
            factor = base.scale_factor(
                statistics.fmean(raw_values),
                summary_value,
                name=f"{method}:{dimension}",
            )
            scales[method][dimension] = factor
            for prompt in range(PROMPT_COUNT):
                rows[(method, prompt)][dimension] = (
                    factor * statistics.fmean(clips[prompt])
                )
            reconstructed = statistics.fmean(
                rows[(method, prompt)][dimension]
                for prompt in range(PROMPT_COUNT)
            )
            if not math.isclose(
                reconstructed,
                summary_value,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{method}:{dimension}: prompt/detail mean mismatch"
                )
    return rows, scales


def audit_duplicate_clip_metric(parts_root: Path) -> dict:
    mismatches = []
    checked = 0
    for method in METHODS:
        overall = base.load_dimension(
            parts_root / method / "overall_consistency" / "results.json",
            "overall_consistency",
        )
        temporal = base.load_dimension(
            parts_root / method / "temporal_style" / "results.json",
            "temporal_style",
        )
        for prompt in range(PROMPT_COUNT):
            if len(overall[prompt]) != len(temporal[prompt]):
                raise ValueError("duplicate metric clip coverage mismatch")
            for clip, (left, right) in enumerate(
                zip(overall[prompt], temporal[prompt])
            ):
                checked += 1
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                    mismatches.append(
                        {
                            "method": method,
                            "prompt_index": prompt,
                            "clip_index": clip,
                            "overall_consistency": left,
                            "temporal_style": right,
                        }
                    )
    return {
        "pair": ["overall_consistency", "temporal_style"],
        "checked_clip_pairs": checked,
        "exact_within_1e-12": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "action": (
            "retain overall_consistency once as semantic_alignment and "
            "exclude temporal_style from diagnostic group averages"
        ),
    }


def corrected_prompt_rows(raw: dict) -> dict:
    return {key: exclusive_scores(row) for key, row in raw.items()}


def quality_prompt_rows(raw: dict) -> dict:
    return {
        key: {"official_quality_score": official_quality_score(row)}
        for key, row in raw.items()
    }


def aggregate(rows: dict) -> dict[str, dict[str, float]]:
    return {
        method: {
            metric: statistics.fmean(
                rows[(method, prompt)][metric]
                for prompt in range(PROMPT_COUNT)
            )
            for metric in rows[(method, 0)]
        }
        for method in METHODS
    }


def dynamic_win_tie_loss(raw: dict, *, reference: str) -> dict:
    deltas = [
        raw[(PRIMARY, prompt)]["dynamic_degree"]
        - raw[(reference, prompt)]["dynamic_degree"]
        for prompt in range(PROMPT_COUNT)
    ]
    return {
        "candidate": PRIMARY,
        "reference": reference,
        "wins": sum(value > 1e-12 for value in deltas),
        "ties": sum(abs(value) <= 1e-12 for value in deltas),
        "losses": sum(value < -1e-12 for value in deltas),
        "mean_delta": statistics.fmean(deltas),
        "per_prompt": [
            {"prompt_index": prompt, "delta": value}
            for prompt, value in enumerate(deltas)
        ],
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v166 Corrected VBench Metrics",
        "",
        "This report uses mutually exclusive diagnostic groups and the",
        "official VBench Quality Score.",
        "",
        f"Duplicate clip pairs checked: {report['duplicate_metric_audit']['checked_clip_pairs']}",
        f"Exact duplicate: **{report['duplicate_metric_audit']['exact_within_1e-12']}**",
        "",
        "| Method | Quality Score | Identity/background | Temporal mechanics | "
        "Semantic | Visual | Dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["quality_ranking"]:
        groups = report["aggregate_exclusive_scores"][method]
        quality = report["aggregate_official_quality_score"][method][
            "official_quality_score"
        ]
        lines.append(
            f"| {method} | {quality:.4f} | "
            f"{groups['identity_background']:.6f} | "
            f"{groups['temporal_mechanics']:.6f} | "
            f"{groups['semantic_alignment']:.6f} | "
            f"{groups['visual_quality']:.6f} | "
            f"{groups['dynamic_degree']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Paired comparisons use the same 16 prompts and bootstrap the",
            "prompt-level mean. This remains development evidence, not a",
            "held-out paper result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = validate_summary(args.vbench_summary)
    raw, scales = load_prompt_rows(
        args.vbench_parts_root,
        summary,
    )
    duplicate = audit_duplicate_clip_metric(args.vbench_parts_root)
    if not duplicate["exact_within_1e-12"]:
        raise ValueError(
            "overall_consistency and temporal_style are not exact duplicates"
        )
    corrected = corrected_prompt_rows(raw)
    quality = quality_prompt_rows(raw)
    aggregate_corrected = aggregate(corrected)
    aggregate_quality = aggregate(quality)
    paired_exclusive = {
        reference: base.paired_comparison(
            corrected,
            candidate=PRIMARY,
            reference=reference,
            seed_offset=index,
        )
        for index, reference in enumerate(REFERENCES)
    }
    paired_quality = {
        reference: base.paired_comparison(
            quality,
            candidate=PRIMARY,
            reference=reference,
            seed_offset=100 + index,
        )["official_quality_score"]
        for index, reference in enumerate(REFERENCES)
    }
    ranking = sorted(
        METHODS,
        key=lambda method: aggregate_quality[method][
            "official_quality_score"
        ],
        reverse=True,
    )
    report = {
        "version": 1,
        "experiment": "v166_corrected_metric_analysis",
        "duplicate_metric_audit": duplicate,
        "exclusive_groups": EXCLUSIVE_GROUPS,
        "official_quality_dimensions": QUALITY_DIMENSIONS,
        "official_constants_source": OFFICIAL_CONSTANTS_SOURCE,
        "vbench_detail_scales": scales,
        "aggregate_exclusive_scores": aggregate_corrected,
        "aggregate_official_quality_score": aggregate_quality,
        "quality_ranking": ranking,
        "primary": PRIMARY,
        "paired_exclusive": paired_exclusive,
        "paired_official_quality": paired_quality,
        "dynamic_win_tie_loss": {
            DIRECTION_MATCH: dynamic_win_tie_loss(
                raw,
                reference=DIRECTION_MATCH,
            ),
            SF: dynamic_win_tie_loss(raw, reference=SF),
        },
        "interpretation": (
            "use mutually exclusive diagnostic groups and the official "
            "Quality Score; never average overall_consistency and "
            "temporal_style into two separate axes for this custom-prompt run"
        ),
        "claim_boundary": (
            "the 16-prompt suite is adaptive development evidence only"
        ),
        "inputs": {
            "vbench_parts_root": str(args.vbench_parts_root.resolve()),
            "vbench_summary": str(args.vbench_summary.resolve()),
            "vbench_summary_sha256": base.sha256(args.vbench_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    print(
        json.dumps(
            {
                "duplicate_metric_audit": duplicate,
                "quality_ranking": ranking,
                "primary_vs_directionmatch": paired_quality[DIRECTION_MATCH],
                "primary_vs_sf": paired_quality[SF],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
