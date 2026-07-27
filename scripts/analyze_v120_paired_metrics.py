#!/usr/bin/env python3
"""Paired, per-prompt analysis for v120 VBench and auxiliary metrics.

This script consumes existing result JSON files. It never regenerates videos
or recomputes model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VIDEO_INDEX = re.compile(r"^(\d+)-(\d+)(?:_|$)")
LOWER_IS_BETTER = {
    "aux.m1_first_last_gap",
    "aux.m3_motion_smoothness",
    "aux.m5_max_flicker",
    "aux.m5_temporal_flickering",
    "aux.m6_clip_text_std",
    "aux.m7_background_drift",
    "aux.m8_loop_score",
}
PRIMARY_METRIC_ORDER = (
    "vbench.subject_consistency",
    "vbench.subject_consistency.inclip",
    "vbench.subject_consistency.clip2clip",
    "vbench.background_consistency",
    "vbench.background_consistency.inclip",
    "vbench.background_consistency.clip2clip",
    "vbench.aesthetic_quality",
    "vbench.imaging_quality",
    "vbench.motion_smoothness",
    "vbench.temporal_flickering",
    "vbench.dynamic_degree",
    "aux.m1_dino_consistency",
    "aux.m2_drift_slope",
    "aux.m5_temporal_flickering",
    "aux.m6_clip_text_alignment",
    "aux.m7_background_consistency",
    "aux.m8_loop_score",
    "aux.composite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vbench",
        action="append",
        default=[],
        metavar="METHOD=RESULTS_JSON",
        help="raw VBench-Long result; repeat once per method",
    )
    parser.add_argument(
        "--comprehensive",
        type=Path,
        help="merged evaluate_comprehensive.py JSON with per_video rows",
    )
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--expected-prompts", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--permutation-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_binding(value: str) -> tuple[str, Path]:
    method, separator, raw_path = value.partition("=")
    if not separator or not method.strip() or not raw_path.strip():
        raise ValueError(f"expected METHOD=RESULTS_JSON, got {value!r}")
    return method.strip(), Path(raw_path).expanduser().resolve()


def prompt_index(video_path: str) -> int:
    path = Path(video_path)
    for part in reversed(path.parts):
        match = VIDEO_INDEX.match(Path(part).stem)
        if match is not None:
            sample_index = int(match.group(2))
            if sample_index != 0:
                raise ValueError(
                    f"only sample index 0 is supported: {video_path}"
                )
            return int(match.group(1))
    raise ValueError(f"cannot recover prompt index from {video_path!r}")


def _detail_lists(value: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(value, (list, tuple)):
        return []
    candidates = []
    for item in value[1:]:
        if (
            isinstance(item, list)
            and item
            and all(isinstance(row, dict) for row in item)
        ):
            candidates.append(item)
    return candidates


def _best_per_video_rows(value: Any) -> list[dict[str, Any]]:
    candidates = _detail_lists(value)
    if not candidates:
        return []

    ranked: list[tuple[int, int, list[dict[str, Any]]]] = []
    for rows in candidates:
        counts: dict[int, int] = defaultdict(int)
        usable = []
        for row in rows:
            score = finite_number(row.get("video_results"))
            video_path = row.get("video_path")
            if score is None or not isinstance(video_path, str):
                continue
            try:
                index = prompt_index(video_path)
            except ValueError:
                continue
            counts[index] += 1
            usable.append(row)
        if counts:
            duplicate_count = sum(count - 1 for count in counts.values())
            ranked.append((len(counts), -duplicate_count, usable))
    if not ranked:
        return []
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def extract_vbench(path: Path) -> dict[str, dict[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"VBench result is not an object: {path}")

    metrics: dict[str, dict[int, float]] = {}
    for dimension, value in payload.items():
        rows = _best_per_video_rows(value)
        if not rows:
            continue
        grouped: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            video_path = row.get("video_path")
            if not isinstance(video_path, str):
                continue
            index = prompt_index(video_path)
            fields = {"": row.get("video_results")}
            for field in (
                "inclip_score",
                "clip2clip_score",
                "mapped_clip2clip_score",
            ):
                if field in row:
                    fields[f".{field.removesuffix('_score')}"] = row[field]
            for suffix, raw_score in fields.items():
                score = finite_number(raw_score)
                if score is not None:
                    grouped[f"vbench.{dimension}{suffix}"][index].append(score)
        for metric, by_index in grouped.items():
            metrics[metric] = {
                index: float(np.mean(values))
                for index, values in by_index.items()
            }
    if not metrics:
        raise ValueError(f"VBench result has no per-video observations: {path}")
    return metrics


def extract_comprehensive(
    path: Path,
) -> dict[str, dict[str, dict[int, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_video")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"comprehensive result has no per_video rows: {path}")

    output: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen: set[tuple[str, int]] = set()
    for key, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid per_video row {key!r}")
        method = row.get("method")
        index = row.get("prompt_index")
        values = row.get("metrics")
        if (
            not isinstance(method, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(values, dict)
        ):
            raise ValueError(f"invalid comprehensive binding in {key!r}")
        pair = (method, index)
        if pair in seen:
            raise ValueError(f"duplicate comprehensive row {pair}")
        seen.add(pair)
        for metric, value in values.items():
            score = finite_number(value)
            if score is not None:
                output[method][f"aux.{metric}"][index] = score
    return {
        method: {metric: dict(by_index) for metric, by_index in metrics.items()}
        for method, metrics in output.items()
    }


def merge_observations(
    *,
    vbench: dict[str, dict[str, dict[int, float]]],
    comprehensive: dict[str, dict[str, dict[int, float]]],
) -> dict[str, dict[str, dict[int, float]]]:
    output: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for source in (vbench, comprehensive):
        for method, metrics in source.items():
            overlap = set(output[method]) & set(metrics)
            if overlap:
                raise ValueError(
                    f"duplicate metric sources for {method}: {sorted(overlap)}"
                )
            output[method].update(metrics)
    return dict(output)


def metric_direction(metric: str) -> int:
    return -1 if metric in LOWER_IS_BETTER else 1


def _comparison_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def paired_statistics(
    candidate: dict[int, float],
    reference: dict[int, float],
    *,
    direction: int,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    indices = sorted(set(candidate) & set(reference))
    if len(indices) < 2:
        raise ValueError(f"{label}: fewer than two paired observations")
    candidate_values = np.asarray([candidate[index] for index in indices])
    reference_values = np.asarray([reference[index] for index in indices])
    raw_delta = candidate_values - reference_values
    improvement = raw_delta * direction

    rng = _comparison_rng(seed, label)
    sampled_indices = rng.integers(
        0,
        len(indices),
        size=(bootstrap_samples, len(indices)),
    )
    bootstrap_means = raw_delta[sampled_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    observed = abs(float(improvement.mean()))
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(permutation_samples, len(indices)),
    )
    null_means = np.abs((improvement * signs).mean(axis=1))
    permutation_p = float(
        (np.count_nonzero(null_means >= observed) + 1)
        / (permutation_samples + 1)
    )

    epsilon = 1e-12
    wins = int(np.count_nonzero(improvement > epsilon))
    losses = int(np.count_nonzero(improvement < -epsilon))
    ties = len(indices) - wins - losses
    return {
        "n": len(indices),
        "prompt_indices": indices,
        "candidate_mean": float(candidate_values.mean()),
        "reference_mean": float(reference_values.mean()),
        "raw_mean_delta": float(raw_delta.mean()),
        "raw_median_delta": float(np.median(raw_delta)),
        "improvement_mean": float(improvement.mean()),
        "bootstrap_95_ci_raw_delta": [
            float(ci_low),
            float(ci_high),
        ],
        "paired_randomization_p": permutation_p,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def analyze(
    observations: dict[str, dict[str, dict[int, float]]],
    *,
    references: list[str],
    candidates: list[str],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    expected_prompts: int | None = None,
) -> dict[str, Any]:
    requested = set(references) | set(candidates)
    missing = sorted(requested - set(observations))
    if missing:
        raise ValueError(f"methods absent from metric inputs: {missing}")
    if expected_prompts is not None:
        if expected_prompts <= 0:
            raise ValueError("expected_prompts must be positive")
        expected_indices = set(range(expected_prompts))
        coverage_failures = {}
        for method in sorted(requested):
            for metric, values in observations[method].items():
                actual_indices = set(values)
                if actual_indices != expected_indices:
                    coverage_failures[f"{method}:{metric}"] = {
                        "missing": sorted(expected_indices - actual_indices),
                        "extra": sorted(actual_indices - expected_indices),
                    }
        if coverage_failures:
            raise ValueError(
                f"per-prompt coverage mismatch: {coverage_failures}"
            )

    comparisons: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for reference in references:
            key = f"{candidate}__vs__{reference}"
            shared_metrics = sorted(
                set(observations[candidate]) & set(observations[reference])
            )
            rows = {}
            for metric in shared_metrics:
                try:
                    rows[metric] = paired_statistics(
                        observations[candidate][metric],
                        observations[reference][metric],
                        direction=metric_direction(metric),
                        bootstrap_samples=bootstrap_samples,
                        permutation_samples=permutation_samples,
                        seed=seed,
                        label=f"{key}:{metric}",
                    )
                except ValueError:
                    continue
            comparisons[key] = {
                "candidate": candidate,
                "reference": reference,
                "metrics": rows,
            }
    return {
        "schema_version": 1,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "expected_prompts": expected_prompts,
        "comparisons": comparisons,
    }


def _fmt(value: Any, digits: int = 5) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# v120 paired metric analysis",
        "",
        (
            "Positive improvement means the candidate is better after applying "
            "the metric direction. Raw deltas always equal candidate minus "
            "reference."
        ),
    ]
    order = {metric: index for index, metric in enumerate(PRIMARY_METRIC_ORDER)}
    for comparison in payload["comparisons"].values():
        lines.extend(
            [
                "",
                (
                    f"## {comparison['candidate']} vs "
                    f"{comparison['reference']}"
                ),
                "",
                (
                    "| Metric | n | Candidate | Reference | Raw delta | "
                    "95% CI | W/T/L | p |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        metrics = sorted(
            comparison["metrics"],
            key=lambda metric: (order.get(metric, len(order)), metric),
        )
        for metric in metrics:
            row = comparison["metrics"][metric]
            ci = row["bootstrap_95_ci_raw_delta"]
            lines.append(
                f"| {metric} | {row['n']} | "
                f"{_fmt(row['candidate_mean'])} | "
                f"{_fmt(row['reference_mean'])} | "
                f"{_fmt(row['raw_mean_delta'])} | "
                f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | "
                f"{row['wins']}/{row['ties']}/{row['losses']} | "
                f"{_fmt(row['paired_randomization_p'], 4)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise SystemExit("resample counts must be positive")
    if args.expected_prompts <= 0:
        raise SystemExit("--expected-prompts must be positive")

    vbench: dict[str, dict[str, dict[int, float]]] = {}
    sources = []
    for raw_binding in args.vbench:
        method, path = parse_binding(raw_binding)
        if method in vbench:
            raise SystemExit(f"duplicate VBench method: {method}")
        vbench[method] = extract_vbench(path)
        sources.append(
            {"kind": "vbench", "method": method, "path": str(path), "sha256": sha256(path)}
        )

    comprehensive = {}
    if args.comprehensive is not None:
        comprehensive_path = args.comprehensive.expanduser().resolve()
        comprehensive = extract_comprehensive(comprehensive_path)
        sources.append(
            {
                "kind": "comprehensive",
                "path": str(comprehensive_path),
                "sha256": sha256(comprehensive_path),
            }
        )
    observations = merge_observations(
        vbench=vbench,
        comprehensive=comprehensive,
    )
    payload = analyze(
        observations,
        references=args.references,
        candidates=args.candidates,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
        expected_prompts=args.expected_prompts,
    )
    payload["sources"] = sources
    payload["method_metric_counts"] = {
        method: {
            metric: len(values)
            for metric, values in sorted(metrics.items())
        }
        for method, metrics in sorted(observations.items())
    }

    for path in (args.output_json, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"[v120-paired] methods={len(observations)} "
        f"comparisons={len(payload['comparisons'])}"
    )


if __name__ == "__main__":
    main()
