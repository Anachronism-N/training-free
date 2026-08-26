#!/usr/bin/env python3
"""Audit v189 Head x Phase structure without changing its frozen generation map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPERIMENT = "v197_threshold_free_head_phase_structure"
SOURCE_EXPERIMENT = "v189_structured_head_phase_profile"
CALLS = 4
LAYERS = 30
HEADS = 12
EXPECTED_CELLS = CALLS * LAYERS * HEADS
PERMUTATION_DRAWS = 10_000
GAIN_THRESHOLDS = (0.0, 0.01, 0.02, 0.05)
WIN_THRESHOLDS = (0.55, 0.60, 0.65)
PRIMARY_GAIN_THRESHOLD = 0.02
PRIMARY_WIN_THRESHOLD = 0.60
VALIDATION_GAIN_THRESHOLD = 0.0
VALIDATION_CI_LOWER = -0.01
MIN_BUDGET_FRACTION = 0.80
MIN_RELATIVE_ENERGY = 0.10

REQUIRED_SCORE_COLUMNS = {
    "operator",
    "call_index",
    "layer",
    "head",
    "discovery_gain",
    "validation_gain",
    "validation_ci_lower",
    "validation_win_fraction",
    "full_budget_fraction",
    "relative_reference_energy",
    "compatible",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value in v189 score table: {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_map(path: Path, expected_sha: str, operator: str) -> np.ndarray:
    if not path.is_file() or sha256(path) != expected_sha:
        raise ValueError(f"v189 compatible map is missing or hash-drifted: {operator}")
    payload = _load_json(path)
    masks = payload.get("coverage_masks")
    array = np.asarray(masks)
    if (
        payload.get("coverage_operator") != operator
        or payload.get("classification") != "compatible"
        or payload.get("call_count") != CALLS
        or payload.get("layer_count") != LAYERS
        or payload.get("head_count") != HEADS
        or array.shape != (CALLS, LAYERS, HEADS)
        or array.dtype != np.bool_
    ):
        raise ValueError(f"invalid v189 compatible map contract: {operator}")
    expected_counts = [int(array[call].sum()) for call in range(CALLS)]
    if payload.get("coverage_count_by_call") != expected_counts:
        raise ValueError(f"v189 compatible map count mismatch: {operator}")
    return array


def load_inputs(analysis_path: Path, score_path: Path) -> tuple[dict, dict]:
    analysis = _load_json(analysis_path)
    if (
        analysis.get("experiment") != SOURCE_EXPERIMENT
        or analysis.get("manual_review_required") is not False
        or not isinstance(analysis.get("operators"), dict)
        or not analysis.get("operators")
    ):
        raise ValueError("invalid v189 analysis contract")
    manifest_path = Path(str(analysis.get("input_manifest", "")))
    if not manifest_path.is_file() or sha256(manifest_path) != analysis.get(
        "input_manifest_sha256"
    ):
        raise ValueError("v189 input manifest is missing or hash-drifted")

    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_SCORE_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError("v189 cell score schema is incomplete")
        raw_rows = list(reader)

    tensors = {}
    expected_operators = set(analysis["operators"])
    observed_operators = {row["operator"] for row in raw_rows}
    if observed_operators != expected_operators:
        raise ValueError("v189 score operators disagree with analysis.json")
    for operator in sorted(expected_operators):
        shape = (CALLS, LAYERS, HEADS)
        fields = {
            key: np.full(shape, np.nan, dtype=np.float64)
            for key in (
                "discovery_gain",
                "validation_gain",
                "validation_ci_lower",
                "validation_win_fraction",
                "full_budget_fraction",
                "relative_reference_energy",
            )
        }
        compatible = np.zeros(shape, dtype=np.bool_)
        seen = set()
        for row in raw_rows:
            if row["operator"] != operator:
                continue
            key = (
                int(row["call_index"]),
                int(row["layer"]),
                int(row["head"]),
            )
            call, layer, head = key
            if (
                key in seen
                or not 0 <= call < CALLS
                or not 0 <= layer < LAYERS
                or not 0 <= head < HEADS
            ):
                raise ValueError(
                    f"duplicate or invalid v189 score cell: {operator}/{key}"
                )
            seen.add(key)
            for field, array in fields.items():
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite v189 score: {operator}/{key}/{field}")
                array[key] = value
            compatible[key] = _parse_bool(row["compatible"])
        if len(seen) != EXPECTED_CELLS or any(
            np.isnan(array).any() for array in fields.values()
        ):
            raise ValueError(f"incomplete v189 score tensor: {operator}")

        map_info = (
            analysis["operators"][operator].get("maps", {}).get("compatible") or {}
        )
        map_path = Path(str(map_info.get("path", "")))
        frozen = _validate_map(map_path, str(map_info.get("sha256", "")), operator)
        if not np.array_equal(frozen, compatible):
            raise ValueError(f"v189 CSV/map membership disagreement: {operator}")

        primary_from_scores = _threshold_mask(
            fields,
            gain_threshold=PRIMARY_GAIN_THRESHOLD,
            win_threshold=PRIMARY_WIN_THRESHOLD,
        )
        if not np.array_equal(primary_from_scores, compatible):
            raise ValueError(f"v189 primary threshold contract drifted: {operator}")
        tensors[operator] = {
            **fields,
            "compatible": compatible,
            "map_path": map_path,
            "map_sha256": map_info["sha256"],
        }
    return analysis, tensors


def _threshold_mask(
    arrays: dict[str, np.ndarray], *, gain_threshold: float, win_threshold: float
) -> np.ndarray:
    return (
        (arrays["discovery_gain"] >= gain_threshold)
        & (arrays["validation_gain"] >= VALIDATION_GAIN_THRESHOLD)
        & (arrays["validation_ci_lower"] >= VALIDATION_CI_LOWER)
        & (arrays["validation_win_fraction"] >= win_threshold)
        & (arrays["full_budget_fraction"] >= MIN_BUDGET_FRACTION)
        & (arrays["relative_reference_energy"] >= MIN_RELATIVE_ENERGY)
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(len(flat), dtype=np.float64)
    start = 0
    while start < len(flat):
        end = start + 1
        while end < len(flat) and flat[order[end]] == flat[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks.reshape(values.shape)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _correlation(_rankdata(left), _rankdata(right))


def _variance_decomposition(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (CALLS, LAYERS, HEADS):
        raise ValueError("Head x Phase tensor has an invalid shape")
    grand = float(values.mean())
    layer_means = values.mean(axis=(0, 2))
    ss_layer = float(CALLS * HEADS * np.square(layer_means - grand).sum())
    ss_call = 0.0
    ss_head = 0.0
    ss_interaction = 0.0
    interaction = np.zeros_like(values)
    call_component = np.zeros_like(values)
    head_component = np.zeros_like(values)
    for layer in range(LAYERS):
        matrix = values[:, layer, :]
        center = float(matrix.mean())
        call_effect = matrix.mean(axis=1) - center
        head_effect = matrix.mean(axis=0) - center
        residual = matrix - center - call_effect[:, None] - head_effect[None, :]
        ss_call += float(HEADS * np.square(call_effect).sum())
        ss_head += float(CALLS * np.square(head_effect).sum())
        ss_interaction += float(np.square(residual).sum())
        interaction[:, layer, :] = residual
        call_component[:, layer, :] = call_effect[:, None]
        head_component[:, layer, :] = head_effect[None, :]
    total = float(np.square(values - grand).sum())
    components = {
        "layer": ss_layer,
        "denoising_phase_within_layer": ss_call,
        "head_within_layer": ss_head,
        "head_by_phase_interaction": ss_interaction,
    }
    return {
        "sum_squares": components,
        "fraction_of_total": {
            key: (value / total if total > 1e-20 else None)
            for key, value in components.items()
        },
        "total_sum_squares": total,
        "interaction": interaction,
        "call_component": call_component,
        "head_component": head_component,
    }


def _contrast(values: np.ndarray, selected: np.ndarray) -> float:
    selected_values = values[selected]
    other_values = values[~selected]
    if not len(selected_values) or not len(other_values):
        raise ValueError("cross-fit contrast requires selected and complement cells")
    return float(selected_values.mean() - other_values.mean())


def _crossfit_axis(
    discovery: np.ndarray,
    validation: np.ndarray,
    *,
    axis: int,
    k: int,
    draws: int,
    seed: int,
) -> dict:
    discovery = np.moveaxis(discovery, axis, -1)
    validation = np.moveaxis(validation, axis, -1)
    categories = discovery.shape[-1]
    if not 0 < k < categories:
        raise ValueError("cross-fit k must leave a non-empty complement")
    observed_selected = np.zeros(discovery.shape, dtype=np.bool_)
    top_indices = np.argpartition(discovery, categories - k, axis=-1)[..., -k:]
    np.put_along_axis(observed_selected, top_indices, True, axis=-1)
    observed = _contrast(validation, observed_selected)

    rng = np.random.default_rng(seed)
    null_values = []
    remaining = draws
    while remaining:
        chunk = min(256, remaining)
        random_scores = rng.random((chunk, *discovery.shape))
        random_indices = np.argpartition(random_scores, categories - k, axis=-1)[
            ..., -k:
        ]
        masks = np.zeros_like(random_scores, dtype=np.bool_)
        np.put_along_axis(masks, random_indices, True, axis=-1)
        selected_sum = np.where(masks, validation[None, ...], 0.0).sum(
            axis=tuple(range(1, masks.ndim))
        )
        selected_count = int(np.prod(discovery.shape[:-1]) * k)
        total_sum = float(validation.sum())
        total_count = validation.size
        null_values.extend(
            (
                selected_sum / selected_count
                - (total_sum - selected_sum) / (total_count - selected_count)
            ).tolist()
        )
        remaining -= chunk
    null = np.asarray(null_values, dtype=np.float64)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    std = float(null.std())
    return {
        "k": k,
        "category_count": categories,
        "selected_cell_count": int(observed_selected.sum()),
        "validation_contrast": observed,
        "null_mean": float(null.mean()),
        "null_std": std,
        "z_score": float((observed - null.mean()) / std) if std > 1e-12 else None,
        "one_sided_permutation_p": p_value,
        "permutation_draws": draws,
    }


def _global_top_fraction(
    discovery: np.ndarray,
    validation: np.ndarray,
    *,
    fraction: float,
    draws: int,
    seed: int,
) -> dict:
    flat_discovery = discovery.reshape(-1)
    flat_validation = validation.reshape(-1)
    count = max(1, round(len(flat_discovery) * fraction))
    selected = np.zeros(len(flat_discovery), dtype=np.bool_)
    selected[np.argpartition(flat_discovery, len(flat_discovery) - count)[-count:]] = (
        True
    )
    observed = _contrast(flat_validation, selected)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        end = min(start + 256, draws)
        random_scores = rng.random((end - start, len(flat_discovery)))
        indices = np.argpartition(random_scores, len(flat_discovery) - count, axis=1)[
            :, -count:
        ]
        selected_sums = flat_validation[indices].sum(axis=1)
        null[start:end] = selected_sums / count - (
            flat_validation.sum() - selected_sums
        ) / (len(flat_validation) - count)
    std = float(null.std())
    return {
        "fraction": fraction,
        "selected_cell_count": count,
        "validation_contrast": observed,
        "null_mean": float(null.mean()),
        "null_std": std,
        "z_score": float((observed - null.mean()) / std) if std > 1e-12 else None,
        "one_sided_permutation_p": float(
            (1 + np.count_nonzero(null >= observed)) / (draws + 1)
        ),
        "permutation_draws": draws,
    }


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _threshold_grid(operator: str, arrays: dict) -> list[dict]:
    primary = arrays["compatible"]
    rows = []
    for gain in GAIN_THRESHOLDS:
        for win in WIN_THRESHOLDS:
            mask = _threshold_mask(arrays, gain_threshold=gain, win_threshold=win)
            rows.append(
                {
                    "operator": operator,
                    "discovery_gain_threshold": gain,
                    "validation_win_threshold": win,
                    "cell_count": int(mask.sum()),
                    "count_by_call": [int(mask[call].sum()) for call in range(CALLS)],
                    "jaccard_with_primary": _jaccard(mask, primary),
                    "head_count_any_call": int(mask.any(axis=0).sum()),
                    "phase_varying_head_count": int(
                        np.logical_and(
                            mask.sum(axis=0) > 0, mask.sum(axis=0) < CALLS
                        ).sum()
                    ),
                }
            )
    return rows


def _topology(mask: np.ndarray) -> dict:
    calls_per_head = mask.sum(axis=0)
    histogram = {
        str(count): int(np.count_nonzero(calls_per_head == count))
        for count in range(CALLS + 1)
    }
    layer_counts = mask.sum(axis=(0, 2)).astype(int).tolist()
    nonzero = np.asarray(layer_counts, dtype=np.float64)
    probabilities = nonzero / nonzero.sum() if nonzero.sum() else nonzero
    entropy = float(
        -sum(value * math.log(value) for value in probabilities if value > 0)
    )
    return {
        "cell_count": int(mask.sum()),
        "count_by_call": [int(mask[call].sum()) for call in range(CALLS)],
        "count_by_layer": layer_counts,
        "heads_by_selected_call_count": histogram,
        "heads_selected_any_call": int(np.count_nonzero(calls_per_head > 0)),
        "heads_selected_all_calls": int(np.count_nonzero(calls_per_head == CALLS)),
        "phase_varying_heads": int(
            np.count_nonzero(np.logical_and(calls_per_head > 0, calls_per_head < CALLS))
        ),
        "active_layer_count": int(np.count_nonzero(nonzero)),
        "layer_selection_entropy": entropy,
    }


def _topk_overlap(discovery: np.ndarray, validation: np.ndarray) -> list[dict]:
    output = []
    flat_d = discovery.reshape(-1)
    flat_v = validation.reshape(-1)
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, round(len(flat_d) * fraction))
        d = set(np.argpartition(flat_d, len(flat_d) - count)[-count:].tolist())
        v = set(np.argpartition(flat_v, len(flat_v) - count)[-count:].tolist())
        output.append(
            {
                "fraction": fraction,
                "cell_count": count,
                "intersection": len(d & v),
                "jaccard": len(d & v) / len(d | v),
                "chance_intersection": count * count / len(flat_d),
            }
        )
    return output


def _diagnostic_level(
    head_test: dict, phase_test: dict, interaction_r: float | None, mean_gain: float
) -> str:
    head = bool(
        head_test["validation_contrast"] > 0
        and head_test["one_sided_permutation_p"] <= 0.05
    )
    phase = bool(
        phase_test["validation_contrast"] > 0
        and phase_test["one_sided_permutation_p"] <= 0.05
    )
    interaction = bool(interaction_r is not None and interaction_r > 0)
    if head and phase and interaction:
        return "joint_head_phase_structure"
    if head and phase:
        return "additive_head_and_phase_structure"
    if head:
        return "head_structure_only"
    if phase:
        return "phase_structure_only"
    if mean_gain > 0:
        return "operator_level_gain_only"
    return "unsupported"


def analyze(
    analysis_path: Path,
    score_path: Path,
    output_dir: Path,
    *,
    draws: int = PERMUTATION_DRAWS,
) -> dict:
    source_analysis, tensors = load_inputs(analysis_path, score_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    operator_reports = {}
    threshold_rows = []
    crossfit_rows = []
    for operator_index, (operator, arrays) in enumerate(sorted(tensors.items())):
        discovery = arrays["discovery_gain"]
        validation = arrays["validation_gain"]
        discovery_decomposition = _variance_decomposition(discovery)
        validation_decomposition = _variance_decomposition(validation)
        head_tests = [
            _crossfit_axis(
                discovery,
                validation,
                axis=2,
                k=k,
                draws=draws,
                seed=1971000 + operator_index * 100 + k,
            )
            for k in (1, 3, 6)
        ]
        phase_test = _crossfit_axis(
            discovery,
            validation,
            axis=0,
            k=1,
            draws=draws,
            seed=1972000 + operator_index,
        )
        global_tests = [
            _global_top_fraction(
                discovery,
                validation,
                fraction=fraction,
                draws=draws,
                seed=1973000 + operator_index * 100 + int(fraction * 100),
            )
            for fraction in (0.01, 0.05, 0.10)
        ]
        interaction_r = _correlation(
            discovery_decomposition["interaction"],
            validation_decomposition["interaction"],
        )
        head_r = _correlation(
            discovery_decomposition["head_component"],
            validation_decomposition["head_component"],
        )
        phase_r = _correlation(
            discovery_decomposition["call_component"],
            validation_decomposition["call_component"],
        )
        grid = _threshold_grid(operator, arrays)
        threshold_rows.extend(grid)
        for family, tests in (
            ("head_identity_within_call_layer", head_tests),
            ("phase_identity_within_layer_head", [phase_test]),
            ("global_cell_ranking", global_tests),
        ):
            for test in tests:
                crossfit_rows.append(
                    {
                        "operator": operator,
                        "family": family,
                        **test,
                    }
                )
        mean_gain = float(validation.mean())
        operator_reports[operator] = {
            "v189_generation_candidate": operator
            in source_analysis.get("generation_candidates", ()),
            "continuous_reproducibility": {
                "pearson_all_cells": _correlation(discovery, validation),
                "spearman_all_cells": _spearman(discovery, validation),
                "sign_agreement": float(np.mean((discovery >= 0) == (validation >= 0))),
                "topk_overlap": _topk_overlap(discovery, validation),
                "head_main_effect_correlation": head_r,
                "phase_main_effect_correlation": phase_r,
                "head_phase_interaction_correlation": interaction_r,
            },
            "discovery_variance_decomposition": {
                key: value
                for key, value in discovery_decomposition.items()
                if not isinstance(value, np.ndarray)
            },
            "validation_variance_decomposition": {
                key: value
                for key, value in validation_decomposition.items()
                if not isinstance(value, np.ndarray)
            },
            "crossfit": {
                "head_identity_topk": head_tests,
                "phase_identity_top1": phase_test,
                "global_top_fraction": global_tests,
            },
            "primary_map_topology": _topology(arrays["compatible"]),
            "threshold_neighborhood": {
                "grid_size": len(grid),
                "minimum_jaccard_with_primary": min(
                    row["jaccard_with_primary"] for row in grid
                ),
                "median_jaccard_with_primary": float(
                    np.median([row["jaccard_with_primary"] for row in grid])
                ),
            },
            "validation_gain_mean_all_cells": mean_gain,
            "diagnostic_structure_level": _diagnostic_level(
                head_tests[1], phase_test, interaction_r, mean_gain
            ),
        }

    report = {
        "version": 1,
        "experiment": EXPERIMENT,
        "diagnostic_only": True,
        "changes_v189_frozen_map": False,
        "generation_gate": "v190_only",
        "permutation_draws": draws,
        "operators": operator_reports,
        "recommendation": "retain_v190_as_the_only_generation_side_causal_gate",
        "manual_review_required": False,
        "claim_boundary": (
            "Cross-split structure and permutation diagnostics can support a profiling "
            "mechanism hypothesis. They cannot establish generated-video benefit, select "
            "a post-hoc threshold, or replace the v190 causal screen."
        ),
        "source": {
            "v189_analysis": str(analysis_path.resolve()),
            "v189_analysis_sha256": sha256(analysis_path),
            "v189_cell_scores": str(score_path.resolve()),
            "v189_cell_scores_sha256": sha256(score_path),
            "v189_input_manifest": source_analysis["input_manifest"],
            "v189_input_manifest_sha256": source_analysis["input_manifest_sha256"],
            "compatible_maps": {
                operator: {
                    "path": str(arrays["map_path"].resolve()),
                    "sha256": arrays["map_sha256"],
                }
                for operator, arrays in tensors.items()
            },
        },
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "threshold_grid.csv", threshold_rows)
    _write_csv(output_dir / "crossfit_tests.csv", crossfit_rows)
    (output_dir / "analysis.md").write_text(render(report), encoding="utf-8")
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty diagnostic table: {path.name}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def render(report: dict) -> str:
    lines = [
        "# v197 Threshold-Free Head x Phase Structure Audit",
        "",
        "- Diagnostic only: `true`",
        "- Changes the v189 frozen map: `false`",
        "- Generated-video gate: `v190_only`",
        "- Manual review required: `false`",
        "",
        "| Operator | Structure level | Spearman | Interaction r | Head top3 p | Phase top1 p | Primary cells | Phase-varying heads |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for operator, row in sorted(report["operators"].items()):
        reproducibility = row["continuous_reproducibility"]
        topology = row["primary_map_topology"]
        head = row["crossfit"]["head_identity_topk"][1]
        phase = row["crossfit"]["phase_identity_top1"]
        values = (
            operator,
            row["diagnostic_structure_level"],
            reproducibility["spearman_all_cells"],
            reproducibility["head_phase_interaction_correlation"],
            head["one_sided_permutation_p"],
            phase["one_sided_permutation_p"],
            topology["cell_count"],
            topology["phase_varying_heads"],
        )
        lines.append(
            f"| {values[0]} | {values[1]} | {_format(values[2])} | "
            f"{_format(values[3])} | {_format(values[4])} | {_format(values[5])} | "
            f"{values[6]} | {values[7]} |"
        )
    lines.extend(
        [
            "",
            (
                "The permutation p-values are diagnostic randomization checks over correlated "
                "cells, not independent-head significance tests. Threshold-grid results cannot "
                "be used to select a new generation map after looking at v190 videos."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v189-analysis", type=Path, required=True)
    parser.add_argument("--cell-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutation-draws", type=int, default=PERMUTATION_DRAWS)
    args = parser.parse_args()
    if args.permutation_draws < 100:
        parser.error("--permutation-draws must be at least 100")
    report = analyze(
        args.v189_analysis,
        args.cell_scores,
        args.output_dir,
        draws=args.permutation_draws,
    )
    levels = ",".join(
        f"{operator}:{row['diagnostic_structure_level']}"
        for operator, row in sorted(report["operators"].items())
    )
    print(f"[v197-analysis] {levels} review=false generation_gate=v190_only")


if __name__ == "__main__":
    main()
