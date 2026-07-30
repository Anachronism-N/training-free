#!/usr/bin/env python3
"""Cluster stable multi-axis head features without fitting to PF labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TAXONOMY_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "lifecycle_kv"
    / "head_taxonomy.py"
)
TAXONOMY_SPEC = importlib.util.spec_from_file_location(
    "v143_cluster_head_taxonomy", TAXONOMY_PATH
)
TAXONOMY = importlib.util.module_from_spec(TAXONOMY_SPEC)
assert TAXONOMY_SPEC.loader is not None
sys.modules[TAXONOMY_SPEC.name] = TAXONOMY
TAXONOMY_SPEC.loader.exec_module(TAXONOMY)

adjusted_rand_index = TAXONOMY.adjusted_rand_index
assign_clusters = TAXONOMY.assign_clusters
deterministic_kmeans = TAXONOMY.deterministic_kmeans
normalized_mutual_information = TAXONOMY.normalized_mutual_information
robust_fit_transform = TAXONOMY.robust_fit_transform
silhouette_score = TAXONOMY.silhouette_score


LAYERS = 30
HEADS = 12
TOTAL_HEADS = LAYERS * HEADS
EPSILON = 1e-12

JOB_SOURCES = {
    "v136_prompt": (
        "head_prompt_job_axes.csv",
        ("cphi_score", "age_js_score"),
    ),
    "v136_temporal": (
        "head_temporal_job_axes.csv",
        (
            "temporal_reach_ratio",
            "middle_recent_margin",
            "old_mass_excess",
            "positive_logit_fraction",
            "sign_switch_rate",
            "spectral_peak_ratio",
        ),
    ),
    "v138_local": (
        "head_local_job_axes.csv",
        (
            "reverse_relative_log",
            "phase_shift_relative_log",
            "freeze_latest_relative_log",
            "value_mismatch_relative_log",
        ),
    ),
    "v138_cross": (
        "head_cross_job_axes.csv",
        ("history_specificity",),
    ),
}

V143_NATURAL_FEATURES = (
    "current_mass",
    "oldest1_mass",
    "middle_mass",
    "recent4_mass",
    "last4_mass",
    "recent4_non_oldest_ratio",
    "history_positive_rate",
    "history_mean_logit",
    "policy_need",
)

V143_AB_FEATURES = (
    "prompt_history_excess",
    "policy_prompt_score",
    "persistent_content",
    "persistent_output",
    "stale_a_mass_B_visible",
)

FEATURE_GROUPS = (
    "prompt_modulation",
    "temporal_allocation",
    "history_intervention",
    "history_specificity",
    "output_policy",
    "episodic_compatibility",
    "switch_plasticity",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _head_index(row: dict) -> int:
    layer, head = int(row["layer"]), int(row["head"])
    if not (0 <= layer < LAYERS and 0 <= head < HEADS):
        raise ValueError(f"invalid head coordinate: layer={layer} head={head}")
    return layer * HEADS + head


def _independent_unit(job_id: str, source: str) -> int:
    if source == "v136_prompt":
        match = re.fullmatch(r"cf_(\d+)_.*", job_id)
    else:
        match = re.search(r"(\d+)$", job_id)
    if match is None:
        raise ValueError(
            f"cannot recover an independent split unit from {source} "
            f"job_id={job_id!r}"
        )
    return int(match.group(1))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    left_rank = _rankdata(left[finite])
    right_rank = _rankdata(right[finite])
    if left_rank.std() <= EPSILON or right_rank.std() <= EPSILON:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _within_layer_residual(values: np.ndarray) -> np.ndarray:
    """Remove the per-layer median while preserving within-layer head order."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (TOTAL_HEADS,):
        raise ValueError(
            f"layer residualization requires {TOTAL_HEADS} heads, got {array.shape}"
        )
    matrix = array.reshape(LAYERS, HEADS)
    if not np.isfinite(matrix).all():
        return np.full_like(array, np.nan)
    return (matrix - np.median(matrix, axis=1, keepdims=True)).reshape(-1)


def _layer_eta_squared(values: np.ndarray) -> float:
    """Fraction of scalar feature variance explained by the layer index."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (TOTAL_HEADS,) or not np.isfinite(array).all():
        return float("nan")
    grand_mean = float(array.mean())
    total = float(np.square(array - grand_mean).sum())
    if total <= EPSILON:
        return 0.0
    matrix = array.reshape(LAYERS, HEADS)
    layer_means = matrix.mean(axis=1)
    between = float(HEADS * np.square(layer_means - grand_mean).sum())
    return between / total


def _feature_group(name: str) -> str:
    if name.startswith("v136_prompt."):
        return "prompt_modulation"
    if name.startswith("v136_temporal."):
        return "temporal_allocation"
    if name.startswith("v138_local."):
        return "history_intervention"
    if name.startswith("v138_cross."):
        return "history_specificity"
    if name == "v143_natural.policy_need":
        return "output_policy"
    if name.startswith("v143_natural."):
        return "temporal_allocation"
    if name.startswith("v143_ab.persistent_"):
        return "episodic_compatibility"
    if name.startswith("v143_ab."):
        return "switch_plasticity"
    raise ValueError(f"no conceptual feature group for {name}")


def _split_job_features(
    path: Path,
    *,
    source: str,
    fields: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = _read_csv(path)
    if not rows:
        return {}
    required = {"job_id", "layer", "head", *fields}
    if not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))
        raise ValueError(f"{path} lacks required columns: {missing}")
    jobs = sorted({str(row["job_id"]) for row in rows})
    units = {
        job: _independent_unit(job, source)
        for job in jobs
    }
    discovery_jobs = {
        job for job, unit in units.items() if unit % 2 == 0
    }
    validation_jobs = set(jobs) - discovery_jobs
    if not discovery_jobs or not validation_jobs:
        raise ValueError(f"{path} cannot form two non-empty independent splits")
    output = {}
    for field in fields:
        discovery_values = defaultdict(list)
        validation_values = defaultdict(list)
        for row in rows:
            value = float(row[field])
            if not np.isfinite(value):
                continue
            target = (
                discovery_values
                if str(row["job_id"]) in discovery_jobs
                else validation_values
            )
            target[_head_index(row)].append(value)
        discovery = np.asarray(
            [
                np.median(discovery_values[index])
                if discovery_values[index]
                else np.nan
                for index in range(TOTAL_HEADS)
            ]
        )
        validation = np.asarray(
            [
                np.median(validation_values[index])
                if validation_values[index]
                else np.nan
                for index in range(TOTAL_HEADS)
            ]
        )
        output[f"{source}.{field}"] = (discovery, validation)
    return output


def _split_prefixed_features(
    path: Path,
    *,
    source: str,
    fields: tuple[str, ...],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict]]:
    rows = _read_csv(path)
    if not rows:
        return {}, []
    by_head = {_head_index(row): row for row in rows}
    if set(by_head) != set(range(TOTAL_HEADS)):
        raise ValueError(f"{path} does not contain exactly 360 heads")
    output = {}
    for field in fields:
        discovery_name = f"discovery_{field}"
        validation_name = f"validation_{field}"
        if discovery_name not in rows[0] or validation_name not in rows[0]:
            continue
        discovery = np.asarray(
            [float(by_head[index][discovery_name]) for index in range(TOTAL_HEADS)]
        )
        validation = np.asarray(
            [float(by_head[index][validation_name]) for index in range(TOTAL_HEADS)]
        )
        output[f"{source}.{field}"] = (discovery, validation)
    return output, rows


def _load_matrix(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([int(value.strip()) for value in row if value.strip()])
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError(f"{path} must be a {LAYERS}x{HEADS} integer matrix")
    return np.asarray(rows, dtype=np.int64).reshape(-1)


def _external_labels(
    natural_rows: list[dict],
    pf_labels: np.ndarray | None,
    v98_labels: np.ndarray | None,
) -> dict[str, np.ndarray]:
    output = {}
    if pf_labels is not None:
        output["pf_official"] = pf_labels
    if v98_labels is not None:
        output["legacy_v98"] = v98_labels
    if natural_rows:
        by_head = {_head_index(row): row for row in natural_rows}
        for column in (
            "forcing_kv_label",
            "head_forcing_label",
            "dummy_forcing_label",
        ):
            if column in natural_rows[0]:
                output[column.removesuffix("_label")] = np.asarray(
                    [by_head[index][column] for index in range(TOTAL_HEADS)]
                )
    return output


def _bootstrap_cluster_stability(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    clusters: int,
    rounds: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for round_index in range(rounds):
        indices = rng.integers(0, values.shape[0], values.shape[0])
        fitted = deterministic_kmeans(
            values[indices],
            clusters,
            restarts=8,
            seed=seed + 1009 * round_index,
        )
        assigned = assign_clusters(values, fitted.centers)
        scores.append(adjusted_rand_index(labels, assigned))
    return float(np.median(scores)), float(np.quantile(scores, 0.10))


def _cluster_margin(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distance = np.sqrt(
        np.maximum(
            0.0,
            ((values[:, None] - centers[None, :]) ** 2).sum(axis=2),
        )
    )
    ordered = np.sort(distance, axis=1)
    return (ordered[:, 1] - ordered[:, 0]) / np.clip(
        ordered[:, 1], EPSILON, None
    )


def analyze(args: argparse.Namespace) -> dict:
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source, (filename, fields) in JOB_SOURCES.items():
        directory = (
            args.v136_analysis_dir
            if source.startswith("v136")
            else args.v138_analysis_dir
        )
        features.update(
            _split_job_features(
                directory / filename,
                source=source,
                fields=fields,
            )
        )
    natural_features, natural_rows = _split_prefixed_features(
        args.v143_analysis_dir / "natural_head_axes.csv",
        source="v143_natural",
        fields=V143_NATURAL_FEATURES,
    )
    ab_features, _ = _split_prefixed_features(
        args.v143_analysis_dir / "ab_head_axes.csv",
        source="v143_ab",
        fields=V143_AB_FEATURES,
    )
    features.update(natural_features)
    features.update(ab_features)

    excluded_groups = set(args.exclude_feature_group or ())
    unknown_groups = excluded_groups - set(FEATURE_GROUPS)
    if unknown_groups:
        raise ValueError(
            f"unknown excluded feature groups: {sorted(unknown_groups)}"
        )
    feature_audit = []
    accepted = []
    coordinate_features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (discovery, validation) in sorted(features.items()):
        raw_finite = np.isfinite(discovery) & np.isfinite(validation)
        residual_discovery = _within_layer_residual(discovery)
        residual_validation = _within_layer_residual(validation)
        residual_finite = (
            np.isfinite(residual_discovery) & np.isfinite(residual_validation)
        )
        raw_rho = _spearman(discovery, validation)
        residual_rho = _spearman(
            residual_discovery, residual_validation
        )
        raw_spread = (
            float(
                np.quantile(discovery[raw_finite], 0.75)
                - np.quantile(discovery[raw_finite], 0.25)
            )
            if raw_finite.any()
            else 0.0
        )
        residual_spread = (
            float(
                np.quantile(residual_discovery[residual_finite], 0.75)
                - np.quantile(residual_discovery[residual_finite], 0.25)
            )
            if residual_finite.any()
            else 0.0
        )
        if args.coordinate_system == "layer_residual":
            coordinate_discovery = residual_discovery
            coordinate_validation = residual_validation
            finite = residual_finite
            rho = residual_rho
            spread = residual_spread
        else:
            coordinate_discovery = discovery
            coordinate_validation = validation
            finite = raw_finite
            rho = raw_rho
            spread = raw_spread
        coordinate_features[name] = (
            coordinate_discovery,
            coordinate_validation,
        )
        conceptual_group = _feature_group(name)
        excluded = conceptual_group in excluded_groups
        passed = bool(
            finite.all()
            and spread > 1e-8
            and rho >= args.min_feature_split_rho
            and not excluded
        )
        if excluded:
            reason = "excluded_feature_group"
        elif not finite.all():
            reason = "non_finite_heads"
        elif spread <= 1e-8:
            reason = "near_zero_discovery_iqr"
        elif rho < args.min_feature_split_rho:
            reason = "insufficient_split_spearman"
        else:
            reason = "accepted"
        feature_audit.append(
            {
                "feature": name,
                "conceptual_group": conceptual_group,
                "coordinate_system": args.coordinate_system,
                "finite_heads": int(finite.sum()),
                "discovery_iqr": spread,
                "split_spearman": rho,
                "raw_discovery_iqr": raw_spread,
                "raw_split_spearman": raw_rho,
                "layer_residual_discovery_iqr": residual_spread,
                "layer_residual_split_spearman": residual_rho,
                "discovery_layer_eta_squared": _layer_eta_squared(discovery),
                "validation_layer_eta_squared": _layer_eta_squared(validation),
                "minimum_split_spearman": args.min_feature_split_rho,
                "excluded_by_group_ablation": int(excluded),
                "accepted": int(passed),
                "reason": reason,
            }
        )
        if passed:
            accepted.append(name)
    if len(accepted) < 3:
        raise RuntimeError(
            "fewer than three split-stable axes are available; run the v143 "
            "profiles and preserve v136/v138 job-level CSV files"
        )

    discovery_source = np.stack(
        [features[name][0] for name in accepted], axis=1
    )
    validation_source = np.stack(
        [features[name][1] for name in accepted], axis=1
    )
    discovery_coordinate = np.stack(
        [coordinate_features[name][0] for name in accepted], axis=1
    )
    validation_coordinate = np.stack(
        [coordinate_features[name][1] for name in accepted], axis=1
    )
    discovery, validation, center, scale = robust_fit_transform(
        discovery_coordinate, validation_coordinate
    )
    assert validation is not None
    feature_groups = [_feature_group(name) for name in accepted]
    group_counts = Counter(feature_groups)
    feature_weights = np.asarray(
        [1.0 / np.sqrt(group_counts[group]) for group in feature_groups],
        dtype=np.float64,
    )
    discovery = discovery * feature_weights[None, :]
    validation = validation * feature_weights[None, :]
    feature_matrix_rows = []
    for flat_head in range(TOTAL_HEADS):
        row = {
            "layer": flat_head // HEADS,
            "head": flat_head % HEADS,
        }
        for index, name in enumerate(accepted):
            column = name.replace(".", "__")
            row[f"discovery_raw__{column}"] = float(
                discovery_source[flat_head, index]
            )
            row[f"validation_raw__{column}"] = float(
                validation_source[flat_head, index]
            )
            row[f"discovery_coordinate__{column}"] = float(
                discovery_coordinate[flat_head, index]
            )
            row[f"validation_coordinate__{column}"] = float(
                validation_coordinate[flat_head, index]
            )
            row[f"discovery_weighted__{column}"] = float(
                discovery[flat_head, index]
            )
            row[f"validation_weighted__{column}"] = float(
                validation[flat_head, index]
            )
        feature_matrix_rows.append(row)
    correlation_rows = []
    for left in range(len(accepted)):
        for right in range(left + 1, len(accepted)):
            correlation_rows.append(
                {
                    "left_feature": accepted[left],
                    "right_feature": accepted[right],
                    "discovery_source_spearman": _spearman(
                        discovery_source[:, left],
                        discovery_source[:, right],
                    ),
                    "validation_source_spearman": _spearman(
                        validation_source[:, left],
                        validation_source[:, right],
                    ),
                    "discovery_coordinate_spearman": _spearman(
                        discovery_coordinate[:, left],
                        discovery_coordinate[:, right],
                    ),
                    "validation_coordinate_spearman": _spearman(
                        validation_coordinate[:, left],
                        validation_coordinate[:, right],
                    ),
                }
            )
    diagnostics = []
    cluster_payload = {}
    for clusters in range(args.min_clusters, args.max_clusters + 1):
        fitted = deterministic_kmeans(
            discovery,
            clusters,
            restarts=args.restarts,
            seed=args.seed + clusters * 100003,
        )
        validation_labels_raw = assign_clusters(validation, fitted.centers)
        validation_labels = validation_labels_raw
        agreement = float(np.mean(fitted.labels == validation_labels))
        ari = adjusted_rand_index(fitted.labels, validation_labels)
        discovery_silhouette = silhouette_score(discovery, fitted.labels)
        validation_silhouette = silhouette_score(
            validation, validation_labels
        )
        counts = Counter(fitted.labels.tolist())
        min_fraction = min(counts.values()) / TOTAL_HEADS
        margin = _cluster_margin(discovery, fitted.centers)
        bootstrap_median, bootstrap_p10 = _bootstrap_cluster_stability(
            discovery,
            fitted.labels,
            clusters=clusters,
            rounds=args.bootstrap_rounds,
            seed=args.seed + clusters * 7919,
        )
        layer_ids = np.repeat(np.arange(LAYERS), HEADS)
        layer_bands = layer_ids // 6
        passed = bool(
            agreement >= 0.8
            and ari >= 0.6
            and bootstrap_median >= 0.75
            and discovery_silhouette >= 0.1
            and min_fraction >= 0.05
            and float(np.mean(margin < 0.05)) <= 0.2
        )
        row = {
            "clusters": clusters,
            "feature_count": len(accepted),
            "split_label_agreement": agreement,
            "split_ari": ari,
            "discovery_silhouette": discovery_silhouette,
            "validation_silhouette": validation_silhouette,
            "bootstrap_ari_median": bootstrap_median,
            "bootstrap_ari_p10": bootstrap_p10,
            "minimum_cluster_fraction": min_fraction,
            "boundary_fraction": float(np.mean(margin < 0.05)),
            "cluster_exact_layer_nmi": normalized_mutual_information(
                fitted.labels, layer_ids
            ),
            "cluster_layer_band_nmi": normalized_mutual_information(
                fitted.labels, layer_bands
            ),
            "passed": int(passed),
        }
        diagnostics.append(row)
        cluster_payload[clusters] = {
            "fit": fitted,
            "validation_labels": validation_labels,
            "margin": margin,
        }
    passing = [row for row in diagnostics if row["passed"]]
    selected = (
        min(row["clusters"] for row in passing)
        if passing
        else None
    )

    external = _external_labels(
        natural_rows,
        _load_matrix(args.pf_labels),
        _load_matrix(args.v98_labels),
    )
    reference_rows = []
    assignment_rows = []
    center_rows = []
    if selected is not None:
        payload = cluster_payload[selected]
        labels = payload["fit"].labels
        validation_labels = payload["validation_labels"]
        for name, reference in external.items():
            reference_rows.append(
                {
                    "reference": name,
                    "ari": adjusted_rand_index(labels, reference),
                    "nmi": normalized_mutual_information(labels, reference),
                    "reference_classes": len(np.unique(reference)),
                }
            )
        for flat_head in range(TOTAL_HEADS):
            row = {
                "layer": flat_head // HEADS,
                "head": flat_head % HEADS,
                "cluster": int(labels[flat_head]),
                "validation_cluster": int(validation_labels[flat_head]),
                "split_agree": int(
                    labels[flat_head] == validation_labels[flat_head]
                ),
                "cluster_margin": float(payload["margin"][flat_head]),
            }
            for name, reference in external.items():
                row[name] = str(reference[flat_head])
            assignment_rows.append(row)
        for cluster in range(selected):
            members = labels == cluster
            row = {
                "cluster": cluster,
                "head_count": int(members.sum()),
            }
            for index, name in enumerate(accepted):
                row[f"source__{name}"] = float(
                    np.median(discovery_source[members, index])
                )
                row[f"coordinate__{name}"] = float(
                    np.median(discovery_coordinate[members, index])
                )
            center_rows.append(row)

    report = {
        "version": 1,
        "coordinate_system": args.coordinate_system,
        "layer_residual_definition": (
            "split-local per-layer median subtraction"
            if args.coordinate_system == "layer_residual"
            else None
        ),
        "minimum_feature_split_spearman": args.min_feature_split_rho,
        "excluded_feature_groups": sorted(excluded_groups),
        "accepted_features": accepted,
        "feature_count": len(accepted),
        "selected_clusters": selected,
        "selection_status": (
            "validated_candidate" if selected is not None else "no_stable_k"
        ),
        "functional_claim_admissible": False,
        "functional_claim_reason": (
            "cluster stability is descriptive; cache interventions must show "
            "cluster-specific causal policy demand before role names are assigned"
        ),
        "scaling": {
            name: {
                "discovery_median": float(center[index]),
                "discovery_scale": float(scale[index]),
                "conceptual_group": feature_groups[index],
                "group_balance_weight": float(feature_weights[index]),
            }
            for index, name in enumerate(accepted)
        },
        "conceptual_group_counts": dict(sorted(group_counts.items())),
        "diagnostics": diagnostics,
        "reference_comparisons": reference_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "feature_audit.csv", feature_audit)
    _write_csv(
        args.output_dir / "head_feature_matrix.csv",
        feature_matrix_rows,
    )
    _write_csv(
        args.output_dir / "feature_correlations.csv",
        correlation_rows,
    )
    _write_csv(args.output_dir / "cluster_diagnostics.csv", diagnostics)
    optional_outputs = (
        args.output_dir / "head_cluster_assignments.csv",
        args.output_dir / "cluster_feature_medians.csv",
        args.output_dir / "reference_comparisons.csv",
    )
    for path in optional_outputs:
        path.unlink(missing_ok=True)
    if assignment_rows:
        _write_csv(args.output_dir / "head_cluster_assignments.csv", assignment_rows)
        _write_csv(args.output_dir / "cluster_feature_medians.csv", center_rows)
    if reference_rows:
        _write_csv(args.output_dir / "reference_comparisons.csv", reference_rows)
    (args.output_dir / "clustering_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# v143 Multi-axis Head Taxonomy",
        "",
        f"- Accepted split-stable features: `{len(accepted)}`",
        f"- Coordinate system: `{args.coordinate_system}`",
        f"- Selected k: `{selected}`",
        f"- Status: `{report['selection_status']}`",
        "- PF and other published labels are post-hoc references only.",
        "- Functional role names remain blocked until causal cache routing passes.",
        "",
        "## k diagnostics",
        "",
        "| k | split agreement | split ARI | silhouette | bootstrap ARI | "
        "min class | layer-band NMI | passed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        summary.append(
            f"| {row['clusters']} | {row['split_label_agreement']:.4f} | "
            f"{row['split_ari']:.4f} | {row['discovery_silhouette']:.4f} | "
            f"{row['bootstrap_ari_median']:.4f} | "
            f"{row['minimum_cluster_fraction']:.4f} | "
            f"{row['cluster_layer_band_nmi']:.4f} | {row['passed']} |"
        )
    (args.output_dir / "clustering_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v136-analysis-dir",
        type=Path,
        default=Path("runs/v134_head_discovery/analysis_multi_axis_v136"),
    )
    parser.add_argument(
        "--v138-analysis-dir",
        type=Path,
        default=Path("runs/v138_history_interventions_v2/analysis"),
    )
    parser.add_argument(
        "--v143-analysis-dir",
        type=Path,
        default=Path("runs/v143_multiaxis_profile/analysis"),
    )
    parser.add_argument(
        "--pf-labels",
        type=Path,
        default=Path(
            "third_party/Pyramid-Forcing/configs/head_configs/best_labels.csv"
        ),
    )
    parser.add_argument(
        "--v98-labels",
        type=Path,
        default=Path("configs/head_maps/legacy_v98_absolute_sign_304_56.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coordinate-system",
        choices=("raw", "layer_residual"),
        default="raw",
        help=(
            "cluster global raw axes or split-local within-layer residuals; "
            "the latter tests head identity beyond layer-wide effects"
        ),
    )
    parser.add_argument("--min-feature-split-rho", type=float, default=0.30)
    parser.add_argument(
        "--exclude-feature-group",
        action="append",
        default=[],
        choices=FEATURE_GROUPS,
        help=(
            "exclude one conceptual feature family; repeat for multiple "
            "families"
        ),
    )
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    report = analyze(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
