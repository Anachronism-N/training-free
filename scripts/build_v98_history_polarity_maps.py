#!/usr/bin/env python3
"""Build PF-independent history-polarity maps from frozen QK statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


SUPPORT_LABEL = 10
SUPPRESS_LABEL = 11
PF_NAMES = {-1: "wave", 1: "anchor", 2: "veil"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--score-artifact", required=True, type=Path)
    parser.add_argument("--pf-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--polarity-thresholds", default="-0.1,0,0.1")
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted(
        {
            float(item.strip())
            for item in str(value).split(",")
            if item.strip()
        }
    )
    if not thresholds or any(not math.isfinite(value) for value in thresholds):
        raise ValueError("polarity thresholds must be finite")
    if 0.0 not in thresholds:
        raise ValueError("polarity thresholds must include the natural zero split")
    return thresholds


def threshold_slug(value: float) -> str:
    if value == 0:
        return "zero"
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def read_matrix(
    path: Path,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != num_layers:
        raise ValueError(
            f"{path}: expected {num_layers} layers, found {len(rows)}"
        )
    for layer, row in enumerate(rows):
        if len(row) != num_heads:
            raise ValueError(
                f"{path}: layer {layer} expected {num_heads} heads, "
                f"found {len(row)}"
            )
    return rows


def read_scores(
    path: Path,
    num_layers: int,
    num_heads: int,
) -> dict[tuple[int, int], dict[str, object]]:
    numeric_fields = {
        "consensus_score",
        "positive_rate",
        "mean_logit",
        "mean_abs_logit",
        "signed_logit_mass",
        "sign_switch_rate",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = num_layers * num_heads
    if len(rows) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} heads, found {len(rows)}"
        )

    result: dict[tuple[int, int], dict[str, object]] = {}
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        if key in result:
            raise ValueError(f"{path}: duplicate head {key}")
        parsed: dict[str, object] = dict(row)
        for field in numeric_fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite {field} for head {key}")
            parsed[field] = value
        signed_mass = float(parsed["signed_logit_mass"])
        if not -1.0 - 1e-6 <= signed_mass <= 1.0 + 1e-6:
            raise ValueError(
                f"{path}: signed_logit_mass outside [-1, 1] for head {key}: "
                f"{signed_mass}"
            )
        result[key] = parsed

    expected = {
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    }
    if set(result) != expected:
        raise ValueError(
            f"{path}: incomplete layer/head grid, "
            f"missing={sorted(expected - set(result))[:8]}"
        )
    return result


def score_map(
    scores: dict[tuple[int, int], dict[str, object]],
    *,
    column: str,
    threshold: float,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    return [
        [
            (
                SUPPORT_LABEL
                if float(scores[(layer, head)][column]) >= threshold
                else SUPPRESS_LABEL
            )
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def random_count_control(
    reference: list[list[int]],
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    result: list[list[int]] = []
    for row in reference:
        support_count = row.count(SUPPORT_LABEL)
        heads = list(range(len(row)))
        rng.shuffle(heads)
        support = set(heads[:support_count])
        result.append(
            [
                SUPPORT_LABEL if head in support else SUPPRESS_LABEL
                for head in range(len(row))
            ]
        )
    return result


def pf_cross_tab(
    matrix: list[list[int]],
    pf_labels: list[list[int]],
) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for pf_label, name in PF_NAMES.items():
        roles = [
            matrix[layer][head]
            for layer, row in enumerate(pf_labels)
            for head, value in enumerate(row)
            if value == pf_label
        ]
        table[name] = {
            "pf_label": pf_label,
            "heads": len(roles),
            "history_supportive": roles.count(SUPPORT_LABEL),
            "history_suppressive": roles.count(SUPPRESS_LABEL),
        }
    return table


def binary_agreement(
    matrix: list[list[int]],
    reference: list[list[int]],
    *,
    positive_label: int,
) -> dict[str, float | int]:
    predicted = flatten(matrix)
    truth = flatten(reference)
    positive = {
        index for index, value in enumerate(predicted) if value == SUPPRESS_LABEL
    }
    reference_positive = {
        index for index, value in enumerate(truth) if value == positive_label
    }
    tp = len(positive & reference_positive)
    fp = len(positive - reference_positive)
    fn = len(reference_positive - positive)
    tn = len(predicted) - tp - fp - fn
    tpr = tp / (tp + fn) if tp + fn else 1.0
    tnr = tn / (tn + fp) if tn + fp else 1.0
    union = positive | reference_positive
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "agreement": (tp + tn) / len(predicted),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "suppressive_jaccard": (
            len(positive & reference_positive) / len(union) if union else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(args.score_artifact.read_text(encoding="utf-8"))
    expected_score_hash = artifact.get("files", {}).get("score_csv_sha256")
    actual_score_hash = sha256(args.scores)
    if expected_score_hash != actual_score_hash:
        raise ValueError(
            "score CSV does not match immutable artifact: "
            f"expected={expected_score_hash} actual={actual_score_hash}"
        )

    scores = read_scores(args.scores, args.num_layers, args.num_heads)
    pf_labels = read_matrix(args.pf_labels, args.num_layers, args.num_heads)
    thresholds = parse_thresholds(args.polarity_thresholds)

    maps: dict[str, list[list[int]]] = {}
    sources: dict[str, dict[str, object]] = {}
    for threshold in thresholds:
        name = f"history_polarity_{threshold_slug(threshold)}"
        maps[name] = score_map(
            scores,
            column="signed_logit_mass",
            threshold=threshold,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
        sources[name] = {
            "family": "history_net_support_polarity",
            "score": (
                "median_over_records("
                "sum_history_qk_logits / sum_abs_history_qk_logits)"
            ),
            "score_column": "signed_logit_mass",
            "support_rule": f"signed_logit_mass >= {threshold}",
            "threshold": threshold,
            "threshold_provenance": (
                "natural_zero_no_pf_labels"
                if threshold == 0
                else "symmetric_robustness_ablation"
            ),
        }

    maps["positive_rate_half"] = score_map(
        scores,
        column="positive_rate",
        threshold=0.5,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    sources["positive_rate_half"] = {
        "family": "history_positive_logit_fraction",
        "score_column": "positive_rate",
        "support_rule": "positive_rate >= 0.5",
        "threshold": 0.5,
        "threshold_provenance": "natural_majority_sign",
    }
    maps["pf_aw_binary_control"] = [
        [
            SUPPORT_LABEL if value in {-1, 1} else SUPPRESS_LABEL
            for value in row
        ]
        for row in pf_labels
    ]
    sources["pf_aw_binary_control"] = {
        "family": "pf_oracle_control_not_proposed_classifier",
        "support": ["anchor", "wave"],
        "suppress": ["veil"],
    }
    maps["pf_ar_binary_control"] = [
        [
            SUPPORT_LABEL if value == 1 else SUPPRESS_LABEL
            for value in row
        ]
        for row in pf_labels
    ]
    sources["pf_ar_binary_control"] = {
        "family": "pf_oracle_control_not_proposed_classifier",
        "support": ["anchor"],
        "suppress": ["wave", "veil"],
    }
    maps["history_polarity_zero_random"] = random_count_control(
        maps["history_polarity_zero"], args.random_seed
    )
    sources["history_polarity_zero_random"] = {
        "family": "layerwise_count_matched_random_control",
        "reference": "history_polarity_zero",
        "seed": args.random_seed,
    }

    manifest_maps: dict[str, dict[str, object]] = {}
    for name, matrix in maps.items():
        values = flatten(matrix)
        if set(values) != {SUPPORT_LABEL, SUPPRESS_LABEL}:
            raise ValueError(
                f"{name}: both neutral role labels must be represented"
            )
        path = args.output_dir / f"{name}.csv"
        write_matrix(path, matrix)
        manifest_maps[name] = {
            **sources[name],
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "label_counts": dict(sorted(Counter(values).items())),
            "pf_cross_tab": pf_cross_tab(matrix, pf_labels),
            "agreement_pf_aw": binary_agreement(
                matrix, pf_labels, positive_label=2
            ),
            "agreement_pf_ar": binary_agreement(
                matrix,
                [
                    [0 if value == 1 else 3 for value in row]
                    for row in pf_labels
                ],
                positive_label=3,
            ),
        }

    assignments_path = args.output_dir / "head_assignments.csv"
    assignment_fields = [
        "layer",
        "head",
        "pf_label",
        "pf_name",
        "mean_logit",
        "mean_abs_logit",
        "signed_logit_mass",
        "positive_rate",
        "sign_switch_rate",
        "consensus_score",
        *maps,
    ]
    with assignments_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fields)
        writer.writeheader()
        for layer in range(args.num_layers):
            for head in range(args.num_heads):
                score = scores[(layer, head)]
                pf_label = pf_labels[layer][head]
                writer.writerow(
                    {
                        "layer": layer,
                        "head": head,
                        "pf_label": pf_label,
                        "pf_name": PF_NAMES.get(pf_label, "unknown"),
                        "mean_logit": score["mean_logit"],
                        "mean_abs_logit": score["mean_abs_logit"],
                        "signed_logit_mass": score["signed_logit_mass"],
                        "positive_rate": score["positive_rate"],
                        "sign_switch_rate": score["sign_switch_rate"],
                        "consensus_score": score["consensus_score"],
                        **{
                            name: matrix[layer][head]
                            for name, matrix in maps.items()
                        },
                    }
                )

    manifest = {
        "version": 1,
        "method": "v98_history_polarity_map_builder",
        "support_label": SUPPORT_LABEL,
        "suppress_label": SUPPRESS_LABEL,
        "reserved_pf_labels": [-1, 1, 2],
        "score_csv": str(args.scores.resolve()),
        "score_csv_sha256": actual_score_hash,
        "score_artifact": str(args.score_artifact.resolve()),
        "score_artifact_sha256": sha256(args.score_artifact),
        "pf_labels": str(args.pf_labels.resolve()),
        "pf_labels_sha256": sha256(args.pf_labels),
        "thresholds": thresholds,
        "maps": manifest_maps,
        "assignments": {
            "path": str(assignments_path.resolve()),
            "sha256": sha256(assignments_path),
        },
        "claims": {
            "primary_classifier": "history_polarity_zero",
            "pf_labels_used_for_primary_classifier": False,
            "pf_labels_used_for_controls_and_posthoc_analysis": True,
            "prompt_sensitivity_used_as_static_classifier": False,
        },
    }
    manifest_path = args.output_dir / "history_polarity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[HistoryPolarityMaps] "
        f"scores={actual_score_hash} maps={len(maps)} "
        f"primary_counts={manifest_maps['history_polarity_zero']['label_counts']} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
