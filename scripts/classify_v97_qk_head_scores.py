#!/usr/bin/env python3
"""Generate reproducible head maps from an immutable QK score artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import build_v96_qk_head_thresholds as qk


HeadKey = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--score-artifact", required=True, type=Path)
    parser.add_argument("--pf-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument(
        "--manual-thresholds",
        default="0.0,0.5,1.0,1.5,2.0",
    )
    parser.add_argument("--main-threshold", type=float, default=1.0)
    parser.add_argument("--sign-thresholds", default="0.5")
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_floats(value: str) -> list[float]:
    result = []
    for item in str(value).split(","):
        if item.strip():
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"non-finite threshold: {item}")
            result.append(number)
    if not result:
        raise ValueError("at least one threshold is required")
    return sorted(set(result))


def threshold_slug(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def load_scores(
    path: Path,
    num_layers: int,
    num_heads: int,
) -> tuple[list[dict], dict[HeadKey, dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = num_layers * num_heads
    if len(rows) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} score rows, found {len(rows)}"
        )
    result: dict[HeadKey, dict] = {}
    numeric_fields = {
        "cfg_raw",
        "semantic_raw",
        "cfg_score",
        "semantic_score",
        "consensus_score",
        "positive_rate",
        "mean_logit",
        "mean_abs_logit",
        "signed_logit_mass",
        "sign_switch_rate",
        "dominant_period",
        "spectral_peak_ratio",
    }
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        if key in result:
            raise ValueError(f"{path}: duplicate score row {key}")
        parsed = dict(row)
        for field in numeric_fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite {field} for {key}")
            parsed[field] = value
        result[key] = parsed
    expected = {
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(f"{path}: missing={missing[:8]} extra={extra[:8]}")
    return rows, result


def matrix_from_threshold(
    scores: dict[HeadKey, dict],
    column: str,
    threshold: float,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    return [
        [
            1 if float(scores[(layer, head)][column]) <= threshold else -1
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]


def matrix_from_sign_rate(
    scores: dict[HeadKey, dict],
    threshold: float,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    return [
        [
            (
                1
                if float(scores[(layer, head)]["positive_rate"]) >= threshold
                else -1
            )
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]


def count_matched_control(
    reference: list[list[int]],
    scores: dict[HeadKey, dict],
    *,
    mode: str,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    result = []
    for layer, row in enumerate(reference):
        responsive_count = row.count(-1)
        heads = list(range(len(row)))
        if mode == "random":
            rng.shuffle(heads)
        elif mode == "reversed":
            heads.sort(
                key=lambda head: float(
                    scores[(layer, head)]["consensus_score"]
                )
            )
        else:
            raise ValueError(f"unsupported control mode: {mode}")
        responsive = set(heads[:responsive_count])
        result.append(
            [-1 if head in responsive else 1 for head in range(len(row))]
        )
    return result


def pf_cross_tab(
    matrix: list[list[int]],
    pf_labels: list[list[int]],
) -> dict[str, dict[str, int]]:
    result = {}
    names = {-1: "wave", 1: "anchor", 2: "veil"}
    for label, name in names.items():
        values = [
            matrix[layer][head]
            for layer, row in enumerate(pf_labels)
            for head, value in enumerate(row)
            if value == label
        ]
        result[name] = {
            "pf_label": label,
            "stable_or_support": values.count(1),
            "responsive_or_suppress": values.count(-1),
        }
    return result


def map_agreement(
    left: list[list[int]],
    right: list[list[int]],
) -> dict[str, float]:
    left_flat = qk.flatten(left)
    right_flat = qk.flatten(right)
    responsive_left = {
        index for index, value in enumerate(left_flat) if value == -1
    }
    responsive_right = {
        index for index, value in enumerate(right_flat) if value == -1
    }
    union = responsive_left | responsive_right
    return {
        "label_agreement": sum(
            left_value == right_value
            for left_value, right_value in zip(left_flat, right_flat)
        )
        / len(left_flat),
        "responsive_jaccard": (
            len(responsive_left & responsive_right) / len(union)
            if union
            else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(args.score_artifact.read_text(encoding="utf-8"))
    expected_hash = artifact.get("files", {}).get("score_csv_sha256")
    actual_hash = sha256(args.scores)
    if expected_hash != actual_hash:
        raise ValueError(
            "score CSV does not match immutable score artifact: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    if artifact.get("score_definition", {}).get("classification") is not None:
        raise ValueError("score artifact unexpectedly contains classification")
    if int(artifact.get("head_count", 0)) != args.num_layers * args.num_heads:
        raise ValueError("score artifact has an invalid head count")

    _, scores = load_scores(
        args.scores, args.num_layers, args.num_heads
    )
    pf_labels = qk.read_matrix(
        args.pf_labels, args.num_layers, args.num_heads
    )
    manual_thresholds = parse_floats(args.manual_thresholds)
    sign_thresholds = parse_floats(args.sign_thresholds)
    if args.main_threshold not in manual_thresholds:
        raise ValueError(
            "main threshold must be included in --manual-thresholds"
        )

    maps: dict[str, list[list[int]]] = {}
    map_sources: dict[str, dict] = {}
    threshold_rows = []
    for threshold in manual_thresholds:
        name = f"prompt_tau_{threshold_slug(threshold)}"
        matrix = matrix_from_threshold(
            scores,
            "consensus_score",
            threshold,
            args.num_layers,
            args.num_heads,
        )
        maps[name] = matrix
        map_sources[name] = {
            "family": "prompt_consensus_manual_threshold",
            "score_column": "consensus_score",
            "threshold": threshold,
        }
        flat = qk.flatten(matrix)
        threshold_rows.append(
            {
                "map": name,
                "score_column": "consensus_score",
                "threshold": threshold,
                "stable_or_support": flat.count(1),
                "responsive_or_suppress": flat.count(-1),
                "minority_fraction": min(flat.count(1), flat.count(-1))
                / len(flat),
            }
        )

    consensus_values = [
        float(scores[key]["consensus_score"]) for key in sorted(scores)
    ]
    gmm_models = [
        qk.fit_gmm_1d(consensus_values, components)
        for components in (1, 2, 3)
    ]
    automatic_thresholds = {
        "prompt_gmm2": qk.gmm_threshold(gmm_models[1]),
        "prompt_otsu": qk.otsu_threshold(consensus_values),
    }
    for name, threshold in automatic_thresholds.items():
        maps[name] = matrix_from_threshold(
            scores,
            "consensus_score",
            threshold,
            args.num_layers,
            args.num_heads,
        )
        map_sources[name] = {
            "family": "prompt_consensus_automatic_threshold",
            "score_column": "consensus_score",
            "threshold": threshold,
        }

    main_name = f"prompt_tau_{threshold_slug(args.main_threshold)}"
    maps[f"{main_name}_random"] = count_matched_control(
        maps[main_name],
        scores,
        mode="random",
        seed=args.random_seed,
    )
    map_sources[f"{main_name}_random"] = {
        "family": "count_matched_layerwise_random_control",
        "reference": main_name,
        "seed": args.random_seed,
    }
    maps[f"{main_name}_reversed"] = count_matched_control(
        maps[main_name],
        scores,
        mode="reversed",
        seed=args.random_seed,
    )
    map_sources[f"{main_name}_reversed"] = {
        "family": "count_matched_layerwise_reversed_control",
        "reference": main_name,
    }

    for threshold in sign_thresholds:
        name = f"sign_rpos_{threshold_slug(threshold)}"
        maps[name] = matrix_from_sign_rate(
            scores, threshold, args.num_layers, args.num_heads
        )
        map_sources[name] = {
            "family": "positive_logit_fraction",
            "score_column": "positive_rate",
            "threshold": threshold,
            "support_rule": "positive_rate >= threshold",
        }

    maps["pf_anchor_vs_rest"] = [
        [1 if value == 1 else -1 for value in row]
        for row in pf_labels
    ]
    map_sources["pf_anchor_vs_rest"] = {
        "family": "pf_label_merge_control",
        "support": ["anchor"],
        "suppress": ["wave", "veil"],
    }
    maps["pf_anchor_wave_vs_veil"] = [
        [-1 if value == 2 else 1 for value in row]
        for row in pf_labels
    ]
    map_sources["pf_anchor_wave_vs_veil"] = {
        "family": "pf_label_merge_control",
        "support": ["anchor", "wave"],
        "suppress": ["veil"],
    }
    maps["pf_native"] = [list(row) for row in pf_labels]
    map_sources["pf_native"] = {
        "family": "pf_native_reference",
        "source": str(args.pf_labels.resolve()),
        "source_sha256": sha256(args.pf_labels),
    }
    for label, class_name in ((1, "anchor"), (-1, "wave"), (2, "veil")):
        name = f"pf_{class_name}_extended_recent"
        maps[name] = [
            [3 if value == label else value for value in row]
            for row in pf_labels
        ]
        map_sources[name] = {
            "family": "pf_class_contribution_ablation",
            "target_class": class_name,
            "target_label": label,
            "replacement_label": 3,
            "replacement_policy": (
                "same_sink_plus_recent5_no_middle"
                if class_name == "veil"
                else "same_sink_plus_recent8_no_middle"
            ),
        }

    manifest_maps = {}
    for name, matrix in maps.items():
        path = args.output_dir / f"{name}.csv"
        qk.write_matrix(path, matrix)
        flat = qk.flatten(matrix)
        valid_labels = {-1, 1} if not name.startswith("pf_") else {-1, 1, 2, 3}
        if not set(flat).issubset(valid_labels):
            raise ValueError(f"{name}: unexpected labels {sorted(set(flat))}")
        manifest_maps[name] = {
            **map_sources[name],
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "label_counts": dict(sorted(Counter(flat).items())),
            "per_layer_label_counts": [
                dict(sorted(Counter(row).items())) for row in matrix
            ],
            "pf_cross_tab": (
                pf_cross_tab(matrix, pf_labels)
                if set(flat).issubset({-1, 1})
                else None
            ),
        }

    binary_names = [
        name
        for name, matrix in maps.items()
        if set(qk.flatten(matrix)).issubset({-1, 1})
    ]
    agreement_payload = {}
    agreement_csv = args.output_dir / "map_agreement.csv"
    with agreement_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "left",
                "right",
                "label_agreement",
                "responsive_jaccard",
            ),
        )
        writer.writeheader()
        for left_index, left in enumerate(binary_names):
            for right in binary_names[left_index + 1 :]:
                item = map_agreement(maps[left], maps[right])
                agreement_payload[f"{left}__{right}"] = item
                writer.writerow({"left": left, "right": right, **item})

    threshold_csv = args.output_dir / "threshold_sweep.csv"
    with threshold_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(threshold_rows[0]),
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    bic = [model["bic"] for model in gmm_models]
    report = {
        "version": 1,
        "method": "offline_head_map_classification_from_immutable_scores",
        "score_artifact": str(args.score_artifact.resolve()),
        "score_csv": str(args.scores.resolve()),
        "score_csv_sha256": actual_hash,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "manual_thresholds": manual_thresholds,
        "main_threshold": args.main_threshold,
        "sign_thresholds": sign_thresholds,
        "automatic_thresholds": automatic_thresholds,
        "gmm": gmm_models,
        "gmm_gates": {
            "bic_1_minus_2": bic[0] - bic[1],
            "bic_3_minus_2": bic[2] - bic[1],
            "two_components_preferred_to_one": bic[0] - bic[1] >= 10.0,
            "two_components_preferred_to_three": bic[1] <= bic[2],
        },
        "maps": manifest_maps,
        "map_agreement": agreement_payload,
        "classification_is_posthoc": True,
        "pf_labels_used_to_construct_prompt_maps": False,
    }
    report_path = args.output_dir / "head_map_classification_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = args.output_dir / "head_map_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "score_csv_sha256": actual_hash,
                "classification_report": str(report_path.resolve()),
                "maps": manifest_maps,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[HeadMapClassification] "
        f"scores={actual_hash} maps={len(maps)} "
        f"manual={manual_thresholds} sign={sign_thresholds} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
