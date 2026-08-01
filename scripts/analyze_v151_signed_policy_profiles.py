#!/usr/bin/env python3
"""Recover signed cache-policy preference responses from v145 profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.analyze_v145_crossed_seed_head_profiles import (
        EPSILON,
        FACTOR_VARIANTS,
        FAMILIES,
        HEADS,
        LAYERS,
        REPLICATES,
        _layer_residual,
        _load_profiles,
        _profile_index,
        _spearman,
        _write_csv,
    )
except ModuleNotFoundError:
    from analyze_v145_crossed_seed_head_profiles import (
        EPSILON,
        FACTOR_VARIANTS,
        FAMILIES,
        HEADS,
        LAYERS,
        REPLICATES,
        _layer_residual,
        _load_profiles,
        _profile_index,
        _spearman,
        _write_csv,
    )


POLICY_NAMES = (
    "boundary_recent",
    "current_only",
    "recent4",
    "recent_budget",
    "uniform_recent",
)
CONTRASTS = {
    "uniform_vs_recent": ("uniform_recent", "recent_budget"),
    "boundary_vs_recent": ("boundary_recent", "recent_budget"),
}
SELECTED_FACTOR = "scene"
SELECTED_CONTRAST = "uniform_vs_recent"
SELECTED_MODE = "noisy"
SELECTED_FRAME = 117
SELECTED_TIMESTEP = 500
PER_LAYER_COUNT = 4
MIN_SPLIT_RHO = 0.30
MIN_SEED_RHO = 0.30
MIN_VALIDATION_RATIO = 1.10
MIN_POSITIVE_LAYER_FRACTION = 0.70
MIN_PARAPHRASE_RATIO = 1.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_errors(record: dict) -> dict[str, torch.Tensor]:
    metrics = record.get("causal_policy_metrics") or {}
    if tuple(sorted(metrics)) != POLICY_NAMES:
        raise RuntimeError(
            "v151 expected policy candidates "
            f"{POLICY_NAMES}, found {tuple(sorted(metrics))}"
        )
    errors = {
        name: metrics[name]["projected_relative_error"].float()
        for name in POLICY_NAMES
    }
    if any(value.shape != (HEADS,) for value in errors.values()):
        raise RuntimeError("v151 policy error vector has the wrong head count")
    if any(
        not bool(torch.isfinite(value).all()) or bool((value < 0).any())
        for value in errors.values()
    ):
        raise RuntimeError("v151 policy errors must be finite and non-negative")
    return errors


def _preference(
    record: dict, *, left: str, right: str
) -> torch.Tensor:
    """Positive values mean the left policy is closer to full history."""

    errors = _policy_errors(record)
    return (errors[right] + EPSILON).log() - (errors[left] + EPSILON).log()


def signed_policy_observations(indexed: dict) -> list[dict]:
    rows = []
    for family in range(FAMILIES):
        split = "discovery" if family % 2 == 0 else "validation"
        for replicate in REPLICATES:
            base_records = indexed[(family, replicate, "base")]["records"]
            for factor in FACTOR_VARIANTS:
                variant_records = indexed[(family, replicate, factor)][
                    "records"
                ]
                if set(base_records) != set(variant_records):
                    raise RuntimeError(
                        f"state mismatch for family={family} factor={factor}"
                    )
                for state in sorted(base_records):
                    mode, frame, timestep, layer = state
                    base = base_records[state]
                    variant = variant_records[state]
                    for contrast, (left, right) in CONTRASTS.items():
                        base_preference = _preference(
                            base, left=left, right=right
                        )
                        variant_preference = _preference(
                            variant, left=left, right=right
                        )
                        response = variant_preference - base_preference
                        for head in range(HEADS):
                            value = float(response[head])
                            rows.append(
                                {
                                    "family_index": family,
                                    "family_split": split,
                                    "seed_replicate": replicate,
                                    "factor": factor,
                                    "contrast": contrast,
                                    "left_policy": left,
                                    "right_policy": right,
                                    "mode": mode,
                                    "current_frame": frame,
                                    "nominal_timestep": timestep,
                                    "layer": layer,
                                    "head": head,
                                    "base_preference": float(
                                        base_preference[head]
                                    ),
                                    "variant_preference": float(
                                        variant_preference[head]
                                    ),
                                    "signed_response": value,
                                    "absolute_response": abs(value),
                                }
                            )
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("v151 encountered an empty aggregation cell")
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise RuntimeError("v151 aggregation produced a non-finite value")
    return result


def aggregate_head_scores(observations: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in observations:
        key = (
            row["factor"],
            row["contrast"],
            row["mode"],
            int(row["current_frame"]),
            int(row["nominal_timestep"]),
            int(row["layer"]),
            int(row["head"]),
        )
        grouped[key].append(row)

    output = []
    for key, values in sorted(grouped.items()):
        factor, contrast, mode, frame, timestep, layer, head = key
        if len(values) != FAMILIES * len(REPLICATES):
            raise RuntimeError(f"v151 incomplete score cell {key}: {len(values)}")
        row = {
            "factor": factor,
            "contrast": contrast,
            "mode": mode,
            "current_frame": frame,
            "nominal_timestep": timestep,
            "layer": layer,
            "head": head,
        }
        for split in ("discovery", "validation", "all"):
            split_values = (
                values
                if split == "all"
                else [value for value in values if value["family_split"] == split]
            )
            row[f"{split}_signed_mean"] = _mean(
                [float(value["signed_response"]) for value in split_values]
            )
            row[f"{split}_abs_mean"] = _mean(
                [float(value["absolute_response"]) for value in split_values]
            )
            row[f"{split}_positive_fraction"] = _mean(
                [float(value["signed_response"] > 0) for value in split_values]
            )
        for replicate in REPLICATES:
            replicate_values = [
                value
                for value in values
                if int(value["seed_replicate"]) == replicate
            ]
            row[f"seed{replicate}_signed_mean"] = _mean(
                [float(value["signed_response"]) for value in replicate_values]
            )
            row[f"seed{replicate}_abs_mean"] = _mean(
                [float(value["absolute_response"]) for value in replicate_values]
            )
        family_sign_matches = []
        for family in range(FAMILIES):
            pair = sorted(
                (
                    value
                    for value in values
                    if int(value["family_index"]) == family
                ),
                key=lambda value: int(value["seed_replicate"]),
            )
            if len(pair) != 2:
                raise RuntimeError(f"v151 incomplete seed pair for {key}")
            left = float(pair[0]["signed_response"])
            right = float(pair[1]["signed_response"])
            family_sign_matches.append(
                float(left == 0 or right == 0 or left * right > 0)
            )
        row["seed_sign_agreement"] = _mean(family_sign_matches)
        output.append(row)
    return output


def _selected_rows(
    head_rows: list[dict],
    *,
    factor: str,
    contrast: str,
    mode: str,
    frame: int,
    timestep: int,
) -> list[dict]:
    selected = [
        row
        for row in head_rows
        if row["factor"] == factor
        and row["contrast"] == contrast
        and row["mode"] == mode
        and int(row["current_frame"]) == frame
        and int(row["nominal_timestep"]) == timestep
    ]
    if len(selected) != LAYERS * HEADS:
        raise RuntimeError(
            "v151 expected a complete 30x12 head grid for "
            f"{factor}/{contrast}/{mode}/f{frame}/t{timestep}, "
            f"found {len(selected)}"
        )
    return sorted(selected, key=lambda row: (int(row["layer"]), int(row["head"])))


def build_signed_maps(head_rows: list[dict]) -> tuple[dict, dict]:
    selected = _selected_rows(
        head_rows,
        factor=SELECTED_FACTOR,
        contrast=SELECTED_CONTRAST,
        mode=SELECTED_MODE,
        frame=SELECTED_FRAME,
        timestep=SELECTED_TIMESTEP,
    )
    by_coordinate = {
        (int(row["layer"]), int(row["head"])): row for row in selected
    }
    maps, diagnostics = build_signed_maps_without_audit(selected)
    diagnostics["group_scores"] = {}
    for group, head_map in maps.items():
        coordinates = [
            (layer, head)
            for layer in range(LAYERS)
            for head in head_map[str(layer)]
        ]
        diagnostics["group_scores"][group] = {
            field: _mean(
                [float(by_coordinate[coordinate][field]) for coordinate in coordinates]
            )
            for field in (
                "discovery_abs_mean",
                "validation_abs_mean",
                "discovery_signed_mean",
                "validation_signed_mean",
            )
        }
    return maps, diagnostics


def feature_audit(head_rows: list[dict]) -> list[dict]:
    lookup = {
        (
            row["factor"],
            row["contrast"],
            row["mode"],
            int(row["current_frame"]),
            int(row["nominal_timestep"]),
            int(row["layer"]),
            int(row["head"]),
        ): row
        for row in head_rows
    }
    contexts = sorted(
        {
            (
                row["factor"],
                row["contrast"],
                row["mode"],
                int(row["current_frame"]),
                int(row["nominal_timestep"]),
            )
            for row in head_rows
        }
    )
    audits = []
    for factor, contrast, mode, frame, timestep in contexts:
        rows = _selected_rows(
            head_rows,
            factor=factor,
            contrast=contrast,
            mode=mode,
            frame=frame,
            timestep=timestep,
        )
        discovery = np.asarray(
            [float(row["discovery_abs_mean"]) for row in rows]
        )
        validation = np.asarray(
            [float(row["validation_abs_mean"]) for row in rows]
        )
        seed0 = np.asarray([float(row["seed0_abs_mean"]) for row in rows])
        seed1 = np.asarray([float(row["seed1_abs_mean"]) for row in rows])
        signed_discovery = np.asarray(
            [float(row["discovery_signed_mean"]) for row in rows]
        )
        signed_validation = np.asarray(
            [float(row["validation_signed_mean"]) for row in rows]
        )
        _, map_diagnostics = build_signed_maps_without_audit(rows)
        paraphrase = np.asarray(
            [
                float(
                    lookup[
                        (
                            "paraphrase",
                            contrast,
                            mode,
                            frame,
                            timestep,
                            int(row["layer"]),
                            int(row["head"]),
                        )
                    ]["validation_abs_mean"]
                )
                for row in rows
            ]
        )
        paraphrase_ratio = float(
            np.median((validation + EPSILON) / (paraphrase + EPSILON))
        )
        split_rho = _spearman(
            _layer_residual(discovery), _layer_residual(validation)
        )
        seed_rho = _spearman(
            _layer_residual(seed0), _layer_residual(seed1)
        )
        signed_split_rho = _spearman(
            _layer_residual(signed_discovery),
            _layer_residual(signed_validation),
        )
        screen = bool(
            split_rho >= MIN_SPLIT_RHO
            and seed_rho >= MIN_SEED_RHO
            and map_diagnostics["validation_high_low_ratio"]
            >= MIN_VALIDATION_RATIO
            and map_diagnostics["validation_positive_layer_fraction"]
            >= MIN_POSITIVE_LAYER_FRACTION
            and (factor == "paraphrase" or paraphrase_ratio >= MIN_PARAPHRASE_RATIO)
        )
        audits.append(
            {
                "factor": factor,
                "contrast": contrast,
                "mode": mode,
                "current_frame": frame,
                "nominal_timestep": timestep,
                "layer_residual_family_split_spearman": split_rho,
                "layer_residual_seed_replicate_spearman": seed_rho,
                "layer_residual_signed_split_spearman": signed_split_rho,
                "validation_high_low_ratio": map_diagnostics[
                    "validation_high_low_ratio"
                ],
                "validation_positive_layer_fraction": map_diagnostics[
                    "validation_positive_layer_fraction"
                ],
                "validation_factor_paraphrase_ratio": paraphrase_ratio,
                "candidate_screen_pass": int(screen),
            }
        )
    return audits


def build_signed_maps_without_audit(selected_rows: list[dict]) -> tuple[dict, dict]:
    if len(selected_rows) != LAYERS * HEADS:
        raise RuntimeError("v151 axis map requires exactly 360 head rows")
    by_coordinate = {
        (int(row["layer"]), int(row["head"])): row for row in selected_rows
    }
    maps = {"low4": {}, "middle4": {}, "high4": {}}
    validation_log_ratios = []
    for layer in range(LAYERS):
        ordered = sorted(
            range(HEADS),
            key=lambda head: (
                float(by_coordinate[(layer, head)]["discovery_abs_mean"]),
                head,
            ),
        )
        low = sorted(ordered[:PER_LAYER_COUNT])
        middle = sorted(ordered[PER_LAYER_COUNT : 2 * PER_LAYER_COUNT])
        high = sorted(ordered[-PER_LAYER_COUNT:])
        groups = {"low4": low, "middle4": middle, "high4": high}
        for group, heads in groups.items():
            maps[group][str(layer)] = heads
        high_score = _mean(
            [
                float(by_coordinate[(layer, head)]["validation_abs_mean"])
                for head in high
            ]
        )
        low_score = _mean(
            [
                float(by_coordinate[(layer, head)]["validation_abs_mean"])
                for head in low
            ]
        )
        validation_log_ratios.append(
            math.log((high_score + EPSILON) / (low_score + EPSILON))
        )
    diagnostics = {
        "validation_high_low_median_log_ratio": float(
            np.median(validation_log_ratios)
        ),
        "validation_high_low_ratio": math.exp(
            float(np.median(validation_log_ratios))
        ),
        "validation_positive_layer_fraction": _mean(
            [float(value > 0) for value in validation_log_ratios]
        ),
    }
    return maps, diagnostics


def analyze(profile_dir: Path, output_dir: Path, expected_count: int) -> dict:
    indexed, profile_audit = _profile_index(
        _load_profiles(profile_dir, expected_count)
    )
    observations = signed_policy_observations(indexed)
    head_rows = aggregate_head_scores(observations)
    audits = feature_audit(head_rows)
    maps, map_diagnostics = build_signed_maps(head_rows)
    selected_audit = [
        row
        for row in audits
        if row["factor"] == SELECTED_FACTOR
        and row["contrast"] == SELECTED_CONTRAST
        and row["mode"] == SELECTED_MODE
        and int(row["current_frame"]) == SELECTED_FRAME
        and int(row["nominal_timestep"]) == SELECTED_TIMESTEP
    ]
    if len(selected_audit) != 1:
        raise RuntimeError("v151 selected signed-policy audit is missing")
    source_screen_pass = bool(selected_audit[0]["candidate_screen_pass"])
    map_payload = {
        "version": 1,
        "selection_uses": "discovery families only",
        "axis": {
            "factor": SELECTED_FACTOR,
            "contrast": SELECTED_CONTRAST,
            "mode": SELECTED_MODE,
            "current_frame": SELECTED_FRAME,
            "nominal_timestep": SELECTED_TIMESTEP,
            "score": "mean absolute signed log-error-ratio response",
        },
        "source_screen_pass": source_screen_pass,
        "screen": selected_audit[0],
        "maps": maps,
        "diagnostics": map_diagnostics,
        "claim_boundary": (
            "This map is an observational candidate. Validation-family "
            "screening does not establish downstream causal leverage."
        ),
    }
    report = {
        "version": 1,
        "profile_count": len(indexed),
        "family_count": FAMILIES,
        "seed_replicates": list(REPLICATES),
        "policy_names": list(POLICY_NAMES),
        "contrasts": {key: list(value) for key, value in CONTRASTS.items()},
        "state_signed_observation_count": len(observations),
        "head_score_count": len(head_rows),
        "feature_audit_count": len(audits),
        "selected_axis": map_payload["axis"],
        "selected_axis_screen": selected_audit[0],
        "source_screen_pass": source_screen_pass,
        "thresholds": {
            "minimum_family_split_spearman": MIN_SPLIT_RHO,
            "minimum_seed_replicate_spearman": MIN_SEED_RHO,
            "minimum_validation_high_low_ratio": MIN_VALIDATION_RATIO,
            "minimum_positive_layer_fraction": MIN_POSITIVE_LAYER_FRACTION,
            "minimum_factor_paraphrase_ratio": MIN_PARAPHRASE_RATIO,
        },
        "claim_boundary": map_payload["claim_boundary"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_contract_audit.csv", profile_audit)
    _write_csv(
        output_dir / "state_signed_policy_responses.csv.gz", observations
    )
    _write_csv(output_dir / "signed_policy_head_scores.csv.gz", head_rows)
    _write_csv(output_dir / "signed_policy_feature_audit.csv", audits)
    map_path = output_dir / "signed_scene_uniform_maps.json"
    map_path.write_text(
        json.dumps(map_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["map_sha256"] = _sha256(map_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# v151 Signed Policy Preference Profiling",
                "",
                f"- Profiles: `{report['profile_count']}`",
                f"- Signed observations: `{len(observations)}`",
                f"- Selected source screen: `{source_screen_pass}`",
                "",
                "The selected scene/uniform-vs-recent map uses discovery "
                "families only. Held-out MovieBench causal confirmation is "
                "required before assigning a functional role.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=160)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.profile_dir, args.output_dir, args.expected_count),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
