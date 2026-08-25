#!/usr/bin/env python3
"""Compare SF and Causal Head x Phase compatibility without generating video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_v189_structured_head_phase import (
    NpEncoder,
    _cell_rows,
    _mask,
    aggregate_operator,
    bootstrap_ci,
)
from prepare_v195_cross_checkpoint_head_phase_profile import (
    CALLS,
    HEADS,
    LAYERS,
    RANDOM_DRAWS,
    RANDOM_SEED,
    load_json,
    sha256,
    verify,
)

PROMPT_WIN_THRESHOLD = 0.60
RANDOM_P_THRESHOLD = 0.05


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("correlation arrays must have the same nontrivial shape")

    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    nonzero = (left != 0) | (right != 0)
    sign_agreement = (
        float(np.mean(np.sign(left[nonzero]) == np.sign(right[nonzero])))
        if np.any(nonzero)
        else None
    )
    return {
        "count": int(left.size),
        "pearson": corr(left, right),
        "spearman": corr(_rankdata(left), _rankdata(right)),
        "sign_agreement_nonzero": sign_agreement,
    }


def _structural_correlations(sf: np.ndarray, cf: np.ndarray) -> dict:
    if sf.shape != (CALLS, LAYERS, HEADS) or cf.shape != sf.shape:
        raise ValueError("v195 cell arrays must be 4 x 30 x 12")
    return {
        "exact_call_layer_head": _correlation(sf, cf),
        "phase_layer_mean_over_heads": _correlation(sf.mean(2), cf.mean(2)),
        "layer_head_mean_over_calls": _correlation(sf.mean(0), cf.mean(0)),
        "layer_mean_over_calls_heads": _correlation(
            sf.mean(axis=(0, 2)), cf.mean(axis=(0, 2))
        ),
        "phase_mean_over_layers_heads": _correlation(
            sf.mean(axis=(1, 2)), cf.mean(axis=(1, 2))
        ),
    }


def _random_control(
    values: np.ndarray,
    selected: np.ndarray,
    *,
    preserve_call_layer: bool,
    draws: int,
    seed: int,
) -> dict:
    if values.shape != (CALLS, LAYERS, HEADS) or selected.shape != values.shape:
        raise ValueError("random-control arrays must be 4 x 30 x 12")
    selected_count = int(selected.sum())
    if selected_count <= 0:
        raise ValueError("v195 frozen route selects no Coverage cells")
    observed = float(values[selected].mean())
    rng = np.random.default_rng(seed)
    random_means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = []
        for call in range(CALLS):
            if preserve_call_layer:
                for layer in range(LAYERS):
                    count = int(selected[call, layer].sum())
                    if count:
                        indices = rng.choice(HEADS, size=count, replace=False)
                        sampled.extend(values[call, layer, indices].tolist())
            else:
                count = int(selected[call].sum())
                if count:
                    indices = rng.choice(LAYERS * HEADS, size=count, replace=False)
                    sampled.extend(values[call].reshape(-1)[indices].tolist())
        if len(sampled) != selected_count:
            raise RuntimeError("v195 random control changed selected cell count")
        random_means[draw] = float(np.mean(sampled))
    greater_equal = int(np.sum(random_means >= observed))
    return {
        "preserve_call_layer": preserve_call_layer,
        "draws": draws,
        "seed": seed,
        "selected_cell_count": selected_count,
        "observed_mean_gain": observed,
        "random_mean": float(random_means.mean()),
        "random_ci95": [
            float(value) for value in np.quantile(random_means, [0.025, 0.975])
        ],
        "observed_percentile": float(np.mean(random_means < observed)),
        "one_sided_empirical_p": float((greater_equal + 1) / (draws + 1)),
    }


def _load_sf_cells(
    path: Path, operator: str
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["operator"] == operator]
    if len(rows) != CALLS * LAYERS * HEADS:
        raise ValueError("v195 did not find one complete SF operator in cell_scores")
    discovery = np.zeros((CALLS, LAYERS, HEADS), dtype=np.float64)
    validation = np.zeros_like(discovery)
    seen = set()
    for row in rows:
        key = (
            int(row["call_index"]),
            int(row["layer"]),
            int(row["head"]),
        )
        if key in seen:
            raise ValueError(f"duplicate SF cell score: {key}")
        seen.add(key)
        discovery[key] = float(row["discovery_gain"])
        validation[key] = float(row["validation_gain"])
    return discovery, validation, rows


def _load_mask(path: Path) -> np.ndarray:
    payload = load_json(path)
    mask = np.asarray(payload.get("coverage_masks"), dtype=np.bool_)
    if mask.shape != (CALLS, LAYERS, HEADS):
        raise ValueError("v195 frozen SF map is not 4 x 30 x 12")
    counts = mask.sum(axis=(1, 2)).astype(int).tolist()
    if counts != payload.get("coverage_count_by_call"):
        raise ValueError("v195 frozen SF map count metadata drifted")
    return mask


def _overlap(left: np.ndarray, right: np.ndarray) -> dict:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    left_count = int(left.sum())
    right_count = int(right.sum())
    return {
        "left_count": left_count,
        "right_count": right_count,
        "intersection": intersection,
        "jaccard": float(intersection / union) if union else 1.0,
        "left_recall": float(intersection / left_count) if left_count else None,
        "right_precision": float(intersection / right_count) if right_count else None,
    }


def _prompt_effects(
    gain: np.ndarray, mask: np.ndarray, prompt_ids: list[int]
) -> list[dict]:
    complement = ~mask
    rows = []
    for prompt in prompt_ids:
        selected_gain = float(gain[prompt][mask].mean())
        complement_gain = float(gain[prompt][complement].mean())
        rows.append(
            {
                "prompt_id": int(prompt),
                "selected_gain": selected_gain,
                "complement_gain": complement_gain,
                "selected_minus_complement": selected_gain - complement_gain,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty v195 table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _recommendation(v194_confirmed: bool, level: str) -> str:
    if v194_confirmed:
        return {
            "exact_head_identity": "freeze_route_with_cross_checkpoint_mechanistic_support",
            "phase_layer_structure": "limit_transfer_mechanism_claim_to_phase_layer_structure",
            "operator_only": "retain_generation_transfer_without_head_identity_claim",
            "unsupported": "inspect_shadow_metric_mismatch_before_any_new_generation",
        }[level]
    return {
        "exact_head_identity": "shadow_metric_transfers_but_generation_does_not_stop_and_diagnose",
        "phase_layer_structure": "reprofile_causal_membership_with_frozen_feature_standard",
        "operator_only": "classifier_is_checkpoint_specific_do_not_advance",
        "unsupported": "stop_cross_checkpoint_route_transfer",
    }[level]


def analyze(
    manifest_path: Path,
    profile_root: Path,
    profile_audit_path: Path,
    output_dir: Path,
) -> dict:
    manifest = verify(manifest_path)
    audit = load_json(profile_audit_path)
    if (
        audit.get("experiment") != "v195_cross_checkpoint_head_phase_profile_audit"
        or audit.get("ok") is not True
        or audit.get("input_manifest_sha256") != sha256(manifest_path)
        or audit.get("operator") != manifest.get("operator")
        or int(audit.get("record_count", -1)) != int(manifest["expected_record_count"])
    ):
        raise ValueError("v195 analysis requires one passing bound profile audit")

    operator = str(manifest["operator"])
    aggregate, aggregate_audit = aggregate_operator(profile_root, operator)
    split = manifest["prompt_split"]
    discovery_ids = [int(value) for value in split["discovery"]]
    validation_ids = [int(value) for value in split["validation"]]
    holdout_ids = [int(value) for value in split["generation_holdout"]]
    if (
        len(discovery_ids) != 64
        or len(validation_ids) != 32
        or len(holdout_ids) != 32
        or set(discovery_ids) | set(validation_ids) | set(holdout_ids)
        != set(range(128))
    ):
        raise ValueError("v195 requires the frozen 64/32/32 v189 split")

    sf_scores_path = Path(manifest["v189_provenance"]["cell_scores"])
    sf_discovery, sf_validation, _ = _load_sf_cells(sf_scores_path, operator)
    route_path = Path(manifest["frozen_sf_route"]["path"])
    frozen_mask = _load_mask(route_path)
    gain = aggregate["gain"]
    cf_discovery = gain[discovery_ids].mean(axis=0)
    cf_validation = gain[validation_ids].mean(axis=0)
    cf_holdout = gain[holdout_ids].mean(axis=0)

    cf_rows = _cell_rows(
        aggregate,
        operator=operator,
        discovery=discovery_ids,
        validation=validation_ids,
    )
    cf_refit_mask = np.asarray(_mask(cf_rows, "compatible"), dtype=np.bool_)
    prompt_rows = _prompt_effects(gain, frozen_mask, holdout_ids)
    selected_prompt_values = np.asarray(
        [row["selected_gain"] for row in prompt_rows], dtype=np.float64
    )
    enrichment_values = np.asarray(
        [row["selected_minus_complement"] for row in prompt_rows], dtype=np.float64
    )
    selected_summary = {
        "prompt_count": len(prompt_rows),
        "mean_gain": float(selected_prompt_values.mean()),
        "bootstrap_ci95": bootstrap_ci(
            selected_prompt_values, seed=RANDOM_SEED + 1, samples=10_000
        ),
        "positive_prompt_fraction": float(np.mean(selected_prompt_values > 0)),
        "enrichment_mean": float(enrichment_values.mean()),
        "enrichment_bootstrap_ci95": bootstrap_ci(
            enrichment_values, seed=RANDOM_SEED + 2, samples=10_000
        ),
        "enrichment_positive_prompt_fraction": float(np.mean(enrichment_values > 0)),
    }
    random_call = _random_control(
        cf_holdout,
        frozen_mask,
        preserve_call_layer=False,
        draws=RANDOM_DRAWS,
        seed=RANDOM_SEED + 10,
    )
    random_call_layer = _random_control(
        cf_holdout,
        frozen_mask,
        preserve_call_layer=True,
        draws=RANDOM_DRAWS,
        seed=RANDOM_SEED + 20,
    )
    selected_positive = bool(
        selected_summary["bootstrap_ci95"][0] > 0
        and selected_summary["positive_prompt_fraction"] >= PROMPT_WIN_THRESHOLD
    )
    selected_enriched = bool(selected_summary["enrichment_bootstrap_ci95"][0] > 0)
    phase_layer_supported = bool(
        selected_positive
        and selected_enriched
        and random_call["one_sided_empirical_p"] <= RANDOM_P_THRESHOLD
    )
    exact_head_supported = bool(
        phase_layer_supported
        and random_call_layer["one_sided_empirical_p"] <= RANDOM_P_THRESHOLD
    )
    if exact_head_supported:
        level = "exact_head_identity"
    elif phase_layer_supported:
        level = "phase_layer_structure"
    elif selected_positive:
        level = "operator_only"
    else:
        level = "unsupported"

    correlations = {
        "sf_validation_vs_cf_validation": _structural_correlations(
            sf_validation, cf_validation
        ),
        "sf_discovery_vs_cf_discovery": _structural_correlations(
            sf_discovery, cf_discovery
        ),
    }
    overlap = {
        "overall": _overlap(frozen_mask, cf_refit_mask),
        "by_call": [
            {"call_index": call, **_overlap(frozen_mask[call], cf_refit_mask[call])}
            for call in range(CALLS)
        ],
        "note": (
            "The Causal-refit map uses the same v189 thresholds for diagnosis only; "
            "it is not a candidate selected for generation."
        ),
    }

    cell_rows = []
    cf_by_key = {
        (int(row["call_index"]), int(row["layer"]), int(row["head"])): row
        for row in cf_rows
    }
    for call in range(CALLS):
        for layer in range(LAYERS):
            for head in range(HEADS):
                key = (call, layer, head)
                row = cf_by_key[key]
                cell_rows.append(
                    {
                        "call_index": call,
                        "layer": layer,
                        "head": head,
                        "frozen_sf_selected": bool(frozen_mask[key]),
                        "cf_refit_selected": bool(cf_refit_mask[key]),
                        "sf_discovery_gain": float(sf_discovery[key]),
                        "sf_validation_gain": float(sf_validation[key]),
                        "cf_discovery_gain": float(cf_discovery[key]),
                        "cf_validation_gain": float(cf_validation[key]),
                        "cf_holdout_gain": float(cf_holdout[key]),
                        "cf_validation_win_fraction": float(
                            row["validation_win_fraction"]
                        ),
                        "cf_full_budget_fraction": float(row["full_budget_fraction"]),
                        "cf_relative_reference_energy": float(
                            row["relative_reference_energy"]
                        ),
                    }
                )

    v194_confirmed = bool(
        manifest["v194_provenance"]["cross_checkpoint_transfer_confirmed"]
    )
    gates = {
        "selected_holdout_gain_positive": selected_positive,
        "selected_enriched_over_complement": selected_enriched,
        "phase_layer_allocation_beats_call_count_random": bool(
            random_call["one_sided_empirical_p"] <= RANDOM_P_THRESHOLD
        ),
        "head_identity_beats_call_layer_count_random": bool(
            random_call_layer["one_sided_empirical_p"] <= RANDOM_P_THRESHOLD
        ),
    }
    report = {
        "version": 1,
        "experiment": "v195_cross_checkpoint_head_phase_profile",
        "diagnostic": True,
        "operator": operator,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "profile_record_count": aggregate_audit["record_count"],
        "frozen_sf_route": manifest["frozen_sf_route"],
        "prompt_split": split,
        "primary_evaluation_split": "generation_holdout",
        "thresholds": {
            "positive_prompt_fraction": PROMPT_WIN_THRESHOLD,
            "random_control_one_sided_p": RANDOM_P_THRESHOLD,
            "positive_and_enrichment_ci_rule": "95% prompt-bootstrap lower bound > 0",
        },
        "holdout_selected_route": selected_summary,
        "random_controls": {
            "call_count_matched": random_call,
            "call_layer_count_matched": random_call_layer,
        },
        "cross_checkpoint_correlations": correlations,
        "causal_refit_overlap": overlap,
        "mechanism_gates": gates,
        "mechanism_support_level": level,
        "exact_head_identity_transfer_supported": exact_head_supported,
        "phase_layer_structure_transfer_supported": phase_layer_supported,
        "operator_compatibility_transfer_supported": selected_positive,
        "v194_generation_transfer_confirmed": v194_confirmed,
        "recommendation": _recommendation(v194_confirmed, level),
        "manual_review_required": False,
        "claim_boundary": manifest["claim_boundary"],
        "source": {
            "input_manifest": str(manifest_path.resolve()),
            "input_manifest_sha256": sha256(manifest_path),
            "profile_audit": str(profile_audit_path.resolve()),
            "profile_audit_sha256": sha256(profile_audit_path),
            "sf_cell_scores": str(sf_scores_path.resolve()),
            "sf_cell_scores_sha256": sha256(sf_scores_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NpEncoder) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "cell_transfer.csv", cell_rows)
    _write_csv(output_dir / "holdout_prompt_effects.csv", prompt_rows)
    (output_dir / "analysis.md").write_text(render(report), encoding="utf-8")
    print(
        "[v195-analysis] "
        f"level={level} recommendation={report['recommendation']} "
        f"v194_confirmed={str(v194_confirmed).lower()}"
    )
    return report


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render(report: dict) -> str:
    selected = report["holdout_selected_route"]
    random_controls = report["random_controls"]
    corr = report["cross_checkpoint_correlations"]["sf_validation_vs_cf_validation"]
    lines = [
        "# v195 Cross-Checkpoint Head x Phase Profile",
        "",
        f"- Operator: `{report['operator']}`",
        f"- v194 generation transfer confirmed: `{str(report['v194_generation_transfer_confirmed']).lower()}`",
        f"- Mechanism support: `{report['mechanism_support_level']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "- Manual review required: `false`",
        "",
        "## Holdout transfer",
        "",
        f"- Frozen selected-cell mean gain: `{selected['mean_gain']:.6f}`",
        f"- Prompt-bootstrap CI95: `{selected['bootstrap_ci95']}`",
        f"- Positive-prompt fraction: `{selected['positive_prompt_fraction']:.4f}`",
        f"- Selected-minus-complement mean: `{selected['enrichment_mean']:.6f}`",
        f"- Enrichment CI95: `{selected['enrichment_bootstrap_ci95']}`",
        "",
        "## Random controls",
        "",
        "| Control | Observed percentile | One-sided p |",
        "|---|---:|---:|",
        f"| call-count matched | {random_controls['call_count_matched']['observed_percentile']:.4f} | {random_controls['call_count_matched']['one_sided_empirical_p']:.6f} |",
        f"| call/layer-count matched | {random_controls['call_layer_count_matched']['observed_percentile']:.4f} | {random_controls['call_layer_count_matched']['one_sided_empirical_p']:.6f} |",
        "",
        "## Structural correlation",
        "",
        "| Resolution | Pearson | Spearman | Sign agreement |",
        "|---|---:|---:|---:|",
    ]
    for name, row in corr.items():
        lines.append(
            f"| {name} | {_fmt(row['pearson'])} | {_fmt(row['spearman'])} | "
            f"{_fmt(row['sign_agreement_nonzero'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact-head support requires the frozen route to be positive on the untouched 32-prompt split, enriched over its complement, and better than both count-matched controls.",
            "- Phase/layer support requires the same positive and enrichment checks plus the call-count control, but not the within-call/layer head-identity control.",
            "- The Causal-refit map is diagnostic only and must not replace the frozen SF map in v194.",
            "- This profile is not video-quality evidence and requests no human review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--profile-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.manifest, args.profile_root, args.profile_audit, args.output_dir)


if __name__ == "__main__":
    main()
