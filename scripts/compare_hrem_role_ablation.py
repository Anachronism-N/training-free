#!/usr/bin/env python3
"""Join HREM role-ablation video metrics with per-cell trace diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


_ALLOWED_ROLE_CONFIG_DIFFERENCES = {
    "head_routing",
    "role_calibration",
    "role_threshold",
    "role_sharpness",
    "role_keep_fraction",
    "role_min_evidence_spread",
    "episode_warmup_blocks",
}


def _controlled_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    readout = dict(config.get("readout") or {})
    for key in _ALLOWED_ROLE_CONFIG_DIFFERENCES:
        readout.pop(key, None)
    runtime = dict(config.get("runtime") or {})
    runtime.pop("run_cell", None)
    return {
        "active_layers": config.get("active_layers"),
        "archive": config.get("archive"),
        "readout": readout,
        "runtime": runtime,
    }


def _different_paths(reference: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(reference, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(reference) | set(current)):
            path = f"{prefix}.{key}" if prefix else key
            differences.extend(_different_paths(reference.get(key), current.get(key), path))
        return differences
    return [] if reference == current else [prefix]


def build_comparison(run_root: Path) -> dict[str, Any]:
    video_metrics = _load(run_root / "metrics_role_ablation.json")
    if video_metrics is None:
        raise FileNotFoundError(run_root / "metrics_role_ablation.json")
    aggregate = video_metrics.get("aggregate") or {}
    paired = video_metrics.get("paired_delta") or {}
    diagnoses = {
        method: _load(run_root / "traces" / f"{method}_diagnosis.json")
        for method in aggregate
    }
    controlled_configs = {
        method: _controlled_config((diagnosis or {}).get("config"))
        for method, diagnosis in diagnoses.items()
        if method != "native_reset"
    }
    reference_config = next(
        (config for config in controlled_configs.values() if config is not None),
        None,
    )
    rows: list[dict[str, Any]] = []
    for method in aggregate:
        diagnosis = diagnoses[method]
        mechanism = (diagnosis or {}).get("metrics") or {}
        findings = [
            str(item.get("code"))
            for item in (diagnosis or {}).get("findings", [])
            if item.get("severity") in {"ERROR", "WARNING"}
        ]
        full = (aggregate.get(method) or {}).get("full") or {}
        background = (aggregate.get(method) or {}).get("background") or {}
        full_delta = (paired.get(method) or {}).get("full") or {}
        role_active = mechanism.get("role_active_head_fraction")
        calibration_valid = mechanism.get("role_calibration_valid_fraction")
        active_jaccard = mechanism.get("role_active_head_jaccard")
        method_config = controlled_configs.get(method)
        if method == "native_reset" or reference_config is None:
            unexpected_config_differences: list[str] = []
        elif method_config is None:
            unexpected_config_differences = ["missing_config"]
        else:
            unexpected_config_differences = _different_paths(
                reference_config,
                method_config,
            )
        disqualifying_findings = {
            "causal_invariant_violation",
            "no_archive_commits",
            "no_accepted_readout",
            "role_gate_not_selective",
            "role_gate_low_contrast",
            "role_evidence_not_discriminative",
            "role_calibration_rejected_all",
            "role_calibration_often_invalid",
            "role_gate_denoise_instability",
            "role_identity_denoise_instability",
        }
        structurally_eligible = (
            method not in {"native_reset", "dual_all_heads"}
            and not disqualifying_findings.intersection(findings)
            and not unexpected_config_differences
            and role_active is not None
            and 0.10 <= float(role_active) <= 0.90
            and (calibration_valid is None or float(calibration_valid) >= 0.90)
            and (active_jaccard is None or float(active_jaccard) >= 0.50)
        )
        rows.append({
            "method": method,
            "full_return_margin": full.get("return_margin"),
            "background_return_margin": background.get("return_margin"),
            "paired_full_return_delta": full_delta.get("return_margin_mean"),
            "positive_prompts": full_delta.get("positive_prompts"),
            "retrieval_accepted_head_fraction": mechanism.get(
                "retrieval_accepted_head_fraction"
            ),
            "role_gate_mean": mechanism.get("head_gate_mean"),
            "role_gate_std": mechanism.get("role_gate_std"),
            "role_active_head_fraction": role_active,
            "role_evidence_spread_median": mechanism.get(
                "role_evidence_spread_median"
            ),
            "role_calibration_valid_fraction": calibration_valid,
            "role_active_head_jaccard": active_jaccard,
            "fusion_delta_to_native_rms": mechanism.get(
                "delta_to_native_rms_median"
            ),
            "episode_warmup_scale_mean": mechanism.get(
                "episode_warmup_scale_mean"
            ),
            "episode_first_block_effective_gate_mean": mechanism.get(
                "episode_first_block_effective_gate_mean"
            ),
            "findings": findings,
            "unexpected_config_differences": unexpected_config_differences,
            "structurally_eligible_for_visual_review": structurally_eligible,
        })
    return {
        "run_root": str(run_root),
        "rows": rows,
        "eligible_methods": [
            row["method"]
            for row in rows
            if row["structurally_eligible_for_visual_review"]
        ],
        "allowed_config_differences": sorted(_ALLOWED_ROLE_CONFIG_DIFFERENCES),
        "selection_note": (
            "Structural eligibility is not a winner declaration. Select a final role gate "
            "only after motion and boundary-artifact review."
        ),
    }


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def print_comparison(report: dict[str, Any]) -> None:
    columns = [
        "method",
        "paired_full_return_delta",
        "positive_prompts",
        "role_gate_mean",
        "role_gate_std",
        "role_active_head_fraction",
        "role_evidence_spread_median",
        "role_calibration_valid_fraction",
        "role_active_head_jaccard",
        "fusion_delta_to_native_rms",
        "episode_warmup_scale_mean",
        "episode_first_block_effective_gate_mean",
        "structurally_eligible_for_visual_review",
    ]
    print("\t".join(columns))
    for row in report["rows"]:
        print("\t".join(_format(row.get(column)) for column in columns))
        if row["findings"]:
            print(f"  findings[{row['method']}]: {','.join(row['findings'])}")
        if row["unexpected_config_differences"]:
            print(
                f"  config_mismatch[{row['method']}]: "
                f"{','.join(row['unexpected_config_differences'])}"
            )
    print(f"eligible_methods: {report['eligible_methods']}")
    print(report["selection_note"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    report = build_comparison(args.run_root)
    print_comparison(report)
    output = args.json_output or (args.run_root / "role_ablation_comparison.json")
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
