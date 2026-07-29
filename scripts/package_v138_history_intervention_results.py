#!/usr/bin/env python3
"""Package bounded v138 analysis artifacts for Git review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ANALYSIS_FILES = (
    "analysis_summary.md",
    "analysis_report.json",
    "head_axes.csv",
    "head_timestep_axes.csv",
    "head_ar_axes.csv",
    "head_timestep_specialization.csv",
    "axis_diagnostics.csv",
    "axis_correlations.csv",
    "profile_contract_audit.csv",
    "donor_audit.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_results(analysis_dir: Path, output_dir: Path) -> dict:
    report_path = analysis_dir / "analysis_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("head_count", 0)) != 360:
        raise ValueError("v138 report does not contain all 360 heads")
    if not bool(report.get("profile_contract_passed", False)):
        raise ValueError("v138 profile contract did not pass")
    if float(report.get("maximum_rope_reconstruction_error", 1.0)) > 5e-3:
        raise ValueError("v138 RoPE reconstruction gate failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ANALYSIS_FILES:
        source = analysis_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_dir / name
        shutil.copy2(source, target)
        files.append(
            {
                "name": name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    inventory = {
        "method": report["method"],
        "recommendation": report["recommendation"],
        "profile_count": report["profile_count"],
        "head_count": report["head_count"],
        "gates": report["gates"],
        "files": files,
        "excluded": [
            "raw projected Q/K descriptors",
            "raw .pt profiles",
            "videos",
            "worker logs",
            "per-job head tables",
        ],
    }
    (output_dir / "bundle_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review_bundle.md").write_text(
        "\n".join(
            [
                "# v138 Review Bundle",
                "",
                f"- Recommendation: `{report['recommendation']}`",
                (
                    "- History-specificity gate: "
                    f"`{report['gates']['history_specificity']}`"
                ),
                f"- Order-axis gate: `{report['gates']['order_axis']}`",
                "",
                "Review `analysis_summary.md`, then `head_axes.csv`, "
                "`axis_correlations.csv`, and `donor_audit.csv`.",
                "",
                "Raw descriptors, profiles, and videos are excluded.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = package_results(args.analysis_dir, args.output_dir)
    print(
        "[v138-package] "
        f"files={len(inventory['files'])} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
