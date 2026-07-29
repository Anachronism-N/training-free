#!/usr/bin/env python3
"""Package bounded v136 analysis artifacts for Git review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ANALYSIS_FILES = (
    "multi_axis_summary.md",
    "multi_axis_report.json",
    "head_axes.csv",
    "head_factor_axes.csv",
    "head_timestep_axes.csv",
    "head_ar_axes.csv",
    "head_natural_timestep_axes.csv",
    "head_natural_ar_axes.csv",
    "head_factor_specialization.csv",
    "head_timestep_specialization.csv",
    "head_ar_specialization.csv",
    "axis_diagnostics.csv",
    "axis_correlations.csv",
    "context_stability.csv",
    "profile_contract_audit.csv",
    "state_eligibility_audit.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_results(analysis_dir: Path, output_dir: Path) -> dict:
    report_path = analysis_dir / "multi_axis_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("head_count", 0)) != 360:
        raise ValueError("v136 report does not contain the complete 360 heads")
    if not bool(report.get("profile_contract_passed", False)):
        raise ValueError("v136 profile contract did not pass")

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
        "profile_counts": report["profile_counts"],
        "head_count": report["head_count"],
        "gates": report["gates"],
        "files": files,
        "excluded": [
            "raw .pt profiles",
            "generated videos",
            "worker logs",
        ],
    }
    (output_dir / "bundle_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review = [
        "# v136 Multi-Axis Review Bundle",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Prompt-axis gate: `{report['gates']['prompt_axis']}`",
        f"- Temporal-axis gate: `{report['gates']['temporal_axis']}`",
        f"- Complete heads: `{report['head_count']}`",
        "",
        "Review order:",
        "",
        "1. `multi_axis_summary.md`",
        "2. `multi_axis_report.json`",
        "3. `head_axes.csv`",
        "4. `head_factor_specialization.csv`",
        "5. `head_timestep_specialization.csv`",
        "6. `context_stability.csv`",
        "7. `profile_contract_audit.csv`",
        "",
        "Raw profiles and videos are deliberately excluded.",
        "",
    ]
    (output_dir / "review_bundle.md").write_text(
        "\n".join(review), encoding="utf-8"
    )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = package_results(args.analysis_dir, args.output_dir)
    print(
        "[v136-package] "
        f"files={len(inventory['files'])} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
