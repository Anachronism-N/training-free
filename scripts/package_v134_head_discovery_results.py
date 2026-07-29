#!/usr/bin/env python3
"""Create a small, Git-friendly v134 result bundle from server artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ANALYSIS_FILES = (
    "analysis_summary.md",
    "classification_report.json",
    "analysis_debug.json",
    "head_scores.csv",
    "head_job_scores.csv",
    "head_map.csv",
    "head_factor_scores.csv",
    "head_timestep_scores.csv",
    "head_ar_scores.csv",
    "temporal_timestep_scores.csv",
    "temporal_ar_scores.csv",
    "family_base_consistency.csv",
    "threshold_sweep.csv",
    "layer_summary.csv",
    "factor_layer_summary.csv",
    "timestep_layer_summary.csv",
    "ar_layer_summary.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_results(run_root: Path, output_dir: Path) -> dict:
    analysis_dir = run_root / "analysis"
    input_dir = run_root / "inputs"
    missing = [
        name for name in ANALYSIS_FILES if not (analysis_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "analysis is incomplete; missing " + ", ".join(missing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {"files": {}, "profile_counts": {}, "video_counts": {}}
    for name in ANALYSIS_FILES:
        source = analysis_dir / name
        target = output_dir / name
        shutil.copy2(source, target)
        inventory["files"][name] = {
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    for name in (
        "suite_metadata.json",
        "controlled128_counterfactual.jsonl",
    ):
        source = input_dir / name
        if source.is_file():
            target = output_dir / name
            shutil.copy2(source, target)
            inventory["files"][name] = {
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
    for stage in ("observational", "counterfactual"):
        inventory["profile_counts"][stage] = len(
            list((run_root / "profiles" / stage).glob("*.pt"))
        )
        inventory["video_counts"][stage] = len(
            list((run_root / "videos" / stage).glob("*.mp4"))
        )
    log_summary = {}
    for stage in ("observational", "counterfactual"):
        stage_rows = []
        for path in sorted((run_root / "logs" / stage).glob("*.log")):
            text = path.read_text(encoding="utf-8", errors="replace")
            suspicious = [
                line
                for line in text.splitlines()
                if any(
                    marker in line.lower()
                    for marker in (
                        "traceback",
                        "runtimeerror",
                        "out of memory",
                        "nan",
                        "polygon",
                    )
                )
            ]
            stage_rows.append(
                {
                    "name": path.name,
                    "profile_begin": text.count("[HeadProfile] begin"),
                    "profile_end": text.count("[HeadProfile] end"),
                    "suspicious_lines": suspicious[:20],
                }
            )
        log_summary[stage] = stage_rows
    (output_dir / "worker_log_summary.json").write_text(
        json.dumps(log_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory["files"]["worker_log_summary.json"] = {
        "bytes": (output_dir / "worker_log_summary.json").stat().st_size,
        "sha256": _sha256(output_dir / "worker_log_summary.json"),
    }
    (output_dir / "run_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = json.loads(
        (output_dir / "classification_report.json").read_text(
            encoding="utf-8"
        )
    )
    review = [
        "# v134 Head Discovery Review Bundle",
        "",
        f"- Acceptance gates: **{report['acceptance_gates']['accepted']}**",
        f"- Label counts: `{report['label_counts']}`",
        f"- Profiles: `{inventory['profile_counts']}`",
        f"- Videos: `{inventory['video_counts']}`",
        "",
        "Review `analysis_summary.md` first, then inspect:",
        "",
        "1. `classification_report.json` for all gates and global statistics.",
        "2. `head_scores.csv` for the 360-head classification and confidence.",
        "3. `head_job_scores.csv` for prompt-family outliers and reproducibility.",
        "4. `head_factor_scores.csv` for identity/action/scene/camera differences.",
        "5. `head_timestep_scores.csv` and `head_ar_scores.csv` for dynamic roles.",
        "6. `temporal_timestep_scores.csv` and `temporal_ar_scores.csv` for history use.",
        "7. `family_base_consistency.csv` to verify matched trajectories across factors.",
        "8. `*_layer_summary.csv` for paper-oriented layer curves.",
        "9. `worker_log_summary.json` for bounded execution diagnostics.",
        "",
        "Do not infer a cache policy from `head_map.csv` when the acceptance gates fail.",
        "",
        "## Human Review Notes",
        "",
        "- Visual anomalies:",
        "- Motion/identity observations:",
        "- Unexpected factor or timestep behavior:",
        "",
    ]
    (output_dir / "review_bundle.md").write_text(
        "\n".join(review), encoding="utf-8"
    )
    print(
        f"[v134-package] output={output_dir} "
        f"profiles={inventory['profile_counts']}"
    )
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_results(args.run_root, args.output_dir)


if __name__ == "__main__":
    main()
