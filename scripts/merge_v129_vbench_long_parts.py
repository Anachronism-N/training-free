#!/usr/bin/env python3
"""Merge validated v129 VBench-Long jobs and build the paper table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from collect_vbench_long_results import collect, write_outputs
from prepare_v129_paper_comparison import (
    ALL_DIMENSIONS,
    CORE_DIMENSIONS,
    SEMANTIC_EXTENSION_DIMENSIONS,
    SOURCES,
)


VIDEO_INDEX = re.compile(r"^(\d+)-(\d+)(?:_|$)")
PROFILE_DIMENSIONS = {
    "core": CORE_DIMENSIONS,
    "semantic_extension": SEMANTIC_EXTENSION_DIMENSIONS,
    "full": ALL_DIMENSIONS,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        for key in ("score", "overall", "mean", "average", "total_score"):
            if key in value:
                score = finite_score(value[key])
                if score is not None:
                    return score
    if isinstance(value, (list, tuple)):
        for item in value:
            score = finite_score(item)
            if score is not None:
                return score
    return None


def collect_video_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"video_path", "video", "path"} and isinstance(
                item, str
            ):
                if item.lower().endswith((".mp4", ".gif")):
                    paths.add(item)
            else:
                paths.update(collect_video_paths(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            paths.update(collect_video_paths(item))
    return paths


def collect_prompt_indices(value: Any) -> set[int]:
    indices: set[int] = set()
    for raw_path in collect_video_paths(value):
        for part in reversed(Path(raw_path).parts):
            match = VIDEO_INDEX.match(Path(part).stem)
            if match is None:
                continue
            if int(match.group(2)) != 0:
                raise ValueError(f"unexpected sample index in {raw_path}")
            indices.add(int(match.group(1)))
            break
    return indices


def validate_manifest(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"comparison manifest is not an object: {path}")
    if (
        payload.get("experiment") != "v129_no_pf_paper_comparison_30s"
        or payload.get("prompt_count") != 128
        or payload.get("num_output_frames") != 120
        or payload.get("seed") != 0
        or payload.get("pf_required") is not False
    ):
        raise ValueError(f"invalid v129 comparison contract: {path}")
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("comparison manifest has no methods")
    keys = [row.get("key") for row in methods if isinstance(row, dict)]
    expected = [source.final_key for source in SOURCES]
    if keys != expected or any("pf" in str(key).lower() for key in keys):
        raise ValueError(f"invalid no-PF method order: {keys!r}")
    expected_profiles = {
        name: list(dimensions)
        for name, dimensions in PROFILE_DIMENSIONS.items()
    }
    if payload.get("metric_profiles") != expected_profiles:
        raise ValueError("metric profile contract differs")
    return payload


def validate_job(
    part_dir: Path,
    *,
    method: str,
    dimension: str,
    manifest_sha: str,
) -> tuple[Any, dict[str, Any]]:
    result_path = part_dir / "results.json"
    marker_path = part_dir / "done.json"
    contract_path = part_dir / "job_contract.json"
    mapping_path = part_dir / "prompt_mapping.json"
    result = load_json(result_path)
    marker = load_json(marker_path)
    contract = load_json(contract_path)
    mapping = load_json(mapping_path)
    if (
        not isinstance(result, dict)
        or dimension not in result
        or finite_score(result[dimension]) is None
    ):
        raise RuntimeError(
            f"missing finite {dimension} score in {result_path}"
        )
    if (
        not isinstance(marker, dict)
        or marker.get("comparison_manifest_sha256") != manifest_sha
        or marker.get("method") != method
        or marker.get("dimension") != dimension
        or marker.get("result_sha256") != sha256(result_path)
        or marker.get("job_contract_sha256") != sha256(contract_path)
        or marker.get("prompt_mapping_sha256") != sha256(mapping_path)
    ):
        raise RuntimeError(f"stale VBench marker: {marker_path}")
    if (
        not isinstance(contract, dict)
        or contract.get("comparison_manifest_sha256") != manifest_sha
        or contract.get("method") != method
        or contract.get("dimension") != dimension
        or marker.get("vbench_commit") != contract.get("vbench_commit")
        or contract.get("prompt_mapping") != "comparison_manifest_exact"
        or mapping.get("comparison_manifest_sha256") != manifest_sha
        or mapping.get("prompt_mapping") != "comparison_manifest_exact"
        or mapping.get("mapped_count") != 128
        or mapping.get("indices") != list(range(128))
    ):
        raise RuntimeError(f"stale VBench job contract: {contract_path}")
    indices = collect_prompt_indices(result[dimension])
    if indices != set(range(128)):
        raise RuntimeError(
            f"{method}:{dimension}: expected 128 prompt indices, "
            f"found {len(indices)}"
        )
    return result[dimension], {
        "method": method,
        "dimension": dimension,
        "score": finite_score(result[dimension]),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "job_contract": str(contract_path),
        "job_contract_sha256": sha256(contract_path),
        "reported_video_paths": len(
            collect_video_paths(result[dimension])
        ),
        "reported_prompt_indices": len(indices),
        "log": str(part_dir / "run.log"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DIMENSIONS),
        default="core",
    )
    parser.add_argument("--parts-root", type=Path)
    parser.add_argument("--combined-root", type=Path)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--vbench-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_root = args.comparison_root.resolve()
    manifest_path = comparison_root / "comparison_manifest.json"
    manifest = validate_manifest(manifest_path)
    manifest_sha = sha256(manifest_path)
    parts_root = (
        args.parts_root
        or comparison_root / "metrics" / "vbench_long_parts"
    ).resolve()
    combined_root = (
        args.combined_root
        or comparison_root / "metrics" / "vbench_long_combined"
    ).resolve()
    summary_root = (
        args.summary_root or comparison_root / "metrics"
    ).resolve()
    methods = [str(row["key"]) for row in manifest["methods"]]
    required = set(PROFILE_DIMENSIONS[args.profile])
    coverage: list[dict[str, Any]] = []
    completed_by_method: dict[str, list[str]] = {}

    for method in methods:
        video_dir = Path(
            next(
                row["video_dir"]
                for row in manifest["methods"]
                if row["key"] == method
            )
        )
        if len(list(video_dir.glob("*.mp4"))) != 128:
            raise RuntimeError(f"{method}: comparison video set changed")
        combined: dict[str, Any] = {}
        for dimension in ALL_DIMENSIONS:
            part_dir = parts_root / method / dimension
            present = all(
                (part_dir / name).is_file()
                for name in (
                    "results.json",
                    "done.json",
                    "job_contract.json",
                    "prompt_mapping.json",
                )
            )
            if not present:
                if dimension in required:
                    raise RuntimeError(
                        f"required VBench job is missing: "
                        f"{method}:{dimension}"
                    )
                continue
            value, row = validate_job(
                part_dir,
                method=method,
                dimension=dimension,
                manifest_sha=manifest_sha,
            )
            combined[dimension] = value
            coverage.append(row)
        missing_required = sorted(required - set(combined))
        if missing_required:
            raise RuntimeError(
                f"{method}: missing required dimensions {missing_required}"
            )
        completed_by_method[method] = [
            dimension
            for dimension in ALL_DIMENSIONS
            if dimension in combined
        ]
        write_json(combined_root / method / "results.json", combined)

    summary = collect(
        combined_root,
        methods,
        list(ALL_DIMENSIONS),
        allow_missing=True,
    )
    summary.update(
        {
            "comparison_manifest": str(manifest_path),
            "comparison_manifest_sha256": manifest_sha,
            "prompt_count": 128,
            "required_profile": args.profile,
            "required_dimensions": list(PROFILE_DIMENSIONS[args.profile]),
            "completed_dimensions": completed_by_method,
        }
    )
    summary_json = summary_root / "vbench_long_summary.json"
    write_outputs(
        summary,
        output_json=summary_json,
        output_csv=summary_root / "vbench_long_summary.csv",
        output_md=summary_root / "vbench_long_summary.md",
    )
    write_json(
        summary_root / "vbench_long_coverage.json",
        {
            "version": 1,
            "comparison_manifest": str(manifest_path),
            "comparison_manifest_sha256": manifest_sha,
            "required_profile": args.profile,
            "methods": methods,
            "jobs": coverage,
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_v129_paper_table.py")),
            "--summary-json",
            str(summary_json),
            "--comparison-manifest",
            str(manifest_path),
            "--vbench-root",
            str(args.vbench_root.resolve()),
            "--output-root",
            str(summary_root / "paper_table"),
        ],
        check=True,
    )
    print(
        f"[merge-v129-vbench] profile={args.profile} "
        f"methods={len(methods)} validated_jobs={len(coverage)} "
        f"summary={summary_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
