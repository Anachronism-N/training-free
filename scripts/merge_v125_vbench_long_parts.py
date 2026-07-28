#!/usr/bin/env python3
"""Merge dimension-sharded VBench-Long outputs into auditable method results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from collect_vbench_long_results import collect, write_outputs


VIDEO_INDEX = re.compile(r"^(\d+)-(\d+)(?:_|$)")
EXPECTED_METHODS = (
    "sf_native",
    "pf_native",
    "ours_landmark_motion1",
    "ours_landmark_retrieval1_age24",
    "ours_landmark_retrieval_motion",
    "ours_prototype_motion1",
    "ours_prototype_retrieval1_age24",
    "ours_prototype_retrieval_motion",
)
EXPECTED_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
)


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
        payload.get("experiment") != "v125_moviebench128_comparison"
        or payload.get("prompt_count") != 128
        or payload.get("num_output_frames") != 120
        or payload.get("seed") != 0
    ):
        raise ValueError(f"invalid comparison contract: {path}")
    methods = payload.get("methods")
    dimensions = payload.get("vbench_long_dimensions")
    if not isinstance(methods, list) or not methods:
        raise ValueError("comparison manifest has no methods")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("comparison manifest has no dimensions")
    keys = [row.get("key") for row in methods if isinstance(row, dict)]
    if keys != list(EXPECTED_METHODS):
        raise ValueError(
            f"comparison manifest methods differ: {keys!r}"
        )
    if dimensions != list(EXPECTED_DIMENSIONS):
        raise ValueError(
            f"comparison manifest dimensions differ: {dimensions!r}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--parts-root", type=Path)
    parser.add_argument("--combined-root", type=Path)
    parser.add_argument("--summary-root", type=Path)
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
    dimensions = [
        str(dimension)
        for dimension in manifest["vbench_long_dimensions"]
    ]

    coverage_rows: list[dict[str, Any]] = []
    for method in methods:
        source_video_dir = comparison_root / "published" / method
        video_count = len(list(source_video_dir.glob("*.mp4")))
        if video_count != 128:
            raise RuntimeError(
                f"{method}: expected 128 comparison videos, found {video_count}"
            )
        combined: dict[str, Any] = {}
        for dimension in dimensions:
            part_dir = parts_root / method / dimension
            result_path = part_dir / "results.json"
            marker_path = part_dir / "done.json"
            contract_path = part_dir / "job_contract.json"
            result = load_json(result_path)
            marker = load_json(marker_path)
            contract = load_json(contract_path)
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
                or marker.get("job_contract_sha256")
                != sha256(contract_path)
            ):
                raise RuntimeError(f"stale VBench marker: {marker_path}")
            if (
                not isinstance(contract, dict)
                or contract.get("comparison_manifest_sha256") != manifest_sha
                or contract.get("method") != method
                or contract.get("dimension") != dimension
                or marker.get("vbench_commit")
                != contract.get("vbench_commit")
            ):
                raise RuntimeError(
                    f"stale VBench job contract: {contract_path}"
                )
            paths = collect_video_paths(result[dimension])
            prompt_indices = collect_prompt_indices(result[dimension])
            if prompt_indices != set(range(128)):
                raise RuntimeError(
                    f"{method}:{dimension}: expected per-prompt coverage "
                    f"[0,128), found {len(prompt_indices)} indices"
                )
            combined[dimension] = result[dimension]
            coverage_rows.append(
                {
                    "method": method,
                    "dimension": dimension,
                    "score": finite_score(result[dimension]),
                    "result": str(result_path),
                    "result_sha256": sha256(result_path),
                    "job_contract": str(contract_path),
                    "job_contract_sha256": sha256(contract_path),
                    "reported_video_paths": len(paths),
                    "reported_prompt_indices": len(prompt_indices),
                    "comparison_video_count": video_count,
                    "log": str(part_dir / "run.log"),
                }
            )
        write_json(combined_root / method / "results.json", combined)

    summary = collect(
        combined_root,
        methods,
        dimensions,
        allow_missing=False,
    )
    summary["comparison_manifest"] = str(manifest_path)
    summary["comparison_manifest_sha256"] = manifest_sha
    summary["prompt_count"] = 128
    write_outputs(
        summary,
        output_json=summary_root / "vbench_long_summary.json",
        output_csv=summary_root / "vbench_long_summary.csv",
        output_md=summary_root / "vbench_long_summary.md",
    )
    write_json(
        summary_root / "vbench_long_coverage.json",
        {
            "version": 1,
            "comparison_manifest": str(manifest_path),
            "comparison_manifest_sha256": manifest_sha,
            "methods": methods,
            "dimensions": dimensions,
            "jobs": coverage_rows,
        },
    )
    print(
        f"[merge-v125-vbench] methods={len(methods)} "
        f"dimensions={len(dimensions)} jobs={len(coverage_rows)} "
        f"summary={summary_root / 'vbench_long_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
