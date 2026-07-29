#!/usr/bin/env python3
"""Assemble SF, the selected method, and completed v132 ablation controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_v100_fast_selection_1video import sha256, write_frozen
from run_v120_moviebench32_main import link_or_validate
from run_v132_binary_memory_ablation import (
    ALL_METHODS,
    EXPERIMENT as CONTROL_EXPERIMENT,
    TIER1_METHODS,
    parse_method_keys,
)


PROMPT_COUNT = 128
DEFAULT_PROMPTS = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
    "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)
CORE_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)
SEMANTIC_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
)


@dataclass(frozen=True)
class Source:
    final_key: str
    root: Path
    source_key: str
    role: str


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--prompts", type=Path, default=env_path("V132_PROMPTS"))
    parser.add_argument(
        "--methods",
        default=os.environ.get("V132_METHODS", ",".join(TIER1_METHODS)),
    )
    parser.add_argument(
        "--control-root", type=Path, default=env_path("V132_CONTROL_ROOT")
    )
    parser.add_argument("--v125-root", type=Path, default=env_path("V125_ROOT"))
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=env_path("V132_COMPARISON_ROOT"),
    )
    args = parser.parse_args()
    try:
        args.method_keys = parse_method_keys(args.methods)
    except ValueError as error:
        parser.error(str(error))
    args.repo_root = args.repo_root.resolve()
    digest = hashlib.sha256(
        ",".join(args.method_keys).encode("ascii")
    ).hexdigest()[:12]
    method_set_id = f"controls{len(args.method_keys)}_{digest}"
    args.prompts = (args.prompts or Path(DEFAULT_PROMPTS)).resolve()
    args.control_root = (
        args.control_root
        or args.repo_root / "runs" / CONTROL_EXPERIMENT / method_set_id
    ).resolve()
    args.v125_root = (
        args.v125_root
        or args.repo_root
        / "runs"
        / "v125_moviebench128_main"
        / "comparison_quality8"
    ).resolve()
    args.comparison_root = (
        args.comparison_root
        or args.repo_root
        / "runs"
        / "v132_binary_memory_ablation_comparison_30s"
        / method_set_id
    ).resolve()
    args.method_set_id = method_set_id
    return args


def prompt_items(path: Path) -> list[dict[str, object]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != PROMPT_COUNT:
        raise ValueError(f"expected 128 prompts, found {len(lines)}")
    return [{"index": index, "text": text} for index, text in enumerate(lines)]


def validate_sources(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    v125_path = args.v125_root / "comparison_manifest.json"
    control_path = args.control_root / "published_manifest.json"
    contract_path = args.control_root / "contracts" / "experiment.json"
    for path in (args.prompts, v125_path, control_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    v125 = load_json(v125_path)
    control = load_json(control_path)
    contract = load_json(contract_path)
    prompt_sha = sha256(args.prompts)

    v125_keys = [row.get("key") for row in v125.get("methods", [])]
    required_v125 = {"sf_native", "ours_prototype_retrieval1_age24"}
    failures = []
    if v125.get("prompt_count") != PROMPT_COUNT:
        failures.append("v125 prompt_count")
    if v125.get("prompt_file_sha256") != prompt_sha:
        failures.append("v125 prompt hash")
    if not required_v125 <= set(v125_keys):
        failures.append("v125 source methods")
    if control.get("experiment") != CONTROL_EXPERIMENT:
        failures.append("control experiment")
    if control.get("prompt_indices") != list(range(PROMPT_COUNT)):
        failures.append("control prompt coverage")
    if control.get("prompt_file_sha256") != prompt_sha:
        failures.append("control prompt hash")
    if [row.get("key") for row in control.get("methods", [])] != list(
        args.method_keys
    ):
        failures.append("control method order")
    if not control.get("ok"):
        failures.append("control audit")
    if contract.get("method_set_id") != args.method_set_id:
        failures.append("control method-set id")
    if control.get("experiment_contract_sha256") != sha256(contract_path):
        failures.append("control contract hash")
    if failures:
        raise RuntimeError("incompatible v132 sources: " + ", ".join(failures))
    return v125, control, contract


def source_video(directory: Path, index: int) -> Path:
    candidates = (
        directory / f"{index:06d}.mp4",
        directory / f"{index:06d}-0.mp4",
    )
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected one prompt {index} video under {directory}")
    return found[0]


def materialize(args: argparse.Namespace, source: Source) -> dict[str, object]:
    source_dir = source.root / "published" / source.source_key
    target_dir = args.comparison_root / "published" / source.final_key
    expected = {f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)}
    modes = {"existing": 0, "hardlink": 0, "symlink": 0}
    total_bytes = 0
    for index in range(PROMPT_COUNT):
        source_path = source_video(source_dir, index)
        total_bytes += source_path.stat().st_size
        mode = link_or_validate(
            source_path, target_dir / f"{index:06d}-0.mp4"
        )
        modes[mode] += 1
    actual = {path.name for path in target_dir.glob("*.mp4")}
    if actual != expected:
        raise RuntimeError(f"incomplete comparison method: {source.final_key}")
    return {
        "key": source.final_key,
        "role": source.role,
        "source_method": source.source_key,
        "source_video_dir": str(source_dir),
        "video_dir": str(target_dir.resolve()),
        "video_count": PROMPT_COUNT,
        "total_bytes": total_bytes,
        "link_modes": modes,
    }


def main() -> None:
    args = parse_args()
    _, _, contract = validate_sources(args)
    sources = (
        Source(
            "sf_native",
            args.v125_root,
            "sf_native",
            "same_backbone_baseline",
        ),
        Source(
            "ours_main",
            args.v125_root,
            "ours_prototype_retrieval1_age24",
            "selected_binary_memory",
        ),
        *(
            Source(key, args.control_root, key, f"ablation_{key}")
            for key in args.method_keys
        ),
    )
    rows = [materialize(args, source) for source in sources]
    manifest = {
        "version": 1,
        "experiment": "v132_binary_memory_ablation_comparison_30s",
        "method_set_id": args.method_set_id,
        "prompt_suite": "AMA MovieGen-128 Qwen Rewrite",
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": sha256(args.prompts),
        "prompt_items": prompt_items(args.prompts),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
            "width": 832,
            "height": 480,
        },
        "seed": 0,
        "reseed_per_prompt": True,
        "pf_required": False,
        "methods": rows,
        "metric_profiles": {
            "core": list(CORE_DIMENSIONS),
            "semantic_extension": list(SEMANTIC_DIMENSIONS),
            "full": list(CORE_DIMENSIONS[:-1])
            + list(SEMANTIC_DIMENSIONS)
            + ["overall_consistency"],
        },
        "head_maps": contract["head_maps"],
        "source_manifests": {
            "v125": {
                "path": str(args.v125_root / "comparison_manifest.json"),
                "sha256": sha256(
                    args.v125_root / "comparison_manifest.json"
                ),
            },
            "controls": {
                "path": str(args.control_root / "published_manifest.json"),
                "sha256": sha256(
                    args.control_root / "published_manifest.json"
                ),
            },
        },
    }
    path = args.comparison_root / "comparison_manifest.json"
    digest = write_frozen(path, manifest)
    print(
        f"[v132-comparison] methods={len(rows)} videos={len(rows) * 128} "
        f"sha256={digest} path={path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
