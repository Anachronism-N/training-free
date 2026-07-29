#!/usr/bin/env python3
"""Assemble the frozen no-PF 30-second MovieBench-128 paper comparison."""

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


PROMPT_COUNT = 128
NUM_OUTPUT_FRAMES = 120
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
SEMANTIC_EXTENSION_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
)
ALL_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)


@dataclass(frozen=True)
class Source:
    final_key: str
    source_group: str
    source_key: str
    role: str


SOURCES = (
    Source("sf_native", "v125", "sf_native", "same_backbone_baseline"),
    Source(
        "deep_forcing",
        "external",
        "deep_forcing",
        "same_checkpoint_external_method",
    ),
    Source(
        "rolling_forcing",
        "external",
        "rolling_forcing",
        "external_trained_system",
    ),
    Source(
        "longlive",
        "external",
        "longlive",
        "external_trained_system",
    ),
    Source(
        "ours_prototype_retrieval_age24",
        "v125",
        "ours_prototype_retrieval1_age24",
        "ours_no_gate_recent_fallback",
    ),
    Source(
        "ours_confidence_recent",
        "internal",
        "ours_prototype_retrieval_conf_recent",
        "ours_confidence_gate_recent_fallback",
    ),
    Source(
        "ours_prototype_retrieval_motion",
        "v125",
        "ours_prototype_retrieval_motion",
        "ours_no_gate_motion_fallback",
    ),
    Source(
        "ours_confidence_motion",
        "internal",
        "ours_prototype_retrieval_conf_motion",
        "ours_confidence_gate_motion_fallback",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def expected_internal_root(repo_root: Path) -> Path:
    candidates = (
        "prototype_retrieval_conf_recent",
        "prototype_retrieval_conf_motion",
    )
    digest = hashlib.sha256(",".join(candidates).encode()).hexdigest()[:12]
    return (
        repo_root
        / "runs"
        / "v129_moviebench128_30s_internal"
        / f"ours_only{len(candidates)}_{digest}"
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", root)),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(os.environ.get("V129_PROMPTS", DEFAULT_PROMPTS)),
    )
    parser.add_argument("--v125-root", type=Path)
    parser.add_argument("--internal-root", type=Path)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.prompts = args.prompts.resolve()
    args.v125_root = (
        args.v125_root
        or args.repo_root
        / "runs"
        / "v125_moviebench128_main"
        / "comparison_quality8"
    ).resolve()
    args.internal_root = (
        args.internal_root or expected_internal_root(args.repo_root)
    ).resolve()
    args.external_root = (
        args.external_root
        or args.repo_root / "runs" / "v129_moviebench128_30s_external"
    ).resolve()
    args.comparison_root = (
        args.comparison_root
        or args.repo_root / "runs" / "v129_paper_comparison_30s"
    ).resolve()
    return args


def prompt_items(path: Path) -> list[dict[str, Any]]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise ValueError(
            f"expected {PROMPT_COUNT} prompts, found {len(prompts)}"
        )
    return [
        {"index": index, "text": prompt}
        for index, prompt in enumerate(prompts)
    ]


def validate_sources(
    args: argparse.Namespace,
    expected_prompt_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    prompt_sha = sha256(args.prompts)
    paths = {
        "v125": args.v125_root / "comparison_manifest.json",
        "internal": args.internal_root / "published_manifest.json",
        "external": args.external_root / "published_manifest.json",
    }
    payloads = {key: load_json(path) for key, path in paths.items()}
    contract_paths = {
        "internal": args.internal_root / "contracts" / "experiment.json",
        "external": args.external_root / "contracts" / "experiment.json",
    }
    contracts = {
        key: load_json(path) for key, path in contract_paths.items()
    }

    v125 = payloads["v125"]
    v125_checks = {
        "experiment": (
            v125.get("experiment"),
            "v125_moviebench128_comparison",
        ),
        "prompt_count": (v125.get("prompt_count"), PROMPT_COUNT),
        "prompt_sha256": (v125.get("prompt_file_sha256"), prompt_sha),
        "prompt_items": (v125.get("prompt_items"), expected_prompt_items),
        "num_output_frames": (
            v125.get("num_output_frames"),
            NUM_OUTPUT_FRAMES,
        ),
        "decoded_video_contract": (
            v125.get("decoded_video_contract"),
            {
                "frames": 477,
                "fps": 16,
                "duration_seconds": 29.8125,
            },
        ),
        "seed": (v125.get("seed"), 0),
        "method_keys": (
            [row.get("key") for row in v125.get("methods", [])],
            [
                "sf_native",
                "pf_native",
                "ours_landmark_motion1",
                "ours_landmark_retrieval1_age24",
                "ours_landmark_retrieval_motion",
                "ours_prototype_motion1",
                "ours_prototype_retrieval1_age24",
                "ours_prototype_retrieval_motion",
            ],
        ),
    }
    internal = payloads["internal"]
    internal_contract = contracts["internal"]
    expected_internal_methods = [
        "ours_prototype_retrieval_conf_recent",
        "ours_prototype_retrieval_conf_motion",
    ]
    internal_checks = {
        "experiment": (
            internal.get("experiment"),
            "v129_moviebench128_30s_internal",
        ),
        "ok": (internal.get("ok"), True),
        "prompt_count": (internal.get("prompt_count"), PROMPT_COUNT),
        "prompt_sha256": (internal.get("prompt_file_sha256"), prompt_sha),
        "contract_sha256": (
            internal.get("experiment_contract_sha256"),
            sha256(contract_paths["internal"]),
        ),
        "method_keys": (
            [row.get("key") for row in internal.get("methods", [])],
            expected_internal_methods,
        ),
        "contract_experiment": (
            internal_contract.get("experiment"),
            "v129_moviebench128_30s_internal",
        ),
        "contract_prompt_count": (
            internal_contract.get("prompt_count"),
            PROMPT_COUNT,
        ),
        "contract_prompt_sha256": (
            internal_contract.get("prompts", {}).get("sha256"),
            prompt_sha,
        ),
        "contract_prompt_items": (
            internal_contract.get("prompts", {}).get("items"),
            expected_prompt_items,
        ),
        "contract_num_output_frames": (
            internal_contract.get("num_output_frames"),
            NUM_OUTPUT_FRAMES,
        ),
        "contract_decoded_video": (
            internal_contract.get("decoded_video_contract"),
            {
                "frames": 477,
                "fps": 16,
                "duration_seconds": 29.8125,
            },
        ),
        "contract_seed": (internal_contract.get("seed"), 0),
        "contract_candidate_keys": (
            internal_contract.get("candidate_keys"),
            [
                "prototype_retrieval_conf_recent",
                "prototype_retrieval_conf_motion",
            ],
        ),
        "contract_method_keys": (
            [
                row.get("key")
                for row in internal_contract.get("methods", [])
            ],
            expected_internal_methods,
        ),
    }
    external = payloads["external"]
    external_contract = contracts["external"]
    expected_external_methods = [
        "deep_forcing",
        "rolling_forcing",
        "longlive",
    ]
    external_checks = {
        "experiment": (
            external.get("experiment"),
            "v129_moviebench128_30s_external",
        ),
        "ok": (external.get("ok"), True),
        "prompt_count": (external.get("prompt_count"), PROMPT_COUNT),
        "prompt_sha256": (external.get("prompt_file_sha256"), prompt_sha),
        "contract_sha256": (
            external.get("experiment_contract_sha256"),
            sha256(contract_paths["external"]),
        ),
        "num_output_frames": (
            external.get("num_output_frames"),
            NUM_OUTPUT_FRAMES,
        ),
        "method_keys": (
            [row.get("key") for row in external.get("methods", [])],
            expected_external_methods,
        ),
        "contract_experiment": (
            external_contract.get("experiment"),
            "v129_moviebench128_30s_external",
        ),
        "contract_prompt_count": (
            external_contract.get("prompt_count"),
            PROMPT_COUNT,
        ),
        "contract_prompt_sha256": (
            external_contract.get("prompts", {}).get("sha256"),
            prompt_sha,
        ),
        "contract_prompt_items": (
            external_contract.get("prompts", {}).get("items"),
            expected_prompt_items,
        ),
        "contract_num_output_frames": (
            external_contract.get("num_output_frames"),
            NUM_OUTPUT_FRAMES,
        ),
        "contract_decoded_video": (
            external_contract.get("decoded_video_contract"),
            {
                "frames": 477,
                "fps": 16,
                "duration_seconds": 29.8125,
                "width": 832,
                "height": 480,
            },
        ),
        "contract_seed": (external_contract.get("seed"), 0),
        "contract_reseed_per_prompt": (
            external_contract.get("reseed_per_prompt"),
            True,
        ),
        "contract_single_process_per_gpu": (
            external_contract.get("assignment", {}).get(
                "single_process_per_gpu"
            ),
            True,
        ),
        "contract_distributed_env_scrubbed": (
            external_contract.get("assignment", {}).get(
                "distributed_launcher_environment_scrubbed"
            ),
            True,
        ),
        "contract_method_keys": (
            [
                row.get("key")
                for row in external_contract.get("methods", [])
            ],
            expected_external_methods,
        ),
    }
    failures = {}
    for group, checks in (
        ("v125", v125_checks),
        ("internal", internal_checks),
        ("external", external_checks),
    ):
        for key, (actual, expected) in checks.items():
            if actual != expected:
                failures[f"{group}.{key}"] = {
                    "actual": actual,
                    "expected": expected,
                }
    if failures:
        raise RuntimeError(
            "source comparison contracts are incompatible: "
            + json.dumps(failures, sort_keys=True)
        )
    return payloads


def source_directory(args: argparse.Namespace, source: Source) -> Path:
    roots = {
        "v125": args.v125_root,
        "internal": args.internal_root,
        "external": args.external_root,
    }
    return roots[source.source_group] / "published" / source.source_key


def source_video(path: Path, prompt_index: int) -> Path:
    candidates = (
        path / f"{prompt_index:06d}.mp4",
        path / f"{prompt_index:06d}-0.mp4",
    )
    found = [candidate for candidate in candidates if candidate.is_file()]
    if len(found) != 1:
        raise RuntimeError(
            f"expected one source for prompt {prompt_index} under {path}, "
            f"found {[str(candidate) for candidate in found]}"
        )
    return found[0]


def materialize_method(
    args: argparse.Namespace,
    source: Source,
) -> dict[str, Any]:
    source_dir = source_directory(args, source)
    target_dir = args.comparison_root / "published" / source.final_key
    expected_source_names = {
        f"{index:06d}.mp4" for index in range(PROMPT_COUNT)
    } | {
        f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)
    }
    observed_source_names = {
        path.name for path in source_dir.glob("*.mp4")
    }
    if len(observed_source_names) != PROMPT_COUNT or not (
        observed_source_names <= expected_source_names
    ):
        raise RuntimeError(
            f"{source.final_key}: invalid source video set under {source_dir}"
        )
    expected_targets = {
        f"{index:06d}-0.mp4" for index in range(PROMPT_COUNT)
    }
    extras = {
        path.name for path in target_dir.glob("*.mp4")
    } - expected_targets
    if extras:
        raise RuntimeError(
            f"{source.final_key}: unexpected comparison files {sorted(extras)}"
        )
    modes = {"existing": 0, "hardlink": 0, "symlink": 0}
    source_bytes = 0
    for prompt_index in range(PROMPT_COUNT):
        video = source_video(source_dir, prompt_index)
        source_bytes += video.stat().st_size
        mode = link_or_validate(
            video,
            target_dir / f"{prompt_index:06d}-0.mp4",
        )
        modes[mode] += 1
    actual_targets = {
        path.name for path in target_dir.glob("*.mp4")
    }
    if actual_targets != expected_targets:
        raise RuntimeError(f"incomplete target method: {source.final_key}")
    return {
        "key": source.final_key,
        "role": source.role,
        "source_group": source.source_group,
        "source_method": source.source_key,
        "source_video_dir": str(source_dir),
        "video_dir": str(target_dir.resolve()),
        "video_count": PROMPT_COUNT,
        "total_bytes": source_bytes,
        "link_modes": modes,
    }


def main() -> None:
    args = parse_args()
    expected_prompts = prompt_items(args.prompts)
    source_payloads = validate_sources(args, expected_prompts)
    method_rows = [
        materialize_method(args, source) for source in SOURCES
    ]
    source_manifests = {
        "v125": args.v125_root / "comparison_manifest.json",
        "internal": args.internal_root / "published_manifest.json",
        "external": args.external_root / "published_manifest.json",
    }
    manifest = {
        "version": 1,
        "experiment": "v129_no_pf_paper_comparison_30s",
        "prompt_suite": "AMA MovieGen-128 Qwen Rewrite",
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": sha256(args.prompts),
        "prompt_items": expected_prompts,
        "num_output_frames": NUM_OUTPUT_FRAMES,
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
        "methods": method_rows,
        "metric_profiles": {
            "core": list(CORE_DIMENSIONS),
            "semantic_extension": list(SEMANTIC_EXTENSION_DIMENSIONS),
            "full": list(ALL_DIMENSIONS),
        },
        "source_manifests": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "experiment": source_payloads[key].get("experiment"),
            }
            for key, path in source_manifests.items()
        },
        "source_contracts": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for key, path in {
                "internal": (
                    args.internal_root / "contracts" / "experiment.json"
                ),
                "external": (
                    args.external_root / "contracts" / "experiment.json"
                ),
            }.items()
        },
    }
    manifest_path = args.comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_path, manifest)
    print(
        f"[v129-comparison] methods={len(method_rows)} "
        f"videos={len(method_rows) * PROMPT_COUNT} "
        f"manifest_sha256={digest} path={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
