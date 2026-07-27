#!/usr/bin/env python3
"""Assemble the audited v125 Qwen-rewrite MovieBench-128 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROMPT_COUNT = 128
COMPARISON_DIR_NAME = "comparison_quality8"
QWEN_PROMPT_PATH = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
    "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)
REWRITE_SCRIPT_PATH = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
    "research_sprint/RollingForcing/scripts/prompt_refine_qwen.py"
)
OURS_CANDIDATES = (
    "landmark_motion1",
    "landmark_retrieval1_age24",
    "landmark_retrieval_motion",
    "prototype_motion1",
    "prototype_retrieval1_age24",
    "prototype_retrieval_motion",
)
SOURCE_METHODS = {
    "sf_native": "sf_native",
    "pf_native": "pf_native",
    "ours_landmark_motion1": "ours_landmark_motion1",
    "ours_retrieval_age24": "ours_landmark_retrieval1_age24",
    "ours_retrieval_motion": "ours_landmark_retrieval_motion",
    "ours_prototype_motion1": "ours_prototype_motion1",
    "ours_prototype_retrieval_age24": (
        "ours_prototype_retrieval1_age24"
    ),
    "ours_prototype_retrieval_motion": (
        "ours_prototype_retrieval_motion"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_frozen(path: Path, payload: Any) -> str:
    encoded = canonical_json(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen artifact differs: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return digest


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def expected_source_root(repo_root: Path) -> Path:
    digest = hashlib.sha256(",".join(OURS_CANDIDATES).encode()).hexdigest()[:12]
    return (
        repo_root
        / "runs"
        / "v125_moviebench128_main"
        / f"ours{len(OURS_CANDIDATES)}_{digest}"
    )


def comparison_name(prompt_index: int) -> str:
    # The trailing "-0" is required by both VBench-Long split reuse and
    # prompt-index recovery in paired statistics.
    return f"{int(prompt_index):06d}-0.mp4"


def validate_source_contract(
    source_root: Path,
    *,
    prompt_path: Path,
    prompt_sha256: str,
    prompts: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = source_root / "published_manifest.json"
    contract_path = source_root / "contracts" / "experiment.json"
    manifest = load_json(manifest_path)
    contract = load_json(contract_path)
    if not isinstance(manifest, dict) or not manifest.get("ok"):
        raise RuntimeError(f"source manifest is not successful: {manifest_path}")
    if not isinstance(contract, dict):
        raise RuntimeError(f"invalid source contract: {contract_path}")
    expected_source_methods = list(SOURCE_METHODS.values())
    actual_source_methods = [
        row.get("key") for row in manifest.get("methods", [])
    ]
    prompt_items = contract.get("prompts", {}).get("items")
    checks = {
        "manifest.prompt_count": (
            manifest.get("prompt_count"),
            PROMPT_COUNT,
        ),
        "manifest.prompt_sha256": (
            manifest.get("prompt_file_sha256"),
            prompt_sha256,
        ),
        "manifest.methods": (
            actual_source_methods,
            expected_source_methods,
        ),
        "contract.experiment": (
            contract.get("experiment"),
            "v125_moviebench128_main",
        ),
        "contract.prompt_count": (
            contract.get("prompt_count"),
            PROMPT_COUNT,
        ),
        "contract.prompt_sha256": (
            contract.get("prompts", {}).get("sha256"),
            prompt_sha256,
        ),
        "contract.candidates": (
            contract.get("candidate_keys"),
            list(OURS_CANDIDATES),
        ),
        "contract.baseline_only": (
            contract.get("baseline_only"),
            False,
        ),
        "contract.frames": (contract.get("num_output_frames"), 120),
        "contract.decoded_video": (
            contract.get("decoded_video_contract"),
            {
                "frames": 477,
                "fps": 16,
                "duration_seconds": 29.8125,
            },
        ),
        "contract.seed": (contract.get("seed"), 0),
        "contract.method_scope": (
            [row.get("key") for row in contract.get("methods", [])],
            expected_source_methods,
        ),
        "contract.prompt_items": (
            len(prompt_items) if isinstance(prompt_items, list) else None,
            PROMPT_COUNT,
        ),
    }
    failures = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if failures:
        raise RuntimeError(
            "v125 source contract is incompatible with comparison: "
            + json.dumps(failures, sort_keys=True)
        )
    expected_prompt_items = [
        {"index": index, "text": text}
        for index, text in enumerate(prompts)
    ]
    if prompt_items != expected_prompt_items:
        raise RuntimeError(
            "v125 source contract prompt items differ from the supplied "
            "Qwen Rewrite"
        )
    if Path(str(contract["prompts"]["path"])).resolve() != prompt_path:
        raise RuntimeError(
            "v125 generation used another prompt path despite matching hash: "
            f"{contract['prompts']['path']}"
        )
    return manifest, contract


def validated_method_sources(
    source_root: Path,
    source_method: str,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    method_dir = (source_root / "published" / source_method).resolve()
    sources: dict[int, Path] = {}
    marker_rows: list[dict[str, Any]] = []
    for prompt_index in range(PROMPT_COUNT):
        target = method_dir / f"{prompt_index:06d}.mp4"
        marker_path = (
            source_root
            / "status"
            / "published"
            / f"{source_method}.p{prompt_index:03d}.json"
        )
        marker = load_json(marker_path)
        if not isinstance(marker, dict):
            raise RuntimeError(f"invalid source marker: {marker_path}")
        source = Path(str(marker.get("source", "")))
        marker_target = Path(str(marker.get("target", "")))
        if (
            marker.get("method") != source_method
            or int(marker.get("prompt_index", -1)) != prompt_index
            or marker_target.resolve() != target.resolve()
            or not source.is_file()
            or not target.is_file()
            or not target.samefile(source)
            or int(marker.get("size", -1)) != target.stat().st_size
        ):
            raise RuntimeError(f"mixed or stale source marker: {marker_path}")
        sources[prompt_index] = target
        marker_rows.append(
            {
                "prompt_index": prompt_index,
                "marker": str(marker_path.resolve()),
                "source": str(source.resolve()),
                "size": target.stat().st_size,
            }
        )
    observed = {path.name for path in method_dir.glob("*.mp4")}
    expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    if observed != expected:
        raise RuntimeError(
            f"unexpected source videos for {source_method}: "
            f"missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )
    return sources, marker_rows


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or not target.samefile(source):
            raise RuntimeError(
                f"comparison target points to another video: {target}"
            )
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def materialize_method(
    comparison_root: Path,
    method: str,
    sources: dict[int, Path],
) -> dict[str, int]:
    target_dir = comparison_root / "published" / method
    expected_names = {
        comparison_name(index) for index in range(PROMPT_COUNT)
    }
    if target_dir.exists():
        extras = {
            path.name for path in target_dir.glob("*.mp4")
        } - expected_names
        if extras:
            raise RuntimeError(
                f"unexpected files in comparison method {method}: "
                f"{sorted(extras)}"
            )
    modes = {"existing": 0, "hardlink": 0, "symlink": 0}
    for prompt_index in range(PROMPT_COUNT):
        mode = link_or_validate(
            sources[prompt_index],
            target_dir / comparison_name(prompt_index),
        )
        modes[mode] += 1
    observed = {path.name for path in target_dir.glob("*.mp4")}
    if observed != expected_names:
        raise RuntimeError(f"incomplete comparison method: {method}")
    return modes


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", root)),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(os.environ.get("V125_PROMPTS", QWEN_PROMPT_PATH)),
    )
    parser.add_argument(
        "--rewrite-script",
        type=Path,
        default=Path(
            os.environ.get("V125_REWRITE_SCRIPT", REWRITE_SCRIPT_PATH)
        ),
    )
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.source_root = (
        args.source_root or expected_source_root(args.repo_root)
    ).resolve()
    args.comparison_root = (
        args.comparison_root
        or args.repo_root
        / "runs"
        / "v125_moviebench128_main"
        / COMPARISON_DIR_NAME
    ).resolve()
    args.prompts = args.prompts.resolve()
    args.rewrite_script = args.rewrite_script.resolve()
    return args


def main() -> None:
    args = parse_args()
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit(
            f"expected {PROMPT_COUNT} non-empty prompts, found {len(prompts)}"
        )
    if not args.rewrite_script.is_file():
        raise SystemExit(
            f"missing Qwen rewrite provenance script: {args.rewrite_script}"
        )
    prompt_sha = sha256(args.prompts)
    source_manifest, source_contract = validate_source_contract(
        args.source_root,
        prompt_path=args.prompts,
        prompt_sha256=prompt_sha,
        prompts=prompts,
    )

    method_rows: list[dict[str, Any]] = []
    for method, source_method in SOURCE_METHODS.items():
        sources, markers = validated_method_sources(
            args.source_root,
            source_method,
        )
        modes = materialize_method(args.comparison_root, method, sources)
        method_rows.append(
            {
                "key": method,
                "role": (
                    "external_baseline"
                    if method in {"sf_native", "pf_native"}
                    else "v125_quality_candidate"
                ),
                "source_experiment": "v125_moviebench128_main",
                "source_method": source_method,
                "source_root": str(args.source_root),
                "source_marker_count": len(markers),
                "video_dir": str(
                    (args.comparison_root / "published" / method).resolve()
                ),
                "link_modes": modes,
            }
        )

    source_contract_path = (
        args.source_root / "contracts" / "experiment.json"
    )
    source_manifest_path = args.source_root / "published_manifest.json"
    comparison_manifest = {
        "version": 2,
        "experiment": "v125_moviebench128_comparison",
        "prompt_suite": "AMA MovieGen-128 Qwen Rewrite",
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": prompt_sha,
        "prompt_items": [
            {"index": index, "text": text}
            for index, text in enumerate(prompts)
        ],
        "rewrite_script": str(args.rewrite_script),
        "rewrite_script_sha256": sha256(args.rewrite_script),
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "seed": 0,
        "reseed_per_prompt": True,
        "methods": method_rows,
        "source": {
            "manifest": str(source_manifest_path.resolve()),
            "manifest_sha256": sha256(source_manifest_path),
            "contract": str(source_contract_path.resolve()),
            "contract_sha256": sha256(source_contract_path),
            "implementation_hashes": source_contract.get(
                "implementation_hashes", {}
            ),
            "published_method_count": len(
                source_manifest.get("methods", [])
            ),
        },
        "vbench_long_dimensions": [
            "subject_consistency",
            "background_consistency",
            "aesthetic_quality",
            "imaging_quality",
            "motion_smoothness",
            "dynamic_degree",
        ],
    }
    manifest_path = args.comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_path, comparison_manifest)
    print(
        f"[v125-comparison] methods={len(method_rows)} "
        f"videos={len(method_rows) * PROMPT_COUNT} "
        f"prompt_sha256={prompt_sha} "
        f"manifest_sha256={digest} path={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
