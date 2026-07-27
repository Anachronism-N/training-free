#!/usr/bin/env python3
"""Evaluate selected v115 methods on a frozen diverse MovieBench-16 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    run_cell,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)
from run_v109_legacy_v98_suppressive_cache_1video import (
    EXPECTED_MAP_SHA256,
    validate_frozen_map,
)
from run_v111_role_event_cache_1video import CELLS as V111_CELLS
from run_v115_role_memory_cache_1video import CELLS as V115_CELLS


EXPERIMENT = "v116_role_memory_diverse16"
PROMPT_COUNT = 16


@dataclass(frozen=True)
class Method:
    key: str
    source_cell: Cell
    role: str


@dataclass(frozen=True)
class PromptItem:
    subset_index: int
    source_index: int
    source_number: int
    tags: tuple[str, ...]
    text: str


_CELLS_BY_NAME = {
    cell.name: cell for cell in (*V111_CELLS, *V115_CELLS)
}
_METHOD_SPECS = {
    # Reused v111 controls.
    "control_all_recent8": (
        "legacy_v98_all_recent8_control",
        "local_control",
    ),
    "control_all_landmark4": (
        "legacy_v98_all_landmark4_control",
        "same_route_control",
    ),
    "control_landmark_recent": (
        "legacy_v98_support_landmark4_suppress_recent8",
        "v111_candidate",
    ),
    "control_landmark_motion2": (
        "legacy_v98_support_landmark4_suppress_motion_pair2",
        "v111_candidate",
    ),
    # Isolated Supportive routes.
    "support_prototype_recent": (
        "legacy_v98_support_prototype4_suppress_recent8",
        "supportive_ablation",
    ),
    "support_snapshot_recent": (
        "legacy_v98_support_snapshot4_suppress_recent8",
        "supportive_ablation",
    ),
    "support_retrieval2_recent": (
        "legacy_v98_support_retrieval2_suppress_recent8",
        "supportive_ablation",
    ),
    "support_retrieval4_recent": (
        "legacy_v98_support_retrieval4_suppress_recent8",
        "supportive_ablation",
    ),
    "support_sparse75_recent": (
        "legacy_v98_support_sparse75_suppress_recent8",
        "supportive_ablation",
    ),
    # Isolated Suppressive routes.
    "landmark_prototype2": (
        "legacy_v98_support_landmark4_suppress_prototype2",
        "suppressive_ablation",
    ),
    "landmark_snapshot2": (
        "legacy_v98_support_landmark4_suppress_snapshot2",
        "suppressive_ablation",
    ),
    "landmark_retrieval2": (
        "legacy_v98_support_landmark4_suppress_retrieval2",
        "suppressive_ablation",
    ),
    "landmark_motion1": (
        "legacy_v98_support_landmark4_suppress_motion_pair1",
        "suppressive_ablation",
    ),
    "landmark_sparse75": (
        "legacy_v98_support_landmark4_suppress_sparse75",
        "suppressive_ablation",
    ),
    # Joint candidates.
    "prototype_motion1": (
        "legacy_v98_support_prototype4_suppress_motion_pair1",
        "joint_candidate",
    ),
    "snapshot_motion1": (
        "legacy_v98_support_snapshot4_suppress_motion_pair1",
        "joint_candidate",
    ),
    "retrieval2_motion1": (
        "legacy_v98_support_retrieval2_suppress_motion_pair1",
        "joint_candidate",
    ),
    "sparse75_motion1": (
        "legacy_v98_support_sparse75_suppress_motion_pair1",
        "joint_candidate",
    ),
    "control_all_prototype4": (
        "legacy_v98_all_prototype4_control",
        "same_route_control",
    ),
    "control_all_snapshot4": (
        "legacy_v98_all_snapshot4_control",
        "same_route_control",
    ),
}
DEFAULT_METHODS = (
    "prototype_motion1",
    "snapshot_motion1",
    "control_landmark_recent",
    "control_all_recent8",
)


def parse_method_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not keys:
        raise ValueError("at least one method is required")
    if len(keys) != len(set(keys)):
        raise ValueError("method list contains duplicates")
    unknown = sorted(set(keys) - set(_METHOD_SPECS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    return keys


def methods_for(keys: tuple[str, ...]) -> tuple[Method, ...]:
    return tuple(
        Method(
            key=key,
            source_cell=_CELLS_BY_NAME[_METHOD_SPECS[key][0]],
            role=_METHOD_SPECS[key][1],
        )
        for key in keys
    )


def load_prompt_items(
    source_path: Path,
    manifest_path: Path,
) -> tuple[list[PromptItem], dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_lines = [
        line.strip()
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_lines) != int(payload["source_prompt_count"]):
        raise ValueError("MovieBench source prompt count changed")
    if sha256(source_path) != str(payload["source_sha256"]):
        raise ValueError("MovieBench source SHA256 changed")
    rows = payload["items"]
    if len(rows) != PROMPT_COUNT:
        raise ValueError(f"manifest must contain {PROMPT_COUNT} prompts")
    items = []
    for expected_subset, row in enumerate(rows):
        subset_index = int(row["subset_index"])
        source_index = int(row["source_index"])
        source_number = int(row["source_number"])
        if subset_index != expected_subset:
            raise ValueError("subset indices must be contiguous and ordered")
        if source_number != source_index + 1:
            raise ValueError("source_number must be one-based source_index")
        if not 0 <= source_index < len(source_lines):
            raise ValueError("source prompt index is outside MovieBench")
        items.append(
            PromptItem(
                subset_index=subset_index,
                source_index=source_index,
                source_number=source_number,
                tags=tuple(str(value) for value in row["tags"]),
                text=source_lines[source_index],
            )
        )
    if len({item.source_index for item in items}) != PROMPT_COUNT:
        raise ValueError("MovieBench-16 source indices must be unique")
    return items, payload


def write_frozen_text(path: Path, text: str) -> str:
    content = text.encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing mixed frozen text artifact: {path}")
        return digest
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_bytes(content)
    try:
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return digest


def published_name(subset_index: int, *, indexed: bool = False) -> str:
    if indexed:
        return f"{int(subset_index):06d}-0_v116.mp4"
    return f"{int(subset_index):06d}.mp4"


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.samefile(source):
            raise RuntimeError(
                f"refusing mixed published video {target}; source={source}"
            )
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source.resolve())
        return "symlink"


def task_cell(method: Method, prompt: PromptItem) -> Cell:
    return replace(
        method.source_cell,
        name=(
            f"{method.key}__s{prompt.subset_index:02d}"
            f"_p{prompt.source_index:03d}"
        ),
        stage="diverse16",
    )


def all_tasks(
    methods: tuple[Method, ...],
    prompts: list[PromptItem],
) -> tuple[tuple[Method, PromptItem, Cell], ...]:
    return tuple(
        (method, prompt, task_cell(method, prompt))
        for method in methods
        for prompt in prompts
    )


def selected_tasks(
    methods: tuple[Method, ...],
    prompts: list[PromptItem],
    *,
    node_rank: int,
    num_nodes: int,
) -> tuple[tuple[Method, PromptItem, Cell], ...]:
    return all_tasks(methods, prompts)[node_rank::num_nodes]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("generate", "audit"),
        default="generate",
    )
    parser.add_argument(
        "--methods",
        default=os.environ.get("V116_METHODS", ",".join(DEFAULT_METHODS)),
        help="comma-separated keys; use --list-methods to inspect choices",
    )
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", "0")),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=int(os.environ.get("NUM_NODES", "1")),
    )
    parser.add_argument(
        "--gpu-list",
        default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", root)),
    )
    parser.add_argument("--out-root", type=Path, default=env_path("OUT_ROOT"))
    parser.add_argument("--pf-repo", type=Path, default=env_path("PF_REPO"))
    parser.add_argument("--pf-config", type=Path, default=env_path("PF_CONFIG"))
    parser.add_argument(
        "--pf-checkpoint",
        type=Path,
        default=env_path("PF_CHECKPOINT"),
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument(
        "--legacy-map",
        type=Path,
        default=env_path("LEGACY_V98_MAP"),
    )
    parser.add_argument(
        "--single-prompts",
        type=Path,
        default=env_path("SINGLE_PROMPTS"),
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=env_path("PROMPT_MANIFEST"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--promotion-approved",
        action="store_true",
        default=os.environ.get("V115_PROMOTION_APPROVED", "0") == "1",
        help="confirm that the v115 one-video review selected these methods",
    )
    args = parser.parse_args()

    if args.list_methods:
        for key, (cell_name, role) in _METHOD_SPECS.items():
            print(f"{key}\t{role}\t{cell_name}")
        raise SystemExit(0)
    try:
        args.method_keys = parse_method_keys(args.methods)
    except ValueError as error:
        parser.error(str(error))
    args.repo_root = args.repo_root.resolve()
    args.pf_repo = (
        args.pf_repo or args.repo_root / "third_party" / "Pyramid-Forcing"
    ).resolve()
    args.pf_config = (
        args.pf_config or args.pf_repo / "configs" / "pyramid-forcing.yaml"
    ).resolve()
    args.pf_checkpoint = (
        args.pf_checkpoint
        or args.pf_repo / "checkpoints" / "self_forcing_dmd.pt"
    ).resolve()
    args.pf_labels = (
        args.pf_labels
        or args.pf_repo / "configs" / "head_configs" / "best_labels.csv"
    ).resolve()
    args.legacy_map = (
        args.legacy_map
        or args.repo_root
        / "configs"
        / "head_maps"
        / "legacy_v98_absolute_sign_304_56.csv"
    ).resolve()
    args.single_prompts = (
        args.single_prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num128.txt"
    ).resolve()
    args.prompt_manifest = (
        args.prompt_manifest
        or args.repo_root / "prompts" / "moviegenbench_diverse16.json"
    ).resolve()
    args.aba_prompts = (
        args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    method_digest = hashlib.sha256(
        ",".join(args.method_keys).encode("utf-8")
    ).hexdigest()[:12]
    args.method_set_id = f"m{len(args.method_keys)}_{method_digest}"
    args.out_root = (
        args.out_root
        or args.repo_root
        / "runs"
        / EXPERIMENT
        / args.method_set_id
    ).resolve()
    args.experiment_name = EXPERIMENT
    return args


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[list[PromptItem], dict]:
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v116 screen requires seed 0")
    if args.mode == "generate" and not args.promotion_approved:
        raise SystemExit(
            "v116 is gated by v115 one-video review; set "
            "V115_PROMOTION_APPROVED=1 after recording the decision"
        )
    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.single_prompts,
        args.prompt_manifest,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    try:
        return load_prompt_items(args.single_prompts, args.prompt_manifest)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid diverse16 prompt manifest: {error}") from error


def experiment_contract(
    args: argparse.Namespace,
    prompts: list[PromptItem],
    prompt_manifest: dict,
    map_audit: dict,
) -> dict:
    methods = methods_for(args.method_keys)
    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        args.repo_root / "scripts" / "run_v115_role_memory_cache_1video.py",
        args.prompt_manifest,
        args.legacy_map,
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "role_event.py",
        args.pf_repo / "pyramidkv" / "role_memory.py",
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "method_keys": list(args.method_keys),
        "seed": 0,
        "prompt_count": PROMPT_COUNT,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "head_membership": {
            "supportive": 304,
            "suppressive": 56,
            "map_sha256": EXPECTED_MAP_SHA256,
        },
        "cache_contract": {
            "max_full_frame_equivalents": 9,
            "clean_kv_only": True,
            "exclusive_dynamic_owner": True,
        },
        "methods": [
            {
                "key": method.key,
                "role": method.role,
                "source_cell": asdict(method.source_cell),
            }
            for method in methods
        ],
        "prompts": {
            "source_path": str(args.single_prompts),
            "source_sha256": sha256(args.single_prompts),
            "manifest_path": str(args.prompt_manifest),
            "manifest_sha256": sha256(args.prompt_manifest),
            "subset_text_sha256": hashlib.sha256(
                "".join(f"{item.text}\n" for item in prompts).encode("utf-8")
            ).hexdigest(),
            "selection_policy": prompt_manifest["selection_policy"],
            "items": [asdict(item) for item in prompts],
        },
        "map_audit": map_audit,
        "pf_labels": {
            "path": str(args.pf_labels),
            "sha256": sha256(args.pf_labels),
        },
        "config": {
            "path": str(args.pf_config),
            "sha256": sha256(args.pf_config),
        },
        "checkpoint": {
            "path": str(args.pf_checkpoint),
            "size": args.pf_checkpoint.stat().st_size,
        },
        "implementation_hashes": {
            str(path.relative_to(args.repo_root)): sha256(path)
            for path in implementation_paths
        },
    }


def publish_video(
    args: argparse.Namespace,
    *,
    method: Method,
    prompt: PromptItem,
    cell: Cell,
    contract_sha256: str,
) -> dict[str, object]:
    source_dir = args.out_root / "videos" / cell.name
    videos = sorted(source_dir.rglob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(
            f"{cell.name}: expected one source video, found {len(videos)}"
        )
    source = videos[0]
    target = (
        args.out_root
        / "published"
        / method.key
        / published_name(prompt.subset_index)
    )
    indexed_target = (
        args.out_root
        / "published_indexed"
        / method.key
        / published_name(prompt.subset_index, indexed=True)
    )
    link_mode = link_or_validate(source, target)
    indexed_link_mode = link_or_validate(source, indexed_target)
    marker = (
        args.out_root
        / "status"
        / "published"
        / f"{method.key}.s{prompt.subset_index:02d}.json"
    )
    payload = {
        "version": 1,
        "experiment_contract_sha256": contract_sha256,
        "method": method.key,
        "subset_index": prompt.subset_index,
        "source_index": prompt.source_index,
        "task_cell": cell.name,
        "source": str(source),
        "target": str(target),
        "indexed_target": str(indexed_target),
        "size": source.stat().st_size,
    }
    write_frozen(marker, payload)
    return {
        "method": method.key,
        "subset_index": prompt.subset_index,
        "source_index": prompt.source_index,
        "status": link_mode,
        "indexed_status": indexed_link_mode,
    }


def run_task(
    args: argparse.Namespace,
    *,
    method: Method,
    prompt: PromptItem,
    cell: Cell,
    gpu: str,
    contract_sha256: str,
) -> dict[str, object]:
    task_args = argparse.Namespace(**vars(args))
    task_args.single_prompt_index = int(prompt.source_index)
    result = run_cell(
        task_args,
        cell=cell,
        gpu=gpu,
        experiment_contract_sha256=contract_sha256,
    )
    published = publish_video(
        args,
        method=method,
        prompt=prompt,
        cell=cell,
        contract_sha256=contract_sha256,
    )
    return {
        **published,
        "generation_status": result["status"],
        "gpu": str(gpu),
    }


def run_worker(
    args: argparse.Namespace,
    *,
    gpu: str,
    tasks: list[tuple[Method, PromptItem, Cell]],
    contract_sha256: str,
) -> list[dict[str, object]]:
    results = []
    for method, prompt, cell in tasks:
        print(
            f"[v116-task] gpu={gpu} method={method.key} "
            f"subset={prompt.subset_index} source={prompt.source_index}",
            flush=True,
        )
        results.append(
            run_task(
                args,
                method=method,
                prompt=prompt,
                cell=cell,
                gpu=gpu,
                contract_sha256=contract_sha256,
            )
        )
    return results


def audit_published(
    args: argparse.Namespace,
    *,
    methods: tuple[Method, ...],
    prompts: list[PromptItem],
    contract_sha256: str,
) -> dict[str, object]:
    failures: list[str] = []
    method_rows: list[dict[str, object]] = []
    expected_names = {
        published_name(prompt.subset_index) for prompt in prompts
    }
    expected_indexed_names = {
        published_name(prompt.subset_index, indexed=True)
        for prompt in prompts
    }
    for method in methods:
        method_dir = args.out_root / "published" / method.key
        indexed_method_dir = (
            args.out_root / "published_indexed" / method.key
        )
        observed = sorted(method_dir.glob("*.mp4"))
        observed_indexed = sorted(indexed_method_dir.glob("*.mp4"))
        actual_names = {path.name for path in observed}
        actual_indexed_names = {path.name for path in observed_indexed}
        if actual_names != expected_names:
            failures.append(
                f"{method.key}: missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}"
            )
        if actual_indexed_names != expected_indexed_names:
            failures.append(
                f"{method.key}: indexed_missing="
                f"{sorted(expected_indexed_names - actual_indexed_names)} "
                f"indexed_extra="
                f"{sorted(actual_indexed_names - expected_indexed_names)}"
            )
        sizes = []
        for prompt in prompts:
            target = (
                method_dir / published_name(prompt.subset_index)
            )
            indexed_target = (
                indexed_method_dir
                / published_name(prompt.subset_index, indexed=True)
            )
            marker = (
                args.out_root
                / "status"
                / "published"
                / f"{method.key}.s{prompt.subset_index:02d}.json"
            )
            if (
                not target.is_file()
                or not indexed_target.is_file()
                or not marker.is_file()
            ):
                failures.append(
                    f"{method.key}:{prompt.subset_index}: missing artifact"
                )
                continue
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                source = Path(payload["source"])
                if (
                    payload["experiment_contract_sha256"] != contract_sha256
                    or payload["method"] != method.key
                    or int(payload["subset_index"]) != prompt.subset_index
                    or int(payload["source_index"]) != prompt.source_index
                    or Path(payload["target"]) != target
                    or Path(payload["indexed_target"]) != indexed_target
                    or not source.is_file()
                    or not target.samefile(source)
                    or not indexed_target.samefile(source)
                    or int(payload["size"]) != target.stat().st_size
                    or indexed_target.stat().st_size
                    != target.stat().st_size
                ):
                    failures.append(
                        f"{method.key}:{prompt.subset_index}: mixed marker"
                    )
                    continue
            except (KeyError, TypeError, ValueError, OSError) as error:
                failures.append(
                    f"{method.key}:{prompt.subset_index}: invalid marker: "
                    f"{error}"
                )
                continue
            sizes.append(target.stat().st_size)
        method_rows.append(
            {
                "key": method.key,
                "role": method.role,
                "video_dir": str(method_dir),
                "indexed_video_dir": str(indexed_method_dir),
                "video_count": len(observed),
                "indexed_video_count": len(observed_indexed),
                "total_bytes": sum(sizes),
            }
        )
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "experiment_contract_sha256": contract_sha256,
        "prompt_count": PROMPT_COUNT,
        "prompt_file": str(
            args.out_root / "prompts" / "moviegenbench_diverse16.txt"
        ),
        "prompt_file_sha256": sha256(
            args.out_root / "prompts" / "moviegenbench_diverse16.txt"
        ),
        "methods": method_rows,
        "failures": failures,
        "ok": not failures,
    }
    if failures:
        write_runtime_json(
            args.out_root / "status" / "published.audit.failed.json",
            payload,
        )
        raise RuntimeError(f"published audit failed: {failures[:5]}")
    write_frozen(args.out_root / "published_manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    prompts, prompt_manifest = validate_inputs(args)
    methods = methods_for(args.method_keys)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "logs",
        "traces",
        "configs",
        "status",
        "status/published",
        "diagnostics",
        "contracts",
        "published",
        "published_indexed",
        "prompts",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)
    subset_prompt_path = (
        args.out_root / "prompts" / "moviegenbench_diverse16.txt"
    )
    subset_prompt_sha = write_frozen_text(
        subset_prompt_path,
        "".join(f"{item.text}\n" for item in prompts),
    )

    map_audit = validate_frozen_map(args.legacy_map, args.pf_labels)
    if map_audit["sha256"] != EXPECTED_MAP_SHA256:
        raise SystemExit("legacy v98 map changed after validation")
    args.head_map_audits = {"legacy": map_audit}
    contract = experiment_contract(
        args,
        prompts,
        prompt_manifest,
        map_audit,
    )
    if contract["prompts"]["subset_text_sha256"] != subset_prompt_sha:
        raise RuntimeError("materialized prompt subset differs from contract")
    contract_path = args.out_root / "contracts" / "experiment.json"
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            args.contract_wait_seconds,
        )
    print(
        "[V116Contract] "
        + canonical_json(
            {
                "method_set_id": args.method_set_id,
                "methods": list(args.method_keys),
                "contract_sha256": contract_sha,
                "map_sha256": map_audit["sha256"],
            }
        )
        .decode("utf-8")
        .strip(),
        flush=True,
    )

    if args.mode == "audit":
        payload = audit_published(
            args,
            methods=methods,
            prompts=prompts,
            contract_sha256=contract_sha,
        )
        print(
            f"[complete] published methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    tasks = list(
        selected_tasks(
            methods,
            prompts,
            node_rank=args.node_rank,
            num_nodes=args.num_nodes,
        )
    )
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    worker_tasks = [tasks[index::len(gpus)] for index in range(len(gpus))]
    worker_tasks = [items for items in worker_tasks if items]
    print(
        f"[v116] node={args.node_rank}/{args.num_nodes} "
        f"tasks={len(tasks)} workers={len(worker_tasks)} "
        f"out={args.out_root}",
        flush=True,
    )

    failures: list[str] = []
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(worker_tasks))) as executor:
        futures = {
            executor.submit(
                run_worker,
                args,
                gpu=gpus[index],
                tasks=items,
                contract_sha256=contract_sha,
            ): gpus[index]
            for index, items in enumerate(worker_tasks)
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                results.extend(future.result())
            except Exception as error:
                failures.append(f"gpu={gpu}: {error}")
                print(f"[failed] gpu={gpu}: {error}", flush=True)

    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "task_count": len(tasks),
        "results": sorted(
            results,
            key=lambda item: (
                str(item["method"]),
                int(item["subset_index"]),
            ),
        ),
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root
        / "status"
        / f"node{args.node_rank}.summary.json"
    )
    write_runtime_json(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
