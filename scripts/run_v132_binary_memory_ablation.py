#!/usr/bin/env python3
"""Run screened or full binary-role memory ablations on MovieBench-128."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import run_v120_moviebench32_main as base
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    read_matrix,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)
from run_v109_legacy_v98_suppressive_cache_1video import (
    EXPECTED_MAP_SHA256,
    validate_frozen_map,
)


EXPERIMENT = "v132_binary_memory_ablation30"
PROMPT_COUNT = 128
NUM_OUTPUT_FRAMES = 120
RANDOM_MAP_SEED = 20260729
DEFAULT_PROMPTS = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/"
    "research_sprint/Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)
SCREEN16 = (0, 1, 4, 7, 13, 15, 17, 24, 33, 47, 61, 67, 75, 84, 109, 124)


CONTROL_CELLS = {
    "random_binary": Cell(
        "random_binary",
        "paper_ablation",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        map_key="random",
    ),
    "inverted_binary": Cell(
        "inverted_binary",
        "paper_ablation",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        map_key="inverted",
    ),
    "all_supportive": Cell(
        "all_supportive",
        "paper_ablation",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        map_key="all_supportive",
    ),
    "all_suppressive": Cell(
        "all_suppressive",
        "paper_ablation",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        map_key="all_suppressive",
    ),
    "no_prototype": Cell(
        "no_prototype",
        "paper_ablation",
        "single",
        support_policy="recent8",
        suppress_policy="retrieval1_age24",
        map_key="legacy",
    ),
    "no_retrieval": Cell(
        "no_retrieval",
        "paper_ablation",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="legacy",
    ),
}
TIER1_METHODS = (
    "random_binary",
    "all_supportive",
    "no_prototype",
    "no_retrieval",
)
ALL_METHODS = tuple(CONTROL_CELLS)
METHOD_ROLES = {
    "random_binary": "count_matched_random_head_partition",
    "inverted_binary": "inverted_head_partition",
    "all_supportive": "all_heads_temporal_prototype",
    "all_suppressive": "all_heads_bounded_retrieval",
    "no_prototype": "remove_supportive_temporal_prototype",
    "no_retrieval": "remove_suppressive_bounded_retrieval",
}


def parse_method_keys(raw: str) -> tuple[str, ...]:
    requested = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("methods must be a non-empty unique list")
    unknown = sorted(set(requested) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    return tuple(key for key in ALL_METHODS if key in requested)


def parse_prompt_indices(raw: str) -> tuple[int, ...]:
    values: set[int] = set()
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        if "-" in item:
            left, separator, right = item.partition("-")
            if not separator:
                raise ValueError(f"invalid prompt interval: {item}")
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"reversed prompt interval: {item}")
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    if not values:
        raise ValueError("prompt selection is empty")
    if min(values) < 0 or max(values) >= PROMPT_COUNT:
        raise ValueError("prompt indices must be within [0, 128)")
    return tuple(sorted(values))


def _matrix_text(matrix: list[list[int]]) -> str:
    return "\n".join(",".join(str(value) for value in row) for row in matrix) + "\n"


def _write_frozen_text(path: Path, text: str) -> str:
    content = text.encode("ascii")
    digest = hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite control map: {path}")
        return digest
    temporary = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary.write_bytes(content)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _map_audit(path: Path, *, kind: str) -> dict[str, object]:
    matrix = read_matrix(path, {10, 11})
    counts = Counter(value for row in matrix for value in row)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "kind": kind,
        "shape": [len(matrix), len(matrix[0])],
        "counts": {str(key): counts[key] for key in sorted(counts)},
    }


def build_control_maps(
    output_root: Path,
    legacy_map: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    legacy = read_matrix(legacy_map, {10, 11})
    flat = [value for row in legacy for value in row]
    if Counter(flat) != Counter({10: 304, 11: 56}):
        raise ValueError("legacy map is not the frozen 304/56 partition")

    inverted = [[11 if value == 10 else 10 for value in row] for row in legacy]
    all_supportive = [[10 for _ in row] for row in legacy]
    all_suppressive = [[11 for _ in row] for row in legacy]
    indices = list(range(len(flat)))
    random.Random(RANDOM_MAP_SEED).shuffle(indices)
    suppressive = set(indices[:56])
    random_flat = [11 if index in suppressive else 10 for index in range(len(flat))]
    random_map = [
        random_flat[offset : offset + len(legacy[0])]
        for offset in range(0, len(random_flat), len(legacy[0]))
    ]

    root = output_root / "control_maps"
    generated = {
        "random": random_map,
        "inverted": inverted,
        "all_supportive": all_supportive,
        "all_suppressive": all_suppressive,
    }
    paths = {"legacy": legacy_map.resolve()}
    for key, matrix in generated.items():
        path = root / f"{key}.csv"
        _write_frozen_text(path, _matrix_text(matrix))
        paths[key] = path.resolve()
    audits = {
        key: _map_audit(path, kind=key) for key, path in paths.items()
    }
    return paths, audits


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("preflight", "generate", "audit"),
        default="generate",
    )
    parser.add_argument(
        "--scope",
        choices=("screen16", "full128", "custom"),
        default=os.environ.get("V132_SCOPE", "screen16"),
    )
    parser.add_argument(
        "--prompt-indices",
        default=os.environ.get("V132_PROMPT_INDICES", ""),
        help="comma-separated indices/ranges; requires --scope custom",
    )
    parser.add_argument(
        "--methods",
        default=os.environ.get("V132_METHODS", ",".join(TIER1_METHODS)),
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--out-root", type=Path, default=env_path("OUT_ROOT"))
    parser.add_argument("--prompts", type=Path, default=env_path("V132_PROMPTS"))
    parser.add_argument("--pf-repo", type=Path, default=env_path("PF_REPO"))
    parser.add_argument("--pf-config", type=Path, default=env_path("PF_CONFIG"))
    parser.add_argument(
        "--pf-checkpoint", type=Path, default=env_path("PF_CHECKPOINT")
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument(
        "--legacy-map", type=Path, default=env_path("LEGACY_V98_MAP")
    )
    parser.add_argument(
        "--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0"))
    )
    parser.add_argument(
        "--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "4"))
    )
    parser.add_argument(
        "--gpu-list", default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7")
    )
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        args.method_keys = parse_method_keys(args.methods)
        if args.scope == "screen16":
            if args.prompt_indices:
                parser.error("--prompt-indices requires --scope custom")
            args.selected_prompt_indices = SCREEN16
        elif args.scope == "full128":
            if args.prompt_indices:
                parser.error("--prompt-indices requires --scope custom")
            args.selected_prompt_indices = tuple(range(PROMPT_COUNT))
        else:
            args.selected_prompt_indices = parse_prompt_indices(
                args.prompt_indices
            )
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
    args.prompts = (args.prompts or Path(DEFAULT_PROMPTS)).resolve()
    args.single_prompts = args.prompts
    args.aba_prompts = (
        args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.aba_prompt_index = 0
    args.single_prompt_index = 0
    args.num_output_frames = NUM_OUTPUT_FRAMES
    args.expected_video_frames = 4 * NUM_OUTPUT_FRAMES - 3
    args.experiment_name = EXPERIMENT
    method_digest = hashlib.sha256(
        ",".join(args.method_keys).encode("ascii")
    ).hexdigest()[:12]
    args.method_set_id = f"controls{len(args.method_keys)}_{method_digest}"
    args.out_root = (
        args.out_root
        or args.repo_root / "runs" / EXPERIMENT / args.method_set_id
    ).resolve()
    return args


def validate_inputs(args: argparse.Namespace) -> list[str]:
    if args.seed != 0:
        raise SystemExit("v132 ablations require seed 0")
    if args.num_nodes <= 0 or not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("require 0 <= NODE_RANK < NUM_NODES")
    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n  " + "\n  ".join(missing))
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit(f"expected 128 prompts, found {len(prompts)}")
    legacy_audit = validate_frozen_map(args.legacy_map, args.pf_labels)
    if legacy_audit["sha256"] != EXPECTED_MAP_SHA256:
        raise SystemExit("legacy v98 map changed after validation")
    return prompts


def methods_for(keys: tuple[str, ...]) -> tuple[base.Method, ...]:
    return tuple(
        base.Method(
            key,
            "pf",
            CONTROL_CELLS[key],
            METHOD_ROLES[key],
        )
        for key in keys
    )


def tasks_for(
    methods: tuple[base.Method, ...],
    prompt_indices: tuple[int, ...],
) -> list[tuple[base.Method, int, Cell]]:
    return [
        (method, prompt_index, base.task_cell(method, prompt_index))
        for method in methods
        for prompt_index in prompt_indices
    ]


def experiment_contract(
    args: argparse.Namespace,
    *,
    prompts: list[str],
    methods: tuple[base.Method, ...],
    map_audits: dict[str, dict[str, object]],
) -> dict[str, object]:
    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        args.repo_root / "scripts" / "run_v120_moviebench32_main.py",
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "role_event.py",
        args.pf_repo / "pyramidkv" / "role_memory.py",
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "prompt_count": PROMPT_COUNT,
        "prompts": {
            "path": str(args.prompts),
            "sha256": sha256(args.prompts),
            "items": [
                {"index": index, "text": prompt}
                for index, prompt in enumerate(prompts)
            ],
        },
        "seed": 0,
        "reseed_per_prompt": True,
        "num_output_frames": NUM_OUTPUT_FRAMES,
        "decoded_video_contract": {
            "frames": args.expected_video_frames,
            "fps": 16,
            "duration_seconds": args.expected_video_frames / 16.0,
            "width": 832,
            "height": 480,
        },
        "methods": [
            {
                "key": method.key,
                "role": method.role,
                "cell": asdict(method.source_cell),
            }
            for method in methods
        ],
        "head_maps": map_audits,
        "random_map_seed": RANDOM_MAP_SEED,
        "checkpoint": {
            "path": str(args.pf_checkpoint),
            "size": args.pf_checkpoint.stat().st_size,
        },
        "implementation_hashes": {
            str(path): sha256(path) for path in implementation_paths
        },
    }


def audit_selection(
    args: argparse.Namespace,
    *,
    methods: tuple[base.Method, ...],
    contract_sha: str,
) -> dict[str, object]:
    failures: list[str] = []
    rows = []
    for method in methods:
        video_dir = args.out_root / "published" / method.key
        indexed_dir = args.out_root / "published_indexed" / method.key
        sizes = []
        for prompt_index in args.selected_prompt_indices:
            target = video_dir / base.published_name(prompt_index)
            indexed = indexed_dir / base.published_name(
                prompt_index, indexed=True
            )
            marker = (
                args.out_root
                / "status"
                / "published"
                / f"{method.key}.p{prompt_index:03d}.json"
            )
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                source = Path(payload["source"])
                if (
                    payload["experiment_contract_sha256"] != contract_sha
                    or payload["method"] != method.key
                    or int(payload["prompt_index"]) != prompt_index
                    or not source.is_file()
                    or not target.is_file()
                    or not indexed.is_file()
                    or not target.samefile(source)
                    or not indexed.samefile(source)
                ):
                    raise ValueError("publication marker mismatch")
                sizes.append(target.stat().st_size)
            except Exception as error:
                failures.append(f"{method.key}:{prompt_index}: {error}")
        rows.append(
            {
                "key": method.key,
                "role": method.role,
                "video_dir": str(video_dir),
                "indexed_video_dir": str(indexed_dir),
                "selected_video_count": len(sizes),
                "total_bytes": sum(sizes),
            }
        )
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "experiment_contract_sha256": contract_sha,
        "prompt_file": str(args.prompts),
        "prompt_file_sha256": sha256(args.prompts),
        "prompt_indices": list(args.selected_prompt_indices),
        "methods": rows,
        "failures": failures,
        "ok": not failures,
    }
    selection_sha = hashlib.sha256(
        ",".join(str(index) for index in args.selected_prompt_indices).encode(
            "ascii"
        )
    ).hexdigest()[:12]
    name = (
        "published_manifest.json"
        if len(args.selected_prompt_indices) == PROMPT_COUNT
        else f"published_manifest.selection_{selection_sha}.json"
    )
    if failures:
        write_runtime_json(args.out_root / "status" / f"{name}.failed", payload)
        raise RuntimeError(f"v132 audit failed: {failures[:5]}")
    write_frozen(args.out_root / name, payload)
    return payload


def main() -> None:
    args = parse_args()
    prompts = validate_inputs(args)
    for name in (
        "logs",
        "traces",
        "configs",
        "status",
        "status/published",
        "diagnostics",
        "contracts",
        "videos",
        "published",
        "published_indexed",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)
    args.head_maps, args.head_map_audits = build_control_maps(
        args.out_root, args.legacy_map
    )
    methods = methods_for(args.method_keys)

    base.EXPERIMENT = EXPERIMENT
    base.PROMPT_COUNT = PROMPT_COUNT
    base.TASK_STAGE = "paper_ablation"
    base.PUBLISHED_TAG = "v132"
    base.RUN_LABEL = "v132"

    contract = experiment_contract(
        args,
        prompts=prompts,
        methods=methods,
        map_audits=args.head_map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path, contract, args.contract_wait_seconds
        )

    all_tasks = tasks_for(methods, args.selected_prompt_indices)
    local_tasks = all_tasks[args.node_rank :: args.num_nodes]
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    print(
        f"[v132] mode={args.mode} scope={args.scope} "
        f"node={args.node_rank}/{args.num_nodes} methods={args.method_keys} "
        f"selected_prompts={len(args.selected_prompt_indices)} "
        f"local_tasks={len(local_tasks)} out={args.out_root}",
        flush=True,
    )
    if args.mode == "preflight":
        print(
            canonical_json(
                {
                    "contract_sha256": contract_sha,
                    "method_set_id": args.method_set_id,
                    "prompt_indices": list(args.selected_prompt_indices),
                    "head_maps": args.head_map_audits,
                }
            )
            .decode("utf-8")
            .strip()
        )
        return
    if args.mode == "audit":
        payload = audit_selection(
            args, methods=methods, contract_sha=contract_sha
        )
        print(
            f"[v132-audit] methods={len(payload['methods'])} "
            f"prompts={len(payload['prompt_indices'])}",
            flush=True,
        )
        return

    worker_tasks = [
        local_tasks[index:: len(gpus)] for index in range(len(gpus))
    ]
    worker_tasks = [items for items in worker_tasks if items]
    results: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(worker_tasks))) as executor:
        futures = {
            executor.submit(
                base.run_worker,
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
    failures.extend(
        f"{row.get('name')}: {row.get('error')}"
        for row in results
        if row.get("status") == "failed"
    )
    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "prompt_indices": list(args.selected_prompt_indices),
        "contract_sha256": contract_sha,
        "results": results,
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root
        / "status"
        / f"{args.scope}.node{args.node_rank}.summary.json"
    )
    write_runtime_json(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
