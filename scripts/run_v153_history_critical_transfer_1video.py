#!/usr/bin/env python3
"""Transfer the v152 one-sided QK head result into 30-second generation.

This is a membership screen, not a broad method comparison.  Every routed
cell has the same sink1 and nine-full-frame-equivalent budget.  QK-top is
compared with bottom-tail, count-matched random, and old-v98 membership before
any larger generation run is allowed.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from analyze_v152_one_sided_history_critical import (
    HEADS_PER_LAYER,
    MANIFEST_FILENAME,
    MAP_FILENAMES,
    audit_binary_map,
)
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    run_cell,
    sha256,
    validate_legacy_map,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


EXPERIMENT = "v153_history_critical_transfer_1video"
DEFAULT_SERVER_PROMPTS = (
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/"
    "Causal-Forcing/prompts/MovieGen_128_qwen.txt"
)

# Label 10 is the tested history-critical route. Label 11 is deliberately
# called default: v152 did not establish a recent-preferring second class.
CELLS = (
    Cell(
        "qk_top4_prototype4_default_recent8",
        "membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "qk_bottom4_control_prototype4_default_recent8",
        "membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="qk_bottom4_control",
    ),
    Cell(
        "qk_random4_control_prototype4_default_recent8",
        "membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="random4_control",
    ),
    Cell(
        "legacy_v98_membership_prototype4_default_recent8",
        "membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="legacy",
    ),
    Cell(
        "qk_top4_all_recent8_control",
        "controls",
        "single",
        support_policy="recent8",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "qk_top4_all_prototype4_control",
        "controls",
        "single",
        support_policy="prototype",
        suppress_policy="prototype",
        map_key="qk_top4",
    ),
    Cell(
        "legacy_v98_prototype4_retrieval1_age24_reference",
        "reference",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        map_key="legacy",
    ),
)


def cells_for_mode(mode: str) -> tuple[Cell, ...]:
    if mode == "all":
        return CELLS
    return tuple(cell for cell in CELLS if cell.stage == mode)


def selected_cells(
    mode: str, *, node_rank: int, num_nodes: int
) -> tuple[Cell, ...]:
    return cells_for_mode(mode)[node_rank::num_nodes]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("membership", "controls", "reference", "all"),
        default="all",
    )
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
        "--pf-checkpoint", type=Path, default=env_path("PF_CHECKPOINT")
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument(
        "--legacy-map", type=Path, default=env_path("LEGACY_V98_MAP")
    )
    parser.add_argument(
        "--map-dir", type=Path, default=env_path("V153_MAP_DIR")
    )
    parser.add_argument(
        "--single-prompts", type=Path, default=env_path("SINGLE_PROMPTS")
    )
    parser.add_argument(
        "--single-prompt-index",
        type=int,
        default=int(os.environ.get("SINGLE_PROMPT_INDEX", "0")),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

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
    args.map_dir = (
        args.map_dir or args.repo_root / "configs" / "head_maps"
    ).resolve()
    args.single_prompts = (
        args.single_prompts or Path(DEFAULT_SERVER_PROMPTS)
    ).resolve()
    args.aba_prompts = (
        args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.out_root = (
        args.out_root or args.repo_root / "runs" / EXPERIMENT
    ).resolve()
    args.experiment_name = EXPERIMENT
    args.num_output_frames = 120
    args.expected_video_frames = 477
    return args


def validate_inputs(args: argparse.Namespace) -> list[str]:
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v153 screen requires seed 0")
    map_paths = [args.map_dir / filename for filename in MAP_FILENAMES.values()]
    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.map_dir / MANIFEST_FILENAME,
        args.single_prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
        *map_paths,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required v153 files: {missing}")
    prompts = [
        line.strip()
        for line in args.single_prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 <= args.single_prompt_index < len(prompts):
        raise SystemExit("--single-prompt-index is outside the prompt file")
    return prompts


def load_and_audit_maps(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    manifest_path = args.map_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = manifest.get("gate_reanalysis") or {}
    if not gate.get("one_sided_transfer_candidate"):
        raise ValueError("manifest does not pass the v152 one-sided transfer gate")
    head_maps = {
        name: args.map_dir / filename for name, filename in MAP_FILENAMES.items()
    }
    head_maps["legacy"] = args.legacy_map
    audits = {}
    for name in MAP_FILENAMES:
        expected = manifest["maps"][name]
        audit = audit_binary_map(
            head_maps[name],
            args.pf_labels,
            expected_label10_per_layer=HEADS_PER_LAYER,
        )
        if audit["sha256"] != expected["sha256"]:
            raise ValueError(
                f"{name} map hash changed: {audit['sha256']} != {expected['sha256']}"
            )
        audits[name] = audit
    audits["legacy"] = validate_legacy_map(args.legacy_map, args.pf_labels)
    return manifest, head_maps, audits


def main() -> None:
    args = parse_args()
    prompts = validate_inputs(args)
    manifest, args.head_maps, args.head_map_audits = load_and_audit_maps(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "logs",
        "traces",
        "configs",
        "status",
        "diagnostics",
        "contracts",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)

    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "analyze_v152_one_sided_history_critical.py",
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        *(args.head_maps[name] for name in MAP_FILENAMES),
        args.map_dir / MANIFEST_FILENAME,
        args.legacy_map,
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "base.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "role_event.py",
        args.pf_repo / "pyramidkv" / "role_memory.py",
    )
    mode_cells = cells_for_mode(args.mode)
    contract = {
        "version": 1,
        "experiment": EXPERIMENT,
        "mode": args.mode,
        "seed": args.seed,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "hypothesis": (
            "Heads in the stable high tail of v152 QK policy margin require "
            "distributed history; the remaining heads are an unclassified "
            "default group, not a claimed recent-preferring class."
        ),
        "cache_contract": {
            "history_critical": "sink1 + TemporalPrototype4 + recent4 = 9",
            "default": "sink1 + recent8 = 9",
            "clean_kv_only": True,
            "exclusive_dynamic_owner": True,
            "original_position_sidecar": True,
            "recent_overlap_forbidden": True,
        },
        "causal_controls": {
            "membership": [
                "qk top4",
                "qk bottom4",
                "count-matched random4",
                "legacy-v98 304/56",
            ],
            "route": ["all recent8", "all TemporalPrototype4"],
            "known_reference": "legacy-v98 Prototype4/Retrieval1(age<=24)",
        },
        "profiling_gate": manifest["gate_reanalysis"],
        "profiling_recurrence": manifest["discovery_validation_recurrence"],
        "maps": args.head_map_audits,
        "cells": [asdict(cell) for cell in mode_cells],
        "prompt": {
            "path": str(args.single_prompts),
            "sha256": sha256(args.single_prompts),
            "index": args.single_prompt_index,
            "text": prompts[args.single_prompt_index],
        },
        "pf_labels": {
            "path": str(args.pf_labels),
            "sha256": sha256(args.pf_labels),
            "use": "audit cross-tab only; not used as v153 head membership",
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
    contract_path = args.out_root / "contracts" / f"{args.mode}.json"
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path, contract, args.contract_wait_seconds
        )

    print(
        "[v153-map-audit] "
        + canonical_json(args.head_map_audits).decode("utf-8").strip(),
        flush=True,
    )
    if args.preflight_only:
        print(
            f"[v153-preflight] PASS mode={args.mode} contract={contract_sha}",
            flush=True,
        )
        return

    cells = selected_cells(
        args.mode, node_rank=args.node_rank, num_nodes=args.num_nodes
    )
    gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    if len(gpus) != len(set(gpus)) or len(gpus) < len(cells):
        raise SystemExit(
            f"node {args.node_rank} needs {len(cells)} unique GPUs; got {gpus}"
        )
    print(
        f"[v153] mode={args.mode} node={args.node_rank}/{args.num_nodes} "
        f"cells={[cell.name for cell in cells]} out={args.out_root}",
        flush=True,
    )

    failures: list[str] = []
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(cells))) as executor:
        futures = {
            executor.submit(
                run_cell,
                args,
                cell=cell,
                gpu=gpus[index],
                experiment_contract_sha256=contract_sha,
            ): cell
            for index, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[done] {result}", flush=True)
            except Exception as error:
                failures.append(f"{cell.name}: {error}")
                print(f"[failed] {cell.name}: {error}", flush=True)

    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "mode": args.mode,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "results": sorted(results, key=lambda item: item["name"]),
        "failures": failures,
        "ok": not failures,
    }
    summary_path = (
        args.out_root
        / "status"
        / f"{args.mode}.node{args.node_rank}.summary.json"
    )
    write_runtime_json(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
