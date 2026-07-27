#!/usr/bin/env python3
"""Screen Suppressive cache policies under the frozen old-v98 304/56 map.

Every cell generates one 30-second video from the same prompt and seed. The
304 Supportive heads use an identical cyclic carrier in every cell, so the
only experimental variable is the cache assigned to the 56 Suppressive heads.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    run_cell,
    sha256,
    validate_legacy_map,
    wait_for_frozen,
    write_frozen,
)


EXPERIMENT = "v109_legacy_v98_suppressive_cache_1video"
EXPECTED_MAP_SHA256 = (
    "379711c1c9e51871b9ad96ca57e85d20723d85faf185cbf08fabd0380bcde373"
)

CELLS = (
    Cell(
        "legacy_v98_all_cyclic_control",
        "carrier",
        "single",
        support_policy="cyclic",
        suppress_policy="cyclic",
    ),
    Cell(
        "legacy_v98_suppress_cyclic_sink3",
        "cache",
        "single",
        support_policy="cyclic",
        suppress_policy="cyclic_sink3",
    ),
    Cell(
        "legacy_v98_suppress_recent8_sink1",
        "cache",
        "single",
        support_policy="cyclic",
        suppress_policy="recent8_sink1",
    ),
    Cell(
        "legacy_v98_suppress_recent5_sink3",
        "cache",
        "single",
        support_policy="cyclic",
        suppress_policy="recent5",
    ),
    Cell(
        "legacy_v98_suppress_merge",
        "cache",
        "single",
        support_policy="cyclic",
        suppress_policy="merge",
    ),
)


def cells_for_mode(mode: str) -> tuple[Cell, ...]:
    if mode == "all":
        return CELLS
    return tuple(cell for cell in CELLS if cell.stage == mode)


def selected_cells(
    mode: str,
    *,
    node_rank: int,
    num_nodes: int,
) -> tuple[Cell, ...]:
    return cells_for_mode(mode)[node_rank::num_nodes]


def validate_frozen_map(map_path: Path, pf_labels: Path) -> dict[str, object]:
    audit = validate_legacy_map(map_path, pf_labels)
    actual_hash = str(audit["sha256"])
    if actual_hash != EXPECTED_MAP_SHA256:
        raise ValueError(
            "old-v98 map hash mismatch: "
            f"expected={EXPECTED_MAP_SHA256} actual={actual_hash}"
        )
    return {
        **audit,
        "classifier": "legacy_v98_absolute_history_logit_sign",
        "support_rule": "sum(history_qk)/sum(abs(history_qk)) >= 0",
        "shift_invariant": False,
        "diagnostic_only": True,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("carrier", "cache", "all"),
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
        "--single-prompts", type=Path, default=env_path("SINGLE_PROMPTS")
    )
    parser.add_argument(
        "--single-prompt-index",
        type=int,
        default=int(os.environ.get("SINGLE_PROMPT_INDEX", "0")),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
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
    args.single_prompts = (
        args.single_prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num32.txt"
    ).resolve()
    args.aba_prompts = (
        args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.out_root = (
        args.out_root or args.repo_root / "runs" / EXPERIMENT
    ).resolve()
    args.experiment_name = EXPERIMENT
    return args


def validate_inputs(args: argparse.Namespace) -> list[str]:
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v109 screen requires seed 0")

    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.single_prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    prompts = [
        line
        for line in args.single_prompts.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not 0 <= args.single_prompt_index < len(prompts):
        raise SystemExit("--single-prompt-index is outside the prompt file")
    return prompts


def main() -> None:
    args = parse_args()
    prompts = validate_inputs(args)
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

    map_audit = validate_frozen_map(args.legacy_map, args.pf_labels)
    args.head_map_audits = {"legacy": map_audit}

    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        args.legacy_map,
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "base.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "cyclic.py",
        args.pf_repo / "pyramidkv" / "merge.py",
    )
    mode_cells = cells_for_mode(args.mode)
    contract = {
        "version": 1,
        "experiment": EXPERIMENT,
        "mode": args.mode,
        "seed": 0,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "causal_question": (
            "With old-v98 304/56 membership fixed and all 304 Supportive "
            "heads held on the same cyclic carrier, which cache is safe and "
            "useful for the 56 Suppressive heads?"
        ),
        "fixed_carrier": {
            "label": 10,
            "count": 304,
            "sink_frames": 1,
            "middle": "cyclic4",
            "recent_frames": 4,
        },
        "decision_rules": {
            "all_cyclic_noisy": (
                "stop; a common cyclic/exclusive-owner regression remains"
            ),
            "sink3_only_noisy": (
                "Suppressive failure is caused by sink contract, not middle"
            ),
            "merge_only_noisy": (
                "large-scale or mixed-membership Merge is unsafe"
            ),
            "recent5_clean_merge_noisy": (
                "prefer local recent retention over KV Merge"
            ),
            "recent8_sink1_clean": (
                "cyclic is not mandatory; local temporal support is sufficient"
            ),
        },
        "cells": [asdict(cell) for cell in mode_cells],
        "prompt": {
            "path": str(args.single_prompts),
            "sha256": sha256(args.single_prompts),
            "index": args.single_prompt_index,
            "text": prompts[args.single_prompt_index],
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
    contract_path = args.out_root / "contracts" / f"{args.mode}.json"
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            args.contract_wait_seconds,
        )

    cells = selected_cells(
        args.mode,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
    )
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if len(gpus) != len(set(gpus)) or len(gpus) < len(cells):
        raise SystemExit(
            f"node {args.node_rank} needs {len(cells)} unique GPUs; got {gpus}"
        )
    print(
        "[V109LegacyMapAudit] "
        + canonical_json(map_audit).decode("utf-8").strip(),
        flush=True,
    )
    print(
        f"[v109] mode={args.mode} node={args.node_rank}/{args.num_nodes} "
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
    write_frozen(summary_path, summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
