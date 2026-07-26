#!/usr/bin/env python3
"""Run the fail-closed one-video recovery after the v100 polygon failures.

The screen first isolates whether routing PF Wave heads through stride causes
the corruption, then evaluates two motion-event integrations without changing
the classifier. The middle-relative map is rebuilt from its frozen score
artifact and must have the documented 33/327 split; the legacy 304/56 map is
not accepted by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    run_cell,
    sha256,
    wait_for_frozen,
    write_frozen,
)
from run_v99_binary_cache_recovery_4node_32gpu import ensure_maps


EXPERIMENT = "v107_polygon_rootcause_1video"
EXPECTED_MAP_COUNTS = {
    "history_polarity_zero": {"10": 33, "11": 327},
    "pf_ar_binary_control": {"10": 172, "11": 188},
    "pf_aw_binary_control": {"10": 328, "11": 32},
}

CELLS = (
    # Root-cause controls. PF-AR keeps Wave on cyclic; PF-AW moves Wave to
    # stride. Their difference is the direct test missing from v100.
    Cell(
        "pf_ar_stride_cyclic_control",
        "rootcause",
        "single",
        map_key="pf_ar_binary_control",
        suppress_policy="cyclic",
    ),
    Cell(
        "pf_aw_stride_cyclic_control",
        "rootcause",
        "single",
        map_key="pf_aw_binary_control",
        suppress_policy="cyclic",
    ),
    Cell(
        "middle_relative_stride_cyclic_control",
        "rootcause",
        "single",
        map_key="history_polarity_zero",
        suppress_policy="cyclic",
    ),
    # Candidate cache screens at fixed middle-relative membership.
    Cell(
        "middle_relative_cyclic4_motion1",
        "candidate",
        "single",
        map_key="history_polarity_zero",
        suppress_policy="cyclic_motion1",
    ),
    Cell(
        "middle_relative_cyclic2_motion2",
        "candidate",
        "single",
        map_key="history_polarity_zero",
        suppress_policy="motion_cyclic",
    ),
    Cell(
        "middle_relative_stride_cyclic_v78",
        "candidate",
        "single",
        map_key="history_polarity_zero",
        suppress_policy="cyclic",
        transition=True,
    ),
    # Scene memory is screened only after the same single-prompt cyclic base.
    Cell(
        "aba_middle_relative_no_episode",
        "aba",
        "aba",
        map_key="history_polarity_zero",
        suppress_policy="cyclic",
    ),
    Cell(
        "aba_middle_relative_episode_bridge1",
        "aba",
        "aba",
        map_key="history_polarity_zero",
        suppress_policy="cyclic",
        scene_cache=True,
        scene_bridge=1,
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


def validate_recovery_maps(
    manifest_path: Path,
    maps: dict[str, dict[str, object]],
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claims = manifest.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("middle-relative map manifest has no claims object")
    required_claims = {
        "primary_classifier": "history_polarity_zero",
        "pf_labels_used_for_primary_classifier": False,
        "common_logit_shift_invariant": True,
        "sink_recent_excluded_from_middle_score": True,
        "probe_policy_balanced": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": claims.get(key)}
        for key, expected in required_claims.items()
        if claims.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"middle-relative map provenance mismatch: {mismatches}"
        )

    audited: dict[str, object] = {}
    for key, expected in EXPECTED_MAP_COUNTS.items():
        item = maps.get(key)
        if item is None:
            raise ValueError(f"map manifest is missing {key}")
        actual = {
            str(label): int(item["label_counts"][str(label)])
            for label in (10, 11)
        }
        if actual != expected:
            raise ValueError(
                f"{key}: expected frozen counts {expected}, got {actual}. "
                "Refusing a likely legacy or mixed map artifact."
            )
        audited[key] = {
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
            "label_counts": actual,
            "pf_cross_tab": item["pf_cross_tab"],
        }
    return {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
            "method": manifest.get("method"),
            "score_csv_sha256": manifest.get("score_csv_sha256"),
            "score_artifact_sha256": manifest.get(
                "score_artifact_sha256"
            ),
            "claims": required_claims,
        },
        "maps": audited,
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
        choices=("rootcause", "candidate", "aba", "all"),
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
        default=int(os.environ.get("NUM_NODES", "4")),
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
    parser.add_argument(
        "--score-root", type=Path, default=env_path("SCORE_ROOT")
    )
    parser.add_argument("--pf-repo", type=Path, default=env_path("PF_REPO"))
    parser.add_argument("--pf-config", type=Path, default=env_path("PF_CONFIG"))
    parser.add_argument(
        "--pf-checkpoint", type=Path, default=env_path("PF_CHECKPOINT")
    )
    parser.add_argument("--pf-labels", type=Path, default=env_path("PF_LABELS"))
    parser.add_argument(
        "--single-prompts", type=Path, default=env_path("SINGLE_PROMPTS")
    )
    parser.add_argument(
        "--aba-prompts", type=Path, default=env_path("ABA_PROMPTS")
    )
    parser.add_argument(
        "--single-prompt-index",
        type=int,
        default=int(os.environ.get("SINGLE_PROMPT_INDEX", "0")),
    )
    parser.add_argument(
        "--aba-prompt-index",
        type=int,
        default=int(os.environ.get("ABA_PROMPT_INDEX", "0")),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-wait-seconds", type=int, default=1800)
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
    args.single_prompts = (
        args.single_prompts
        or args.pf_repo / "prompts" / "MovieGenVideoBench_num32.txt"
    ).resolve()
    args.aba_prompts = (
        args.aba_prompts
        or args.repo_root / "prompts" / "paper_scene_switch_sf_3.txt"
    ).resolve()
    args.out_root = (
        args.out_root or args.repo_root / "runs" / EXPERIMENT
    ).resolve()
    args.score_root = (
        args.score_root
        or args.repo_root / "runs" / "v98_middle_relative_scores"
    ).resolve()
    args.experiment_name = EXPERIMENT
    return args


def validate_inputs(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v107 screen requires seed 0")
    required = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.single_prompts,
        args.aba_prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
        args.repo_root / "scripts" / "build_v98_history_polarity_maps.py",
        args.score_root / "scores" / "qk_head_scores.csv",
        args.score_root / "scores" / "qk_head_score_artifact.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    single_lines = [
        line
        for line in args.single_prompts.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    aba_lines = [
        line
        for line in args.aba_prompts.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not 0 <= args.single_prompt_index < len(single_lines):
        raise SystemExit("--single-prompt-index is outside the prompt file")
    if not 0 <= args.aba_prompt_index < len(aba_lines):
        raise SystemExit("--aba-prompt-index is outside the ABA prompt file")
    segments = [
        part
        for part in aba_lines[args.aba_prompt_index].split("||")
        if part.strip()
    ]
    if len(segments) != 3:
        raise SystemExit("selected ABA prompt must contain exactly 3 segments")
    return single_lines, aba_lines


def main() -> None:
    args = parse_args()
    single_lines, aba_lines = validate_inputs(args)
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

    manifest_path, maps = ensure_maps(args)
    map_audit = validate_recovery_maps(manifest_path, maps)
    args.head_maps = {
        key: Path(str(maps[key]["path"])).resolve()
        for key in EXPECTED_MAP_COUNTS
    }
    args.head_map_audits = map_audit["maps"]
    # Compatibility fallback for shared v100 execution helpers. Every v107
    # non-native cell resolves through the explicit map_key dictionary.
    args.legacy_map = args.head_maps["history_polarity_zero"]

    mode_cells = cells_for_mode(args.mode)
    implementation_paths = (
        Path(__file__).resolve(),
        args.repo_root / "scripts" / "run_v100_fast_selection_1video.py",
        args.repo_root
        / "scripts"
        / "run_v99_binary_cache_recovery_4node_32gpu.py",
        args.repo_root / "scripts" / "build_v98_history_polarity_maps.py",
        args.pf_repo / "inference.py",
        args.pf_repo / "pipeline" / "causal_inference.py",
        args.pf_repo / "pipeline" / "pyramidkv_config.py",
        args.pf_repo / "pyramidkv" / "adaptive_cache.py",
        args.pf_repo / "pyramidkv" / "base.py",
        args.pf_repo / "pyramidkv" / "factory.py",
        args.pf_repo / "pyramidkv" / "motion_event.py",
        args.pf_repo / "pyramidkv" / "policy_overrides.py",
        args.pf_repo / "pyramidkv" / "cyclic.py",
        args.pf_repo / "pyramidkv" / "stride.py",
    )
    contract = {
        "version": 1,
        "experiment": EXPERIMENT,
        "mode": args.mode,
        "seed": 0,
        "num_output_frames": 120,
        "causal_question": (
            "Does moving PF Wave heads from cyclic to stride cause the "
            "polygon corruption observed with the legacy 304/56 map?"
        ),
        "decision_rule": {
            "pf_ar_clean_pf_aw_noisy": "Wave-to-stride routing is causal",
            "pf_ar_clean_pf_aw_clean": (
                "the legacy 304/56 membership or its interactions are causal"
            ),
            "pf_ar_noisy": (
                "a common current-code binary-path regression remains"
            ),
        },
        "cells": [asdict(cell) for cell in mode_cells],
        "single_prompt": {
            "path": str(args.single_prompts),
            "sha256": sha256(args.single_prompts),
            "index": args.single_prompt_index,
            "text": single_lines[args.single_prompt_index],
        },
        "aba_prompt": {
            "path": str(args.aba_prompts),
            "sha256": sha256(args.aba_prompts),
            "index": args.aba_prompt_index,
            "text": aba_lines[args.aba_prompt_index],
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
        "[V107MapAudit] "
        + canonical_json(map_audit).decode("utf-8").strip(),
        flush=True,
    )
    print(
        f"[v107] mode={args.mode} node={args.node_rank}/{args.num_nodes} "
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
