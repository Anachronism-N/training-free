#!/usr/bin/env python3
"""Run the v100 one-video mechanism selection before broad ablations.

The fast screen separates three decisions:
1. Responsive-head middle-cache policy.
2. Small, previously motivated add-ons.
3. A-B-A role-aware scene archive/recall.

Every cell generates exactly one 30-second video and writes auditable policy,
motion-event, transition, and scene-switch traces as applicable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRACE_LAYERS = (0, 7, 15, 23, 29)
EXPECTED_VIDEO_FRAMES = 477
EXPECTED_VIDEO_FPS = 16.0
EXPECTED_VIDEO_WIDTH = 832
EXPECTED_VIDEO_HEIGHT = 480
LEGACY_COUNTS = {10: 304, 11: 56}
LEGACY_PF_CROSS_TAB = {
    "wave": {10: 133, 11: 23},
    "anchor": {10: 169, 11: 3},
    "veil": {10: 2, 11: 30},
}
MOTION_SIGNATURE_POLICY_MODES = {
    "reservoir2_multiscaledir1": "multiscale_direction",
    "reservoir2_multiscalemotion1": "multiscale_magnitude",
    "reservoir2_multiscalepareto1": "pareto_multiscale_magnitude",
    "reservoir2_multiscaleconsensus1": (
        "consensus_multiscale_magnitude"
    ),
    "reservoir2_multiscalequeryweighted1": (
        "query_weighted_multiscale_magnitude"
    ),
    "reservoir2_multiscalebottleneck1": (
        "bottleneck_multiscale_magnitude"
    ),
    "reservoir2_staterankmotion1": (
        "state_ranked_multiscale_magnitude"
    ),
    "reservoir2_deficitstaterankmotion1": (
        "deficit_state_ranked_multiscale_magnitude"
    ),
}

RESERVOIR_MOTION_POLICIES = {
    "reservoir2_motion1",
    "reservoir2_freshmotion1",
    "reservoir2_statemotion1",
    "reservoir2_freshmotion4",
    "reservoir2_statemotion1_strict",
    "reservoir2_stateage12motion1",
    "reservoir2_statebalancedmotion1",
    "reservoir2_directionmatch1",
    "reservoir2_directionfresh1",
    "reservoir2_dirstaletie003",
    "reservoir2_dirstaletie005",
    *MOTION_SIGNATURE_POLICY_MODES,
}
FRESH_MOTION_POLICIES = {
    "reservoir2_freshmotion1",
    "reservoir2_statemotion1",
    "reservoir2_freshmotion4",
    "reservoir2_statemotion1_strict",
    "reservoir2_stateage12motion1",
    "reservoir2_statebalancedmotion1",
    "reservoir2_directionmatch1",
    "reservoir2_directionfresh1",
    "reservoir2_dirstaletie003",
    "reservoir2_dirstaletie005",
    *MOTION_SIGNATURE_POLICY_MODES,
}
STATE_MATCH_POLICIES = {
    "reservoir2_statemotion1",
    "reservoir2_freshmotion4",
    "reservoir2_statemotion1_strict",
    "reservoir2_stateage12motion1",
    "reservoir2_statebalancedmotion1",
    "reservoir2_directionmatch1",
    "reservoir2_directionfresh1",
    "reservoir2_dirstaletie003",
    "reservoir2_dirstaletie005",
    *MOTION_SIGNATURE_POLICY_MODES,
}
DIRECTION_ONLY_POLICIES = {
    "reservoir2_directionmatch1",
    "reservoir2_directionfresh1",
    "reservoir2_dirstaletie003",
    "reservoir2_dirstaletie005",
    *MOTION_SIGNATURE_POLICY_MODES,
}
DIRECTION_STALE_TIE_MARGINS = {
    "reservoir2_dirstaletie003": 0.03,
    "reservoir2_dirstaletie005": 0.05,
}


def expected_state_match_contract(policy: str) -> dict[str, Any] | None:
    if policy not in STATE_MATCH_POLICIES:
        return None
    direction_only = policy in DIRECTION_ONLY_POLICIES
    return {
        "state_archive_capacity": 4,
        "state_max_read_age": (
            12 if policy == "reservoir2_stateage12motion1" else 24
        ),
        "state_min_similarity": (
            -1.0
            if policy
            in {
                "reservoir2_freshmotion4",
                "reservoir2_directionmatch1",
                "reservoir2_directionfresh1",
                *DIRECTION_STALE_TIE_MARGINS,
                *MOTION_SIGNATURE_POLICY_MODES,
            }
            else 0.0
            if policy == "reservoir2_statemotion1_strict"
            else -0.25
        ),
        "state_min_direction_similarity": (
            -1.0
            if policy == "reservoir2_freshmotion4"
            else 0.1
            if policy
            in {
                "reservoir2_statemotion1_strict",
                "reservoir2_directionmatch1",
                "reservoir2_directionfresh1",
                *DIRECTION_STALE_TIE_MARGINS,
                *MOTION_SIGNATURE_POLICY_MODES,
            }
            else 0.0
        ),
        "state_selection_order": (
            ["recency"]
            if policy == "reservoir2_freshmotion4"
            else ["direction_similarity", "recency"]
            if direction_only
            else ["direction_similarity", "state_similarity", "recency"]
        ),
        "state_recency_weight": (
            0.25
            if policy
            in {
                "reservoir2_statebalancedmotion1",
                "reservoir2_directionfresh1",
            }
            else 0.0
        ),
        "state_similarity_weight": 0.0 if direction_only else 0.5,
        "state_fallback_to_newest": direction_only,
        "state_direction_tie_margin": DIRECTION_STALE_TIE_MARGINS.get(
            policy,
            0.0,
        ),
        "state_stale_tie_age": (
            12 if policy in DIRECTION_STALE_TIE_MARGINS else 0
        ),
        "state_motion_signature_mode": MOTION_SIGNATURE_POLICY_MODES.get(
            policy,
            "none",
        ),
    }


@dataclass(frozen=True)
class Cell:
    name: str
    stage: str
    prompt_kind: str
    support_policy: str = "stride"
    suppress_policy: str | None = "motion_cyclic"
    transition: bool = False
    scene_cache: bool = False
    scene_bridge: int = 1
    scene_manual: bool = False
    motion_top_k: int = 1
    variance_refresh: bool = False
    map_key: str = "legacy"
    history_budget_profile: str = "default"
    max_full_frame_equivalents: int = 9
    retrieval_abstain: bool = False
    retrieval_min_similarity: float = -0.25
    retrieval_min_margin: float = 0.0

    @property
    def native(self) -> bool:
        return self.suppress_policy is None

    @property
    def uses_motion(self) -> bool:
        return self.suppress_policy in {
            "motion",
            "motion_cyclic",
            "cyclic_motion1",
        }

    @property
    def uses_role_event(self) -> bool:
        policies = {
            str(self.support_policy),
            str(self.suppress_policy),
        }
        return bool(
            policies
            & {
                "landmark",
                "motion_pair",
                "motion_pair1",
                "landmark_motion",
                "retrieval",
                "retrieval2",
                "retrieval1",
                "retrieval1_age24",
                "retrieval1_motion1_age24",
                "prototype",
                "prototype2",
                "reservoir2_motion1",
                "reservoir2_freshmotion1",
                "reservoir2_statemotion1",
                "reservoir2_freshmotion4",
                "reservoir2_statemotion1_strict",
                "reservoir2_stateage12motion1",
                "reservoir2_statebalancedmotion1",
                "reservoir2_directionmatch1",
                "reservoir2_directionfresh1",
                "profile_anchor",
                "snapshot",
                "snapshot2",
                "sparse75",
            }
        )


CELLS = (
    # 1. Responsive cache selection.
    Cell("single_pf_native", "responsive", "single", suppress_policy=None),
    Cell(
        "legacy_v98_stride_cyclic_sink1",
        "responsive",
        "single",
        suppress_policy="cyclic",
    ),
    Cell(
        "legacy_v98_stride_cyclic_sink3",
        "responsive",
        "single",
        suppress_policy="cyclic_sink3",
    ),
    Cell(
        "legacy_v98_stride_motion4",
        "responsive",
        "single",
        suppress_policy="motion",
    ),
    Cell(
        "legacy_v98_stride_motion2_cyclic2",
        "responsive",
        "single",
        suppress_policy="motion_cyclic",
    ),
    Cell(
        "legacy_v98_stride_recent8",
        "responsive",
        "single",
        suppress_policy="recent8",
    ),
    # 2. Small add-ons. Only v78 is already validated; the others are screens.
    Cell(
        "legacy_v98_motion2_cyclic2_v78",
        "tricks",
        "single",
        transition=True,
    ),
    Cell(
        "legacy_v98_hybrid_motion2_cyclic2",
        "tricks",
        "single",
        support_policy="hybrid",
    ),
    Cell(
        "legacy_v98_hybrid_motion2_cyclic2_v78",
        "tricks",
        "single",
        support_policy="hybrid",
        transition=True,
    ),
    Cell(
        "legacy_v98_motion2_cyclic2_variance_refresh",
        "tricks",
        "single",
        variance_refresh=True,
    ),
    # 3. A-B-A switch/return. Manual matching is diagnostic, not a final claim.
    Cell("aba_pf_native", "aba", "aba", suppress_policy=None),
    Cell(
        "aba_motion_no_episode",
        "aba",
        "aba",
    ),
    Cell(
        "aba_motion_episode_bridge1",
        "aba",
        "aba",
        scene_cache=True,
        scene_bridge=1,
    ),
    Cell(
        "aba_motion_episode_hard",
        "aba",
        "aba",
        scene_cache=True,
        scene_bridge=0,
    ),
    Cell(
        "aba_motion_episode_manual_bridge1",
        "aba",
        "aba",
        scene_cache=True,
        scene_bridge=1,
        scene_manual=True,
    ),
    Cell(
        "aba_cyclic_sink3_episode_bridge1",
        "aba",
        "aba",
        suppress_policy="cyclic_sink3",
        scene_cache=True,
        scene_bridge=1,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
    content = canonical_json(payload)
    digest = hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise RuntimeError(f"refusing to overwrite mixed contract: {path}")
    else:
        temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temp.write_bytes(content)
        try:
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()
    return digest


def write_runtime_json(path: Path, payload: Any) -> None:
    """Atomically replace mutable status output without weakening contracts."""

    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_bytes(content)
    try:
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def wait_for_frozen(path: Path, payload: Any, timeout_seconds: int) -> str:
    expected = hashlib.sha256(canonical_json(payload)).hexdigest()
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(2)
    if not path.is_file():
        raise TimeoutError(f"timed out waiting for {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"mixed frozen artifact: {path} expected={expected} actual={actual}"
        )
    return actual


def read_matrix(path: Path, allowed: set[int]) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError(f"{path}: expected a complete 30x12 matrix")
    observed = {value for row in rows for value in row}
    if not observed.issubset(allowed):
        raise ValueError(
            f"{path}: labels {sorted(observed)} are not within {sorted(allowed)}"
        )
    return rows


def validate_legacy_map(
    legacy_path: Path,
    pf_path: Path,
) -> dict[str, Any]:
    legacy = read_matrix(legacy_path, {10, 11})
    pf = read_matrix(pf_path, {-1, 1, 2})
    counts = Counter(value for row in legacy for value in row)
    if dict(counts) != LEGACY_COUNTS:
        raise ValueError(
            f"legacy v98 counts changed: {dict(counts)} != {LEGACY_COUNTS}"
        )
    pf_names = {-1: "wave", 1: "anchor", 2: "veil"}
    cross_tab: dict[str, dict[int, int]] = {}
    for pf_label, name in pf_names.items():
        role_counts = Counter(
            legacy[layer][head]
            for layer in range(30)
            for head in range(12)
            if pf[layer][head] == pf_label
        )
        cross_tab[name] = {
            10: int(role_counts[10]),
            11: int(role_counts[11]),
        }
    if cross_tab != LEGACY_PF_CROSS_TAB:
        raise ValueError(
            f"legacy v98/PF cross-tab changed: {cross_tab} "
            f"!= {LEGACY_PF_CROSS_TAB}"
        )
    return {
        "path": str(legacy_path),
        "sha256": sha256(legacy_path),
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "pf_cross_tab": {
            name: {str(key): value for key, value in sorted(values.items())}
            for name, values in cross_tab.items()
        },
    }


def selected_cells(
    mode: str,
    *,
    node_rank: int,
    num_nodes: int,
) -> tuple[Cell, ...]:
    candidates = CELLS if mode == "all" else tuple(
        cell for cell in CELLS if cell.stage == mode
    )
    return tuple(candidates[node_rank::num_nodes])


def resolve_head_map(args: argparse.Namespace, cell: Cell) -> Path:
    if cell.native:
        return args.pf_labels
    head_maps = getattr(args, "head_maps", None)
    if head_maps is not None:
        try:
            return Path(head_maps[cell.map_key])
        except KeyError as error:
            raise ValueError(
                f"{cell.name}: unknown head-map key {cell.map_key!r}"
            ) from error
    if cell.map_key != "legacy":
        raise ValueError(
            f"{cell.name}: map {cell.map_key!r} requires args.head_maps"
        )
    return args.legacy_map


def run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    if log_path is None:
        completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write(
                "[command] "
                + subprocess.list2cmdline([str(value) for value in command])
                + "\n"
            )
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{subprocess.list2cmdline(command)}"
        )


def audit_video(
    args: argparse.Namespace,
    *,
    cell: Cell,
    output: Path,
    report: Path,
    log: Path,
) -> dict[str, Any]:
    prompt_index = (
        args.single_prompt_index
        if cell.prompt_kind == "single"
        else args.aba_prompt_index
    )
    command = [
        sys.executable,
        str(args.repo_root / "scripts" / "audit_indexed_videos.py"),
        "--video-dir",
        str(output),
        "--start-idx",
        str(prompt_index),
        "--end-idx",
        str(prompt_index + 1),
        "--expected-frames",
        str(
            int(
                getattr(
                    args,
                    "expected_video_frames",
                    EXPECTED_VIDEO_FRAMES,
                )
            )
        ),
        "--expected-fps",
        str(EXPECTED_VIDEO_FPS),
        "--expected-width",
        str(EXPECTED_VIDEO_WIDTH),
        "--expected-height",
        str(EXPECTED_VIDEO_HEIGHT),
        "--fps-tolerance",
        "0.05",
        "--allow-outside-interval",
        "--output-json",
        str(report),
    ]
    run_checked(command, log_path=log)
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"video audit failed: {report}")
    return payload


def expected_policy(
    cell: Cell,
    label: int,
) -> tuple[tuple[str, ...], int, int, str]:
    result: tuple[tuple[str, ...], int, int, str]
    if label == 10:
        if cell.support_policy == "profile_anchor":
            result = (
                ("TemporalProfileAnchorStrategy",),
                0,
                4,
                "temporal_profile_anchor",
            )
        elif cell.support_policy == "recent8_exact":
            result = ((), 0, 8, "stride")
        elif cell.support_policy == "recent8":
            result = ((), 1, 8, "stride")
        elif cell.support_policy == "landmark":
            result = (
                ("SemanticLandmarkStrategy",),
                1,
                4,
                "semantic_landmark",
            )
        elif cell.support_policy == "motion_pair":
            result = (("CoherentMotionStrategy",), 1, 4, "coherent_motion")
        elif cell.support_policy == "motion_pair1":
            result = (("CoherentMotionStrategy",), 1, 6, "coherent_motion")
        elif cell.support_policy == "landmark_motion":
            result = (
                ("SemanticLandmarkStrategy", "CoherentMotionStrategy"),
                1,
                4,
                "landmark_motion",
            )
        elif cell.support_policy in {"retrieval", "retrieval2"}:
            result = (
                ("SemanticRetrievalStrategy",),
                1,
                6 if cell.support_policy == "retrieval2" else 4,
                "semantic_retrieval",
            )
        elif cell.support_policy in {"retrieval1", "retrieval1_age24"}:
            result = (
                ("SemanticRetrievalStrategy",),
                1,
                7,
                "semantic_retrieval",
            )
        elif cell.support_policy == "retrieval1_motion1_age24":
            result = (
                ("CoherentMotionStrategy", "SemanticRetrievalStrategy"),
                1,
                5,
                "retrieval_motion",
            )
        elif cell.support_policy in {"prototype", "prototype2"}:
            result = (
                ("TemporalPrototypeStrategy",),
                1,
                6 if cell.support_policy == "prototype2" else 4,
                "temporal_prototype",
            )
        elif cell.support_policy == "reservoir":
            result = (
                ("TemporalReservoirStrategy",),
                1,
                4,
                "temporal_reservoir",
            )
        elif cell.support_policy in RESERVOIR_MOTION_POLICIES:
            result = (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            )
        elif cell.support_policy in {"snapshot", "snapshot2"}:
            result = (
                ("UniqueSnapshotStrategy",),
                1,
                6 if cell.support_policy == "snapshot2" else 4,
                "unique_snapshot",
            )
        elif cell.support_policy == "sparse75":
            result = (("SparseSnapshotStrategy",), 1, 5, "sparse_snapshot")
        elif cell.support_policy == "cyclic":
            result = (("CyclicStrategy",), 1, 4, "osc")
        elif cell.support_policy == "hybrid":
            result = (
                ("CyclicStrategy", "StrideStrategy"),
                3,
                4,
                "stride",
            )
        else:
            result = (("StrideStrategy",), 3, 4, "stride")
    else:
        if label != 11 or cell.suppress_policy is None:
            raise ValueError(
                f"no binary policy for label={label} cell={cell.name}"
            )
        result = {
            "merge": (("MergeStrategy",), 3, 4, "merge"),
            "cyclic": (("CyclicStrategy",), 1, 4, "osc"),
            "cyclic_sink3": (("CyclicStrategy",), 3, 4, "osc"),
            "motion": (("MotionEventStrategy",), 3, 4, "motion_event"),
            "motion_cyclic": (
                ("CyclicStrategy", "MotionEventStrategy"),
                3,
                4,
                "motion_cyclic",
            ),
            "cyclic_motion1": (
                ("CyclicStrategy", "MotionEventStrategy"),
                1,
                4,
                "motion_cyclic",
            ),
            "recent": ((), 3, 4, "stride"),
            "recent5": ((), 3, 5, "stride"),
            "recent8": ((), 3, 8, "stride"),
            "recent8_sink1": ((), 1, 8, "stride"),
            "landmark": (
                ("SemanticLandmarkStrategy",),
                1,
                4,
                "semantic_landmark",
            ),
            "motion_pair": (
                ("CoherentMotionStrategy",),
                1,
                4,
                "coherent_motion",
            ),
            "motion_pair1": (
                ("CoherentMotionStrategy",),
                1,
                6,
                "coherent_motion",
            ),
            "landmark_motion": (
                ("SemanticLandmarkStrategy", "CoherentMotionStrategy"),
                1,
                4,
                "landmark_motion",
            ),
            "retrieval": (
                ("SemanticRetrievalStrategy",),
                1,
                4,
                "semantic_retrieval",
            ),
            "retrieval2": (
                ("SemanticRetrievalStrategy",),
                1,
                6,
                "semantic_retrieval",
            ),
            "retrieval1": (
                ("SemanticRetrievalStrategy",),
                1,
                7,
                "semantic_retrieval",
            ),
            "retrieval1_age24": (
                ("SemanticRetrievalStrategy",),
                1,
                7,
                "semantic_retrieval",
            ),
            "retrieval1_motion1_age24": (
                ("CoherentMotionStrategy", "SemanticRetrievalStrategy"),
                1,
                5,
                "retrieval_motion",
            ),
            "prototype": (
                ("TemporalPrototypeStrategy",),
                1,
                4,
                "temporal_prototype",
            ),
            "prototype2": (
                ("TemporalPrototypeStrategy",),
                1,
                6,
                "temporal_prototype",
            ),
            "reservoir": (
                ("TemporalReservoirStrategy",),
                1,
                4,
                "temporal_reservoir",
            ),
            "reservoir2_motion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_freshmotion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_statemotion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_freshmotion4": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_statemotion1_strict": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_stateage12motion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_statebalancedmotion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_directionmatch1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_directionfresh1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_dirstaletie003": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_dirstaletie005": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_multiscaledir1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_multiscalemotion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_staterankmotion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_deficitstaterankmotion1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_multiscalepareto1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_multiscaleconsensus1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_multiscalequeryweighted1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "reservoir2_multiscalebottleneck1": (
                ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
                1,
                4,
                "reservoir_motion",
            ),
            "profile_anchor": (
                ("TemporalProfileAnchorStrategy",),
                0,
                4,
                "temporal_profile_anchor",
            ),
            "recent8_exact": ((), 0, 8, "stride"),
            "snapshot": (
                ("UniqueSnapshotStrategy",),
                1,
                4,
                "unique_snapshot",
            ),
            "snapshot2": (
                ("UniqueSnapshotStrategy",),
                1,
                6,
                "unique_snapshot",
            ),
            "sparse75": (
                ("SparseSnapshotStrategy",),
                1,
                5,
                "sparse_snapshot",
            ),
        }[cell.suppress_policy]

    if cell.history_budget_profile == "sink3_extra":
        result = (result[0], 3, result[2], result[3])
    elif cell.history_budget_profile == "sink3_budget9":
        if (
            cell.support_policy != "landmark"
            or cell.suppress_policy != "motion_pair1"
        ):
            raise ValueError(
                "sink3_budget9 requires landmark/motion_pair1"
            )
        result = (result[0], 3, 4, result[3])
    elif cell.history_budget_profile == "profile_exact8":
        if cell.support_policy not in {"profile_anchor", "recent8_exact"} or (
            cell.suppress_policy not in {"profile_anchor", "recent8_exact"}
        ):
            raise ValueError(
                "profile_exact8 requires profile_anchor/recent8_exact"
            )
        result = (
            result[0],
            0,
            4
            if (label == 10 and cell.support_policy == "profile_anchor")
            or (label == 11 and cell.suppress_policy == "profile_anchor")
            else 8,
            result[3],
        )
    elif cell.history_budget_profile != "default":
        raise ValueError(
            f"unknown history budget profile {cell.history_budget_profile!r}"
        )
    return result


def audit_policy_trace(
    path: Path,
    *,
    cell: Cell,
    head_map: Path,
    report_path: Path,
) -> dict[str, Any]:
    labels = read_matrix(
        head_map,
        {-1, 1, 2} if cell.native else {10, 11},
    )
    failures: list[str] = []
    records = 0
    observed: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: {error}")
                continue
            if event.get("event") != "middle_selection":
                failures.append(
                    f"line {line_number}: unexpected event {event.get('event')}"
                )
                continue
            records += 1
            try:
                layer = int(event["layer"])
                head = int(event["head"])
                label = int(event["label"])
                strategies = tuple(
                    str(item["name"]) for item in event["strategies"]
                )
                observed.add((layer, head))
                if label != labels[layer][head]:
                    failures.append(
                        f"line {line_number}: label {label} != map "
                        f"{labels[layer][head]}"
                    )
                if cell.native:
                    continue
                actual = (
                    strategies,
                    int(event["sink_frames"]),
                    int(event["recent_frames"]),
                    str(event["policy_type"]),
                )
                expected = expected_policy(cell, label)
                if actual != expected:
                    failures.append(
                        f"line {line_number}: policy {actual} != {expected}"
                    )
                if not bool(event["explicit_composition_owns_dynamic"]):
                    failures.append(
                        f"line {line_number}: dynamic owner is not exclusive"
                    )
                if not bool(event["cache_contract_pass"]):
                    failures.append(
                        f"line {line_number}: cache contract failed "
                        f"{event['cache_contract_violations']}"
                    )
                if "TemporalReservoirStrategy" in strategies:
                    union_count = int(event["union_frame_count"])
                    if union_count > 4:
                        failures.append(
                            f"line {line_number}: reservoir middle has "
                            f"{union_count} frames, expected at most 4"
                        )
                    frame_seqlen = int(event.get("frame_seqlen", 0))
                    read_tokens = sum(
                        int(event.get(field, 0))
                        for field in (
                            "sink_token_count",
                            "union_token_count",
                            "recent_token_count",
                        )
                    )
                    if (
                        frame_seqlen <= 0
                        or read_tokens
                        > cell.max_full_frame_equivalents * frame_seqlen
                    ):
                        failures.append(
                            f"line {line_number}: reservoir read has "
                            f"{read_tokens} tokens at frame_seqlen="
                            f"{frame_seqlen}, expected at most "
                            f"{cell.max_full_frame_equivalents} FFE"
                        )
                    item = next(
                        row
                        for row in event["strategies"]
                        if row["name"] == "TemporalReservoirStrategy"
                    )
                    state = item.get("state")
                    if not isinstance(state, dict):
                        failures.append(
                            f"line {line_number}: reservoir state missing"
                        )
                    else:
                        anchors = [
                            int(value)
                            for value in state.get("anchor_frame_ids", [])
                        ]
                        pending = [
                            int(value)
                            for value in state.get("pending_frame_ids", [])
                        ]
                        active_policy = (
                            cell.support_policy
                            if label == 10
                            else cell.suppress_policy
                        )
                        expected_capacity = (
                            2
                            if active_policy in RESERVOIR_MOTION_POLICIES
                            else 4
                        )
                        if int(state.get("capacity", -1)) != expected_capacity:
                            failures.append(
                                f"line {line_number}: reservoir capacity "
                                f"{state.get('capacity')} != "
                                f"{expected_capacity}"
                            )
                        if (
                            len(anchors) > expected_capacity
                            or len(anchors) != len(set(anchors))
                        ):
                            failures.append(
                                f"line {line_number}: invalid reservoir "
                                f"anchors {anchors}"
                            )
                        if len(pending) > int(state.get("defer_frames", 4)):
                            failures.append(
                                f"line {line_number}: reservoir pending "
                                f"overflow {pending}"
                            )
                        if anchors != sorted(anchors):
                            failures.append(
                                f"line {line_number}: reservoir anchors "
                                "are not sorted"
                            )
                if "TemporalProfileAnchorStrategy" in strategies:
                    union_count = int(event["union_frame_count"])
                    if union_count > 4:
                        failures.append(
                            f"line {line_number}: profile-anchor middle has "
                            f"{union_count} frames, expected at most 4"
                        )
                    frame_seqlen = int(event.get("frame_seqlen", 0))
                    read_tokens = sum(
                        int(event.get(field, 0))
                        for field in (
                            "sink_token_count",
                            "union_token_count",
                            "recent_token_count",
                        )
                    )
                    if (
                        frame_seqlen <= 0
                        or read_tokens
                        > cell.max_full_frame_equivalents * frame_seqlen
                    ):
                        failures.append(
                            f"line {line_number}: profile-anchor read has "
                            f"{read_tokens} tokens at frame_seqlen="
                            f"{frame_seqlen}, expected at most "
                            f"{cell.max_full_frame_equivalents} FFE"
                        )
                    item = next(
                        row
                        for row in event["strategies"]
                        if row["name"] == "TemporalProfileAnchorStrategy"
                    )
                    state = item.get("state")
                    if not isinstance(state, dict):
                        failures.append(
                            f"line {line_number}: profile-anchor state missing"
                        )
                    else:
                        targets = [
                            int(value)
                            for value in state.get("target_frame_ids", [])
                        ]
                        anchors = [
                            int(value)
                            for value in state.get("anchor_frame_ids", [])
                        ]
                        if targets != [0, 37, 75, 112]:
                            failures.append(
                                f"line {line_number}: profile targets "
                                f"{targets} != [0, 37, 75, 112]"
                            )
                        if (
                            len(anchors) > 4
                            or len(anchors) != len(set(anchors))
                            or not set(anchors).issubset(targets)
                            or int(state.get("physical_frame_count", -1))
                            != len(anchors)
                        ):
                            failures.append(
                                f"line {line_number}: invalid profile "
                                f"anchors {anchors}"
                            )
                if cell.uses_role_event:
                    union_count = int(event["union_frame_count"])
                    if union_count > 4:
                        failures.append(
                            f"line {line_number}: role-event middle has "
                            f"{union_count} frames, expected at most 4"
                        )
                    token_fields = {
                        "frame_seqlen",
                        "sink_token_count",
                        "union_token_count",
                        "recent_token_count",
                    }
                    if token_fields.issubset(event):
                        frame_seqlen = int(event["frame_seqlen"])
                        actual_tokens = (
                            int(event["sink_token_count"])
                            + int(event["union_token_count"])
                            + int(event["recent_token_count"])
                        )
                        if (
                            frame_seqlen <= 0
                            or actual_tokens
                            > cell.max_full_frame_equivalents * frame_seqlen
                        ):
                            failures.append(
                                f"line {line_number}: role-event read has "
                                f"{actual_tokens} tokens at frame_seqlen="
                                f"{frame_seqlen}, expected at most "
                                f"{cell.max_full_frame_equivalents} "
                                "full-frame equivalents"
                            )
                    else:
                        actual_frames = (
                            int(event["sink_frame_count"])
                            + union_count
                            + int(event["recent_frame_count"])
                        )
                        if actual_frames > cell.max_full_frame_equivalents:
                            failures.append(
                                f"line {line_number}: role-event read has "
                                f"{actual_frames} frames, expected at most "
                                f"{cell.max_full_frame_equivalents}"
                            )
                    for item in event["strategies"]:
                        name = str(item["name"])
                        state = item.get("state")
                        if not isinstance(state, dict):
                            failures.append(
                                f"line {line_number}: {name} has no "
                                "auditable state"
                            )
                            continue
                        if name == "SemanticLandmarkStrategy":
                            frame_ids = [
                                int(value)
                                for value in state.get(
                                    "anchor_frame_ids",
                                    [],
                                )
                            ]
                            if len(frame_ids) > int(state["capacity"]):
                                failures.append(
                                    f"line {line_number}: landmark bank "
                                    "exceeds capacity"
                                )
                        elif name == "CoherentMotionStrategy":
                            pairs = state.get("pair_frame_ids", [])
                            active_policy = (
                                cell.support_policy
                                if label == 10
                                else cell.suppress_policy
                            )
                            if (
                                active_policy in RESERVOIR_MOTION_POLICIES
                                and int(state.get("pair_capacity", -1)) != 1
                            ):
                                failures.append(
                                    f"line {line_number}: hybrid motion "
                                    "pair capacity must be one"
                                )
                            expected_pair_age = (
                                12
                                if active_policy in FRESH_MOTION_POLICIES
                                else 24
                            )
                            expected_stale_refresh = (
                                active_policy in FRESH_MOTION_POLICIES
                            )
                            if (
                                int(state.get("max_pair_age", -1))
                                != expected_pair_age
                            ):
                                failures.append(
                                    f"line {line_number}: motion-pair "
                                    f"max age {state.get('max_pair_age')} != "
                                    f"{expected_pair_age}"
                                )
                            if bool(
                                state.get(
                                    "stale_refresh_bypass_quantile",
                                    False,
                                )
                            ) != expected_stale_refresh:
                                failures.append(
                                    f"line {line_number}: stale-refresh "
                                    "quantile policy mismatch"
                                )
                            expected_state_contract = (
                                expected_state_match_contract(active_policy)
                            )
                            expected_state_match = (
                                expected_state_contract is not None
                            )
                            if bool(state.get("state_match", False)) != (
                                expected_state_match
                            ):
                                failures.append(
                                    f"line {line_number}: state-match "
                                    "policy mismatch"
                                )
                            read_pair = [
                                int(value)
                                for value in item.get("frame_ids", [])
                            ]
                            read_pairs_atomic = (
                                len(read_pair) % 2 == 0
                                and all(
                                    read_pair[offset] + 1
                                    == read_pair[offset + 1]
                                    for offset in range(
                                        0,
                                        len(read_pair),
                                        2,
                                    )
                                )
                            )
                            if not read_pairs_atomic:
                                failures.append(
                                    f"line {line_number}: coherent-motion "
                                    f"read is not an atomic pair: {read_pair}"
                                )
                            if expected_state_match:
                                assert expected_state_contract is not None
                                actual_state_contract = {
                                    key: state.get(key)
                                    for key in expected_state_contract
                                }
                                if actual_state_contract != expected_state_contract:
                                    failures.append(
                                        f"line {line_number}: state-match "
                                        "frozen parameters changed: "
                                        f"{actual_state_contract} != "
                                        f"{expected_state_contract}"
                                    )
                                last_retrieval = state.get(
                                    "last_retrieval",
                                    {},
                                )
                                if last_retrieval:
                                    required_retrieval = {
                                        "eligible_before_age",
                                        "eligible",
                                        "direction_available",
                                        "candidates",
                                        "selected",
                                        "legacy_selected",
                                        "newest_passing",
                                        "newest_age_eligible",
                                        "compatible_candidate_count",
                                        "fallback_used",
                                        "fallback_reason",
                                        "read_budget_preserved",
                                        "selection_mode",
                                        "selection_changed_from_legacy",
                                        "selected_age",
                                        "selected_compatibility",
                                        "selected_score",
                                        "reason",
                                    }
                                    missing_retrieval = sorted(
                                        required_retrieval
                                        - set(last_retrieval)
                                    )
                                    if missing_retrieval:
                                        failures.append(
                                            f"line {line_number}: "
                                            "state-match retrieval is missing "
                                            f"{missing_retrieval}"
                                        )
                            pair_bank_capacity = (
                                int(state.get("state_archive_capacity", -1))
                                if active_policy in STATE_MATCH_POLICIES
                                else int(state["pair_capacity"])
                            )
                            if len(pairs) > pair_bank_capacity:
                                failures.append(
                                    f"line {line_number}: motion-pair bank "
                                    "exceeds pair capacity"
                                )
                            normalized_pairs = []
                            for pair_index, pair in enumerate(pairs):
                                if (
                                    not isinstance(pair, (list, tuple))
                                    or len(pair) != 2
                                ):
                                    failures.append(
                                        f"line {line_number}: motion pair "
                                        f"{pair_index} is malformed: {pair!r}"
                                    )
                                    continue
                                start_t, end_t = (int(pair[0]), int(pair[1]))
                                normalized_pairs.append((start_t, end_t))
                                if start_t + 1 != end_t:
                                    failures.append(
                                        f"line {line_number}: motion pair "
                                        f"{pair_index} is not adjacent: "
                                        f"{pair!r}"
                                    )
                            unique_frames = {
                                value
                                for pair in normalized_pairs
                                for value in pair
                            }
                            frame_bank_capacity = (
                                pair_bank_capacity * 2
                                if active_policy in STATE_MATCH_POLICIES
                                else int(state["capacity"])
                            )
                            if len(unique_frames) > frame_bank_capacity:
                                failures.append(
                                    f"line {line_number}: motion-pair bank "
                                    "exceeds frame capacity"
                                )
                            min_spacing = int(state["min_pair_spacing"])
                            end_times = sorted(
                                end_t for _, end_t in normalized_pairs
                            )
                            if any(
                                right - left < min_spacing
                                for left, right in zip(
                                    end_times,
                                    end_times[1:],
                                )
                            ):
                                failures.append(
                                    f"line {line_number}: motion-pair bank "
                                    f"violates end-time spacing {min_spacing}: "
                                    f"{end_times}"
                                )
                            last_decision = state.get("last_decision", {})
                            if last_decision.get("candidate_pair") is not None:
                                required = {
                                    "bank_size_before",
                                    "filling",
                                    "victim_age",
                                    "retained_pair_end_ts",
                                    "spacing_checks",
                                    "spacing_ok",
                                    "motion_quantile_pass",
                                    "motion_ok",
                                    "replacement_ok",
                                }
                                missing = sorted(
                                    required - set(last_decision)
                                )
                                if missing:
                                    failures.append(
                                        f"line {line_number}: motion-pair "
                                        "decision is missing debug fields "
                                        f"{missing}"
                                    )
                                if (
                                    active_policy in FRESH_MOTION_POLICIES
                                    and "stale_quantile_bypass"
                                    not in last_decision
                                ):
                                    failures.append(
                                        f"line {line_number}: fresh-motion "
                                        "decision is missing "
                                        "stale_quantile_bypass"
                                    )
                                if last_decision.get(
                                    "stale_quantile_bypass"
                                ) and not last_decision.get("victim_stale"):
                                    failures.append(
                                        f"line {line_number}: quantile "
                                        "bypass used for a non-stale pair"
                                    )
                        elif name == "SemanticRetrievalStrategy":
                            archive = [
                                int(value)
                                for value in state.get(
                                    "archive_frame_ids",
                                    [],
                                )
                            ]
                            if len(archive) > int(state["archive_capacity"]):
                                failures.append(
                                    f"line {line_number}: retrieval archive "
                                    "exceeds capacity"
                                )
                            selected = state.get(
                                "last_retrieval",
                                {},
                            ).get("selected", [])
                            if len(selected) > int(state["capacity"]):
                                failures.append(
                                    f"line {line_number}: retrieval read "
                                    "exceeds top-k capacity"
                                )
                            retrieval = state.get("last_retrieval", {})
                            max_age = state.get("max_age")
                            if max_age is not None:
                                required = {
                                    "eligible_before_age",
                                    "eligible",
                                    "age_filtered",
                                    "max_age",
                                }
                                missing = sorted(required - set(retrieval))
                                if missing:
                                    failures.append(
                                        f"line {line_number}: retrieval "
                                        f"age audit fields missing {missing}"
                                    )
                                for selected_item in selected:
                                    if int(selected_item.get("age", -1)) < 0:
                                        failures.append(
                                            f"line {line_number}: retrieval "
                                            "selection has invalid age"
                                        )
                                    elif int(selected_item["age"]) > int(max_age):
                                        failures.append(
                                            f"line {line_number}: retrieval "
                                            f"age {selected_item['age']} "
                                            f"exceeds max_age={max_age}"
                                        )
                            if cell.retrieval_abstain:
                                reason = retrieval.get("reason")
                                valid_reasons = {
                                    "empty",
                                    "age_gate",
                                    "similarity_gate",
                                    "margin_gate",
                                    "selected",
                                }
                                if (
                                    not state.get(
                                        "abstain_on_low_confidence",
                                        False,
                                    )
                                    or reason not in valid_reasons
                                    or not math.isclose(
                                        float(
                                            state.get(
                                                "min_similarity",
                                                float("nan"),
                                            )
                                        ),
                                        float(
                                            cell.retrieval_min_similarity
                                        ),
                                        abs_tol=1e-9,
                                    )
                                    or not math.isclose(
                                        float(
                                            state.get(
                                                "min_margin",
                                                float("nan"),
                                            )
                                        ),
                                        float(cell.retrieval_min_margin),
                                        abs_tol=1e-9,
                                    )
                                ):
                                    failures.append(
                                        f"line {line_number}: retrieval "
                                        "confidence-gate contract mismatch"
                                    )
                                if reason in {
                                    "similarity_gate",
                                    "margin_gate",
                                } and selected:
                                    failures.append(
                                        f"line {line_number}: gated retrieval "
                                        "must abstain"
                                    )
                                if reason in {
                                    "similarity_gate",
                                    "margin_gate",
                                    "selected",
                                } and retrieval.get(
                                    "top1_similarity"
                                ) is None:
                                    failures.append(
                                        f"line {line_number}: retrieval gate "
                                        "is missing top-1 similarity"
                                    )
                        elif name == "TemporalPrototypeStrategy":
                            spans = state.get("prototype_spans", [])
                            medoids = state.get("prototype_medoid_ids", [])
                            counts = state.get("prototype_counts", [])
                            if (
                                len(spans) > int(state["capacity"])
                                or len(spans) != len(medoids)
                                or len(spans) != len(counts)
                            ):
                                failures.append(
                                    f"line {line_number}: prototype bank "
                                    "shape/capacity mismatch"
                                )
                            for span, medoid, count in zip(
                                spans,
                                medoids,
                                counts,
                            ):
                                if (
                                    len(span) != 2
                                    or int(span[0]) > int(medoid)
                                    or int(medoid) > int(span[1])
                                    or int(count) <= 0
                                ):
                                    failures.append(
                                        f"line {line_number}: invalid "
                                        f"prototype {span}/{medoid}/{count}"
                                    )
                        elif name in {
                            "UniqueSnapshotStrategy",
                            "SparseSnapshotStrategy",
                        }:
                            frames = state.get("snapshot_frame_ids", [])
                            token_counts = state.get(
                                "snapshot_token_counts",
                                [],
                            )
                            if (
                                len(frames) > int(state["capacity"])
                                or len(frames) != len(token_counts)
                                or any(int(value) <= 0 for value in token_counts)
                            ):
                                failures.append(
                                    f"line {line_number}: snapshot bank "
                                    "shape/capacity mismatch"
                                )
                            if name == "SparseSnapshotStrategy":
                                ratio = float(state["keep_ratio"])
                                if "frame_seqlen" not in event:
                                    failures.append(
                                        f"line {line_number}: sparse "
                                        "snapshot trace lacks frame_seqlen"
                                    )
                                    continue
                                expected_max = math.ceil(
                                    int(event["frame_seqlen"]) * ratio
                                )
                                if any(
                                    int(value) > expected_max
                                    for value in token_counts
                                ):
                                    failures.append(
                                        f"line {line_number}: sparse "
                                        "snapshot exceeds keep ratio"
                                    )
            except (IndexError, KeyError, TypeError, ValueError) as error:
                failures.append(f"line {line_number}: malformed event: {error}")
    if records == 0:
        failures.append("trace has no middle_selection events")
    expected_layers = set(TRACE_LAYERS)
    observed_layers = {layer for layer, _ in observed}
    if not expected_layers.issubset(observed_layers):
        failures.append(
            f"missing trace layers {sorted(expected_layers - observed_layers)}"
        )
    payload = {
        "version": 1,
        "cell": cell.name,
        "records": records,
        "observed_pairs": len(observed),
        "failures": failures[:100],
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"policy trace audit failed for {cell.name}: {failures[:5]}"
        )
    return payload


def audit_motion_trace(
    path: Path,
    *,
    cell: Cell,
    legacy_map: Path,
    report_path: Path,
) -> dict[str, Any]:
    labels = read_matrix(legacy_map, {10, 11})
    expected_layers = {
        layer
        for layer in TRACE_LAYERS
        if 11 in labels[layer]
    }
    observed_layers: set[int] = set()
    failures: list[str] = []
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
                if event["event"] != "motion_event_update":
                    raise ValueError(f"unexpected event {event['event']!r}")
                layer = int(event["layer"])
                num_frames = int(event["num_frames"])
                offsets = [int(value) for value in event["selected_offsets"]]
                selected_ids = [
                    int(value) for value in event["selected_frame_ids"]
                ]
                scores = [float(value) for value in event["all_scores"]]
                expected_head_count = labels[layer].count(11)
                if int(event["responsive_head_count"]) != expected_head_count:
                    raise ValueError("responsive head count differs from map")
                if len(scores) != num_frames:
                    raise ValueError("score count differs from block frames")
                if len(offsets) != min(cell.motion_top_k, num_frames):
                    raise ValueError("selected offset count differs from top-k")
                if any(
                    offset < 0 or offset >= num_frames for offset in offsets
                ):
                    raise ValueError("selected offset outside block")
                if selected_ids != [
                    int(event["frame_start_t"]) + offset
                    for offset in offsets
                ]:
                    raise ValueError("selected frame ids do not match offsets")
                if any(not math.isfinite(value) or value < 0 for value in scores):
                    raise ValueError("motion scores must be finite and non-negative")
                records += 1
                observed_layers.add(layer)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"line {line_number}: {error}")
    if records == 0:
        failures.append("motion trace has no records")
    if not expected_layers.issubset(observed_layers):
        failures.append(
            "motion trace is missing responsive sampled layers: "
            f"{sorted(expected_layers - observed_layers)}"
        )
    payload = {
        "version": 1,
        "cell": cell.name,
        "records": records,
        "observed_layers": sorted(observed_layers),
        "expected_layers": sorted(expected_layers),
        "failures": failures[:100],
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"motion trace audit failed for {cell.name}: {failures[:5]}"
        )
    return payload


def audit_role_event_trace(
    path: Path,
    *,
    cell: Cell,
    head_map: Path,
    report_path: Path,
) -> dict[str, Any]:
    labels = read_matrix(head_map, {10, 11})

    def policy_routes(policy: str | None) -> dict[str, int]:
        routes: dict[str, int] = {}
        if policy in {"landmark", "landmark_motion"}:
            routes["landmark"] = 4 if policy == "landmark" else 2
        if policy in {
            "motion_pair",
            "motion_pair1",
            "landmark_motion",
            "retrieval1_motion1_age24",
        } | RESERVOIR_MOTION_POLICIES:
            routes["motion"] = 2 if policy == "motion_pair" else 1
        if policy in {
            "retrieval",
            "retrieval2",
            "retrieval1",
            "retrieval1_age24",
            "retrieval1_motion1_age24",
        }:
            routes["retrieval"] = (
                4
                if policy == "retrieval"
                else 2
                if policy == "retrieval2"
                else 1
            )
        if policy in {"prototype", "prototype2"}:
            routes["prototype"] = 4 if policy == "prototype" else 2
        if policy in {"snapshot", "snapshot2"}:
            routes["snapshot"] = 4 if policy == "snapshot" else 2
        if policy == "sparse75":
            routes["sparse"] = 4
        return routes

    support_routes = policy_routes(cell.support_policy)
    suppress_routes = policy_routes(cell.suppress_policy)
    shared_routes = {
        prefix
        for prefix, capacity in support_routes.items()
        if suppress_routes.get(prefix) == capacity
    }

    def policy_groups(policy: str | None, label: int) -> set[str]:
        routes = policy_routes(policy)
        return {
            (
                f"{prefix}:all"
                if prefix in shared_routes
                else f"{prefix}:{label}"
            )
            for prefix in routes
        }

    expected_heads_by_group: dict[tuple[int, str], list[int]] = {}
    support_groups = policy_groups(cell.support_policy, 10)
    suppress_groups = policy_groups(cell.suppress_policy, 11)
    for layer in TRACE_LAYERS:
        for label, groups in (
            (10, support_groups),
            (11, suppress_groups),
        ):
            for key in groups:
                head_ids = (
                    list(range(len(labels[layer])))
                    if key.endswith(":all")
                    else [
                        head
                        for head, value in enumerate(labels[layer])
                        if int(value) == label
                    ]
                )
                if head_ids:
                    expected_heads_by_group[(layer, key)] = head_ids
    expected = set(expected_heads_by_group)

    observed: set[tuple[int, str]] = set()
    failures: list[str] = []
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
                if event["event"] != "role_event_features":
                    raise ValueError(f"unexpected event {event['event']!r}")
                layer = int(event["layer"])
                context_key = str(event["context_key"])
                role_text, label_text = context_key.split(":", maxsplit=1)
                if role_text not in {
                    "landmark",
                    "motion",
                    "retrieval",
                    "prototype",
                    "snapshot",
                    "sparse",
                }:
                    raise ValueError("unknown role-event context prefix")
                if label_text not in {"all", "10", "11"}:
                    raise ValueError(
                        "role-event suffix must be all, 10, or 11"
                    )
                expected_heads = expected_heads_by_group.get(
                    (layer, context_key)
                )
                if expected_heads is None:
                    raise ValueError(
                        f"unexpected role-event group {(layer, context_key)}"
                    )
                observed_heads = [int(value) for value in event["head_ids"]]
                if observed_heads != expected_heads:
                    raise ValueError(
                        f"head ids {observed_heads} != {expected_heads}"
                    )
                if int(event["head_count"]) != len(expected_heads):
                    raise ValueError("head count differs from map")
                num_frames = int(event["num_frames"])
                motion_scores = [
                    float(value) for value in event["motion_scores"]
                ]
                semantic_scores = [
                    float(value)
                    for value in event["adjacent_semantic_similarity"]
                ]
                if len(motion_scores) != num_frames:
                    raise ValueError("motion score count differs from frames")
                if len(semantic_scores) != max(0, num_frames - 1):
                    raise ValueError(
                        "semantic score count differs from adjacent edges"
                    )
                if any(
                    not math.isfinite(value) or value < 0.0
                    for value in motion_scores
                ):
                    raise ValueError(
                        "role-event motion scores must be finite and non-negative"
                    )
                if any(
                    not math.isfinite(value) or not -1.001 <= value <= 1.001
                    for value in semantic_scores
                ):
                    raise ValueError(
                        "role-event semantic scores must be finite cosines"
                    )
                if role_text == "sparse":
                    token_summary = event.get("token_score_summary")
                    if (
                        not isinstance(token_summary, dict)
                        or int(token_summary.get("tokens_per_frame", 0)) <= 0
                        or not all(
                            math.isfinite(float(token_summary[key]))
                            for key in ("min", "max", "mean")
                        )
                    ):
                        raise ValueError(
                            "sparse role-event group has invalid token scores"
                        )
                records += 1
                observed.add((layer, context_key))
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                failures.append(f"line {line_number}: {error}")
    if records == 0:
        failures.append("role-event trace has no records")
    missing = expected - observed
    if missing:
        failures.append(
            f"role-event trace is missing groups {sorted(missing)}"
        )
    unexpected = observed - expected
    if unexpected:
        failures.append(
            f"role-event trace has unexpected groups {sorted(unexpected)}"
        )
    payload = {
        "version": 1,
        "cell": cell.name,
        "records": records,
        "observed": [
            {"layer": layer, "context_key": key}
            for layer, key in sorted(observed)
        ],
        "expected": [
            {"layer": layer, "context_key": key}
            for layer, key in sorted(expected)
        ],
        "failures": failures[:100],
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"role-event trace audit failed for {cell.name}: {failures[:5]}"
        )
    return payload


def audit_scene_trace(
    path: Path,
    *,
    cell: Cell,
    report_path: Path,
) -> dict[str, Any]:
    events = [
        json.loads(raw)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    failures: list[str] = []
    if len(events) != 3:
        failures.append(f"expected 3 scene events, found {len(events)}")
    canonical = [
        int(event.get("canonical_scene_id", -1)) for event in events
    ]
    if canonical != [0, 1, 0]:
        failures.append(f"expected canonical A-B-A ids [0,1,0], got {canonical}")
    if events:
        if not bool(events[0].get("initial")):
            failures.append("first scene event is not initial")
        if any(
            event.get("event") != "scene_cache_switch"
            for event in events
        ):
            failures.append("unexpected scene trace event")
    if len(events) >= 3:
        if cell.scene_manual:
            if events[2].get("decision") != "manual_recall":
                failures.append("manual A2 event is not manual_recall")
        elif events[2].get("decision") != "recall":
            failures.append("automatic A2 event is not recall")
        if (
            int(events[2].get("action_counts", {}).get("restore_scene", 0))
            <= 0
        ):
            failures.append("A2 did not restore any stride scene bank")
        if float(events[2].get("similarity", -1.0)) < float(
            events[2].get("threshold", 0.0)
        ):
            failures.append("A2 recall similarity is below threshold")
    payload = {
        "version": 1,
        "cell": cell.name,
        "events": len(events),
        "canonical_scene_ids": canonical,
        "decisions": [event.get("decision") for event in events],
        "similarities": [event.get("similarity") for event in events],
        "failures": failures,
        "ok": not failures,
    }
    write_frozen(report_path, payload)
    if failures:
        raise RuntimeError(
            f"scene trace audit failed for {cell.name}: {failures}"
        )
    return payload


def inference_command(
    args: argparse.Namespace,
    *,
    cell: Cell,
    output: Path,
    transition_trace: Path,
    scene_trace: Path,
) -> tuple[list[str], Path, Path, int]:
    prompts = (
        args.single_prompts
        if cell.prompt_kind == "single"
        else args.aba_prompts
    )
    prompt_index = (
        args.single_prompt_index
        if cell.prompt_kind == "single"
        else args.aba_prompt_index
    )
    head_map = resolve_head_map(args, cell)
    command = [
        sys.executable,
        "inference.py",
        "--config_path",
        str(args.pf_config),
        "--checkpoint_path",
        str(args.pf_checkpoint),
        "--data_path",
        str(prompts),
        "--output_folder",
        str(output),
        "--num_output_frames",
        str(int(getattr(args, "num_output_frames", 120))),
        "--seed",
        str(args.seed),
        "--num_samples",
        "1",
        "--use_ema",
        "--save_with_index",
        "--start_idx",
        str(prompt_index),
        "--end_idx",
        str(prompt_index + 1),
        "--reseed_per_prompt",
        "--pyramidkv_head_config_path",
        str(head_map),
    ]
    if not cell.native:
        command.extend(
            [
                "--pyramidkv_history_polarity",
                "--pyramidkv_history_support_policy",
                cell.support_policy,
                "--pyramidkv_history_suppress_policy",
                str(cell.suppress_policy),
                "--pyramidkv_history_budget_profile",
                cell.history_budget_profile,
                "--pyramidkv_motion_event_top_k",
                str(cell.motion_top_k),
                "--pyramidkv_motion_event_sample_tokens",
                "64",
            ]
        )
        if cell.retrieval_abstain:
            command.extend(
                [
                    "--pyramidkv_semantic_retrieval_abstain",
                    "--pyramidkv_semantic_retrieval_min_similarity",
                    str(float(cell.retrieval_min_similarity)),
                    "--pyramidkv_semantic_retrieval_min_margin",
                    str(float(cell.retrieval_min_margin)),
                ]
            )
    if cell.transition:
        command.extend(
            [
                "--pyramidkv_cache_transition",
                "--pyramidkv_cache_transition_mode",
                "full",
                "--pyramidkv_cache_transition_min_reliability",
                ".55",
                "--pyramidkv_cache_transition_min_novelty",
                ".01",
                "--pyramidkv_cache_transition_max_commit_fraction",
                ".75",
                "--pyramidkv_cache_transition_stagger_period",
                "1",
                "--pyramidkv_cache_transition_max_age_blocks",
                "6",
                "--pyramidkv_cache_transition_branches",
                "cond",
                "--pyramidkv_cache_transition_denoise_weight",
                "2",
                "--pyramidkv_cache_transition_trace_path",
                str(transition_trace),
                "--pyramidkv_cache_transition_debug",
            ]
        )
    if cell.variance_refresh:
        command.extend(
            [
                "--pyramidkv_history_value_renorm_strength",
                ".5",
                "--pyramidkv_history_value_recent_frames",
                "4",
                "--pyramidkv_history_value_gate_lambda",
                "3",
                "--pyramidkv_history_value_labels",
                "10,11",
                "--pyramidkv_history_value_layer_start",
                "10",
                "--pyramidkv_history_value_layer_end",
                "20",
                "--pyramidkv_history_value_moment_mode",
                "variance_only",
                "--pyramidkv_history_value_target_frames",
                "8",
                "--pyramidkv_history_value_max_std_ratio",
                "1.5",
            ]
        )
    if cell.scene_cache:
        command.extend(
            [
                "--pyramidkv_scene_cache",
                "--pyramidkv_scene_cache_match_mode",
                "idf",
                "--pyramidkv_scene_cache_similarity_threshold",
                ".20",
                "--pyramidkv_scene_cache_max_scenes",
                "8",
                "--pyramidkv_scene_cache_bridge_recent_frames",
                str(cell.scene_bridge),
                "--pyramidkv_scene_cache_trace_path",
                str(scene_trace),
                "--pyramidkv_scene_cache_debug",
            ]
        )
        if cell.scene_manual:
            command.extend(
                ["--pyramidkv_scene_cache_manual_ids", "0,1,0"]
            )
    return command, args.pf_repo, head_map, prompt_index


def run_cell(
    args: argparse.Namespace,
    *,
    cell: Cell,
    gpu: str,
    experiment_contract_sha256: str,
) -> dict[str, str]:
    output = args.out_root / "videos" / cell.name
    log = args.out_root / "logs" / f"{cell.name}.log"
    policy_trace = args.out_root / "traces" / f"{cell.name}.policy.jsonl"
    motion_trace = args.out_root / "traces" / f"{cell.name}.motion.jsonl"
    role_event_trace = (
        args.out_root / "traces" / f"{cell.name}.role_event.jsonl"
    )
    transition_trace = (
        args.out_root / "traces" / f"{cell.name}.transition.jsonl"
    )
    scene_trace = args.out_root / "traces" / f"{cell.name}.scene.jsonl"
    config_path = args.out_root / "configs" / f"{cell.name}.json"
    marker = args.out_root / "status" / f"{cell.name}.done.json"
    video_report = args.out_root / "diagnostics" / f"{cell.name}.video.json"
    policy_report = (
        args.out_root / "diagnostics" / f"{cell.name}.policy.json"
    )
    motion_report = (
        args.out_root / "diagnostics" / f"{cell.name}.motion.json"
    )
    role_event_report = (
        args.out_root / "diagnostics" / f"{cell.name}.role_event.json"
    )
    scene_report = (
        args.out_root / "diagnostics" / f"{cell.name}.scene.json"
    )
    command, cwd, head_map, prompt_index = inference_command(
        args,
        cell=cell,
        output=output,
        transition_trace=transition_trace,
        scene_trace=scene_trace,
    )
    head_map_audit = getattr(args, "head_map_audits", {}).get(cell.map_key)
    cell_config = {
        "version": 1,
        "experiment": getattr(
            args, "experiment_name", "v100_fast_selection_1video"
        ),
        "experiment_contract_sha256": experiment_contract_sha256,
        "cell": asdict(cell),
        "gpu": str(gpu),
        "prompt_index": int(prompt_index),
        "prompt_path": str(
            args.single_prompts
            if cell.prompt_kind == "single"
            else args.aba_prompts
        ),
        "head_map": str(head_map),
        "head_map_sha256": sha256(head_map),
        "head_map_audit": head_map_audit,
        "command": command,
    }
    config_sha = write_frozen(config_path, cell_config)

    if marker.is_file():
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if completed.get("config_sha256") != config_sha:
            raise RuntimeError(f"stale completion marker: {marker}")
        audit_video(
            args,
            cell=cell,
            output=output,
            report=video_report,
            log=video_report.with_suffix(".log"),
        )
        audit_policy_trace(
            policy_trace,
            cell=cell,
            head_map=head_map,
            report_path=policy_report,
        )
        if cell.uses_motion:
            audit_motion_trace(
                motion_trace,
                cell=cell,
                legacy_map=head_map,
                report_path=motion_report,
            )
        if cell.uses_role_event:
            audit_role_event_trace(
                role_event_trace,
                cell=cell,
                head_map=head_map,
                report_path=role_event_report,
            )
        if cell.scene_cache:
            audit_scene_trace(
                scene_trace,
                cell=cell,
                report_path=scene_report,
            )
        return {"name": cell.name, "status": "resumed"}

    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.mp4")):
        import shutil
        shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
    for stale in (
        policy_trace,
        motion_trace,
        role_event_trace,
        transition_trace,
        scene_trace,
        policy_report,
        motion_report,
        role_event_report,
        scene_report,
        video_report,
    ):
        if stale.exists():
            stale.unlink()

    env = os.environ.copy()
    for key in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    root_python = str(args.repo_root / "src")
    env["PYTHONPATH"] = (
        root_python
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "COMMIT_FORCING_ENABLE": "0",
            "PYRAMIDKV_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_STRATEGY": "0",
            "PYRAMIDKV_USE_CPP_PACK": "0",
            "PYRAMIDKV_USE_CPP_PACK_OUTPUT": "0",
            "PYRAMIDKV_USE_MEGA_CACHE": "0",
            "PYRAMIDKV_USE_MEGA_ATTN": "0",
            "PYRAMIDKV_CONTIG_ANCHOR_STORE": "0",
            "PYRAMIDKV_HEAD_MAP_DEBUG": "1",
            "PYRAMIDKV_POLICY_TRACE_PATH": str(policy_trace),
            "PYRAMIDKV_POLICY_TRACE_LAYERS": ",".join(
                str(value) for value in TRACE_LAYERS
            ),
            "PYRAMIDKV_POLICY_TRACE_STRIDE": "3",
            "PYRAMIDKV_POLICY_TRACE_MAX_RECORDS": "30000",
        }
    )
    if cell.uses_motion:
        env.update(
            {
                "PYRAMIDKV_MOTION_TRACE_PATH": str(motion_trace),
                "PYRAMIDKV_MOTION_TRACE_LAYERS": ",".join(
                    str(value) for value in TRACE_LAYERS
                ),
                "PYRAMIDKV_MOTION_DEBUG": "1",
            }
        )
    else:
        env.pop("PYRAMIDKV_MOTION_TRACE_PATH", None)
        env.pop("PYRAMIDKV_MOTION_DEBUG", None)
    if cell.uses_role_event:
        env.update(
            {
                "PYRAMIDKV_ROLE_EVENT_TRACE_PATH": str(role_event_trace),
                "PYRAMIDKV_ROLE_EVENT_TRACE_LAYERS": ",".join(
                    str(value) for value in TRACE_LAYERS
                ),
                "PYRAMIDKV_ROLE_EVENT_DEBUG": "1",
            }
        )
    else:
        env.pop("PYRAMIDKV_ROLE_EVENT_TRACE_PATH", None)
        env.pop("PYRAMIDKV_ROLE_EVENT_DEBUG", None)

    print(
        f"[run] stage={cell.stage} cell={cell.name} gpu={gpu} "
        f"prompt={prompt_index} map={cell.map_key} "
        f"map_sha256={sha256(head_map)}",
        flush=True,
    )
    run_checked(command, cwd=cwd, env=env, log_path=log)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    failure_signatures = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
        "PyramidKVPolicyTraceError",
        "PyramidKVMotionTraceError",
        "PyramidKVRoleEventTraceError",
    )
    hits = [
        signature for signature in failure_signatures
        if signature in log_text
    ]
    if hits:
        raise RuntimeError(f"{cell.name}: failure signatures in log: {hits}")
    if "[PyramidKVRuntimePolicy]" not in log_text:
        raise RuntimeError(f"{cell.name}: runtime policy marker is missing")
    if not cell.native:
        required = (
            "[HistoryPolarityPolicy]",
            "legacy_pf_labels=false",
            "exclusive_owner=true",
        )
        if any(value not in log_text for value in required):
            raise RuntimeError(
                f"{cell.name}: neutral exclusive-policy marker is incomplete"
            )
    if cell.scene_cache and "[SceneCacheSwitch]" not in log_text:
        raise RuntimeError(f"{cell.name}: scene-cache switch marker is missing")
    if cell.uses_motion and "[PyramidKVMotionEvent]" not in log_text:
        raise RuntimeError(f"{cell.name}: motion-event marker is missing")
    if (
        cell.uses_role_event
        and "[PyramidKVRoleEvent]" not in log_text
    ):
        raise RuntimeError(f"{cell.name}: role-event marker is missing")

    video = audit_video(
        args,
        cell=cell,
        output=output,
        report=video_report,
        log=video_report.with_suffix(".log"),
    )
    policy = audit_policy_trace(
        policy_trace,
        cell=cell,
        head_map=head_map,
        report_path=policy_report,
    )
    motion = None
    if cell.uses_motion:
        motion = audit_motion_trace(
            motion_trace,
            cell=cell,
            legacy_map=head_map,
            report_path=motion_report,
        )
    role_event = None
    if cell.uses_role_event:
        role_event = audit_role_event_trace(
            role_event_trace,
            cell=cell,
            head_map=head_map,
            report_path=role_event_report,
        )
    scene = None
    if cell.scene_cache:
        scene = audit_scene_trace(
            scene_trace,
            cell=cell,
            report_path=scene_report,
        )
    if cell.transition and not transition_trace.is_file():
        raise RuntimeError(f"{cell.name}: transition trace is missing")

    marker_payload = {
        "version": 1,
        "cell": cell.name,
        "config_sha256": config_sha,
        "log_sha256": sha256(log),
        "video_fingerprint": video["input_fingerprint"],
        "policy_records": policy["records"],
        "motion_records": None if motion is None else motion["records"],
        "role_event_records": (
            None if role_event is None else role_event["records"]
        ),
        "scene_events": None if scene is None else scene["events"],
        "completed_at_unix": int(time.time()),
    }
    write_frozen(marker, marker_payload)
    return {"name": cell.name, "status": "completed"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("responsive", "tricks", "aba", "all"),
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
        "--legacy-map", type=Path, default=env_path("LEGACY_MAP")
    )
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
    parser.add_argument("--contract-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--reproduce-known-broken-map",
        action="store_true",
        help=(
            "Explicitly reproduce the historical v100 304/56-map screen. "
            "The non-native cells are known to produce polygon noise."
        ),
    )
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
        / "runs"
        / "v98_history_polarity"
        / "maps"
        / "history_polarity_zero.csv"
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
        args.out_root
        or args.repo_root / "runs" / "v100_fast_selection_1video"
    ).resolve()
    return args


def main() -> None:
    args = parse_args()
    if not args.reproduce_known_broken_map:
        raise SystemExit(
            "v100 is frozen as a historical 304/56-map reproduction and is "
            "disabled by default after docs/106. Use "
            "scripts/run_v107_polygon_rootcause_1video.py for recovery, or "
            "pass --reproduce-known-broken-map only to reproduce v100."
        )
    if args.num_nodes <= 0:
        raise SystemExit("--num-nodes must be positive")
    if not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("--node-rank must be within [0, num-nodes)")
    if args.seed != 0:
        raise SystemExit("the frozen v100 fast screen requires seed 0")
    paths = (
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.single_prompts,
        args.aba_prompts,
        args.repo_root / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    single_lines = [
        value for value in args.single_prompts.read_text(
            encoding="utf-8"
        ).splitlines() if value.strip()
    ]
    aba_lines = [
        value for value in args.aba_prompts.read_text(
            encoding="utf-8"
        ).splitlines() if value.strip()
    ]
    if not 0 <= args.single_prompt_index < len(single_lines):
        raise SystemExit("--single-prompt-index is outside the prompt file")
    if not 0 <= args.aba_prompt_index < len(aba_lines):
        raise SystemExit("--aba-prompt-index is outside the ABA prompt file")
    if len(
        [
            part for part in aba_lines[args.aba_prompt_index].split("||")
            if part.strip()
        ]
    ) != 3:
        raise SystemExit("selected ABA prompt must contain exactly 3 segments")

    args.out_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "traces", "configs", "status", "diagnostics"):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)
    legacy = validate_legacy_map(args.legacy_map, args.pf_labels)
    all_mode_cells = (
        CELLS if args.mode == "all"
        else tuple(cell for cell in CELLS if cell.stage == args.mode)
    )
    contract = {
        "version": 1,
        "experiment": "v100_fast_selection_1video",
        "mode": args.mode,
        "seed": args.seed,
        "num_output_frames": 120,
        "expected_decoded_frames": EXPECTED_VIDEO_FRAMES,
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
        "legacy_v98_map": legacy,
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
        "cells": [asdict(cell) for cell in all_mode_cells],
        "implementation_hashes": {
            str(path.relative_to(args.repo_root)): sha256(path)
            for path in (
                args.pf_repo / "inference.py",
                args.pf_repo / "pipeline" / "causal_inference.py",
                args.pf_repo / "pipeline" / "pyramidkv_config.py",
                args.pf_repo / "pyramidkv" / "adaptive_cache.py",
                args.pf_repo / "pyramidkv" / "base.py",
                args.pf_repo / "pyramidkv" / "factory.py",
                args.pf_repo / "pyramidkv" / "motion_event.py",
                args.pf_repo / "pyramidkv" / "policy_overrides.py",
                args.pf_repo / "pyramidkv" / "stride.py",
            )
        },
    }
    contract_path = (
        args.out_root / "contracts" / f"{args.mode}.contract.json"
    )
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path,
            contract,
            timeout_seconds=args.contract_wait_seconds,
        )

    cells = selected_cells(
        args.mode,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
    )
    gpus = [
        value.strip() for value in args.gpu_list.split(",") if value.strip()
    ]
    if len(gpus) < len(cells) or len(gpus) != len(set(gpus)):
        raise SystemExit(
            f"node {args.node_rank} needs {len(cells)} unique GPUs; "
            f"received {gpus}"
        )
    print(
        "[v100] "
        f"mode={args.mode} node={args.node_rank}/{args.num_nodes} "
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
    node_summary = {
        "version": 1,
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
    write_frozen(summary_path, node_summary)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
