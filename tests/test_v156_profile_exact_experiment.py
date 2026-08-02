from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v156_vbench as analysis  # noqa: E402
import prepare_v156_vbench_comparison as vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v156_profile_exact_moviebench16 as v156  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402
from pyramidkv.temporal_reservoir import (  # noqa: E402
    TemporalProfileAnchorStrategy,
)


def test_profile_anchor_replays_exact_v152_uniform8_old_frames() -> None:
    strategy = TemporalProfileAnchorStrategy()
    strategy.reset(1)
    for t_val in range(117):
        key = torch.full((2, 3), float(t_val))
        value = key + 0.5
        positions = torch.tensor([[t_val, 0, 0], [t_val, 0, 1]])
        strategy.update(
            0,
            key,
            value,
            positions,
            2,
            t_val,
            t_vals=[t_val],
        )
    state = strategy.debug_state(0)
    assert state["target_frame_ids"] == [0, 37, 75, 112]
    assert state["anchor_frame_ids"] == [0, 37, 75, 112]
    assert state["physical_frame_count"] == 4
    assert "pending_frame_ids" not in state
    selected = strategy.collect(0, 116, 113, -1)
    assert [anchor.t for anchor in selected] == [0, 37, 75, 112]
    assert all(anchor.source_kind == "temporal_profile_anchor" for anchor in selected)


def test_v156_grid_is_exact_budgeted_and_evenly_sharded() -> None:
    v156.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")

    assert tuple(method.key for method in methods) == v156.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 112
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [28, 28, 28, 28]
    assert len(v156.REUSE_METHODS) * v156.PROMPT_COUNT == 32


def test_v156_profile_routes_match_the_frozen_v152_budget() -> None:
    for cell in v156.V156_CELLS[:3]:
        assert expected_policy(cell, 10) == (
            ("TemporalProfileAnchorStrategy",),
            0,
            4,
            "temporal_profile_anchor",
        )
        assert expected_policy(cell, 11) == ((), 0, 8, "stride")
        assert cell.max_full_frame_equivalents == 8

    all_profile = v156.V156_CELLS[3]
    assert expected_policy(all_profile, 10) == expected_policy(all_profile, 11)
    all_recent = v156.V156_CELLS[4]
    assert expected_policy(all_recent, 10) == expected_policy(all_recent, 11)
    assert expected_policy(all_recent, 10) == ((), 0, 8, "stride")


def test_v156_vbench_contract_and_promotion_gate() -> None:
    assert len(vbench.DIMENSIONS) == 16
    assert len(vbench.CORE_EVALUATION_DIMENSIONS) == 9
    rows = {
        method: {
            dimension: 0.7 for dimension in vbench.CORE_EVALUATION_DIMENSIONS
        }
        for method in vbench.METHODS
    }
    for dimension in (
        "subject_consistency",
        "background_consistency",
        "overall_consistency",
    ):
        rows[analysis.PRIMARY][dimension] = 0.8
    report = analysis.analyze(
        {
            "methods": rows,
            "dimensions": list(vbench.CORE_EVALUATION_DIMENSIONS),
            "missing": [],
        }
    )
    assert report["metric_promotion_gate"] is True

    rows[analysis.MEMBERSHIP_CONTROLS[0]]["dynamic_degree"] = 1.0
    report = analysis.analyze(
        {
            "methods": rows,
            "dimensions": list(vbench.CORE_EVALUATION_DIMENSIONS),
            "missing": [],
        }
    )
    assert report["metric_promotion_gate"] is False
