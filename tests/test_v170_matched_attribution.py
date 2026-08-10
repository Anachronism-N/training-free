from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_v120_moviebench32_main as parent
import run_v170_matched_attribution_moviebench16 as runner
import v170_matched_attribution_contract as contract


def test_v170_schedule_has_complete_balanced_coverage() -> None:
    rows = contract.full_schedule()
    assert len(rows) == 64
    assert {(row["method"], row["prompt_index"]) for row in rows} == {
        (method, prompt)
        for method in contract.METHODS
        for prompt in range(contract.PROMPT_COUNT)
    }
    for node_rank in range(contract.NUM_NODES):
        node_rows = contract.node_schedule(node_rank)
        assert {row["gpu_slot"] for row in node_rows} == set(range(8))
        for gpu_slot in range(8):
            gpu_rows = [row for row in node_rows if row["gpu_slot"] == gpu_slot]
            assert len(gpu_rows) == 2
            assert len({row["prompt_index"] for row in gpu_rows}) == 1
            assert [row["order"] for row in gpu_rows] == [0, 1]
    query_first = sum(
        row["order"] == 0 and row["method"] in contract.QUERY_METHODS for row in rows
    )
    assert query_first == 16
    assert sum(row["method"] in contract.QUERY_METHODS for row in rows) == 32


def test_v170_parent_grid_contains_only_four_replica_methods() -> None:
    previous = os.environ.get("PYRAMIDKV_POLICY_TRACE_HEADS")
    os.environ["PYRAMIDKV_POLICY_TRACE_HEADS"] = "0"
    try:
        runner.configure_parent_runner()
    finally:
        if previous is None:
            os.environ.pop("PYRAMIDKV_POLICY_TRACE_HEADS", None)
        else:
            os.environ["PYRAMIDKV_POLICY_TRACE_HEADS"] = previous
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="ours")
    assert tuple(method.key for method in methods) == contract.METHODS
    assert parent.INCLUDE_PF_BASELINE is False
    assert parent.ALLOW_PARTIAL_SCOPE is True


def test_v170_cells_change_only_selector_and_replica_identity() -> None:
    cells = {cell.name: asdict(cell) for cell in runner.V170_CELLS}
    for lane in ("a", "b"):
        reference = cells[f"v170_v166_{lane}"]
        candidate = cells[f"v170_queryweighted_{lane}"]
        differences = {key for key in reference if reference[key] != candidate[key]}
        assert differences == {"name", "stage", "support_policy"}
        assert reference["support_policy"] == "reservoir2_multiscalemotion1"
        assert candidate["support_policy"] == "reservoir2_multiscalequeryweighted1"
        assert reference["suppress_policy"] == candidate["suppress_policy"]
        assert reference["map_key"] == candidate["map_key"] == "middle10"


def test_v170_shell_and_trace_head_filter_are_frozen() -> None:
    generation = (SCRIPTS / "run_v170_matched_attribution_moviebench16.sh").read_text(
        encoding="utf-8"
    )
    adaptive = (
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    assert "export V120_OURS_ONLY=1" in generation
    assert "export PYRAMIDKV_POLICY_TRACE_HEADS=0" in generation
    assert 'NUM_NODES}" == "4' in generation
    assert "analyze_v170_full_layer_trace.py" in generation
    assert "PYRAMIDKV_POLICY_TRACE_HEADS" in adaptive
    assert "head_idx not in self._policy_trace_heads" in adaptive
