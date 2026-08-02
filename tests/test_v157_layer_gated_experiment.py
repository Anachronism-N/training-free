from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v157_blind_review as blind  # noqa: E402
import analyze_v157_vbench as analysis  # noqa: E402
import prepare_v157_vbench_comparison as vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v157_layer_gated_moviebench16 as v157  # noqa: E402
from build_v157_layer_gate_maps import (  # noqa: E402
    MAP_SPECS,
    build_manifest,
)
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


def test_v157_maps_are_count_matched_and_cover_depth_controls() -> None:
    manifest = build_manifest()
    assert set(manifest["maps"]) == set(MAP_SPECS)
    assert all(
        row["selected_head_count"] == 120
        for row in manifest["maps"].values()
    )
    block_layers = set(MAP_SPECS["early10"])
    block_layers |= set(MAP_SPECS["middle10"])
    block_layers |= set(MAP_SPECS["late10"])
    assert block_layers == set(range(30))
    assert not set(MAP_SPECS["early10"]) & set(MAP_SPECS["middle10"])
    assert not set(MAP_SPECS["middle10"]) & set(MAP_SPECS["late10"])


def test_v157_grid_reuses_half_and_shards_evenly() -> None:
    v157.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v157.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 128
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [32, 32, 32, 32]
    assert len(v157.REUSE_METHODS) * v157.PROMPT_COUNT == 64


def test_v157_layer_candidates_change_only_membership() -> None:
    for cell in v157.V157_CELLS[:4]:
        assert expected_policy(cell, 10) == (
            ("TemporalReservoirStrategy",),
            1,
            4,
            "temporal_reservoir",
        )
        assert expected_policy(cell, 11) == ((), 1, 8, "stride")
        assert cell.max_full_frame_equivalents == 9


def _synthetic_rows() -> dict[str, dict[str, float]]:
    return {
        method: {
            dimension: 0.7 for dimension in vbench.CORE_EVALUATION_DIMENSIONS
        }
        for method in vbench.METHODS
    }


def test_v157_metric_gate_requires_motion_and_stability_recovery() -> None:
    rows = _synthetic_rows()
    recent = analysis.RECENT
    all_reservoir = analysis.ALL_RESERVOIR
    candidate = vbench.LAYER_CANDIDATES[0]
    rows[all_reservoir]["temporal_flickering"] = 0.68
    rows[all_reservoir]["motion_smoothness"] = 0.68
    rows[candidate]["dynamic_degree"] = 0.73
    report = analysis.analyze(
        {
            "methods": rows,
            "dimensions": list(vbench.CORE_EVALUATION_DIMENSIONS),
            "missing": [],
        }
    )
    assert report["candidate_gates"][candidate]["passes"] is True
    assert report["metric_promotion_gate"] is True

    rows[candidate]["dynamic_degree"] = rows[recent]["dynamic_degree"] + 0.01
    report = analysis.analyze(
        {
            "methods": rows,
            "dimensions": list(vbench.CORE_EVALUATION_DIMENSIONS),
            "missing": [],
        }
    )
    assert report["candidate_gates"][candidate]["passes"] is False


def test_v157_blind_primary_is_predeclared_not_metric_selected() -> None:
    assert blind.PRIMARY == "ours_layer_interleaved10_reservoir4"
    assert set(blind.REQUIRED_CONTROLS) == (
        set(vbench.LAYER_CANDIDATES)
        - {blind.PRIMARY}
        | {
            "ours_all_reservoir4_reference",
            "ours_all_recent8_reference",
        }
    )
