from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v158_blind_review as blind  # noqa: E402
import analyze_v158_vbench as analysis  # noqa: E402
import prepare_v158_vbench_comparison as vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v158_interleaved_budget_moviebench16 as v158  # noqa: E402
from build_v158_interleaved_budget_maps import (  # noqa: E402
    MAP_SPECS,
    build_manifest,
)
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


def test_v158_maps_are_strictly_nested_and_freeze_v157_reference() -> None:
    manifest = build_manifest()
    assert manifest["nested"] is True
    assert [len(layers) for layers in MAP_SPECS.values()] == [6, 8, 10, 12]
    sets = [set(layers) for layers in MAP_SPECS.values()]
    assert all(left < right for left, right in zip(sets, sets[1:]))
    assert MAP_SPECS["interleaved10"] == (
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
    )
    assert [
        row["selected_head_count"] for row in manifest["maps"].values()
    ] == [72, 96, 120, 144]


def test_v158_grid_generates_48_reuses_80_and_shards_evenly() -> None:
    v158.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v158.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 128
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [32, 32, 32, 32]
    assert len(v158.NEW_METHODS) * v158.PROMPT_COUNT == 48
    assert len(v158.REUSE_METHODS) * v158.PROMPT_COUNT == 80


def test_v158_budget_candidates_change_only_layer_membership() -> None:
    for cell in v158.V158_CELLS[:4]:
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


def test_v158_metric_gate_is_only_for_predeclared_interleaved8() -> None:
    rows = _synthetic_rows()
    rows[analysis.ALL_RESERVOIR]["temporal_flickering"] = 0.68
    rows[analysis.ALL_RESERVOIR]["motion_smoothness"] = 0.68
    rows[vbench.PRIMARY]["dynamic_degree"] = 0.73
    payload = {
        "methods": rows,
        "dimensions": list(vbench.CORE_EVALUATION_DIMENSIONS),
        "missing": [],
    }
    report = analysis.analyze(payload)
    assert report["primary_confirmation_gate"] is True
    assert "candidate_gates" not in report

    rows[vbench.PRIMARY]["dynamic_degree"] = (
        rows[analysis.RECENT]["dynamic_degree"] + 0.01
    )
    assert analysis.analyze(payload)["primary_confirmation_gate"] is False


def test_v158_launch_authorization_requires_passed_v157_blind(
    tmp_path: Path, monkeypatch
) -> None:
    report_path = tmp_path / "v157_blind_review_report.json"
    monkeypatch.setenv("V158_V157_BLIND_REPORT", str(report_path))
    assert v158.load_blind_authorization()["ready"] is False
    report_path.write_text(
        json.dumps(
            {
                "experiment": "v157_layer_gated_moviebench16_blind_review",
                "primary": "ours_layer_interleaved10_reservoir4",
                "prompt_count": 16,
                "human_promotion_gate": True,
            }
        ),
        encoding="utf-8",
    )
    assert v158.load_blind_authorization()["ready"] is True


def test_v158_blind_gate_uses_only_preregistered_promotion_controls() -> None:
    assert blind.PROMOTION_CONTROLS == (
        "ours_interleaved10_reservoir4_reference",
        "ours_all_recent8_reference",
    )
    assert set(blind.CONTEXTUAL_CONTROLS) == {
        "ours_middle10_reservoir4_reference",
        "ours_all_reservoir4_reference",
    }
    assert vbench.PRIMARY not in blind.COMPARATORS


def test_v158_blind_gate_accepts_ties_and_rejects_severe_failures(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        blind.base,
        "bootstrap_mean_ci",
        lambda values, *, seed, samples=5000: (0.0, 0.0),
    )
    rows = []
    for method in vbench.METHODS:
        for prompt in range(vbench.PROMPT_COUNT):
            rows.append(
                {
                    "method": method,
                    "prompt_index": prompt,
                    **{column: 0.0 for column in blind.base.RATING_COLUMNS},
                    blind.base.SEVERE_COLUMN: 0,
                }
            )
    report = blind.analyze(rows)
    assert report["human_promotion_gate"] is True
    primary_rows = [row for row in rows if row["method"] == vbench.PRIMARY]
    primary_rows[0][blind.base.SEVERE_COLUMN] = 1
    primary_rows[1][blind.base.SEVERE_COLUMN] = 1
    report = blind.analyze(rows)
    assert report["human_gate_checks"]["primary_severe_failures"] is False
    assert report["human_promotion_gate"] is False
