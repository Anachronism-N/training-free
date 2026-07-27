from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v120 = _load(
    "v120_moviebench32_runner_no_torch",
    SCRIPTS / "run_v120_moviebench32_main.py",
)


def test_v120_candidate_parser_limits_main_table_scope():
    assert v120.parse_candidate_keys("landmark_motion1") == (
        "landmark_motion1",
    )
    assert v120.parse_candidate_keys(
        "landmark_retrieval1_age24,landmark_retrieval_motion"
    ) == (
        "landmark_retrieval1_age24",
        "landmark_retrieval_motion",
    )
    with pytest.raises(ValueError, match="at most two"):
        v120.parse_candidate_keys(
            "landmark_motion1,prototype_motion1,landmark_retrieval2"
        )
    with pytest.raises(ValueError, match="unknown"):
        v120.parse_candidate_keys("not_a_candidate")


def test_v120_default_table_has_paired_sf_pf_and_ours():
    methods = v120.methods_for(v120.DEFAULT_CANDIDATES)
    assert [(method.key, method.engine) for method in methods] == [
        ("sf_native", "sf"),
        ("pf_native", "pf"),
        ("ours_landmark_motion1", "pf"),
    ]
    assert methods[1].source_cell.native
    assert methods[1].source_cell.map_key == "pf"
    ours = methods[2].source_cell
    assert ours.support_policy == "landmark"
    assert ours.suppress_policy == "motion_pair1"


def test_v120_four_node_partition_is_balanced_and_complete():
    methods = v120.methods_for(v120.DEFAULT_CANDIDATES)
    shards = [
        v120.selected_tasks(methods, node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    identities = [
        (method.key, prompt_index)
        for shard in shards
        for method, prompt_index, _ in shard
    ]

    assert [len(shard) for shard in shards] == [24, 24, 24, 24]
    assert len(identities) == len(set(identities)) == 3 * 32
    assert {
        prompt_index for _, prompt_index in identities
    } == set(range(32))


def test_v120_exposes_v119_promotable_allocations():
    methods = v120.methods_for(
        ("landmark_retrieval_motion", "landmark_motion1_sink3_budget9")
    )
    hybrid = methods[2].source_cell
    sink3 = methods[3].source_cell

    assert hybrid.suppress_policy == "retrieval1_motion1_age24"
    assert hybrid.history_budget_profile == "default"
    assert sink3.support_policy == "landmark"
    assert sink3.suppress_policy == "motion_pair1"
    assert sink3.history_budget_profile == "sink3_budget9"
    assert sink3.max_full_frame_equivalents == 9


def test_v120_vbench_script_requires_32_prompt_manifest_and_six_dimensions():
    text = (SCRIPTS / "run_v120_vbench_long.sh").read_text(encoding="utf-8")
    assert 'int(payload.get("prompt_count", 0)) != 32' in text
    assert "V119_PROMOTION_APPROVED" in text
    for dimension in (
        "subject_consistency",
        "background_consistency",
        "aesthetic_quality",
        "imaging_quality",
        "motion_smoothness",
        "dynamic_degree",
    ):
        assert dimension in text
