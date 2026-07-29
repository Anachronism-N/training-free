from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_v132_binary_memory_ablation as v132
from run_v100_fast_selection_1video import expected_policy, read_matrix


def test_v132_method_tiers_and_screen_are_frozen():
    assert v132.TIER1_METHODS == (
        "random_binary",
        "all_supportive",
        "no_prototype",
        "no_retrieval",
    )
    assert len(v132.ALL_METHODS) == 6
    assert len(v132.SCREEN16) == 16
    assert len(set(v132.SCREEN16)) == 16
    assert min(v132.SCREEN16) == 0
    assert max(v132.SCREEN16) == 124


def test_v132_prompt_index_parser_supports_ranges():
    assert v132.parse_prompt_indices("0,4-6,127") == (0, 4, 5, 6, 127)


def test_v132_control_maps_are_deterministic_and_counted(tmp_path):
    legacy = (
        ROOT
        / "configs"
        / "head_maps"
        / "legacy_v98_absolute_sign_304_56.csv"
    )
    paths, audits = v132.build_control_maps(tmp_path, legacy)
    first_random = paths["random"].read_bytes()
    paths2, audits2 = v132.build_control_maps(tmp_path, legacy)
    assert paths2["random"].read_bytes() == first_random
    assert audits2 == audits

    expected = {
        "legacy": {10: 304, 11: 56},
        "random": {10: 304, 11: 56},
        "inverted": {10: 56, 11: 304},
        "all_supportive": {10: 360},
        "all_suppressive": {11: 360},
    }
    for key, counts in expected.items():
        matrix = read_matrix(paths[key], {10, 11})
        assert Counter(value for row in matrix for value in row) == counts


def test_v132_cells_have_valid_nine_ffe_routes():
    for key, cell in v132.CONTROL_CELLS.items():
        labels = {
            value
            for row in (
                [[10] * 12] * 30
                if key == "all_supportive"
                else [[11] * 12] * 30
                if key == "all_suppressive"
                else read_matrix(
                    ROOT
                    / "configs"
                    / "head_maps"
                    / "legacy_v98_absolute_sign_304_56.csv",
                    {10, 11},
                )
            )
            for value in row
        }
        for label in labels:
            _, sink, recent, _ = expected_policy(cell, label)
            middle = 4 if label == 10 and cell.support_policy == "prototype" else 0
            middle = (
                1
                if label == 11
                and cell.suppress_policy == "retrieval1_age24"
                else middle
            )
            assert sink + middle + recent <= 9
