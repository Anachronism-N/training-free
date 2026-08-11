from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_v120_moviebench32_main as parent  # noqa: E402
import run_v172_relative_depth_moviebench16 as v172  # noqa: E402
from build_v172_relative_depth_maps import MAP_SPECS  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


SELECTED_POLICY = (
    ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
    1,
    4,
    "reservoir_motion",
)
RECENT_POLICY = ((), 1, 8, "stride")


def test_v172_grid_reuses_two_methods_and_generates_seven() -> None:
    v172.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")

    assert tuple(method.key for method in methods) == v172.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 144
    assert len(v172.NEW_METHODS) * v172.PROMPT_COUNT == 112
    assert len(v172.REUSE_METHODS) * v172.PROMPT_COUNT == 32
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [36, 36, 36, 36]


def test_v172_center_third_exactly_reproduces_v166_middle10() -> None:
    assert MAP_SPECS["center_1of3"] == tuple(range(10, 20))
    v172_map = (
        ROOT
        / "configs"
        / "head_maps"
        / "v172_depth_center_1of3_multiscale_motion.csv"
    )
    v157_map = (
        ROOT
        / "configs"
        / "head_maps"
        / "v157_layer_middle10_reservoir.csv"
    )
    v172_bytes = v172_map.read_bytes().replace(b"\r\n", b"\n")
    v157_bytes = v157_map.read_bytes().replace(b"\r\n", b"\n")
    assert v172_bytes == v157_bytes


def test_v172_cells_isolate_layer_allocation() -> None:
    regular = v172.V172_CELLS[:-1]
    all_layer = v172.V172_CELLS[-1]

    for cell in regular:
        assert expected_policy(cell, 10) == SELECTED_POLICY
        assert expected_policy(cell, 11) == RECENT_POLICY
    assert expected_policy(all_layer, 10) == SELECTED_POLICY
    assert expected_policy(all_layer, 11) == SELECTED_POLICY
    assert all_layer.map_key == "center_1of3"


def test_v172_depth_maps_pass_binary_audit() -> None:
    args = SimpleNamespace(
        pf_labels=(
            PF_ROOT / "configs" / "head_configs" / "best_labels.csv"
        )
    )
    manifest, paths, audits = v172.load_depth_maps(args)

    assert set(paths) == set(MAP_SPECS)
    assert set(audits) == set(MAP_SPECS)
    for key, layers in MAP_SPECS.items():
        assert manifest["maps"][key]["selected_layer_count"] == len(layers)
        assert audits[key]["counts"]["10"] == len(layers) * 12
