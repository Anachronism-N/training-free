from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
RUNNER_PATH = SCRIPTS / "run_v101_paper_ablation_4node.py"
spec = importlib.util.spec_from_file_location(
    "v101_paper_ablation_runner", RUNNER_PATH
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def candidate_args(**overrides):
    values = {
        "candidate_support": "stride",
        "candidate_suppress": "motion_cyclic",
        "candidate_transition": True,
        "threshold_control": "m0p1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_paper_ablation_builds_exactly_eight_unique_methods():
    methods = runner.build_methods(candidate_args())
    signatures = {
        (
            method.map_key,
            method.support_policy,
            method.suppress_policy,
            method.transition,
        )
        for method in methods
    }

    assert len(methods) == 8
    assert len({method.name for method in methods}) == 8
    assert len(signatures) == 8
    assert methods[0].name == "ours_full"


def test_candidate_toggles_change_only_the_intended_factor():
    methods = {
        method.name: method for method in runner.build_methods(candidate_args())
    }
    full = methods["ours_full"]
    transition = methods["ablate_transition_toggle"]
    support = methods["ablate_support_route_toggle"]

    assert transition.map_key == full.map_key
    assert transition.support_policy == full.support_policy
    assert transition.suppress_policy == full.suppress_policy
    assert transition.transition is not full.transition

    assert support.map_key == full.map_key
    assert support.suppress_policy == full.suppress_policy
    assert support.transition == full.transition
    assert support.support_policy != full.support_policy


def test_route_replacements_never_duplicate_selected_route():
    for selected in runner.ALLOWED_SUPPRESS_POLICIES:
        methods = runner.build_methods(
            candidate_args(candidate_suppress=selected)
        )
        route_cells = [
            method
            for method in methods
            if method.name.startswith("ablate_responsive_")
        ]

        assert len(route_cells) == 2
        assert len({method.suppress_policy for method in route_cells}) == 2
        assert all(
            method.suppress_policy != selected for method in route_cells
        )


def test_membership_controls_keep_candidate_cache_routes():
    methods = {
        method.name: method for method in runner.build_methods(candidate_args())
    }
    full = methods["ours_full"]

    for name, map_key in (
        ("control_random_membership", "random"),
        ("control_pf_aw_membership", "pf_aw"),
        ("control_threshold_m0p1", "threshold_m0p1"),
    ):
        control = methods[name]
        assert control.map_key == map_key
        assert control.support_policy == full.support_policy
        assert control.suppress_policy == full.suppress_policy
        assert control.transition == full.transition


def test_four_frozen_node_intervals_cover_moviebench_128_once():
    intervals = [(rank * 32, (rank + 1) * 32) for rank in range(4)]
    indices = [
        index
        for start, end in intervals
        for index in range(start, end)
    ]

    assert indices == list(range(128))


def test_method_policy_cell_preserves_runtime_route_contract():
    full = runner.build_methods(candidate_args())[0]
    cell = full.policy_cell()

    assert cell.support_policy == "stride"
    assert cell.suppress_policy == "motion_cyclic"
    assert cell.transition
    assert cell.uses_motion


def test_inference_command_freezes_30s_reseed_and_exclusive_role_route(tmp_path):
    full = runner.build_methods(candidate_args())[0]
    args = SimpleNamespace(
        pf_config=tmp_path / "pf.yaml",
        pf_checkpoint=tmp_path / "model.pt",
        prompts=tmp_path / "prompts.txt",
    )
    command = runner.inference_command(
        args,
        method=full,
        head_map=tmp_path / "heads.csv",
        output=tmp_path / "videos",
        start=32,
        end=64,
        transition_trace=tmp_path / "transition.jsonl",
    )

    assert command[command.index("--num_output_frames") + 1] == "120"
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--start_idx") + 1] == "32"
    assert command[command.index("--end_idx") + 1] == "64"
    assert "--reseed_per_prompt" in command
    assert "--pyramidkv_history_polarity" in command
    assert (
        command[command.index("--pyramidkv_history_support_policy") + 1]
        == "stride"
    )
    assert (
        command[command.index("--pyramidkv_history_suppress_policy") + 1]
        == "motion_cyclic"
    )
    assert "--pyramidkv_cache_transition" in command
