from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


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


addons = _load(
    "v129_noncache_addons_for_tests",
    SCRIPTS / "run_v129_noncache_addons.py",
)
prompt_prep = _load(
    "v129_addon_prompt_prep_for_tests",
    SCRIPTS / "prepare_v129_addon_prompts.py",
)
analyzer = _load(
    "v129_addon_analyzer_for_tests",
    SCRIPTS / "analyze_v129_noncache_addons.py",
)


def test_addon_matrix_changes_no_cache_policy():
    assert set(addons.VALUE_VARIANTS) == {
        "value_var_s025",
        "value_var_s050",
        "value_var_s050_mid",
        "value_var_s050_mid_t3",
    }
    cells = (addons.CONTROL_CELL, *addons.ADDON_CELLS)
    assert all(cell.support_policy == "prototype" for cell in cells)
    assert all(cell.suppress_policy == "retrieval1_age24" for cell in cells)
    assert all(cell.max_full_frame_equivalents == 9 for cell in cells)
    assert addons.CONTROL_CELL.variance_refresh is False
    assert all(cell.variance_refresh for cell in addons.ADDON_CELLS)
    assert set(addons.ADDON_CANDIDATE_SPECS) == {
        "value_control",
        *addons.VALUE_VARIANTS,
    }


def test_variant_command_is_supportive_only_and_variance_only(tmp_path):
    base_command = [
        sys.executable,
        "inference.py",
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

    def fake_base(*_args, **_kwargs):
        return list(base_command), tmp_path, tmp_path / "map.csv", 0

    original = addons._base_inference_command
    addons._base_inference_command = fake_base
    try:
        cell = addons.Cell(
            "ours_value_var_s050_mid_t3__p000",
            "test",
            "single",
            support_policy="prototype",
            suppress_policy="retrieval1_age24",
            variance_refresh=True,
        )
        command, _, _, _ = addons.addon_inference_command(
            object(),
            cell=cell,
            output=tmp_path / "videos",
            transition_trace=tmp_path / "transition.jsonl",
            scene_trace=tmp_path / "scene.jsonl",
        )
    finally:
        addons._base_inference_command = original

    assert command[1] == str(addons.TRACE_WRAPPER)
    assert command[command.index("--pyramidkv_history_value_labels") + 1] == "10"
    assert (
        command[command.index("--pyramidkv_history_value_moment_mode") + 1]
        == "variance_only"
    )
    assert (
        command[command.index("--pyramidkv_history_value_target_frames") + 1]
        == "4"
    )
    assert (
        command[
            command.index("--pyramidkv_history_value_transition_lambda") + 1
        ]
        == "3.0"
    )


def test_prompt_index_parser_rejects_duplicates_and_out_of_range():
    assert prompt_prep.parse_indices("0,7,127") == (0, 7, 127)
    for invalid in ("", "0,0", "-1", "128"):
        try:
            prompt_prep.parse_indices(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid indices: {invalid}")


def test_trace_helpers_are_strict():
    assert analyzer.method_from_task("ours_value_control__p000") == (
        "ours_value_control"
    )
    assert analyzer.finite_values([0.1, 0.2]) == [0.1, 0.2]
    try:
        analyzer.finite_values([float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite trace sample was accepted")


def test_master_script_does_not_modify_main_v129_entrypoint():
    source = (SCRIPTS / "run_v129_noncache_addons.sh").read_text(
        encoding="utf-8"
    )
    assert "run_v129_noncache_addons.py" in source
    assert "run_v129_no_pf_10h.sh" not in source
    assert "value_control" in source
    assert "value_var_s050_mid_t3" in source
