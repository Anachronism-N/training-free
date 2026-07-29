from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load():
    path = SCRIPTS / "run_v132_moviebench128_60s.py"
    spec = importlib.util.spec_from_file_location("v132_long60_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v132_long60_is_sf_plus_one_selected_method():
    module = _load()
    runner = module.runner
    assert runner.EXPERIMENT == "v132_moviebench128_60s"
    assert runner.PROMPT_COUNT == 128
    assert runner.NUM_OUTPUT_FRAMES == 240
    assert runner.INCLUDE_PF_BASELINE is False
    assert runner.DEFAULT_CANDIDATES == ("prototype_retrieval1_age24",)
    methods = runner.methods_for(runner.DEFAULT_CANDIDATES)
    assert [method.key for method in methods] == [
        "sf_native",
        "ours_prototype_retrieval1_age24",
    ]


def test_v132_long60_vbench_expects_thirty_two_second_clips():
    splitter = (
        SCRIPTS / "prepare_v129_vbench_splits.py"
    ).read_text(encoding="utf-8")
    wrapper = (
        SCRIPTS / "run_v132_long60_vbench.sh"
    ).read_text(encoding="utf-8")
    assert "clips_per_video = num_output_frames // 8" in splitter
    assert "VBENCH_EXPECTED_NUM_OUTPUT_FRAMES=240" in wrapper
