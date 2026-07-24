from pathlib import Path
import importlib.util


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_pf_transition_controls.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_pf_transition_controls",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pf_controls_preserve_intended_classes() -> None:
    source = [[-1, 1, 2, -1, 1, 2]]
    controls = MODULE.build_controls(source)

    assert controls["pf_binary"] == [[-1, 1, -1, -1, 1, -1]]
    assert controls["wave_only"] == [[-1, 0, 0, -1, 0, 0]]
    assert controls["veil_only"] == [[0, 0, -1, 0, 0, -1]]
    assert controls["anchor_only"] == [[0, 1, 0, 0, 1, 0]]
    assert controls["wave_anchor"] == [[-1, 1, 0, -1, 1, 0]]
    assert controls["veil_anchor"] == [[0, 1, -1, 0, 1, -1]]
