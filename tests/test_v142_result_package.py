import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "package_v142_output_causal_results.py"
)
SPEC = importlib.util.spec_from_file_location("v142_package", SCRIPT)
V142 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V142)


def test_v142_package_copies_only_bounded_review_artifacts(tmp_path):
    analysis = tmp_path / "analysis"
    inputs = tmp_path / "inputs"
    output = tmp_path / "package"
    analysis.mkdir()
    inputs.mkdir()
    for name in V142.ANALYSIS_FILES:
        (analysis / name).write_text(f"{name}\n", encoding="utf-8")
    for name in V142.INPUT_FILES:
        (inputs / name).write_text("{}\n", encoding="utf-8")
    inventory = V142.package(analysis, inputs, output)
    names = {entry["name"] for entry in inventory["files"]}
    assert names == set(V142.ANALYSIS_FILES) | set(V142.INPUT_FILES)
    assert not any(path.suffix == ".pt" for path in output.iterdir())
