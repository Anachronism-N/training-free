import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "package_v141_full_prompt_switch_results.py"
)
SPEC = importlib.util.spec_from_file_location("v141_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v141_package_excludes_state_observations(tmp_path):
    analysis = tmp_path / "analysis"
    output = tmp_path / "bundle"
    analysis.mkdir()
    report = {
        "method": "v141",
        "recommendation": "candidate",
        "profile_count": 32,
        "profile_contract_passed": True,
        "gates": {"full_prompt_switch_axis": True},
    }
    for name in MODULE.ANALYSIS_FILES:
        payload = "{}\n" if name.endswith(".json") else "test\n"
        if name == "analysis_report.json":
            payload = json.dumps(report)
        (analysis / name).write_text(payload, encoding="utf-8")
    (analysis / "state_observations.csv").write_text(
        "large\n", encoding="utf-8"
    )

    inventory = MODULE.package_results(analysis, output)
    assert len(inventory["files"]) == len(MODULE.ANALYSIS_FILES)
    assert not (output / "state_observations.csv").exists()
    assert (output / "bundle_inventory.json").is_file()
