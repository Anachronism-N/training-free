import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "package_v136_multi_axis_results.py"
)
SPEC = importlib.util.spec_from_file_location("v136_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_package_is_allowlisted_and_excludes_raw_profiles(tmp_path):
    analysis = tmp_path / "analysis"
    output = tmp_path / "bundle"
    analysis.mkdir()
    report = {
        "method": "v136",
        "recommendation": "dual_axis_prompt_and_temporal",
        "profile_counts": {"observational": 128, "counterfactual": 128},
        "head_count": 360,
        "profile_contract_passed": True,
        "gates": {"prompt_axis": True, "temporal_axis": True},
    }
    for name in MODULE.ANALYSIS_FILES:
        payload = "{}\n" if name.endswith(".json") else "test\n"
        if name == "multi_axis_report.json":
            payload = json.dumps(report)
        (analysis / name).write_text(payload, encoding="utf-8")
    (analysis / "raw.pt").write_bytes(b"raw")
    (analysis / "video.mp4").write_bytes(b"video")

    inventory = MODULE.package_results(analysis, output)
    assert len(inventory["files"]) == len(MODULE.ANALYSIS_FILES)
    assert not (output / "raw.pt").exists()
    assert not (output / "video.mp4").exists()
    assert (output / "bundle_inventory.json").is_file()
    assert (output / "review_bundle.md").is_file()
