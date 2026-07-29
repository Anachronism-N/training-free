import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "package_v138_history_intervention_results.py"
)
SPEC = importlib.util.spec_from_file_location("v138_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_package_excludes_raw_descriptors(tmp_path):
    analysis = tmp_path / "analysis"
    output = tmp_path / "bundle"
    analysis.mkdir()
    report = {
        "method": "v138",
        "recommendation": "history_specificity_plus_order_axis",
        "profile_count": 128,
        "head_count": 360,
        "profile_contract_passed": True,
        "maximum_rope_reconstruction_error": 1e-6,
        "gates": {
            "history_specificity": True,
            "order_axis": True,
        },
    }
    for name in MODULE.ANALYSIS_FILES:
        payload = "{}\n" if name.endswith(".json") else "test\n"
        if name == "analysis_report.json":
            payload = json.dumps(report)
        (analysis / name).write_text(payload, encoding="utf-8")
    (analysis / "raw.pt").write_bytes(b"raw")
    (analysis / "head_local_job_axes.csv").write_text(
        "large\n", encoding="utf-8"
    )

    inventory = MODULE.package_results(analysis, output)
    assert len(inventory["files"]) == len(MODULE.ANALYSIS_FILES)
    assert not (output / "raw.pt").exists()
    assert not (output / "head_local_job_axes.csv").exists()
    assert (output / "bundle_inventory.json").is_file()
