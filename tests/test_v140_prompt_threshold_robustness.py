import csv
import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v140_prompt_threshold_robustness.py"
)
SPEC = importlib.util.spec_from_file_location("v140_threshold", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_held_out_query_adjusted_threshold_recovers_stable_split(tmp_path):
    source = tmp_path / "head_prompt_job_axes.csv"
    fields = [
        "job_id",
        "layer",
        "head",
        "cphi_score",
        "query_score",
        "native_score",
        "current_key_score",
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in range(4):
            for layer in range(30):
                for head in range(12):
                    signal = 0.0
                    if layer > 0:
                        signal = 0.6 if head < 6 else -0.6
                    query = 0.2 + family * 0.001
                    writer.writerow(
                        {
                            "job_id": f"cf_{family:02d}_identity",
                            "layer": layer,
                            "head": head,
                            "cphi_score": signal + query,
                            "query_score": query,
                            "native_score": 0.0,
                            "current_key_score": 0.0,
                        }
                    )

    rows = MODULE._load_rows(source, expected_jobs=4)
    report = MODULE.analyze(
        rows,
        output_dir=tmp_path / "analysis",
        expected_jobs=4,
    )
    query_report = report["score_reports"]["query_adjusted"]
    assert query_report["discovery_validation_spearman"] == 1.0
    assert query_report["thresholds"]["zero"]["label_agreement"] == 1.0
    assert query_report["thresholds"]["zero"]["validation_positive"] == 174
    assert query_report["zero_threshold_gate"] is True
    assert (tmp_path / "analysis" / "threshold_sweep.csv").is_file()
    assert (tmp_path / "analysis" / "threshold_summary.md").is_file()
