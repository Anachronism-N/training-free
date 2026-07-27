from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v116_candidate_metrics.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "v116_candidate_metrics_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load()


def _write_fixture(run_root: Path) -> list[str]:
    methods = ["landmark_recent8", "landmark_motion1"]
    (run_root / "published_manifest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "prompt_count": 16,
                "methods": [{"key": method} for method in methods],
            }
        ),
        encoding="utf-8",
    )
    for method in methods:
        per_video = {}
        for prompt_index in range(16):
            dino = 0.80 + prompt_index * 0.001
            acceleration = 2.0
            if method == "landmark_motion1":
                dino += 0.01
                acceleration = 1.5
            per_video[f"{method}/{prompt_index:06d}"] = {
                "method": method,
                "prompt_index": prompt_index,
                "metrics": {
                    "composite": dino,
                    "m1_dino_consistency": dino,
                    "m3_motion_smoothness": acceleration,
                },
            }
        output = run_root / "metrics" / "auxiliary" / method
        output.mkdir(parents=True)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "per_method": {
                        method: {
                            "composite": 0.9,
                            "m1_dino_consistency": 0.9,
                            "m3_motion_smoothness": acceleration,
                        }
                    },
                    "per_video": per_video,
                }
            ),
            encoding="utf-8",
        )
    return methods


def test_paired_analysis_normalizes_metric_directions(tmp_path):
    methods = _write_fixture(tmp_path)
    loaded_methods, prompt_count = analysis.load_manifest(tmp_path)
    per_prompt, aggregates = analysis.load_auxiliary(
        tmp_path,
        loaded_methods,
        prompt_count,
    )
    rows = analysis.paired_rows(
        per_prompt,
        reference="landmark_recent8",
        bootstrap_samples=200,
    )

    assert loaded_methods == methods
    dino = next(
        row for row in rows
        if row["method"] == "landmark_motion1"
        and row["metric"] == "m1_dino_consistency"
    )
    motion = next(
        row for row in rows
        if row["method"] == "landmark_motion1"
        and row["metric"] == "m3_motion_smoothness"
    )
    assert abs(dino["mean_improvement"] - 0.01) < 1e-9
    assert abs(motion["mean_improvement"] - 0.5) < 1e-9
    assert dino["wins"] == motion["wins"] == 16
    assert aggregates["landmark_recent8"]["composite"] == 0.9


def test_auxiliary_loader_fails_on_missing_prompt(tmp_path):
    methods = _write_fixture(tmp_path)
    path = (
        tmp_path
        / "metrics"
        / "auxiliary"
        / methods[1]
        / "results.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["per_video"].pop(f"{methods[1]}/000015")
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        analysis.load_auxiliary(tmp_path, methods, 16)
    except ValueError as error:
        assert "coverage mismatch" in str(error)
    else:
        raise AssertionError("missing paired prompt must fail closed")


def test_summary_combines_vbench_and_auxiliary_outputs(tmp_path):
    methods = _write_fixture(tmp_path)
    metrics = tmp_path / "metrics"
    (metrics / "vbench_long_summary.json").write_text(
        json.dumps(
            {
                "methods": {
                    method: {
                        dimension: 0.7 + index * 0.01
                        for index, dimension in enumerate(
                            analysis.VBENCH_DIMENSIONS
                        )
                    }
                    for method in methods
                }
            }
        ),
        encoding="utf-8",
    )
    _, aggregates = analysis.load_auxiliary(tmp_path, methods, 16)
    vbench = analysis.load_vbench(tmp_path, methods)
    summaries = analysis.method_summary(
        methods,
        aggregates,
        vbench,
        reference=methods[0],
    )
    output = tmp_path / "analysis"
    analysis.write_csv(output / "summary.csv", summaries)
    analysis.write_markdown(
        output / "summary.md",
        summaries,
        [],
        reference=methods[0],
        has_vbench=True,
    )

    assert summaries[0]["vbench_dynamic_degree"] == 0.75
    assert (output / "summary.csv").is_file()
    assert "Dynamic degree" in (output / "summary.md").read_text(
        encoding="utf-8"
    )
