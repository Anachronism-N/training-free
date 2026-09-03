from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = load_module(
    "v202_v183_metric_correction",
    SCRIPTS / "analyze_v202_v183_metric_correction.py",
)


def test_corrected_dynamic_loader_accepts_torchvision_raft_shape(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "PROMPT_COUNT", 4)
    monkeypatch.setattr(module, "CLIPS_PER_VIDEO", 3)
    for method in module.METHODS:
        directory = tmp_path / method / "dynamic_degree"
        directory.mkdir(parents=True)
        legacy = []
        continuous = []
        for prompt in range(4):
            for clip in range(3):
                path = f"/split_clip/{prompt:06d}-0/{prompt:06d}-0_{clip:03d}.mp4"
                legacy.append({"video_path": path, "video_results": True})
                continuous.append(
                    {
                        "video_path": f"{path}/{prompt:06d}-0.mp4",
                        "video_results": 1.0,
                    }
                )
        (directory / "fixture_eval_results.json").write_text(
            json.dumps({"dynamic_degree": [1.0, legacy, continuous]}),
            encoding="utf-8",
        )
    values, provenance = module.load_corrected_dynamic(tmp_path)
    assert values == {
        method: {prompt: 1.0 for prompt in range(4)} for method in module.METHODS
    }
    assert all(row["all_one"] for row in provenance.values())
    assert all(row["clip_count"] == 12 for row in provenance.values())


def _raw_score(value: float) -> dict:
    return {
        "subject_consistency": value,
        "background_consistency": value,
        "temporal_flickering": value,
        "motion_smoothness": value,
        "aesthetic_quality": value,
        "imaging_quality": value,
        "dynamic_degree": 1.0,
        "overall_consistency": value,
    }


def test_corrected_analysis_rejects_false_v183_gain() -> None:
    manifest = {
        "experiment": module.SOURCE_EXPERIMENT,
        "prompt_count": module.PROMPT_COUNT,
        "methods": [{"key": method} for method in module.METHODS],
    }
    summary = {
        "methods": {
            method: _raw_score(0.90 if method == "sf_native" else 0.89)
            for method in module.METHODS
        },
        "missing": [],
    }
    rows = {}
    for method in module.METHODS:
        offset = 0.0 if method == "sf_native" else -0.01
        for prompt in range(module.PROMPT_COUNT):
            rows[(method, prompt)] = {
                "official_quality_score": 85.0 + offset,
                "identity_background": 0.90 + offset,
                "temporal_mechanics": 0.95 + offset,
                "semantic_alignment": 0.55 + offset,
                "visual_quality": 0.70 + offset,
                "dynamic_degree": 1.0,
            }
    provenance = {method: {"all_one": True} for method in module.METHODS}
    report = module.analyze(
        manifest,
        summary,
        rows,
        dynamic_provenance=provenance,
    )
    assert report["recommendation"] == (
        "no_v183_method_improves_sf_after_dynamic_correction"
    )
    assert report["passing_candidates"] == []
    assert report["corrected_dynamic_degree"]["used_for_method_ranking"] is False
    assert report["manual_review_required"] is False


def test_v202_runner_is_zero_gpu_and_keeps_old_results_immutable() -> None:
    runner = (SCRIPTS / "run_v202_v183_metric_correction.sh").read_text(
        encoding="utf-8"
    )
    assert "CUDA_VISIBLE_DEVICES" not in runner
    assert "analyze_v202_v183_metric_correction.py" in runner
    assert "old_raft_broken" not in runner
    assert "rm -" not in runner
