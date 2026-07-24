import json
from pathlib import Path

import pytest

from scripts.collect_vbench_long_results import collect
from scripts.analyze_v93_moviebench import HEAD_METHODS, analyze
from scripts.merge_comprehensive_results import merge
from scripts.prepare_blind_review import _videos


def test_collect_vbench_accepts_scalar_and_nested_scores(tmp_path: Path):
    first = tmp_path / "pf"
    second = tmp_path / "ours"
    first.mkdir()
    second.mkdir()
    (first / "results.json").write_text(
        json.dumps(
            {
                "subject_consistency": 0.8,
                "background_consistency": [0.7, {"video": 0.2}],
            }
        ),
        encoding="utf-8",
    )
    (second / "results.json").write_text(
        json.dumps(
            {
                "subject_consistency": {"score": 0.9},
                "background_consistency": {"overall": 0.75},
            }
        ),
        encoding="utf-8",
    )

    result = collect(
        tmp_path,
        ["pf", "ours"],
        ["subject_consistency", "background_consistency"],
        allow_missing=False,
    )

    assert result["methods"]["pf"]["subject_consistency"] == 0.8
    assert result["methods"]["pf"]["background_consistency"] == 0.7
    assert result["methods"]["ours"]["subject_consistency"] == 0.9


def test_merge_comprehensive_enforces_method_and_video_coverage(tmp_path: Path):
    first = tmp_path / "pf.json"
    second = tmp_path / "ours.json"
    first.write_text(
        json.dumps({"per_method": {"pf": {"num_videos": 32, "composite": 0.5}}}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"per_method": {"ours": {"num_videos": 32, "composite": 0.6}}}),
        encoding="utf-8",
    )

    result = merge(
        [first, second],
        expected_methods=["pf", "ours"],
        expected_videos=32,
    )

    assert set(result["per_method"]) == {"pf", "ours"}
    with pytest.raises(ValueError, match="video counts"):
        merge([first], expected_methods=["pf"], expected_videos=128)


def test_blind_review_orders_videos_by_numeric_prompt_index(tmp_path: Path):
    for index in range(12):
        (tmp_path / f"{index}-0_ema.mp4").write_bytes(b"video")

    videos = _videos(tmp_path, 12)

    assert [path.name for path in videos] == [
        f"{index}-0_ema.mp4" for index in range(12)
    ]


def test_head_analysis_applies_causal_and_replica_screen():
    comprehensive = {}
    vbench = {}
    temporal = {}
    for method in HEAD_METHODS:
        base = 0.7
        if method == "prompt_pfcount_read_v78":
            base = 0.8
        elif method == "prompt_replica_read_v78":
            base = 0.795
        elif method == "pf_binary_read_v78":
            base = 0.803
        elif method in {"prompt_inverse_read_v78", "prompt_random_read_v78"}:
            base = 0.6
        comprehensive[method] = {
            "m1_dino_consistency": base,
            "m1_min_stability": base,
            "m2_drift_slope": -0.01,
            "m7_background_consistency": base,
            "composite": base,
        }
        vbench[method] = {
            "subject_consistency": base,
            "background_consistency": base,
            "aesthetic_quality": base,
            "imaging_quality": base,
            "dynamic_degree": 0.5,
        }
        temporal[method] = {"count": 32, "mean": 1.0, "median": 1.0}

    result = analyze(
        mode="head32",
        comprehensive_payload={"per_method": comprehensive},
        temporal=temporal,
        vbench_payload={"methods": vbench},
        traces={},
        label_manifest=None,
    )

    assert result["classification_screen"]["automated_screen_passed"]
