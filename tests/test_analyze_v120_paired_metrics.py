from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v120_paired_metrics as analysis


def test_extract_vbench_prefers_per_video_rows(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        """
{
  "subject_consistency": [
    0.8,
    [
      {
        "video_path": "/x/000000-0_v120.mp4",
        "inclip_score": 0.9,
        "clip2clip_score": 0.7,
        "mapped_clip2clip_score": 0.8,
        "video_results": 0.85
      },
      {
        "video_path": "/x/000001-0_v120.mp4",
        "inclip_score": 0.8,
        "clip2clip_score": 0.6,
        "mapped_clip2clip_score": 0.7,
        "video_results": 0.75
      }
    ]
  ],
  "aesthetic_quality": [
    0.6,
    [
      {"video_path": "/x/000000-0_v120/000.mp4", "video_results": 0.5},
      {"video_path": "/x/000000-0_v120/001.mp4", "video_results": 0.7},
      {"video_path": "/x/000001-0_v120/000.mp4", "video_results": 0.4},
      {"video_path": "/x/000001-0_v120/001.mp4", "video_results": 0.6}
    ],
    [
      {"video_path": "/x/000000-0_v120.mp4", "video_results": 0.6},
      {"video_path": "/x/000001-0_v120.mp4", "video_results": 0.5}
    ]
  ]
}
""".strip(),
        encoding="utf-8",
    )

    metrics = analysis.extract_vbench(path)

    assert metrics["vbench.subject_consistency"] == {0: 0.85, 1: 0.75}
    assert metrics["vbench.subject_consistency.inclip"] == {0: 0.9, 1: 0.8}
    assert metrics["vbench.subject_consistency.clip2clip"] == {0: 0.7, 1: 0.6}
    assert metrics["vbench.aesthetic_quality"] == {0: 0.6, 1: 0.5}


def test_extract_comprehensive_requires_unique_method_prompt_rows(tmp_path):
    path = tmp_path / "comprehensive.json"
    path.write_text(
        """
{
  "per_video": {
    "ours/a.mp4": {
      "method": "ours",
      "prompt_index": 0,
      "metrics": {"m1_dino_consistency": 0.9}
    },
    "ours/b.mp4": {
      "method": "ours",
      "prompt_index": 0,
      "metrics": {"m1_dino_consistency": 0.8}
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate comprehensive row"):
        analysis.extract_comprehensive(path)


def test_paired_statistics_respects_lower_is_better_direction():
    row = analysis.paired_statistics(
        {0: 0.1, 1: 0.2, 2: 0.3},
        {0: 0.2, 1: 0.3, 2: 0.4},
        direction=-1,
        bootstrap_samples=100,
        permutation_samples=100,
        seed=7,
        label="loop",
    )

    assert row["raw_mean_delta"] == pytest.approx(-0.1)
    assert row["improvement_mean"] == pytest.approx(0.1)
    assert (row["wins"], row["ties"], row["losses"]) == (3, 0, 0)


def test_analyze_builds_every_candidate_reference_pair():
    observations = {
        "sf": {"aux.m1_dino_consistency": {0: 0.7, 1: 0.8}},
        "pf": {"aux.m1_dino_consistency": {0: 0.9, 1: 0.9}},
        "ours": {"aux.m1_dino_consistency": {0: 0.8, 1: 0.85}},
    }

    payload = analysis.analyze(
        observations,
        references=["sf", "pf"],
        candidates=["ours"],
        bootstrap_samples=100,
        permutation_samples=100,
        seed=11,
        expected_prompts=2,
    )

    assert set(payload["comparisons"]) == {
        "ours__vs__sf",
        "ours__vs__pf",
    }


def test_analyze_rejects_incomplete_prompt_coverage():
    observations = {
        "sf": {"aux.m1_dino_consistency": {0: 0.7, 1: 0.8}},
        "ours": {"aux.m1_dino_consistency": {0: 0.8}},
    }

    with pytest.raises(ValueError, match="per-prompt coverage mismatch"):
        analysis.analyze(
            observations,
            references=["sf"],
            candidates=["ours"],
            bootstrap_samples=100,
            permutation_samples=100,
            seed=11,
            expected_prompts=2,
        )
