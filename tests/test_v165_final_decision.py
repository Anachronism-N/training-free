from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v165_final_decision as final  # noqa: E402
import prepare_v165_minimal_review as review  # noqa: E402
import run_v165_vbench_long as vbench  # noqa: E402
from prepare_v165_vbench_comparison import (  # noqa: E402
    DIMENSIONS,
    METHODS,
    PROMPT_COUNT,
)
from v165_decision_contract import (  # noqa: E402
    DIRECTION_FRESH,
    DIRECTION_MATCH,
    PRIMARY,
    SF,
    STATE_MOTION,
    TIE_003,
)


GROUP_VALUES = {
    SF: (0.700, 0.800, 0.650, 0.600),
    DIRECTION_MATCH: (0.705, 0.805, 0.655, 10 / 15),
    TIE_003: (0.700, 0.802, 0.651, 0.600),
    PRIMARY: (0.704, 0.807, 0.654, 10 / 15),
    DIRECTION_FRESH: (0.702, 0.803, 0.653, 0.600),
    STATE_MOTION: (0.703, 0.804, 0.652, 0.600),
}


def dimension_row(method: str) -> dict[str, float]:
    history, temporal, visual, dynamic = GROUP_VALUES[method]
    return {
        "subject_consistency": history,
        "background_consistency": history,
        "temporal_flickering": temporal,
        "motion_smoothness": temporal,
        "overall_consistency": history,
        "dynamic_degree": dynamic,
        "aesthetic_quality": visual,
        "imaging_quality": visual,
        "temporal_style": temporal,
    }


def summary_payload() -> dict:
    return {
        "version": 1,
        "experiment": "v165_direction_stale_tie_vbench16",
        "methods": {method: dimension_row(method) for method in METHODS},
        "dimensions": list(DIMENSIONS),
        "sources": {},
        "missing": [],
    }


def write_vbench_parts(root: Path) -> None:
    for method in METHODS:
        row = dimension_row(method)
        for dimension in DIMENSIONS:
            records = []
            detail_value = row[dimension]
            if dimension == "imaging_quality":
                detail_value *= 100.0
            for prompt in range(PROMPT_COUNT):
                for clip in range(final.CLIPS_PER_VIDEO):
                    video_result = detail_value
                    if dimension == "dynamic_degree":
                        video_result = clip < round(
                            detail_value * final.CLIPS_PER_VIDEO
                        )
                    records.append(
                        {
                            "video_path": (
                                f"/split/{prompt:06d}-0_{clip:03d}.mp4/video.mp4"
                            ),
                            "video_results": video_result,
                        }
                    )
            target = root / method / dimension / "results.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({dimension: [row[dimension], records]}),
                encoding="utf-8",
            )


def write_temporal(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "prompt_index", *final.TEMPORAL_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            for prompt in range(PROMPT_COUNT):
                writer.writerow(
                    {
                        "method": method,
                        "prompt_index": prompt,
                        "late_motion_ratio": 1.0,
                        "temporal_jump": 1.0,
                        "appearance_outlier_fraction": 0.02,
                        "flow_accel_outlier_fraction": 0.05,
                        "dark_frame_fraction": 0.0,
                        "bright_frame_fraction": 0.0,
                        "low_contrast_frame_fraction": 0.0,
                        "edge_density_outlier_fraction": 0.01,
                    }
                )


def write_comprehensive(path: Path) -> None:
    per_video = {}
    for method in METHODS:
        history, temporal, _, _ = GROUP_VALUES[method]
        for prompt in range(PROMPT_COUNT):
            per_video[f"{method}/{prompt}"] = {
                "method": method,
                "prompt_index": prompt,
                "prompt": f"Synthetic prompt {prompt}",
                "metrics": {
                    "m1_dino_consistency": history,
                    "m1_first_last_gap": 1.0 - history,
                    "m1_min_stability": history,
                    "m3_motion_smoothness": 1.0 - temporal,
                    "m5_max_flicker": 1.0 - temporal,
                    "m5_temporal_flickering": 1.0 - temporal,
                    "m6_clip_text_alignment": 0.30,
                    "m7_background_consistency": history,
                    "m7_background_drift": 1.0 - history,
                    "m8_loop_score": 0.01,
                },
            }
    path.write_text(
        json.dumps({"per_video": per_video}),
        encoding="utf-8",
    )


def write_contract_inputs(root: Path) -> tuple[Path, Path]:
    trace = root / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "experiment": "v165_direction_stale_tie_trace",
                "mechanism_gate": True,
                "methods": {
                    PRIMARY: {
                        "aggregate": {
                            "mechanism_gate": True,
                            "changed_count": 57,
                            "contract_failure_count": 0,
                            "read_budget_violation_count": 0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    published = root / "published_manifest.json"
    published.write_text(
        json.dumps(
            {
                "experiment": "v165_direction_stale_tie_moviebench16",
                "ok": True,
                "prompt_count": PROMPT_COUNT,
                "methods": [
                    {"key": method, "indexed_video_count": PROMPT_COUNT}
                    for method in METHODS
                ],
            }
        ),
        encoding="utf-8",
    )
    return trace, published


def test_vbench_collect_analysis_exposes_compatibility_gate() -> None:
    report = vbench.analyze(summary_payload())
    assert report["development_candidate_gate"] is True
    assert report["metric_promotion_gate"] is True
    assert report["primary_candidate"] == PRIMARY
    assert len(report["development_gates"]) == 8
    assert "compatibility field" in report["claim_boundary"]


def test_final_decision_and_four_video_review(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    write_vbench_parts(parts)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(summary_payload()), encoding="utf-8")
    temporal = tmp_path / "temporal.csv"
    write_temporal(temporal)
    comprehensive = tmp_path / "comprehensive.json"
    write_comprehensive(comprehensive)
    trace, published = write_contract_inputs(tmp_path)
    output = tmp_path / "analysis" / "decision.json"
    args = argparse.Namespace(
        vbench_parts_root=parts,
        vbench_summary=summary,
        temporal_csv=temporal,
        comprehensive_json=comprehensive,
        trace_report=trace,
        published_manifest=published,
        output=output,
    )
    report = final.analyze(args)
    assert report["development_candidate_gate"] is True
    assert report["recommendation"] == "targeted_review_then_heldout_confirmation"
    assert report["candidate_specific_safety_flags"] == []
    assert report["review_plan"]["video_count"] == 4
    assert report["review_plan"]["prompt_count"] == 2
    assert report["vbench_detail_scale_factors"][PRIMARY]["imaging_quality"] == 0.01

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report), encoding="utf-8")
    run_root = tmp_path / "run"
    for row in report["review_plan"]["rows"]:
        for method in (PRIMARY, DIRECTION_MATCH):
            source = (
                run_root
                / "published_indexed"
                / method
                / f"{row['prompt_index']:06d}-0_v165.mp4"
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{method}:{row['prompt_index']}".encode())
    review_root = tmp_path / "review"
    manifest = review.prepare(
        argparse.Namespace(
            decision=output,
            run_root=run_root,
            output_root=review_root,
        )
    )
    assert manifest["ok"] is True
    assert manifest["video_count"] == 4
    assert review.prepare(
        argparse.Namespace(
            decision=output,
            run_root=run_root,
            output_root=review_root,
        )
    ) == manifest
    sheet_path = review_root / "reviewer" / "review_sheet.csv"
    with sheet_path.open(
        encoding="utf-8", newline=""
    ) as handle:
        sheet = list(csv.DictReader(handle))
    assert len(sheet) == 4
    assert all("method" not in row for row in sheet)
    sheet[0]["overall_preference_-2_to_2"] = "2"
    with sheet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(sheet)
    assert review.prepare(
        argparse.Namespace(
            decision=output,
            run_root=run_root,
            output_root=review_root,
        )
    ) == manifest
    with sheet_path.open(encoding="utf-8", newline="") as handle:
        preserved = list(csv.DictReader(handle))
    assert preserved[0]["overall_preference_-2_to_2"] == "2"
    key = json.loads(
        (review_root / "private" / "blind_key.json").read_text(encoding="utf-8")
    )
    assert {row["method"] for row in key["rows"]} == {
        PRIMARY,
        DIRECTION_MATCH,
    }
