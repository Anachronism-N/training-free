from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_v207_context_budget_phase_screen as v207_inputs  # noqa: E402
import analyze_v208_paper_confirmation as analysis  # noqa: E402
import prepare_v208_paper_confirmation as v208  # noqa: E402


def write_prompts(path: Path) -> None:
    path.write_text(
        "\n".join(f"Qwen rewritten MovieGen prompt {index}" for index in range(128))
        + "\n",
        encoding="utf-8",
    )


def v207_artifacts(
    tmp_path: Path, *, selected: list[str], parity_pass: bool = True
) -> tuple[Path, Path, Path, Path]:
    prompts = tmp_path / "moviegen128.txt"
    write_prompts(prompts)
    v207_root = tmp_path / "v207_inputs"
    v207_inputs.prepare(prompts, v207_root)
    comparison = tmp_path / "comparison_manifest.json"
    comparison.write_text(
        json.dumps({"experiment": "v207_context_budget_phase_vbench_screen32"}),
        encoding="utf-8",
    )
    report = tmp_path / "v207_report.json"
    report.write_text(
        json.dumps(
            {
                "experiment": "v207_context_budget_phase_vbench_screen32",
                "recommendation": (
                    "advance_best_equal_budget_retrieval_to_fresh128"
                    if selected
                    else "do_not_advance_v207_no_sf_and_recent_gain"
                ),
                "selected_for_fresh128": selected,
                "manual_review_required_for_decision": False,
                "source": {
                    "comparison_manifest": str(comparison.resolve()),
                    "comparison_manifest_sha256": v207_inputs.sha256(comparison),
                },
            }
        ),
        encoding="utf-8",
    )
    parity = tmp_path / "parity_report.json"
    parity.write_text(
        json.dumps(
            {
                "experiment": "v207_sf_runtime_parity",
                "parity_pass": parity_pass,
                "decision": (
                    "parity_pass_proceed_to_budget_phase_screen"
                    if parity_pass
                    else "stop_fix_adaptive_recent21_before_large_scale"
                ),
            }
        ),
        encoding="utf-8",
    )
    return prompts, v207_root / "manifest.json", report, parity


def test_v208_freezes_promoted_method_for_30_and_60_seconds(tmp_path: Path) -> None:
    prompts, v207_manifest, report, parity = v207_artifacts(
        tmp_path, selected=["retrieval21_early2"]
    )
    output = tmp_path / "v208_inputs"
    payload = v208.prepare(prompts, v207_manifest, report, parity, output)
    verified = v208.verify(output / "manifest.json")

    assert verified == payload
    assert payload["method_order"] == ["sf_native", "recent21", "ours"]
    assert payload["frozen_parent_method"] == "retrieval21_early2"
    assert payload["methods"]["ours"]["schedule"] == "early2"
    assert payload["methods"]["ours"]["coverage_noisy_calls"] == [0, 1]
    assert [row["num_output_frames"] for row in payload["scopes"]] == [120, 240]
    assert [row["decoded_video_contract"]["frames"] for row in payload["scopes"]] == [477, 957]
    assert sum(
        row["selection_split"] == "v208_holdout96"
        for row in payload["prompt_items"]
    ) == 96


def test_v208_fails_closed_without_v207_promotion(tmp_path: Path) -> None:
    prompts, v207_manifest, report, parity = v207_artifacts(tmp_path, selected=[])
    with pytest.raises(ValueError, match="automatically promoted"):
        v208.prepare(
            prompts, v207_manifest, report, parity, tmp_path / "v208_inputs"
        )


def test_v208_fails_closed_without_numerical_parity(tmp_path: Path) -> None:
    prompts, v207_manifest, report, parity = v207_artifacts(
        tmp_path, selected=["retrieval21_early2"], parity_pass=False
    )
    with pytest.raises(ValueError, match="numerical SF parity"):
        v208.prepare(
            prompts, v207_manifest, report, parity, tmp_path / "v208_inputs"
        )


def test_v208_runner_is_paper_aligned_and_hard_gated() -> None:
    runner = (SCRIPTS / "run_v208_paper_confirmation_32gpu.sh").read_text(
        encoding="utf-8"
    )
    preparer = (SCRIPTS / "prepare_v208_paper_confirmation.py").read_text(
        encoding="utf-8"
    )

    assert "generate30|generate60" in runner
    assert "NODE_RANK" in runner and "NUM_NODES" in runner
    assert "--v207-report" in runner
    assert "--parity-report" in runner
    assert "--pyramidkv_cache_compatibility_read_budget_frames 21" in runner
    assert "selected_for_fresh128" in preparer
    assert "advance_best_equal_budget_retrieval_to_fresh128" in preparer


def test_v208_confirmatory_gate_requires_sf_and_recent_gain() -> None:
    manifest = {
        "experiment": "v208_main30_vbench_long",
        "scope": "main30",
        "num_output_frames": 120,
        "claim_boundary": "unit-test boundary",
        "prompt_items": [
            {
                "index": index, "source_index": index,
                "selection_split": (
                    "v207_development32" if index in v207_inputs.SOURCE_INDICES else "v208_holdout96"
                ),
            }
            for index in range(128)
        ],
    }
    values = {
        "sf_native": (80.00, 0.9700, 0.9800, 0.240, 0.650),
        "recent21": (80.05, 0.9702, 0.9801, 0.241, 0.651),
        "ours": (80.45, 0.9710, 0.9810, 0.246, 0.656),
    }
    rows = {}
    for method, (quality, identity, temporal, semantic, visual) in values.items():
        for prompt in range(128):
            rows[(method, prompt)] = {
                "quality_without_dynamic_degree": quality,
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": semantic,
                "visual_quality": visual,
                "dynamic_degree": 1.0,
            }
    rows_by_window = {window: rows for window in ("full", "early_half", "late_half")}
    temporal_defaults = {
        feature: 0.0 for feature in analysis.v190.TEMPORAL_FEATURES
    }
    temporal_defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal_rows = {
        (method, prompt): dict(temporal_defaults)
        for method in v208.METHOD_ORDER
        for prompt in range(128)
    }
    report = analysis.analyze_from_rows(manifest, rows_by_window, temporal_rows)

    assert report["paper_confirmation_pass"] is True
    assert report["sf_positive_support"]["pass"] is True
    assert report["recent21_positive_support"]["pass"] is True
    assert report["recommendation"] == "accept_v208_main30_paper_result"
    assert report["primary_prompt_count"] == 96
    assert report["cohorts"]["all128"]["prompt_count"] == 128
    assert report["historically_fresh_prompts_verified"] is False

    # A large development-set gain cannot rescue a candidate that ties Recent
    # on every held-out prompt, even if its all128 mean remains positive.
    for prompt in range(128):
        if prompt not in v207_inputs.SOURCE_INDICES:
            rows[("ours", prompt)] = dict(rows[("recent21", prompt)])
    report = analysis.analyze_from_rows(manifest, rows_by_window, temporal_rows)
    assert report["paper_confirmation_pass"] is False
    assert report["recent21_positive_support"]["pass"] is False


def test_v208_vbench_pipeline_has_full_late_and_motion_checks() -> None:
    shell = (SCRIPTS / "run_v208_vbench_long.sh").read_text(encoding="utf-8")
    analyzer = (SCRIPTS / "analyze_v208_paper_confirmation.py").read_text(
        encoding="utf-8"
    )
    preparer = (SCRIPTS / "prepare_v208_vbench_comparison.py").read_text(
        encoding="utf-8"
    )

    assert "SCOPE=main30|long60" in shell
    assert "compute_temporal_jump_diagnostic.py" in shell
    assert "run_v193_camera_motion.sh" in shell
    assert '"late_half"' in analyzer
    assert "quality_without_dynamic_degree" in analyzer
    assert "Dynamic" in preparer and "excluded" in preparer


def test_v208_rejects_swapped_split_membership(tmp_path: Path) -> None:
    paths = v207_artifacts(tmp_path, selected=["retrieval21_early1"])
    output = tmp_path / "v208_inputs"
    payload = v208.prepare(*paths, output)
    items = payload["prompt_items"]
    items[0]["selection_split"], items[1]["selection_split"] = (
        items[1]["selection_split"], items[0]["selection_split"]
    )
    (output / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split drift"):
        v208.verify(output / "manifest.json")


def test_runtime_binding_rejects_checkpoint_and_code_drift(tmp_path: Path, monkeypatch) -> None:
    import bind_v208_runtime as runtime

    manifest = tmp_path / "input.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime, "verify", lambda _: {"source": {
        "sf_parity_report_sha256": "report",
        "sf_parity_report": str(tmp_path / "parity" / "report" / "parity_report.json"),
    }})
    hashes = {"src/lifecycle_kv/parity_trace.py": "source-hash"}
    monkeypatch.setattr(runtime, "runtime_hashes", lambda _: dict(hashes))
    artifact_paths = {}
    for key in ("checkpoint", "sf_config", "pf_adaptive_config"):
        artifact_paths[key] = tmp_path / key
        artifact_paths[key].write_text(key, encoding="utf-8")
    parity_path = tmp_path / "parity" / "inputs" / "manifest.json"
    parity_path.parent.mkdir(parents=True)
    parity_path.write_text(json.dumps({
        "experiment": "v207_sf_runtime_parity", "runtime_paths": hashes,
        "run_order": ["sf_native_a"],
        "artifacts": {key: {"sha256": runtime.sha256(path)} for key, path in artifact_paths.items()},
    }), encoding="utf-8")
    meta_path = tmp_path / "parity" / "traces" / "sf_native_a" / "trace_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps({"contract_sha256": runtime.sha256(parity_path)}), encoding="utf-8")
    output = tmp_path / "runtime.json"
    args = (manifest, parity_path, tmp_path, artifact_paths["checkpoint"],
            artifact_paths["sf_config"], artifact_paths["pf_adaptive_config"], output)
    first = runtime.bind(*args)
    assert runtime.bind(*args) == first
    hashes["src/lifecycle_kv/parity_trace.py"] = "changed"
    with pytest.raises(ValueError, match="runtime changed"):
        runtime.bind(*args)
    hashes["src/lifecycle_kv/parity_trace.py"] = "source-hash"
    artifact_paths["checkpoint"].write_text("different checkpoint", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint stamp changed"):
        runtime.bind(*args)


@pytest.mark.parametrize("scope,frames", [("main30", 120), ("long60", 240)])
def test_v208_split_accepts_both_frozen_durations(tmp_path: Path, scope: str, frames: int) -> None:
    import prepare_v208_vbench_splits as splitter

    manifest = {
        "experiment": f"v208_{scope}_vbench_long", "scope": scope,
        "prompt_count": 128, "num_output_frames": frames,
        "methods": [{"key": key} for key in v208.METHOD_ORDER],
    }
    (tmp_path / "comparison_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    splitter.configure(tmp_path)
    assert splitter.base.NUM_OUTPUT_FRAMES == frames
    manifest["num_output_frames"] = 99
    (tmp_path / "comparison_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        splitter.configure(tmp_path)
