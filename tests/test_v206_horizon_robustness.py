from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_v205_fixture(tmp_path: Path, module) -> tuple[Path, Path]:
    helper = load_module(
        "v205_test_helper", ROOT / "tests" / "test_v205_horizon_confirmation.py"
    )
    v205_prepare = load_module(
        "v205_prepare_for_v206", SCRIPTS / "prepare_v205_horizon_confirmation.py"
    )
    v201_inputs = helper.build_v201_fixture(tmp_path, v205_prepare)
    output = tmp_path / "v205_inputs"
    frozen = v205_prepare.prepare(*v201_inputs, output)
    input_path = output / "manifest.json"
    candidate = frozen["selected_v201_candidates"][0]

    published = tmp_path / "v205_published.json"
    generation = tmp_path / "v205_generation.json"
    write_json(published, {"ok": True})
    write_json(generation, {"ok": True})
    comparison = {
        "experiment": "v205_profile_disjoint_head_phase_horizon_vbench",
        "confirmatory": True,
        "prompt_count": 128,
        "selected_v201_candidates": [candidate],
        "source": {
            "input_manifest_sha256": module.sha256(input_path),
            "published_manifest": str(published),
            "published_manifest_sha256": module.sha256(published),
            "generation_contract": str(generation),
            "generation_contract_sha256": module.sha256(generation),
        },
    }
    comparison_path = tmp_path / "v205_comparison.json"
    write_json(comparison_path, comparison)
    source = {
        "comparison_manifest": str(comparison_path),
        "comparison_manifest_sha256": module.sha256(comparison_path),
    }
    for key, content in (
        ("vbench_summary", "{}\n"),
        ("temporal_diagnostics", "method,prompt_index\n"),
        ("temporal_contract", "{}\n"),
    ):
        path = tmp_path / f"v205_{key}"
        path.write_text(content, encoding="utf-8")
        source[key] = str(path)
        source[f"{key}_sha256"] = module.sha256(path)
    decision = {
        "experiment": "v205_profile_disjoint_head_phase_horizon_vbench",
        "confirmatory": True,
        "paper_efficacy_support": True,
        "recommendation": "freeze_horizon_candidate_for_seed_length_checkpoint_replication",
        "confirmed_candidates": [candidate],
        "long_horizon_supported_candidates": [candidate],
        "candidate_status": {
            candidate: {
                "confirmation_pass": True,
                "positive_support": {
                    "interval_supported_axes": [
                        {
                            "window": "full",
                            "metric": "quality_without_dynamic_degree",
                        },
                        {
                            "window": "late_half",
                            "metric": "identity_background",
                        },
                    ]
                },
            }
        },
        "source": source,
    }
    decision_path = tmp_path / "v205_decision.json"
    write_json(decision_path, decision)
    return decision_path, input_path


def test_v206_preparer_freezes_seed_and_systematic_long_scopes(tmp_path: Path) -> None:
    module = load_module(
        "v206_prepare", SCRIPTS / "prepare_v206_horizon_robustness.py"
    )
    decision, v205_input = build_v205_fixture(tmp_path, module)
    output = tmp_path / "v206"
    payload = module.prepare(decision, v205_input, output)
    assert module.verify(output / "manifest.json") == payload
    assert tuple(row["key"] for row in payload["scopes"]) == module.SCOPE_KEYS
    seed = module.scope_config(payload, "seed20600_30s_128")
    long = module.scope_config(payload, "long60_seed20500_32")
    assert seed["prompt_count"] == 128 and seed["seed"] == 20600
    assert long["num_output_frames"] == 240 and long["seed"] == 20500
    assert long["prompt_positions_in_v205"] == list(range(0, 128, 4))
    assert long["prompt_source_indices"] == list(range(128, 256, 4))


def test_v206_preparer_fails_closed_without_v205_confirmation(tmp_path: Path) -> None:
    module = load_module(
        "v206_prepare_fail", SCRIPTS / "prepare_v206_horizon_robustness.py"
    )
    decision_path, v205_input = build_v205_fixture(tmp_path, module)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["paper_efficacy_support"] = False
    write_json(decision_path, decision)
    with pytest.raises(ValueError, match="confirmed v205"):
        module.prepare(decision_path, v205_input, tmp_path / "rejected")


def test_v206_seed_scope_requires_target_replication_and_temporal_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "v206_analysis", SCRIPTS / "analyze_v206_horizon_robustness.py"
    )
    candidate = "retrieval_horizon_top10"
    methods = ("sf_native", candidate)
    rows = {}
    for method in methods:
        for prompt in range(128):
            gain = 0.0 if method == "sf_native" else 1.0
            rows[(method, prompt)] = {
                "quality_without_dynamic_degree": 80.0 + gain,
                "official_quality_score": 80.0 + gain,
                "identity_background": 0.97 + gain * 0.002,
                "temporal_mechanics": 0.98 + gain * 0.002,
                "semantic_alignment": 0.55 + gain * 0.003,
                "visual_quality": 0.65 + gain * 0.003,
                "dynamic_degree": 1.0,
            }
    monkeypatch.setattr(
        module,
        "load_window_rows",
        lambda *args, **kwargs: {
            "full": rows,
            "early_half": rows,
            "late_half": rows,
        },
    )
    manifest = {
        "scope": "seed20600_30s_128",
        "scope_role": "same_prompt_new_seed_replication",
        "confirmed_v205_candidates": [candidate],
        "v205_positive_axes_to_replicate": {
            candidate: ["full:quality_without_dynamic_degree"]
        },
        "prompt_count": 128,
        "prompt_items": [
            {"source_index": 128 + prompt, "text": f"prompt {prompt}"}
            for prompt in range(128)
        ],
        "methods": [
            {"key": method, "video_dir": f"/videos/{method}"}
            for method in methods
        ],
        "claim_boundary": "unit-test boundary",
    }
    summary = {"methods": {method: {} for method in methods}}
    defaults = {feature: 0.0 for feature in module.v190.TEMPORAL_FEATURES}
    defaults.update(
        {
            "flow_speed_median": 0.5,
            "motion_coverage_fraction": 0.9,
            "late_motion_ratio": 1.0,
            "temporal_jump": 1.0,
        }
    )
    temporal = {
        (method, prompt): dict(defaults)
        for method in methods
        for prompt in range(128)
    }
    report = module.analyze_scope(
        manifest, summary, Path("unused"), temporal_rows=temporal
    )
    assert report["passing_candidates"] == [candidate]
    status = report["candidate_status"][candidate]
    assert status["v205_target_replication"]["pass"] is True
    assert status["automatic_temporal_guard"]["automatic_safety_pass"] is True
    assert report["manual_review_required_for_scope_pass"] is False


def test_v206_runners_include_both_scopes_and_camera_motion() -> None:
    generation = (SCRIPTS / "run_v206_horizon_robustness_32gpu.sh").read_text(
        encoding="utf-8"
    )
    evaluation = (SCRIPTS / "run_v206_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    for scope in ("seed20600_30s_128", "long60_seed20500_32"):
        assert scope in generation and scope in evaluation
    assert "generate-all" in generation
    assert "head_phase_horizon" in generation
    assert "verify_sampling_parity" in generation
    assert "tracked v206 runtime has unstaged changes" in generation
    assert "scopes/*/motion" in generation
    assert "motion-compute" in evaluation and "motion-analyze" in evaluation
    assert "analyze_v206_continuous_motion.py" in evaluation


def test_v206_motion_analysis_enumerates_every_confirmed_candidate() -> None:
    module = load_module(
        "v206_motion_candidates", SCRIPTS / "analyze_v206_continuous_motion.py"
    )
    candidates = ("landmark_horizon_top10", "retrieval_horizon_top10")
    manifest = {
        "experiment": "v206_horizon_robustness_vbench",
        "confirmatory": True,
        "primary_baseline": "sf_native",
        "confirmed_v205_candidates": list(candidates),
        "methods": [
            {"key": "sf_native", "runtime": "sf_native"},
            *(
                {
                    "key": candidate,
                    "runtime": "head_phase_horizon_cache_runtime",
                }
                for candidate in candidates
            ),
        ],
    }
    assert module.candidate_controls(manifest) == [
        (candidate, ("sf_native",)) for candidate in candidates
    ]
