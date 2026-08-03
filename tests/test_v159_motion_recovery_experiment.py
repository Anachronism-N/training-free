from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_v159_blind_review as blind  # noqa: E402
import analyze_v159_vbench as vbench_analysis  # noqa: E402
import prepare_v159_vbench_comparison as vbench  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v159_motion_coherent_reservoir_moviebench16 as v159  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v159_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)
history_polarity_policy_overrides = (
    POLICY_MODULE.history_polarity_policy_overrides
)


def test_v159_grid_generates_only_three_new_methods() -> None:
    v159.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v159.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 128
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [32, 32, 32, 32]
    assert len(v159.NEW_METHODS) * v159.PROMPT_COUNT == 48
    assert len(v159.REUSE_METHODS) * v159.PROMPT_COUNT == 80


def test_v159_reuse_loader_requires_complete_hashed_v157_source(
    tmp_path: Path, monkeypatch
) -> None:
    pf_config = tmp_path / "pyramid-forcing.yaml"
    pf_config.write_text("num_frame_per_block: 3\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    pf_runtime = v159.load_pf_runtime_contract(pf_config, checkpoint)
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir(parents=True)
    contract_path = contract_dir / "experiment.json"
    contract_path.write_text(
        json.dumps(
            {
                "prompt_suite": {"prompt_file_sha256": "prompt-sha"},
                "pf": {
                    "config_sha256": pf_runtime["config_sha256"],
                    "checkpoint_size": pf_runtime["checkpoint_size"],
                },
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for source_method in v159.REUSE_METHODS.values():
        video_dir = tmp_path / "videos" / source_method
        video_dir.mkdir(parents=True)
        for prompt_index in range(v159.PROMPT_COUNT):
            (video_dir / f"{prompt_index:06d}.mp4").write_bytes(b"video")
        rows.append({"key": source_method, "video_dir": str(video_dir)})
    published = {
        "ok": True,
        "experiment": "v157_layer_gated_moviebench16",
        "prompt_count": v159.PROMPT_COUNT,
        "experiment_contract_sha256": v159.sha256(contract_path),
        "methods": rows,
    }
    (tmp_path / "published_manifest.json").write_text(
        json.dumps(published), encoding="utf-8"
    )
    monkeypatch.setenv("V159_REUSE_V157_ROOT", str(tmp_path))

    source = v159.load_v157_source(
        {"prompt_file_sha256": "prompt-sha"},
        pf_runtime=pf_runtime,
    )
    assert set(source["sources"]) == set(v159.REUSE_METHODS)
    assert source["experiment_contract_sha256"] == v159.sha256(contract_path)
    assert source["matched_pf_runtime"]["num_frame_per_block"] == 3


def test_v159_rejects_a_config_without_adjacent_motion_edges(
    tmp_path: Path,
) -> None:
    pf_config = tmp_path / "pyramid-forcing.yaml"
    pf_config.write_text("num_frame_per_block: 1\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    try:
        v159.load_pf_runtime_contract(pf_config, checkpoint)
    except ValueError as error:
        assert "num_frame_per_block=3" in str(error)
    else:
        raise AssertionError("single-frame PF config must be rejected")


def test_v159_dual_timescale_policy_is_exactly_budget_matched() -> None:
    primary = v159.V159_CELLS[0]
    motion_only = v159.V159_CELLS[1]
    assert primary.uses_role_event is True
    assert primary.uses_motion is False
    assert expected_policy(primary, 10) == (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    assert expected_policy(primary, 11) == ((), 1, 8, "stride")
    assert expected_policy(motion_only, 10) == (
        ("CoherentMotionStrategy",),
        1,
        4,
        "coherent_motion",
    )
    overrides = history_polarity_policy_overrides(
        "reservoir2_motion1", "recent8_sink1"
    )
    assert overrides["pyramidkv_label_temporal_reservoir_capacity_map"] == {
        "10": 2,
        "11": 0,
    }
    assert overrides[
        "pyramidkv_label_coherent_motion_pair_capacity_map"
    ] == {"10": 1, "11": 0}
    assert overrides["pyramidkv_label_sink_frames_map"] == {"10": 1, "11": 1}
    assert overrides["pyramidkv_label_recent_frames_map"] == {
        "10": 4,
        "11": 8,
    }
    assert overrides["pyramidkv_hybrid_middle_enabled"] is True
    assert overrides["pyramidkv_composition_owns_dynamic"] is True


def test_v157_diagnosis_freezes_the_actual_failure_localization() -> None:
    path = (
        ROOT
        / "docs"
        / "results"
        / "v157_layer_gated_moviebench16"
        / "v157_motion_failure_diagnosis.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    motion = report["primary_minus_all_reservoir_motion"]
    assert motion["mean"] == -0.3125
    assert motion["deficit_prompts"] == [0, 1, 5, 6, 7, 15]
    assert motion["positive_count"] == 0
    assert report["primary_specific_severe_prompts"] == [0, 8, 15]


def _metric_payload() -> dict:
    dimensions = vbench.CORE_EVALUATION_DIMENSIONS
    return {
        "methods": {
            method: {dimension: 0.7 for dimension in dimensions}
            for method in vbench.METHODS
        },
        "dimensions": list(dimensions),
        "missing": [],
    }


def test_v159_vbench_is_a_safety_gate_not_a_promotion_gate() -> None:
    payload = _metric_payload()
    report = vbench_analysis.analyze(payload)
    assert report["metric_safety_gate"] is True
    assert report["human_motion_confirmation_required"] is True
    assert report["metric_promotion_gate"] is False

    payload["methods"][vbench.PRIMARY]["dynamic_degree"] = 0.67
    report = vbench_analysis.analyze(payload)
    assert report["metric_safety_gate"] is False


def _human_rows(*, primary_motion: float) -> list[dict]:
    rows = []
    for prompt_index in range(blind.PROMPT_COUNT):
        for method in blind.METHODS:
            rows.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    "identity_continuity_-2_to_2": 1.0,
                    "background_continuity_-2_to_2": 1.0,
                    "motion_quality_-2_to_2": (
                        primary_motion if method == blind.PRIMARY else 0.0
                    ),
                    "overall_preference_-2_to_2": 1.0,
                    "severe_failure_0_or_1": 0,
                }
            )
    return rows


def test_v159_human_recovery_gate_requires_real_motion_gain() -> None:
    report = blind.analyze(_human_rows(primary_motion=0.5))
    assert report["exploratory_recovery_gate"] is True
    assert report["human_promotion_gate"] is False

    report = blind.analyze(_human_rows(primary_motion=0.0))
    assert report["recovery_checks"]["motion_gain_over_reservoir4"] is False
    assert report["exploratory_recovery_gate"] is False
