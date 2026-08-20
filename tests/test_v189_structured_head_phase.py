from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


def test_v189_preparer_freezes_split_and_operator_contract(tmp_path: Path) -> None:
    module = load_module(
        "v189_prepare",
        SCRIPTS / "prepare_v189_structured_head_phase_profile.py",
    )
    source = tmp_path / "prompts.txt"
    source.write_text(
        "\n".join(f"prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "inputs"
    payload = module.prepare(source, output)
    assert module.verify(output / "manifest.json") == payload
    split = payload["prompt_split"]
    assert tuple(len(split[key]) for key in split) == (64, 32, 32)
    assert set(split["discovery"]).isdisjoint(split["validation"])
    assert set(split["discovery"]).isdisjoint(split["generation_holdout"])
    assert set(split["validation"]).isdisjoint(split["generation_holdout"])
    assert payload["operators"] == ["landmark", "retrieval"]
    assert payload["teacher_max_budget_ffe"] == 13
    assert payload["teacher_candidates"] == ["recent", "coverage"]


class DummyCache:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.num_heads = 12
        self.calls = []

    def set_cache_compatibility_active_policy(self, policy: str, **kwargs) -> None:
        self.calls.append((policy, kwargs))


def _phase_payload() -> dict:
    masks = [
        [[[False for _ in range(12)] for _ in range(30)]][0]
        for _ in range(4)
    ]
    masks[0][0][1] = True
    masks[0][0][7] = True
    masks[1][1] = [True] * 12
    return {
        "version": 1,
        "map_id": "unit-phase-map",
        "coverage_operator": "landmark",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "coverage_masks": masks,
    }


def test_v189_head_phase_schedule_routes_each_layer_and_keeps_clean_recent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_module(
        "v189_schedule",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "denoise_schedule.py",
    )
    path = tmp_path / "map.json"
    path.write_text(json.dumps(_phase_payload()), encoding="utf-8")
    monkeypatch.setenv(module.CACHE_COMPAT_DENOISE_SCHEDULE_ENV, "head_phase")
    monkeypatch.setenv(module.CACHE_COMPAT_HEAD_PHASE_MAP_ENV, str(path))
    layer0 = DummyCache(0)
    layer1 = DummyCache(1)
    result = module.set_cache_compatibility_denoise_state(
        [layer0, layer1],
        call_index=0,
        call_count=4,
        update_mode="noisy",
        current_start=120,
    )
    assert result == "mixed"
    assert layer0.calls[-1][0] == "mixed"
    assert sum(layer0.calls[-1][1]["coverage_head_mask"]) == 2
    assert layer1.calls[-1][0] == "recent"
    assert sum(layer1.calls[-1][1]["coverage_head_mask"]) == 0
    assert layer0.calls[-1][1]["phase_map_id"] == "unit-phase-map"

    result = module.set_cache_compatibility_denoise_state(
        [layer0, layer1],
        call_index=None,
        call_count=4,
        update_mode="clean",
        current_start=120,
    )
    assert result == "recent"
    assert layer0.calls[-1][0] == "recent"
    assert layer0.calls[-1][1]["coverage_head_mask"] is None


def test_v189_phase_map_rejects_integer_masks(tmp_path: Path) -> None:
    module = load_module(
        "v189_schedule_invalid",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "denoise_schedule.py",
    )
    payload = _phase_payload()
    payload["coverage_masks"][0][0][0] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid head mask"):
        module.load_cache_compatibility_head_phase_map(path)


def test_v189_classifier_keeps_phase_specific_cell_without_cross_call_gate() -> None:
    analysis = load_module(
        "v189_analysis",
        SCRIPTS / "analyze_v189_structured_head_phase.py",
    )
    prepare = load_module(
        "v189_prepare_split",
        SCRIPTS / "prepare_v189_structured_head_phase_profile.py",
    )
    discovery, validation, _ = prepare.frozen_split()
    gain = np.zeros((128, 4, 30, 12), dtype=np.float64)
    gain[:, 0, 2, 3] = 0.10
    # A call-invariant compatible cell should not be called phase-selective.
    gain[:, :, 4, 5] = 0.08
    aggregate = {
        "gain": gain,
        "energy": np.ones_like(gain),
        "full_budget": np.ones_like(gain),
    }
    rows = analysis._cell_rows(
        aggregate,
        operator="landmark",
        discovery=discovery,
        validation=validation,
    )
    by_key = {
        (row["call_index"], row["layer"], row["head"]): row for row in rows
    }
    assert by_key[(0, 2, 3)]["compatible"] is True
    assert by_key[(0, 2, 3)]["phase_selective"] is True
    for call in range(4):
        assert by_key[(call, 4, 5)]["compatible"] is True
        assert by_key[(call, 4, 5)]["phase_selective"] is False


def test_v189_runtime_contract_is_representation_complete_and_traceable() -> None:
    profiler = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "cache_compat_profile.py"
    ).read_text(encoding="utf-8")
    core = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "core.py"
    ).read_text(encoding="utf-8")
    cache = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    runner = (
        SCRIPTS / "run_v189_structured_head_phase_profile_32gpu.sh"
    ).read_text(encoding="utf-8")
    assert '"v189": {' in profiler
    assert '"policies": ("recent", "coverage")' in profiler
    assert 'profile_contract in {"v176", "v177", "v189"}' in core
    assert 'contract == "v189"' in cache
    assert "compatibility_head_mask" in cache
    assert "--cache_compat_profile_coverage_operator" in inference
    assert "--pyramidkv_cache_compatibility_head_phase_map" in inference
    assert "candidate_representation_subset_checks" in runner
    assert "profile128" in runner


def test_v190_preparer_builds_count_and_phase_controls(tmp_path: Path) -> None:
    module = load_module(
        "v190_prepare", SCRIPTS / "prepare_v190_head_phase_causal_screen.py"
    )
    source = tmp_path / "source.txt"
    source.write_text(
        "\n".join(f"prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    profile_map = tmp_path / "all_heads.csv"
    profile_map.write_text(
        "\n".join(",".join("10" for _ in range(12)) for _ in range(30))
        + "\n",
        encoding="ascii",
    )
    holdout = list(range(96, 128))
    v189_manifest = {
        "experiment": "v189_structured_head_phase_profile",
        "source_prompt_file": str(source.resolve()),
        "source_prompt_file_sha256": module.sha256(source),
        "profile_map": str(profile_map.resolve()),
        "profile_map_sha256": module.sha256(profile_map),
        "prompt_split": {
            "discovery": list(range(64)),
            "validation": list(range(64, 96)),
            "generation_holdout": holdout,
        },
    }
    manifest_path = tmp_path / "v189_manifest.json"
    manifest_path.write_text(json.dumps(v189_manifest), encoding="utf-8")
    masks = [
        [[False for _ in range(12)] for _ in range(30)] for _ in range(4)
    ]
    masks[0][2][3] = True
    masks[1][2][7] = True
    primary = {
        "version": 1,
        "map_id": "primary-map",
        "coverage_operator": "landmark",
        "call_count": 4,
        "layer_count": 30,
        "head_count": 12,
        "coverage_masks": masks,
        "coverage_count_by_call": [1, 1, 0, 0],
    }
    primary_path = tmp_path / "primary.json"
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    analysis = {
        "experiment": "v189_structured_head_phase_profile",
        "recommendation": "advance_head_phase_maps_to_causal_screen",
        "input_manifest_sha256": module.sha256(manifest_path),
        "generation_candidates": ["landmark"],
        "operators": {
            "landmark": {
                "maps": {
                    "compatible": {
                        "path": str(primary_path.resolve()),
                        "sha256": module.sha256(primary_path),
                        "map_id": "primary-map",
                    }
                }
            }
        },
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "v190"
    payload = module.prepare(manifest_path, analysis_path, output)
    assert module.verify(output / "manifest.json") == payload
    assert payload["method_order"] == [
        "all_recent",
        "landmark_compatible",
        "landmark_membership_shift",
        "landmark_phase_shift",
        "landmark_dense_phase",
    ]
    primary_counts = payload["methods"]["landmark_compatible"][
        "coverage_count_by_call"
    ]
    membership_counts = payload["methods"]["landmark_membership_shift"][
        "coverage_count_by_call"
    ]
    phase_counts = payload["methods"]["landmark_phase_shift"][
        "coverage_count_by_call"
    ]
    assert membership_counts == primary_counts
    assert phase_counts == [0, 1, 1, 0]
    assert payload["methods"]["landmark_dense_phase"][
        "coverage_count_by_call"
    ] == [360, 360, 0, 0]


def test_v190_analyzer_requires_baseline_membership_and_phase_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "v190_analysis", SCRIPTS / "analyze_v190_head_phase_causal_screen.py"
    )
    methods = (
        "all_recent",
        "landmark_compatible",
        "landmark_membership_shift",
        "landmark_phase_shift",
        "landmark_dense_phase",
    )
    roles = {
        "all_recent": "local_control",
        "landmark_compatible": "primary_head_phase",
        "landmark_membership_shift": "layer_count_matched_membership_control",
        "landmark_phase_shift": "call_count_matched_phase_control",
        "landmark_dense_phase": "same_active_calls_dense_control",
    }
    manifest = {
        "experiment": "v190_head_phase_causal_vbench_screen32",
        "prompt_count": 32,
        "methods": [
            {"key": method, "role": roles[method], "operator": "landmark"}
            for method in methods
        ],
    }
    summary = {"methods": {method: {} for method in methods}, "missing": []}
    values = {
        "all_recent": (80.0, 0.9700, 0.9800, 0.40),
        "landmark_compatible": (80.5, 0.9705, 0.9805, 0.44),
        "landmark_membership_shift": (80.2, 0.9700, 0.9800, 0.41),
        "landmark_phase_shift": (80.2, 0.9700, 0.9800, 0.41),
        "landmark_dense_phase": (80.3, 0.9698, 0.9798, 0.43),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(32):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.24,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    monkeypatch.setattr(module.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module.base, "derived_rows", lambda *args, **kwargs: rows)
    report = module.analyze(manifest, summary, Path("unused"))
    assert report["statuses"]["landmark_compatible"]["baseline_pass"] is True
    assert (
        report["statuses"]["landmark_compatible"]["controls"][
            "head_membership"
        ]["supported"]
        is True
    )
    assert (
        report["statuses"]["landmark_compatible"]["controls"][
            "phase_membership"
        ]["supported"]
        is True
    )
    assert report["recommendation"] == "advance_head_phase_method_to_fresh128"
