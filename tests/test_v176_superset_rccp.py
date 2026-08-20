from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v173_cache_compatibility as base  # noqa: E402
import analyze_v176_superset_rccp as v176  # noqa: E402


def test_v176_frozen_split_is_disjoint_and_exhaustive() -> None:
    discovery, validation, generation = v176.frozen_prompt_split(
        list(range(128))
    )
    assert len(discovery) == 64
    assert len(validation) == 32
    assert len(generation) == 32
    assert set(discovery).isdisjoint(validation)
    assert set(discovery).isdisjoint(generation)
    assert set(validation).isdisjoint(generation)
    assert set(discovery) | set(validation) | set(generation) == set(range(128))
    assert v176.frozen_prompt_split(list(range(128))) == (
        discovery,
        validation,
        generation,
    )


def test_v176_split_rejects_partial_profile() -> None:
    with pytest.raises(ValueError, match="complete prompt ids"):
        v176.frozen_prompt_split(list(range(127)))


def test_v176_contract_uses_fair_teacher_and_four_calls() -> None:
    contract = base.PROFILE_CONTRACTS["v176"]
    assert contract["version"] == 2
    assert contract["method"] == "superset_residual_cache_compatibility"
    assert contract["expected_budget"] == {
        "recent": 9,
        "coverage": 9,
        "episode": 9,
        "union": 17,
    }
    assert contract["calls"] == [0, 1, 2, 3]
    assert contract["expected_records_per_prompt_layer"] == 48
    assert contract["trace_layers"] == {0, 10, 20, 29}
    assert contract["reference_superset"] is True


def test_v176_static_wiring_preserves_v173_contract() -> None:
    profile = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "cache_compat_profile.py"
    ).read_text(encoding="utf-8")
    cache = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
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
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    runner = (SCRIPTS / "run_v176_superset_rccp_32gpu.sh").read_text(
        encoding="utf-8"
    )

    assert '"v173": {' in profile and '"v176": {' in profile
    assert "CACHE_COMPAT_PROFILE_CONTRACT" in cache
    assert 'source_kind="episode_reservoir"' in cache
    assert 'if contract == "v189"' in cache
    assert 'if contract in {"v176", "v177"}' in cache
    assert "candidate_physical_superset_verified" in core
    assert "teacher is not a cache-" in core
    assert "--cache_compat_profile_contract" in inference
    assert "--cache_compat_profile_call_indices 0,1,2,3" in runner
    assert "--skip_video_decode" in runner
    assert "V176_OUT_ROOT" in runner
    assert "run_v175" not in runner


def test_v176_synthetic_oracle_and_policy_agreement() -> None:
    errors = {"recent": 1.0, "coverage": 0.5, "episode": 0.8}
    policy, margin = v176._oracle_policy(errors)
    assert policy == "coverage"
    assert margin == pytest.approx(0.3)
    left = np.ones((1, 12), dtype=np.int8)
    right = left.copy()
    right[0, 0] = 0
    assert v176._agreement(left, right) == pytest.approx(11 / 12)


def test_v176_analyzer_declares_untouched_generation_and_hard_negatives() -> None:
    source = (SCRIPTS / "analyze_v176_superset_rccp.py").read_text(
        encoding="utf-8"
    )
    assert "generation_prompts_used_for_membership" in source
    assert "global_bh_q_le_0p10" in source
    assert "same_nonlocal_9_of_12_without_competitor" in source
    assert "call_and_ar_consistency" in source
    assert "full_budget_ge_0p80" in source
    assert "hard_negative_0" not in source  # generated programmatically
    assert 'range(4)' in source
    assert "mean_nonlocal_jaccard" in source
    assert "threshold_sensitivity" in source


def test_v176_single_pass_aggregate_shapes() -> None:
    records = []
    for prompt in range(base.PROMPTS):
        for layer in range(base.LAYERS):
            for call in range(4):
                for frame in (12, 21):
                    records.append(
                        {
                            "prompt_id": prompt,
                            "layer": layer,
                            "current_frame": frame,
                            "call_index": call,
                            "reference_residual_energy": [1.0] * base.HEADS,
                            "policies": {
                                policy: {
                                    "residual_relative_mse": [
                                        0.1 + policy_index * 0.1
                                    ]
                                    * base.HEADS
                                }
                                for policy_index, policy in enumerate(base.POLICIES)
                            },
                            "budgets": {
                                policy: {
                                    "per_sequence_frame_equivalents": [9]
                                    * base.HEADS
                                }
                                for policy in base.POLICIES
                            },
                        }
                    )
    result = v176.aggregate_profile(records)
    assert result["errors"].shape == (128, 30, 12, 3)
    assert result["call_errors"].shape == (4, 128, 30, 12, 3)
    assert result["ar_errors"].shape == (2, 128, 30, 12, 3)
    assert result["budget_full"].shape == (128, 30, 12, 3)
    assert np.all(np.argmin(result["errors"], axis=-1) == 0)


def test_v176_map_labels_remain_pf_independent_namespace() -> None:
    assert base.LABELS == {"recent": 20, "coverage": 21, "episode": 22}
    assert not set(base.LABELS.values()).intersection({-1, 1, 2})
    assert Counter(base.LABELS.values()) == Counter({20: 1, 21: 1, 22: 1})


def test_v173_analyzer_defaults_remain_unchanged() -> None:
    parser_source = (SCRIPTS / "analyze_v173_cache_compatibility.py").read_text(
        encoding="utf-8"
    )
    audit_source = (SCRIPTS / "audit_v173_cache_compatibility.py").read_text(
        encoding="utf-8"
    )
    profile_source = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "cache_compat_profile.py"
    ).read_text(encoding="utf-8")
    assert 'contract: str = "v173"' in parser_source
    assert 'default="v173"' in parser_source
    assert 'default="v173"' in audit_source
    assert 'CACHE_COMPAT_PROFILE_CONTRACT", "v173"' in profile_source
