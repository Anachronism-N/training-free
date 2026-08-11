from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # The local planning machine has no model runtime.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v173_cache_compatibility as analysis  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_compatibility_policy_has_three_neutral_equal_budget_routes() -> None:
    module = load_module(
        "v173_policy_overrides",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "policy_overrides.py",
    )
    fields = module.cache_compatibility_policy_overrides(capacity=12345)

    assert fields["pyramidkv_code_map"] == {
        "20": 12345,
        "21": 12345,
        "22": 12345,
    }
    assert fields["pyramidkv_label_sink_frames_map"] == {
        "20": 1,
        "21": 1,
        "22": 1,
    }
    assert fields["pyramidkv_label_recent_frames_map"] == {
        "20": 8,
        "21": 4,
        "22": 4,
    }
    assert fields["pyramidkv_label_temporal_reservoir_capacity_map"] == {
        "20": 0,
        "21": 4,
        "22": 2,
    }
    assert fields["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
        "20": 0,
        "21": 0,
        "22": 1,
    }
    assert fields["pyramidkv_composition_owns_dynamic"] is True

    profile = module.history_polarity_policy_overrides(
        module.CACHE_COMPAT_PROFILE_POLICY,
        module.CACHE_COMPAT_PROFILE_POLICY,
    )
    assert profile["pyramidkv_label_temporal_reservoir_capacity_map"] == {
        "10": 4,
        "11": 4,
    }
    assert profile["pyramidkv_label_coherent_motion_pair_capacity_map"] == {
        "10": 1,
        "11": 1,
    }


@pytest.mark.skipif(torch is None, reason="torch is available on the GPU server")
def test_residual_space_metric_uses_head_slice_of_output_projection() -> None:
    module = load_module(
        "v173_cache_compat_profile",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "cache_compat_profile.py",
    )
    module._records.clear()
    reference = torch.ones(1, 4, 2, 2)
    outputs = {
        "recent": reference * 0.90,
        "coverage": reference * 0.50,
        "episode": reference * 0.75,
        "union": reference,
    }
    budgets = {
        policy: {
            "max_frame_equivalents": 15 if policy == "union" else 9,
        }
        for policy in (*module.POLICIES, "union")
    }
    capture = {
        "prompt_id": 0,
        "layer": 0,
        "current_start": 0,
        "current_frame": 12,
        "cache_update_mode": "noisy",
        "cfg_branch": "cond",
        "call_index": 0,
        "frame_seqlen": 4,
    }
    module.record_cache_compatibility_outputs(
        outputs=outputs,
        output_projection_weight=torch.eye(4),
        capture=capture,
        budget_metadata=budgets,
    )

    record = module._records[-1]
    recent = record["policies"]["recent"]["residual_relative_mse"][0]
    episode = record["policies"]["episode"]["residual_relative_mse"][0]
    coverage = record["policies"]["coverage"]["residual_relative_mse"][0]
    assert recent < episode < coverage
    assert math.isclose(recent, 0.01, rel_tol=1e-5)


def synthetic_records() -> list[dict]:
    records = []
    best_by_head = ("recent", "coverage", "episode")
    for prompt in range(analysis.PROMPTS):
        for layer in range(analysis.LAYERS):
            for call_index, frame in ((0, 12), (2, 21)):
                policies = {}
                for policy in analysis.POLICIES:
                    errors = []
                    for head in range(analysis.HEADS):
                        best = best_by_head[head % len(best_by_head)]
                        errors.append(0.10 if policy == best else 0.30)
                    policies[policy] = {
                        "residual_relative_mse": errors,
                    }
                budgets = {
                    policy: {
                        "per_sequence_frame_equivalents": [
                            analysis.EXPECTED_BUDGET[policy]
                        ]
                        * analysis.HEADS
                    }
                    for policy in (*analysis.POLICIES, "union")
                }
                records.append(
                    {
                        "prompt_id": prompt,
                        "layer": layer,
                        "heads": analysis.HEADS,
                        "current_frame": frame,
                        "call_index": call_index,
                        "policies": policies,
                        "budgets": budgets,
                    }
                )
    return records


def test_analyzer_discovers_stable_operator_assignments_and_controls() -> None:
    rows, labels = analysis.analyze_heads(
        synthetic_records(),
        calibration_prompts=64,
        bootstrap_samples=50,
    )

    assert len(rows) == analysis.LAYERS * analysis.HEADS
    assert all(row["supported"] for row in rows)
    assert Counter(row["assigned_policy"] for row in rows) == {
        "recent": 120,
        "coverage": 120,
        "episode": 120,
    }
    controls = analysis.build_control_maps(labels)
    assert set(controls) == {
        "matched",
        "swapped",
        "all_recent",
        "all_coverage",
        "all_episode",
        "random_count_matched_0",
        "random_count_matched_1",
        "random_count_matched_2",
        "random_count_matched_3",
    }
    for layer in range(analysis.LAYERS):
        matched_counts = Counter(controls["matched"][layer])
        for replica in range(4):
            assert Counter(
                controls[f"random_count_matched_{replica}"][layer]
            ) == matched_counts
        for left, right in zip(
            controls["matched"][layer], controls["swapped"][layer]
        ):
            if left == analysis.LABELS["coverage"]:
                assert right == analysis.LABELS["episode"]
            elif left == analysis.LABELS["episode"]:
                assert right == analysis.LABELS["coverage"]
            else:
                assert right == left


def test_v173_shell_exposes_automated_profile_lifecycle() -> None:
    shell = (SCRIPTS / "run_v173_cache_compat_profile_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "profile128" in shell
    assert "MovieGen_128_qwen.txt" in shell
    assert "reservoir4_multiscalemotion1" in shell
    assert "cache_compat_profile_output" in shell
    assert "audit_v173_cache_compatibility.py" in shell
    assert "analyze_v173_cache_compatibility.py" in shell

    generation = (
        SCRIPTS / "run_v174_cache_compat_generation_32gpu.sh"
    ).read_text(encoding="utf-8")
    assert "screen32" in generation and "confirm128" in generation
    assert "matched,swapped,random_count_matched_0" in generation
    assert "--pyramidkv_cache_compatibility_policy" in generation
    assert "pf_native" not in generation

    evaluation = (SCRIPTS / "run_v174_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_v174_vbench_comparison.py" in evaluation
    assert "analyze_v174_paired_metrics.py" in evaluation
    assert "V174_SCOPE" in evaluation

    cache_source = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    assert cache_source.count(
        '"recent" if self.cache_compat_profile_enabled else None'
    ) == 2
    assert "cache compatibility middle budget exceeded" in cache_source
