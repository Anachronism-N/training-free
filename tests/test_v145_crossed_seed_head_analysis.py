import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v145_crossed_seed_head_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("v145_analysis_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_delta_direction_is_seed_reproducible_and_factor_specific():
    torch.manual_seed(145)
    base0 = torch.randn(12, 4, 16)
    base1 = torch.randn(12, 4, 16)
    semantic_direction = torch.randn(12, 1, 16) * 0.1
    other_direction = -semantic_direction
    variant0 = base0 + semantic_direction
    variant1 = base1 + semantic_direction
    other1 = base1 + other_direction

    same = MODULE._delta_direction_cosine(
        base0, variant0, base1, variant1
    )
    different = MODULE._delta_direction_cosine(
        base0, variant0, base1, other1
    )
    assert torch.allclose(same, torch.ones_like(same), atol=1e-5)
    assert torch.allclose(
        different, -torch.ones_like(different), atol=1e-5
    )


def test_aligned_cosine_and_layer_residual_are_headwise():
    torch.manual_seed(146)
    value = torch.randn(12, 3, 4, 16)
    assert torch.allclose(
        MODULE._aligned_cosine(value, value),
        torch.ones(12),
        atol=1e-5,
    )
    layer_offsets = np.repeat(np.arange(30, dtype=float), 12)
    head_pattern = np.tile(np.arange(12, dtype=float), 30)
    residual = MODULE._layer_residual(
        100.0 * layer_offsets + head_pattern
    )
    expected = np.tile(np.arange(12, dtype=float) - 5.5, 30)
    assert np.allclose(residual, expected)


def test_policy_shift_detects_changed_candidate_preference():
    def record(first):
        return {
            "causal_policy_metrics": {
                "a": {
                    "projected_relative_error": torch.ones(12) * first
                },
                "b": {
                    "projected_relative_error": torch.ones(12)
                },
            }
        }

    unchanged = MODULE._policy_shift(record(1.0), record(1.0))
    changed = MODULE._policy_shift(record(1.0), record(3.0))
    assert torch.allclose(unchanged, torch.zeros_like(unchanged))
    assert torch.all(changed > 0)


def test_crossed_state_analysis_uses_paired_seed_records(monkeypatch):
    monkeypatch.setattr(MODULE, "FAMILIES", 2)
    monkeypatch.setattr(MODULE, "LAYERS", 2)
    monkeypatch.setattr(MODULE, "HEADS", 2)
    monkeypatch.setattr(MODULE, "TOTAL_HEADS", 4)
    indexed = {}
    torch.manual_seed(147)
    factor_directions = {
        variant: torch.randn(2, 1, 3) * 0.05
        for variant in MODULE.FACTOR_VARIANTS
    }
    for family in range(2):
        for replicate in MODULE.REPLICATES:
            base_records = {}
            for layer in range(2):
                state = ("clean", 63, 0, layer)
                query = torch.randn(2, 2, 3)
                key = torch.randn(2, 2, 2, 3)
                value = torch.randn(2, 2, 2, 3)
                base_records[state] = {
                    "query_projection": query,
                    "history_key_projection": key,
                    "history_value_projection": value,
                    "history_value_rms": torch.ones(2, 2, 2),
                    "causal_policy_metrics": {
                        "recent": {
                            "projected_relative_error": torch.ones(2)
                        },
                        "uniform": {
                            "projected_relative_error": torch.ones(2) * 2
                        },
                    },
                }
            indexed[(family, replicate, "base")] = {
                "records": base_records
            }
            for factor_index, variant in enumerate(
                MODULE.FACTOR_VARIANTS, start=1
            ):
                records = {}
                direction = factor_directions[variant]
                for state, base in base_records.items():
                    records[state] = {
                        "query_projection": (
                            base["query_projection"] + direction
                        ),
                        "history_key_projection": (
                            base["history_key_projection"]
                            + direction[:, None]
                        ),
                        "history_value_projection": (
                            base["history_value_projection"]
                            + direction[:, None]
                        ),
                        "history_value_rms": (
                            base["history_value_rms"]
                            * (1.0 + factor_index * 0.01)
                        ),
                        "causal_policy_metrics": (
                            base["causal_policy_metrics"]
                        ),
                    }
                indexed[(family, replicate, variant)] = {
                    "records": records
                }

    observations = MODULE._state_observations(indexed)
    assert len(observations) == 32
    assert {
        row["variant"] for row in observations
    } == set(MODULE.FACTOR_VARIANTS)
    assert all(np.isfinite(row["q_shift_mean"]) for row in observations)
    assert all(
        row["q_delta_seed_cosine"] > 0.99 for row in observations
    )
    family_rows = MODULE._aggregate_family_heads(observations)
    head_rows = MODULE._aggregate_heads(family_rows)
    assert len(family_rows) == 32
    assert len(head_rows) == 16
