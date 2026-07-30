import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v144_factorized_mechanism_profiles.py"
)
SPEC = importlib.util.spec_from_file_location(
    "v144_factor_analysis_test", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_descriptor_similarity_and_policy_shift_are_headwise():
    torch.manual_seed(144)
    query = torch.randn(12, 4, 16)
    history = query[:, None].repeat(1, 3, 1, 1)
    aligned = MODULE._descriptor_similarity(query, history)
    opposite = MODULE._descriptor_similarity(query, -history)
    assert torch.allclose(aligned, torch.ones_like(aligned), atol=1e-5)
    assert torch.allclose(opposite, -torch.ones_like(opposite), atol=1e-5)

    def record(scale):
        return {
            "causal_policy_metrics": {
                "a": {
                    "projected_relative_error": torch.ones(12) * scale,
                },
                "b": {
                    "projected_relative_error": torch.ones(12),
                },
            }
        }

    unchanged = MODULE._policy_shift(record(1.0), record(1.0))
    changed = MODULE._policy_shift(record(1.0), record(3.0))
    assert torch.allclose(unchanged, torch.zeros_like(unchanged))
    assert torch.all(changed > 0)


def test_dominant_factor_diagnostic_allows_unresolved_heads():
    rows = []
    for family in range(16):
        for layer in range(30):
            for head in range(12):
                row = {
                    "family_index": family,
                    "layer": layer,
                    "head": head,
                }
                for factor_index, factor in enumerate(
                    MODULE.SEMANTIC_FACTORS
                ):
                    value = 0.01 * layer
                    if head == factor_index:
                        value += 2.0
                    row[
                        f"{factor}.compatibility_loss_excess_seed"
                    ] = value
                rows.append(row)
    head_rows, audit, dominant = MODULE._head_axes(rows)
    assert len(head_rows) == 360
    assert len(audit) == 4
    assert dominant["discovery_counts"]["unresolved"] > 0
    for factor in MODULE.SEMANTIC_FACTORS:
        assert dominant["discovery_counts"][factor] > 0
    assert dominant["split_agreement"] == 1.0
    assert dominant["both_splits_resolved_count"] > 0
    assert dominant["both_splits_resolved_agreement"] == 1.0
    assert dominant["both_splits_unresolved_count"] > 0


def test_context_seed_excess_is_paired_before_family_aggregation():
    observations = []
    for family, seed_value, identity_value in (
        (0, 3.0, 10.0),
        (1, 7.0, 20.0),
    ):
        for variant, value in (
            ("seed_control", seed_value),
            ("identity", identity_value),
        ):
            row = {
                "family_index": family,
                "variant": variant,
                "mode": "clean",
                "current_frame": 117,
                "nominal_timestep": 0,
                "layer": 0,
                "head": 0,
            }
            for measure in MODULE.MEASURES:
                row[measure] = value
            observations.append(row)

    augmented = MODULE._context_observations_with_seed_excess(observations)
    corrected = [
        row
        for row in augmented
        if row["variant"] == "identity_excess_seed"
    ]
    assert [row["query_shift"] for row in corrected] == [7.0, 13.0]
