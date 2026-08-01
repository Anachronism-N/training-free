import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_v151_signed_policy_profiles import (
    HEADS,
    LAYERS,
    _preference,
    build_signed_maps_without_audit,
)


def _record(uniform, recent):
    metrics = {}
    values = {
        "boundary_recent": 0.4,
        "current_only": 0.8,
        "recent4": 0.5,
        "recent_budget": recent,
        "uniform_recent": uniform,
    }
    for name, value in values.items():
        metrics[name] = {
            "projected_relative_error": torch.full((HEADS,), value)
        }
    return {"causal_policy_metrics": metrics}


def test_preference_sign_has_physical_direction():
    preference = _preference(
        _record(uniform=0.2, recent=0.4),
        left="uniform_recent",
        right="recent_budget",
    )
    assert torch.all(preference > 0)
    reversed_preference = _preference(
        _record(uniform=0.8, recent=0.4),
        left="uniform_recent",
        right="recent_budget",
    )
    assert torch.all(reversed_preference < 0)


def test_signed_map_uses_discovery_scores_and_validates_on_holdout():
    rows = []
    for layer in range(LAYERS):
        for head in range(HEADS):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "discovery_abs_mean": float(head + 1),
                    "validation_abs_mean": float(head + 2),
                }
            )
    maps, diagnostics = build_signed_maps_without_audit(rows)
    assert maps["low4"]["0"] == [0, 1, 2, 3]
    assert maps["middle4"]["0"] == [4, 5, 6, 7]
    assert maps["high4"]["0"] == [8, 9, 10, 11]
    assert diagnostics["validation_high_low_ratio"] > 2.0
    assert diagnostics["validation_positive_layer_fraction"] == 1.0
