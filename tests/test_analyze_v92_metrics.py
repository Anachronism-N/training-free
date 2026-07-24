import pytest

from scripts.analyze_v92_metrics import EXPECTED_METHODS, analyze


def _metrics(dino: float) -> dict[str, float]:
    return {
        "m1_dino_consistency": dino,
        "m1_min_stability": dino - 0.05,
        "m2_drift_slope": -0.002,
        "m7_background_consistency": 0.90,
        "composite": 0.51,
    }


def test_analyze_v92_factorizes_topology_and_prompt_controls():
    methods = {name: _metrics(0.80) for name in EXPECTED_METHODS}
    methods["prompt_pfcount_read_v78"] = _metrics(0.82)
    methods["prompt_inverse_read_v78"] = _metrics(0.80)
    methods["prompt_random_read_v78"] = _metrics(0.81)
    methods["pf_binary_read_v78"] = _metrics(0.823)
    temporal = {
        name: {"count": 16, "mean": 1.5, "median": 1.5}
        for name in EXPECTED_METHODS
    }
    coherence = {
        "prompt_pfcount_read_v78": {
            "acceptance_rate": 0.6,
            "coherence": {"age_spread": {"mean": 2.0}},
        }
    }

    result = analyze(
        {"per_method": methods},
        temporal,
        coherence,
        {"method": "test"},
    )

    assert result["classification_gate"]["metric_passed"] is True
    assert (
        result["comparisons"]["prompt_vs_inverse"]["delta"][
            "m1_dino_consistency"
        ]
        == pytest.approx(0.02)
    )
    row = next(
        item
        for item in result["ranking"]
        if item["method"] == "prompt_pfcount_read_v78"
    )
    assert row["acceptance_rate"] == 0.6
    assert row["age_spread"] == 2.0


def test_analyze_v92_rejects_missing_method():
    methods = {name: _metrics(0.80) for name in EXPECTED_METHODS[:-1]}

    try:
        analyze({"per_method": methods}, {}, {}, None)
    except ValueError as error:
        assert EXPECTED_METHODS[-1] in str(error)
    else:
        raise AssertionError("missing method must fail")
