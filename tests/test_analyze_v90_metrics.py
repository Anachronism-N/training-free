from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_v90_metrics.py"
SPEC = importlib.util.spec_from_file_location("analyze_v90_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _metrics(dino: float) -> dict[str, float]:
    return {
        "m1_dino_consistency": dino,
        "m1_min_stability": dino - 0.05,
        "m2_drift_slope": -0.002,
        "m7_background_consistency": 0.90,
        "composite": 0.51,
    }


def test_analyze_v90_computes_paired_gate_and_candidate_ranking() -> None:
    per_method = {
        "pf": _metrics(0.80),
        "v78": _metrics(0.81),
        "pf_s1": _metrics(0.82),
        "v78_s1": _metrics(0.83),
        "pf_s2": _metrics(0.84),
        "v78_s2": _metrics(0.83),
        "pf_s3": _metrics(0.81),
        "v78_s3": _metrics(0.82),
        "pf_binary_balanced": _metrics(0.80),
        "learned_balanced": _metrics(0.79),
        "candidate_a": _metrics(0.85),
        "candidate_b": _metrics(0.84),
    }
    temporal = {
        name: {"count": 16, "mean": 1.5, "median": 1.5}
        for name in per_method
    }
    coherence = {
        "candidate_a": {
            "acceptance_rate": 0.6,
            "coherence": {"age_spread": {"mean": 2.0}},
        }
    }

    result = MODULE.analyze(
        {"per_method": per_method},
        temporal,
        coherence,
    )

    assert result["paired_summary"]["m1_dino_consistency"]["count"] == 4
    assert result["paired_summary"]["m1_dino_consistency"]["nonnegative"] == 3
    assert result["v78_identity_gate"]["passed"] is True
    assert [row["method"] for row in result["candidates"]] == [
        "candidate_a",
        "candidate_b",
    ]
    assert result["candidates"][0]["trace"]["acceptance_rate"] == 0.6
