from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v170_matched_metrics as analysis
import v170_matched_attribution_contract as contract


def synthetic_rows(
    *,
    delta_a: float = 0.2,
    delta_b: float = 0.2,
    replica_shift: float = 0.02,
) -> dict:
    rows = {}
    for prompt in range(contract.PROMPT_COUNT):
        base = 1.0 + 0.01 * prompt
        rows[(contract.V166_A, prompt)] = {"metric": base}
        rows[(contract.QUERY_A, prompt)] = {"metric": base + delta_a}
        rows[(contract.V166_B, prompt)] = {"metric": base + replica_shift}
        rows[(contract.QUERY_B, prompt)] = {"metric": base + replica_shift + delta_b}
    return rows


def test_matched_report_separates_effect_from_replica_noise() -> None:
    report = analysis.matched_metric_report(
        synthetic_rows(),
        metric="metric",
        seed=170,
    )
    assert abs(report["lane_a"]["mean"] - 0.2) < 1e-12
    assert abs(report["lane_b"]["mean"] - 0.2) < 1e-12
    assert abs(report["matched_effect"]["mean"] - 0.2) < 1e-12
    assert abs(report["replica_noise"]["mean"] - 0.02) < 1e-12
    assert report["effect_exceeds_mean_replica_noise"] is True
    assert report["order_strata"]["query_first"]["count"] == 16
    assert report["order_strata"]["query_second"]["count"] == 16


def fake_report(value: float, *, above_noise: bool = True) -> dict:
    return {
        "lane_a": {"mean": value},
        "lane_b": {"mean": value},
        "matched_effect": {"mean": value},
        "effect_exceeds_mean_replica_noise": above_noise,
    }


def test_development_gate_requires_all_axes_and_quality_above_noise() -> None:
    passing = {metric: fake_report(0.1) for metric in analysis.GATE_METRICS}
    decision = analysis.development_decision(passing, mechanism_gate=True)
    assert decision["attribution_gate"] is True

    failing = dict(passing)
    failing["identity_background"] = fake_report(-0.001)
    decision = analysis.development_decision(failing, mechanism_gate=True)
    assert decision["attribution_gate"] is False
    assert "without_additional_manual_review" in decision["recommendation"]
