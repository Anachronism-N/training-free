from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_v171_demand_gated_counterfactual as offline  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v171_demand_gated_motion_moviebench16 as runner  # noqa: E402
import v171_demand_gated_contract as contract  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v171_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def _candidate(
    pair: list[int],
    *,
    local_cosine: float,
    context_cosine: float,
    local_norm: float,
    context_norm: float,
    query_local: float = 1.0,
    query_context: float = 1.0,
) -> dict:
    return {
        "pair": pair,
        "state_similarity": 0.99,
        "local_direction_similarity": local_cosine,
        "context_direction_similarity": context_cosine,
        "query_local_magnitude": query_local,
        "candidate_local_magnitude": local_norm,
        "query_context_magnitude_per_step": query_context,
        "candidate_context_magnitude_per_step": context_norm,
        "state_pass": True,
        "direction_pass": True,
    }


def _deficit(triggered: bool) -> dict:
    ratio = 0.25 if triggered else 1.25
    return {
        "ready": True,
        "triggered": triggered,
        "local_median": 4.0,
        "context_median_per_step": 4.0,
        "local_ratio": ratio,
        "context_ratio": ratio,
        "rule": "both_scales_below_online_median",
    }


def test_v171_baseline_calibration_does_not_change_healthy_reads() -> None:
    candidates = [
        _candidate(
            [2, 3],
            local_cosine=0.8,
            context_cosine=0.8,
            local_norm=4.0,
            context_norm=4.0,
        ),
        _candidate(
            [21, 22],
            local_cosine=0.9,
            context_cosine=0.9,
            local_norm=1.0,
            context_norm=1.0,
        ),
    ]
    healthy = contract.expected_selection(
        candidates,
        method=contract.DEFICIT_BASELINE,
        motion_deficit=_deficit(False),
    )
    deficit = contract.expected_selection(
        candidates,
        method=contract.DEFICIT_BASELINE,
        motion_deficit=_deficit(True),
    )
    assert healthy["baseline"] == [21, 22]
    assert healthy["selected"] == healthy["baseline"]
    assert healthy["reason"] == "healthy_motion_signature_recall"
    assert deficit["selected"] == [2, 3]
    assert deficit["deficit_baseline"] == [2, 3]
    assert deficit["reason"] == (
        "motion_deficit_baseline_calibrated_recall"
    )


def test_v171_policy_aliases_change_only_selector_mode() -> None:
    mode_key = "pyramidkv_label_coherent_motion_state_motion_signature_mode_map"
    policies = (
        "reservoir2_multiscalemotion1",
        "reservoir2_deficitquery1",
        "reservoir2_deficitbaseline1",
    )
    configs = {
        policy: POLICY_MODULE.history_polarity_policy_overrides(
            policy,
            "recent8_sink1",
        )
        for policy in policies
    }
    baseline = configs[policies[0]]
    for policy in policies[1:]:
        assert {
            key for key in baseline if baseline[key] != configs[policy][key]
        } == {mode_key}
        assert configs[policy][mode_key]["10"] == (
            POLICY_MODULE.MOTION_SIGNATURE_POLICY_MODES[policy]
        )


def test_v171_grid_reuses_16_and_generates_32_videos() -> None:
    previous = os.environ.get("PYRAMIDKV_POLICY_TRACE_HEADS")
    os.environ["PYRAMIDKV_POLICY_TRACE_HEADS"] = "0"
    try:
        runner.configure_parent_runner()
    finally:
        if previous is None:
            os.environ.pop("PYRAMIDKV_POLICY_TRACE_HEADS", None)
        else:
            os.environ["PYRAMIDKV_POLICY_TRACE_HEADS"] = previous
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="ours")
    assert tuple(method.key for method in methods) == contract.METHODS
    assert len(parent.all_tasks(methods)) == 48
    assert len(runner.REUSE_METHODS) * contract.PROMPT_COUNT == 16
    assert len(runner.NEW_METHODS) * contract.PROMPT_COUNT == 32
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [12, 12, 12, 12]


def test_v171_cells_freeze_cache_and_change_only_selector() -> None:
    rows = {cell.support_policy: asdict(cell) for cell in runner.V171_CELLS}
    reference = rows["reservoir2_multiscalemotion1"]
    for policy in ("reservoir2_deficitquery1", "reservoir2_deficitbaseline1"):
        candidate = rows[policy]
        assert {
            key for key in reference if reference[key] != candidate[key]
        } == {"name", "stage", "support_policy"}
        assert candidate["suppress_policy"] == "recent8_sink1"
        assert candidate["map_key"] == "middle10"


def test_v171_checked_in_offline_gate_matches_frozen_v170_trace() -> None:
    report = offline.analyze(offline.DEFAULT_TRACE_DIR)
    assert report["offline_gate"] is True
    assert report["coverage"]["full_query_weighted_changes"] == 258
    assert report["methods"][contract.DEFICIT_QUERY]["changed"] == 86
    assert report["methods"][contract.DEFICIT_BASELINE]["changed"] == 180
    assert all(
        report["methods"][method]["healthy_changed"] == 0
        for method in contract.CANDIDATES
    )
    checked_in = json.loads(offline.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert checked_in["source"]["trace_sha256"] == report["source"][
        "trace_sha256"
    ]


def test_v171_shell_uses_32_gpus_and_no_default_manual_review() -> None:
    generation = (
        SCRIPTS / "run_v171_demand_gated_motion_moviebench16.sh"
    ).read_text(encoding="utf-8")
    evaluation = (SCRIPTS / "run_v171_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert 'NUM_NODES="${NUM_NODES:-4}"' in generation
    assert "export PYRAMIDKV_POLICY_TRACE_HEADS=0" in generation
    assert "analyze_v171_demand_gated_trace.py" in generation
    assert "prepare-review" not in evaluation
