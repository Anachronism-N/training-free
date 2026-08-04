from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
for path in (SCRIPTS, PF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_v160_adaptive_review as human  # noqa: E402
import analyze_v160_automated_screen as screen  # noqa: E402
import analyze_v160_fresh_motion_trace as trace_audit  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v160_fresh_motion_moviebench16 as v160  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


POLICY_SPEC = importlib.util.spec_from_file_location(
    "v160_policy_overrides_no_torch",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


def test_v160_grid_generates_only_one_new_method() -> None:
    v160.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
    assert tuple(method.key for method in methods) == v160.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 80
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [20, 20, 20, 20]
    assert len(v160.NEW_METHODS) * v160.PROMPT_COUNT == 16
    assert len(v160.REUSE_METHODS) * v160.PROMPT_COUNT == 64


def test_v160_reuse_loader_freezes_v159_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "v159"
    results = tmp_path / "results"
    (run_root / "contracts").mkdir(parents=True)
    results.mkdir()
    pf_runtime = {"config_sha256": "pf-config", "checkpoint_size": 123}
    contract_path = run_root / "contracts" / "experiment.json"
    contract_path.write_text(
        json.dumps(
            {
                "prompt_suite": {"prompt_file_sha256": "prompt-sha"},
                "pf": pf_runtime,
            }
        ),
        encoding="utf-8",
    )
    source_methods = set(v160.REUSE_METHODS.values())
    published_path = run_root / "published_manifest.json"
    published_path.write_text(
        json.dumps(
            {
                "ok": True,
                "experiment": "v159_motion_coherent_reservoir_moviebench16",
                "prompt_count": 16,
                "experiment_contract_sha256": v160.sha256(contract_path),
                "methods": [{"key": method} for method in source_methods],
            }
        ),
        encoding="utf-8",
    )
    for method in source_methods:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for prompt_index in range(16):
            (video_dir / f"{prompt_index:06d}.mp4").write_bytes(b"video")
    diagnostics = results / "v159_diagnostics.tar.gz"
    diagnostics.write_bytes(b"diagnostics")
    (results / "v159_motion_pair_trace_diagnosis.json").write_text(
        json.dumps(
            {
                "experiment": "v159_motion_pair_trace_diagnosis",
                "source": {"sha256": v160.sha256(diagnostics)},
                "diagnosis": {
                    "dominant_rejection": "motion_quantile_gate",
                    "max_pair_age_is_not_a_hard_refresh_bound": True,
                },
                "methods": {method: {} for method in v160.v159.NEW_METHODS},
            }
        ),
        encoding="utf-8",
    )
    metric_methods = {method: {} for method in source_methods}
    metric_methods["ours_middle10_reservoir2_motionpair1"] = {
        "dynamic_degree": 0.7458333333333333
    }
    metric_methods["ours_middle10_reservoir4_reference"] = {
        "dynamic_degree": 0.7791666666666667
    }
    (results / "vbench_core9_summary.json").write_text(
        json.dumps({"methods": metric_methods}),
        encoding="utf-8",
    )
    monkeypatch.setattr(v160, "v159_run_root", lambda: run_root)
    monkeypatch.setattr(
        v160,
        "_frozen_result_path",
        lambda name: results / name,
    )
    source = v160.load_v159_source(
        {"prompt_file_sha256": "prompt-sha"},
        pf_runtime=pf_runtime,
    )
    assert set(source["sources"]) == set(v160.REUSE_METHODS)
    assert source["trace_diagnosis_sha256"] == v160.sha256(
        results / "v159_motion_pair_trace_diagnosis.json"
    )


def test_v160_policy_is_an_isolated_freshness_change() -> None:
    primary = v160.V160_CELLS[0]
    reference = v160.V160_CELLS[1]
    assert expected_policy(primary, 10) == expected_policy(reference, 10) == (
        ("CoherentMotionStrategy", "TemporalReservoirStrategy"),
        1,
        4,
        "reservoir_motion",
    )
    fresh = POLICY_MODULE.history_polarity_policy_overrides(
        primary.support_policy,
        primary.suppress_policy,
    )
    old = POLICY_MODULE.history_polarity_policy_overrides(
        reference.support_policy,
        reference.suppress_policy,
    )
    differing = {
        key for key in fresh if fresh[key] != old[key]
    }
    assert differing == {
        "pyramidkv_label_coherent_motion_max_pair_age_map",
        "pyramidkv_label_coherent_motion_stale_refresh_map",
    }
    assert fresh["pyramidkv_label_coherent_motion_max_pair_age_map"] == {
        "10": 12,
        "11": 24,
    }
    assert fresh["pyramidkv_label_coherent_motion_stale_refresh_map"] == {
        "10": True,
        "11": False,
    }


def test_freshness_maps_are_wired_only_to_coherent_motion_constructor() -> None:
    source = (PF_ROOT / "pyramidkv" / "factory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id: {keyword.arg for keyword in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"CoherentMotionStrategy", "SemanticLandmarkStrategy"}
    }
    freshness = {"max_pair_age", "stale_refresh_bypass_quantile"}
    assert freshness.issubset(calls["CoherentMotionStrategy"])
    assert freshness.isdisjoint(calls["SemanticLandmarkStrategy"])


def test_v159_trace_diagnosis_freezes_the_actual_mechanism_failure() -> None:
    path = (
        ROOT
        / "docs"
        / "results"
        / "v159_motion_coherent_reservoir_moviebench16"
        / "v159_motion_pair_trace_diagnosis.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["experiment"] == "v159_motion_pair_trace_diagnosis"
    assert report["analyzer_sha256"] == v160.sha256(
        SCRIPTS / "analyze_v159_motion_pair_trace.py"
    )
    assert report["diagnosis"]["dominant_rejection"] == "motion_quantile_gate"
    assert report["diagnosis"]["max_pair_age_is_not_a_hard_refresh_bound"] is True
    primary = report["methods"][
        "ours_interleaved10_reservoir2_motionpair1"
    ]
    assert primary["prompt_count"] == 16
    assert primary["updates_per_prompt"] == [40]
    assert primary["accepted_per_prompt"]["mean"] == 6.125
    assert primary["pair_age_max"] == 61.0


def _diagnostic_rows() -> dict[tuple[str, int], dict[str, float]]:
    rows = {}
    for method_index, method in enumerate(screen.METHODS):
        for prompt_index in range(screen.PROMPT_COUNT):
            base = 0.1 * method_index + 0.01 * prompt_index
            rows[(method, prompt_index)] = {
                feature: (
                    base + feature_index * 0.001
                    if direction > 0
                    else 1.0 - base - feature_index * 0.001
                )
                for feature_index, (feature, direction) in enumerate(
                    screen.TEMPORAL_FEATURES.items()
                )
            }
    return rows


def test_automatic_screen_selects_two_unique_four_prompt_waves() -> None:
    rows = screen.score_prompts(_diagnostic_rows(), {})
    prompts = [
        {"tags": [f"tag-{index}", f"group-{index % 4}"]}
        for index in range(screen.PROMPT_COUNT)
    ]
    plan = screen.choose_review_prompts(rows, prompts)
    wave1 = [row["prompt_index"] for row in plan["wave1"]]
    wave2 = [row["prompt_index"] for row in plan["wave2"]]
    assert len(wave1) == len(set(wave1)) == 4
    assert len(wave2) == len(set(wave2)) == 4
    assert set(wave1).isdisjoint(wave2)
    assert plan["videos_per_wave"] == 12
    assert plan["maximum_review_videos"] == 24
    assert plan["selection_is_diagnostic_only"] is True


def test_fresh_motion_trace_counts_only_actual_below_quantile_bypass(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "primary.policy.jsonl"
    rows = []
    for index in range(40):
        bypass = index == 1
        decision = (
            {
                "accepted": True,
                "victim_stale": True,
                "motion_quantile_pass": False,
                "motion_ok": True,
                "stale_quantile_bypass": True,
                "reason": "stale_quantile_refresh",
                "candidate_pair": [10, 11],
                "motion": 0.2,
                "motion_threshold": 0.5,
                "victim_end_t": 3,
            }
            if bypass
            else {}
        )
        rows.append(
            {
                "event": "middle_selection",
                "layer": 15,
                "head": 0,
                "label": 10,
                "cache_contract_pass": True,
                "explicit_composition_owns_dynamic": True,
                "sink_frames": 1,
                "recent_frames": 4,
                "union_frame_count": 2,
                "sync_t": index * 3,
                "strategies": [
                    {
                        "name": "CoherentMotionStrategy",
                        "state": {
                            "max_pair_age": 12,
                            "stale_refresh_bypass_quantile": True,
                            "pair_capacity": 1,
                            "pair_frame_ids": [[10, 11]],
                            "accepted_count": 1,
                            "rejected_count": 38,
                            "evicted_count": 1,
                            "last_decision": decision,
                        },
                    }
                ],
            }
        )
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = trace_audit.analyze_prompt(trace, 0)
    assert report["stale_quantile_bypass_count"] == 1
    assert report["reason_counts"] == {"stale_quantile_refresh": 1}


def _human_rows(
    prompt_indices: range,
    *,
    primary_scores: list[float],
) -> list[dict]:
    rows = []
    for offset, prompt_index in enumerate(prompt_indices):
        for method in screen.REVIEW_METHODS:
            value = primary_scores[offset] if method == screen.PRIMARY else 0.0
            rows.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    **{dimension: value for dimension in human.DIMENSIONS},
                    human.SEVERE: 0,
                }
            )
    return rows


def test_adaptive_human_review_stops_or_expands_by_frozen_rules() -> None:
    safety = {"automatic_safety_screen": True}
    early = human.analyze(
        safety,
        _human_rows(range(4), primary_scores=[1.0, 1.0, 1.0, 0.0]),
        None,
    )
    assert early["wave1_decision"] == "exploratory_pass_stop_after_wave1"
    assert early["review_complete"] is True

    tied = human.analyze(
        safety,
        _human_rows(range(4), primary_scores=[0.0, 0.0, 0.0, 0.0]),
        None,
    )
    assert tied["wave1_decision"] == "continue_wave2"
    assert tied["exploratory_recovery_gate"] is False

    wave1 = _human_rows(range(4), primary_scores=[1.0, 1.0, -1.0, -1.0])
    pending = human.analyze(safety, wave1, None)
    assert pending["wave1_decision"] == "continue_wave2"
    assert pending["review_complete"] is False
    wave2 = _human_rows(range(4, 8), primary_scores=[1.0, 1.0, 1.0, 1.0])
    combined = human.analyze(safety, wave1, wave2)
    assert combined["review_complete"] is True
    assert combined["exploratory_recovery_gate"] is True
    assert combined["adaptive_review_is_not_paper_evidence"] is True


def test_v160_shells_expose_generation_and_adaptive_review_actions() -> None:
    generation = (SCRIPTS / "run_v160_fresh_motion_moviebench16.sh").read_text(
        encoding="utf-8"
    )
    automation = (SCRIPTS / "run_v160_automated_screen.sh").read_text(
        encoding="utf-8"
    )
    assert "reservoir2_freshmotionpair1" in generation
    assert "V160_REUSE_V159_ROOT" in generation
    assert "--workers" in automation
    assert "analyze_v160_fresh_motion_trace.py" in automation
    assert "evaluate_comprehensive.py" in automation
    assert "review-wave1" in automation
    assert "review-wave2" in automation
