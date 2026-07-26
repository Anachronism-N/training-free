import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_v98_history_polarity_4node_32gpu.sh"
POSTPROCESS = ROOT / "scripts" / "postprocess_v98_history_polarity.sh"


def _first_array(script: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\(\n(.*?)^\)", script, flags=re.MULTILINE | re.DOTALL)
    assert match is not None
    return [
        line.strip().strip('"')
        for line in match.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_primary_matrix_is_local_method_by_node_shard():
    runner = RUNNER.read_text(encoding="utf-8")
    methods = _first_array(runner, "METHODS")

    assert methods == [
        "sf_native",
        "pf_native",
        "pf_explicit_parity",
        "pf_aw_hybrid_merge",
        "history_polarity_hybrid_merge",
        "history_polarity_stride_merge",
        "history_polarity_zero_random_hybrid_merge",
        "positive_rate_half_hybrid_merge",
    ]
    assert "history_polarity_hybrid_merge_v78" not in methods
    assert "method_index=$local_slot" in runner
    assert "shard=$NODE_RANK" in runner
    assert "global_rank=$((NODE_RANK * 8 + local_slot))" not in runner
    assert '"gpu_slot_expression"' in runner
    assert "GPU_SLOT_MAPPING=" in runner
    assert 'global_manifest.get("GPU_SLOT_MAPPING")' in POSTPROCESS.read_text(
        encoding="utf-8"
    )


def test_runner_freezes_global_contract_media_and_python_path():
    runner = RUNNER.read_text(encoding="utf-8")

    assert "runs/v98_middle_relative_scores" in runner
    assert "EXPECTED_SCORE_ARTIFACT_VERSION=2" in runner
    assert "v98_middle_relative_qk_head_scores" in runner
    assert "middle_relative_logit_margin" in runner
    assert 'payload.get("accepted") is not True' in runner
    assert "experiment_contract.json" in runner
    assert '"version": 2' in runner
    assert '"run_fingerprint"' in runner
    assert '"tracked_worktree_dirty"' in runner
    assert "--untracked-files=all" in runner
    assert "--validate-only" in runner
    assert "calibration/deployment" in runner
    assert '"policies": policies' in runner
    assert '"policy_trace_branches": ["cond"]' in runner
    assert "experiment_contract_sha256" in runner
    assert "EXPECTED_VIDEO_FRAMES:-$((FRAMES * 4 - 3))" in runner
    assert "--expected-frames \"$EXPECTED_VIDEO_FRAMES\"" in runner
    assert "--expected-fps \"$EXPECTED_VIDEO_FPS\"" in runner
    assert "--expected-width \"$EXPECTED_VIDEO_WIDTH\"" in runner
    assert "--expected-height \"$EXPECTED_VIDEO_HEIGHT\"" in runner
    assert "export PYRAMIDKV_CPP_STRATEGY=0" in runner
    assert "export PYRAMIDKV_USE_CPP_PACK_OUTPUT=0" in runner
    assert "export PYRAMIDKV_USE_MEGA_ATTN=0" in runner
    assert 'PRELOAD_PYRAMIDKV="${PRELOAD_PYRAMIDKV:-0}"' in runner
    assert "--pyramidkv_cache_transition_branches cond" in runner
    assert '["uniform_stride", "uniform_merge"]' in runner
    assert "counterfactual_prompt_pair" in runner
    assert 'PYRAMIDKV_POLICY_TRACE_MAX_RECORDS:-60000' in runner
    assert 'v98 protocol requires FRAMES=120' in runner
    assert 'v98 protocol requires SEED=0' in runner
    assert "must match the canonical MovieBench-$EXPECTED file" in runner
    assert "v98-main128-gate" in runner
    assert 'contract.get("phase") != "primary"' in runner
    assert "screen32/main128 method policies or maps differ" in runner
    assert "NODE_RUN_LOCK" in runner


def test_v78_is_an_isolated_matched_followup():
    runner = RUNNER.read_text(encoding="utf-8")
    postprocess = POSTPROCESS.read_text(encoding="utf-8")

    assert 'if [[ "$FOLLOWUP_V78" == "1" ]]' in runner
    assert 'RUN_ROOT="$OUT_ROOT/followup_v78"' in runner
    assert "followup_history_polarity_hybrid_merge_base" in runner
    assert "followup_history_polarity_hybrid_merge_v78" in runner
    assert "follow-up requires completed primary node" in runner
    assert '"primary_gate_evidence"' in runner
    assert "PRIMARY_ANALYSIS_SHA256" in runner
    assert "PRIMARY_BLIND_VERIFICATION_SHA256" in runner
    assert "primary analysis input is stale" in runner
    assert "primary experiment semantics do not match this follow-up" in runner
    assert 'contract.get("phase") != "primary"' in runner
    assert 'contract.get("mode") != expected_mode' in runner
    assert 'prompt_contract.get("sha256") != digest(prompts_path)' in runner
    assert 'score_contract.get("map_manifest_sha256")' in runner
    assert runner.rfind("import csv", 0, runner.index("csv.reader")) >= 0
    assert 'RUN_ROOT="$PRIMARY_RUN_ROOT/followup_v78"' in postprocess
    assert "v78 follow-up cannot feed the primary go/no-go analyzer" in postprocess


def test_postprocess_hard_gates_and_fingerprints_metrics():
    postprocess = POSTPROCESS.read_text(encoding="utf-8")

    freeze_gate = postprocess.index("--verify-frozen")
    vbench_run = postprocess.index("--mode long_custom_input")
    comprehensive_run = postprocess.index("evaluate_comprehensive.py")
    temporal_run = postprocess.index("compute_temporal_jump_diagnostic.py")
    assert freeze_gate < vbench_run
    assert freeze_gate < comprehensive_run
    assert freeze_gate < temporal_run
    assert "--experiment-contract \"$EXPERIMENT_CONTRACT\"" in postprocess
    assert '--blind-scorecard "$BLIND_REVIEW/scorecard.csv"' in postprocess
    assert '--blind-key "$BLIND_PRIVATE/key_private.json"' in postprocess
    assert '--blind-verification "$METRICS/blind_frozen_verification.json"' in (
        postprocess
    )
    assert '--metric-manifest "$METRICS/metric_manifest.json"' in postprocess
    assert "videos changed after generation" in postprocess
    assert "metric_input_fingerprint" in postprocess
    assert "RESUME_VALID" in postprocess
    assert "*_eval_results.json" in postprocess
    assert "vbench_version.lock.env" in postprocess
    assert "input_count" in postprocess
    assert "POSTPROCESS_RUN_LOCK" in postprocess
    assert '--expected-videos "$((EXPECTED * ${#METHODS[@]}))"' in postprocess
    assert '"${VIDEO_FILES[@]}" --frame-step' in postprocess
