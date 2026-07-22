from scripts.analyze_hrem_v2_debug import analyze_records


def test_nominal_trace_has_no_error() -> None:
    records = [
        {
            "event": "commit",
            "layer_idx": 15,
            "archive_k_rms": 0.5,
            "archive_v_rms": 0.6,
            "episode_counts": {"0": 12, "1": 12},
        },
        {
            "event": "boundary",
            "previous_episode_id": 1,
            "current_episode_id": 2,
            "archive_preserved": True,
        },
        {
            "event": "readout",
            "layer_idx": 15,
            "block_index": 20,
            "current_episode_id": 2,
            "previous_episode_id": 1,
            "allowed_episode_id": 0,
            "episode_decision": {"winner_episode_id": 0},
            "accepted_head_count": 8,
            "head_count": 12,
            "confidence_mean": 0.4,
            "head_gate_mean": 0.5,
            "effective_weight_mean": 0.02,
            "alignment_positive_fraction": 0.6,
            "delta_to_native_rms": 0.03,
        },
    ]
    report = analyze_records(records)
    assert not any(finding["severity"] == "ERROR" for finding in report["findings"])
    assert report["metrics"]["accepted_head_fraction"] == 8 / 12


def test_previous_episode_admission_is_a_hard_error() -> None:
    records = [
        {"event": "commit", "layer_idx": 15},
        {
            "event": "readout",
            "layer_idx": 15,
            "block_index": 20,
            "current_episode_id": 2,
            "previous_episode_id": 1,
            "allowed_episode_id": 1,
            "accepted_head_count": 1,
            "head_count": 12,
            "delta_to_native_rms": 0.01,
        },
    ]
    report = analyze_records(records)
    codes = {finding["code"] for finding in report["findings"]}
    assert "causal_invariant_violation" in codes
    assert report["violations"]


def test_populated_archive_without_readout_reports_gate_failure() -> None:
    records = [
        {"event": "commit", "layer_idx": 15},
        {
            "event": "readout_abstain",
            "layer_idx": 15,
            "episode_decision": {"abstain_reason": "cue_disagreement"},
        },
    ]
    report = analyze_records(records)
    assert report["abstain_reasons"] == {"cue_disagreement": 1}
    assert any(
        finding["code"] == "no_accepted_readout" for finding in report["findings"]
    )
