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


def test_role_and_retrieval_head_fractions_are_reported_separately() -> None:
    base = {
        "event": "readout",
        "layer": 15,
        "trajectory_id": 0,
        "current_start": 80,
        "block_index": 20,
        "current_episode_id": 2,
        "previous_episode_id": 1,
        "allowed_episode_id": 0,
        "head_routing": "role_evidence",
        "accepted_head_count": 2,
        "head_count": 2,
        "head_gate_mean": 0.5,
        "head_gate_std": 0.3,
        "head_gate_active_fraction": 0.5,
        "role_evidence_spread": 0.2,
        "role_calibration_valid": True,
        "confidence_mean": 0.4,
        "alignment_positive_fraction": 0.5,
        "effective_weight_mean": 0.02,
        "delta_to_native_rms": 0.03,
    }
    first = {**base, "attention_call_index": 0, "head_role": {"gate": [0.8, 0.2]}}
    second = {**base, "attention_call_index": 1, "head_role": {"gate": [0.7, 0.3]}}
    report = analyze_records([{"event": "commit", "layer": 15}, first, second])

    assert report["metrics"]["retrieval_accepted_head_fraction"] == 1.0
    assert report["metrics"]["role_active_head_fraction"] == 0.5
    assert report["metrics"]["role_active_head_jaccard"] == 1.0


def test_layer_and_denoising_call_diagnostics_are_factorized() -> None:
    base = {
        "event": "readout",
        "trajectory_id": 0,
        "current_start": 80,
        "current_episode_id": 0,
        "recall_scope": "intra_episode",
        "allow_current_episode": True,
        "allowed_episode_id": 0,
        "selected_episode_ids": [0],
        "selected_frame_ages": [20],
        "recent_exclude_frames": 12,
        "memory_start_frame": 36,
        "current_frame": 80,
        "selected_indices_valid": True,
        "interval_sidecar_valid": True,
        "episode_sidecar_valid": True,
        "accepted_head_count": 2,
        "head_count": 2,
        "confidence_mean": 0.4,
        "retrieval_margin_mean": 0.2,
        "retrieval_entropy_mean": 0.6,
        "head_gate_mean": 0.5,
        "effective_weight_mean": 0.02,
        "alignment_positive_fraction": 0.5,
    }
    records = [
        {"event": "commit", "layer": 15},
        {**base, "layer": 15, "attention_call_index": 0, "delta_to_native_rms": 0.01},
        {**base, "layer": 15, "attention_call_index": 1, "delta_to_native_rms": 0.03},
        {**base, "layer": 16, "attention_call_index": 0, "delta_to_native_rms": 0.05},
    ]

    report = analyze_records(records)

    assert report["per_attention_call"]["0"]["readouts"] == 2
    assert report["per_attention_call"]["1"]["delta_to_native_rms_median"] == 0.03
    assert report["per_layer"]["15"]["retrieval_margin_mean"] == 0.2
    assert report["per_layer_attention_call"]["layer_16/call_0"]["readouts"] == 1


def test_low_role_evidence_spread_is_reported() -> None:
    records = [
        {"event": "commit", "layer": 15},
        {
            "event": "readout",
            "layer": 15,
            "current_start": 80,
            "block_index": 20,
            "current_episode_id": 2,
            "previous_episode_id": 1,
            "allowed_episode_id": 0,
            "head_routing": "role_evidence",
            "accepted_head_count": 2,
            "head_count": 2,
            "head_gate_mean": 0.5,
            "head_gate_std": 0.001,
            "head_gate_active_fraction": 0.5,
            "role_evidence_spread": 0.001,
            "role_calibration_valid": True,
            "confidence_mean": 0.4,
            "alignment_positive_fraction": 0.5,
            "effective_weight_mean": 0.02,
            "delta_to_native_rms": 0.03,
            "head_role": {"gate": [0.501, 0.499]},
        },
    ]
    report = analyze_records(records)
    codes = {finding["code"] for finding in report["findings"]}
    assert "role_gate_low_contrast" in codes
    assert "role_evidence_not_discriminative" in codes


def test_fail_closed_role_abstain_contributes_to_calibration_validity() -> None:
    records = [
        {"event": "commit", "layer": 15},
        {
            "event": "readout_abstain",
            "layer": 15,
            "current_start": 80,
            "current_episode_id": 2,
            "head_routing": "role_evidence",
            "reason": "role_evidence_spread_below_min",
            "role_evidence_spread": 0.001,
            "role_calibration_valid": False,
        },
    ]
    report = analyze_records(records)
    codes = {finding["code"] for finding in report["findings"]}
    assert report["metrics"]["role_calibration_valid_fraction"] == 0.0
    assert "role_calibration_often_invalid" in codes
    assert "role_calibration_rejected_all" in codes
    assert "no_accepted_readout" not in codes


def test_episode_warmup_scale_is_reported() -> None:
    records = [
        {
            "event": "config",
            "readout": {"episode_warmup_blocks": 2},
        },
        {"event": "commit", "layer": 15},
        {
            "event": "readout",
            "layer": 15,
            "current_episode_id": 2,
            "previous_episode_id": 1,
            "allowed_episode_id": 0,
            "accepted_head_count": 6,
            "head_count": 12,
            "confidence_mean": 0.4,
            "head_gate_mean": 0.5,
            "alignment_positive_fraction": 0.5,
            "effective_weight_mean": 0.01,
            "delta_to_native_rms": 0.02,
            "episode_block_index": 0,
            "episode_warmup_scale": 1 / 3,
            "effective_gate": 1 / 30,
        },
    ]

    report = analyze_records(records)
    assert report["metrics"]["episode_warmup_scale_min"] == 1 / 3
    assert report["metrics"]["episode_first_block_effective_gate_mean"] == 1 / 30
    assert not any(
        finding["code"] == "episode_warmup_not_observed"
        for finding in report["findings"]
    )


def test_empty_boundary_archive_snapshot_is_an_error() -> None:
    records = [
        {"event": "commit", "layer": 15},
        {
            "event": "boundary",
            "archive_preserved": True,
            "archive_state": {"archive_layers": []},
        },
    ]

    report = analyze_records(records)
    assert any(
        finding["code"] == "boundary_archive_snapshot_empty"
        and finding["severity"] == "ERROR"
        for finding in report["findings"]
    )


def test_nominal_intra_episode_trace_accepts_old_same_episode_frames() -> None:
    records = [
        {"event": "commit", "layer": 15},
        {
            "event": "readout",
            "layer": 15,
            "current_frame": 48,
            "block_index": 16,
            "current_episode_id": 0,
            "previous_episode_id": None,
            "recall_scope": "intra_episode",
            "allow_current_episode": True,
            "allowed_episode_id": 0,
            "memory_start_frame": 36,
            "recent_exclude_frames": 12,
            "selected_indices_valid": True,
            "interval_sidecar_valid": True,
            "episode_sidecar_valid": True,
            "selected_episode_ids": [0, 0],
            "selected_frame_ages": [20, 32],
            "episode_decision": {"winner_episode_id": 0},
            "accepted_head_count": 8,
            "head_count": 12,
            "confidence_mean": 0.4,
            "head_gate_mean": 1.0,
            "effective_weight_mean": 0.02,
            "alignment_positive_fraction": 0.6,
            "delta_to_native_rms": 0.03,
        },
    ]

    report = analyze_records(records)

    assert not any(finding["severity"] == "ERROR" for finding in report["findings"])
    assert report["metrics"]["intra_episode_readouts"] == 1
    assert report["metrics"]["intra_selected_frame_age_median"] == 26


def test_intra_episode_recent_or_foreign_selection_is_a_hard_error() -> None:
    records = [
        {"event": "commit", "layer": 15},
        {
            "event": "readout",
            "layer": 15,
            "current_frame": 48,
            "block_index": 16,
            "current_episode_id": 0,
            "recall_scope": "intra_episode",
            "allow_current_episode": True,
            "allowed_episode_id": 0,
            "memory_start_frame": 36,
            "recent_exclude_frames": 12,
            "selected_indices_valid": True,
            "interval_sidecar_valid": True,
            "episode_sidecar_valid": True,
            "selected_episode_ids": [1],
            "selected_frame_ages": [12],
            "accepted_head_count": 1,
            "head_count": 12,
            "delta_to_native_rms": 0.01,
        },
    ]

    report = analyze_records(records)

    assert any(
        finding["code"] == "causal_invariant_violation"
        and finding["severity"] == "ERROR"
        for finding in report["findings"]
    )
    assert len(report["violations"]) == 2
